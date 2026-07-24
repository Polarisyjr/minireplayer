from __future__ import annotations

import ast
import contextvars
import importlib
import inspect
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from minireplay.observation import recorded_output_result_contract
from minireplay.sdk import (
    composite_scope,
    current_context,
    last_llm_attempt_id,
    llm_scope,
    run_dispatch,
    run_dispatch_async,
    run_tool,
    run_tool_async,
)
from minireplay.serialization import jsonable
from minireplay.util import atomic_write_json, read_json, sha256_json

from .gate import gated_terminal_callable
from .patching import method_identity, patch_method
from .result_replay import encode_framework_output, restore_framework_output
from .state import state

_TASK_SUBMISSIONS = itertools.count()
_COMPOSITE_TOOL_IMPLEMENTATIONS = frozenset(
    {
        "camel.toolkits.browser_toolkit.AsyncBrowserToolkit.browse_url",
        "camel.toolkits.browser_toolkit.BrowserToolkit.browse_url",
    }
)

# owl's own launcher tags every model it builds with the role that selected the
# endpoint (`make_model` in run_gaia_workforce_vllm_flex.py). That attribute is the
# only place the native role survives to the model object, so read it rather than
# invent a second one — a renamed copy here silently fails every LLM call.
_ROLE_ATTRIBUTE = "_agent_replay_role"


def _native_model_role(model: Any) -> str:
    role = getattr(model, _ROLE_ATTRIBUTE, None)
    if not isinstance(role, str) or not role:
        raise RuntimeError(
            "Owl model request has no native role identity "
            f"({type(model).__module__}.{type(model).__qualname__} has no {_ROLE_ATTRIBUTE})"
        )
    return role


def _worker_signature(worker: Any) -> dict[str, Any]:
    agent = getattr(worker, "worker", None)
    tools = getattr(agent, "_internal_tools", {})
    return {
        "class": f"{type(worker).__module__}.{type(worker).__qualname__}",
        "description": str(getattr(worker, "description", "")),
        "tools": sorted(str(name) for name in tools) if isinstance(tools, dict) else [],
    }


def _workforce_identity_record(workforce: Any) -> dict[str, Any]:
    context = current_context()
    workers = [
        {
            "logical_id": f"worker-{index}",
            "runtime_id": str(worker.node_id),
            "signature": _worker_signature(worker),
        }
        for index, worker in enumerate(workforce._children)
    ]
    if not workers or len({item["runtime_id"] for item in workers}) != len(workers):
        raise RuntimeError("Owl workforce runtime identities are empty or ambiguous")
    return {
        "schema_version": "native-agent-replay.owl-runtime-identities/v1",
        "actor_id": str(context["actor_id"]),
        "workers": workers,
    }


def _record_or_bind_workforce(workforce: Any) -> dict[str, str]:
    current = _workforce_identity_record(workforce)
    root = Path(os.environ["NATIVE_REPLAY_RUNTIME_IDENTITY_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{current['actor_id']}.json"
    if path.exists():
        if read_json(path) != current:
            raise RuntimeError("Owl workforce runtime identity inventory changed within one run")
    else:
        atomic_write_json(path, current)

    if os.environ.get("NATIVE_REPLAY_MODE") != "replay":
        return {}
    source_inventory = json.loads(os.environ["NATIVE_REPLAY_SOURCE_IDENTITY_BINDINGS"])
    source = source_inventory.get(current["actor_id"])
    if not isinstance(source, dict):
        raise RuntimeError(f"Owl replay has no source identity inventory for {current['actor_id']}")
    source_workers = source.get("workers")
    if not isinstance(source_workers, list) or len(source_workers) != len(current["workers"]):
        raise RuntimeError("Owl source/replay worker identity inventory size drift")

    mapping: dict[str, str] = {}
    for recorded, runtime in zip(source_workers, current["workers"], strict=True):
        if (
            recorded.get("logical_id") != runtime["logical_id"]
            or recorded.get("signature") != runtime["signature"]
        ):
            raise RuntimeError("Owl source/replay logical worker inventory drift")
        source_id = recorded.get("runtime_id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("Owl source worker runtime identity is invalid")
        mapping[source_id] = runtime["runtime_id"]
    return mapping


def _bind_assignee_response(response: Any, mapping: dict[str, str]) -> None:
    message = getattr(response, "msg", None)
    parsed = getattr(message, "parsed", None)
    assignee = getattr(parsed, "assignee_id", None)
    if isinstance(assignee, str):
        if assignee not in mapping:
            raise RuntimeError(f"Owl response selected an unknown source worker ID: {assignee!r}")
        parsed.assignee_id = mapping[assignee]
        return

    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            return
    if not isinstance(value, dict) or not isinstance(value.get("assignee_id"), str):
        return
    source_id = value["assignee_id"]
    if source_id not in mapping:
        raise RuntimeError(f"Owl response selected an unknown source worker ID: {source_id!r}")
    value["assignee_id"] = mapping[source_id]
    message.content = json.dumps(value, separators=(",", ":"))


def _find_assignee_factory(original):
    def wrapped(self, *args, **kwargs):
        mapping = _record_or_bind_workforce(self)
        if not mapping:
            return original(self, *args, **kwargs)
        coordinator = self.coordinator_agent
        original_step = coordinator.step

        def bound_step(*step_args, **step_kwargs):
            response = original_step(*step_args, **step_kwargs)
            _bind_assignee_response(response, mapping)
            return response

        coordinator.step = bound_step
        try:
            return original(self, *args, **kwargs)
        finally:
            coordinator.step = original_step

    return wrapped


def _tool_name(tool: Any) -> str:
    schema = getattr(tool, "openai_tool_schema", {})
    name = schema.get("function", {}).get("name") if isinstance(schema, dict) else None
    return str(name or getattr(tool.func, "__name__", "unknown"))


def _tool_implementation(tool: Any) -> str:
    function = tool.func
    return f"{function.__module__}.{getattr(function, '__qualname__', function.__name__)}"


def _tool_arguments(tool: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(tool.func).bind_partial(*args, **kwargs)
        return jsonable(dict(bound.arguments))
    except (TypeError, ValueError):
        return {"args": jsonable(args), "kwargs": jsonable(kwargs)}


def _cover_invoked_tool(tool: Any) -> tuple[str, str]:
    name = _tool_name(tool)
    implementation = _tool_implementation(tool)
    state().cover("implementation", implementation, tool_name=name)
    return name, implementation


def _is_composite_tool(tool: Any) -> bool:
    return _tool_implementation(tool) in _COMPOSITE_TOOL_IMPLEMENTATIONS


def _requested_tool(agent: Any, name: str) -> Any | None:
    tools = getattr(agent, "_internal_tools", {})
    return tools.get(name) if isinstance(tools, dict) else None


def _tool_timeout(tool: Any) -> float | None:
    owner = getattr(getattr(tool, "func", None), "__self__", None)
    value = getattr(owner, "timeout", None)
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


# ---- model-free primitives inside a top-level tool -------------------------
#
# owl already draws the line this adapter needs: a step whose complete input is in
# its arguments is a replayable primitive, and a step backed by a model is an LLM
# call. `browse_url` is not one operation but a loop of them — open, then observe
# (screenshot + a VL call) and act, then close — and the same holds for document
# and video extraction. Recording only the outer FunctionTool left all of that
# inside one opaque slot: 64 tool records where owl's own lane had 176, with the
# per-step arguments, ordering and outcomes invisible to the ledger.
#
# Every primitive below really executes on replay, exactly as before; what changes
# is that each is now a slot of its own, so its count, order, identity and returned
# observation are checked instead of assumed.


def _primitive_tool(name: str, implementation: str, arguments: dict[str, Any], invoke):
    state().cover("implementation", implementation, tool_name=name)
    return run_tool(
        name=name,
        implementation=implementation,
        arguments=arguments,
        invoke=invoke,
        result_encoder=encode_framework_output,
        result_contract=recorded_output_result_contract(),
        result_replayer=restore_framework_output,
    )


async def _primitive_tool_async(
    name: str,
    implementation: str,
    arguments: dict[str, Any],
    invoke,
    *,
    result_encoder=encode_framework_output,
    result_replayer=restore_framework_output,
):
    state().cover("implementation", implementation, tool_name=name)
    return await run_tool_async(
        name=name,
        implementation=implementation,
        arguments=arguments,
        invoke=invoke,
        result_encoder=result_encoder,
        result_contract=recorded_output_result_contract(),
        result_replayer=result_replayer,
    )


def _run_tool_primitive_factory(original):
    """owl's own wrapper for model-free primitives — document and video steps.

    It takes the callable, so the primitive becomes a slot without the adapter
    needing to know which toolkit raised it.
    """

    def wrapped(*args, **kwargs):
        name = str(kwargs.get("name", "primitive"))
        toolkit = str(kwargs.get("toolkit", "unknown"))
        function = kwargs.get("function")
        if function is None:
            return original(*args, **kwargs)
        return _primitive_tool(
            name,
            f"camel.replay_capture.{toolkit}.{name}",
            jsonable(kwargs.get("arguments", {})),
            lambda: original(*args, **kwargs),
        )

    return wrapped


def _browser_action_factory(original):
    """`async_act` — one model-free browser action, e.g. a click or a fill.

    Its `(success, info)` goes straight into the toolkit's trajectory history and
    from there into the next prompt, so the recorded outcome is what fixes the
    browser's onward control flow.
    """

    async def wrapped(self, action_code, *args, **kwargs):
        # A video-question action contains its own VLM request and model-free
        # extraction primitives.  It is orchestration for the same reason as
        # browse_url, so do not add an outer browser_action slot around them.
        from camel.toolkits.browser_toolkit import (
            MODEL_BACKED_BROWSER_ACTIONS,
            extract_function_name,
            normalize_browser_action_code,
        )

        normalized = normalize_browser_action_code(action_code)
        if extract_function_name(normalized) in MODEL_BACKED_BROWSER_ACTIONS:
            return await original(self, action_code, *args, **kwargs)
        return await _primitive_tool_async(
            "browser_action",
            "camel.toolkits.browser_toolkit.AsyncBaseBrowser.async_act",
            {"action_code": jsonable(normalized)},
            lambda: original(self, action_code, *args, **kwargs),
        )

    return wrapped


def _browser_init_factory(original):
    async def wrapped(self, *args, **kwargs):
        if current_context()["composite_lane"] is None:
            return await original(self, *args, **kwargs)
        return await _primitive_tool_async(
            "browser_open",
            "camel.toolkits.browser_toolkit.AsyncBaseBrowser.async_init",
            {"headless": bool(self.headless)},
            lambda: original(self, *args, **kwargs),
        )

    return wrapped


def _browser_visit_factory(original):
    async def wrapped(self, url, *args, **kwargs):
        context = current_context()
        # visit_page is also one possible browser_action implementation.  In
        # that case the enclosing action is already the primitive boundary.
        if context["composite_lane"] is None or context["tool_call_id"] is not None:
            return await original(self, url, *args, **kwargs)
        return await _primitive_tool_async(
            "browser_visit_page",
            "camel.toolkits.browser_toolkit.AsyncBaseBrowser.async_visit_page",
            {
                "url": jsonable(url),
                "timeout": jsonable(kwargs.get("timeout", 30000)),
                "max_retries": jsonable(kwargs.get("max_retries", 2)),
            },
            lambda: original(self, url, *args, **kwargs),
        )

    return wrapped


def _browser_observe_result(value: Any) -> dict[str, Any]:
    # The PIL image is native input to the following VLM call and is not a
    # framework observation we can safely substitute.  Record the durable path
    # identity while replay executes the native screenshot operation again.
    path = value[1] if isinstance(value, tuple) and len(value) == 2 else None
    return {"output": {"screenshot_path": jsonable(path)}}


def _keep_native_browser_observation(native: Any, _recorded: Any) -> Any:
    return native


def _browser_observe_factory(original):
    async def wrapped(self, save_image=False, *args, **kwargs):
        if current_context()["composite_lane"] is None:
            return await original(self, save_image, *args, **kwargs)
        return await _primitive_tool_async(
            "browser_observe",
            "camel.toolkits.browser_toolkit.AsyncBaseBrowser.async_get_som_screenshot",
            {"save_image": bool(save_image)},
            lambda: original(self, save_image, *args, **kwargs),
            result_encoder=_browser_observe_result,
            result_replayer=_keep_native_browser_observation,
        )

    return wrapped


def _browser_close_factory(original):
    async def wrapped(self, *args, **kwargs):
        return await _primitive_tool_async(
            "browser_close",
            "camel.toolkits.browser_toolkit.AsyncBrowserToolkit._close_browser_primitive",
            {},
            lambda: original(self, *args, **kwargs),
        )

    return wrapped


def _call_factory(original):
    def wrapped(self, *args, **kwargs):
        if _is_composite_tool(self):
            return original(self, *args, **kwargs)
        name, implementation = _cover_invoked_tool(self)
        return run_tool(
            name=name,
            implementation=implementation,
            arguments=_tool_arguments(self, args, kwargs),
            invoke=lambda: original(self, *args, **kwargs),
            result_encoder=encode_framework_output,
            result_contract=recorded_output_result_contract(),
            result_replayer=restore_framework_output,
            semantic_timeout_s=_tool_timeout(self),
        )

    return wrapped


def _async_call_factory(original):
    async def wrapped(self, *args, **kwargs):
        if _is_composite_tool(self):
            return await original(self, *args, **kwargs)
        name, implementation = _cover_invoked_tool(self)
        return await run_tool_async(
            name=name,
            implementation=implementation,
            arguments=_tool_arguments(self, args, kwargs),
            invoke=lambda: original(self, *args, **kwargs),
            result_encoder=encode_framework_output,
            result_contract=recorded_output_result_contract(),
            result_replayer=restore_framework_output,
            semantic_timeout_s=_tool_timeout(self),
        )

    return wrapped


def _execute_tool_factory(original):
    dispatcher_identity = f"{original.__module__}.{original.__qualname__}"

    def wrapped(self, tool_call_request):
        trigger = last_llm_attempt_id()
        if not isinstance(trigger, str) or not trigger:
            raise RuntimeError("Owl tool dispatch has no generating LLM attempt")
        name = str(tool_call_request.tool_name)
        tool = _requested_tool(self, name)
        if tool is not None and _is_composite_tool(tool):
            with composite_scope(
                name=name,
                model_call_id=str(tool_call_request.tool_call_id),
            ):
                return original(self, tool_call_request)
        return run_dispatch(
            name=name,
            arguments=jsonable(tool_call_request.args),
            parser_identity="camel.agents._types.ToolCallRequest",
            dispatcher_identity=dispatcher_identity,
            native_call_id=str(tool_call_request.tool_call_id),
            origin_kind="llm_structured",
            trigger_id=trigger,
            model_call_id=str(tool_call_request.tool_call_id),
            invoke=lambda: original(self, tool_call_request),
        )

    return wrapped


def _aexecute_tool_factory(original):
    dispatcher_identity = f"{original.__module__}.{original.__qualname__}"

    async def wrapped(self, tool_call_request):
        trigger = last_llm_attempt_id()
        if not isinstance(trigger, str) or not trigger:
            raise RuntimeError("Owl async tool dispatch has no generating LLM attempt")
        name = str(tool_call_request.tool_name)
        tool = _requested_tool(self, name)
        if tool is not None and _is_composite_tool(tool):
            with composite_scope(
                name=name,
                model_call_id=str(tool_call_request.tool_call_id),
            ):
                return await original(self, tool_call_request)
        return await run_dispatch_async(
            name=name,
            arguments=jsonable(tool_call_request.args),
            parser_identity="camel.agents._types.ToolCallRequest",
            dispatcher_identity=dispatcher_identity,
            native_call_id=str(tool_call_request.tool_call_id),
            origin_kind="llm_structured",
            trigger_id=trigger,
            model_call_id=str(tool_call_request.tool_call_id),
            invoke=lambda: original(self, tool_call_request),
        )

    return wrapped


def _record_chat_registry(agent: Any, phase: str) -> None:
    inventory: list[dict[str, Any]] = []
    for name, tool in sorted(agent._internal_tools.items()):
        implementation = _tool_implementation(tool)
        composite = _is_composite_tool(tool)
        registry_identity = f"camel.FunctionTool:{implementation}"
        state().cover("implementation", implementation, tool_name=str(name))
        state().cover("registry", registry_identity, tool_name=str(name), native_id=str(name))
        inventory.append(
            {
                "name": str(name),
                "native_id": str(name),
                "registry_identity": registry_identity,
                "implementation_identity": implementation,
                "dispatch_supported": not composite,
                "composite_scope": composite,
            }
        )
    for name, schema in sorted(agent._external_tool_schemas.items()):
        inventory.append(
            {
                "name": str(name),
                "native_id": str(name),
                "registry_identity": f"camel.external:{sha256_json(schema)}",
                "implementation_identity": None,
                "dispatch_supported": False,
            }
        )
    state().snapshot_registry("camel.agents.chat_agent.ChatAgent", inventory, phase=phase)


def _chat_init_factory(original):
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _record_chat_registry(self, "initialize")
        return result

    return wrapped


def _registry_mutation_factory(phase: str):
    def factory(original):
        def wrapped(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            _record_chat_registry(self, phase)
            return result

        return wrapped

    return factory


def _planning_factory(role: str):
    def factory(original):
        def wrapped(self, *args, **kwargs):
            with llm_scope(role):
                return original(self, *args, **kwargs)

        return wrapped

    return factory


def _role_target(role: str) -> str:
    try:
        values = json.loads(os.environ.get("NATIVE_REPLAY_ROLE_TARGETS", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Owl role target mapping: {exc}") from exc
    if not isinstance(values, dict):
        raise RuntimeError("NATIVE_REPLAY_ROLE_TARGETS must be a JSON object")
    target = values.get(role, current_context()["target_id"])
    if not isinstance(target, str) or not target:
        raise RuntimeError(f"Owl model role has no replay target: {role}")
    return target


def _model_run_factory(original):
    def wrapped(self, *args, **kwargs):
        role = _native_model_role(self)
        with llm_scope(role, target_id=_role_target(role)):
            return original(self, *args, **kwargs)

    return wrapped


def _model_arun_factory(original):
    async def wrapped(self, *args, **kwargs):
        role = _native_model_role(self)
        with llm_scope(role, target_id=_role_target(role)):
            return await original(self, *args, **kwargs)

    return wrapped


def _submit_factory(original):
    def wrapped(self, function, /, *args, **kwargs):
        if getattr(function, "__name__", "") != "_proc_run_one" or not args:
            return original(self, function, *args, **kwargs)
        task = args[0]
        if not isinstance(task, dict) or "task_id" not in task:
            raise RuntimeError("Owl native task submission has no task_id")
        # Completion order changes slightly between record and replay, so a
        # global refill submission number does not identify the same GAIA task.
        # The dataset task_id does: record binds every real task to the process
        # lane that ran it, and replay resolves that task back to the recorded
        # lane regardless of which runtime worker becomes free first.
        source_actor = str(task["task_id"])
        return original(
            self,
            gated_terminal_callable,
            function,
            source_actor,
            "owl-task",
            args,
            kwargs,
            "process-worker",
        )

    return wrapped


def _thread_submit_factory(original):
    def wrapped(self, function, /, *args, **kwargs):
        context = contextvars.copy_context()
        return original(self, context.run, function, *args, **kwargs)

    return wrapped


def _serial_source_actor() -> str:
    """Name the actor owl's serial path is about to run.

    Recording resolves no task list, so an actor names itself here exactly as it
    does on the pool path; only replay has an inventory to be pinned against. The
    index is the one identity available before the benchmark loads its tasks, and
    it is the same on both sides, so record and replay derive the same name.
    """

    try:
        actor_map = json.loads(os.environ.get("NATIVE_REPLAY_ACTOR_MAP", "{}"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Owl actor map: {exc}") from exc
    if not isinstance(actor_map, dict):
        raise RuntimeError("NATIVE_REPLAY_ACTOR_MAP must be a JSON object")
    if actor_map:
        if len(actor_map) != 1:
            raise RuntimeError("Owl C1 serial workload must declare exactly one actor")
        return str(next(iter(actor_map)))

    index = os.environ.get("GAIA_TEST_IDX", "").strip()
    if not index:
        raise RuntimeError(
            "Owl serial workload has no task index to name its actor from "
            "(GAIA_TEST_IDX is unset; the sweep normally exports it from --idx)"
        )
    if "," in index:
        raise RuntimeError(f"Owl serial workload expected a single task index, got {index!r}")
    return f"idx-{index}"


def _serial_workforce_factory(original):
    def wrapped(self, *args, **kwargs):
        if os.environ.get("NATIVE_REPLAY_SCOPE") != "C1":
            return original(self, *args, **kwargs)
        submission = next(_TASK_SUBMISSIONS)
        source_actor_id = (
            _serial_source_actor() if submission == 0 else f"refill-{submission:06d}"
        )
        return gated_terminal_callable(
            original,
            str(source_actor_id),
            "owl-task",
            (self, *args),
            kwargs,
            "serial-worker",
        )

    return wrapped


def install() -> None:
    importlib.import_module("camel.toolkits")
    from camel.agents.chat_agent import ChatAgent
    from camel.models.base_model import BaseModelBackend
    from camel.toolkits.browser_toolkit import AsyncBaseBrowser, AsyncBrowserToolkit
    from camel.toolkits.function_tool import FunctionTool
    from utils.enhanced_workforce import OwlWorkforce
    from utils.gaia import GAIABenchmark

    sync_dispatcher = method_identity(ChatAgent._execute_tool)
    async_dispatcher = method_identity(ChatAgent._aexecute_tool)

    patch_method(ChatAgent, "__init__", _chat_init_factory)
    patch_method(ChatAgent, "add_tool", _registry_mutation_factory("add_tool"))
    patch_method(ChatAgent, "add_external_tool", _registry_mutation_factory("add_external_tool"))
    patch_method(ChatAgent, "remove_tool", _registry_mutation_factory("remove_tool"))
    patch_method(
        ChatAgent,
        "remove_external_tool",
        _registry_mutation_factory("remove_external_tool"),
    )
    patch_method(FunctionTool, "__call__", _call_factory)
    patch_method(FunctionTool, "async_call", _async_call_factory)
    patch_method(ChatAgent, "_execute_tool", _execute_tool_factory)
    patch_method(ChatAgent, "_aexecute_tool", _aexecute_tool_factory)
    patch_method(OwlWorkforce, "_find_assignee", _find_assignee_factory)
    patch_method(BaseModelBackend, "run", _model_run_factory)
    patch_method(BaseModelBackend, "arun", _model_arun_factory)
    patch_method(AsyncBrowserToolkit, "_task_planning", _planning_factory("owl-browser-plan"))
    patch_method(
        AsyncBrowserToolkit,
        "_task_replanning",
        _planning_factory("owl-browser-replan"),
    )
    patch_method(
        AsyncBrowserToolkit,
        "_get_final_answer",
        _planning_factory("owl-browser-final"),
    )
    patch_method(GAIABenchmark, "run_workforce_with_retry", _serial_workforce_factory)

    # Model-free primitives inside the top-level tools. `browse_url` and the
    # document/video extractions are loops of these, and until now the whole loop
    # was one opaque slot.
    from camel.utils import replay_capture

    patch_method(replay_capture, "run_tool_primitive", _run_tool_primitive_factory)
    patch_method(AsyncBaseBrowser, "async_init", _browser_init_factory)
    patch_method(AsyncBaseBrowser, "async_visit_page", _browser_visit_factory)
    patch_method(
        AsyncBaseBrowser,
        "async_get_som_screenshot",
        _browser_observe_factory,
    )
    patch_method(AsyncBrowserToolkit, "async_act", _browser_action_factory)
    patch_method(AsyncBrowserToolkit, "_close_browser_primitive", _browser_close_factory)
    state().mark("owl-tool-primitive")
    state().mark("owl-browser-primitive")

    state().mark("owl-function-tool")
    state().mark("owl-composite-scope")
    state().mark("owl-dispatch-ledger")
    state().mark("owl-live-browser-approximate")
    state().mark("owl-model-routing")
    state().mark("owl-runtime-identity-binding")
    state().cover("parser", "camel.agents._types.ToolCallRequest")
    state().cover("dispatcher", sync_dispatcher)
    state().cover("dispatcher", async_dispatcher)

    if os.environ.get("NATIVE_REPLAY_PROCESS_ROLE", "framework") == "framework":
        patch_method(ProcessPoolExecutor, "submit", _submit_factory)
    patch_method(ThreadPoolExecutor, "submit", _thread_submit_factory)
    state().mark("owl-process-pool-gate")
    state().mark("owl-thread-context")
    state().mark("owl-serial-gate")
