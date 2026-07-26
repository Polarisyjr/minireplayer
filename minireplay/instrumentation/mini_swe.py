from __future__ import annotations

import contextvars
import json
import os
import threading
from functools import lru_cache
from time import monotonic_ns
from typing import Any

from minireplay.observation import recorded_output_result_contract
from minireplay.placement import declared_docker_cpuset
from minireplay.sdk import (
    BoundaryClient,
    capture_subprocess_launches,
    current_context,
    last_llm_attempt_id,
    llm_identity_headers,
    report_task_terminal,
    reset_context,
    run_dispatch,
    set_context,
)
from minireplay.serialization import jsonable

from .gate import bind_task_lane, ready_and_wait, target_for_actor
from .patching import method_identity, patch_method
from .state import state

_DISPATCH_QUEUE: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "minireplay_miniswe_dispatch_queue", default=None
)
_TOOL_RESULT_CONTRACT = recorded_output_result_contract()
_LOGICAL_CONTAINER_ID = "native-replay.container/mini-swe-workspace"

_TERMINAL_SNAPSHOT_COMMAND = r"""python3 - <<'PY'
import base64
import hashlib
import json
import os
import stat
import subprocess


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return base64.b64encode(value).decode("ascii")


head = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")
diff = subprocess.check_output(["git", "diff", "--binary", "--no-ext-diff", "HEAD"])
status = subprocess.check_output(
    ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all"]
)
untracked = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard", "-z"]
).split(b"\0")
entries = []
for path in sorted(item for item in untracked if item):
    metadata = os.lstat(path)
    entry = {"path_base64": encoded(path), "mode": stat.S_IMODE(metadata.st_mode)}
    if stat.S_ISREG(metadata.st_mode):
        content = open(path, "rb").read()
        entry.update({"kind": "file", "size": len(content), "sha256": digest(content)})
    elif stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        target_bytes = os.fsencode(target)
        entry.update({"kind": "symlink", "target_base64": encoded(target_bytes)})
    else:
        entry["kind"] = "other"
    entries.append(entry)
print(
    json.dumps(
        {
            "schema_version": "native-agent-replay.mini-swe-terminal/v1",
            "head": head,
            "tracked_diff_base64": encoded(diff),
            "tracked_diff_sha256": digest(diff),
            "status_base64": encoded(status),
            "status_sha256": digest(status),
            "untracked": entries,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY"""


@lru_cache(maxsize=1)
def _pin_cpu_hooks():
    """Load the optional external placement hook only for declared experiments."""

    if not os.environ.get("PIN_CPU_POLICY_PATH"):
        return None
    try:
        from pin_cpu import hooks
    except ImportError as exc:
        raise RuntimeError(
            "PIN_CPU_POLICY_PATH is set but the pin_cpu package is not importable"
        ) from exc
    return hooks


def _environment_factory(original):
    def wrapped(config, instance):
        source_actor = str(instance.get("instance_id", ""))
        if not source_actor:
            raise RuntimeError("mini-swe environment has no instance_id")
        actor = bind_task_lane(source_actor, f"executor-thread:{threading.get_ident()}")
        tokens = set_context(
            actor_id=actor,
            process_role="mini-swe-environment-setup",
            session_id=source_actor,
            llm_role="agent",
            target_id=target_for_actor(actor),
        )
        try:
            return original(config, instance)
        finally:
            reset_context(tokens)

    return wrapped


def _container_start_factory(original):
    def wrapped(self, *args, **kwargs):
        run_id = os.environ["NATIVE_REPLAY_RUN_ID"]
        label = f"native-replay.run={run_id}"
        run_args = list(self.config.run_args)
        actor = current_context()["actor_id"]
        cpuset = declared_docker_cpuset()
        if cpuset is not None:
            if any(
                value == "--cpuset-cpus" or value.startswith("--cpuset-cpus=") for value in run_args
            ):
                raise RuntimeError("mini-swe Docker run already declares a CPU set")
            run_args.extend(["--cpuset-cpus", cpuset])
        hooks = _pin_cpu_hooks()
        placement = None
        if hooks is not None:
            if cpuset is not None:
                raise RuntimeError(
                    "pin_cpu placement cannot be combined with config.cpuset"
                )
            placement = hooks.container_placement(str(actor))
            for option, value in (
                ("--cpuset-cpus", placement["cpus"]),
                ("--cpuset-mems", placement["mems"]),
            ):
                if value is None:
                    continue
                if any(
                    item == option or item.startswith(f"{option}=") for item in run_args
                ):
                    raise RuntimeError(
                        f"mini-swe Docker run already declares {option}"
                    )
                run_args.extend([option, str(value)])
            for name, value in sorted(placement["labels"].items()):
                run_args.extend(["--label", f"{name}={value}"])
            for name, value in sorted(placement.get("env", {}).items()):
                if any(
                    item == "--env" and following.split("=", 1)[0] == name
                    for item, following in zip(run_args, run_args[1:], strict=False)
                ):
                    raise RuntimeError(
                        f"mini-swe Docker run already declares --env {name}"
                    )
                run_args.extend(["--env", f"{name}={value}"])
            cgroup_parent = placement.get("cgroup_parent")
            if cgroup_parent is not None:
                if any(
                    item == "--cgroup-parent"
                    or item.startswith("--cgroup-parent=")
                    for item in run_args
                ):
                    raise RuntimeError(
                        "mini-swe Docker run already declares --cgroup-parent"
                    )
                run_args.extend(["--cgroup-parent", str(cgroup_parent)])
        for index, value in enumerate(run_args):
            if (
                value == "--label"
                and index + 1 < len(run_args)
                and run_args[index + 1].startswith("native-replay.run=")
            ):
                raise RuntimeError("mini-swe Docker run already has a replay ownership label")
        run_args.extend(["--label", label, "--label", "native-replay.role=mini-swe-workspace"])
        if actor != "unknown":
            run_args.extend(["--label", f"native-replay.actor={actor}"])
        self.config.run_args = run_args
        self._pin_cpu_actor = str(actor) if hooks is not None and placement is not None else None
        try:
            return original(self, *args, **kwargs)
        except Exception:
            if hooks is not None and self._pin_cpu_actor is not None:
                hooks.release_container(self._pin_cpu_actor)
                self._pin_cpu_actor = None
            raise

    return wrapped


def _container_cleanup_factory(original):
    def wrapped(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            actor = getattr(self, "_pin_cpu_actor", None)
            if actor is not None:
                hooks = _pin_cpu_hooks()
                if hooks is not None:
                    hooks.release_container(actor)
                self._pin_cpu_actor = None

    return wrapped


def _terminal_workspace_state(environment, native_execute) -> dict[str, Any]:
    result = native_execute(
        environment,
        {"command": _TERMINAL_SNAPSHOT_COMMAND},
        timeout=120,
    )
    if result.get("returncode") != 0 or result.get("exception_info"):
        raise RuntimeError(f"mini-swe terminal workspace capture failed: {result}")
    try:
        value = json.loads(result["output"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("mini-swe terminal workspace capture returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("mini-swe terminal workspace capture returned a non-object")
    return value


def _terminal_result(result: Any, workspace: dict[str, Any]) -> dict[str, Any]:
    value = jsonable(result)
    if isinstance(value, dict):
        return {**value, "native_workspace": workspace}
    return {"framework_result": value, "native_workspace": workspace}


def _logical_container_children(
    children: list[dict[str, Any]], container_id: str | None
) -> list[dict[str, Any]]:
    if not container_id:
        raise RuntimeError("mini-swe native child evidence has no container ID")
    normalized: list[dict[str, Any]] = []
    for child in children:
        value = jsonable(child)
        command = value.get("command") if isinstance(value, dict) else None
        if isinstance(command, list):
            value["command"] = [
                _LOGICAL_CONTAINER_ID if item == container_id else item for item in command
            ]
        normalized.append(value)
    return normalized


def _restore_tool_observation(native_result: Any, recorded_result: Any) -> Any:
    if not isinstance(native_result, dict) or not isinstance(recorded_result, dict):
        raise RuntimeError("mini-swe replay cannot restore a non-object tool observation")
    return dict(recorded_result)


def _agent_run_factory(original, native_execute):
    def wrapped(self, *args, **kwargs):
        source_actor = getattr(self, "instance_id", "")
        if not source_actor:
            raise RuntimeError("mini-swe native replay requires an instance_id")
        tokens = ready_and_wait(
            source_actor,
            native_lane_key=f"executor-thread:{threading.get_ident()}",
            process_role="mini-swe-agent",
            session_id=source_actor,
            llm_role="agent",
        )
        try:
            try:
                result = original(self, *args, **kwargs)
            except Exception as exc:
                completed_at_ns = monotonic_ns()
                workspace = _terminal_workspace_state(self.env, native_execute)
                report_task_terminal(
                    status="failure",
                    completed_at_ns=completed_at_ns,
                    task={"source_actor_id": source_actor},
                    result={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "native_workspace": workspace,
                    },
                )
                raise
            completed_at_ns = monotonic_ns()
            workspace = _terminal_workspace_state(self.env, native_execute)
            report_task_terminal(
                result=_terminal_result(result, workspace),
                completed_at_ns=completed_at_ns,
                task={"source_actor_id": source_actor},
            )
            return result
        finally:
            reset_context(tokens)

    return wrapped


def _docker_execute_factory(original):
    implementation = method_identity(original)

    def child_evidence(self, action, cwd, context, exit_code):
        command = str(action.get("command", ""))
        daemon_command = " ".join([*map(str, self.config.interpreter), command])
        return [
            {
                "kind": "container-exec",
                "container_role": "mini-swe-workspace",
                "owner_actor": str(context["actor_id"]),
                "launcher": "mini-swe.DockerEnvironment.execute",
                "command": jsonable(action),
                "daemon_command": daemon_command,
                "daemon_exit_code": exit_code,
                "cwd": jsonable(cwd or self.config.cwd),
                "shell": False,
                "executable": None,
            }
        ]

    def wrapped(self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None):
        context = current_context()
        if context["actor_id"] == "unknown":
            return original(self, action, cwd, timeout=timeout)
        queue = _DISPATCH_QUEUE.get()
        if queue is None and context["process_role"] == "mini-swe-environment-setup":
            return original(self, action, cwd, timeout=timeout)
        if not queue:
            raise RuntimeError("mini-swe environment execution escaped execute_actions dispatch")
        dispatch = queue.pop(0)
        arguments = {
            "action": jsonable(action),
            "cwd": jsonable(cwd),
            "timeout": timeout,
        }
        if arguments != dispatch["arguments"]:
            raise RuntimeError("mini-swe parsed action changed before environment dispatch")

        def execute_native():
            boundary = BoundaryClient()
            semantic_timeout_s = float(timeout or self.config.timeout)
            reservation = boundary.start(
                "tool",
                dispatch_id=str(current_context()["dispatch_id"]),
                name="docker.exec",
                implementation=implementation,
                arguments=arguments,
                result_contract=_TOOL_RESULT_CONTRACT,
                semantic_timeout_s=semantic_timeout_s,
            )
            native_children: list[dict[str, Any]] = []
            original_interpreter = None
            if os.environ.get("PIN_CPU_TOOL_AFFINITY") == "1":
                hooks = _pin_cpu_hooks()
                if hooks is None:
                    raise RuntimeError(
                        "PIN_CPU_TOOL_AFFINITY needs a declared pin_cpu policy"
                    )
                tool_cpus = hooks.tool_cpu_set(action)
                if tool_cpus is None:
                    raise RuntimeError(
                        "PIN_CPU_TOOL_AFFINITY is enabled for a non-tool policy"
                    )
                original_interpreter = list(self.config.interpreter)
                if original_interpreter[:1] == ["taskset"]:
                    # The prefix is applied by mutating a field shared by the whole
                    # environment, which is safe only because mini-swe runs one
                    # action at a time per container. A nested entry would silently
                    # bind this action to the outer action's CPUs and leak the
                    # prefix into every later action, so fail instead.
                    raise RuntimeError(
                        "mini-swe interpreter already carries a pin_cpu affinity prefix"
                    )
                # taskset sets affinity and execs the existing interpreter in the
                # same container process. There is no host wrapper or polling
                # controller in this measured path.
                self.config.interpreter = [
                    "taskset",
                    "-c",
                    tool_cpus,
                    *original_interpreter,
                ]
            try:
                with capture_subprocess_launches() as native_children:
                    result = original(self, action, cwd, timeout=timeout)
            except Exception as exc:
                # Submitted is an intentional terminal control-flow exception. Its
                # native payload is still the result of the command that just ran.
                terminal = exc.__class__.__name__ == "Submitted"
                boundary.complete(
                    reservation["reservation_id"],
                    status="ok" if terminal else "error",
                    result=(
                        {"terminal_exception": terminal, "payload": jsonable(exc.args)}
                        if terminal
                        else {"error_type": type(exc).__name__, "message": str(exc)}
                    ),
                    logical_frames=[],
                    side_effects={},
                    child_processes=[
                        *child_evidence(self, action, cwd, context, 0),
                        *_logical_container_children(native_children, self.container_id),
                    ],
                    native_execution=True,
                )
                raise
            finally:
                if original_interpreter is not None:
                    self.config.interpreter = original_interpreter
            extra = result.get("extra")
            completion = boundary.complete(
                reservation["reservation_id"],
                status=(
                    "timeout"
                    if isinstance(extra, dict)
                    and extra.get("exception_type") == "TimeoutExpired"
                    else "ok"
                ),
                result=jsonable(result),
                logical_frames=[],
                side_effects={},
                child_processes=[
                    *child_evidence(
                        self,
                        action,
                        cwd,
                        context,
                        int(result.get("returncode", -1)),
                    ),
                    *_logical_container_children(native_children, self.container_id),
                ],
                native_execution=True,
            )
            if completion.get("result_replay_required") is True:
                result = _restore_tool_observation(
                    result,
                    completion.get("framework_result"),
                )
            return result

        return run_dispatch(
            name="docker.exec",
            arguments=arguments,
            parser_identity=str(dispatch["parser_identity"]),
            dispatcher_identity=implementation,
            native_call_id=str(dispatch["native_call_id"]),
            origin_kind=str(dispatch["origin_kind"]),
            trigger_id=str(dispatch["trigger_id"]),
            model_call_id=dispatch.get("model_call_id"),
            invoke=execute_native,
        )

    return wrapped


def _execute_actions_factory(original):
    parser_identity = method_identity(original)

    def wrapped(self, message):
        if _DISPATCH_QUEUE.get() is not None:
            raise RuntimeError("nested mini-swe execute_actions dispatch queue")
        trigger = last_llm_attempt_id()
        if not isinstance(trigger, str) or not trigger:
            raise RuntimeError("mini-swe parsed actions have no generating LLM attempt")
        actions = message.get("extra", {}).get("actions", [])
        queue = []
        for index, action in enumerate(actions):
            canonical = jsonable(action)
            model_call_id = action.get("tool_call_id") if isinstance(action, dict) else None
            queue.append(
                {
                    "arguments": {"action": canonical, "cwd": "", "timeout": None},
                    "parser_identity": parser_identity,
                    "native_call_id": str(model_call_id or f"{trigger}:action:{index}"),
                    "origin_kind": "llm_structured" if model_call_id else "llm_parsed",
                    "trigger_id": trigger,
                    "model_call_id": model_call_id,
                }
            )
        token = _DISPATCH_QUEUE.set(queue)
        try:
            result = original(self, message)
            if queue:
                raise RuntimeError(f"mini-swe left {len(queue)} parsed actions undispatched")
            return result
        finally:
            _DISPATCH_QUEUE.reset(token)

    return wrapped


def _parse_actions_factory(original, format_error_type):
    parser_identity = method_identity(original)

    def wrapped(self, response):
        try:
            return original(self, response)
        except format_error_type as exc:
            trigger = last_llm_attempt_id()
            if not isinstance(trigger, str) or not trigger:
                raise RuntimeError("mini-swe rejected model calls have no LLM attempt") from exc
            choices = getattr(response, "choices", None)
            choice = choices[0] if isinstance(choices, list) and choices else None
            message = getattr(choice, "message", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            rejection = jsonable(getattr(exc, "messages", str(exc)))
            for index, tool_call in enumerate(tool_calls):
                call_id = getattr(tool_call, "id", None)
                if not isinstance(call_id, str) or not call_id:
                    raise RuntimeError(
                        "mini-swe rejected structured call has no provider call ID"
                    ) from exc
                run_dispatch(
                    name="mini-swe.parser-rejection",
                    arguments={
                        "choice_index": 0,
                        "tool_index": index,
                        "tool_call": jsonable(tool_call),
                        "format_error": rejection,
                    },
                    parser_identity=parser_identity,
                    dispatcher_identity=parser_identity,
                    native_call_id=call_id,
                    origin_kind="llm_structured",
                    trigger_id=trigger,
                    model_call_id=call_id,
                    invoke=lambda: None,
                )
            raise

    return wrapped


def _litellm_factory(original):
    def wrapped(*args, **kwargs):
        headers = dict(kwargs.get("extra_headers") or {})
        headers.update(llm_identity_headers())
        kwargs["extra_headers"] = headers
        return original(*args, **kwargs)

    return wrapped


def install() -> None:
    import minisweagent.run.benchmarks.swebench as swebench
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.exceptions import FormatError
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent

    parser_identity = method_identity(DefaultAgent.execute_actions)
    rejection_parser_identity = method_identity(LitellmModel._parse_actions)
    execution_identity = method_identity(DockerEnvironment.execute)

    patch_method(
        ProgressTrackingAgent,
        "run",
        lambda original: _agent_run_factory(original, DockerEnvironment.execute),
    )
    patch_method(swebench, "get_sb_environment", _environment_factory)
    patch_method(DefaultAgent, "execute_actions", _execute_actions_factory)
    patch_method(
        LitellmModel,
        "_parse_actions",
        lambda original: _parse_actions_factory(original, FormatError),
    )
    patch_method(DockerEnvironment, "_start_container", _container_start_factory)
    if _pin_cpu_hooks() is not None:
        # Only a declared placement experiment needs the slot to be returned, and
        # only it may depend on this framework version owning a ``cleanup`` method.
        patch_method(DockerEnvironment, "cleanup", _container_cleanup_factory)
    patch_method(DockerEnvironment, "execute", _docker_execute_factory)
    state().mark("mini-swe-agent-gate")
    state().mark("mini-swe-environment-actor-binding")
    state().mark("mini-swe-docker-execute")
    state().mark("mini-swe-dispatch-ledger")
    state().mark("mini-swe-parser-rejection-ledger")
    state().mark("mini-swe-docker-ownership")
    state().cover("parser", parser_identity)
    state().cover("parser", rejection_parser_identity)
    state().cover("dispatcher", rejection_parser_identity)
    state().cover("dispatcher", execution_identity)
    state().cover("implementation", execution_identity)
    state().cover(
        "registry",
        "minisweagent.environments.docker.DockerEnvironment.action-schema",
        tool_name="docker.exec",
        native_id="execute_actions.action",
    )
    state().snapshot_registry(
        "minisweagent.DefaultAgent.actions",
        [
            {
                "name": "docker.exec",
                "registry_identity": (
                    "minisweagent.environments.docker.DockerEnvironment.action-schema"
                ),
                "implementation_identity": execution_identity,
                "dispatch_supported": True,
            }
        ],
        phase="install",
    )

    import litellm

    patch_method(litellm, "completion", _litellm_factory)
    state().mark("mini-swe-litellm-identity")
