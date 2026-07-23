from __future__ import annotations

import contextvars
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from minireplay.observation import recorded_output_result_contract
from minireplay.placement import declared_docker_cpuset
from minireplay.sdk import (
    current_context,
    last_llm_attempt_id,
    record_container_shell_command,
    record_persistent_process_command,
    run_dispatch,
    run_dispatch_async,
    run_tool,
    run_tool_async,
)
from minireplay.serialization import jsonable

from .gate import gated_terminal_callable, resolve_actor
from .patching import method_identity, patch_method
from .result_replay import restore_object_field
from .state import state


def _result_status(result) -> str:
    success = getattr(result, "success", None)
    error_code = getattr(result, "error_code", None)
    error = str(getattr(result, "error", "") or "").lower()
    if "timed out" in error or "timeout" in error:
        return "timeout"
    return "error" if success is False or error_code not in {None, 0} else "ok"


def _native_tool_timeout(tool) -> float | None:
    if type(tool).__name__ != "BashTool":
        return None
    session = getattr(tool, "_session", None)
    value = getattr(session, "_timeout", 120.0)
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _semantic_tool_result(result):
    payload = jsonable(result)
    if isinstance(payload, dict):
        payload.pop("started_at_ns", None)
        payload.pop("ended_at_ns", None)
    return payload


def _docker_shell_factory(original):
    implementation = method_identity(original)

    def wrapped(self, command, timeout=300):
        result = original(self, command, timeout)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], int)
            or isinstance(result[0], bool)
        ):
            raise RuntimeError("Trae persistent shell returned an invalid execution receipt")
        record_container_shell_command(
            launcher=implementation,
            container_role="trae-container-agent",
            command=str(command),
            exit_code=result[0],
            shell_executable="/bin/bash",
        )
        return result

    return wrapped


def _bash_session_run_factory(original):
    implementation = method_identity(original)

    async def wrapped(self, command):
        result = await original(self, command)
        process = getattr(self, "_process", None)
        runtime_pid = getattr(process, "pid", None)
        exit_code = getattr(result, "error_code", None)
        record_persistent_process_command(
            launcher=implementation,
            container_role="trae-container-agent",
            command=str(command),
            exit_code=exit_code,
            shell_executable=str(getattr(self, "command", "/bin/bash")),
            runtime_pid=runtime_pid,
        )
        return result

    return wrapped


def _dispatch_factory(original):
    implementation = method_identity(original)

    async def wrapped(self, tool_call):
        trigger = last_llm_attempt_id()
        if not isinstance(trigger, str) or not trigger:
            raise RuntimeError("Trae tool dispatch has no generating LLM attempt")
        return await run_dispatch_async(
            name=str(tool_call.name),
            arguments=jsonable(tool_call.arguments),
            parser_identity="trae_agent.tools.base.ToolCall",
            dispatcher_identity=implementation,
            native_call_id=str(tool_call.call_id),
            origin_kind="llm_structured",
            trigger_id=trigger,
            model_call_id=str(tool_call.call_id),
            invoke=lambda: original(self, tool_call),
        )

    return wrapped


def _native_tool_factory(original):
    implementation = method_identity(original)
    tool_name = original.__qualname__.rsplit(".", 1)[0]

    async def wrapped(self, arguments):
        return await run_tool_async(
            name=str(getattr(self, "name", tool_name)),
            implementation=implementation,
            arguments=jsonable(arguments),
            invoke=lambda: original(self, arguments),
            result_encoder=_semantic_tool_result,
            result_status=_result_status,
            semantic_timeout_s=_native_tool_timeout(self),
            result_contract=recorded_output_result_contract("/output"),
            result_replayer=lambda native, recorded: restore_object_field(
                native, recorded, "output"
            ),
        )

    return wrapped


def _executor_init_factory(original):
    def wrapped(self, tools, *args, **kwargs):
        result = original(self, tools, *args, **kwargs)
        inventory = []
        for tool in tools:
            patch_method(type(tool), "execute", _native_tool_factory)
            implementation = method_identity(type(tool).execute)
            registry_identity = f"{type(tool).__module__}.{type(tool).__qualname__}"
            tool_name = str(tool.name)
            state().cover("implementation", implementation, tool_name=tool_name)
            state().cover(
                "registry",
                registry_identity,
                tool_name=tool_name,
                native_id=self._normalize_name(tool_name),
            )
            inventory.append(
                {
                    "name": tool_name,
                    "native_id": self._normalize_name(tool_name),
                    "registry_identity": registry_identity,
                    "implementation_identity": implementation,
                    "dispatch_supported": True,
                }
            )
        state().snapshot_registry(
            "trae_agent.tools.base.ToolExecutor",
            inventory,
            phase="initialize",
        )
        return result

    return wrapped


def _docker_tool_factory(original):
    implementation = method_identity(original)

    def wrapped(self, tool_call):
        trigger = last_llm_attempt_id()
        if not isinstance(trigger, str) or not trigger:
            raise RuntimeError("Trae Docker tool dispatch has no generating LLM attempt")
        arguments = jsonable(tool_call.arguments)
        return run_dispatch(
            name=str(tool_call.name),
            arguments=arguments,
            parser_identity="trae_agent.tools.base.ToolCall",
            dispatcher_identity=implementation,
            native_call_id=str(tool_call.call_id),
            origin_kind="llm_structured",
            trigger_id=trigger,
            model_call_id=str(tool_call.call_id),
            invoke=lambda: run_tool(
                name=str(tool_call.name),
                implementation=implementation,
                arguments=arguments,
                invoke=lambda: original(self, tool_call),
                result_encoder=_semantic_tool_result,
                result_status=_result_status,
                semantic_timeout_s=300.0,
                result_contract=recorded_output_result_contract("/result"),
                result_replayer=lambda native, recorded: restore_object_field(
                    native, recorded, "result"
                ),
            ),
        )

    return wrapped


def _argument(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _instance_from_path(command: list[str]) -> str | None:
    for flag in ("--candidates", "--candidate_path", "--output", "--result_path"):
        value = _argument(command, flag)
        if value is None:
            continue
        parts = Path(value).parts
        if "per_instance" in parts:
            index = parts.index("per_instance")
            if index + 1 < len(parts):
                return parts[index + 1]
    return None


def _subprocess_factory(original):
    def wrapped(command, *args, **kwargs):
        if not isinstance(command, (list, tuple)) or not command:
            return original(command, *args, **kwargs)
        values = [str(value) for value in command]
        module = _argument(values, "-m")
        source_actor = _argument(values, "--instance_ids") or _instance_from_path(values)
        if source_actor is None:
            return original(command, *args, **kwargs)
        actor = resolve_actor(source_actor)
        if module == "evaluation.generate_candidates_swebench":
            logical_role = "generate"
            process_role = "trae-generate"
        elif module == "evaluation.regression_test_swebench":
            logical_role = "prune"
            process_role = "trae-prune"
        elif any(Path(value).name == "selector.py" for value in values):
            logical_role = "select"
            process_role = "trae-select"
        else:
            return original(command, *args, **kwargs)
        role_targets = json.loads(os.environ.get("NATIVE_REPLAY_ROLE_TARGETS", "{}"))
        target_map = json.loads(os.environ.get("NATIVE_REPLAY_TARGET_MAP", "{}"))
        target = role_targets.get(logical_role, role_targets.get(process_role))
        if target is None:
            target = target_map.get(actor, os.environ.get("NATIVE_REPLAY_TARGET_ID", "default"))
        environment = dict(kwargs.get("env") or os.environ)
        environment.update(
            {
                "NATIVE_REPLAY_ACTOR_ID": actor,
                "NATIVE_REPLAY_SESSION_ID": actor,
                "NATIVE_REPLAY_TARGET_ID": target,
                "NATIVE_REPLAY_PROCESS_ROLE": process_role,
            }
        )
        values = _rewrite_stage_endpoints(values, actor)
        kwargs["env"] = environment
        return original(values, *args, **kwargs)

    return wrapped


def _rewrite_stage_endpoints(command: list[str], actor: str) -> list[str]:
    values = list(command)
    proxy = os.environ["NATIVE_REPLAY_PROXY_URL"].rstrip("/") + "/v1"
    if "--base-urls" in values:
        values[values.index("--base-urls") + 1] = proxy
    for flag in ("--config-file", "--config_file"):
        config = _argument(values, flag)
        if config is None:
            continue
        source = Path(config)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        providers = payload.get("model_providers") if isinstance(payload, dict) else None
        if not isinstance(providers, dict) or not providers:
            raise RuntimeError(f"Trae config has no model provider inventory: {source}")
        changed = 0
        for provider in providers.values():
            if isinstance(provider, dict) and "base_url" in provider:
                provider["base_url"] = proxy
                changed += 1
        if changed == 0:
            raise RuntimeError(f"Trae config has no base_url to redirect: {source}")
        root = Path(os.environ["NATIVE_REPLAY_RUN_ROOT"]) / "trae-configs"
        root.mkdir(parents=True, exist_ok=True)
        stage = "generate" if "generate_candidates" in " ".join(values) else Path(values[0]).stem
        target = root / f"{actor}-{stage}.yaml"
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        values[values.index(flag) + 1] = str(target)
    return values


def _submit_factory(original):
    def wrapped(self, function, /, *args, **kwargs):
        if (
            os.environ.get("NATIVE_REPLAY_PROCESS_ROLE", "framework") == "framework"
            and getattr(function, "__name__", "") == "tracked_run"
            and args
        ):
            source_actor = str(args[0])
            return original(
                self,
                gated_terminal_callable,
                function,
                source_actor,
                "trae-chain",
                args,
                kwargs,
            )
        context = contextvars.copy_context()
        return original(self, context.run, function, *args, **kwargs)

    return wrapped


def _docker_run_factory(original):
    package_root = Path(os.environ["NATIVE_REPLAY_PACKAGE_ROOT"])
    run_root = Path(os.environ["NATIVE_REPLAY_RUN_ROOT"])

    def wrapped(self, *args, **kwargs):
        labels = dict(kwargs.get("labels") or {})
        existing = labels.get("native-replay.run")
        run_id = os.environ["NATIVE_REPLAY_RUN_ID"]
        if existing is not None and existing != run_id:
            raise RuntimeError("Trae Docker container has a conflicting replay owner")
        context = current_context()
        actor = context["actor_id"]
        labels["native-replay.run"] = run_id
        labels["native-replay.role"] = (
            "trae-setup" if actor == "unknown" else "trae-container-agent"
        )
        if actor != "unknown":
            labels["native-replay.actor"] = str(actor)
        kwargs["labels"] = labels
        cpuset = declared_docker_cpuset()
        if cpuset is not None:
            existing_cpuset = kwargs.get("cpuset_cpus")
            if existing_cpuset not in {None, cpuset}:
                raise RuntimeError("Trae Docker run conflicts with the declared CPU set")
            kwargs["cpuset_cpus"] = cpuset
        if actor == "unknown":
            gate = os.environ.get("NATIVE_REPLAY_START_GATE")
            if gate and Path(gate).is_file():
                raise RuntimeError("post-gate Trae container has no native actor identity")
            return original(self, *args, **kwargs)

        parsed = urlsplit(os.environ["NATIVE_REPLAY_PROXY_URL"])
        if parsed.hostname is None or parsed.hostname.startswith("127."):
            raise RuntimeError("Trae container replay requires a container-reachable proxy")
        volumes = dict(kwargs.get("volumes") or {})
        for source in (package_root, run_root):
            if str(source) in volumes:
                raise RuntimeError(f"Trae container already mounts replay path: {source}")
        volumes[str(package_root)] = {"bind": "/opt/native-replay", "mode": "ro"}
        volumes[str(run_root)] = {"bind": "/opt/native-replay-run", "mode": "rw"}
        kwargs["volumes"] = volumes

        environment = kwargs.get("environment") or {}
        if not isinstance(environment, dict):
            raise RuntimeError("Trae replay Docker environment must be a mapping")
        environment = {str(key): str(value) for key, value in environment.items()}
        forwarded = {
            name: value
            for name, value in os.environ.items()
            if name.startswith("NATIVE_REPLAY_")
            and name
            not in {
                "NATIVE_REPLAY_READY_DIR",
                "NATIVE_REPLAY_START_GATE",
                "NATIVE_REPLAY_TERMINAL_DIR",
                "NATIVE_REPLAY_INSTRUMENTATION_STATUS_DIR",
                "NATIVE_REPLAY_PACKAGE_ROOT",
                "NATIVE_REPLAY_RUN_ROOT",
            }
        }
        forwarded.update(
            {
                "NATIVE_REPLAY_ACTOR_ID": str(actor),
                "NATIVE_REPLAY_SESSION_ID": str(context["session_id"]),
                "NATIVE_REPLAY_PROCESS_ROLE": "trae-container-agent",
                "NATIVE_REPLAY_CONTAINER_ROLE": "trae-container-agent",
                "NATIVE_REPLAY_TARGET_ID": str(context["target_id"]),
                "NATIVE_REPLAY_INSTRUMENTATION_STATUS_DIR": (
                    "/opt/native-replay-run/stage/instrumentation"
                ),
                "PYTHONPATH": "/opt/native-replay/bootstrap:/opt/native-replay",
                "OPENAI_BASE_URL": os.environ["OPENAI_BASE_URL"],
                "OPENAI_API_BASE": os.environ["OPENAI_API_BASE"],
                "LITELLM_BASE_URL": os.environ["LITELLM_BASE_URL"],
            }
        )
        environment.update(forwarded)
        kwargs["environment"] = environment
        return original(self, *args, **kwargs)

    return wrapped


def install() -> None:
    from docker.models.containers import ContainerCollection
    from trae_agent.agent.docker_manager import DockerManager
    from trae_agent.tools.base import ToolExecutor
    from trae_agent.tools.bash_tool import _BashSession
    from trae_agent.tools.docker_tool_executor import DockerToolExecutor

    dispatcher_identity = method_identity(ToolExecutor.execute_tool_call)
    docker_dispatcher_identity = method_identity(DockerToolExecutor._execute_in_docker)
    docker_shell_identity = method_identity(DockerManager._execute_interactive)

    patch_method(ToolExecutor, "__init__", _executor_init_factory)
    patch_method(ToolExecutor, "execute_tool_call", _dispatch_factory)
    patch_method(DockerToolExecutor, "_execute_in_docker", _docker_tool_factory)
    patch_method(DockerManager, "_execute_interactive", _docker_shell_factory)
    patch_method(_BashSession, "run", _bash_session_run_factory)
    state().mark("trae-tool-executor")
    state().mark("trae-dispatch-ledger")
    state().mark("trae-docker-tool-executor")
    state().mark("trae-persistent-shell-receipt")
    state().mark("trae-bash-session-receipt")
    state().cover("parser", "trae_agent.tools.base.ToolCall")
    state().cover("dispatcher", dispatcher_identity)
    state().cover("dispatcher", docker_dispatcher_identity)
    state().cover("implementation", docker_dispatcher_identity)
    state().cover("implementation", docker_shell_identity)

    role = os.environ.get("NATIVE_REPLAY_PROCESS_ROLE", "framework")
    patch_method(ThreadPoolExecutor, "submit", _submit_factory)
    state().mark("trae-thread-context")
    if role != "trae-container-agent":
        patch_method(ContainerCollection, "run", _docker_run_factory)
        state().mark("trae-docker-ownership")

    if role in {
        "framework",
        "trae-orchestrator",
    }:
        patch_method(subprocess, "run", _subprocess_factory)
        state().mark("trae-stage-gate")
        state().mark("trae-chain-gate")
