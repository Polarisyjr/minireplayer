from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import os
from pathlib import Path

from minireplay.sdk import (
    BoundaryClient,
    current_context,
    report_task_terminal,
    reset_context,
    run_grader,
    set_context,
)
from minireplay.serialization import jsonable
from minireplay.util import sha256_json

from .gate import resolve_actor
from .patching import method_identity, patch_method
from .state import state

_SPAWN_ENV: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "minireplay_coral_spawn_env", default=None
)


def _plugin_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets/opencode_minireplay_plugin.js"


def _target_for(actor: str) -> str:
    value = json.loads(os.environ.get("NATIVE_REPLAY_TARGET_MAP", "{}"))
    target = value.get(actor)
    if target is None and "--agent-" in actor:
        agent = actor.rsplit("--agent-", 1)[1]
        roles = json.loads(os.environ.get("NATIVE_REPLAY_ROLE_TARGETS", "{}"))
        target = roles.get(f"agent-{agent}")
    target = target or os.environ.get("NATIVE_REPLAY_TARGET_ID", "default")
    if not isinstance(target, str) or not target:
        raise RuntimeError(f"invalid CORAL target for {actor}")
    return target


def _task_source_id() -> str:
    value = os.environ.get("NATIVE_REPLAY_CORAL_TASK_ID")
    if not isinstance(value, str) or not value:
        raise RuntimeError("CORAL replay process has no task identity")
    return value


def _agent_source_id(agent_id: str) -> str:
    return f"{_task_source_id()}--{agent_id}"


def _agent_actor(agent_id: str) -> str:
    return resolve_actor(_agent_source_id(agent_id))


def _inject_plugin(worktree: Path, model: str) -> None:
    config_path = worktree / ".opencode" / "opencode.json"
    if not config_path.is_file():
        raise RuntimeError(f"CORAL OpenCode config does not exist: {config_path}")
    value = json.loads(config_path.read_text())
    plugins = list(value.get("plugin") or [])
    plugin = _plugin_path().as_uri()
    if plugin not in plugins:
        plugins.append(plugin)
    value["plugin"] = plugins
    plugin_inventory = []
    for item in plugins:
        identity = str(item[0] if isinstance(item, list) and item else item)
        supported = identity == plugin
        plugin_inventory.append(
            {
                "name": identity,
                "native_id": sha256_json(item),
                "registry_identity": identity,
                "implementation_identity": identity if supported else None,
                "dispatch_supported": supported,
            }
        )
        if supported:
            state().cover("plugin", identity)
    state().snapshot_registry(
        "opencode.config.plugins",
        plugin_inventory,
        phase="runtime-start",
    )
    providers = value.get("provider")
    if not isinstance(providers, dict) or not providers:
        raise RuntimeError("CORAL OpenCode config has no provider inventory")
    provider_name = model.split("/", 1)[0] if "/" in model else None
    if provider_name is None:
        if len(providers) != 1:
            raise RuntimeError("CORAL model without provider is ambiguous")
        provider_name = next(iter(providers))
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        raise RuntimeError(f"CORAL OpenCode provider does not exist: {provider_name}")
    options = provider.setdefault("options", {})
    if not isinstance(options, dict):
        raise RuntimeError(f"CORAL OpenCode provider options are invalid: {provider_name}")
    options["baseURL"] = f"{os.environ['NATIVE_REPLAY_PROXY_URL'].rstrip('/')}/v1"
    temporary = config_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, config_path)


def _runtime_start_factory(original):
    def wrapped(self, worktree_path, *args, **kwargs):
        worktree = Path(worktree_path)
        source_actor = (worktree / ".coral_agent_id").read_text().strip()
        actor = _agent_actor(source_actor)
        bound = inspect.signature(original).bind(self, worktree_path, *args, **kwargs)
        bound.apply_defaults()
        _inject_plugin(worktree, str(bound.arguments["model"]))
        bound.arguments["gateway_url"] = f"{os.environ['NATIVE_REPLAY_PROXY_URL'].rstrip('/')}/v1"
        spawn_env = {
            "NATIVE_REPLAY_ACTOR_ID": actor,
            "NATIVE_REPLAY_SESSION_ID": actor,
            "NATIVE_REPLAY_PROCESS_ROLE": "coral-opencode",
            "NATIVE_REPLAY_TARGET_ID": _target_for(actor),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_MODELS_FETCH": "true",
        }
        token = _SPAWN_ENV.set(spawn_env)
        try:
            return original(*bound.args, **bound.kwargs)
        finally:
            _SPAWN_ENV.reset(token)

    return wrapped


def _clean_env_factory(original):
    def wrapped():
        value = original()
        extra = _SPAWN_ENV.get()
        if extra:
            value.update(extra)
        return value

    return wrapped


def _grade_factory(original):
    implementation = method_identity(original)

    def wrapped(attempt, config_path, coral_dir, config):
        actor = _agent_actor(str(attempt.agent_id))
        tokens = set_context(
            actor_id=actor,
            process_role="coral-grader",
            session_id=f"{actor}/grader/{attempt.commit_hash}",
            llm_role="coral-grader",
        )
        try:
            kind = "entrypoint" if config.grader.entrypoint else "python"
            return run_grader(
                implementation=implementation,
                grader_kind=kind,
                trigger_id=str(attempt.commit_hash),
                invoke=lambda: original(attempt, config_path, coral_dir, config),
                result_encoder=jsonable,
                timeout_s=float(config.grader.timeout),
            )
        finally:
            reset_context(tokens)

    return wrapped


def _artifact_id(logical_path: str, version: int) -> str:
    digest = hashlib.sha256(logical_path.encode()).hexdigest()[:24]
    return f"artifact-{digest}-v{version}"


def _attempt_path(coral_dir, attempt) -> Path:
    return Path(coral_dir) / "public" / "attempts" / f"{attempt.commit_hash}.json"


def _attempt_version(attempt) -> int:
    return 1 if attempt.status == "pending" else 2


def _attempt_payload(attempt) -> bytes:
    # Match coral.hub.attempts.write_attempt exactly. Hash the object handled by
    # this operation rather than reopening a path another grader may replace.
    return json.dumps(attempt.to_dict(), indent=2).encode()


def _write_attempt_factory(original):
    def wrapped(coral_dir, attempt):
        context_tokens = None
        if current_context()["actor_id"] == "unknown":
            actor = _agent_actor(str(attempt.agent_id))
            context_tokens = set_context(
                actor_id=actor,
                process_role="coral-artifact-producer",
                session_id=f"{actor}/attempt/{attempt.commit_hash}",
                llm_role="coral-grader",
            )
        path = _attempt_path(coral_dir, attempt)
        logical = str(jsonable(path))
        version = _attempt_version(attempt)
        boundary = BoundaryClient()
        reservation = boundary.start(
            "artifact",
            logical_path=logical,
            operation="create" if version == 1 else "write",
            version=version,
            record_id_hint=_artifact_id(logical, version),
        )
        try:
            result = original(coral_dir, attempt)
            payload = _attempt_payload(attempt)
            boundary.complete(
                reservation["reservation_id"],
                status="ok",
                physical_path=str(jsonable(path)),
                process_role=str(current_context()["process_role"]),
                bytes_sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
                mode=path.stat().st_mode & 0o7777,
                triggered_by=[
                    value
                    for value in (
                        current_context().get("tool_call_id"),
                        current_context().get("grader_attempt_id"),
                    )
                    if value
                ],
                read_from=None,
                native_execution=True,
            )
            return result
        finally:
            if context_tokens is not None:
                reset_context(context_tokens)

    return wrapped


def _read_attempt_factory(original):
    def wrapped(coral_dir, commit_hash):
        result = original(coral_dir, commit_hash)
        if result is None:
            return None
        context_tokens = None
        if current_context()["actor_id"] == "unknown":
            actor = _agent_actor(str(result.agent_id))
            context_tokens = set_context(
                actor_id=actor,
                process_role="coral-artifact-consumer",
                session_id=f"{actor}/attempt/{result.commit_hash}",
                llm_role="coral-grader",
            )
        path = _attempt_path(coral_dir, result)
        logical = str(jsonable(path))
        version = _attempt_version(result)
        boundary = BoundaryClient()
        reservation = boundary.start(
            "artifact",
            logical_path=logical,
            operation="read",
            version=version,
        )
        try:
            payload = _attempt_payload(result)
            boundary.complete(
                reservation["reservation_id"],
                status="ok",
                physical_path=str(jsonable(path)),
                process_role=str(current_context()["process_role"]),
                bytes_sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
                mode=path.stat().st_mode & 0o7777,
                triggered_by=[],
                read_from=_artifact_id(logical, version),
                native_execution=True,
            )
            return result
        finally:
            if context_tokens is not None:
                reset_context(context_tokens)

    return wrapped


def _install_attempt_artifacts() -> None:
    import coral.hub.attempts as attempts

    patch_method(attempts, "write_attempt", _write_attempt_factory)
    patch_method(attempts, "read_attempt", _read_attempt_factory)
    state().mark("coral-shared-artifacts")


def _start_all_factory(original):
    def wrapped(self, *args, **kwargs):
        source = _task_source_id()
        actor = resolve_actor(source)
        tokens = set_context(
            actor_id=actor,
            process_role="coral-team",
            session_id=source,
            llm_role="coral-team",
            target_id=_target_for(actor),
        )
        self._minireplay_task_tokens = tokens
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            report_task_terminal(
                actor_id=resolve_actor(source),
                status="failure",
                result={"error_type": type(exc).__name__, "message": str(exc)},
            )
            reset_context(tokens)
            self._minireplay_task_tokens = None
            raise

    return wrapped


def _team_terminal(self) -> dict[str, object]:
    return {
        "task_id": _task_source_id(),
        "one_shot_terminal": bool(self._one_shot_terminal),
        "one_shot_failure": self._one_shot_failure,
        "max_total_turns": int(self.config.agents.max_total_turns),
        "turn_count": int(self._turn_count()),
        "restart_counts": {
            name: int(value) for name, value in sorted(self._restart_counts.items())
        },
        "eval_counts": {
            name: int(value) for name, value in sorted(self._agent_eval_counts.items())
        },
        "agents": [
            {
                "agent_id": str(handle.agent_id),
                "return_code": (handle.process.poll() if handle.process is not None else None),
                "restarts": int(self._restart_counts.get(handle.agent_id, 0)),
            }
            for handle in sorted(self.handles, key=lambda item: str(item.agent_id))
        ],
    }


def _monitor_factory(original):
    def wrapped(self, *args, **kwargs):
        tokens = getattr(self, "_minireplay_task_tokens", None)
        if tokens is None:
            raise RuntimeError("CORAL monitor entered without a gated native task")
        try:
            try:
                result = original(self, *args, **kwargs)
            except Exception as exc:
                report_task_terminal(
                    actor_id=resolve_actor(_task_source_id()),
                    status="failure",
                    result={"error_type": type(exc).__name__, "message": str(exc)},
                )
                raise
            terminal = _team_terminal(self)
            success = bool(self._one_shot_terminal and self._one_shot_failure is None)
            report_task_terminal(
                actor_id=resolve_actor(_task_source_id()),
                status="success" if success else "failure",
                result=terminal,
            )
            return result
        finally:
            reset_context(tokens)
            self._minireplay_task_tokens = None

    return wrapped


def install() -> None:
    _install_attempt_artifacts()
    role = os.environ.get("NATIVE_REPLAY_PROCESS_ROLE", "framework")
    if role == "coral-tool-child":
        return

    import coral.agent.builtin.opencode as opencode
    import coral.agent.manager as manager
    import coral.grader.daemon as daemon

    patch_method(opencode.OpenCodeRuntime, "start", _runtime_start_factory)
    patch_method(opencode, "_clean_env", _clean_env_factory)
    patch_method(manager.AgentManager, "start_all", _start_all_factory)
    patch_method(manager.AgentManager, "monitor_loop", _monitor_factory)
    patch_method(daemon, "_grade_one", _grade_factory)
    patch_method(daemon, "write_attempt", _write_attempt_factory)
    state().mark("coral-opencode-plugin")
    state().mark("coral-native-grader")
    state().mark("coral-opencode-routing")
    state().mark("coral-native-terminal")
    state().mark("coral-dispatch-ledger")
    state().cover("parser", "opencode.message.part.tool")
    state().cover("dispatcher", "opencode.tool.execute.before")
    state().cover("implementation", os.environ["NATIVE_REPLAY_OPENCODE_IDENTITY"])
