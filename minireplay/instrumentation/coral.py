from __future__ import annotations

import contextvars
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

from minireplay.coral_control import grader_recorded
from minireplay.replay_control import session_prefix_consumed
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

from .gate import bind_task_lane, ready_and_wait, resolve_actor
from .patching import method_identity, patch_method
from .state import state

_SPAWN_ENV: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "minireplay_coral_spawn_env", default=None
)
_INVOCATION_COUNTS: dict[str, int] = {}


def _plugin_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets/opencode_native_replay_plugin.js"


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


def _team_integer(name: str) -> int:
    value = os.environ.get(name)
    try:
        parsed = int(value) if value is not None else -1
    except ValueError as exc:
        raise RuntimeError(f"CORAL team has invalid {name}: {value!r}") from exc
    if parsed < 0:
        raise RuntimeError(f"CORAL team has no {name}")
    return parsed


def _team_lane_metadata() -> dict[str, object]:
    team_size = _team_integer("NATIVE_REPLAY_CORAL_TEAM_SIZE")
    if team_size != 4:
        raise RuntimeError("CORAL replay requires exactly four agents per team")
    return {
        "framework": "coral",
        "lane_kind": "team",
        "concurrency_unit": "coral-team",
        "team_slot": _team_integer("NATIVE_REPLAY_CORAL_TEAM_SLOT"),
        "slot_generation": _team_integer("NATIVE_REPLAY_CORAL_SLOT_GENERATION"),
        "run_index": _team_integer("NATIVE_REPLAY_CORAL_RUN_INDEX"),
        "team_size": team_size,
        "source_task_id": os.environ.get("NATIVE_REPLAY_CORAL_SOURCE_TASK_ID", _task_source_id()),
    }


def _agent_source_id(agent_id: str) -> str:
    return f"{_task_source_id()}--{agent_id}"


def _agent_actor(agent_id: str) -> str:
    if not agent_id.startswith("agent-"):
        raise RuntimeError(f"invalid CORAL agent identity: {agent_id!r}")
    try:
        agent_index = int(agent_id.removeprefix("agent-"))
    except ValueError as exc:
        raise RuntimeError(f"invalid CORAL agent identity: {agent_id!r}") from exc
    team = _team_lane_metadata()
    if not 1 <= agent_index <= int(team["team_size"]):
        raise RuntimeError(f"CORAL agent is outside its four-agent team: {agent_id!r}")
    return bind_task_lane(
        _agent_source_id(agent_id),
        actor_metadata={
            **team,
            "lane_kind": "agent",
            "agent_id": agent_id,
            "agent_index": agent_index,
            "parent_actor_id": resolve_actor(_task_source_id()),
        },
    )


def _agent_target(agent_id: str) -> str:
    # Resolve role routing from the framework-native ID before ``resolve_actor``
    # hashes the task/agent composite into a filesystem-safe ledger actor.
    return _target_for(_agent_source_id(agent_id))


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
    options["baseURL"] = os.environ["NATIVE_REPLAY_PROXY_URL"].rstrip("/")
    temporary = config_path.with_suffix(f".json.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, config_path)


def _runtime_start_factory(original):
    def wrapped(self, worktree_path, *args, **kwargs):
        worktree = Path(worktree_path)
        source_actor = (worktree / ".coral_agent_id").read_text().strip()
        actor = _agent_actor(source_actor)
        invocation_key = _agent_source_id(source_actor)
        invocation_index = _INVOCATION_COUNTS.get(invocation_key, 0)
        _INVOCATION_COUNTS[invocation_key] = invocation_index + 1
        invocation_id = f"{actor}/invocation-{invocation_index}"
        bound = inspect.signature(original).bind(self, worktree_path, *args, **kwargs)
        bound.apply_defaults()
        _inject_plugin(worktree, str(bound.arguments["model"]))
        bound.arguments["gateway_url"] = os.environ["NATIVE_REPLAY_PROXY_URL"].rstrip("/")
        spawn_env = {
            "NATIVE_REPLAY_ACTOR_ID": actor,
            "NATIVE_REPLAY_SESSION_ID": actor,
            "NATIVE_REPLAY_INVOCATION_ID": invocation_id,
            "NATIVE_REPLAY_INVOCATION_INDEX": str(invocation_index),
            "NATIVE_REPLAY_PROCESS_ROLE": "coral-opencode",
            "NATIVE_REPLAY_TARGET_ID": _agent_target(source_actor),
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
        logical_attempt_path = _attempt_logical_path(coral_dir, attempt, actor)
        input_artifact_id = _artifact_id(logical_attempt_path, _attempt_version(attempt))
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
                # The daemon reads the pending attempt before entering _grade_one.
                # Declare that input explicitly; the completed v2 write is captured
                # by write_attempt while the grader is active.
                artifact_versions=lambda: [input_artifact_id],
                timeout_s=float(config.grader.timeout),
            )
        finally:
            reset_context(tokens)

    return wrapped


def _artifact_id(logical_path: str, version: int) -> str:
    digest = hashlib.sha256(logical_path.encode()).hexdigest()[:24]
    return f"artifact-{digest}-v{version}"


def _attempt_logical_path(coral_dir, attempt, actor: str) -> str:
    # CORAL stores attempts below a run-name-specific results directory. That
    # physical directory changes between record and replay. A git commit hash is
    # also unstable because its timestamp changes. The ancestry generation and
    # tree hash stay fixed while still rejecting a different evaluated snapshot.
    root = Path(coral_dir).resolve().parent
    candidates = (root / "repo", root)
    repo = next((value for value in candidates if (value / ".git").exists()), None)
    if repo is None:
        raise RuntimeError(f"cannot locate CORAL attempt repo from {coral_dir}")

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"cannot resolve CORAL attempt identity for {attempt.commit_hash}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    tree = git("rev-parse", f"{attempt.commit_hash}^{{tree}}")
    generation = git("rev-list", "--count", str(attempt.commit_hash))
    return f"/coral-attempts/{actor}/g{generation}-{tree}.json"


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
        actor = str(current_context()["actor_id"])
        if actor == "unknown":
            actor = _agent_actor(str(attempt.agent_id))
            context_tokens = set_context(
                actor_id=actor,
                process_role="coral-artifact-producer",
                session_id=f"{actor}/attempt/{attempt.commit_hash}",
                llm_role="coral-grader",
            )
        path = _attempt_path(coral_dir, attempt)
        logical = _attempt_logical_path(coral_dir, attempt, actor)
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


def _install_attempt_artifacts() -> None:
    import coral.hub.attempts as attempts

    patch_method(attempts, "write_attempt", _write_attempt_factory)
    state().mark("coral-shared-artifacts")


def _start_all_factory(original):
    def wrapped(self, *args, **kwargs):
        source = _task_source_id()
        actor = resolve_actor(source)
        tokens = ready_and_wait(
            source,
            process_role="coral-team",
            session_id=source,
            llm_role="coral-team",
            target_id=_target_for(actor),
            actor_metadata=_team_lane_metadata(),
        )
        self._minireplay_task_tokens = tokens
        # Full replay can reach its first eval while start_all is still launching
        # sibling agents. Preserve the pre-launch snapshot so monitor_loop does not
        # mistake those early attempt files for work that predates this run.
        self._minireplay_monitor_seen_override = set()
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


def _get_seen_attempts_factory(original):
    def wrapped(self):
        override = getattr(self, "_minireplay_monitor_seen_override", None)
        if override is not None:
            del self._minireplay_monitor_seen_override
            return set(override)
        return original(self)

    return wrapped


def _current_invocation(agent_id: str) -> tuple[str, str] | None:
    actor = _agent_actor(agent_id)
    invocation_key = _agent_source_id(agent_id)
    invocation_index = _INVOCATION_COUNTS.get(invocation_key, 0) - 1
    if invocation_index < 0:
        return None
    return actor, f"{actor}/invocation-{invocation_index}/root-0"


def _recorded_control(actor: str, invocation_index: int) -> dict[str, object] | None:
    controls = json.loads(os.environ.get("NATIVE_REPLAY_CORAL_CONTROLS", "[]"))
    for control in controls:
        if (
            isinstance(control, dict)
            and control.get("actor_id") == actor
            and control.get("invocation_index") == invocation_index
        ):
            return control
    return None


def _schedule_interrupt_and_resume_factory(original):
    def wrapped(self, idx, prompt, prompt_source=None):
        live_replay_heartbeat = (
            os.environ.get("NATIVE_REPLAY_MODE") == "replay"
            and isinstance(prompt_source, str)
            and prompt_source.startswith("heartbeat:")
        )
        if live_replay_heartbeat:
            # Replay consumes recorded controls below. Live grader completion
            # order must never invent, suppress, or reorder an invocation.
            return False
        return original(self, idx, prompt, prompt_source=prompt_source)

    return wrapped


def _request_interrupt_factory(original):
    def wrapped(self, *, at_turn_boundary=False):
        recorded_restart = (
            at_turn_boundary
            and os.environ.get("NATIVE_REPLAY_MODE") == "replay"
            and bool(getattr(self, "_minireplay_recorded_restart", False))
        )
        if not recorded_restart:
            return original(self, at_turn_boundary=at_turn_boundary)
        return original(self, at_turn_boundary=False)

    return wrapped


def _advance_pending_resumes_factory(original):
    def wrapped(self):
        if os.environ.get("NATIVE_REPLAY_MODE") == "replay":
            run_root = Path(os.environ["NATIVE_REPLAY_RUN_ROOT"])
            pending = getattr(self, "_pending_resumes", {})
            for idx, handle in enumerate(self.handles):
                if handle.agent_id in pending:
                    continue
                current = _current_invocation(str(handle.agent_id))
                if current is None:
                    continue
                actor, root_session = current
                current_index = int(
                    root_session.split("/invocation-", 1)[1].split("/", 1)[0]
                )
                control = _recorded_control(actor, current_index + 1)
                if control is None:
                    continue
                trigger = control.get("trigger_grader_attempt_id")
                if not session_prefix_consumed(run_root, actor, root_session):
                    continue
                if not grader_recorded(
                    run_root,
                    str(trigger) if isinstance(trigger, str) else None,
                ):
                    continue

                handle._minireplay_recorded_restart = True
                try:
                    scheduled = self._schedule_interrupt_and_resume(
                        idx,
                        str(control["prompt"]),
                        prompt_source=f"minireplay-recorded:{control['source']}",
                    )
                finally:
                    if hasattr(handle, "_minireplay_recorded_restart"):
                        del handle._minireplay_recorded_restart
                if not scheduled:
                    raise RuntimeError(
                        f"recorded CORAL restart was rejected for {actor} "
                        f"invocation {current_index + 1}"
                    )
        return original(self)

    return wrapped


def _team_terminal(self) -> dict[str, object]:
    return {
        "task_id": _task_source_id(),
        "one_shot_terminal": bool(self._one_shot_terminal),
        "one_shot_failure": self._one_shot_failure,
        "termination_reason": getattr(self, "_termination_reason", None),
        # CORAL's manager clock is wall/epoch time, despite the historical
        # ``_at_ns`` field name. Keep the unit explicit at the recording boundary.
        "replay_cutoff_at_epoch_ns": getattr(self, "_replay_cutoff_at_ns", None),
        "run_dir": str(self.paths.run_dir) if getattr(self, "paths", None) is not None else None,
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
            # Replay stops the native driver as soon as every fixed slot is
            # delivered. That can interrupt the manager before its next polling
            # tick sets ``_one_shot_terminal``; absence of a native terminal is
            # then expected, not a task failure. Only publish a terminal the
            # manager itself reached, or a real one-shot failure it diagnosed.
            if self._one_shot_terminal or self._one_shot_failure is not None:
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
    import coral.agent.runtime as runtime
    import coral.grader.daemon as daemon

    patch_method(opencode.OpenCodeRuntime, "start", _runtime_start_factory)
    patch_method(opencode, "_clean_env", _clean_env_factory)
    patch_method(manager.AgentManager, "start_all", _start_all_factory)
    patch_method(manager.AgentManager, "_get_seen_attempts", _get_seen_attempts_factory)
    patch_method(
        manager.AgentManager,
        "_schedule_interrupt_and_resume",
        _schedule_interrupt_and_resume_factory,
    )
    patch_method(
        manager.AgentManager,
        "_advance_pending_resumes",
        _advance_pending_resumes_factory,
    )
    patch_method(manager.AgentManager, "monitor_loop", _monitor_factory)
    patch_method(runtime.AgentHandle, "request_interrupt", _request_interrupt_factory)
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
