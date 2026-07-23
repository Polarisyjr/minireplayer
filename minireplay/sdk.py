from __future__ import annotations

import builtins
import contextvars
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from .errors import MismatchError
from .lane_record import (
    local_complete,
    local_composite_scope_complete,
    local_composite_scope_start,
    local_start,
)
from .observation import exact_result_contract
from .serialization import jsonable
from .util import atomic_write_json, monotonic_ns

T = TypeVar("T")

_actor = contextvars.ContextVar("minireplay_actor", default="unknown")
_process_role = contextvars.ContextVar("minireplay_process_role", default="unknown")
_parent_span = contextvars.ContextVar("minireplay_parent_span", default=None)
_session = contextvars.ContextVar("minireplay_session", default="unknown")
_llm_role = contextvars.ContextVar("minireplay_llm_role", default="agent")
_target = contextvars.ContextVar("minireplay_target", default="default")
_dispatch = contextvars.ContextVar("minireplay_dispatch", default=None)
_tool_call = contextvars.ContextVar("minireplay_tool_call", default=None)
_composite_lane = contextvars.ContextVar("minireplay_composite_lane", default=None)
_grader_call = contextvars.ContextVar("minireplay_grader_call", default=None)
_last_llm_attempt = contextvars.ContextVar("minireplay_last_llm_attempt", default=None)
_captured_children: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "minireplay_captured_children", default=None
)
_captured_llm_attempts: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "minireplay_captured_llm_attempts", default=None
)
_captured_tool_calls: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "minireplay_captured_tool_calls", default=None
)
_captured_dispatch_executions: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "minireplay_dispatch_executions", default=None
)


def _raise_framework_exception(completion: dict[str, Any]) -> None:
    recorded = completion.get("framework_exception")
    if recorded is None:
        return
    if not isinstance(recorded, dict):
        raise MismatchError("recorded framework exception is not an object")
    name = recorded.get("error_type")
    message = recorded.get("message")
    if not isinstance(name, str) or not isinstance(message, str):
        raise MismatchError("recorded framework exception is incomplete")
    exception_type = getattr(builtins, name, None)
    if not isinstance(exception_type, type) or not issubclass(exception_type, Exception):
        raise RuntimeError(f"{name}: {message}")
    raise exception_type(message)


_captured_artifacts: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "minireplay_captured_artifacts", default=None
)


def set_context(
    *,
    actor_id: str,
    process_role: str,
    parent_span_id: str | None = None,
    session_id: str | None = None,
    llm_role: str | None = None,
    target_id: str | None = None,
):
    return (
        _actor.set(actor_id),
        _process_role.set(process_role),
        _parent_span.set(parent_span_id),
        _session.set(session_id or actor_id),
        _llm_role.set(llm_role or process_role),
        _target.set(target_id or os.environ.get("NATIVE_REPLAY_TARGET_ID", "default")),
    )


def reset_context(tokens) -> None:
    actor, role, parent, session, llm_role, target = tokens
    _actor.reset(actor)
    _process_role.reset(role)
    _parent_span.reset(parent)
    _session.reset(session)
    _llm_role.reset(llm_role)
    _target.reset(target)


def current_context() -> dict[str, str | None]:
    actor = _actor.get()
    inherited = actor == "unknown"
    return {
        "actor_id": os.environ.get("NATIVE_REPLAY_ACTOR_ID", actor) if inherited else actor,
        "process_role": (
            os.environ.get("NATIVE_REPLAY_PROCESS_ROLE", _process_role.get())
            if inherited
            else _process_role.get()
        ),
        "parent_span_id": (
            os.environ.get("NATIVE_REPLAY_PARENT_SPAN_ID", _parent_span.get())
            if inherited
            else _parent_span.get()
        ),
        "session_id": (
            os.environ.get("NATIVE_REPLAY_SESSION_ID", _session.get())
            if inherited
            else _session.get()
        ),
        "llm_role": (
            os.environ.get("NATIVE_REPLAY_LLM_ROLE", _llm_role.get())
            if inherited
            else _llm_role.get()
        ),
        "target_id": (
            os.environ.get("NATIVE_REPLAY_TARGET_ID", _target.get()) if inherited else _target.get()
        ),
        "dispatch_id": _dispatch.get(),
        "tool_call_id": _tool_call.get(),
        "composite_lane": _composite_lane.get(),
        "grader_attempt_id": _grader_call.get(),
    }


def llm_identity_headers() -> dict[str, str]:
    headers = {
        "X-Native-Replay-Actor": _actor.get(),
        "X-Native-Replay-Session": _session.get(),
        "X-Native-Replay-Role": _llm_role.get(),
        "X-Native-Replay-Target": _target.get(),
    }
    parent = _parent_span.get()
    if parent is not None:
        headers["X-Native-Replay-Parent-Span"] = parent
    return headers


def remember_llm_attempt(attempt_id: str | None) -> None:
    if attempt_id:
        _last_llm_attempt.set(attempt_id)
        attempts = _captured_llm_attempts.get()
        if attempts is not None and attempt_id not in attempts:
            attempts.append(attempt_id)


def last_llm_attempt_id() -> str | None:
    return _last_llm_attempt.get()


def record_subprocess_launch(
    launcher: str,
    *,
    runtime_pid: int | None = None,
    command: Any = None,
    cwd: Any = None,
    shell: bool = False,
    executable: Any = None,
) -> None:
    records = _captured_children.get()
    if records is None:
        return
    container_role = os.environ.get("NATIVE_REPLAY_CONTAINER_ROLE")
    record: dict[str, Any] = {
        "kind": "container-process" if container_role else "native-subprocess",
        "owner_actor": _actor.get(),
        "launcher": launcher,
        "command": jsonable(command),
        "cwd": jsonable(cwd),
        "shell": shell,
        "executable": jsonable(executable),
    }
    if container_role:
        record["container_role"] = container_role
    elif runtime_pid is not None:
        record["runtime_pid"] = runtime_pid
    records.append(record)


@contextmanager
def capture_subprocess_launches():
    """Collect native subprocess receipts for an adapter-owned operation."""

    records: list[dict[str, Any]] = []
    token = _captured_children.set(records)
    try:
        yield records
    finally:
        _captured_children.reset(token)


def record_container_shell_command(
    *,
    launcher: str,
    container_role: str,
    command: str,
    exit_code: int,
    shell_executable: str,
) -> None:
    """Bind one persistent-container-shell command to the active native tool."""

    records = _captured_children.get()
    if records is None:
        return
    if not launcher or not container_role or not command or not shell_executable:
        raise ValueError("container shell command evidence is incomplete")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError("container shell command exit code is invalid")
    records.append(
        {
            "kind": "container-shell-command",
            "owner_actor": _actor.get(),
            "container_role": container_role,
            "launcher": launcher,
            "command": command,
            "exit_code": exit_code,
            "shell_executable": shell_executable,
        }
    )


def record_persistent_process_command(
    *,
    launcher: str,
    container_role: str,
    command: str,
    exit_code: int,
    shell_executable: str,
    runtime_pid: int,
) -> None:
    """Bind a command to a long-lived native shell process and its descendants."""

    records = _captured_children.get()
    if records is None:
        return
    if not launcher or not container_role or not command or not shell_executable:
        raise ValueError("persistent process command evidence is incomplete")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError("persistent process command exit code is invalid")
    if not isinstance(runtime_pid, int) or isinstance(runtime_pid, bool) or runtime_pid <= 0:
        raise ValueError("persistent process command runtime PID is invalid")
    records.append(
        {
            "kind": "container-persistent-process-command",
            "owner_actor": _actor.get(),
            "container_role": container_role,
            "launcher": launcher,
            "command": command,
            "exit_code": exit_code,
            "shell_executable": shell_executable,
            "runtime_pid": runtime_pid,
        }
    )


def _finish_children(token, explicit: Callable[[], list[Any]]) -> list[Any]:
    captured = list(_captured_children.get() or [])
    _captured_children.reset(token)
    return [*explicit(), *captured]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


@contextmanager
def llm_scope(role: str, *, target_id: str | None = None):
    role_token = _llm_role.set(role)
    target_token = _target.set(target_id) if target_id is not None else None
    previous = _last_llm_attempt.set(None)
    try:
        yield
    finally:
        completed_attempt = _last_llm_attempt.get()
        _last_llm_attempt.reset(previous)
        if completed_attempt is not None:
            # Preserve the innermost native model call for an enclosing
            # browser/grader operation that must link its result to that call.
            _last_llm_attempt.set(completed_attempt)
        if target_token is not None:
            _target.reset(target_token)
        _llm_role.reset(role_token)


@contextmanager
def composite_scope(*, name: str, model_call_id: str):
    """Enter a diagnostic-only composite orchestration scope.

    A composite such as Owl's ``browse_url`` is a container for model calls and
    replayable primitives, not a tool operation of its own.  It therefore creates
    no boundary reservation and no parent span.  Its stable model-call lane is
    inherited by standalone primitives so concurrent composites remain independent.
    """

    if not name or not model_call_id:
        raise ValueError("composite scope requires a name and model call ID")
    lane = f"model-call:{model_call_id}"
    lane_token = _composite_lane.set(lane)
    root_value = os.environ.get("NATIVE_REPLAY_LANE_EVENT_DIR")
    record_scope = os.environ.get("NATIVE_REPLAY_MODE") == "record" and bool(root_value)
    root = Path(str(root_value)) if record_scope else None
    scope_id = None
    if root is not None:
        scope_id = local_composite_scope_start(
            root=root,
            actor_id=_actor.get(),
            session_id=_session.get(),
            name=name,
            causal_lane=lane,
            started_at_ns=time.monotonic_ns(),
        )
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        if root is not None and scope_id is not None:
            local_composite_scope_complete(
                root=root,
                actor_id=_actor.get(),
                session_id=_session.get(),
                scope_id=scope_id,
                ended_at_ns=time.monotonic_ns(),
                status=status,
            )
        _composite_lane.reset(lane_token)


def report_task_terminal(
    *,
    result: Any,
    status: str = "success",
    actor_id: str | None = None,
    completed_at_ns: int | None = None,
    task: Any = None,
) -> None:
    """Publish the native top-level task result exactly once.

    This channel carries the object produced by the framework itself.  It is
    validation evidence, never a source of replacement values for replay.
    """

    actor = actor_id or _actor.get()
    if status not in {"success", "failure"}:
        raise ValueError(f"invalid native terminal status: {status}")
    if completed_at_ns is not None and (
        not isinstance(completed_at_ns, int) or completed_at_ns < 0
    ):
        raise ValueError("native terminal completion time must be a non-negative integer")
    terminal_dir = os.environ.get("NATIVE_REPLAY_TERMINAL_DIR")
    if not terminal_dir or actor == "unknown":
        raise RuntimeError("native replay terminal channel has no resolved actor")
    if task is None:
        try:
            task_map = json.loads(os.environ["NATIVE_REPLAY_TASK_MAP"])
            task = task_map.get(actor, {"dynamic_actor": actor})
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                f"native replay terminal task mapping is invalid for {actor}"
            ) from exc
    path = os.path.join(terminal_dir, f"{actor}.json")
    value = {
        "schema_version": "native-agent-replay.task-terminal/v1",
        "run_id": os.environ["NATIVE_REPLAY_RUN_ID"],
        "actor_id": actor,
        "task": jsonable(task),
        "status": status,
        "result": jsonable(result),
        "pid": os.getpid(),
        "completed_at_ns": monotonic_ns() if completed_at_ns is None else completed_at_ns,
    }
    target = os.fspath(path)
    if os.path.exists(target):
        target = os.path.join(
            terminal_dir,
            f"{actor}.{os.getpid()}.{value['completed_at_ns']}.json",
        )
    atomic_write_json(Path(target), value)


class BoundaryClient:
    def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
        self.endpoint = (endpoint or os.environ.get("NATIVE_REPLAY_BOUNDARY_URL", "")).rstrip("/")
        self.token = token or os.environ.get("NATIVE_REPLAY_BOUNDARY_TOKEN", "")
        if not self.endpoint or not self.token:
            raise RuntimeError("native replay boundary endpoint and token are required")
        self._reservations: dict[str, tuple[str, str, str, str]] = {}
        lane_dir = os.environ.get("NATIVE_REPLAY_LANE_EVENT_DIR")
        self._local_root = (
            Path(lane_dir)
            if os.environ.get("NATIVE_REPLAY_MODE") == "record" and lane_dir
            else None
        )

    def _post(self, path: str, value: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=json.dumps(value, separators=(",", ":")).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                message = json.load(exc).get("error", str(exc))
            except Exception:
                message = str(exc)
            raise MismatchError(message) from exc
        if not isinstance(result, dict):
            raise RuntimeError("boundary returned a non-object response")
        return result

    def start(self, kind: str, **fields: Any) -> dict[str, Any]:
        payload = {
            "kind": kind,
            "actor_id": _actor.get(),
            "process_role": _process_role.get(),
            "parent_span_id": _parent_span.get(),
            "started_at_ns": time.monotonic_ns(),
            **fields,
        }
        result = (
            local_start(root=self._local_root, payload=payload)
            if self._local_root is not None
            else self._post("/v1/boundary/start", payload)
        )
        reservation = result.get("reservation_id")
        record_id = result.get("record_id")
        if isinstance(reservation, str) and isinstance(record_id, str):
            actor_id = str(payload["actor_id"])
            session_id = str(payload.get("session_id") or actor_id)
            self._reservations[reservation] = (kind, record_id, actor_id, session_id)
            if kind == "tool":
                executions = _captured_dispatch_executions.get()
                if (
                    executions is not None
                    and payload.get("dispatch_id") is not None
                    and _tool_call.get() is None
                    and record_id not in executions
                ):
                    executions.append(record_id)
                records = _captured_tool_calls.get()
                if records is not None and record_id not in records:
                    records.append(record_id)
        return result

    def complete(self, reservation_id: str, **fields: Any) -> dict[str, Any]:
        payload = {
            "reservation_id": reservation_id,
            "ended_at_ns": time.monotonic_ns(),
            **fields,
        }
        reservation = self._reservations.pop(reservation_id, None)
        if self._local_root is not None:
            if reservation is None:
                raise RuntimeError("unknown local boundary reservation")
            payload["_lane_session_id"] = reservation[3]
            result = local_complete(
                root=self._local_root,
                reservation=reservation[:3],
                payload=payload,
            )
        else:
            result = self._post("/v1/boundary/complete", payload)
        if reservation is not None and reservation[0] == "artifact":
            records = _captured_artifacts.get()
            if records is not None and reservation[1] not in records:
                records.append(reservation[1])
        return result


@contextmanager
def parent_span(span_id: str) -> Iterator[None]:
    token = _parent_span.set(span_id)
    try:
        yield
    finally:
        _parent_span.reset(token)


def _dispatch_resolution(executions: list[str], *, failed: bool) -> tuple[str, str | None]:
    if len(executions) > 1:
        raise MismatchError("one framework dispatch entered multiple native tool executions")
    if executions:
        return "executed", executions[0]
    return ("failed-before-entry" if failed else "rejected"), None


def run_dispatch(
    *,
    name: str,
    arguments: dict[str, Any],
    parser_identity: str,
    dispatcher_identity: str,
    native_call_id: str,
    origin_kind: str,
    trigger_id: str,
    invoke: Callable[[], T],
    model_call_id: str | None = None,
    client: BoundaryClient | None = None,
) -> T:
    boundary = client or BoundaryClient()
    reservation = boundary.start(
        "dispatch",
        session_id=_session.get(),
        parser_identity=parser_identity,
        dispatcher_identity=dispatcher_identity,
        native_call_id=native_call_id,
        name=name,
        arguments=arguments,
        origin={
            "kind": origin_kind,
            "trigger_id": trigger_id,
            "model_call_id": model_call_id,
        },
    )
    dispatch_token = _dispatch.set(reservation["record_id"])
    execution_token = _captured_dispatch_executions.set([])
    try:
        with parent_span(reservation["span_id"]):
            result = invoke()
    except BaseException:
        executions = list(_captured_dispatch_executions.get() or [])
        _captured_dispatch_executions.reset(execution_token)
        _dispatch.reset(dispatch_token)
        status, execution = _dispatch_resolution(executions, failed=True)
        boundary.complete(
            reservation["reservation_id"],
            status=status,
            execution_call_id=execution,
        )
        raise
    executions = list(_captured_dispatch_executions.get() or [])
    _captured_dispatch_executions.reset(execution_token)
    _dispatch.reset(dispatch_token)
    status, execution = _dispatch_resolution(executions, failed=False)
    boundary.complete(
        reservation["reservation_id"],
        status=status,
        execution_call_id=execution,
    )
    return result


async def run_dispatch_async(
    *,
    name: str,
    arguments: dict[str, Any],
    parser_identity: str,
    dispatcher_identity: str,
    native_call_id: str,
    origin_kind: str,
    trigger_id: str,
    invoke: Callable[[], Awaitable[T]],
    model_call_id: str | None = None,
    client: BoundaryClient | None = None,
) -> T:
    boundary = client or BoundaryClient()
    reservation = boundary.start(
        "dispatch",
        session_id=_session.get(),
        parser_identity=parser_identity,
        dispatcher_identity=dispatcher_identity,
        native_call_id=native_call_id,
        name=name,
        arguments=arguments,
        origin={
            "kind": origin_kind,
            "trigger_id": trigger_id,
            "model_call_id": model_call_id,
        },
    )
    dispatch_token = _dispatch.set(reservation["record_id"])
    execution_token = _captured_dispatch_executions.set([])
    try:
        with parent_span(reservation["span_id"]):
            result = await invoke()
    except BaseException:
        executions = list(_captured_dispatch_executions.get() or [])
        _captured_dispatch_executions.reset(execution_token)
        _dispatch.reset(dispatch_token)
        status, execution = _dispatch_resolution(executions, failed=True)
        boundary.complete(
            reservation["reservation_id"],
            status=status,
            execution_call_id=execution,
        )
        raise
    executions = list(_captured_dispatch_executions.get() or [])
    _captured_dispatch_executions.reset(execution_token)
    _dispatch.reset(dispatch_token)
    status, execution = _dispatch_resolution(executions, failed=False)
    boundary.complete(
        reservation["reservation_id"],
        status=status,
        execution_call_id=execution,
    )
    return result


def run_tool(
    *,
    name: str,
    implementation: str,
    arguments: dict[str, Any],
    invoke: Callable[[], T],
    result_encoder: Callable[[T], Any] = lambda value: value,
    side_effects: Callable[[], dict[str, Any]] = dict,
    logical_frames: Callable[[], list[Any]] = list,
    child_processes: Callable[[], list[Any]] = list,
    result_status: Callable[[T], str] = lambda _value: "ok",
    result_contract: dict[str, Any] | None = None,
    result_replayer: Callable[[T, Any], T] | None = None,
    semantic_timeout_s: float | None = None,
    client: BoundaryClient | None = None,
) -> Any:
    boundary = client or BoundaryClient()
    composite_lane = _composite_lane.get()
    dispatch_id = None if composite_lane is not None else _dispatch.get()
    if composite_lane is None and (not isinstance(dispatch_id, str) or not dispatch_id):
        raise MismatchError("native tool execution escaped the framework dispatch ledger")
    direct_execution = composite_lane is None and _tool_call.get() is None
    reservation = boundary.start(
        "tool",
        dispatch_id=dispatch_id,
        causal_lane=composite_lane,
        name=name,
        implementation=implementation,
        arguments=arguments,
        result_contract=result_contract or exact_result_contract(),
        semantic_timeout_s=semantic_timeout_s,
    )
    executions = _captured_dispatch_executions.get()
    if (
        direct_execution
        and executions is not None
        and reservation["record_id"] not in executions
    ):
        executions.append(reservation["record_id"])
    cpu_started_ns = time.thread_time_ns()
    child_token = _captured_children.set([])
    try:
        tool_token = _tool_call.set(reservation["record_id"])
        try:
            with parent_span(reservation["span_id"]):
                native_result = invoke()
        finally:
            _tool_call.reset(tool_token)
    except BaseException as exc:
        children = _finish_children(child_token, child_processes)
        completion = boundary.complete(
            reservation["reservation_id"],
            status="error",
            result={"error_type": type(exc).__name__, "message": str(exc)},
            logical_frames=logical_frames(),
            side_effects=side_effects(),
            child_processes=children,
            native_execution=True,
            exception_raised=True,
            cpu_seconds=max(0.0, (time.thread_time_ns() - cpu_started_ns) / 1e9),
        )
        _raise_framework_exception(completion)
        raise
    children = _finish_children(child_token, child_processes)
    encoded = result_encoder(native_result)
    native_status = result_status(native_result)
    if native_status not in {"ok", "error", "timeout"}:
        raise ValueError(f"invalid native tool status: {native_status}")
    completion = boundary.complete(
        reservation["reservation_id"],
        status=native_status,
        result=encoded,
        logical_frames=logical_frames(),
        side_effects=side_effects(),
        child_processes=children,
        native_execution=True,
        cpu_seconds=max(0.0, (time.thread_time_ns() - cpu_started_ns) / 1e9),
    )
    _raise_framework_exception(completion)
    if completion.get("result_replay_required") is True:
        if result_replayer is None:
            raise MismatchError(f"native tool {name!r} requires an adapter-owned result replayer")
        native_result = result_replayer(native_result, completion.get("framework_result"))
    # Exact contracts return the native value untouched. Typed contracts let the
    # adapter restore only the validated source observation in that native type.
    return native_result


async def run_tool_async(
    *,
    name: str,
    implementation: str,
    arguments: dict[str, Any],
    invoke: Callable[[], Awaitable[T]],
    result_encoder: Callable[[T], Any] = lambda value: value,
    side_effects: Callable[[], dict[str, Any]] = dict,
    logical_frames: Callable[[], list[Any]] = list,
    child_processes: Callable[[], list[Any]] = list,
    result_status: Callable[[T], str] = lambda _value: "ok",
    result_contract: dict[str, Any] | None = None,
    result_replayer: Callable[[T, Any], T] | None = None,
    semantic_timeout_s: float | None = None,
    client: BoundaryClient | None = None,
) -> Any:
    boundary = client or BoundaryClient()
    composite_lane = _composite_lane.get()
    dispatch_id = None if composite_lane is not None else _dispatch.get()
    if composite_lane is None and (not isinstance(dispatch_id, str) or not dispatch_id):
        raise MismatchError("native tool execution escaped the framework dispatch ledger")
    direct_execution = composite_lane is None and _tool_call.get() is None
    reservation = boundary.start(
        "tool",
        dispatch_id=dispatch_id,
        causal_lane=composite_lane,
        name=name,
        implementation=implementation,
        arguments=arguments,
        result_contract=result_contract or exact_result_contract(),
        semantic_timeout_s=semantic_timeout_s,
    )
    executions = _captured_dispatch_executions.get()
    if (
        direct_execution
        and executions is not None
        and reservation["record_id"] not in executions
    ):
        executions.append(reservation["record_id"])
    cpu_started_ns = time.thread_time_ns()
    child_token = _captured_children.set([])
    try:
        tool_token = _tool_call.set(reservation["record_id"])
        try:
            with parent_span(reservation["span_id"]):
                native_result = await invoke()
        finally:
            _tool_call.reset(tool_token)
    except BaseException as exc:
        children = _finish_children(child_token, child_processes)
        completion = boundary.complete(
            reservation["reservation_id"],
            status="error",
            result={"error_type": type(exc).__name__, "message": str(exc)},
            logical_frames=logical_frames(),
            side_effects=side_effects(),
            child_processes=children,
            native_execution=True,
            exception_raised=True,
            cpu_seconds=max(0.0, (time.thread_time_ns() - cpu_started_ns) / 1e9),
        )
        _raise_framework_exception(completion)
        raise
    children = _finish_children(child_token, child_processes)
    native_status = result_status(native_result)
    if native_status not in {"ok", "error", "timeout"}:
        raise ValueError(f"invalid native tool status: {native_status}")
    completion = boundary.complete(
        reservation["reservation_id"],
        status=native_status,
        result=result_encoder(native_result),
        logical_frames=logical_frames(),
        side_effects=side_effects(),
        child_processes=children,
        native_execution=True,
        cpu_seconds=max(0.0, (time.thread_time_ns() - cpu_started_ns) / 1e9),
    )
    _raise_framework_exception(completion)
    if completion.get("result_replay_required") is True:
        if result_replayer is None:
            raise MismatchError(f"native tool {name!r} requires an adapter-owned result replayer")
        native_result = result_replayer(native_result, completion.get("framework_result"))
    return native_result


def record_artifact(
    *,
    logical_path: str,
    physical_path: str,
    operation: str,
    version: int,
    bytes_sha256: str,
    size: int,
    mode: int,
    triggered_by: list[str],
    read_from: str | None,
    event_id: str | None = None,
    client: BoundaryClient | None = None,
) -> str:
    boundary = client or BoundaryClient()
    reservation = boundary.start(
        "artifact",
        logical_path=logical_path,
        operation=operation,
        version=version,
        record_id_hint=event_id,
    )
    boundary.complete(
        reservation["reservation_id"],
        status="ok",
        physical_path=physical_path,
        process_role=_process_role.get(),
        bytes_sha256=bytes_sha256,
        size=size,
        mode=mode,
        triggered_by=triggered_by,
        read_from=read_from,
        native_execution=True,
    )
    return reservation["record_id"]


def run_grader(
    *,
    implementation: str,
    grader_kind: str,
    trigger_id: str,
    invoke: Callable[[], T],
    result_encoder: Callable[[T], Any] = lambda value: value,
    child_processes: Callable[[], list[Any]] = list,
    llm_attempt_ids: Callable[[], list[str]] = list,
    tool_call_ids: Callable[[], list[str]] = list,
    artifact_versions: Callable[[], list[str]] = list,
    timeout_s: float | None = None,
    client: BoundaryClient | None = None,
) -> T:
    boundary = client or BoundaryClient()
    reservation = boundary.start(
        "grader",
        implementation=implementation,
        grader_kind=grader_kind,
        trigger_id=trigger_id,
        semantic_timeout_s=timeout_s,
    )
    cpu_started_ns = time.thread_time_ns()
    child_token = _captured_children.set([])
    llm_token = _captured_llm_attempts.set([])
    tool_token = _captured_tool_calls.set([])
    artifact_token = _captured_artifacts.set([])

    def finish_evidence() -> tuple[list[Any], list[str], list[str], list[str]]:
        children = _finish_children(child_token, child_processes)
        captured_llm = list(_captured_llm_attempts.get() or [])
        captured_tools = list(_captured_tool_calls.get() or [])
        captured_artifacts = list(_captured_artifacts.get() or [])
        _captured_llm_attempts.reset(llm_token)
        _captured_tool_calls.reset(tool_token)
        _captured_artifacts.reset(artifact_token)
        return (
            children,
            _unique([*llm_attempt_ids(), *captured_llm]),
            _unique([*tool_call_ids(), *captured_tools]),
            _unique([*artifact_versions(), *captured_artifacts]),
        )

    try:
        grader_token = _grader_call.set(reservation["record_id"])
        try:
            with parent_span(reservation["span_id"]):
                result = invoke()
        finally:
            _grader_call.reset(grader_token)
    except BaseException as exc:
        children, llm_ids, tool_ids, artifact_ids = finish_evidence()
        boundary.complete(
            reservation["reservation_id"],
            status="error",
            result={"error_type": type(exc).__name__, "message": str(exc)},
            child_processes=children,
            llm_attempt_ids=llm_ids,
            tool_call_ids=tool_ids,
            artifact_versions=artifact_ids,
            native_execution=True,
            cpu_seconds=max(0.0, (time.thread_time_ns() - cpu_started_ns) / 1e9),
        )
        raise
    children, llm_ids, tool_ids, artifact_ids = finish_evidence()
    boundary.complete(
        reservation["reservation_id"],
        status="ok",
        result=result_encoder(result),
        child_processes=children,
        llm_attempt_ids=llm_ids,
        tool_call_ids=tool_ids,
        artifact_versions=artifact_ids,
        native_execution=True,
        cpu_seconds=max(0.0, (time.thread_time_ns() - cpu_started_ns) / 1e9),
    )
    return result
