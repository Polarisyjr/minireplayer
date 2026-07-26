from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from minireplay.errors import InfrastructureError, MismatchError
from minireplay.sdk import current_context, report_task_terminal, reset_context, set_context
from minireplay.util import atomic_write_json, monotonic_ns, read_json

_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
_DYNAMIC = re.compile(r"^refill-[0-9]{6}(?:--agent-[0-9]+)?$")
_GATED: set[tuple[int, str]] = set()
_NATIVE_LANES: dict[tuple[int, str], str] = {}
_TASK_LANES: dict[tuple[int, str], str] = {}
_BINDING_SEQUENCE = 0
_LOCK = threading.Lock()
_IDENTITY_ENV = {
    "actor_id": "NATIVE_REPLAY_ACTOR_ID",
    "process_role": "NATIVE_REPLAY_PROCESS_ROLE",
    "session_id": "NATIVE_REPLAY_SESSION_ID",
    "llm_role": "NATIVE_REPLAY_LLM_ROLE",
    "target_id": "NATIVE_REPLAY_TARGET_ID",
}


def _export_context() -> dict[str, str | None]:
    context = current_context()
    previous: dict[str, str | None] = {}
    for field, name in _IDENTITY_ENV.items():
        previous[name] = os.environ.get(name)
        value = context[field]
        if isinstance(value, str) and value:
            os.environ[name] = value
    return previous


def _restore_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _mapping(name: str) -> dict[str, Any]:
    raw = os.environ.get(name, "{}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InfrastructureError(f"invalid {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise InfrastructureError(f"{name} must be a JSON object")
    return value


def derive_actor_id(source_id: str) -> str:
    """Name an actor from its framework-native identity, with no pre-declared inventory.

    Recording does not resolve a task list up front, so an actor names itself the
    first time it reaches the gate. A framework id that is already filesystem-safe
    is kept verbatim because it stays readable in evidence; anything else (CORAL
    passes a task path) collapses to a stable content-derived id. The function is
    pure, so record and replay derive the same name without coordinating.
    """

    if _SAFE.fullmatch(source_id) is not None:
        return source_id
    return "task-" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]


def resolve_actor(source_id: str) -> str:
    mapping = _mapping("NATIVE_REPLAY_ACTOR_MAP")
    actor = mapping.get(source_id)
    if actor is None:
        actor = derive_actor_id(source_id) if source_id else source_id
    if not isinstance(actor, str) or _SAFE.fullmatch(actor) is None:
        raise MismatchError(f"unmapped or invalid native actor: {source_id!r}")
    allowed = os.environ.get("NATIVE_REPLAY_ACTORS")
    if allowed:
        try:
            inventory = json.loads(allowed)
        except json.JSONDecodeError as exc:
            raise InfrastructureError(f"invalid NATIVE_REPLAY_ACTORS: {exc}") from exc
        if actor not in inventory and _DYNAMIC.fullmatch(actor) is None:
            raise MismatchError(f"native actor is outside the recording inventory: {actor}")
    return actor


def _arrival_offset(actor_id: str) -> float:
    value = _mapping("NATIVE_REPLAY_ARRIVAL_OFFSETS").get(actor_id, 0.0)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise InfrastructureError(f"invalid arrival offset for {actor_id}: {value!r}")
    return float(value)


def target_for_actor(actor_id: str) -> str:
    target = _mapping("NATIVE_REPLAY_TARGET_MAP").get(
        actor_id,
        os.environ.get("NATIVE_REPLAY_TARGET_ID", "default"),
    )
    if not isinstance(target, str) or not target:
        raise InfrastructureError(f"invalid LLM target for {actor_id}: {target!r}")
    return target


@contextlib.contextmanager
def _causal_lane_lock(
    actor_id: str,
    native_lane_key: str | None,
) -> Iterator[None]:
    """Keep refill tasks for one recorded lane sequential across pool workers.

    Replay restores a refill task's recorded actor even when ``ProcessPoolExecutor``
    assigns it to a different runtime worker. The earlier task for that actor may
    still be running there, so identity mapping alone is insufficient: both tasks
    would consume one LLM/tool lane concurrently. A run-local advisory lock makes
    the causal lane serial without requiring process affinity from Owl's scheduler.
    """

    root = os.environ.get("NATIVE_REPLAY_LANE_BINDING_DIR")
    if native_lane_key is None or not root:
        yield
        return
    lock_dir = Path(root) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{actor_id}.lock").open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def bind_task_lane(
    source_actor_id: str,
    native_lane_key: str | None = None,
    *,
    actor_metadata: dict[str, Any] | None = None,
) -> str:
    """Bind one native task/session to a causal lane.

    Adapters which own a refill scheduler pass its stable, process-local execution
    slot (worker/thread/process) as ``native_lane_key``. Recording names that slot
    after its first task and persists every task-to-lane binding. Replay ignores
    fresh scheduling order and resolves the task through the bundle's actor map.
    Adapters without refill omit the key and retain one task per lane.
    """

    global _BINDING_SEQUENCE

    if native_lane_key is not None and (
        not isinstance(native_lane_key, str) or not native_lane_key
    ):
        raise InfrastructureError("native lane key must be a non-empty string")
    if actor_metadata is not None and not isinstance(actor_metadata, dict):
        raise InfrastructureError("actor metadata must be an object")
    resolved = resolve_actor(source_actor_id)
    pid = os.getpid()
    with _LOCK:
        if native_lane_key is not None and os.environ.get("NATIVE_REPLAY_MODE") == "record":
            actor_id = _NATIVE_LANES.setdefault((pid, native_lane_key), resolved)
        else:
            actor_id = resolved
        task_key = (pid, source_actor_id)
        first_binding = task_key not in _TASK_LANES
        previous = _TASK_LANES.setdefault(task_key, actor_id)
        if previous != actor_id:
            raise MismatchError(
                f"native task {source_actor_id!r} changed lanes: {previous!r} -> {actor_id!r}"
            )
        binding_dir = os.environ.get("NATIVE_REPLAY_LANE_BINDING_DIR")
        if first_binding and (native_lane_key is not None or actor_metadata) and binding_dir:
            sequence = _BINDING_SEQUENCE
            _BINDING_SEQUENCE += 1
            value = {
                "schema_version": "native-agent-replay.lane-binding/v1",
                "actor_id": actor_id,
                "source_actor_id": source_actor_id,
                "native_lane_key": native_lane_key,
                "pid": pid,
                "bound_at_ns": monotonic_ns(),
            }
            if actor_metadata:
                value["actor_metadata"] = actor_metadata
            atomic_write_json(
                Path(binding_dir) / f"{pid}-{sequence:06d}.json",
                value,
            )
        return actor_id


def ready_and_wait(
    source_actor_id: str,
    *,
    native_lane_key: str | None = None,
    process_role: str,
    session_id: str | None = None,
    llm_role: str | None = None,
    target_id: str | None = None,
    actor_metadata: dict[str, Any] | None = None,
) -> tuple[Any, ...]:
    actor_id = bind_task_lane(
        source_actor_id,
        native_lane_key,
        actor_metadata=actor_metadata,
    )
    target_id = target_id or target_for_actor(actor_id)
    key = (os.getpid(), actor_id)
    with _LOCK:
        if key in _GATED:
            return set_context(
                actor_id=actor_id,
                process_role=process_role,
                session_id=session_id,
                llm_role=llm_role,
                target_id=target_id,
            )
        _GATED.add(key)

    ready_dir = Path(os.environ["NATIVE_REPLAY_READY_DIR"])
    gate = Path(os.environ["NATIVE_REPLAY_START_GATE"])
    run_id = os.environ["NATIVE_REPLAY_RUN_ID"]
    if gate.exists():
        payload = read_json(gate)
        if payload.get("run_id") != run_id:
            raise InfrastructureError("start gate belongs to another run")
        return set_context(
            actor_id=actor_id,
            process_role=process_role,
            session_id=session_id or actor_id,
            llm_role=llm_role,
            target_id=target_id,
        )
    ready_dir.mkdir(parents=True, exist_ok=True)
    ready = {
        "schema_version": "native-agent-replay.actor-ready/v1",
        "run_id": run_id,
        "actor_id": actor_id,
        "source_actor_id": source_actor_id,
        "process_role": process_role,
        "pid": os.getpid(),
        "ready_at_ns": monotonic_ns(),
    }
    if actor_metadata:
        ready["actor_metadata"] = actor_metadata
    atomic_write_json(ready_dir / f"{actor_id}.json", ready)
    timeout_s = float(os.environ.get("NATIVE_REPLAY_GATE_TIMEOUT_S", "1800"))
    deadline = time.monotonic() + timeout_s
    while not gate.is_file():
        if time.monotonic() >= deadline:
            raise InfrastructureError(f"timed out waiting for replay gate: {actor_id}")
        time.sleep(0.005)
    payload = read_json(gate)
    if payload.get("run_id") != run_id:
        raise InfrastructureError("start gate belongs to another run")
    opened_ns = payload.get("opened_at_ns")
    if not isinstance(opened_ns, int):
        raise InfrastructureError("start gate has no monotonic timestamp")
    release_ns = opened_ns + int(_arrival_offset(actor_id) * 1e9)
    while monotonic_ns() < release_ns:
        time.sleep(min(0.005, max(0.0, (release_ns - monotonic_ns()) / 1e9)))
    return set_context(
        actor_id=actor_id,
        process_role=process_role,
        session_id=session_id or actor_id,
        llm_role=llm_role,
        target_id=target_id,
    )


def gated_callable(function, source_actor_id: str, process_role: str, args, kwargs):
    tokens = ready_and_wait(source_actor_id, process_role=process_role)
    previous = _export_context()
    try:
        return function(*args, **kwargs)
    finally:
        _restore_environment(previous)
        reset_context(tokens)


def gated_terminal_callable(
    function,
    source_actor_id: str,
    process_role: str,
    args,
    kwargs,
    native_lane_key: str | None = None,
):
    tokens = ready_and_wait(
        source_actor_id,
        native_lane_key=native_lane_key,
        process_role=process_role,
    )
    try:
        actor_id = str(current_context()["actor_id"])
        with _causal_lane_lock(actor_id, native_lane_key):
            previous = _export_context()
            try:
                result = function(*args, **kwargs)
                successful = not (
                    isinstance(result, tuple) and len(result) >= 2 and result[1] != "ok"
                )
                report_task_terminal(
                    result=result,
                    status="success" if successful else "failure",
                )
                return result
            except Exception as exc:
                report_task_terminal(
                    status="failure",
                    result={"error_type": type(exc).__name__, "message": str(exc)},
                )
                raise
            finally:
                _restore_environment(previous)
    finally:
        reset_context(tokens)
