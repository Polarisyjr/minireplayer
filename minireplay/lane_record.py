"""Lane-local recording for native operation boundaries.

Recording has no expected bundle to claim, so making a framework worker wait for a
central ``/start`` reservation only perturbs the workload.  Each process therefore
appends start/complete events directly to an actor-local log.  After the native
sweep stops, the supervisor materializes those events through ``BoundaryLedger``;
that keeps path binding, result projection and the on-disk record schemas in one
place while moving all of that work outside the measured hot path.

Replay deliberately does not use this path.  It must validate a claim before native
work begins, and continues to use the boundary service for that gate.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from pathlib import Path
from typing import Any

from .constants import LANE_RECORD_EVENT_SCHEMA
from .util import append_jsonl, iter_jsonl, require

_ID_PREFIX = {
    "dispatch": "dispatch",
    "tool": "tool",
    "grader": "grader",
    "artifact": "artifact",
}
_LANE_LOCKS: dict[Path, threading.Lock] = {}
_LANE_LOCKS_GUARD = threading.Lock()


def _lane_path(root: Path, actor_id: str, session_id: str) -> Path:
    identity = f"{actor_id}\0{session_id}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return root / f"lane-{digest}.jsonl"


def _path_lock(path: Path) -> threading.Lock:
    with _LANE_LOCKS_GUARD:
        return _LANE_LOCKS.setdefault(path, threading.Lock())


def append_lane_event(root: Path, event: dict[str, Any]) -> None:
    """Append one complete event atomically with respect to its causal lane."""

    actor_id = str(event["actor_id"])
    session_id = str(event.get("session_id") or actor_id)
    path = _lane_path(root, actor_id, session_id)
    # mini-swe uses worker threads, while other adapters may have more than one
    # callback in a session.  A per-file lock preserves event order without making
    # unrelated lanes wait for one another.
    with _path_lock(path):
        append_jsonl(path, event)


def local_start(
    *,
    root: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    kind = str(payload.get("kind"))
    require(kind in _ID_PREFIX, f"invalid local boundary kind: {kind!r}")
    actor_id = payload.get("actor_id")
    require(isinstance(actor_id, str) and bool(actor_id), "local boundary actor is required")
    session_id = str(payload.get("session_id") or actor_id)
    record_id = f"{_ID_PREFIX[kind]}-{secrets.token_hex(12)}"
    span_id = f"span-{secrets.token_hex(12)}"
    reservation_id = secrets.token_urlsafe(24)
    append_lane_event(
        root,
        {
            "schema_version": LANE_RECORD_EVENT_SCHEMA,
            "event": "start",
            "actor_id": actor_id,
            "session_id": session_id,
            "reservation_id": reservation_id,
            "record_id": record_id,
            "span_id": span_id,
            "at_ns": payload["started_at_ns"],
            "payload": payload,
        },
    )
    return {
        "reservation_id": reservation_id,
        "record_id": record_id,
        "span_id": span_id,
    }


def local_complete(
    *,
    root: Path,
    reservation: tuple[str, str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    kind, record_id, actor_id = reservation
    session_id = str(payload.pop("_lane_session_id", actor_id))
    append_lane_event(
        root,
        {
            "schema_version": LANE_RECORD_EVENT_SCHEMA,
            "event": "complete",
            "actor_id": actor_id,
            "session_id": session_id,
            "reservation_id": payload["reservation_id"],
            "record_id": record_id,
            "kind": kind,
            "at_ns": payload["ended_at_ns"],
            "payload": payload,
        },
    )
    response: dict[str, Any] = {"valid": True}
    if kind == "tool":
        response["result_replay_required"] = False
    return response


def _events(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if not root.is_dir():
        return values
    for path in sorted(root.glob("lane-*.jsonl")):
        values.extend(iter_jsonl(path))
    for value in values:
        require(
            value.get("schema_version") == LANE_RECORD_EVENT_SCHEMA,
            "lane record: unsupported event schema",
        )
        require(value.get("event") in {"start", "complete"}, "lane record: invalid event")
        require(isinstance(value.get("at_ns"), int), "lane record: invalid event time")
    # Completion timestamps are captured before their append.  Sorting by those
    # observed times reconstructs the causal order even when different lane files
    # are drained in filesystem order.
    return sorted(values, key=lambda value: (int(value["at_ns"]), value["event"] != "start"))


def materialize_lane_recording(
    *,
    event_dir: Path,
    stage_dir: Path,
    cutoff_at_ns: int,
    auth_token: str,
    adapter: str,
    run_root: Path,
    repo: Path,
) -> list[dict[str, Any]]:
    """Build closed ledgers and cutoff tails from lane-local start/end events."""

    # Framework interpreters only need the lightweight append path above and do
    # not necessarily install aiohttp.  Keep the server-side ledger import lazy so
    # importing ``minireplay.sdk`` never pulls service dependencies into them.
    from .boundary import BoundaryLedger

    ledger = BoundaryLedger(
        mode="record",
        stage_dir=stage_dir,
        auth_token=auth_token,
        adapter=adapter,
        run_root=run_root,
        repo=repo,
    )
    reservations: dict[str, str] = {}
    started: set[str] = set()
    completed: set[str] = set()

    for event in _events(event_dir):
        local_id = str(event["reservation_id"])
        if event["event"] == "start":
            require(local_id not in started, f"lane record: duplicate start {local_id}")
            started.add(local_id)
            if int(event["at_ns"]) > cutoff_at_ns:
                continue
            payload = dict(event["payload"])
            payload["record_id_hint"] = str(event["record_id"])
            payload["span_id_hint"] = str(event["span_id"])
            opened = ledger.start(payload)
            reservations[local_id] = str(opened["reservation_id"])
            continue

        require(local_id in started, f"lane record: completion before start {local_id}")
        require(local_id not in completed, f"lane record: duplicate completion {local_id}")
        completed.add(local_id)
        materialized_id = reservations.get(local_id)
        if materialized_id is None or int(event["at_ns"]) > cutoff_at_ns:
            continue
        payload = dict(event["payload"])
        payload["reservation_id"] = materialized_id
        ledger.complete(payload)

    return ledger.freeze_source_cutoff(cutoff_at_ns)
