"""Cross-process control facts that are not replayable work."""

from __future__ import annotations

from pathlib import Path

from .util import atomic_write_json, sha256_json

SESSION_PREFIX_SCHEMA = "minireplay.session-prefix-consumed/v1"


def session_prefix_marker(run_root: Path, actor_id: str, session_id: str) -> Path:
    digest = sha256_json({"actor_id": actor_id, "session_id": session_id})
    return run_root / "session-prefix-consumed" / f"{digest}.json"


def mark_session_prefix_consumed(run_root: Path, actor_id: str, session_id: str) -> None:
    atomic_write_json(
        session_prefix_marker(run_root, actor_id, session_id),
        {
            "schema_version": SESSION_PREFIX_SCHEMA,
            "actor_id": actor_id,
            "session_id": session_id,
        },
        mode=0o644,
    )


def session_prefix_consumed(run_root: Path, actor_id: str, session_id: str) -> bool:
    return session_prefix_marker(run_root, actor_id, session_id).is_file()
