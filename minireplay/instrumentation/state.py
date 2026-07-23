from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minireplay.constants import INSTRUMENTATION_COVERAGE_KINDS
from minireplay.serialization import jsonable
from minireplay.util import atomic_write_json, monotonic_ns, sha256_json


@dataclass
class InstrumentationState:
    adapter: str
    process_role: str
    installed: set[str] = field(default_factory=set)
    failures: list[str] = field(default_factory=list)
    coverage_entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    registry_snapshots: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark(self, hook: str) -> None:
        with self._lock:
            self.installed.add(hook)
            self._write()

    def fail(self, message: str) -> None:
        with self._lock:
            self.failures.append(message)
            self._write()

    def cover(
        self,
        kind: str,
        identity: str,
        *,
        tool_name: str | None = None,
        native_id: str | None = None,
    ) -> None:
        if kind not in INSTRUMENTATION_COVERAGE_KINDS:
            raise ValueError(f"invalid native replay coverage kind: {kind}")
        if not identity or "*" in identity:
            raise ValueError("coverage identity must be exact and non-empty")
        entry: dict[str, Any] = {"kind": kind, "identity": identity}
        if tool_name is not None:
            if not tool_name or tool_name == "*":
                raise ValueError("coverage tool name must be exact and non-empty")
            entry["tool_name"] = tool_name
        if native_id is not None:
            if not native_id or native_id == "*":
                raise ValueError("coverage native ID must be exact and non-empty")
            entry["native_id"] = native_id
        with self._lock:
            self.coverage_entries[sha256_json(entry)] = entry
            self._write()

    def snapshot_registry(
        self,
        owner: str,
        entries: list[dict[str, Any]],
        *,
        phase: str,
    ) -> None:
        if not owner or "*" in owner or not phase:
            raise ValueError("registry snapshot requires exact owner and phase")
        normalized = jsonable(entries)
        if not isinstance(normalized, list):
            raise ValueError("registry snapshot entries must be a list")
        snapshot = {
            "owner": owner,
            "phase": phase,
            "entries": sorted(normalized, key=sha256_json),
        }
        with self._lock:
            self.registry_snapshots.append(snapshot)
            self._write()

    def _write(self) -> None:
        raw = os.environ.get("NATIVE_REPLAY_INSTRUMENTATION_STATUS_DIR")
        if not raw:
            return
        root = Path(raw)
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            root / f"{os.getpid()}.json",
            {
                "schema_version": "native-agent-replay.instrumentation-status/v1",
                "pid": os.getpid(),
                "adapter": self.adapter,
                "process_role": self.process_role,
                "actor_id": os.environ.get("NATIVE_REPLAY_ACTOR_ID"),
                "session_id": os.environ.get("NATIVE_REPLAY_SESSION_ID"),
                "target_id": os.environ.get("NATIVE_REPLAY_TARGET_ID"),
                "installed": sorted(self.installed),
                "failures": list(self.failures),
                "coverage_entries": sorted(self.coverage_entries.values(), key=sha256_json),
                "registry_snapshots": list(self.registry_snapshots),
                "updated_at_ns": monotonic_ns(),
            },
            mode=0o644,
        )


STATE: InstrumentationState | None = None


def initialize(adapter: str) -> InstrumentationState:
    global STATE
    if STATE is not None:
        if STATE.adapter != adapter:
            raise RuntimeError(
                f"native replay instrumentation already initialized for {STATE.adapter}"
            )
        return STATE
    STATE = InstrumentationState(
        adapter=adapter,
        process_role=os.environ.get("NATIVE_REPLAY_PROCESS_ROLE", "framework"),
    )
    STATE.mark("bootstrap")
    return STATE


def state() -> InstrumentationState:
    if STATE is None:
        raise RuntimeError("native replay instrumentation is not initialized")
    return STATE
