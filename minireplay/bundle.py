"""Bundle build and load.

A bundle is the closed causal prefix of one source run plus the tails that were
still running when the sweep cut the window. It is built once, from a recording's
staged ledgers, and is read-only afterwards.

All expensive validation happens here, at load time, exactly once per run. The
per-slot check that runs inside a replay compares precomputed digests only (see
``boundary.BoundaryLedger._claim``), so the measured window carries no structural
comparison cost.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    BUNDLE_FILES,
    LANE_BUNDLE_EVENT_SCHEMA,
    LEDGER_FILES,
    LEDGER_ID_FIELD,
    MANIFEST_SCHEMA,
)
from .errors import ValidationError
from .schema import (
    validate_artifact,
    validate_artifact_graph,
    validate_causal_graph,
    validate_dispatch,
    validate_grader,
    validate_llm,
    validate_manifest,
    validate_span,
    validate_terminal,
    validate_tool,
)
from .util import (
    append_jsonl,
    atomic_write_json,
    ensure_empty_directory,
    iter_jsonl,
    read_json,
    require,
    sha256_json,
    unique_strings,
)

_VALIDATOR = {
    "dispatch": validate_dispatch,
    "tool": validate_tool,
    "grader": validate_grader,
    "artifact": validate_artifact,
}


@dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: dict[str, Any]
    llm: list[dict[str, Any]]
    dispatches: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    graders: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    spans: list[dict[str, Any]]
    cutoff_tails: dict[str, Any]
    terminal: dict[str, Any]

    @property
    def adapter(self) -> str:
        return str(self.manifest["adapter"])

    def records(self, kind: str) -> list[dict[str, Any]]:
        return {
            "dispatch": self.dispatches,
            "tool": self.tools,
            "grader": self.graders,
            "artifact": self.artifacts,
        }[kind]

    def actor_ids(self) -> list[str]:
        return [str(actor["actor_id"]) for actor in self.manifest["actors"]]

    def actor_map(self) -> dict[str, str]:
        """Recorded framework identity -> logical actor id."""

        mapping: dict[str, str] = {}
        for actor in self.manifest["actors"]:
            actor_id = str(actor["actor_id"])
            sources = [actor.get("source_actor_id"), *actor.get("source_actor_ids", [])]
            for source in sources:
                if not isinstance(source, str) or not source:
                    continue
                require(
                    source not in mapping or mapping[source] == actor_id,
                    f"source actor {source!r} maps to more than one lane",
                )
                mapping[source] = actor_id
        return mapping


def counts_of(
    *,
    llm: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    graders: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "llm": len(llm),
        "dispatch": len(dispatches),
        "tool": len(tools),
        "grader": len(graders),
        "artifact": len(artifacts),
    }


def build_bundle(
    *,
    stage_dir: Path,
    output: Path,
    bundle_id: str,
    adapter: str,
    workload: dict[str, Any],
    actors: list[dict[str, Any]],
    window: dict[str, int],
    terminal: dict[str, Any],
    cutoff_tails: dict[str, Any],
    llm_models: dict[str, Any] | None = None,
    identity_bindings: dict[str, Any] | None = None,
) -> Bundle:
    ensure_empty_directory(output)
    llm = list(iter_jsonl(stage_dir / "llm.jsonl"))
    dispatches = list(iter_jsonl(stage_dir / "dispatches.jsonl"))
    tools = list(iter_jsonl(stage_dir / "tools.jsonl"))
    graders = list(iter_jsonl(stage_dir / "graders.jsonl"))
    artifacts = list(iter_jsonl(stage_dir / "artifacts.jsonl"))
    spans = list(iter_jsonl(stage_dir / "spans.jsonl"))

    by_actor: dict[str, list[tuple[str, dict[str, Any]]]] = {
        str(actor["actor_id"]): [] for actor in actors
    }

    def add(kind: str, records: list[dict[str, Any]]) -> None:
        for record in records:
            actor_id = str(record.get("actor_id", ""))
            require(actor_id in by_actor, f"bundle records name undeclared actors: {[actor_id]}")
            by_actor[actor_id].append((kind, record))

    add("llm", llm)
    add("dispatch", dispatches)
    add("tool", tools)
    add("grader", graders)
    add("artifact", artifacts)
    add("span", spans)
    add("cutoff-operation", list(cutoff_tails.get("operations", [])))
    add("cutoff-llm", list(cutoff_tails.get("llm_requests", [])))

    lanes: list[dict[str, Any]] = []
    for index, actor in enumerate(actors):
        actor_id = str(actor["actor_id"])
        digest = hashlib.sha256(actor_id.encode()).hexdigest()[:12]
        relative = Path("lanes") / f"{index:06d}-{digest}" / "events.jsonl"
        events = sorted(by_actor[actor_id], key=_lane_issue_order)
        lane_path = output / relative
        lane_path.parent.mkdir(parents=True, exist_ok=True)
        lane_path.touch()
        for sequence, (kind, record) in enumerate(events):
            append_jsonl(
                lane_path,
                {
                    "schema_version": LANE_BUNDLE_EVENT_SCHEMA,
                    "actor_id": actor_id,
                    "sequence": sequence,
                    "kind": kind,
                    "record": record,
                },
            )
        lane_counts = {
            kind: sum(1 for observed, _record in events if observed == kind)
            for kind in (
                "llm",
                "dispatch",
                "tool",
                "grader",
                "artifact",
                "span",
                "cutoff-operation",
                "cutoff-llm",
            )
        }
        lanes.append(
            {
                "actor_id": actor_id,
                "path": relative.as_posix(),
                "counts": lane_counts,
            }
        )

    atomic_write_json(output / "terminal.json", terminal)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "bundle_id": bundle_id,
        "adapter": adapter,
        "workload": workload,
        "cutoff_policy": "evidence-only",
        "actors": actors,
        "lanes": lanes,
        "window": window,
        "counts": counts_of(
            llm=llm,
            dispatches=dispatches,
            tools=tools,
            graders=graders,
            artifacts=artifacts,
        ),
        # What each endpoint answered to `/v1/models`, so a tool-only replay can
        # answer the same discovery without a vLLM behind it.
        "llm_models": dict(llm_models or {}),
        # Per-actor framework-internal runtime identities (owl names its workers
        # with object addresses, which differ every run). Replay maps the recorded
        # names onto the live ones so a recorded routing decision still selects the
        # same logical worker.
        "identity_bindings": dict(identity_bindings or {}),
    }
    atomic_write_json(output / "manifest.json", manifest)
    return load_bundle(output)


def load_bundle(root: Path) -> Bundle:
    root = root.expanduser().resolve()
    require(root.is_dir(), f"bundle does not exist: {root}")
    for relative in BUNDLE_FILES:
        require((root / relative).is_file(), f"bundle is missing a required file: {relative}")

    manifest = read_json(root / "manifest.json")
    validate_manifest(manifest)

    records = _load_lane_records(root, manifest)
    llm = records["llm"]
    for record in llm:
        validate_llm(record)
    unique_strings((record["attempt_id"] for record in llm), "llm attempts")

    ledgers: dict[str, list[dict[str, Any]]] = {}
    for kind in LEDGER_FILES:
        entries = records[kind]
        for record in entries:
            _VALIDATOR[kind](record)
        unique_strings(
            (record[LEDGER_ID_FIELD[kind]] for record in entries),
            f"{kind} records",
        )
        ledgers[kind] = entries

    spans = records["span"]
    for span in spans:
        validate_span(span)
    unique_strings((span["span_id"] for span in spans), "spans")

    validate_causal_graph(spans=spans, dispatches=ledgers["dispatch"], tools=ledgers["tool"])
    validate_artifact_graph(ledgers["artifact"])

    terminal = read_json(root / "terminal.json")
    validate_terminal(terminal)

    cutoff_tails = {
        "operations": records["cutoff-operation"],
        "llm_requests": records["cutoff-llm"],
    }
    _validate_cutoff_tails(cutoff_tails)
    _require_cutoff_tails_disjoint(cutoff_tails, ledgers, llm)

    observed = counts_of(
        llm=llm,
        dispatches=ledgers["dispatch"],
        tools=ledgers["tool"],
        graders=ledgers["grader"],
        artifacts=ledgers["artifact"],
    )
    require(
        manifest["counts"] == observed,
        f"bundle counts disagree with its records: manifest={manifest['counts']} "
        f"observed={observed}",
    )

    _require_actor_coverage(manifest, ledgers, llm)

    return Bundle(
        root=root,
        manifest=manifest,
        llm=llm,
        dispatches=ledgers["dispatch"],
        tools=ledgers["tool"],
        graders=ledgers["grader"],
        artifacts=ledgers["artifact"],
        spans=spans,
        cutoff_tails=cutoff_tails,
        terminal=terminal,
    )


_LANE_KINDS = frozenset(
    {
        "llm",
        "dispatch",
        "tool",
        "grader",
        "artifact",
        "span",
        "cutoff-operation",
        "cutoff-llm",
    }
)


def _lane_issue_order(item: tuple[str, dict[str, Any]]) -> tuple[int, int]:
    kind, record = item
    started = record.get("started_at_ns")
    if not isinstance(started, int):
        started = record.get("source_started_at_ns")
    if not isinstance(started, int):
        started = record.get("completed_at_ns", 0)
    # A causal record precedes its duplicate diagnostic span at the same instant.
    return (int(started), 1 if kind == "span" else 0)


def _load_lane_records(root: Path, manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = {kind: [] for kind in _LANE_KINDS}
    for lane in manifest["lanes"]:
        actor_id = str(lane["actor_id"])
        relative = Path(str(lane["path"]))
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"bundle lane has an unsafe path: {relative}",
        )
        path = root / relative
        require(path.is_file(), f"bundle lane is missing: {relative}")
        observed_counts = {kind: 0 for kind in _LANE_KINDS}
        for expected_sequence, event in enumerate(iter_jsonl(path)):
            require(
                event.get("schema_version") == LANE_BUNDLE_EVENT_SCHEMA,
                f"bundle lane {actor_id}: unsupported event schema",
            )
            require(
                event.get("actor_id") == actor_id,
                f"bundle lane {actor_id}: event names another actor",
            )
            require(
                event.get("sequence") == expected_sequence,
                f"bundle lane {actor_id}: non-contiguous sequence",
            )
            kind = event.get("kind")
            require(kind in _LANE_KINDS, f"bundle lane {actor_id}: unknown kind {kind!r}")
            record = event.get("record")
            require(isinstance(record, dict), f"bundle lane {actor_id}: record is not an object")
            require(
                record.get("actor_id") == actor_id,
                f"bundle lane {actor_id}: record names another actor",
            )
            values[str(kind)].append(record)
            observed_counts[str(kind)] += 1
        require(
            lane["counts"] == observed_counts,
            f"bundle lane {actor_id}: manifest counts disagree with its events",
        )
    return values


def _validate_cutoff_tails(value: dict[str, Any]) -> None:
    require(isinstance(value, dict), "cutoff tails: expected an object")
    for section in ("operations", "llm_requests"):
        entries = value.get(section)
        require(isinstance(entries, list), f"cutoff tails: {section} must be a list")
        for entry in entries:
            require(isinstance(entry, dict), f"cutoff tails: {section} entry must be an object")
            require(
                entry.get("cutoff_truncated") is True,
                f"cutoff tails: {section} entry is not marked truncated",
            )
            elapsed = entry.get("elapsed_ns")
            require(
                isinstance(elapsed, int) and not isinstance(elapsed, bool) and elapsed >= 0,
                f"cutoff tails: {section} entry has an invalid elapsed_ns",
            )
            require(
                isinstance(entry.get("actor_id"), str) and bool(entry["actor_id"]),
                f"cutoff tails: {section} entry has no actor",
            )


def _require_cutoff_tails_disjoint(
    cutoff_tails: dict[str, Any],
    ledgers: dict[str, list[dict[str, Any]]],
    llm: list[dict[str, Any]],
) -> None:
    """A call cannot be both closed before cutoff and still running at cutoff."""

    closed = {
        kind: {str(record[LEDGER_ID_FIELD[kind]]) for record in records}
        for kind, records in ledgers.items()
    }
    closed["llm"] = {str(record["attempt_id"]) for record in llm}

    for entry in cutoff_tails["operations"]:
        kind = entry.get("kind")
        require(kind in ledgers, f"cutoff tails: unknown operation kind {kind!r}")
        record_id = entry.get("record_id")
        require(
            isinstance(record_id, str) and bool(record_id),
            "cutoff tails: operation entry has no record_id",
        )
        require(
            record_id not in closed[str(kind)],
            f"cutoff tails: {record_id} is both closed and cutoff-truncated",
        )

    for entry in cutoff_tails["llm_requests"]:
        attempt_id = entry.get("attempt_id")
        require(
            isinstance(attempt_id, str) and bool(attempt_id),
            "cutoff tails: LLM entry has no attempt_id",
        )
        require(
            attempt_id not in closed["llm"],
            f"cutoff tails: {attempt_id} is both closed and cutoff-truncated",
        )


def _require_actor_coverage(
    manifest: dict[str, Any],
    ledgers: dict[str, list[dict[str, Any]]],
    llm: list[dict[str, Any]],
) -> None:
    """Every record must belong to a declared actor.

    An unknown actor means the recording saw work the manifest cannot account for,
    which would make replay's per-actor slot queues incomplete before it starts.
    """

    declared = {str(actor["actor_id"]) for actor in manifest["actors"]}
    seen: set[str] = {str(record["actor_id"]) for record in llm}
    for records in ledgers.values():
        seen.update(str(record["actor_id"]) for record in records)
    unknown = sorted(seen - declared)
    if unknown:
        raise ValidationError(f"bundle records name undeclared actors: {unknown}")


def bundle_id_for(workload: dict[str, Any]) -> str:
    return sha256_json(workload)[:16]
