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
import json
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

    def cutoff_source_actor_ids(self) -> set[str]:
        """Framework task identities still live at the source cutoff.

        A refill scheduler can run several source tasks in one logical lane. A
        terminal for the lane's first task therefore does not prove that its
        refill task also completed before the recording window closed.
        """

        recorded_sources = set(self.actor_map())
        terminal_sources: set[str] = set()
        for terminal in self.terminal.get("task_terminals", []):
            if not isinstance(terminal, dict):
                continue
            task = terminal.get("task")
            source = task.get("source_actor_id") if isinstance(task, dict) else None
            if isinstance(source, str) and source:
                terminal_sources.add(source)
                continue
            # Older/synthetic bundles may only carry actor_id. Treat it as a
            # source identity only when the manifest maps that exact value.
            actor_id = terminal.get("actor_id")
            if isinstance(actor_id, str) and actor_id in recorded_sources:
                terminal_sources.add(actor_id)
        return recorded_sources - terminal_sources

    def cutoff_actor_ids(self) -> set[str]:
        """Logical lanes with at least one source task open at cutoff."""

        mapping = self.actor_map()
        cutoff = {
            mapping[source]
            for source in self.cutoff_source_actor_ids()
            if source in mapping
        }
        terminal_actors = {
            str(terminal["actor_id"])
            for terminal in self.terminal.get("task_terminals", [])
            if isinstance(terminal, dict) and isinstance(terminal.get("actor_id"), str)
        }
        # Preserve support for dynamic actors whose manifests predate explicit
        # source_actor_ids.
        cutoff.update(set(self.actor_ids()) - terminal_actors)
        return cutoff


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


def _concurrency_window(
    *,
    adapter: str,
    workload: dict[str, Any],
    actors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if adapter != "coral":
        return None
    declared = [actor for actor in actors if isinstance(actor.get("lane"), dict)]
    if not declared:
        return None
    require(
        len(declared) == len(actors),
        "CORAL actor inventory mixes hierarchical and flat lanes",
    )
    size = int(workload.get("concurrency", 0))
    team_size = int(workload.get("coral_team_size", 0))
    require(size > 0, "CORAL concurrency window has no team slots")
    require(team_size == 4, "CORAL concurrency window must use four-agent teams")

    attempts: dict[tuple[int, int, int], dict[str, Any]] = {}
    for actor in actors:
        actor_id = str(actor["actor_id"])
        lane = actor["lane"]
        require(
            lane.get("concurrency_unit") == "coral-team",
            f"CORAL actor {actor_id!r} has the wrong concurrency unit",
        )
        slot = int(lane.get("team_slot", -1))
        generation = int(lane.get("slot_generation", -1))
        run_index = int(lane.get("run_index", -1))
        require(
            0 <= slot < size and generation >= 0 and run_index >= 0,
            f"CORAL actor {actor_id!r} has an invalid team slot",
        )
        require(
            int(lane.get("team_size", -1)) == team_size,
            f"CORAL actor {actor_id!r} changed team size",
        )
        key = (slot, generation, run_index)
        attempt = attempts.setdefault(
            key,
            {
                "team_slot": slot,
                "slot_generation": generation,
                "run_index": run_index,
                "source_task_id": lane.get("source_task_id"),
                "team_actor_id": None,
                "agent_actor_ids": {},
                "agent_parent_ids": set(),
            },
        )
        require(
            attempt["source_task_id"] == lane.get("source_task_id"),
            f"CORAL team slot {slot} generation {generation} mixes source tasks",
        )
        kind = lane.get("lane_kind")
        if kind == "team":
            require(
                attempt["team_actor_id"] is None,
                f"CORAL team slot {slot} generation {generation} has two parent lanes",
            )
            attempt["team_actor_id"] = actor_id
            continue
        require(kind == "agent", f"CORAL actor {actor_id!r} has an invalid lane kind")
        agent_index = int(lane.get("agent_index", -1))
        require(
            1 <= agent_index <= team_size,
            f"CORAL actor {actor_id!r} is outside its team",
        )
        require(
            lane.get("parent_actor_id") is not None,
            f"CORAL actor {actor_id!r} has no parent team lane",
        )
        attempt["agent_parent_ids"].add(str(lane["parent_actor_id"]))
        members = attempt["agent_actor_ids"]
        require(
            agent_index not in members,
            f"CORAL team slot {slot} generation {generation} duplicates agent-{agent_index}",
        )
        members[agent_index] = actor_id

    slots: list[dict[str, Any]] = []
    for slot in range(size):
        slot_attempts = [
            value
            for (observed_slot, _generation, _run_index), value in attempts.items()
            if observed_slot == slot
        ]
        slot_attempts.sort(key=lambda value: value["slot_generation"])
        require(bool(slot_attempts), f"CORAL concurrency window never occupied team slot {slot}")
        require(
            [value["slot_generation"] for value in slot_attempts]
            == list(range(len(slot_attempts))),
            f"CORAL team slot {slot} has a non-contiguous refill history",
        )
        rendered: list[dict[str, Any]] = []
        for attempt in slot_attempts:
            require(
                isinstance(attempt["team_actor_id"], str),
                f"CORAL team slot {slot} generation {attempt['slot_generation']} has no parent",
            )
            members = attempt["agent_actor_ids"]
            require(
                attempt["agent_parent_ids"] == {attempt["team_actor_id"]},
                f"CORAL team slot {slot} generation "
                f"{attempt['slot_generation']} has detached agent lanes",
            )
            require(
                sorted(members) == list(range(1, team_size + 1)),
                f"CORAL team slot {slot} generation "
                f"{attempt['slot_generation']} is not one complete four-agent group",
            )
            rendered.append(
                {
                    **{
                        key: value
                        for key, value in attempt.items()
                        if key not in {"agent_actor_ids", "agent_parent_ids"}
                    },
                    "agent_actor_ids": [members[index] for index in range(1, team_size + 1)],
                }
            )
        slots.append({"team_slot": slot, "attempts": rendered})
    return {
        "unit": "coral-team",
        "size": size,
        "team_size": team_size,
        "target_agent_lanes": size * team_size,
        "refill_unit": "whole-team",
        "slots": slots,
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
    coral_controls: list[dict[str, Any]] | None = None,
) -> Bundle:
    ensure_empty_directory(output)
    llm = list(iter_jsonl(stage_dir / "llm.jsonl"))
    dispatches = list(iter_jsonl(stage_dir / "dispatches.jsonl"))
    tools = list(iter_jsonl(stage_dir / "tools.jsonl"))
    graders = list(iter_jsonl(stage_dir / "graders.jsonl"))
    artifacts = list(iter_jsonl(stage_dir / "artifacts.jsonl"))
    spans = list(iter_jsonl(stage_dir / "spans.jsonl"))
    cutoff_tails = materialize_pre_dispatch_tails(
        adapter=adapter,
        llm=llm,
        dispatches=dispatches,
        cutoff_tails=cutoff_tails,
    )

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

    concurrency_window = _concurrency_window(
        adapter=adapter,
        workload=workload,
        actors=actors,
    )
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
    if concurrency_window is not None:
        manifest["concurrency_window"] = concurrency_window
        manifest["coral_controls"] = list(coral_controls or [])
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

    cutoff_tails = {
        "operations": records["cutoff-operation"],
        "llm_requests": records["cutoff-llm"],
    }
    _validate_cutoff_tails(cutoff_tails)
    _require_cutoff_tails_disjoint(cutoff_tails, ledgers, llm)
    validate_causal_graph(
        spans=spans,
        dispatches=ledgers["dispatch"],
        tools=ledgers["tool"],
        open_parent_span_ids={
            str(record["span_id"])
            for record in cutoff_tails["operations"]
            if record.get("replay_entry") == "enter-and-preserve-descendants"
            and isinstance(record.get("span_id"), str)
            and record["span_id"]
        },
    )
    validate_artifact_graph(ledgers["artifact"])

    terminal = read_json(root / "terminal.json")
    validate_terminal(terminal)

    _require_coral_dispatch_coverage(
        adapter=str(manifest["adapter"]),
        llm=llm,
        dispatches=ledgers["dispatch"],
        cutoff_tails=cutoff_tails,
    )

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
            causal_lane = entry.get("causal_lane")
            require(
                causal_lane is None or (isinstance(causal_lane, str) and bool(causal_lane)),
                f"cutoff tails: {section} entry has an invalid causal_lane",
            )
            if section == "operations":
                require(
                    entry.get("replay_entry", "block-before-entry")
                    in {"block-before-entry", "enter-and-preserve-descendants"},
                    "cutoff tails: operation entry has an invalid replay_entry",
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


def _response_tool_calls(value: Any) -> list[dict[str, Any]]:
    """Reassemble OpenAI tool calls from streamed or non-streamed responses."""

    by_slot: dict[tuple[int, int], dict[str, str]] = {}

    def consume(payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice_position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            choice_index = choice.get("index", choice_position)
            container = choice.get("delta")
            if not isinstance(container, dict):
                container = choice.get("message")
            if not isinstance(container, dict):
                continue
            calls = container.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call_position, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue
                call_index = call.get("index", call_position)
                key = (int(choice_index), int(call_index))
                current = by_slot.setdefault(
                    key, {"native_call_id": "", "name": "", "raw_arguments": ""}
                )
                call_id = call.get("id")
                if isinstance(call_id, str):
                    current["native_call_id"] += call_id
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                arguments = function.get("arguments")
                if isinstance(name, str):
                    current["name"] += name
                if isinstance(arguments, str):
                    current["raw_arguments"] += arguments

    chunks = value.get("chunks") if isinstance(value, dict) else None
    if isinstance(chunks, list):
        for chunk in chunks:
            if isinstance(chunk, dict):
                consume(chunk.get("payload"))
    else:
        consume(value)

    result: list[dict[str, Any]] = []
    for call in by_slot.values():
        if not call["native_call_id"]:
            continue
        try:
            arguments = json.loads(call["raw_arguments"] or "{}")
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"completed LLM tool call {call['native_call_id']!r} has invalid arguments"
            ) from exc
        require(
            isinstance(arguments, dict),
            f"completed LLM tool call {call['native_call_id']!r} arguments are not an object",
        )
        result.append({**call, "arguments": arguments})
    return result


def materialize_pre_dispatch_tails(
    *,
    adapter: str,
    llm: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
    cutoff_tails: dict[str, Any],
) -> dict[str, Any]:
    """Account for completed model calls whose native dispatch never began.

    This is a recording-boundary rule, not an adapter rule.  The LLM response is
    fixed work; if the source stopped before dispatch there is correctly no native
    operation record.  A synthetic zero-duration marker preserves that exact
    pre-entry boundary so replay can return the captured LLM response without
    executing a tool the source never entered.

    CORAL's composite ``task`` call is the one adapter-specific extension: when
    closed child work exists below the unentered parent call, replay enters the
    parent wrapper only far enough to preserve those descendants.
    """

    rendered = {
        "operations": [dict(record) for record in cutoff_tails.get("operations", [])],
        "llm_requests": [dict(record) for record in cutoff_tails.get("llm_requests", [])],
    }
    observed = {
        (str(record["actor_id"]), str(record["native_call_id"]))
        for record in [*dispatches, *rendered["operations"]]
        if isinstance(record.get("native_call_id"), str) and record["native_call_id"]
    }
    claimed_child_parents: set[str] = set()

    def child_parent_for(
        *,
        actor_id: str,
        parent_session: str,
        after_ns: int,
    ) -> str | None:
        candidates = sorted(
            (
                int(child["started_at_ns"]),
                str(child["parent_span_id"]),
            )
            for child in llm
            if child.get("actor_id") == actor_id
            and isinstance(child.get("session_id"), str)
            and str(child["session_id"]).startswith(f"{parent_session}/child-")
            and isinstance(child.get("parent_span_id"), str)
            and child["parent_span_id"]
            and str(child["parent_span_id"]) not in claimed_child_parents
            and int(child["started_at_ns"]) >= after_ns
        )
        if not candidates:
            return None
        parent_span_id = candidates[0][1]
        claimed_child_parents.add(parent_span_id)
        return parent_span_id

    for attempt in llm:
        actor_id = str(attempt["actor_id"])
        for call in _response_tool_calls(attempt["response"]):
            native_call_id = str(call["native_call_id"])
            identity = (actor_id, native_call_id)
            if identity in observed:
                continue
            digest = hashlib.sha256(f"{actor_id}\0{native_call_id}".encode()).hexdigest()[:24]
            arguments = call["arguments"]
            parent_session = str(attempt.get("session_id") or "")
            child_parent_span = (
                child_parent_for(
                    actor_id=actor_id,
                    parent_session=parent_session,
                    after_ns=int(attempt["ended_at_ns"]),
                )
                if adapter == "coral" and call["name"] == "task" and parent_session
                else None
            )
            replay_entry = (
                "enter-and-preserve-descendants"
                if child_parent_span is not None
                else "block-before-entry"
            )
            dispatch_id = f"predispatch-{digest}"
            dispatch_span_id = f"span-predispatch-{digest}"
            common = {
                "cutoff_truncated": True,
                "pre_dispatch": child_parent_span is None,
                "replay_entry": replay_entry,
                "actor_id": actor_id,
                "process_role": attempt.get("process_role", "agent"),
                "lane": f"model-call:{native_call_id}",
                "causal_lane": f"model-call:{native_call_id}",
                "session_id": attempt.get("session_id"),
                "origin": {
                    "kind": "llm_structured",
                    "trigger_id": attempt["attempt_id"],
                    "model_call_id": native_call_id,
                },
                "name": call["name"],
                "arguments": arguments,
                "native_arguments": arguments,
                "arguments_sha256": sha256_json(arguments),
                "source_started_at_ns": attempt["ended_at_ns"],
                "elapsed_ns": 0,
            }
            rendered["operations"].append(
                {
                    **common,
                    "kind": "dispatch",
                    "dispatch_id": dispatch_id,
                    "record_id": dispatch_id,
                    "span_id": dispatch_span_id,
                    "parser_identity": "opencode.message.part.tool",
                    "dispatcher_identity": "opencode.tool.execute.before",
                    "native_call_id": native_call_id,
                }
            )
            if child_parent_span is not None:
                tool_id = f"pretool-{digest}"
                rendered["operations"].append(
                    {
                        **common,
                        "pre_dispatch": False,
                        "kind": "tool",
                        "call_id": tool_id,
                        "record_id": tool_id,
                        # The child LLM spans already name the live task tool as
                        # their parent. Reuse that exact missing span identity to
                        # reconnect the closed descendant graph.
                        "span_id": child_parent_span,
                        "parent_span_id": dispatch_span_id,
                        "dispatch_id": dispatch_id,
                        "implementation": "opencode-native-replay-plugin",
                        "result_contract": {
                            "schema_version": "native-agent-replay.result-contract/v2",
                            "kind": "recorded-observation",
                            "fields": [
                                {"json_pointer": "/output", "optional": True},
                                {"json_pointer": "/error", "optional": True},
                                {
                                    "json_pointer": "/metadata/parentSessionId",
                                    "optional": True,
                                },
                                {
                                    "json_pointer": "/metadata/sessionId",
                                    "optional": True,
                                },
                            ],
                        },
                    }
                )
            observed.add(identity)
    return rendered


def _require_coral_dispatch_coverage(
    *,
    adapter: str,
    llm: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
    cutoff_tails: dict[str, Any],
) -> None:
    """CORAL must account for every provider-emitted tool call.

    OpenCode normally publishes ``tool.execute.before``, but built-in tools can
    bypass that hook. Without this gate a source bundle can silently omit the
    parent operation and causal closure can then discard completed subagent work.
    """

    if adapter != "coral":
        return

    emitted = {
        (str(record["actor_id"]), call_id)
        for record in llm
        for call_id in (
            str(call["native_call_id"]) for call in _response_tool_calls(record["response"])
        )
    }
    observed = {
        (str(record["actor_id"]), str(record["native_call_id"]))
        for record in dispatches
        if isinstance(record.get("native_call_id"), str) and record["native_call_id"]
    }
    observed.update(
        (str(record["actor_id"]), str(record["native_call_id"]))
        for record in cutoff_tails["operations"]
        if isinstance(record.get("native_call_id"), str) and record["native_call_id"]
    )
    missing = sorted(emitted - observed)
    if missing:
        sample = ", ".join(f"{actor}:{call_id}" for actor, call_id in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise ValidationError(
            f"CORAL LLM tool calls have no dispatch or cutoff evidence: {sample}{suffix}"
        )


def bundle_id_for(workload: dict[str, Any]) -> str:
    return sha256_json(workload)[:16]
