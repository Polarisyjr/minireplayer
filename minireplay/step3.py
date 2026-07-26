"""Export every source recording as a Step3-compatible timeline.

Step3 is an observation format, not a sweep profiler.  The sweep therefore stays
at ``-s none`` while the recorder projects its exact LLM/tool ledgers onto one row
per causal actor lane.  Composite orchestration envelopes are diagnostic context,
never tools: they are drawn as transparent outlines and are excluded from work
coverage. Calls that were still active at the source cutoff are included as
truncated spans.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .lane_record import composite_scope_rows
from .timeline import build_timeline
from .util import atomic_write, atomic_write_json, canonical_json

STEP3_SCHEMA = "minireplay.step3/v1"


def _visual_chain(actor: str, session_id: Any) -> str:
    """Keep child-session work on the owning CORAL agent lane.

    A CORAL subagent is an OpenCode session nested inside one agent invocation,
    not another independently scheduled CORAL lane.  The session remains on each
    raw row for audit, while the dashed ``task`` scope shows the nesting.
    """

    del session_id
    return actor


def _child_session(actor: str, session_id: Any) -> str | None:
    if not isinstance(session_id, str) or "/child-" not in session_id:
        return None
    if not session_id.startswith(f"{actor}/"):
        return None
    return session_id


def _chain_actor(chain: str) -> str:
    return chain.split("::", 1)[0]


def _chain_session(chain: str) -> str | None:
    if "::" not in chain:
        return None
    actor, relative = chain.split("::", 1)
    return f"{actor}/{relative}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_json(row) + b"\n" for row in rows))


def _port(target_id: Any) -> int | None:
    match = re.search(r"(\d+)$", str(target_id or ""))
    return int(match.group(1)) if match else None


def _epoch(
    at_ns: int,
    *,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
) -> float:
    bounded = min(max(at_ns, gate_at_ns), terminal_at_ns)
    return round((gate_at_epoch_ns + bounded - gate_at_ns) / 1e9, 9)


def _tail_end_ns(
    tail: dict[str, Any],
    *,
    started_at_ns: int,
    terminal_at_ns: int,
) -> int:
    """Return when an observed cutoff tail actually stopped being active.

    ``terminal_at_ns`` is the recorder window boundary. A framework can kill a
    lane earlier, and an interrupted HTTP stream can close earlier still. The
    elapsed duration captured at freeze time is therefore the primary endpoint;
    the other timestamps are upper bounds and provenance.
    """

    elapsed = tail.get("elapsed_ns")
    ended = (
        started_at_ns + int(elapsed)
        if isinstance(elapsed, int) and elapsed >= 0
        else terminal_at_ns
    )
    for key in ("interrupted_at_ns", "lane_terminated_at_ns"):
        upper = tail.get(key)
        if isinstance(upper, int):
            ended = min(ended, upper)
    return max(started_at_ns, min(ended, terminal_at_ns))


def _llm_rows(
    records: list[dict[str, Any]],
    tails: list[dict[str, Any]],
    *,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        response = record.get("response")
        request_id = response.get("id") if isinstance(response, dict) else None
        actor = str(record.get("actor_id") or "?")
        chain = _visual_chain(actor, record.get("session_id"))
        row = {
                "ts_start": _epoch(
                    int(record["started_at_ns"]),
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "ts_end": _epoch(
                    int(record["ended_at_ns"]),
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "role": str(record.get("role") or "agent"),
                "chain": chain,
                "port": _port(record.get("target_id")),
                "request_id": request_id or record.get("attempt_id"),
                "prompt_tokens": len(record.get("prompt_token_ids") or []),
                "completion_tokens": len(record.get("response_token_ids") or []),
                "e2e_s": round(
                    (int(record["ended_at_ns"]) - int(record["started_at_ns"])) / 1e9,
                    6,
                ),
                "source": "minireplay-llm-boundary",
            }
        if (session_id := _child_session(actor, record.get("session_id"))) is not None:
            row["actor_chain"] = actor
            row["session_id"] = session_id
            row["work_scope"] = "subagent"
        rows.append(row)
    for tail in tails:
        started = int(tail["started_at_ns"])
        ended = _tail_end_ns(
            tail,
            started_at_ns=started,
            terminal_at_ns=terminal_at_ns,
        )
        actor = str(tail.get("actor_id") or "?")
        chain = _visual_chain(actor, tail.get("session_id"))
        row = {
                "ts_start": _epoch(
                    started,
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "ts_end": _epoch(
                    ended,
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "role": str(tail.get("role") or "agent"),
                "chain": chain,
                "port": _port(tail.get("target_id")),
                "request_id": tail.get("attempt_id"),
                "prompt_tokens": None,
                "completion_tokens": None,
                "e2e_s": round((ended - started) / 1e9, 6),
                "timeline_kind": "truncated",
                "source": "minireplay-cutoff-tail",
            }
        if (session_id := _child_session(actor, tail.get("session_id"))) is not None:
            row["actor_chain"] = actor
            row["session_id"] = session_id
            row["work_scope"] = "subagent"
        rows.append(row)
    return sorted(rows, key=lambda row: (row["ts_start"], row["ts_end"]))


def _tool_rows(
    records: list[dict[str, Any]],
    tails: list[dict[str, Any]],
    *,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steps: defaultdict[str, int] = defaultdict(int)

    def add(record: dict[str, Any], *, truncated: bool) -> None:
        actor = str(record.get("actor_id") or "?")
        chain = _visual_chain(actor, record.get("session_id"))
        steps[chain] += 1
        started = int(record.get("started_at_ns", record.get("source_started_at_ns", gate_at_ns)))
        ended = (
            _tail_end_ns(
                record,
                started_at_ns=started,
                terminal_at_ns=terminal_at_ns,
            )
            if truncated
            else int(record["ended_at_ns"])
        )
        status = str(record.get("status") or "ok")
        row: dict[str, Any] = {
            "ts_start": _epoch(
                started,
                gate_at_ns=gate_at_ns,
                gate_at_epoch_ns=gate_at_epoch_ns,
                terminal_at_ns=terminal_at_ns,
            ),
            "ts_end": _epoch(
                ended,
                gate_at_ns=gate_at_ns,
                gate_at_epoch_ns=gate_at_epoch_ns,
                terminal_at_ns=terminal_at_ns,
            ),
            "chain": chain,
            "stage": "agent",
            "tool": str(record.get("name") or "tool"),
            "tools": None,
            "success": status in {"ok", "executed", "success"},
            "step": steps[chain],
            "n_calls": 1,
            "source": "minireplay-cutoff-tail" if truncated else "minireplay-tool-boundary",
        }
        if truncated:
            row["timeline_kind"] = "truncated"
        if (session_id := _child_session(actor, record.get("session_id"))) is not None:
            row["actor_chain"] = actor
            row["session_id"] = session_id
            row["work_scope"] = "subagent"
        rows.append(row)

    for record in records:
        add(record, truncated=False)
    for tail in tails:
        add(tail, truncated=True)
    return sorted(rows, key=lambda row: (row["ts_start"], row["ts_end"]))


def _lane_termination_rows(
    task_terminals: list[dict[str, Any]],
    actor_lanes: dict[str, dict[str, Any]],
    *,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
) -> list[dict[str, Any]]:
    """Project one CORAL team cutoff onto each of its four child agent lanes."""

    children: defaultdict[str, list[str]] = defaultdict(list)
    for actor, lane in actor_lanes.items():
        if (
            isinstance(lane, dict)
            and lane.get("concurrency_unit") == "coral-team"
            and lane.get("lane_kind") == "agent"
            and isinstance(lane.get("parent_actor_id"), str)
        ):
            children[str(lane["parent_actor_id"])].append(actor)

    rows: list[dict[str, Any]] = []
    terminal_epoch_ns = gate_at_epoch_ns + terminal_at_ns - gate_at_ns
    for terminal in task_terminals:
        if not isinstance(terminal, dict):
            continue
        team_actor = terminal.get("actor_id")
        result = terminal.get("result")
        if not isinstance(team_actor, str) or not isinstance(result, dict):
            continue
        cutoff_epoch_ns = result.get("replay_cutoff_at_epoch_ns")
        reason = result.get("termination_reason")
        if not isinstance(cutoff_epoch_ns, int) or not isinstance(reason, str) or not reason:
            continue
        bounded_epoch_ns = min(
            max(cutoff_epoch_ns, gate_at_epoch_ns),
            terminal_epoch_ns,
        )
        terminated_at_ns = gate_at_ns + bounded_epoch_ns - gate_at_epoch_ns
        seconds_since_gate = round(
            (bounded_epoch_ns - gate_at_epoch_ns) / 1e9,
            9,
        )
        terminated_at = round(bounded_epoch_ns / 1e9, 9)
        for actor in sorted(children.get(team_actor, [])):
            rows.append(
                {
                    "chain": actor,
                    "team_chain": team_actor,
                    "ts_terminated": terminated_at,
                    "terminated_at_ns": terminated_at_ns,
                    "terminated_at_epoch_ns": bounded_epoch_ns,
                    "seconds_since_gate": seconds_since_gate,
                    "reason": reason,
                    "timeline_kind": "lane-termination",
                    "source": "coral-team-manager",
                }
            )
    return sorted(rows, key=lambda row: (row["ts_terminated"], row["chain"]))


def _event_epoch_ns(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # OpenCode JSON stream timestamps are Unix milliseconds.
        return int(float(value) * 1e6)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1e9)
    return None


def _coral_restart_rows(
    task_terminals: list[dict[str, Any]],
    actor_lanes: dict[str, dict[str, Any]],
    *,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
) -> list[dict[str, Any]]:
    """Harvest native CORAL invocation boundaries without treating them as work."""

    agent_actors: dict[tuple[str, str], str] = {}
    for actor, lane in actor_lanes.items():
        if (
            isinstance(lane, dict)
            and lane.get("concurrency_unit") == "coral-team"
            and lane.get("lane_kind") == "agent"
            and isinstance(lane.get("parent_actor_id"), str)
            and isinstance(lane.get("agent_id"), str)
        ):
            agent_actors[(str(lane["parent_actor_id"]), str(lane["agent_id"]))] = actor

    terminal_epoch_ns = gate_at_epoch_ns + terminal_at_ns - gate_at_ns
    rows: list[dict[str, Any]] = []
    for terminal in task_terminals:
        team_actor = terminal.get("actor_id")
        result = terminal.get("result")
        if not isinstance(team_actor, str) or not isinstance(result, dict):
            continue
        run_dir = result.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            continue
        log_dir = Path(run_dir) / ".coral" / "public" / "logs"
        if not log_dir.is_dir():
            continue

        grouped: defaultdict[str, list[tuple[int, Path]]] = defaultdict(list)
        for path in sorted(log_dir.glob("agent-*.log")):
            match = re.fullmatch(r"(agent-\d+)\.(\d+)\.log", path.name)
            if match:
                grouped[match.group(1)].append((int(match.group(2)), path))

        for agent_id, entries in grouped.items():
            actor = agent_actors.get((team_actor, agent_id))
            if actor is None:
                continue
            invocations: list[dict[str, Any]] = []
            for sequence, path in sorted(entries):
                parsed: list[dict[str, Any]] = []
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            value = json.loads(line)
                            if isinstance(value, dict):
                                parsed.append(value)
                except (OSError, ValueError):
                    continue
                prompt = next(
                    (value for value in parsed if value.get("type") == "coral"),
                    {},
                )
                numeric_times = [
                    timestamp
                    for value in parsed
                    if isinstance(value.get("timestamp"), (int, float))
                    and (timestamp := _event_epoch_ns(value["timestamp"])) is not None
                ]
                launch = _event_epoch_ns(prompt.get("timestamp"))
                first = min(numeric_times) if numeric_times else launch
                last = max(numeric_times) if numeric_times else launch
                if first is None or last is None:
                    continue
                invocations.append(
                    {
                        "sequence": sequence,
                        "first": first,
                        "last": last,
                        "source": str(prompt.get("source") or "restart"),
                    }
                )

            for previous, resumed in zip(invocations, invocations[1:], strict=False):
                started_epoch_ns = max(int(previous["last"]), gate_at_epoch_ns)
                ended_epoch_ns = min(int(resumed["first"]), terminal_epoch_ns)
                if ended_epoch_ns <= started_epoch_ns:
                    continue
                heartbeat = str(resumed["source"]).startswith("heartbeat:")
                rows.append(
                    {
                        "chain": actor,
                        "team_chain": team_actor,
                        "agent_id": agent_id,
                        "invocation_index": int(resumed["sequence"]),
                        "ts_start": round(started_epoch_ns / 1e9, 9),
                        "ts_end": round(ended_epoch_ns / 1e9, 9),
                        "start_seconds_since_gate": round(
                            (started_epoch_ns - gate_at_epoch_ns) / 1e9,
                            9,
                        ),
                        "end_seconds_since_gate": round(
                            (ended_epoch_ns - gate_at_epoch_ns) / 1e9,
                            9,
                        ),
                        "source_prompt": resumed["source"],
                        "timeline_kind": "heartbeat" if heartbeat else "restart",
                        "source": "coral-native-invocation-log",
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["start_seconds_since_gate"],
            row["chain"],
            row["invocation_index"],
        ),
    )


def _composite_rows(
    records: list[dict[str, Any]],
    *,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        actor = str(record["actor_id"])
        chain = _visual_chain(actor, record.get("session_id"))
        row = {
                "scope_id": record["scope_id"],
                "chain": chain,
                "name": record["name"],
                "causal_lane": record["causal_lane"],
                "ts_start": _epoch(
                    int(record["started_at_ns"]),
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "ts_end": _epoch(
                    int(record["ended_at_ns"]),
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "timeline_kind": (
                    "truncated" if record.get("cutoff_truncated") is True else "scope"
                ),
                "source": "minireplay-composite-scope",
            }
        if (session_id := _child_session(actor, record.get("session_id"))) is not None:
            row["actor_chain"] = actor
            row["session_id"] = session_id
            row["work_scope"] = "subagent"
        rows.append(row)
    return sorted(rows, key=lambda row: (row["ts_start"], row["ts_end"]))


def _coral_task_composites(
    records: list[dict[str, Any]],
    tails: list[dict[str, Any]],
    *,
    terminal_at_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Render built-in task as a parent scope, not duplicated tool work."""

    scopes: list[dict[str, Any]] = []
    tool_records: list[dict[str, Any]] = []
    for record in records:
        if record.get("name") != "task":
            tool_records.append(record)
            continue
        scopes.append(
            {
                "scope_id": f"coral-task-{record['call_id']}",
                "actor_id": record["actor_id"],
                "session_id": record.get("session_id"),
                "name": "task",
                "causal_lane": record.get("causal_lane") or "",
                "started_at_ns": record["started_at_ns"],
                "ended_at_ns": record["ended_at_ns"],
                "cutoff_truncated": False,
            }
        )

    tool_tails: list[dict[str, Any]] = []
    for tail in tails:
        if tail.get("name") != "task":
            tool_tails.append(tail)
            continue
        started = int(tail.get("started_at_ns", tail.get("source_started_at_ns", 0)))
        scopes.append(
            {
                "scope_id": f"coral-task-{tail['record_id']}",
                "actor_id": tail["actor_id"],
                "session_id": tail.get("session_id"),
                "name": "task",
                "causal_lane": tail.get("causal_lane") or "",
                "started_at_ns": started,
                "ended_at_ns": _tail_end_ns(
                    tail,
                    started_at_ns=started,
                    terminal_at_ns=terminal_at_ns,
                ),
                "cutoff_truncated": True,
            }
        )
    return tool_records, tool_tails, scopes


def _tool_sessions(
    records: list[dict[str, Any]],
    tails: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
    dispatch_tails: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inherit the OpenCode session carried by each tool's dispatch."""

    sessions: dict[str, str] = {}
    for record in [*dispatches, *dispatch_tails]:
        dispatch_id = record.get("dispatch_id", record.get("record_id"))
        session_id = record.get("session_id")
        if isinstance(dispatch_id, str) and isinstance(session_id, str):
            sessions[dispatch_id] = session_id

    def bind(record: dict[str, Any]) -> dict[str, Any]:
        rendered = dict(record)
        if not isinstance(rendered.get("session_id"), str):
            session_id = sessions.get(str(rendered.get("dispatch_id") or ""))
            if session_id is not None:
                rendered["session_id"] = session_id
        return rendered

    return [bind(record) for record in records], [bind(record) for record in tails]


def _legacy_owl_composites(
    records: list[dict[str, Any]],
    tails: list[dict[str, Any]],
    *,
    terminal_at_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Promote old Owl ``browse_url`` tool rows into diagnostic scopes.

    Recordings made before transparent composite scopes cannot be rewritten into
    new replay bundles, but their visualization can stop misclassifying the outer
    orchestration as work. The nested records that survived the old cutoff closure
    remain untouched.
    """

    composite_records: list[dict[str, Any]] = []
    tool_records: list[dict[str, Any]] = []
    for record in records:
        if record.get("name") != "browse_url":
            tool_records.append(record)
            continue
        composite_records.append(
            {
                "scope_id": f"legacy-{record['call_id']}",
                "actor_id": record["actor_id"],
                "name": "browse_url",
                "causal_lane": record.get("causal_lane") or "",
                "started_at_ns": record["started_at_ns"],
                "ended_at_ns": record["ended_at_ns"],
                "cutoff_truncated": False,
            }
        )

    tool_tails: list[dict[str, Any]] = []
    for tail in tails:
        if tail.get("name") != "browse_url":
            tool_tails.append(tail)
            continue
        composite_records.append(
            {
                "scope_id": f"legacy-{tail['record_id']}",
                "actor_id": tail["actor_id"],
                "name": "browse_url",
                "causal_lane": tail.get("causal_lane") or "",
                "started_at_ns": tail["source_started_at_ns"],
                "ended_at_ns": terminal_at_ns,
                "cutoff_truncated": True,
            }
        )
    return tool_records, tool_tails, composite_records


def _write_text(
    path: Path,
    *,
    llm_rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    composite_rows: list[dict[str, Any]],
    restart_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
    gate_epoch_s: float,
    window_s: float,
    timeline: dict[str, Any],
    view_kind: str,
    actor_lanes: dict[str, dict[str, Any]],
) -> None:
    activities: list[tuple[float, float, str]] = []
    for row in llm_rows:
        suffix = " [truncated]" if row.get("timeline_kind") == "truncated" else ""
        activities.append(
            (
                float(row["ts_start"]) - gate_epoch_s,
                float(row["ts_end"]) - gate_epoch_s,
                f"LLM:{row['role']} [{_lane_label(str(row['chain']), actor_lanes)}]{suffix}",
            )
        )
    for row in tool_rows:
        suffix = " [truncated]" if row.get("timeline_kind") == "truncated" else ""
        activities.append(
            (
                float(row["ts_start"]) - gate_epoch_s,
                float(row["ts_end"]) - gate_epoch_s,
                f"tool:{row['tool']} [{_lane_label(str(row['chain']), actor_lanes)}]{suffix}",
            )
        )
    for row in composite_rows:
        suffix = " [open at cutoff]" if row.get("timeline_kind") == "truncated" else ""
        activities.append(
            (
                float(row["ts_start"]) - gate_epoch_s,
                float(row["ts_end"]) - gate_epoch_s,
                f"scope:{row['name']} [{_lane_label(str(row['chain']), actor_lanes)}]{suffix}",
            )
        )
    for row in restart_rows:
        start = float(row["start_seconds_since_gate"])
        end = float(row["end_seconds_since_gate"])
        activities.append(
            (
                start,
                end,
                (
                    f"CONTROL:{row['timeline_kind']} -> "
                    f"invocation-{row['invocation_index']} "
                    f"[{_lane_label(str(row['chain']), actor_lanes)}]"
                ),
            )
        )
    for row in termination_rows:
        at = float(row["seconds_since_gate"])
        activities.append(
            (
                at,
                at,
                (
                    f"TEAM TERMINATED:{row['reason']} "
                    f"[{_lane_label(str(row['chain']), actor_lanes)}]; "
                    "no lane work after this marker"
                ),
            )
        )
    activities.sort()
    gate_label = "replay" if view_kind == "replay" else "source"
    lines = [
        (
            f"# minireplay Step3 {view_kind} timeline "
            f"(t0={gate_epoch_s:.6f} epoch; seconds since {gate_label} gate)"
        ),
        f"# {view_kind} window: 0.0 .. {window_s:.3f} ({window_s:.3f}s)",
        f"# {'start':>9} {'end':>9} {'dur':>9}  lane",
        "# " + "-" * 76,
    ]
    for start, end, label in activities:
        lines.append(f"  {start:9.3f} {end:9.3f} {end - start:9.3f}  {label}")
    lines.extend(["", "== blanks >= 2s (no LLM/tool activity) =="])
    if timeline["gaps"]:
        for gap in timeline["gaps"]:
            lines.append(
                f"  {gap['duration_s']:7.3f}s  {gap['start_s']:9.3f} .. "
                f"{gap['end_s']:9.3f}  {gap['attribution']}"
            )
    else:
        lines.append("  (none)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lane_label(actor: str, actor_lanes: dict[str, dict[str, Any]]) -> str:
    chain = actor
    actor = _chain_actor(chain)
    lane = actor_lanes.get(actor)
    if not isinstance(lane, dict) or lane.get("concurrency_unit") != "coral-team":
        base = actor
    else:
        source = str(lane.get("source_task_id") or "")
        task = source.rsplit("/", 1)[-1] if source else actor
        if lane.get("lane_kind") == "agent":
            base = (
                f"slot-{int(lane['team_slot']):02d}/"
                f"g{int(lane['slot_generation']):02d}/"
                f"{lane['agent_id']} · {task}"
            )
        else:
            base = (
                f"slot-{int(lane['team_slot']):02d}/"
                f"g{int(lane['slot_generation']):02d}/team · {task}"
            )
    session = _chain_session(chain)
    if session is None:
        return base
    relative = session.removeprefix(f"{actor}/")
    return f"↳ {base}/{relative} · subagent"


def _lane_sort_key(
    actor: str,
    first_start: dict[str, float],
    actor_lanes: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    chain = actor
    actor = _chain_actor(chain)
    child = _chain_session(chain)
    lane = actor_lanes.get(actor)
    if isinstance(lane, dict) and lane.get("concurrency_unit") == "coral-team":
        return (
            0,
            int(lane.get("team_slot", -1)),
            int(lane.get("slot_generation", -1)),
            int(lane.get("agent_index", 0)),
            0 if child is None else 1,
            child or "",
            chain,
        )
    return (1, first_start[chain], actor, 0 if child is None else 1, child or "")


def _render_png(
    path: Path,
    *,
    llm_rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    composite_rows: list[dict[str, Any]],
    restart_rows: list[dict[str, Any]],
    termination_rows: list[dict[str, Any]],
    gate_epoch_s: float,
    window_s: float,
    gaps: list[dict[str, Any]],
    view_kind: str,
    actor_lanes: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # LLM and tool work belong to the same causal actor.  Keeping one row per
    # actor makes LLM -> tool -> LLM transitions directly visible and avoids the
    # misleading global LLM band that used to hide cross-lane interleaving.
    all_rows = [*llm_rows, *tool_rows, *composite_rows]
    first_start: dict[str, float] = {}
    for row in all_rows:
        actor = str(row["chain"])
        first_start[actor] = min(first_start.get(actor, float("inf")), float(row["ts_start"]))
    for row in termination_rows:
        actor = str(row["chain"])
        first_start[actor] = min(
            first_start.get(actor, float("inf")),
            gate_epoch_s + float(row["seconds_since_gate"]),
        )
    for row in restart_rows:
        actor = str(row["chain"])
        first_start[actor] = min(
            first_start.get(actor, float("inf")),
            gate_epoch_s + float(row["start_seconds_since_gate"]),
        )
    lanes = sorted(
        first_start,
        key=lambda actor: _lane_sort_key(actor, first_start, actor_lanes),
    )
    lane_y = {lane: index for index, lane in enumerate(reversed(lanes))}
    height = max(2.4, 0.38 * max(1, len(lanes)) + 1.5)
    # Keep the timeline itself wide while reserving a right rail for the legend.
    # Owl has enough primitive kinds that an in-axes legend otherwise hides the
    # first two actor lanes — exactly where long browser composites tend to be.
    fig, ax = plt.subplots(figsize=(18, height))

    # A gap is represented by literal white space.  Coloring every globally idle
    # interval red made "no LLM/tool right now" look like a cutoff, especially
    # when a CORAL restart control was still pending.  Actual team termination
    # has its own exact red tick and hatched post-cutoff region below.

    # Composite orchestration is context, not replayable work. Draw only an
    # unfilled envelope behind its child LLM/tool spans; it must never hide them
    # or contribute to timeline coverage.
    for row in composite_rows:
        start = float(row["ts_start"]) - gate_epoch_s
        end = float(row["ts_end"]) - gate_epoch_s
        edge = "#d62728" if row.get("timeline_kind") == "truncated" else "#777777"
        ax.broken_barh(
            [(start, max(0.0, end - start))],
            (lane_y[str(row["chain"])] - 0.4, 0.8),
            facecolors="none",
            edgecolors=edge,
            linewidth=1.0,
            linestyles="dashed",
            zorder=1,
        )

    for row in llm_rows:
        start = float(row["ts_start"]) - gate_epoch_s
        end = float(row["ts_end"]) - gate_epoch_s
        color = "#d62728" if row.get("timeline_kind") == "truncated" else "#444444"
        ax.broken_barh(
            [(start, max(0.0, end - start))],
            (lane_y[str(row["chain"])] - 0.35, 0.7),
            facecolors=color,
            edgecolors="black",
            linewidth=0.25,
            zorder=2,
        )

    for row in restart_rows:
        actor = str(row["chain"])
        start = float(row["start_seconds_since_gate"])
        end = float(row["end_seconds_since_gate"])
        color = "#0f766e" if row["timeline_kind"] == "heartbeat" else "#7c3aed"
        ax.broken_barh(
            [(start, max(0.0, end - start))],
            (lane_y[actor] - 0.3, 0.6),
            facecolors=color,
            edgecolors=color,
            linewidth=0.6,
            alpha=0.28,
            hatch="\\\\",
            zorder=1.5,
        )

    tool_names = sorted({str(row["tool"]) for row in tool_rows})
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(index % 20) for index, name in enumerate(tool_names)}
    for row in tool_rows:
        start = float(row["ts_start"]) - gate_epoch_s
        end = float(row["ts_end"]) - gate_epoch_s
        color = "#d62728" if row.get("timeline_kind") == "truncated" else colors[row["tool"]]
        ax.broken_barh(
            [(start, max(0.0, end - start))],
            (lane_y[str(row["chain"])] - 0.35, 0.7),
            facecolors=color,
            edgecolors="black",
            linewidth=0.25,
            zorder=2,
        )

    # This is a state boundary, not work. The short per-lane red tick identifies
    # the exact team-manager kill time; the faint hatch explicitly means that the
    # remaining source window contains no possible work for that lane.
    for row in termination_rows:
        actor = str(row["chain"])
        at = float(row["seconds_since_gate"])
        for chain, y in lane_y.items():
            if _chain_actor(chain) != actor:
                continue
            ax.broken_barh(
                [(at, max(0.0, window_s - at))],
                (y - 0.4, 0.8),
                facecolors="#d62728",
                edgecolors="#d62728",
                linewidth=0.0,
                alpha=0.055,
                hatch="///",
                zorder=0.5,
            )
            ax.vlines(at, y - 0.42, y + 0.42, color="#b2182b", linewidth=1.8, zorder=3)

    if lanes:
        ordered = list(reversed(lanes))
        ax.set_yticks(
            range(len(ordered)),
            [_lane_label(actor, actor_lanes) for actor in ordered],
        )
        previous_group: tuple[int, int] | None = None
        for y, actor in enumerate(ordered):
            lane = actor_lanes.get(_chain_actor(actor), {})
            if lane.get("concurrency_unit") != "coral-team":
                continue
            group = (int(lane["team_slot"]), int(lane["slot_generation"]))
            if previous_group is not None and group != previous_group:
                ax.axhline(y - 0.5, color="#bbbbbb", linewidth=0.7, zorder=0)
            previous_group = group
    else:
        ax.set_yticks([])
    ax.set_xlim(0, max(window_s, 0.001))
    gate_label = "replay" if view_kind == "replay" else "source"
    ax.set_xlabel(f"seconds since {gate_label} gate")
    ax.set_title(f"{view_kind} timeline: per-lane LLM generation and native tool execution")
    ax.grid(axis="x", alpha=0.2)
    legend = [Patch(facecolor="#444444", label="LLM")]
    legend.extend(Patch(facecolor=colors[name], label=name) for name in tool_names[:12])
    if composite_rows:
        legend.append(
            Patch(
                facecolor="none",
                edgecolor="#777777",
                linestyle="dashed",
                label="composite scope",
            )
        )
    if any(row.get("timeline_kind") == "truncated" for row in [*llm_rows, *tool_rows]):
        legend.append(Patch(facecolor="#d62728", label="truncated at cutoff"))
    if any(row["timeline_kind"] == "restart" for row in restart_rows):
        legend.append(
            Patch(
                facecolor="#7c3aed",
                edgecolor="#7c3aed",
                alpha=0.28,
                hatch="\\\\",
                label="agent restart (control, not work)",
            )
        )
    if any(row["timeline_kind"] == "heartbeat" for row in restart_rows):
        legend.append(
            Patch(
                facecolor="#0f766e",
                edgecolor="#0f766e",
                alpha=0.28,
                hatch="\\\\",
                label="heartbeat restart (control, not work)",
            )
        )
    if termination_rows:
        legend.append(
            Patch(
                facecolor="#d62728",
                edgecolor="#b2182b",
                alpha=0.12,
                hatch="///",
                label="team terminated; no work after",
            )
        )
    ax.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_step3(
    *,
    output: Path,
    run_id: str,
    framework: str,
    records: dict[str, list[dict[str, Any]]],
    cutoff_tails: dict[str, Any],
    scope_event_dir: Path | None = None,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
    view_kind: str = "recording",
    actor_lanes: dict[str, dict[str, Any]] | None = None,
    task_terminals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write Step3 raw streams plus text/PNG views for one source recording."""

    root = output / "step3"
    actor_lanes = dict(actor_lanes or {})
    raw = root / "raw"
    views = root / "views"
    raw.mkdir(parents=True, exist_ok=True)
    views.mkdir(parents=True, exist_ok=True)

    llm_tails = list(cutoff_tails.get("llm_requests", []))
    operation_tails = list(cutoff_tails.get("operations", []))
    dispatch_tails = [tail for tail in operation_tails if tail.get("kind") == "dispatch"]
    tool_tails = [tail for tail in operation_tails if tail.get("kind") == "tool"]
    tool_records, tool_tails = _tool_sessions(
        list(records.get("tool", [])),
        tool_tails,
        list(records.get("dispatch", [])),
        dispatch_tails,
    )
    legacy_composites: list[dict[str, Any]] = []
    if framework == "owl":
        tool_records, tool_tails, legacy_composites = _legacy_owl_composites(
            tool_records,
            tool_tails,
            terminal_at_ns=terminal_at_ns,
        )
    if framework == "coral":
        tool_records, tool_tails, coral_composites = _coral_task_composites(
            tool_records,
            tool_tails,
            terminal_at_ns=terminal_at_ns,
        )
        legacy_composites.extend(coral_composites)
    llm_rows = _llm_rows(
        records.get("llm", []),
        llm_tails,
        gate_at_ns=gate_at_ns,
        gate_at_epoch_ns=gate_at_epoch_ns,
        terminal_at_ns=terminal_at_ns,
    )
    tool_rows = _tool_rows(
        tool_records,
        tool_tails,
        gate_at_ns=gate_at_ns,
        gate_at_epoch_ns=gate_at_epoch_ns,
        terminal_at_ns=terminal_at_ns,
    )
    scopes = (
        composite_scope_rows(scope_event_dir, terminal_at_ns) if scope_event_dir is not None else []
    )
    composite_rows = _composite_rows(
        [*scopes, *legacy_composites],
        gate_at_ns=gate_at_ns,
        gate_at_epoch_ns=gate_at_epoch_ns,
        terminal_at_ns=terminal_at_ns,
    )
    termination_rows = _lane_termination_rows(
        list(task_terminals or []),
        actor_lanes,
        gate_at_ns=gate_at_ns,
        gate_at_epoch_ns=gate_at_epoch_ns,
        terminal_at_ns=terminal_at_ns,
    )
    restart_rows = _coral_restart_rows(
        list(task_terminals or []),
        actor_lanes,
        gate_at_ns=gate_at_ns,
        gate_at_epoch_ns=gate_at_epoch_ns,
        terminal_at_ns=terminal_at_ns,
    )
    _write_jsonl(raw / "llm_spans.jsonl", llm_rows)
    _write_jsonl(raw / "tool_events.jsonl", tool_rows)
    _write_jsonl(raw / "composite_scopes.jsonl", composite_rows)
    _write_jsonl(raw / "restart_events.jsonl", restart_rows)
    _write_jsonl(raw / "lane_terminations.jsonl", termination_rows)
    for name in ("engine_occupancy.jsonl", "container_setup.jsonl"):
        (raw / name).touch()

    timeline_records = {
        "llm": [
            {
                "started_at_ns": int((row["ts_start"] * 1e9) - gate_at_epoch_ns + gate_at_ns),
                "ended_at_ns": int((row["ts_end"] * 1e9) - gate_at_epoch_ns + gate_at_ns),
                "name": f"llm:{row['role']}",
                "actor_id": row["chain"],
                "status": row.get("timeline_kind") or "ok",
            }
            for row in llm_rows
        ],
        "tool": [
            {
                "started_at_ns": int((row["ts_start"] * 1e9) - gate_at_epoch_ns + gate_at_ns),
                "ended_at_ns": int((row["ts_end"] * 1e9) - gate_at_epoch_ns + gate_at_ns),
                "name": row["tool"],
                "actor_id": row["chain"],
                "status": row.get("timeline_kind") or ("ok" if row["success"] else "error"),
            }
            for row in tool_rows
        ],
    }
    timeline = build_timeline(
        records=timeline_records,
        gate_at_ns=gate_at_ns,
        terminal_at_ns=terminal_at_ns,
    )
    gate_epoch_s = gate_at_epoch_ns / 1e9
    _write_text(
        views / "timeline.txt",
        llm_rows=llm_rows,
        tool_rows=tool_rows,
        composite_rows=composite_rows,
        restart_rows=restart_rows,
        termination_rows=termination_rows,
        gate_epoch_s=gate_epoch_s,
        window_s=timeline["window_s"],
        timeline=timeline,
        view_kind=view_kind,
        actor_lanes=actor_lanes,
    )
    _render_png(
        views / "timeline.png",
        llm_rows=llm_rows,
        tool_rows=tool_rows,
        composite_rows=composite_rows,
        restart_rows=restart_rows,
        termination_rows=termination_rows,
        gate_epoch_s=gate_epoch_s,
        window_s=timeline["window_s"],
        gaps=timeline["gaps"],
        view_kind=view_kind,
        actor_lanes=actor_lanes,
    )
    metadata = {
        "schema_version": STEP3_SCHEMA,
        "run_id": run_id,
        "framework": framework,
        "source": "minireplay-replay" if view_kind == "replay" else "minireplay-record",
        "window": {
            "gate_at_ns": gate_at_ns,
            "gate_at_epoch_ns": gate_at_epoch_ns,
            "terminal_at_ns": terminal_at_ns,
            "duration_s": timeline["window_s"],
        },
        "counts": {
            "llm": len(llm_rows),
            "tool": len(tool_rows),
            "truncated_llm": len(llm_tails),
            "truncated_tool": len(tool_tails),
            "composite_scope": len(composite_rows),
            "truncated_composite_scope": sum(
                row.get("timeline_kind") == "truncated" for row in composite_rows
            ),
            "lane_termination": len(termination_rows),
            "restart": sum(row["timeline_kind"] == "restart" for row in restart_rows),
            "heartbeat_restart": sum(row["timeline_kind"] == "heartbeat" for row in restart_rows),
        },
        "timeline": timeline,
    }
    if actor_lanes:
        metadata["actor_lanes"] = actor_lanes
    atomic_write_json(root / "metadata.json", metadata)
    return metadata
