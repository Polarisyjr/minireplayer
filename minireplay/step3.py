"""Export every source recording as a Step3-compatible timeline.

Step3 is an observation format, not a sweep profiler.  The sweep therefore stays
at ``-s none`` while the recorder projects its exact LLM/tool ledgers onto one row
per causal actor lane.  Composite orchestration envelopes are diagnostic context,
never tools: they are drawn as transparent outlines and are excluded from work
coverage. Calls that were still active at the source cutoff are included as
truncated spans.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .lane_record import composite_scope_rows
from .timeline import build_timeline
from .util import atomic_write, atomic_write_json, canonical_json

STEP3_SCHEMA = "minireplay.step3/v1"


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
        rows.append(
            {
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
                "chain": str(record.get("actor_id") or "?"),
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
        )
    for tail in tails:
        started = int(tail["started_at_ns"])
        rows.append(
            {
                "ts_start": _epoch(
                    started,
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "ts_end": _epoch(
                    terminal_at_ns,
                    gate_at_ns=gate_at_ns,
                    gate_at_epoch_ns=gate_at_epoch_ns,
                    terminal_at_ns=terminal_at_ns,
                ),
                "role": str(tail.get("role") or "agent"),
                "chain": str(tail.get("actor_id") or "?"),
                "port": _port(tail.get("target_id")),
                "request_id": tail.get("attempt_id"),
                "prompt_tokens": None,
                "completion_tokens": None,
                "e2e_s": round((terminal_at_ns - started) / 1e9, 6),
                "timeline_kind": "truncated",
                "source": "minireplay-cutoff-tail",
            }
        )
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
        steps[actor] += 1
        started = int(record.get("started_at_ns", record.get("source_started_at_ns", gate_at_ns)))
        ended = terminal_at_ns if truncated else int(record["ended_at_ns"])
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
            "chain": actor,
            "stage": "agent",
            "tool": str(record.get("name") or "tool"),
            "tools": None,
            "success": status in {"ok", "executed", "success"},
            "step": steps[actor],
            "n_calls": 1,
            "source": "minireplay-cutoff-tail" if truncated else "minireplay-tool-boundary",
        }
        if truncated:
            row["timeline_kind"] = "truncated"
        rows.append(row)

    for record in records:
        add(record, truncated=False)
    for tail in tails:
        add(tail, truncated=True)
    return sorted(rows, key=lambda row: (row["ts_start"], row["ts_end"]))


def _composite_rows(
    records: list[dict[str, Any]],
    *,
    gate_at_ns: int,
    gate_at_epoch_ns: int,
    terminal_at_ns: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "scope_id": record["scope_id"],
                "chain": record["actor_id"],
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
        )
    return sorted(rows, key=lambda row: (row["ts_start"], row["ts_end"]))


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
    gate_epoch_s: float,
    window_s: float,
    timeline: dict[str, Any],
    view_kind: str,
) -> None:
    activities: list[tuple[float, float, str]] = []
    for row in llm_rows:
        suffix = " [truncated]" if row.get("timeline_kind") == "truncated" else ""
        activities.append(
            (
                float(row["ts_start"]) - gate_epoch_s,
                float(row["ts_end"]) - gate_epoch_s,
                f"LLM:{row['role']} [{row['chain']}]{suffix}",
            )
        )
    for row in tool_rows:
        suffix = " [truncated]" if row.get("timeline_kind") == "truncated" else ""
        activities.append(
            (
                float(row["ts_start"]) - gate_epoch_s,
                float(row["ts_end"]) - gate_epoch_s,
                f"tool:{row['tool']} [{row['chain']}]{suffix}",
            )
        )
    for row in composite_rows:
        suffix = " [open at cutoff]" if row.get("timeline_kind") == "truncated" else ""
        activities.append(
            (
                float(row["ts_start"]) - gate_epoch_s,
                float(row["ts_end"]) - gate_epoch_s,
                f"scope:{row['name']} [{row['chain']}]{suffix}",
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


def _render_png(
    path: Path,
    *,
    llm_rows: list[dict[str, Any]],
    tool_rows: list[dict[str, Any]],
    composite_rows: list[dict[str, Any]],
    gate_epoch_s: float,
    window_s: float,
    gaps: list[dict[str, Any]],
    view_kind: str,
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
    lanes = sorted(first_start, key=lambda actor: (first_start[actor], actor))
    lane_y = {lane: index for index, lane in enumerate(reversed(lanes))}
    height = max(2.4, 0.38 * max(1, len(lanes)) + 1.5)
    # Keep the timeline itself wide while reserving a right rail for the legend.
    # Owl has enough primitive kinds that an in-axes legend otherwise hides the
    # first two actor lanes — exactly where long browser composites tend to be.
    fig, ax = plt.subplots(figsize=(18, height))

    for gap in gaps:
        ax.axvspan(gap["start_s"], gap["end_s"], color="#d6483f", alpha=0.12, zorder=0)

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

    if lanes:
        ordered = list(reversed(lanes))
        ax.set_yticks(range(len(ordered)), ordered)
    else:
        ax.set_yticks([])
    ax.set_xlim(0, max(window_s, 0.001))
    gate_label = "replay" if view_kind == "replay" else "source"
    ax.set_xlabel(f"seconds since {gate_label} gate")
    ax.set_title(
        f"{view_kind} timeline: per-lane LLM generation and native tool execution"
    )
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
) -> dict[str, Any]:
    """Write Step3 raw streams plus text/PNG views for one source recording."""

    root = output / "step3"
    raw = root / "raw"
    views = root / "views"
    raw.mkdir(parents=True, exist_ok=True)
    views.mkdir(parents=True, exist_ok=True)

    llm_tails = list(cutoff_tails.get("llm_requests", []))
    tool_tails = [tail for tail in cutoff_tails.get("operations", []) if tail.get("kind") == "tool"]
    tool_records = list(records.get("tool", []))
    legacy_composites: list[dict[str, Any]] = []
    if framework == "owl":
        tool_records, tool_tails, legacy_composites = _legacy_owl_composites(
            tool_records,
            tool_tails,
            terminal_at_ns=terminal_at_ns,
        )
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
        composite_scope_rows(scope_event_dir, terminal_at_ns)
        if scope_event_dir is not None
        else []
    )
    composite_rows = _composite_rows(
        [*scopes, *legacy_composites],
        gate_at_ns=gate_at_ns,
        gate_at_epoch_ns=gate_at_epoch_ns,
        terminal_at_ns=terminal_at_ns,
    )
    _write_jsonl(raw / "llm_spans.jsonl", llm_rows)
    _write_jsonl(raw / "tool_events.jsonl", tool_rows)
    _write_jsonl(raw / "composite_scopes.jsonl", composite_rows)
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
        gate_epoch_s=gate_epoch_s,
        window_s=timeline["window_s"],
        timeline=timeline,
        view_kind=view_kind,
    )
    _render_png(
        views / "timeline.png",
        llm_rows=llm_rows,
        tool_rows=tool_rows,
        composite_rows=composite_rows,
        gate_epoch_s=gate_epoch_s,
        window_s=timeline["window_s"],
        gaps=timeline["gaps"],
        view_kind=view_kind,
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
        },
        "timeline": timeline,
    }
    atomic_write_json(root / "metadata.json", metadata)
    return metadata
