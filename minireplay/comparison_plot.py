"""Render a recording and any number of validated replays on shared lane views."""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from collections.abc import Sequence
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .bundle import Bundle, load_bundle
from .constants import COMPARISON_SCHEMA
from .util import atomic_write, atomic_write_json, read_json, require

_RUN_COLORS = (
    "#3B82F6",
    "#F59E0B",
    "#10B981",
    "#8B5CF6",
    "#EC4899",
    "#06B6D4",
    "#84CC16",
    "#F97316",
    "#6366F1",
    "#14B8A6",
)
_RUN_MARKERS = ("o", "D", "s", "^", "v", "P", "X", "*", "<", ">")
_SPAN_COLORS = {"llm": "#3B82F6", "tool": "#F59E0B"}
_FORMATS = frozenset({"png", "svg"})


def _closed_by_actor(timeline: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    spans = timeline.get("spans")
    require(isinstance(spans, list), "timeline spans must be a list")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        require(isinstance(span, dict), "timeline span must be an object")
        if span.get("lane") not in {"llm", "tool"}:
            continue
        require(
            span.get("name") != "browse_url",
            "browse_url is a composite scope and cannot be a filled tool span",
        )
        actor = span.get("actor_id")
        require(isinstance(actor, str) and bool(actor), "timeline span has no actor")
        start = span.get("start_s")
        end = span.get("end_s")
        require(
            isinstance(start, (int, float))
            and not isinstance(start, bool)
            and isinstance(end, (int, float))
            and not isinstance(end, bool)
            and float(end) >= float(start),
            f"timeline span for {actor} has invalid bounds",
        )
        grouped.setdefault(actor, []).append(span)
    return grouped


def _lane_stats(spans: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not spans:
        return {
            "first_s": None,
            "closed_end_s": None,
            "wallclock_s": 0.0,
            "llm_count": 0,
            "tool_count": 0,
            "llm_busy_s": 0.0,
            "tool_busy_s": 0.0,
        }
    first = min(float(span["start_s"]) for span in spans)
    end = max(float(span["end_s"]) for span in spans)
    llm = [span for span in spans if span["lane"] == "llm"]
    tools = [span for span in spans if span["lane"] == "tool"]
    return {
        "first_s": first,
        "closed_end_s": end,
        "wallclock_s": end - first,
        "llm_count": len(llm),
        "tool_count": len(tools),
        "llm_busy_s": sum(float(span["end_s"]) - float(span["start_s"]) for span in llm),
        "tool_busy_s": sum(float(span["end_s"]) - float(span["start_s"]) for span in tools),
    }


def _rounded(
    stats: dict[str, float | int | None],
) -> dict[str, float | int | None]:
    return {
        key: round(value, 6) if isinstance(value, float) else value for key, value in stats.items()
    }


def _short_actor(actor: str) -> str:
    return actor.split("-", 1)[0]


def _actor_lanes(bundle: Bundle) -> dict[str, dict[str, Any]]:
    return {
        str(actor["actor_id"]): actor["lane"]
        for actor in bundle.manifest["actors"]
        if isinstance(actor.get("lane"), dict)
    }


def _display_actor(actor: str, actor_lanes: dict[str, dict[str, Any]]) -> str:
    lane = actor_lanes.get(actor)
    if not isinstance(lane, dict) or lane.get("concurrency_unit") != "coral-team":
        return _short_actor(actor)
    source = str(lane.get("source_task_id") or "")
    task = source.rsplit("/", 1)[-1] if source else _short_actor(actor)
    return (
        f"S{int(lane['team_slot']):02d}/"
        f"G{int(lane['slot_generation']):02d}/"
        f"A{int(lane['agent_index'])}  {task}"
    )


def _work_actors(bundle: Bundle) -> list[str]:
    window = bundle.manifest.get("concurrency_window")
    if not isinstance(window, dict):
        return bundle.actor_ids()
    return [
        str(actor)
        for slot in window["slots"]
        for attempt in slot["attempts"]
        for actor in attempt["agent_actor_ids"]
    ]


def _tail_counts(
    tails: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    llm: dict[str, int] = {}
    tool: dict[str, int] = {}
    for tail in tails.get("llm_requests", []):
        actor = str(tail["actor_id"])
        llm[actor] = llm.get(actor, 0) + 1
    for tail in tails.get("operations", []):
        if tail.get("kind") != "tool":
            continue
        actor = str(tail["actor_id"])
        tool[actor] = tool.get(actor, 0) + 1
    return llm, tool


def _spread(values: list[float]) -> dict[str, float | int]:
    require(bool(values), "cannot calculate an empty spread")
    mean = statistics.fmean(values)
    low, high = min(values), max(values)
    return {
        "n": len(values),
        "mean": round(mean, 6),
        "min": round(low, 6),
        "max": round(high, 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "relative_spread": round((high - low) / mean, 6) if mean > 0 else 0.0,
    }


def _percentile(values: list[float], percentile: float) -> float:
    require(bool(values), "cannot calculate an empty percentile")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _validate_operation_counts(
    *,
    bundle: Bundle,
    metrics: dict[str, Any],
    run_label: str,
) -> None:
    operations = metrics.get("operations")
    require(isinstance(operations, dict), f"{run_label}: metrics operations must be an object")
    for kind, expected in bundle.manifest["counts"].items():
        observed = operations.get(kind, {}).get("count", 0)
        require(
            observed == expected,
            f"{run_label}: observed {observed} {kind} records, expected {expected}",
        )


def _load_run(
    *,
    root: Path,
    label: str,
    bundle: Bundle,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    require(root.is_dir(), f"{label}: run directory does not exist: {root}")
    metrics = read_json(root / "metrics.json")
    timeline = read_json(root / "timeline.json")
    verdict = read_json(root / "verdict.json")
    require(bool(verdict.get("valid")), f"{label}: run is invalid ({verdict.get('reason')})")
    _validate_operation_counts(bundle=bundle, metrics=metrics, run_label=label)
    grouped = _closed_by_actor(timeline)
    return {
        "root": root,
        "label": label,
        "metrics": metrics,
        "timeline": timeline,
        "verdict": verdict,
        "groups": grouped,
    }


def _run_names(replay_count: int) -> list[str]:
    return ["record", *(f"replay{index}" for index in range(1, replay_count + 1))]


def _run_labels(runs: list[dict[str, Any]]) -> list[str]:
    labels = ["Record"]
    for index, run in enumerate(runs[1:], 1):
        mode = run["metrics"].get("replay_mode")
        prefix = "Full" if mode == "full" else "Tool-only" if mode == "tool-only" else "Replay"
        labels.append(f"{prefix} replay {index}")
    return labels


def _build_summary(
    *,
    bundle: Bundle,
    run_names: list[str],
    runs: list[dict[str, Any]],
    actors: list[str],
) -> dict[str, Any]:
    llm_tails, tool_tails = _tail_counts(bundle.cutoff_tails)
    actor_lanes = _actor_lanes(bundle)
    rows: list[dict[str, Any]] = []
    for index, actor in enumerate(actors, 1):
        stats = [_lane_stats(run["groups"].get(actor, [])) for run in runs]
        reference = stats[0]
        for candidate in stats[1:]:
            require(
                candidate["llm_count"] == reference["llm_count"]
                and candidate["tool_count"] == reference["tool_count"],
                f"closed-work count drift for actor {actor}",
            )
        row: dict[str, Any] = {
            "lane": index,
            "actor_id": actor,
            "label": _display_actor(actor, actor_lanes),
            "llm_count": reference["llm_count"],
            "tool_count": reference["tool_count"],
            "source_llm_tail_count": llm_tails.get(actor, 0),
            "source_tool_tail_count": tool_tails.get(actor, 0),
        }
        for run_name, stats_for_run in zip(run_names, stats, strict=True):
            row[run_name] = _rounded(stats_for_run)
        rows.append(row)

    replay_metrics = [run["metrics"] for run in runs[1:]]
    replay_names = run_names[1:]
    lane_ranges = [
        max(float(row[name]["wallclock_s"]) for name in replay_names)
        - min(float(row[name]["wallclock_s"]) for name in replay_names)
        for row in rows
    ]
    replay_spread: dict[str, Any] = {
        "makespan_s": _spread([float(metrics["makespan_seconds"]) for metrics in replay_metrics]),
        "busy_span_s": _spread([float(metrics["busy_span_seconds"]) for metrics in replay_metrics]),
        "lane_wallclock_range_mean_s": round(statistics.fmean(lane_ranges), 6),
        "lane_wallclock_range_p95_s": round(_percentile(lane_ranges, 0.95), 6),
        "lane_wallclock_range_max_s": round(max(lane_ranges, default=0.0), 6),
    }
    if len(replay_names) == 2:
        first_name, second_name = replay_names
        makespans = [float(run["metrics"]["makespan_seconds"]) for run in runs[1:]]
        deltas = [
            float(row[second_name]["wallclock_s"]) - float(row[first_name]["wallclock_s"])
            for row in rows
        ]
        replay_spread["pairwise"] = {
            "second_minus_first_makespan_s": round(makespans[1] - makespans[0], 6),
            "second_minus_first_makespan_pct": round(
                100 * (makespans[1] - makespans[0]) / makespans[0], 3
            )
            if makespans[0]
            else 0.0,
            "lane_wallclock_delta_mean_s": round(statistics.fmean(deltas), 6),
            "lane_wallclock_delta_median_s": round(statistics.median(deltas), 6),
            "lane_wallclock_delta_p95_abs_s": round(
                _percentile([abs(value) for value in deltas], 0.95), 6
            ),
            "lane_wallclock_delta_max_abs_s": round(
                max((abs(value) for value in deltas), default=0.0), 6
            ),
        }

    return {
        "schema_version": COMPARISON_SCHEMA,
        "definition": {
            "lane_wallclock_s": ("last closed LLM/tool end minus first closed LLM/tool start"),
            "closed_end_s": "last closed LLM/tool end since concurrency gate",
            "dispatch": "omitted because it wraps native tool execution",
            "browse_url": "composite scope only; excluded from filled spans and work counts",
            "cutoff_tails": "source diagnostics only; excluded from replay and lane wallclock",
        },
        "bundle": {
            "bundle_id": bundle.manifest["bundle_id"],
            "adapter": bundle.adapter,
            "workload": bundle.manifest["workload"],
        },
        "batch": {
            run_name: {
                "root": str(run["root"]),
                "run_id": run["metrics"]["run_id"],
                "makespan_s": run["metrics"]["makespan_seconds"],
                "busy_span_s": run["metrics"]["busy_span_seconds"],
                "replay_mode": run["metrics"].get("replay_mode"),
            }
            for run_name, run in zip(run_names, runs, strict=True)
        },
        "fixed_work": bundle.manifest["counts"],
        "source_cutoff_tails": {
            "llm": len(bundle.cutoff_tails.get("llm_requests", [])),
            "operation": len(bundle.cutoff_tails.get("operations", [])),
        },
        "replay_spread": replay_spread,
        "lanes": rows,
    }


def _write_summary(
    *,
    output: Path,
    prefix: str,
    summary: dict[str, Any],
    run_names: list[str],
) -> tuple[Path, Path]:
    json_path = output / f"{prefix}-per-lane.json"
    csv_path = output / f"{prefix}-per-lane.csv"
    atomic_write_json(json_path, summary)

    base_fields = [
        "lane",
        "actor_id",
        "llm_count",
        "tool_count",
        "source_llm_tail_count",
        "source_tool_tail_count",
    ]
    stat_fields = [
        "first_s",
        "closed_end_s",
        "wallclock_s",
        "llm_busy_s",
        "tool_busy_s",
    ]
    fields = [
        *base_fields,
        *(f"{name}_{field}" for name in run_names for field in stat_fields),
    ]
    payload = io.StringIO(newline="")
    writer = csv.DictWriter(payload, fieldnames=fields)
    writer.writeheader()
    for row in summary["lanes"]:
        flat = {field: row[field] for field in base_fields}
        for run_name in run_names:
            for field in stat_fields:
                flat[f"{run_name}_{field}"] = row[run_name][field]
        writer.writerow(flat)
    atomic_write(csv_path, payload.getvalue().encode("utf-8"))
    return json_path, csv_path


def _save_figure(
    *,
    figure: Any,
    output: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for output_format in formats:
        path = output / f"{stem}.{output_format}"
        figure.savefig(
            path,
            dpi=dpi if output_format == "png" else None,
            facecolor=figure.get_facecolor(),
        )
        paths[output_format] = path
    plt.close(figure)
    return paths


def _plot_wallclock(
    *,
    output: Path,
    prefix: str,
    workload_label: str,
    run_labels: list[str],
    run_names: list[str],
    runs: list[dict[str, Any]],
    summary: dict[str, Any],
    formats: Sequence[str],
) -> dict[str, Path]:
    rows = summary["lanes"]
    labels = [
        f"L{row['lane']:02d}  {row.get('label', _short_actor(row['actor_id']))}" for row in rows
    ]
    figure_height = max(9.2, 5.0 + 0.36 * len(rows) + 0.12 * len(runs))
    fig = plt.figure(figsize=(15.5, figure_height), facecolor="#fbfcfe")
    batch_weight = max(1.35, 0.42 * len(runs))
    grid = GridSpec(2, 1, height_ratios=[batch_weight, 4.65], hspace=0.25, figure=fig)
    batch_axis = fig.add_subplot(grid[0])
    lane_axis = fig.add_subplot(grid[1])
    for axis in (batch_axis, lane_axis):
        axis.set_facecolor("#fbfcfe")

    colors = list(islice(cycle(_RUN_COLORS), len(runs)))
    markers = list(islice(cycle(_RUN_MARKERS), len(runs)))
    batch_values = [float(run["metrics"]["makespan_seconds"]) for run in runs]
    bars = batch_axis.barh(
        run_labels,
        batch_values,
        color=colors,
        edgecolor="#1F2937",
        height=0.55,
    )
    for bar, value in zip(bars, batch_values, strict=True):
        batch_axis.text(
            value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}s",
            va="center",
            fontsize=10,
        )
    replay_values = batch_values[1:]
    if len(replay_values) == 2:
        delta = replay_values[1] - replay_values[0]
        detail = (
            f"Replay 2 − Replay 1 = {delta:+.3f}s ({100 * delta / replay_values[0]:+.2f}%)"
            if replay_values[0]
            else f"Replay 2 − Replay 1 = {delta:+.3f}s"
        )
    else:
        detail = f"replay range = {max(replay_values) - min(replay_values):.3f}s"
    batch_axis.set_xlim(0, max(batch_values) + max(2.0, 0.06 * max(batch_values)))
    batch_axis.set_title(
        f"Batch wallclock  ·  {detail}",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    batch_axis.grid(axis="x", color="#CBD5E1", alpha=0.7)
    batch_axis.spines[["top", "right", "left"]].set_visible(False)
    batch_axis.tick_params(axis="y", length=0)

    values = [[float(row[run_name]["wallclock_s"]) for row in rows] for run_name in run_names]
    y = list(range(len(rows)))
    for lane_index in y:
        lane_values = [value[lane_index] for value in values]
        lane_axis.plot(
            [min(lane_values), max(lane_values)],
            [lane_index, lane_index],
            color="#94A3B8",
            linewidth=1.2,
            zorder=1,
        )
    for run_label, value, color, marker in zip(run_labels, values, colors, markers, strict=True):
        lane_axis.scatter(
            value,
            y,
            s=48,
            color=color,
            edgecolor="#1F2937",
            linewidth=0.5,
            marker=marker,
            label=f"{run_label} lane wallclock",
            zorder=3,
        )
    lane_axis.set_yticks(y, labels, fontsize=9)
    lane_axis.invert_yaxis()
    lane_max = max((item for value in values for item in value), default=0.0)
    lane_axis.set_xlim(0, lane_max + max(2.0, 0.06 * lane_max))
    lane_axis.set_xlabel(
        "Lane wallclock (seconds): first closed LLM/tool start → last closed LLM/tool end"
    )
    lane_axis.set_title(
        f"Per-lane closed-prefix wallclock ({len(rows)} actor lanes)",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    lane_axis.grid(axis="x", color="#CBD5E1", alpha=0.7)
    lane_axis.spines[["top", "right", "left"]].set_visible(False)
    lane_axis.tick_params(axis="y", length=0)
    lane_axis.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=min(4, len(runs)),
        frameon=False,
    )

    replay_word = "replay" if len(runs) == 2 else "replays"
    fig.suptitle(
        f"{workload_label}: record and {len(runs) - 1} validated {replay_word}",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.075,
        0.017,
        (
            f"Every run contains {summary['fixed_work']['llm']} closed LLM slots "
            f"and {summary['fixed_work']['tool']} native tools. browse_url is a "
            "composite scope, not a filled tool; source cutoff tails are evidence-only."
        ),
        fontsize=9.5,
        color="#64748B",
    )
    fig.subplots_adjust(left=0.18, right=0.96, top=0.93, bottom=0.075)
    return _save_figure(
        figure=fig,
        output=output,
        stem=f"{prefix}-batch-and-per-lane-wallclock",
        formats=formats,
        dpi=180,
    )


def _plot_timeline(
    *,
    output: Path,
    prefix: str,
    workload_label: str,
    run_labels: list[str],
    runs: list[dict[str, Any]],
    actors: list[str],
    source_tails: dict[str, Any],
    formats: Sequence[str],
    actor_lanes: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    labels = [
        f"L{index:02d}  {_display_actor(actor, actor_lanes)}"
        for index, actor in enumerate(actors, 1)
    ]
    figure_height = max(9.0, 5.0 + 0.36 * len(actors))
    fig, raw_axes = plt.subplots(
        1,
        len(runs),
        figsize=(9 * len(runs), figure_height),
        sharey=True,
        facecolor="#fbfcfe",
    )
    axes = list(raw_axes)
    max_window = max(float(run["metrics"]["makespan_seconds"]) for run in runs)
    actor_y = {actor: index for index, actor in enumerate(actors)}
    for run_index, (axis, run_label, run) in enumerate(zip(axes, run_labels, runs, strict=True)):
        axis.set_facecolor("#fbfcfe")
        for yi in range(len(actors)):
            if yi % 2 == 0:
                axis.axhspan(
                    yi - 0.45,
                    yi + 0.45,
                    color="#EEF2F7",
                    alpha=0.75,
                    zorder=0,
                )
            if yi > 0:
                current = actor_lanes.get(actors[yi], {})
                previous = actor_lanes.get(actors[yi - 1], {})
                current_group = (
                    current.get("team_slot"),
                    current.get("slot_generation"),
                )
                previous_group = (
                    previous.get("team_slot"),
                    previous.get("slot_generation"),
                )
                if current_group != previous_group:
                    axis.axhline(yi - 0.5, color="#94A3B8", linewidth=0.8, zorder=1)
        for actor in actors:
            yi = actor_y[actor]
            for span in run["groups"].get(actor, []):
                start = float(span["start_s"])
                width = max(0.001, float(span["end_s"]) - start)
                kind = str(span["lane"])
                axis.broken_barh(
                    [(start, width)],
                    (yi - 0.31, 0.62),
                    facecolors=_SPAN_COLORS[kind],
                    edgecolors="#1F2937",
                    linewidth=0.18,
                    zorder=3,
                )
        if run_index == 0:
            gate = int(run["metrics"]["gate_at_ns"])
            end = float(run["metrics"]["makespan_seconds"])
            tails = [
                (
                    str(tail["actor_id"]),
                    int(tail["started_at_ns"]),
                    tail,
                )
                for tail in source_tails.get("llm_requests", [])
            ]
            tails.extend(
                (
                    str(tail["actor_id"]),
                    int(tail.get("source_started_at_ns", tail.get("started_at_ns"))),
                    tail,
                )
                for tail in source_tails.get("operations", [])
                if tail.get("kind") == "tool"
            )
            for actor, started_at_ns, tail in tails:
                start = (started_at_ns - gate) / 1e9
                elapsed_ns = tail.get("elapsed_ns")
                tail_end = (
                    (started_at_ns + int(elapsed_ns) - gate) / 1e9
                    if isinstance(elapsed_ns, int) and elapsed_ns >= 0
                    else end
                )
                for key in ("interrupted_at_ns", "lane_terminated_at_ns"):
                    upper = tail.get(key)
                    if isinstance(upper, int):
                        tail_end = min(tail_end, (upper - gate) / 1e9)
                tail_end = max(start, min(tail_end, end))
                axis.broken_barh(
                    [(start, max(0.001, tail_end - start))],
                    (actor_y[actor] - 0.31, 0.62),
                    facecolors="none",
                    edgecolors="#DC2626",
                    linewidth=0.9,
                    hatch="////",
                    zorder=5,
                )
        axis.axvline(
            float(run["metrics"]["makespan_seconds"]),
            color="#111827",
            linewidth=1.15,
            zorder=6,
        )
        axis.set_xlim(0, max_window + max(1.0, 0.02 * max_window))
        axis.set_xlabel("Seconds since concurrency gate")
        axis.set_title(
            f"{run_label} · batch {run['metrics']['makespan_seconds']:.3f}s",
            fontsize=14,
            fontweight="bold",
        )
        axis.grid(axis="x", color="#CBD5E1", alpha=0.65)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    axes[0].set_yticks(list(range(len(actors))), labels, fontsize=9)
    axes[0].invert_yaxis()
    legend = [
        Patch(
            facecolor=_SPAN_COLORS["llm"],
            edgecolor="#1F2937",
            label="LLM engine execution",
        ),
        Patch(
            facecolor=_SPAN_COLORS["tool"],
            edgecolor="#1F2937",
            label="Native tool execution",
        ),
        Patch(
            facecolor="none",
            edgecolor="#DC2626",
            hatch="////",
            label="Source cutoff tail (not replayed)",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            linewidth=1.15,
            label="Batch boundary",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.945),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    replay_word = "replay" if len(runs) == 2 else "replays"
    fig.suptitle(
        (
            f"{workload_label} full causal-lane timeline: record and "
            f"{len(runs) - 1} validated {replay_word}"
        ),
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.065,
        0.022,
        (
            "Each row is one actor lane. Dispatch wrappers and browse_url composite "
            "envelopes are omitted; only real LLM and native tool execution is filled."
        ),
        fontsize=9.5,
        color="#64748B",
    )
    fig.subplots_adjust(
        left=max(0.06, 0.12 * 3 / len(runs)),
        right=0.985,
        top=0.89,
        bottom=0.09,
        wspace=0.08,
    )
    return _save_figure(
        figure=fig,
        output=output,
        stem=f"{prefix}-full-chain-lanes-timeline",
        formats=formats,
        dpi=160,
    )


def _default_label(bundle: Bundle) -> str:
    workload = bundle.manifest["workload"]
    framework = str(workload.get("framework", bundle.adapter))
    framework_label = framework.replace("-", " ").title()
    concurrency = workload.get("concurrency", len(bundle.manifest["actors"]))
    return f"{framework_label} C{concurrency}"


def _default_prefix(bundle: Bundle, replay_count: int) -> str:
    concurrency = bundle.manifest["workload"].get("concurrency", len(bundle.manifest["actors"]))
    replay_word = "replay" if replay_count == 1 else "replays"
    raw = f"{bundle.adapter}-c{concurrency}-record-{replay_count}-{replay_word}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-").lower()


def render_comparison(
    *,
    bundle_dir: Path,
    source_dir: Path,
    run_dirs: Sequence[Path],
    output_dir: Path,
    prefix: str | None = None,
    label: str | None = None,
    formats: Sequence[str] = ("svg", "png"),
) -> dict[str, Any]:
    """Validate the fixed work, then write wallclock and causal-lane comparisons."""

    require(bool(run_dirs), "plot-comparison needs at least one replay run")
    selected_formats = tuple(dict.fromkeys(formats))
    require(bool(selected_formats), "plot-comparison needs at least one output format")
    unknown_formats = sorted(set(selected_formats) - _FORMATS)
    require(not unknown_formats, f"unsupported plot formats: {unknown_formats}")

    bundle = load_bundle(bundle_dir)
    runs = [
        _load_run(root=source_dir, label="record", bundle=bundle),
        *(
            _load_run(root=root, label=f"replay {index}", bundle=bundle)
            for index, root in enumerate(run_dirs, 1)
        ),
    ]
    actors = _work_actors(bundle)
    declared = set(bundle.actor_ids())
    for run in runs:
        unknown_actors = sorted(set(run["groups"]) - declared)
        require(
            not unknown_actors,
            f"{run['label']}: timeline contains undeclared actors: {unknown_actors}",
        )
        observed_counts = {
            kind: sum(
                1 for spans in run["groups"].values() for span in spans if span["lane"] == kind
            )
            for kind in ("llm", "tool")
        }
        for kind, observed in observed_counts.items():
            require(
                observed == bundle.manifest["counts"][kind],
                f"{run['label']}: timeline has {observed} {kind} spans, "
                f"expected {bundle.manifest['counts'][kind]}",
            )

    if not isinstance(bundle.manifest.get("concurrency_window"), dict):
        actors.sort(
            key=lambda actor: (
                min(
                    (float(span["start_s"]) for span in runs[0]["groups"].get(actor, [])),
                    default=math.inf,
                ),
                actor,
            )
        )
    run_names = _run_names(len(run_dirs))
    run_labels = _run_labels(runs)
    summary = _build_summary(
        bundle=bundle,
        run_names=run_names,
        runs=runs,
        actors=actors,
    )

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_prefix = prefix or _default_prefix(bundle, len(run_dirs))
    require(
        bool(selected_prefix)
        and selected_prefix not in {".", ".."}
        and Path(selected_prefix).name == selected_prefix,
        "plot prefix must be a non-empty filename stem",
    )
    workload_label = label or _default_label(bundle)
    summary_path, csv_path = _write_summary(
        output=output,
        prefix=selected_prefix,
        summary=summary,
        run_names=run_names,
    )
    wallclock_paths = _plot_wallclock(
        output=output,
        prefix=selected_prefix,
        workload_label=workload_label,
        run_labels=run_labels,
        run_names=run_names,
        runs=runs,
        summary=summary,
        formats=selected_formats,
    )
    timeline_paths = _plot_timeline(
        output=output,
        prefix=selected_prefix,
        workload_label=workload_label,
        run_labels=run_labels,
        runs=runs,
        actors=actors,
        source_tails=bundle.cutoff_tails,
        formats=selected_formats,
        actor_lanes=_actor_lanes(bundle),
    )
    return {
        "summary": str(summary_path),
        "csv": str(csv_path),
        "wallclock": {output_format: str(path) for output_format, path in wallclock_paths.items()},
        "timeline": {output_format: str(path) for output_format, path in timeline_paths.items()},
        "replay_spread": summary["replay_spread"],
    }
