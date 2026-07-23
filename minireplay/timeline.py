"""Unify every activity lane onto one clock and name the blanks.

Borrowed from step3's gap analysis: a per-lane view shows a lane as idle whenever
its work is happening somewhere else, which hides whether the run was actually
doing anything. Merging the lanes makes a real blank visible, and a blank that no
lane explains is the signal that something ran outside the instrumentation.
"""

from __future__ import annotations

from typing import Any

from .constants import TIMELINE_SCHEMA

DEFAULT_MIN_GAP_S = 2.0

_LANE_OF = {
    "llm": "llm",
    "tool": "tool",
    "dispatch": "dispatch",
    "grader": "grader",
    "artifact": "artifact",
}


def _intervals(records: dict[str, list[dict[str, Any]]], gate_at_ns: int) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for kind, entries in records.items():
        lane = _LANE_OF.get(kind, kind)
        for entry in entries:
            start = entry.get("started_at_ns")
            end = entry.get("ended_at_ns", entry.get("completed_at_ns"))
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            spans.append(
                {
                    "lane": lane,
                    "name": str(entry.get("name") or entry.get("grader_kind") or lane),
                    "actor_id": entry.get("actor_id"),
                    "status": entry.get("status"),
                    "start_s": round((start - gate_at_ns) / 1e9, 4),
                    "end_s": round((end - gate_at_ns) / 1e9, 4),
                }
            )
    spans.sort(key=lambda item: (item["start_s"], item["end_s"]))
    return spans


def _merge(spans: list[dict[str, Any]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for span in spans:
        start, end = span["start_s"], span["end_s"]
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def find_gaps(
    spans: list[dict[str, Any]],
    *,
    window_s: float,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
) -> list[dict[str, Any]]:
    """Stretches where no lane was doing anything at all."""

    merged = _merge(spans)
    gaps: list[dict[str, Any]] = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor >= min_gap_s:
            gaps.append({"start_s": round(cursor, 3), "end_s": round(start, 3)})
        cursor = max(cursor, end)
    if window_s - cursor >= min_gap_s:
        gaps.append({"start_s": round(cursor, 3), "end_s": round(window_s, 3)})
    for gap in gaps:
        gap["duration_s"] = round(gap["end_s"] - gap["start_s"], 3)
        gap["attribution"] = _attribute(gap, spans)
    return gaps


def _attribute(gap: dict[str, Any], spans: list[dict[str, Any]]) -> str:
    """Name what a blank most likely is, from what surrounds it.

    This is a hint for a human reading the timeline, not a verdict. Real
    attribution needs the run's logs, and the design deliberately leaves that to
    inspection rather than to a rule table that would go stale.
    """

    if gap["start_s"] <= 0.001:
        return "startup: framework had not reached its first instrumented call"
    before = [s for s in spans if s["end_s"] <= gap["start_s"] + 0.001]
    after = [s for s in spans if s["start_s"] >= gap["end_s"] - 0.001]
    if not after:
        return "tail: no further instrumented activity before the window closed"
    previous = max(before, key=lambda s: s["end_s"], default=None)
    following = min(after, key=lambda s: s["start_s"])
    if previous is None:
        return f"before first {following['lane']}:{following['name']}"
    return (
        f"between {previous['lane']}:{previous['name']} and "
        f"{following['lane']}:{following['name']} (uninstrumented framework work)"
    )


def build_timeline(
    *,
    records: dict[str, list[dict[str, Any]]],
    gate_at_ns: int,
    terminal_at_ns: int,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
) -> dict[str, Any]:
    # A run that failed before its gate has no window, and therefore no timeline.
    # The test is on the window, not on the gate's value: zero is a legitimate
    # clock origin.
    if terminal_at_ns <= gate_at_ns:
        return {
            "schema_version": TIMELINE_SCHEMA,
            "window_s": 0.0,
            "busy_s": 0.0,
            "coverage_fraction": 0.0,
            "lanes": {},
            "gaps": [],
            "unattributed_gap_seconds": 0.0,
            "busy_span_s": 0.0,
            "first_activity_s": None,
            "last_activity_s": None,
            "spans": [],
        }
    window_s = (terminal_at_ns - gate_at_ns) / 1e9
    spans = _intervals(records, gate_at_ns)
    gaps = find_gaps(spans, window_s=window_s, min_gap_s=min_gap_s)
    busy_s = sum(end - start for start, end in _merge(spans))
    per_lane: dict[str, dict[str, Any]] = {}
    for span in spans:
        lane = per_lane.setdefault(span["lane"], {"count": 0, "busy_s": 0.0})
        lane["count"] += 1
        lane["busy_s"] = round(lane["busy_s"] + (span["end_s"] - span["start_s"]), 3)
    # The span from first to last instrumented activity. A recording's window is
    # set by the sweep and usually outlasts the work — an agent that submits early
    # leaves the rest of the window idle — so `window_s` is not comparable between a
    # recording and a replay. This is: both measure the same thing.
    merged = _merge(spans)
    first = merged[0][0] if merged else None
    last = merged[-1][1] if merged else None
    return {
        "schema_version": TIMELINE_SCHEMA,
        "window_s": round(window_s, 3),
        "busy_s": round(busy_s, 3),
        "busy_span_s": round(last - first, 3) if merged else 0.0,
        "first_activity_s": round(first, 3) if merged else None,
        "last_activity_s": round(last, 3) if merged else None,
        "coverage_fraction": round(busy_s / window_s, 4) if window_s > 0 else 0.0,
        "lanes": per_lane,
        "gaps": gaps,
        "unattributed_gap_seconds": round(sum(gap["duration_s"] for gap in gaps), 3),
        "spans": spans,
    }
