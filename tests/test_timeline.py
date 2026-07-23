"""Timeline merging and blank detection."""

from __future__ import annotations

from minireplay.timeline import build_timeline, find_gaps

S = 1_000_000_000


def entry(kind_start: int, kind_end: int, *, name: str = "shell") -> dict:
    return {
        "started_at_ns": kind_start,
        "ended_at_ns": kind_end,
        "name": name,
        "actor_id": "actor-0",
        "status": "ok",
    }


def test_overlapping_lanes_merge_into_one_busy_span() -> None:
    """A lane looks idle whenever its work moved elsewhere; merging hides that."""

    timeline = build_timeline(
        records={
            "tool": [entry(0, 10 * S)],
            "llm": [entry(5 * S, 15 * S)],
        },
        gate_at_ns=0,
        terminal_at_ns=15 * S,
    )
    assert timeline["gaps"] == []
    assert timeline["busy_s"] == 15.0
    assert timeline["coverage_fraction"] == 1.0


def test_a_real_blank_is_reported() -> None:
    timeline = build_timeline(
        records={"tool": [entry(0, 2 * S), entry(10 * S, 12 * S)]},
        gate_at_ns=0,
        terminal_at_ns=12 * S,
    )
    assert len(timeline["gaps"]) == 1
    gap = timeline["gaps"][0]
    assert (gap["start_s"], gap["end_s"], gap["duration_s"]) == (2.0, 10.0, 8.0)
    assert "uninstrumented" in gap["attribution"]


def test_a_blank_before_the_first_call_is_named_startup() -> None:
    timeline = build_timeline(
        records={"tool": [entry(5 * S, 6 * S)]},
        gate_at_ns=0,
        terminal_at_ns=6 * S,
    )
    assert timeline["gaps"][0]["attribution"].startswith("startup")


def test_a_trailing_blank_is_named_tail() -> None:
    timeline = build_timeline(
        records={"tool": [entry(0, 1 * S)]},
        gate_at_ns=0,
        terminal_at_ns=10 * S,
    )
    assert timeline["gaps"][-1]["attribution"].startswith("tail")


def test_short_blanks_are_below_the_reporting_threshold() -> None:
    spans = [
        {"lane": "tool", "name": "a", "start_s": 0.0, "end_s": 1.0},
        {"lane": "tool", "name": "b", "start_s": 1.5, "end_s": 2.0},
    ]
    assert find_gaps(spans, window_s=2.0, min_gap_s=2.0) == []


def test_lane_totals_are_reported_per_kind() -> None:
    timeline = build_timeline(
        records={"tool": [entry(0, 2 * S)], "llm": [entry(0, 1 * S)]},
        gate_at_ns=0,
        terminal_at_ns=2 * S,
    )
    assert timeline["lanes"]["tool"] == {"count": 1, "busy_s": 2.0}
    assert timeline["lanes"]["llm"] == {"count": 1, "busy_s": 1.0}


def test_busy_span_excludes_a_window_that_outlasts_the_work() -> None:
    """A recording's window is the sweep's, not the work's.

    An agent that submits early leaves the rest of the window idle, so `window_s`
    for a recording and `window_s` for a replay measure different things. This is
    the number that measures the same thing in both.
    """

    timeline = build_timeline(
        records={"tool": [entry(2 * S, 12 * S)]},
        gate_at_ns=0,
        terminal_at_ns=60 * S,  # the sweep kept running for another 48s
    )
    assert timeline["window_s"] == 60.0
    assert timeline["busy_span_s"] == 10.0
    assert timeline["first_activity_s"] == 2.0
    assert timeline["last_activity_s"] == 12.0
    assert timeline["unattributed_gap_seconds"] == 50.0  # 2s head + 48s tail


def test_busy_span_spans_internal_gaps() -> None:
    """It is first-to-last, not the sum of busy intervals; `busy_s` is that."""

    timeline = build_timeline(
        records={"tool": [entry(0, 1 * S), entry(9 * S, 10 * S)]},
        gate_at_ns=0,
        terminal_at_ns=10 * S,
    )
    assert timeline["busy_span_s"] == 10.0
    assert timeline["busy_s"] == 2.0


def test_a_run_with_no_activity_has_no_span() -> None:
    timeline = build_timeline(records={"tool": []}, gate_at_ns=0, terminal_at_ns=10 * S)
    assert timeline["busy_span_s"] == 0.0
    assert timeline["first_activity_s"] is None
