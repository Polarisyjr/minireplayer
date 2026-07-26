"""Cutoff pruning must leave a causally closed prefix."""

from __future__ import annotations

from pathlib import Path

from minireplay.cutoff import close_stage_at_cutoff
from minireplay.util import append_jsonl, iter_jsonl
from tests.support import dispatch, llm, span, tool


def stage(tmp_path: Path, **records) -> Path:
    root = tmp_path / "stage"
    root.mkdir()
    for relative in (
        "llm.jsonl",
        "spans.jsonl",
        "dispatches.jsonl",
        "tools.jsonl",
        "graders.jsonl",
        "artifacts.jsonl",
    ):
        (root / relative).touch()
    for relative, entries in records.items():
        for entry in entries:
            append_jsonl(root / relative, entry)
    return root


def ids(root: Path, relative: str, field: str) -> set[str]:
    return {str(record[field]) for record in iter_jsonl(root / relative)}


def test_tool_without_its_dispatch_is_dropped(tmp_path: Path) -> None:
    """The dispatch finished after the window; its tool cannot stand alone."""

    root = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [dispatch(dispatch_id="d-kept", execution_call_id="t-kept")],
            "tools.jsonl": [
                tool(call_id="t-kept", dispatch_id="d-kept"),
                tool(call_id="t-orphan", dispatch_id="d-dropped"),
            ],
        },
    )
    close_stage_at_cutoff(root)
    assert ids(root, "tools.jsonl", "call_id") == {"t-kept"}


def test_dispatch_pointing_at_a_dropped_tool_is_dropped(tmp_path: Path) -> None:
    root = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [
                dispatch(dispatch_id="d-ok", execution_call_id="t-ok"),
                dispatch(dispatch_id="d-dangling", execution_call_id="t-missing"),
                dispatch(dispatch_id="d-rejected", execution_call_id=None),
            ],
            "tools.jsonl": [tool(call_id="t-ok", dispatch_id="d-ok")],
        },
    )
    close_stage_at_cutoff(root)
    assert ids(root, "dispatches.jsonl", "dispatch_id") == {"d-ok", "d-rejected"}


def test_pruning_cascades_to_a_fixed_point(tmp_path: Path) -> None:
    """Dropping one record can orphan another; the pass must iterate."""

    root = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [dispatch(dispatch_id="d-0", execution_call_id="t-missing")],
            "tools.jsonl": [tool(call_id="t-child", dispatch_id="d-0")],
        },
    )
    close_stage_at_cutoff(root)
    assert ids(root, "dispatches.jsonl", "dispatch_id") == set()
    assert ids(root, "tools.jsonl", "call_id") == set()


def test_missing_parent_drops_the_whole_descendant_branch(tmp_path: Path) -> None:
    root = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [dispatch(dispatch_id="d-0", execution_call_id="tool-0")],
            "tools.jsonl": [tool(call_id="tool-0", dispatch_id="d-0")],
            "spans.jsonl": [
                span("span-d-0", parent="span-gone"),
                span("span-tool-0", parent="span-d-0"),
            ],
        },
    )
    close_stage_at_cutoff(root)
    assert ids(root, "dispatches.jsonl", "dispatch_id") == set()
    assert ids(root, "tools.jsonl", "call_id") == set()
    assert list(iter_jsonl(root / "spans.jsonl")) == []


def test_llm_below_an_ordinary_cutoff_tool_is_not_fixed_work(tmp_path: Path) -> None:
    child = llm(attempt_id="llm-child")
    root = stage(
        tmp_path,
        **{
            "llm.jsonl": [child],
            "spans.jsonl": [
                {
                    **span(child["span_id"], parent="span-tool-cutoff"),
                    "kind": "llm",
                    "name": "llm:browser",
                }
            ],
        },
    )
    close_stage_at_cutoff(root)
    assert ids(root, "llm.jsonl", "attempt_id") == set()
    assert list(iter_jsonl(root / "spans.jsonl")) == []


def test_closed_llm_below_a_composite_task_cutoff_is_preserved(tmp_path: Path) -> None:
    child = llm(attempt_id="llm-child")
    root = stage(
        tmp_path,
        **{
            "llm.jsonl": [child],
            "spans.jsonl": [
                {
                    **span(child["span_id"], parent="span-task-cutoff"),
                    "kind": "llm",
                    "name": "llm:coral-subagent",
                }
            ],
        },
    )
    close_stage_at_cutoff(
        root,
        cutoff_tails={
            "operations": [
                {
                    "span_id": "span-task-cutoff",
                    "replay_entry": "enter-and-preserve-descendants",
                }
            ]
        },
    )

    assert ids(root, "llm.jsonl", "attempt_id") == {"llm-child"}
    assert ids(root, "spans.jsonl", "span_id") == {child["span_id"]}


def test_report_counts_what_was_discarded(tmp_path: Path) -> None:
    root = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [dispatch(dispatch_id="d-0", execution_call_id="tool-0")],
            "tools.jsonl": [
                tool(call_id="tool-0", dispatch_id="d-0"),
                tool(call_id="t-orphan", dispatch_id="d-gone"),
            ],
        },
    )
    report = close_stage_at_cutoff(root)
    assert report["discarded"]["tool"] == 1
    assert report["discarded"]["dispatch"] == 0
