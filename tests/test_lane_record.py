"""Record-side lane logs and their post-window materialization."""

from __future__ import annotations

from pathlib import Path

from minireplay.lane_record import (
    composite_scope_rows,
    local_complete,
    local_composite_scope_complete,
    local_composite_scope_start,
    local_start,
    materialize_lane_recording,
)
from minireplay.observation import exact_result_contract
from minireplay.util import iter_jsonl


def _complete(
    root: Path,
    opened: dict,
    *,
    kind: str,
    actor_id: str,
    ended_at_ns: int,
    **fields,
) -> None:
    local_complete(
        root=root,
        reservation=(kind, opened["record_id"], actor_id),
        payload={
            "reservation_id": opened["reservation_id"],
            "ended_at_ns": ended_at_ns,
            **fields,
        },
    )


def test_lane_events_materialize_closed_prefix_and_cutoff_tail(tmp_path: Path) -> None:
    events = tmp_path / "lane-events"
    stage = tmp_path / "stage"
    stage.mkdir()
    actor = "actor-0"

    dispatch = local_start(
        root=events,
        payload={
            "kind": "dispatch",
            "actor_id": actor,
            "session_id": actor,
            "process_role": "agent",
            "parent_span_id": None,
            "started_at_ns": 100,
            "parser_identity": "parser",
            "dispatcher_identity": "dispatcher",
            "native_call_id": "native-0",
            "name": "shell",
            "arguments": {"command": "echo hi"},
            "origin": {"kind": "llm_structured", "trigger_id": "llm-0"},
        },
    )
    tool = local_start(
        root=events,
        payload={
            "kind": "tool",
            "actor_id": actor,
            "session_id": actor,
            "process_role": "agent",
            "parent_span_id": dispatch["span_id"],
            "started_at_ns": 110,
            "dispatch_id": dispatch["record_id"],
            "name": "shell",
            "implementation": "native-shell",
            "arguments": {"command": "echo hi"},
            "result_contract": exact_result_contract(),
            "semantic_timeout_s": None,
        },
    )
    _complete(
        events,
        tool,
        kind="tool",
        actor_id=actor,
        ended_at_ns=190,
        status="ok",
        result={"output": "hi"},
        native_execution=True,
    )
    _complete(
        events,
        dispatch,
        kind="dispatch",
        actor_id=actor,
        ended_at_ns=200,
        status="executed",
        execution_call_id=tool["record_id"],
    )
    active = local_start(
        root=events,
        payload={
            "kind": "grader",
            "actor_id": actor,
            "session_id": actor,
            "process_role": "agent",
            "parent_span_id": None,
            "started_at_ns": 250,
            "implementation": "native-grader",
            "grader_kind": "test",
            "trigger_id": tool["record_id"],
        },
    )

    tails = materialize_lane_recording(
        event_dir=events,
        stage_dir=stage,
        cutoff_at_ns=300,
        auth_token="unused",
        adapter="mini-swe",
        run_root=tmp_path,
        repo=tmp_path,
    )

    dispatches = list(iter_jsonl(stage / "dispatches.jsonl"))
    tools = list(iter_jsonl(stage / "tools.jsonl"))
    spans = list(iter_jsonl(stage / "spans.jsonl"))
    assert dispatches[0]["dispatch_id"] == dispatch["record_id"]
    assert dispatches[0]["execution_call_id"] == tool["record_id"]
    assert tools[0]["call_id"] == tool["record_id"]
    assert dispatches[0]["causal_lane"] == "session:actor-0|trigger:llm-0"
    assert tools[0]["causal_lane"] == dispatches[0]["causal_lane"]
    assert {span["span_id"] for span in spans} == {dispatch["span_id"], tool["span_id"]}
    assert len(tails) == 1
    assert tails[0]["kind"] == "grader"
    assert tails[0]["record_id"] == active["record_id"]
    assert tails[0]["elapsed_ns"] == 50


def test_different_actors_write_different_lane_files(tmp_path: Path) -> None:
    for actor in ("actor-a", "actor-b"):
        local_start(
            root=tmp_path,
            payload={
                "kind": "grader",
                "actor_id": actor,
                "session_id": actor,
                "process_role": "agent",
                "parent_span_id": None,
                "started_at_ns": 1,
                "implementation": "native-grader",
                "grader_kind": "test",
                "trigger_id": "llm-0",
            },
        )

    assert len(list(tmp_path.glob("lane-*.jsonl"))) == 2


def test_composite_scope_is_diagnostic_only(tmp_path: Path) -> None:
    events = tmp_path / "lane-events"
    stage = tmp_path / "stage"
    stage.mkdir()
    actor = "actor-0"
    scope_id = local_composite_scope_start(
        root=events,
        actor_id=actor,
        session_id=actor,
        name="browse_url",
        causal_lane="model-call:browse-0",
        started_at_ns=100,
    )
    local_composite_scope_complete(
        root=events,
        actor_id=actor,
        session_id=actor,
        scope_id=scope_id,
        ended_at_ns=250,
        status="ok",
    )

    tails = materialize_lane_recording(
        event_dir=events,
        stage_dir=stage,
        cutoff_at_ns=300,
        auth_token="unused",
        adapter="owl",
        run_root=tmp_path,
        repo=tmp_path,
    )

    assert tails == []
    assert not (stage / "dispatches.jsonl").exists()
    assert not (stage / "tools.jsonl").exists()
    assert composite_scope_rows(events, 300) == [
        {
            "scope_id": scope_id,
            "actor_id": actor,
            "session_id": actor,
            "name": "browse_url",
            "causal_lane": "model-call:browse-0",
            "started_at_ns": 100,
            "ended_at_ns": 250,
            "status": "ok",
            "cutoff_truncated": False,
        }
    ]


def test_open_composite_scope_is_not_a_cutoff_tail(tmp_path: Path) -> None:
    events = tmp_path / "lane-events"
    stage = tmp_path / "stage"
    stage.mkdir()
    scope_id = local_composite_scope_start(
        root=events,
        actor_id="actor-0",
        session_id="actor-0",
        name="browse_url",
        causal_lane="model-call:browse-0",
        started_at_ns=100,
    )

    tails = materialize_lane_recording(
        event_dir=events,
        stage_dir=stage,
        cutoff_at_ns=300,
        auth_token="unused",
        adapter="owl",
        run_root=tmp_path,
        repo=tmp_path,
    )

    assert tails == []
    assert composite_scope_rows(events, 300)[0] == {
        "scope_id": scope_id,
        "actor_id": "actor-0",
        "session_id": "actor-0",
        "name": "browse_url",
        "causal_lane": "model-call:browse-0",
        "started_at_ns": 100,
        "ended_at_ns": 300,
        "status": "truncated",
        "cutoff_truncated": True,
    }
