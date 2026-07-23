"""A bundle must replay from a different directory than the one that recorded it.

Tool arguments name real directories. If those went into the bundle verbatim, every
replay would either fail its digest compare or run against the recording's
workspace. Both sides reduce to logical names for comparison, and expand back to the
live directories before the call is made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minireplay.boundary import BoundaryLedger
from minireplay.contracts import run_path_map, to_physical
from tests.support import dispatch, make_bundle, tool


def ledger(tmp_path: Path, run: str, bundle=None, *, mode="record", adapter="trae"):
    run_root = tmp_path / run
    run_root.mkdir(parents=True, exist_ok=True)
    stage = run_root / "stage"
    stage.mkdir(exist_ok=True)
    return BoundaryLedger(
        mode=mode,
        stage_dir=stage,
        auth_token="token",
        adapter=adapter,
        run_root=run_root,
        repo=tmp_path / "repo",
        bundle=bundle,
    )


def start(run_root: Path) -> dict:
    """Trae declares `path` as a path field, so it is bound; `command` is free text."""

    return {
        "kind": "tool",
        "actor_id": "actor-0",
        "process_role": "agent",
        "started_at_ns": 100,
        "dispatch_id": "dispatch-0",
        "name": "edit",
        "implementation": "trae.edit",
        "arguments": {"path": str(run_root / "workspace" / "a.py")},
        "result_contract": tool()["result_contract"],
    }


def test_run_directories_reduce_to_logical_names(tmp_path: Path) -> None:
    mapping = run_path_map(tmp_path / "run-a", tmp_path / "repo")
    from minireplay.contracts import bind_typed_fields

    bound = bind_typed_fields(
        "trae", {"path": str(tmp_path / "run-a" / "workspace" / "a.py")}, mapping
    )
    assert bound == {"path": "/native-workspace/a.py"}


def test_logical_names_expand_into_the_live_run(tmp_path: Path) -> None:
    physical = to_physical(
        "trae", {"path": "/native-workspace/a.py"}, tmp_path / "run-b", tmp_path / "repo"
    )
    assert physical == {"path": str((tmp_path / "run-b" / "workspace").resolve() / "a.py")}


def test_a_recording_replays_from_a_different_directory(tmp_path: Path) -> None:
    """Record in run-a, replay in run-b: the claim must still match."""

    recorder = ledger(tmp_path, "run-a")
    reservation = recorder.start(start(tmp_path / "run-a"))
    recorder.complete(
        {
            "reservation_id": reservation["reservation_id"],
            "ended_at_ns": 300,
            "status": "ok",
            "result": {"output": "ok"},
            "native_execution": True,
        }
    )
    recorded = [
        line
        for line in (tmp_path / "run-a" / "stage" / "tools.jsonl").read_text().splitlines()
        if line
    ]
    assert len(recorded) == 1
    import json

    record = json.loads(recorded[0])
    # The bundle stores the logical form as identity, and the real one as evidence.
    assert record["arguments"] == {"path": "/native-workspace/a.py"}
    assert record["native_arguments"]["path"].endswith("run-a/workspace/a.py")

    record["call_id"] = "tool-0"
    record["dispatch_id"] = "dispatch-0"
    replayer = ledger(
        tmp_path,
        "run-b",
        make_bundle(tools=[record], dispatches=[dispatch()]),
        mode="replay",
    )
    # Same logical call, different directory: this must claim cleanly.
    claimed = replayer.start(start(tmp_path / "run-b"))
    assert claimed["record_id"] == "tool-0"


def test_dispatch_arguments_come_back_bound_to_this_run(tmp_path: Path) -> None:
    recorded = dispatch(
        dispatch_id="dispatch-0",
        name="edit",
        arguments={"path": "/native-workspace/a.py"},
        execution_call_id=None,
    )
    replayer = ledger(tmp_path, "run-c", make_bundle(dispatches=[recorded], tools=[]),
                      mode="replay")
    response = replayer.start(
        {
            "kind": "dispatch",
            "actor_id": "actor-0",
            "process_role": "agent",
            "started_at_ns": 100,
            "parser_identity": "parser",
            "dispatcher_identity": "dispatcher",
            "native_call_id": "dispatch-0",
            "name": "edit",
            "arguments": {"path": str(tmp_path / "run-c" / "workspace" / "a.py")},
            "origin": {"kind": "llm_structured", "trigger_id": "llm-0"},
        }
    )
    expected = str((tmp_path / "run-c" / "workspace").resolve() / "a.py")
    assert response["execution_arguments"] == {"path": expected}


def test_a_patch_factory_that_returns_nothing_fails_by_name() -> None:
    """The wrapper is what gets installed, so losing it must not fail obscurely.

    Without this guard the omission surfaced several frames later inside functools
    as `'NoneType' object has no attribute '__module__'`, naming neither the adapter
    nor the patch point, and only when the framework actually started.
    """

    from minireplay.instrumentation.patching import patch_method

    class Owner:
        def method(self):
            return None

    with pytest.raises(RuntimeError) as failure:
        patch_method(Owner, "method", lambda original: None)

    assert "must return the wrapper" in str(failure.value)
    assert "method" in str(failure.value)
