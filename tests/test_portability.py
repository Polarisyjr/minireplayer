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
    replayer = ledger(
        tmp_path, "run-c", make_bundle(dispatches=[recorded], tools=[]), mode="replay"
    )
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


def test_coral_worktree_outside_evidence_dir_rebinds_to_live_invocation(
    tmp_path: Path,
) -> None:
    source_workspace = tmp_path / "repo/frameworks/CORAL/results/source/agents/agent-3"
    source_workspace.mkdir(parents=True)
    source_payload = {
        "kind": "dispatch",
        "actor_id": "actor-3",
        "session_id": "actor-3/invocation-0/root-0",
        "process_role": "coral-opencode",
        "started_at_ns": 100,
        "parser_identity": "opencode.message.part.tool",
        "dispatcher_identity": "opencode.tool.execute.before",
        "native_call_id": "call-edit",
        "name": "edit",
        "arguments": {"filePath": str(source_workspace / "solution.cpp")},
        "workspace_path": str(source_workspace),
        "origin": {
            "kind": "llm_structured",
            "trigger_id": "llm-0",
            "model_call_id": "call-edit",
        },
    }
    recorder = ledger(tmp_path, "run-source", adapter="coral")
    reservation = recorder.start(source_payload)
    recorder.complete(
        {
            "reservation_id": reservation["reservation_id"],
            "ended_at_ns": 200,
            "status": "executed",
            "execution_call_id": "tool-0",
        }
    )
    import json

    recorded = json.loads((tmp_path / "run-source/stage/dispatches.jsonl").read_text().strip())
    assert recorded["arguments"] == {"filePath": "/native-workspace/solution.cpp"}

    live_workspace = tmp_path / "repo/frameworks/CORAL/results/replay/agents/agent-3"
    live_workspace.mkdir(parents=True)
    replayer = ledger(
        tmp_path,
        "run-replay",
        make_bundle(dispatches=[recorded], tools=[]),
        mode="replay",
        adapter="coral",
    )
    live_payload = {
        **source_payload,
        "arguments": {"filePath": str(live_workspace / "solution.cpp")},
        "workspace_path": str(live_workspace),
    }
    response = replayer.start(live_payload)

    assert response["execution_arguments"] == {"filePath": str(live_workspace / "solution.cpp")}


def test_coral_forced_source_path_is_claimed_then_rebound_to_live_worktree(
    tmp_path: Path,
) -> None:
    source_workspace = tmp_path / "repo/frameworks/CORAL/results/source/agents/agent-2"
    source_workspace.mkdir(parents=True)
    source_payload = {
        "kind": "dispatch",
        "actor_id": "actor-2",
        "session_id": "actor-2/invocation-0/root-0",
        "process_role": "coral-opencode",
        "started_at_ns": 100,
        "parser_identity": "opencode.message.part.tool",
        "dispatcher_identity": "opencode.tool.execute.before",
        "native_call_id": "call-read",
        "name": "read",
        "arguments": {"filePath": str(source_workspace / "statement.txt")},
        "workspace_path": str(source_workspace),
        "origin": {
            "kind": "llm_structured",
            "trigger_id": "llm-0",
            "model_call_id": "call-read",
        },
    }
    recorder = ledger(tmp_path, "run-source", adapter="coral")
    reservation = recorder.start(source_payload)
    recorder.complete(
        {
            "reservation_id": reservation["reservation_id"],
            "ended_at_ns": 200,
            "status": "executed",
            "execution_call_id": None,
        }
    )
    import json

    recorded = json.loads((tmp_path / "run-source/stage/dispatches.jsonl").read_text().strip())
    live_workspace = tmp_path / "repo/frameworks/CORAL/results/replay/agents/agent-2"
    live_workspace.mkdir(parents=True)
    replayer = ledger(
        tmp_path,
        "run-replay",
        make_bundle(dispatches=[recorded], tools=[]),
        mode="replay",
        adapter="coral",
    )
    # Forced decoding faithfully emits the source response, including its absolute
    # path. The selected record proves that exact source path before it is rebound.
    forced_payload = {
        **source_payload,
        "workspace_path": str(live_workspace),
    }
    response = replayer.start(forced_payload)

    assert response["execution_arguments"] == {
        "filePath": str(live_workspace / "statement.txt")
    }


def test_coral_bash_command_rebinds_only_a_typed_proven_source_workspace(
    tmp_path: Path,
) -> None:
    """Support old bundles recorded before native_workspace_path was persisted."""

    actor = "actor-2"
    source_workspace = tmp_path / "repo/frameworks/CORAL/results/source/agents/agent-2"
    live_workspace = tmp_path / "repo/frameworks/CORAL/results/replay/agents/agent-2"
    source_workspace.mkdir(parents=True)
    live_workspace.mkdir(parents=True)

    path_record = dispatch(
        dispatch_id="dispatch-read",
        actor_id=actor,
        name="read",
        arguments={"filePath": "/native-workspace/statement.txt"},
        execution_call_id=None,
        session_id=f"{actor}/invocation-0/root-0",
    )
    path_record["native_arguments"] = {
        "filePath": str(source_workspace / "statement.txt")
    }
    path_record["causal_lane"] = "model-call:call-read"

    source_command = f"cd {source_workspace} && coral eval -m test"
    command_record = dispatch(
        dispatch_id="dispatch-bash",
        actor_id=actor,
        name="bash",
        arguments={"command": source_command},
        execution_call_id="tool-bash",
        session_id=f"{actor}/invocation-0/root-0",
    )
    command_record["native_arguments"] = {"command": source_command}
    command_record["causal_lane"] = "model-call:call-bash"
    command_record["origin"]["model_call_id"] = "call-bash"
    tool_record = tool(
        call_id="tool-bash",
        dispatch_id="dispatch-bash",
        actor_id=actor,
        name="bash",
        causal_lane="model-call:call-bash",
        arguments={"command": source_command},
    )
    tool_record["native_arguments"] = {"command": source_command}

    replayer = ledger(
        tmp_path,
        "run-replay",
        make_bundle(dispatches=[path_record, command_record], tools=[tool_record]),
        mode="replay",
        adapter="coral",
    )
    response = replayer.start(
        {
            "kind": "dispatch",
            "actor_id": actor,
            "session_id": f"{actor}/invocation-0/root-0",
            "process_role": "coral-opencode",
            "started_at_ns": 100,
            "parser_identity": "opencode.message.part.tool",
            "dispatcher_identity": "opencode.tool.execute.before",
            "native_call_id": "call-bash",
            "name": "bash",
            "arguments": {"command": source_command},
            "workspace_path": str(live_workspace),
            "origin": {
                "kind": "llm_structured",
                "trigger_id": "llm-0",
                "model_call_id": "call-bash",
            },
        }
    )

    assert response["execution_arguments"] == {
        "command": f"cd {live_workspace} && coral eval -m test"
    }
    tool_response = replayer.start(
        {
            "kind": "tool",
            "actor_id": actor,
            "process_role": "coral-opencode",
            "started_at_ns": 101,
            "dispatch_id": "dispatch-bash",
            "name": "bash",
            "implementation": "native-shell",
            "arguments": response["execution_arguments"],
            "workspace_path": str(live_workspace),
            "result_contract": tool_record["result_contract"],
        }
    )
    assert tool_response["record_id"] == "tool-bash"


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
