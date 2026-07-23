"""Slot claiming: ordering, drift, concurrency and observation replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from minireplay.boundary import BoundaryLedger
from minireplay.errors import MismatchError
from tests.support import dispatch, make_bundle, tool


def ledger(tmp_path: Path, bundle, *, adapter: str = "mini-swe", **kwargs) -> BoundaryLedger:
    return BoundaryLedger(
        mode="replay",
        stage_dir=tmp_path,
        auth_token="token",
        adapter=adapter,
        run_root=tmp_path,
        repo=tmp_path,
        bundle=bundle,
        **kwargs,
    )


def start_tool(actor_id: str = "actor-0", **overrides) -> dict:
    payload = {
        "kind": "tool",
        "actor_id": actor_id,
        "process_role": "agent",
        "started_at_ns": 100,
        "dispatch_id": "dispatch-0",
        "name": "shell",
        "implementation": "native-shell",
        "arguments": {"command": "echo hi"},
        "result_contract": tool()["result_contract"],
    }
    payload.update(overrides)
    return payload


def test_claim_rejects_argument_drift(tmp_path: Path) -> None:
    led = ledger(tmp_path, make_bundle())
    with pytest.raises(MismatchError, match="invocation drift"):
        led.start(start_tool(arguments={"command": "rm -rf /"}))


def test_claim_rejects_reorder_within_one_actor(tmp_path: Path) -> None:
    """The second recorded call exists, but claiming it out of turn is still drift."""

    first = tool(call_id="tool-0", arguments={"command": "first"})
    second = tool(call_id="tool-1", dispatch_id="dispatch-1", arguments={"command": "second"})
    led = ledger(tmp_path, make_bundle(tools=[first, second]))
    with pytest.raises(MismatchError, match="invocation drift"):
        led.start(start_tool(arguments={"command": "second"}))


def test_claim_is_independent_across_actors(tmp_path: Path) -> None:
    """Two actors never compete for a slot, so completion order cannot matter."""

    a = tool(call_id="tool-a", actor_id="actor-a", arguments={"command": "a"})
    b = tool(call_id="tool-b", actor_id="actor-b", arguments={"command": "b"})
    bundle = make_bundle(
        tools=[a, b],
        dispatches=[
            dispatch(dispatch_id="dispatch-a", actor_id="actor-a", execution_call_id="tool-a"),
            dispatch(dispatch_id="dispatch-b", actor_id="actor-b", execution_call_id="tool-b"),
        ],
        actors=["actor-a", "actor-b"],
    )
    led = ledger(tmp_path, bundle)

    # Claim the second actor first: interleaving across actors is natural concurrency.
    assert led.start(start_tool("actor-b", arguments={"command": "b"}))["record_id"] == "tool-b"
    assert led.start(start_tool("actor-a", arguments={"command": "a"}))["record_id"] == "tool-a"


def test_coral_sessions_are_separate_lanes(tmp_path: Path) -> None:
    """One CORAL actor runs concurrent sessions; they must not order each other."""

    root = dispatch(dispatch_id="d-root", session_id="s/root", arguments={"n": "root"},
                    execution_call_id=None)
    child = dispatch(dispatch_id="d-child", session_id="s/root/child", arguments={"n": "child"},
                     execution_call_id=None)
    led = ledger(tmp_path, make_bundle(dispatches=[child, root], tools=[]), adapter="coral")

    def start_dispatch(session: str, name: str) -> dict:
        return {
            "kind": "dispatch",
            "actor_id": "actor-0",
            "session_id": session,
            "process_role": "coral-opencode",
            "started_at_ns": 100,
            "parser_identity": "parser",
            "dispatcher_identity": "dispatcher",
            "native_call_id": name,
            "name": "shell",
            "arguments": {"n": name},
            "origin": {"kind": "llm_structured", "trigger_id": "llm-0"},
        }

    # Recorded order is [child, root]; claiming root first is fine, different lane.
    assert led.start(start_dispatch("s/root", "root"))["record_id"] == "d-root"
    assert led.start(start_dispatch("s/root/child", "child"))["record_id"] == "d-child"


def test_fast_claim_skips_the_digest_compare(tmp_path: Path) -> None:
    led = ledger(tmp_path, make_bundle(), fast_claim=True)
    assert led.start(start_tool(arguments={"command": "anything"}))["record_id"] == "tool-0"


def test_completion_returns_the_recorded_observation(tmp_path: Path) -> None:
    """The native tool ran; the framework still sees what the recording saw."""

    recorded = tool(result={"output": "recorded", "exit_code": 0})
    led = ledger(tmp_path, make_bundle(tools=[recorded]))
    reservation = led.start(start_tool())
    completion = led.complete(
        {
            "reservation_id": reservation["reservation_id"],
            "ended_at_ns": 300,
            "status": "ok",
            "result": {"output": "this run produced something else", "exit_code": 0},
            "native_execution": True,
        }
    )
    assert completion["result_replay_required"] is True
    assert completion["framework_result"] == {"output": "recorded", "exit_code": 0}


def test_completion_restores_a_recorded_exception(tmp_path: Path) -> None:
    recorded = tool(
        result={"error_type": "TimeoutError", "message": "timed out"},
        status="error",
        exception_raised=True,
    )
    led = ledger(tmp_path, make_bundle(tools=[recorded]))
    reservation = led.start(start_tool())
    completion = led.complete(
        {
            "reservation_id": reservation["reservation_id"],
            "ended_at_ns": 300,
            "status": "ok",  # this run happened to succeed
            "result": {"output": "fine"},
            "native_execution": True,
        }
    )
    assert completion["framework_exception"] == {
        "error_type": "TimeoutError",
        "message": "timed out",
    }


def test_native_result_is_kept_alongside_the_replayed_one(tmp_path: Path) -> None:
    """Evidence must stay honest about which value was real and which was replayed."""

    led = ledger(tmp_path, make_bundle(tools=[tool(result={"output": "recorded"})]))
    reservation = led.start(start_tool())
    led.complete(
        {
            "reservation_id": reservation["reservation_id"],
            "ended_at_ns": 300,
            "status": "ok",
            "result": {"output": "actual"},
            "native_execution": True,
        }
    )
    written = (tmp_path / "tools.jsonl").read_text()
    assert '"result":{"output":"recorded"}' in written.replace(" ", "")
    assert '"native_result":{"output":"actual"}' in written.replace(" ", "")


def test_unexpected_operation_is_rejected(tmp_path: Path) -> None:
    led = ledger(tmp_path, make_bundle(tools=[]))
    with pytest.raises(MismatchError, match="unexpected native tool"):
        led.start(start_tool())


def test_missing_operations_are_reported(tmp_path: Path) -> None:
    led = ledger(tmp_path, make_bundle())
    with pytest.raises(MismatchError, match="missing native operations"):
        led.assert_consumed()


def test_expected_complete_ignores_diagnostic_tail(tmp_path: Path) -> None:
    bundle = make_bundle(
        tools=[],
        dispatches=[],
        cutoff_tails={
            "operations": [
                {
                    "cutoff_truncated": True,
                    "kind": "dispatch",
                    "record_id": "dispatch-0",
                    "actor_id": "actor-0",
                    "elapsed_ns": 10_000_000_000,
                    "source_started_at_ns": 300,
                }
            ],
            "llm_requests": [],
        },
    )
    led = ledger(tmp_path, bundle)
    assert led.expected_complete() is True
