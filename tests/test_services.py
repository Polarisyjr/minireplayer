"""Cross-ledger completion rules for concurrent replay prefixes."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from minireplay.errors import MismatchError, WorkloadComplete
from minireplay.llm_store import RequestIdentity
from minireplay.services import ReplayServices
from tests.support import dispatch, llm, make_bundle, tool

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def services(tmp_path: Path, bundle) -> ReplayServices:
    return ReplayServices(
        mode="replay",
        stage_dir=tmp_path,
        auth_token="token",
        adapter="mini-swe",
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
        run_root=tmp_path,
        repo=tmp_path,
        bundle=bundle,
    )


def start_tool(actor_id: str, call: str) -> dict:
    return {
        "kind": "tool",
        "actor_id": actor_id,
        "process_role": "agent",
        "started_at_ns": 100,
        "dispatch_id": f"dispatch-{call}",
        "name": "shell",
        "implementation": "native-shell",
        "arguments": {"command": call},
        "result_contract": tool()["result_contract"],
    }


def start_dispatch(actor_id: str, call: str) -> dict:
    return {
        "kind": "dispatch",
        "actor_id": actor_id,
        "process_role": "agent",
        "started_at_ns": 90,
        "parser_identity": "parser",
        "dispatcher_identity": "dispatcher",
        "native_call_id": f"dispatch-{call}",
        "name": "shell",
        "arguments": {"command": call},
        "origin": {"kind": "llm_structured", "trigger_id": f"llm-{actor_id}"},
    }


def complete_operation(service: ReplayServices, reservation: dict) -> None:
    service.boundary.complete(
        {
            "reservation_id": reservation["reservation_id"],
            "ended_at_ns": 300,
            "status": "ok",
            "result": {"output": "ok", "exit_code": 0},
            "native_execution": True,
        }
    )


def test_source_live_actor_can_stop_at_its_own_cutoff(tmp_path: Path) -> None:
    records = [
        tool(call_id="tool-a", dispatch_id="dispatch-a", actor_id="a", arguments={"command": "a"}),
        tool(call_id="tool-b", dispatch_id="dispatch-b", actor_id="b", arguments={"command": "b"}),
    ]
    bundle = make_bundle(
        actors=["a", "b"],
        tools=records,
        dispatches=[
            dispatch(
                dispatch_id="dispatch-a",
                actor_id="a",
                execution_call_id="tool-a",
                arguments={"command": "a"},
            ),
            dispatch(
                dispatch_id="dispatch-b",
                actor_id="b",
                execution_call_id="tool-b",
                arguments={"command": "b"},
            ),
        ],
        llm_records=[llm(attempt_id="llm-a", actor_id="a", session_id="a")],
    )
    service = services(tmp_path, bundle)

    for operation in (start_dispatch("a", "a"), start_tool("a", "a")):
        reservation = service.boundary.start(operation)
        complete_operation(service, reservation)
    service.llm._claim(
        RequestIdentity("a", "a", "agent", "vllm-8000", None),
        BODY,
        "chat.completions",
    )

    assert service.expected_complete() is False  # actor b still has native work
    assert service.cutoff_actor_complete("a") is False
    assert service.cutoff_actor_prefix_consumed("a") is True
    with pytest.raises(WorkloadComplete):
        service.boundary.start(start_tool("a", "extra"))


def test_terminal_actor_does_not_hide_an_extra_operation(tmp_path: Path) -> None:
    bundle = make_bundle(
        actors=["done", "pending"],
        tools=[tool(call_id="tool-pending", actor_id="pending")],
        dispatches=[
            dispatch(
                dispatch_id="dispatch-pending",
                actor_id="pending",
                execution_call_id="tool-pending",
            )
        ],
    )
    bundle.terminal["task_terminals"] = [{"actor_id": "done", "status": "success"}]
    service = services(tmp_path, bundle)

    assert service.cutoff_actor_complete("done") is False
    with pytest.raises(MismatchError, match="unexpected native tool"):
        service.boundary.start(start_tool("done", "extra"))


def test_refill_lane_remains_a_cutoff_actor_after_initial_task_terminal(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(actors=["lane-0"])
    bundle.manifest["actors"] = [
        {
            "actor_id": "lane-0",
            "source_actor_id": "initial",
            "source_actor_ids": ["initial", "refill"],
        }
    ]
    bundle.terminal["task_terminals"] = [
        {
            "actor_id": "lane-0",
            "status": "success",
            "task": {"source_actor_id": "initial"},
        }
    ]
    service = services(tmp_path, bundle)

    assert service.cutoff_actor_prefix_consumed("lane-0") is True
    with pytest.raises(WorkloadComplete):
        service.boundary.start(start_tool("lane-0", "extra"))


def test_cutoff_dispatch_is_stopped_before_native_entry(tmp_path: Path) -> None:

    tail = dispatch(
        dispatch_id="dispatch-tail",
        actor_id="a",
        execution_call_id=None,
        arguments={"command": "tail"},
    )
    tail.update(
        {
            "cutoff_truncated": True,
            "kind": "dispatch",
            "record_id": "dispatch-tail",
            "source_started_at_ns": 300,
            "elapsed_ns": 10_000_000_000,
        }
    )
    bundle = make_bundle(
        actors=["a"],
        tools=[],
        dispatches=[],
        cutoff_tails={"operations": [tail], "llm_requests": []},
    )
    service = services(tmp_path, bundle)

    assert service.expected_complete() is True
    assert service.cutoff_actor_prefix_consumed("a") is True
    with pytest.raises(WorkloadComplete):
        service.boundary.start(start_dispatch("a", "tail"))


def test_cutoff_snapshot_runs_on_each_ledger_service_loop(tmp_path: Path) -> None:
    service = services(tmp_path, make_bundle(tools=[], dispatches=[]))
    seen: dict[str, int] = {}

    def freeze_boundary(_cutoff: int) -> list:
        seen["boundary"] = threading.get_ident()
        return []

    def freeze_llm(_cutoff: int) -> list:
        seen["llm"] = threading.get_ident()
        return []

    service.boundary.freeze_source_cutoff = freeze_boundary
    service.llm.freeze_source_cutoff = freeze_llm
    service.start()
    try:
        assert service.freeze_source_cutoff(100) == {"operations": [], "llm_requests": []}
        boundary_thread = service._threads["boundary"]
        llm_thread = service._threads["llm"]
        assert boundary_thread.ident != llm_thread.ident
        assert seen == {"boundary": boundary_thread.ident, "llm": llm_thread.ident}
    finally:
        service.stop()
