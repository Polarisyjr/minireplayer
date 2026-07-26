"""A killed CORAL task keeps completed subagent work as fixed replay slots."""

from __future__ import annotations

from pathlib import Path

from minireplay.boundary import BoundaryLedger
from minireplay.bundle import build_bundle
from minireplay.constants import TERMINAL_SCHEMA
from minireplay.cutoff import close_stage_at_cutoff
from minireplay.llm_store import LLMStore, RequestIdentity
from minireplay.util import append_jsonl, sha256_json
from tests.support import dispatch, llm, span, tool


def test_partial_task_preserves_closed_child_and_reenters_parent(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    for relative in (
        "llm.jsonl",
        "spans.jsonl",
        "dispatches.jsonl",
        "tools.jsonl",
        "graders.jsonl",
        "artifacts.jsonl",
    ):
        (stage / relative).touch()

    actor = "actor-0"
    parent_arguments = {"description": "research", "subagent_type": "general"}
    parent_lane = "model-call:call-task"
    parent_dispatch = {
        **dispatch(
            dispatch_id="dispatch-task",
            actor_id=actor,
            name="task",
            arguments=parent_arguments,
            execution_call_id=None,
            session_id="root",
        ),
        "cutoff_truncated": True,
        "kind": "dispatch",
        "record_id": "dispatch-task",
        "native_call_id": "call-task",
        "causal_lane": parent_lane,
        "source_started_at_ns": 30,
        "elapsed_ns": 50,
        "replay_entry": "enter-and-preserve-descendants",
    }
    parent_dispatch["origin"]["model_call_id"] = "call-task"
    parent_tool = {
        **tool(
            call_id="tool-task",
            dispatch_id="dispatch-task",
            actor_id=actor,
            name="task",
            causal_lane=parent_lane,
            arguments=parent_arguments,
        ),
        "cutoff_truncated": True,
        "kind": "tool",
        "record_id": "tool-task",
        "source_started_at_ns": 31,
        "elapsed_ns": 49,
        "replay_entry": "enter-and-preserve-descendants",
        "parent_span_id": parent_dispatch["span_id"],
    }

    child = llm(
        attempt_id="llm-child-complete",
        actor_id=actor,
        session_id="root/child-0",
        role="coral-subagent",
    )
    child_dispatch = dispatch(
        dispatch_id="dispatch-child",
        actor_id=actor,
        name="read",
        arguments={"filePath": "README.md"},
        execution_call_id="tool-child",
        session_id="root/child-0",
    )
    child_dispatch["causal_lane"] = "model-call:call-child"
    child_tool = tool(
        call_id="tool-child",
        dispatch_id="dispatch-child",
        actor_id=actor,
        name="read",
        causal_lane="model-call:call-child",
        arguments={"filePath": "README.md"},
    )
    child_spans = [
        {
            **span(child["span_id"], parent=parent_tool["span_id"], actor_id=actor),
            "kind": "llm",
            "name": "llm:coral-subagent",
        },
        {
            **span(
                child_dispatch["span_id"],
                parent=parent_tool["span_id"],
                actor_id=actor,
            ),
            "kind": "dispatch",
            "name": "read",
        },
        {
            **span(
                child_tool["span_id"],
                parent=child_dispatch["span_id"],
                actor_id=actor,
            ),
            "kind": "tool",
            "name": "read",
        },
    ]
    for relative, records in (
        ("llm.jsonl", [child]),
        ("dispatches.jsonl", [child_dispatch]),
        ("tools.jsonl", [child_tool]),
        ("spans.jsonl", child_spans),
    ):
        for record in records:
            append_jsonl(stage / relative, record)

    unfinished_child = {
        "cutoff_truncated": True,
        "attempt_id": "llm-child-unfinished",
        "span_id": "span-llm-child-unfinished",
        "parent_span_id": parent_tool["span_id"],
        "actor_id": actor,
        "session_id": "root/child-0",
        "role": "coral-subagent",
        "target_id": "vllm-8000",
        "api": "chat.completions",
        "sequence": 1,
        "request": {"model": "m", "messages": []},
        "request_sha256": sha256_json({"model": "m", "messages": []}),
        "request_shape_sha256": "0" * 64,
        "started_at_ns": 70,
        "elapsed_ns": 10,
    }
    cutoff_tails = {
        "operations": [parent_dispatch, parent_tool],
        "llm_requests": [unfinished_child],
    }
    close_stage_at_cutoff(stage, cutoff_tails=cutoff_tails)

    bundle = build_bundle(
        stage_dir=stage,
        output=tmp_path / "bundle",
        bundle_id="partial-task",
        adapter="coral",
        workload={"framework": "coral", "concurrency": 1, "duration_s": 180},
        actors=[{"actor_id": actor, "source_actor_id": "task/agent-1"}],
        window={"gate_at_ns": 0, "terminal_at_ns": 100},
        terminal={
            "schema_version": TERMINAL_SCHEMA,
            "status": "success",
            "task_terminals": [],
        },
        cutoff_tails=cutoff_tails,
    )

    assert [record["attempt_id"] for record in bundle.llm] == ["llm-child-complete"]
    assert [record["dispatch_id"] for record in bundle.dispatches] == ["dispatch-child"]
    assert [record["call_id"] for record in bundle.tools] == ["tool-child"]
    assert bundle.cutoff_tails["llm_requests"] == [unfinished_child]

    boundary = BoundaryLedger(
        mode="replay",
        stage_dir=tmp_path / "replay-stage",
        auth_token="token",
        adapter="coral",
        run_root=tmp_path,
        repo=tmp_path,
        bundle=bundle,
    )
    parent_reservation = boundary.start(
        {
            "kind": "dispatch",
            "actor_id": actor,
            "session_id": "root",
            "process_role": "coral-opencode",
            "started_at_ns": 200,
            "parser_identity": "parser",
            "dispatcher_identity": "dispatcher",
            "native_call_id": "call-task",
            "name": "task",
            "arguments": parent_arguments,
            "origin": {
                "kind": "llm_structured",
                "trigger_id": "llm-parent",
                "model_call_id": "call-task",
            },
        }
    )
    assert parent_reservation["record_id"] == "dispatch-task"

    llm_store = LLMStore(
        mode="replay",
        stage_dir=tmp_path / "llm-replay-stage",
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
        bundle=bundle,
    )
    claimed_child = llm_store._claim(  # noqa: SLF001 - integration boundary probe
        RequestIdentity(
            actor_id=actor,
            session_id="root/child-0",
            role="coral-subagent",
            target_id="vllm-8000",
            parent_span_id=parent_tool["span_id"],
        ),
        child["request"],
        "chat.completions",
    )
    assert claimed_child["attempt_id"] == "llm-child-complete"
