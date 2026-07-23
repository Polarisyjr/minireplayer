"""Cutoff tails are recorded as diagnostics but excluded from replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from minireplay.boundary import BoundaryLedger
from minireplay.errors import WorkloadComplete
from minireplay.llm_store import LLMStore
from minireplay.util import sha256_json
from tests.support import llm, make_bundle
from tests.test_llm_store import BODY, identity


def ledger(tmp_path: Path, bundle) -> BoundaryLedger:
    return BoundaryLedger(
        mode="replay",
        stage_dir=tmp_path,
        auth_token="token",
        adapter="mini-swe",
        run_root=tmp_path,
        repo=tmp_path,
        bundle=bundle,
    )


def operation_tail() -> dict:
    return {
        "cutoff_truncated": True,
        "kind": "tool",
        "call_id": "tool-tail",
        "record_id": "tool-tail",
        "span_id": "span-tool-tail",
        "actor_id": "actor-0",
        "lane": None,
        "name": "shell",
        "implementation": "native-shell",
        "arguments": {"command": "sleep 30"},
        "arguments_sha256": sha256_json({"command": "sleep 30"}),
        "source_started_at_ns": 100,
        "elapsed_ns": 5_000_000_000,
    }


def test_operation_tail_is_diagnostic_not_claimable(tmp_path: Path) -> None:
    bundle = make_bundle(
        dispatches=[],
        tools=[],
        cutoff_tails={"operations": [operation_tail()], "llm_requests": []},
    )
    service = ledger(tmp_path, bundle)

    assert service.expected_complete() is True
    assert service.outstanding()["truncated"]["tails"] == []


def test_llm_tail_is_diagnostic_not_claimable(tmp_path: Path) -> None:
    tail = dict(llm(attempt_id="llm-tail", sequence=1))
    tail.update({"cutoff_truncated": True, "elapsed_ns": 5_000_000_000})
    store = LLMStore(
        mode="replay",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
        bundle=make_bundle(
            dispatches=[],
            tools=[],
            llm_records=[llm(attempt_id="llm-0", sequence=0)],
            cutoff_tails={"operations": [], "llm_requests": [tail]},
        ),
    )

    expected = store._claim(identity(), BODY, "chat.completions")
    assert expected["attempt_id"] == "llm-0"
    assert store.expected_complete() is False
    store._write_replay_attempt(expected, identity(), 100, 200)
    store._delivered.add("llm-0")
    assert store.expected_complete() is True
    with pytest.raises(WorkloadComplete, match="recorded window closed"):
        store._claim(identity(), BODY, "chat.completions")


def test_recording_captures_an_llm_request_left_in_flight(tmp_path: Path) -> None:
    store = LLMStore(
        mode="record",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
    )
    store._inflight["llm-live"] = {
        "cutoff_truncated": True,
        "attempt_id": "llm-live",
        "span_id": "span-llm-live",
        "actor_id": "actor-0",
        "session_id": "actor-0",
        "role": "agent",
        "target_id": "vllm-8000",
        "api": "chat.completions",
        "sequence": 3,
        "request": BODY,
        "request_sha256": sha256_json(BODY),
        "request_shape_sha256": "0" * 64,
        "started_at_ns": 1_000_000_000,
    }

    tails = store.freeze_source_cutoff(4_000_000_000)

    assert len(tails) == 1
    assert tails[0]["attempt_id"] == "llm-live"
    assert tails[0]["elapsed_ns"] == 3_000_000_000
