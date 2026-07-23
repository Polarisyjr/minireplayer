"""LLM slot claiming and the request-shape projection."""

from __future__ import annotations

from pathlib import Path

import pytest

from minireplay.errors import MismatchError
from minireplay.llm_store import LLMStore, RequestIdentity, request_shape
from minireplay.util import sha256_json
from tests.support import llm, make_bundle


def store(tmp_path: Path, records: list[dict]) -> LLMStore:
    return LLMStore(
        mode="replay",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
        bundle=make_bundle(llm_records=records, tools=[], dispatches=[]),
    )


def identity(actor: str = "actor-0", session: str = "actor-0", role: str = "agent"):
    return RequestIdentity(
        actor_id=actor, session_id=session, role=role, target_id="vllm-8000", parent_span_id=None
    )


BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_claims_in_recorded_order(tmp_path: Path) -> None:
    records = [llm(attempt_id="llm-0", sequence=0), llm(attempt_id="llm-1", sequence=1)]
    store_ = store(tmp_path, records)
    assert store_._claim(identity(), BODY, "chat.completions")["attempt_id"] == "llm-0"
    assert store_._claim(identity(), BODY, "chat.completions")["attempt_id"] == "llm-1"


def test_rejects_an_extra_request(tmp_path: Path) -> None:
    store_ = store(tmp_path, [llm()])
    store_._claim(identity(), BODY, "chat.completions")
    with pytest.raises(MismatchError, match="unexpected LLM request"):
        store_._claim(identity(), BODY, "chat.completions")


def test_rejects_api_drift(tmp_path: Path) -> None:
    store_ = store(tmp_path, [llm()])
    with pytest.raises(MismatchError, match="API drift"):
        store_._claim(identity(), BODY, "responses")


def test_rejects_a_structurally_different_request(tmp_path: Path) -> None:
    store_ = store(tmp_path, [llm()])
    changed = {"model": "m", "messages": [{"role": "system", "content": "hi"}]}
    with pytest.raises(MismatchError, match="request drift"):
        store_._claim(identity(), changed, "chat.completions")


def test_roles_and_sessions_are_separate_queues(tmp_path: Path) -> None:
    """Two lanes of one actor do not order each other."""

    records = [
        llm(attempt_id="llm-agent", role="agent", sequence=0),
        llm(attempt_id="llm-grader", role="grader", sequence=0),
    ]
    store_ = store(tmp_path, records)
    assert store_._claim(identity(role="grader"), BODY, "chat.completions")["attempt_id"] == (
        "llm-grader"
    )
    assert store_._claim(identity(role="agent"), BODY, "chat.completions")["attempt_id"] == (
        "llm-agent"
    )


def test_shape_tolerates_free_text_but_not_structure() -> None:
    """A prompt legitimately carries this run's paths; its shape must still match."""

    recorded = {"model": "m", "messages": [{"role": "user", "content": "read /run/aaa/f.py"}]}
    same_shape = {"model": "m", "messages": [{"role": "user", "content": "read /run/bbb/f.py"}]}
    different_role = {"model": "m", "messages": [{"role": "system", "content": "read /run/a/f.py"}]}
    different_config = {"model": "other", "messages": recorded["messages"]}

    assert sha256_json(request_shape(recorded)) == sha256_json(request_shape(same_shape))
    assert sha256_json(request_shape(recorded)) != sha256_json(request_shape(different_role))
    assert sha256_json(request_shape(recorded)) != sha256_json(request_shape(different_config))


def test_shape_notices_a_wildly_different_prompt_size() -> None:
    small = {"messages": [{"role": "user", "content": "hi"}]}
    huge = {"messages": [{"role": "user", "content": "x" * 5000}]}
    assert sha256_json(request_shape(small)) != sha256_json(request_shape(huge))


def test_expected_complete_requires_every_attempt(tmp_path: Path) -> None:
    store_ = store(
        tmp_path, [llm(attempt_id="llm-0", sequence=0), llm(attempt_id="llm-1", sequence=1)]
    )
    assert store_.expected_complete() is False
    store_._claim(identity(), BODY, "chat.completions")
    assert store_.expected_complete() is False
    store_._claim(identity(), BODY, "chat.completions")
    assert store_.expected_complete() is True


def test_model_call_index_links_a_dispatch_to_its_attempt(tmp_path: Path) -> None:
    """A dispatch names the provider tool-call ID; this is how it finds its parent."""

    store_ = store(tmp_path, [])
    store_._index_model_calls(
        "llm-7",
        {"choices": [{"message": {"tool_calls": [{"id": "call_abc", "type": "function"}]}}]},
    )
    assert store_.attempt_for_model_call("call_abc") == "llm-7"
    assert store_.attempt_for_model_call("call_unknown") is None


def test_inline_image_payloads_do_not_carry_a_size_class() -> None:
    """A live screenshot's byte length must not decide whether a replay is valid.

    Owl's browser toolkit sends the real viewport to its VL endpoint, so the bytes
    differ every run. The media type is structural and stays; the length is a
    browser internal (design §5) and must not reach the gate.
    """

    def request(pixels: int) -> dict:
        return {
            "model": "Qwen2.5-VL-7B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "act as a web agent"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + "A" * pixels},
                        },
                    ],
                }
            ],
        }

    small, large = request(39_266), request(426_258)
    assert request_shape(small) == request_shape(large)

    # The media type is still part of the shape, and so is everything structural.
    jpeg = request(39_266)
    jpeg["messages"][0]["content"][1]["image_url"]["url"] = "data:image/jpeg;base64,AAAA"
    assert request_shape(jpeg) != request_shape(small)

    dropped = request(39_266)
    dropped["messages"][0]["content"].pop(0)
    assert request_shape(dropped) != request_shape(small)
