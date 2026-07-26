"""LLM slot claiming and the request-shape projection."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from minireplay import llm_store
from minireplay.errors import MismatchError, WorkloadComplete
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


def test_holds_a_completed_lane_while_a_sibling_lane_finishes(tmp_path: Path) -> None:
    records = [
        llm(attempt_id="llm-agent", role="agent"),
        llm(attempt_id="llm-grader", role="grader"),
    ]
    store_ = store(tmp_path, records)
    agent = store_._claim(identity(role="agent"), BODY, "chat.completions")
    store_._write_replay_attempt(agent, identity(role="agent"), 100, 200)
    store_._delivered.add("llm-agent")

    assert store_.expected_complete() is False
    with pytest.raises(WorkloadComplete, match="recorded window closed"):
        store_._claim(identity(role="agent"), BODY, "chat.completions")


def test_rejects_api_drift(tmp_path: Path) -> None:
    store_ = store(tmp_path, [llm()])
    with pytest.raises(MismatchError, match="API drift"):
        store_._claim(identity(), BODY, "responses")


def test_rejects_a_structurally_different_request(tmp_path: Path) -> None:
    store_ = store(tmp_path, [llm()])
    changed = {"model": "m", "messages": [{"role": "system", "content": "hi"}]}
    with pytest.raises(
        MismatchError,
        match=r"first difference: \$\.payload\.messages\[0\]\.role",
    ):
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
    first = store_._claim(identity(), BODY, "chat.completions")
    assert store_.expected_complete() is False
    second = store_._claim(identity(), BODY, "chat.completions")
    # Claiming every queue slot is not completion: the engine/audit and the
    # framework-visible HTTP response can still be in flight.
    assert store_.expected_complete() is False
    store_._write_replay_attempt(first, identity(), 100, 200)
    store_._delivered.add("llm-0")
    assert store_.expected_complete() is False
    store_._write_replay_attempt(second, identity(), 200, 300)
    assert store_.expected_complete() is False
    store_._delivered.add("llm-1")
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


def test_stream_indexes_tool_call_before_framework_can_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    store_ = LLMStore(
        mode="record",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
    )
    attempt_id = "llm-stream"
    payload = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"id": "call_stream", "type": "function", "function": {"name": "read"}}
                    ]
                }
            }
        ]
    }
    wire = (f"data: {json.dumps(payload, separators=(',', ':'))}\n\ndata: [DONE]\n\n").encode()

    class Content:
        async def iter_any(self):
            yield wire

    class Upstream:
        status = 200
        content = Content()

    class Response:
        async def prepare(self, _request):
            pass

        async def write(self, data):
            if b"call_stream" in data:
                assert store_.attempt_for_model_call("call_stream") == attempt_id

        async def write_eof(self):
            pass

    monkeypatch.setattr(
        llm_store.web,
        "StreamResponse",
        lambda **_kwargs: Response(),
    )

    async def capture_engine(observed_attempt_id, committed):
        assert observed_attempt_id == attempt_id
        assert committed == []
        return {"capture": "complete"}

    monkeypatch.setattr(store_, "_capture_engine", capture_engine)
    asyncio.run(
        store_._record_stream(
            object(),
            Upstream(),
            identity(),
            BODY,
            "chat.completions",
            attempt_id,
            0,
            100,
        )
    )
    written = json.loads((tmp_path / "llm.jsonl").read_text().strip())
    assert written["engine"] == {"capture": "complete"}


def test_completed_record_stream_tolerates_client_close_before_http_eof(
    tmp_path: Path, monkeypatch
) -> None:
    store_ = LLMStore(
        mode="record",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
    )

    class Content:
        async def iter_any(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    class Upstream:
        status = 200
        content = Content()

    class Response:
        async def prepare(self, _request):
            pass

        async def write(self, _data):
            pass

        async def write_eof(self):
            raise ConnectionResetError("Cannot write to closing transport")

    monkeypatch.setattr(llm_store.web, "StreamResponse", lambda **_kwargs: Response())

    async def capture_engine(_attempt_id, _committed):
        return {"capture": "complete"}

    monkeypatch.setattr(store_, "_capture_engine", capture_engine)
    asyncio.run(
        store_._record_stream(
            object(),
            Upstream(),
            identity(),
            BODY,
            "chat.completions",
            "llm-complete-reset",
            0,
            100,
        )
    )

    written = json.loads((tmp_path / "llm.jsonl").read_text().strip())
    assert written["attempt_id"] == "llm-complete-reset"


def test_terminal_payload_tolerates_client_close_before_done_marker(
    tmp_path: Path, monkeypatch
) -> None:
    store_ = LLMStore(
        mode="record",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
    )
    terminal = {
        "choices": [
            {
                "delta": {},
                "finish_reason": "tool_calls",
            }
        ]
    }

    class Content:
        async def iter_any(self):
            yield (
                f"data: {json.dumps(terminal, separators=(',', ':'))}\n\ndata: [DONE]\n\n"
            ).encode()

    class Upstream:
        status = 200
        content = Content()

    class Response:
        async def prepare(self, _request):
            pass

        async def write(self, data):
            if b"[DONE]" in data:
                raise ConnectionResetError("Cannot write to closing transport")

        async def write_eof(self):
            raise ConnectionResetError("Cannot write to closing transport")

    monkeypatch.setattr(llm_store.web, "StreamResponse", lambda **_kwargs: Response())

    async def capture_engine(_attempt_id, _committed):
        return {"capture": "complete"}

    monkeypatch.setattr(store_, "_capture_engine", capture_engine)
    asyncio.run(
        store_._record_stream(
            object(),
            Upstream(),
            identity(),
            BODY,
            "chat.completions",
            "llm-terminal-reset",
            0,
            100,
        )
    )

    written = json.loads((tmp_path / "llm.jsonl").read_text().strip())
    assert written["response"]["chunks"] == [{"done": False, "payload": terminal}]


def test_incomplete_record_stream_becomes_a_cutoff_tail_on_client_close(
    tmp_path: Path, monkeypatch
) -> None:
    store_ = LLMStore(
        mode="record",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
    )

    class Content:
        async def iter_any(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    class Upstream:
        status = 200
        content = Content()

    class Response:
        async def prepare(self, _request):
            pass

        async def write(self, _data):
            pass

        async def write_eof(self):
            raise ConnectionResetError("Cannot write to closing transport")

    monkeypatch.setattr(llm_store.web, "StreamResponse", lambda **_kwargs: Response())
    attempt_id = "llm-partial-reset"
    store_._inflight[attempt_id] = {
        "attempt_id": attempt_id,
        "actor_id": "actor-0",
        "started_at_ns": 100,
    }

    asyncio.run(
        store_._record_stream(
            object(),
            Upstream(),
            identity(),
            BODY,
            "chat.completions",
            attempt_id,
            0,
            100,
        )
    )

    assert not (tmp_path / "llm.jsonl").exists()
    tail = store_.freeze_source_cutoff(10**30)[0]
    assert tail["attempt_id"] == attempt_id
    assert tail["interruption"] == "client-disconnected-before-terminal"
    assert tail["partial_response"]["chunks"][0]["payload"]["choices"][0]["delta"] == {
        "content": "partial"
    }
    assert tail["elapsed_ns"] == tail["interrupted_at_ns"] - 100


def test_full_replay_drains_forced_stream_without_decoding_it_as_json(
    tmp_path: Path, monkeypatch
) -> None:
    store_ = store(tmp_path, [])
    store_.force_secret = "secret"
    store_.audit_namespace = "run"
    expected_engine = {
        "committed_sample_start": 0,
        "sampled_token_count": 1,
    }
    expected = {
        "attempt_id": "llm-stream",
        "request": {"model": "m", "stream": True},
        "stream": True,
        "status_code": 200,
        "prompt_token_ids": [1],
        "response_token_ids": [2],
        "engine": expected_engine,
    }
    drained: list[bytes] = []

    class Content:
        async def iter_any(self):
            for value in (b"data: first\n\n", b"data: [DONE]\n\n"):
                drained.append(value)
                yield value

    class Upstream:
        status = 200
        content = Content()

        async def json(self):
            raise AssertionError("a forced SSE response must not be decoded as JSON")

    class Post:
        async def __aenter__(self):
            return Upstream()

        async def __aexit__(self, *_args):
            pass

    class Client:
        def post(self, *_args, **_kwargs):
            return Post()

    class Audit:
        async def wait(self, *_args, **_kwargs):
            return {"audit": "complete"}

    store_.client = Client()
    store_.audit = Audit()
    monkeypatch.setattr(llm_store, "forced_upstream_body", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        llm_store,
        "engine_evidence",
        lambda _record, _committed: expected_engine,
    )

    asyncio.run(
        store_._run_upstream(
            identity(),
            "chat.completions",
            expected,
        )
    )

    assert drained == [b"data: first\n\n", b"data: [DONE]\n\n"]


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


def test_claim_tolerates_live_multimodal_history_eviction(tmp_path: Path) -> None:
    """Fresh screenshot cost may change retained history, not the current turn."""

    inline_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": "act on the current browser viewport"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA"},
            },
        ],
    }
    recorded = {
        "model": "vision",
        "messages": [
            {"role": "system", "content": "browse safely"},
            {"role": "assistant", "content": "older action"},
            inline_turn,
            {"role": "assistant", "content": "most recent action"},
        ],
        "max_tokens": 4096,
    }
    observed = {
        "model": "vision",
        "messages": [
            {"role": "system", "content": "browse safely"},
            inline_turn,
        ],
        "max_tokens": 4096,
    }
    store_ = store(tmp_path, [llm(request=recorded, role="browser_web")])

    claimed = store_._claim(identity(role="browser_web"), observed, "chat.completions")

    assert claimed["attempt_id"] == "llm-0"
