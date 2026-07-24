"""Replay projection for Owl's concurrently checked live document chunks."""

from pathlib import Path

import pytest

from minireplay.errors import MismatchError
from minireplay.llm_store import LLMStore, RequestIdentity
from tests.support import llm, make_bundle


def _body(document: str, *, query: str = "the stable query") -> dict:
    return {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Check this retrieved chunk.\n"
                    f"<document_part>\n{document}\n</document_part>\n"
                    f"<query>\n{query}\n</query>"
                ),
            }
        ],
        "max_tokens": 4096,
    }


def _store(tmp_path: Path, recorded: dict) -> LLMStore:
    return LLMStore(
        mode="replay",
        stage_dir=tmp_path,
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
        bundle=make_bundle(
            llm_records=[llm(request=recorded, role="document")],
            tools=[],
            dispatches=[],
        ),
    )


def _identity() -> RequestIdentity:
    return RequestIdentity(
        actor_id="actor-0",
        session_id="actor-0",
        role="document",
        target_id="vllm-8000",
        parent_span_id=None,
    )


def test_document_claim_ignores_only_live_chunk_body(tmp_path: Path) -> None:
    store = _store(tmp_path, _body("short"))

    claimed = store._claim(
        _identity(),
        _body("a different live chunk " * 5_000),
        "chat.completions",
    )

    assert claimed["attempt_id"] == "llm-0"


def test_document_claim_still_rejects_query_drift(tmp_path: Path) -> None:
    store = _store(tmp_path, _body("short", query="q"))

    with pytest.raises(MismatchError, match="live document projection"):
        store._claim(
            _identity(),
            _body("a different live chunk " * 5_000, query="q" * 5_000),
            "chat.completions",
        )
