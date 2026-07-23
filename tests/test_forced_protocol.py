"""The proxy and the engine sign the same bytes.

`minireplay.forced` runs in this process; `native_replay_vllm.protocol` runs inside
the vLLM container. They are deliberately separate implementations so that a change
to one cannot silently be assumed by the other — which means something has to check
that they still agree. That is this file.

A signature mismatch would surface as an opaque rejection deep inside the engine, so
it is worth catching in a unit test instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from minireplay import forced as proxy

PLUGIN_SRC = Path(__file__).parent.parent / "vllm_plugin" / "src"
sys.path.insert(0, str(PLUGIN_SRC))

engine = pytest.importorskip(
    "native_replay_vllm.protocol",
    reason="the vLLM plugin source is not present",
)

SECRET = "a-shared-secret"
REQUEST_ID = "nr-0123456789abcdef"
TOKENS = [1, 22, 333, 4444, 0, 999999]


def test_token_payloads_are_byte_identical() -> None:
    assert proxy.token_payload(REQUEST_ID, TOKENS) == engine.token_payload(REQUEST_ID, TOKENS)


def test_token_signatures_agree() -> None:
    assert proxy.sign_tokens(SECRET, REQUEST_ID, TOKENS) == engine.sign_tokens(
        SECRET, REQUEST_ID, TOKENS
    )


def test_capture_signatures_agree() -> None:
    assert proxy.sign_capture(SECRET, REQUEST_ID) == engine.sign_capture(SECRET, REQUEST_ID)


def test_sampler_window_payloads_are_byte_identical() -> None:
    assert proxy.sampler_window_payload(REQUEST_ID, 3, 17) == engine.sampler_window_payload(
        REQUEST_ID, 3, 17
    )


def test_sampler_window_signatures_agree() -> None:
    assert proxy.sign_sampler_window(SECRET, REQUEST_ID, 3, 17) == engine.sign_sampler_window(
        SECRET, REQUEST_ID, 3, 17
    )


def test_an_empty_token_sequence_still_agrees() -> None:
    assert proxy.sign_tokens(SECRET, REQUEST_ID, []) == engine.sign_tokens(SECRET, REQUEST_ID, [])


def test_a_different_secret_changes_the_signature() -> None:
    """Guard against a signature that ignores its key."""

    assert proxy.sign_tokens(SECRET, REQUEST_ID, TOKENS) != proxy.sign_tokens(
        "other-secret", REQUEST_ID, TOKENS
    )


def test_token_order_is_part_of_the_signature() -> None:
    assert proxy.sign_tokens(SECRET, REQUEST_ID, [1, 2]) != proxy.sign_tokens(
        SECRET, REQUEST_ID, [2, 1]
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], ([], 0)),
        ([11, 22], ([11, 22], 0)),
        ([11, 22, -1], ([11, 22], 1)),
        ([11, -1, -1], ([11], 2)),
    ],
)
def test_async_output_placeholders_are_split_from_the_resolved_prefix(
    values: list[int], expected: tuple[list[int], int]
) -> None:
    assert engine.resolved_output_prefix(values) == expected


def test_async_output_placeholders_must_be_a_trailing_run() -> None:
    with pytest.raises(ValueError, match="not a trailing run"):
        engine.resolved_output_prefix([11, -1, 22])
