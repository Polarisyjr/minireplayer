from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Sequence

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")


def resolved_output_prefix(output_token_ids: Sequence[int]) -> tuple[list[int], int]:
    """Split vLLM's committed output from async-scheduling placeholders.

    With async scheduling, vLLM optimistically appends one or more ``-1``
    entries until the preceding GPU-to-CPU token copy is visible.  Real token
    IDs must still form an exact prefix; a real ID after a placeholder would
    mean the engine's bookkeeping no longer has the shape replay relies on.
    """

    values = list(output_token_ids)
    try:
        first_placeholder = values.index(-1)
    except ValueError:
        first_placeholder = len(values)
    if any(token_id != -1 for token_id in values[first_placeholder:]):
        raise ValueError("native replay output placeholders are not a trailing run")
    return values[:first_placeholder], len(values) - first_placeholder


def token_payload(request_id: str, token_ids: Sequence[int]) -> bytes:
    return f"{request_id}:{','.join(str(token) for token in token_ids)}".encode()


def sign_tokens(secret: str, request_id: str, token_ids: Sequence[int]) -> str:
    return hmac.new(
        secret.encode(), token_payload(request_id, token_ids), hashlib.sha256
    ).hexdigest()


def sign_capture(secret: str, request_id: str) -> str:
    return hmac.new(secret.encode(), f"capture:{request_id}".encode(), hashlib.sha256).hexdigest()


def sampler_window_payload(
    request_id: str,
    committed_sample_start: int,
    sampled_token_count: int,
) -> bytes:
    return (
        f"sampler-window:{request_id}:{committed_sample_start}:"
        f"{sampled_token_count}"
    ).encode()


def sign_sampler_window(
    secret: str,
    request_id: str,
    committed_sample_start: int,
    sampled_token_count: int,
) -> str:
    return hmac.new(
        secret.encode(),
        sampler_window_payload(
            request_id,
            committed_sample_start,
            sampled_token_count,
        ),
        hashlib.sha256,
    ).hexdigest()


def validate_sampler_window(
    *,
    secret: str,
    request_id: str,
    committed_sample_start: object,
    sampled_token_count: object,
    committed_token_count: int,
    signature: object,
) -> tuple[int, int]:
    if (
        not isinstance(committed_sample_start, int)
        or isinstance(committed_sample_start, bool)
        or not isinstance(sampled_token_count, int)
        or isinstance(sampled_token_count, bool)
        or committed_sample_start < 0
        or sampled_token_count < committed_sample_start + committed_token_count
    ):
        raise ValueError("invalid native replay sampler window")
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature,
        sign_sampler_window(
            secret,
            request_id,
            committed_sample_start,
            sampled_token_count,
        ),
    ):
        raise ValueError("invalid native replay sampler-window signature")
    return committed_sample_start, sampled_token_count


def validate_request(
    *,
    secret: str,
    request_id: object,
    token_ids: object,
    signature: object,
) -> tuple[str, list[int]]:
    if not isinstance(request_id, str) or REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise ValueError("invalid native replay request ID")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(isinstance(token, int) and token >= 0 for token in token_ids)
    ):
        raise ValueError("native replay token sequence must be a non-empty integer list")
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature,
        sign_tokens(secret, request_id, token_ids),
    ):
        raise ValueError("invalid native replay token signature")
    return request_id, list(token_ids)


def validate_capture(*, secret: str, request_id: object, signature: object) -> str:
    if not isinstance(request_id, str) or REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise ValueError("invalid native replay capture ID")
    if not isinstance(signature, str) or not hmac.compare_digest(
        signature, sign_capture(secret, request_id)
    ):
        raise ValueError("invalid native replay capture signature")
    return request_id


def audit_payload(
    *,
    request_id: str,
    prompt_token_ids: Sequence[int],
    sampled_token_ids: Sequence[int],
    forced_token_ids: Sequence[int],
    forced_count: int,
    status: str,
    pid: int,
    mode: str,
    committed_sample_start: int | None,
    expected_sampled_token_count: int | None,
) -> bytes:
    value = {
        "schema_version": "native-agent-replay.vllm-audit/v4",
        "request_id": request_id,
        "mode": mode,
        "prompt_token_ids": list(prompt_token_ids),
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_sha256": hashlib.sha256(
            json.dumps(list(prompt_token_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "sampled_token_ids": list(sampled_token_ids),
        "sampled_token_count": len(sampled_token_ids),
        "sampled_token_sha256": hashlib.sha256(
            json.dumps(list(sampled_token_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "forced_token_ids": list(forced_token_ids),
        "forced_token_count": len(forced_token_ids),
        "forced_token_sha256": hashlib.sha256(
            json.dumps(list(forced_token_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "forced_count": forced_count,
        "committed_sample_start": committed_sample_start,
        "expected_sampled_token_count": expected_sampled_token_count,
        "status": status,
        "pid": pid,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
