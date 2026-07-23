from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from .protocol import (
    audit_payload,
    resolved_output_prefix,
    validate_capture,
    validate_request,
    validate_sampler_window,
)

_AUDIT_LOCK = threading.Lock()
_WRITTEN_AUDITS: dict[str, bytes] = {}


def _prompt_drift_detail(actual: list[int], expected: list[int]) -> str:
    shared = min(len(actual), len(expected))
    first = next((index for index in range(shared) if actual[index] != expected[index]), shared)
    start = max(0, first - 3)
    end = first + 4

    def digest(tokens: list[int]) -> str:
        payload = json.dumps(tokens, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    return (
        f"actual_len={len(actual)}, expected_len={len(expected)}, first_diff={first}, "
        f"actual_sha256={digest(actual)}, expected_sha256={digest(expected)}, "
        f"actual_window={actual[start:end]}, expected_window={expected[start:end]}"
    )


@dataclass
class RequestState:
    request_id: str
    mode: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    sampled_token_ids: list[int]
    output_token_ids: list[int]
    committed_sample_start: int | None = None
    expected_sampled_token_count: int | None = None
    forced_count: int = 0


class ForcedSequenceProcessor(LogitsProcessor):
    """Keep normal sampling intact, then commit a signed recorded token sequence.

    `apply` deliberately leaves logits untouched. The companion vLLM sampler
    patch calls `native_replay_override` only after `Sampler.sample` returns.
    """

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        del vllm_config, is_pin_memory
        self._secret = os.environ.get("NATIVE_REPLAY_FORCE_SECRET", "")
        if not self._secret:
            raise RuntimeError("NATIVE_REPLAY_FORCE_SECRET is required")
        audit_path = os.environ.get("NATIVE_REPLAY_FORCE_AUDIT")
        if not audit_path:
            raise RuntimeError("NATIVE_REPLAY_FORCE_AUDIT is required")
        self._audit_path = Path(audit_path)
        # Every tensor-parallel rank must apply the same forced sequence, but a
        # request has exactly one engine audit. vLLM assigns local rank zero to
        # cuda:0 inside each serving instance.
        self._writes_audit = device.type != "cuda" or device.index in (None, 0)
        self._states: dict[int, RequestState] = {}
        self._seen_request_ids: set[str] = set()
        self._discard_mask: tuple[bool, ...] | None = None

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        forced = "native_replay_request_id" in extra
        forced_fields = {
            "native_replay_request_id",
            "native_replay_prompt_token_ids",
            "native_replay_prompt_signature",
            "native_replay_token_ids",
            "native_replay_signature",
            "native_replay_committed_sample_start",
            "native_replay_sampled_token_count",
            "native_replay_sampler_signature",
        }
        capture = "native_replay_capture_id" in extra
        capture_fields = {"native_replay_capture_id", "native_replay_capture_signature"}
        if forced and not forced_fields <= extra.keys():
            raise ValueError("incomplete native replay forced-token request")
        if capture and not capture_fields <= extra.keys():
            raise ValueError("incomplete native replay capture request")
        if forced and capture:
            raise ValueError("native replay request cannot capture and force simultaneously")

    def is_argmax_invariant(self) -> bool:
        return True

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def _new_state(
        self,
        params: SamplingParams,
        prompt_ids: list[int],
        _output_ids: list[int],
    ) -> RequestState | None:
        extra = params.extra_args or {}
        if "native_replay_capture_id" in extra:
            request_id = validate_capture(
                secret=self._secret,
                request_id=extra.get("native_replay_capture_id"),
                signature=extra.get("native_replay_capture_signature"),
            )
            state = RequestState(
                request_id=request_id,
                mode="capture",
                prompt_token_ids=list(prompt_ids),
                token_ids=[],
                sampled_token_ids=[],
                output_token_ids=_output_ids,
            )
        elif "native_replay_request_id" in extra:
            request_id, token_ids = validate_request(
                secret=self._secret,
                request_id=extra.get("native_replay_request_id"),
                token_ids=extra.get("native_replay_token_ids"),
                signature=extra.get("native_replay_signature"),
            )
            _, expected_prompt_ids = validate_request(
                secret=self._secret,
                request_id=request_id,
                token_ids=extra.get("native_replay_prompt_token_ids"),
                signature=extra.get("native_replay_prompt_signature"),
            )
            actual_prompt_ids = list(prompt_ids)
            if actual_prompt_ids != expected_prompt_ids:
                detail = _prompt_drift_detail(actual_prompt_ids, expected_prompt_ids)
                raise ValueError(
                    f"native replay prompt token drift for {request_id}: {detail}"
                )
            if params.max_tokens is not None and params.max_tokens < len(token_ids):
                raise ValueError("native replay token sequence exceeds source max_tokens")
            committed_sample_start, sampled_token_count = validate_sampler_window(
                secret=self._secret,
                request_id=request_id,
                committed_sample_start=extra.get("native_replay_committed_sample_start"),
                sampled_token_count=extra.get("native_replay_sampled_token_count"),
                committed_token_count=len(token_ids),
                signature=extra.get("native_replay_sampler_signature"),
            )
            state = RequestState(
                request_id=request_id,
                mode="force",
                prompt_token_ids=list(prompt_ids),
                token_ids=token_ids,
                sampled_token_ids=[],
                output_token_ids=_output_ids,
                committed_sample_start=committed_sample_start,
                expected_sampled_token_count=sampled_token_count,
            )
        else:
            return None
        if request_id in self._seen_request_ids:
            raise ValueError(f"native replay request ID reused: {request_id}")
        self._seen_request_ids.add(request_id)
        return state

    def _write_audit(self, state: RequestState) -> None:
        if not self._writes_audit:
            return
        if state.mode == "capture":
            status = "capture_complete"
        else:
            start = state.committed_sample_start
            expected = state.expected_sampled_token_count
            status = (
                "complete"
                if isinstance(start, int)
                and isinstance(expected, int)
                and state.forced_count == len(state.token_ids)
                and len(state.sampled_token_ids) == expected
                and state.sampled_token_ids[start : start + len(state.token_ids)]
                == state.token_ids
                else "incomplete"
            )
        payload = audit_payload(
            request_id=state.request_id,
            prompt_token_ids=state.prompt_token_ids,
            sampled_token_ids=state.sampled_token_ids,
            forced_token_ids=state.token_ids if state.mode == "force" else [],
            forced_count=state.forced_count,
            status=status,
            pid=os.getpid(),
            mode=state.mode,
            committed_sample_start=state.committed_sample_start,
            expected_sampled_token_count=state.expected_sampled_token_count,
        )
        with _AUDIT_LOCK:
            previous = _WRITTEN_AUDITS.get(state.request_id)
            if previous is not None:
                if previous != payload:
                    raise RuntimeError(
                        f"conflicting native replay audits for request {state.request_id}"
                    )
                return
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                self._audit_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
                0o644,
            )
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            _WRITTEN_AUDITS[state.request_id] = payload

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        # vLLM may recycle a removed request's persistent-batch index for an
        # added request in the same update. Honor BatchUpdate's required
        # removed -> added -> moved ordering so the completed request is
        # audited before the new state occupies its slot.
        for index in batch_update.removed:
            state = self._states.pop(index, None)
            if state is not None:
                self._write_audit(state)
        for index, params, prompt_ids, output_ids in batch_update.added:
            # GPUInputBatch.pop_removed() consumes a recycled destination from
            # BatchUpdate.removed before publishing this update. Finalize the
            # completed state that still occupies the slot before replacing it.
            replaced = self._states.pop(index, None)
            if replaced is not None:
                self._write_audit(replaced)
            state = self._new_state(params, prompt_ids, output_ids)
            if state is not None:
                self._states[index] = state
        for first, second, direction in batch_update.moved:
            first_state = self._states.pop(first, None)
            second_state = self._states.pop(second, None)
            if first_state is not None:
                self._states[second] = first_state
            if second_state is not None:
                if direction == MoveDirectionality.SWAP:
                    self._states[first] = second_state
                else:
                    # condense() consumes the removed destination before it
                    # reports a unidirectional move. The destination's old
                    # request is complete and must be audited before overwrite.
                    self._write_audit(second_state)

    def native_replay_set_discard_mask(self, discarded) -> None:
        self._discard_mask = tuple(bool(value) for value in discarded)

    def native_replay_override(self, sampled: torch.Tensor) -> torch.Tensor:
        if self._discard_mask is None:
            raise RuntimeError("native replay valid-sample mask was not published")
        for request_index, state in self._states.items():
            if request_index >= len(self._discard_mask):
                raise RuntimeError("native replay request index exceeds valid-sample mask")
            if self._discard_mask[request_index]:
                continue
            natural_token_id = int(sampled[request_index].item())
            if state.mode == "capture":
                state.sampled_token_ids.append(natural_token_id)
                continue
            start = state.committed_sample_start
            expected = state.expected_sampled_token_count
            if not isinstance(start, int) or not isinstance(expected, int):
                raise RuntimeError(
                    f"native replay sampler contract missing for {state.request_id}"
                )
            sample_index = len(state.sampled_token_ids)
            if sample_index >= expected:
                raise RuntimeError(
                    f"native replay sampled beyond source count for {state.request_id}"
                )
            position = sample_index - start
            if 0 <= position < len(state.token_ids):
                try:
                    resolved_output, _ = resolved_output_prefix(
                        state.output_token_ids
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"native replay malformed output placeholders for "
                        f"{state.request_id}"
                    ) from exc
                if (
                    len(resolved_output) > position
                    or resolved_output != state.token_ids[: len(resolved_output)]
                    or len(state.output_token_ids) > position + 1
                ):
                    raise RuntimeError(
                        f"native replay commit-position drift for {state.request_id}: "
                        f"resolved_output={len(resolved_output)}, "
                        f"logical_output={len(state.output_token_ids)}, "
                        f"expected={position}"
                    )
                token_id = state.token_ids[position]
                sampled[request_index] = token_id
                state.forced_count += 1
            elif position >= len(state.token_ids):
                try:
                    resolved_output, _ = resolved_output_prefix(
                        state.output_token_ids
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"native replay malformed output placeholders for "
                        f"{state.request_id}"
                    ) from exc
                forced_width = min(len(resolved_output), len(state.token_ids))
                if resolved_output[:forced_width] != state.token_ids[:forced_width]:
                    raise RuntimeError(
                        f"native replay committed output drift before suffix for "
                        f"{state.request_id}"
                    )
            state.sampled_token_ids.append(int(sampled[request_index].item()))
        return sampled
