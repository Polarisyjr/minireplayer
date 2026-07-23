from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from .errors import MismatchError, ValidationError
from .util import require, sha256_json


def token_payload(request_id: str, token_ids: list[int]) -> bytes:
    return f"{request_id}:{','.join(str(token) for token in token_ids)}".encode()


def sign_tokens(secret: str, request_id: str, token_ids: list[int]) -> str:
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
    return (f"sampler-window:{request_id}:{committed_sample_start}:{sampled_token_count}").encode()


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


def token_digest(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def vllm_work_identity(audit: dict[str, Any]) -> str:
    sampled = audit.get("sampled_token_count")
    forced = audit.get("forced_token_count")
    committed_start = audit.get("committed_sample_start")
    require(
        isinstance(sampled, int)
        and not isinstance(sampled, bool)
        and isinstance(forced, int)
        and not isinstance(forced, bool)
        and isinstance(committed_start, int)
        and not isinstance(committed_start, bool)
        and sampled >= forced >= 0,
        "invalid vLLM sampler work counts",
    )
    require(
        0 <= committed_start <= sampled - forced,
        "invalid vLLM committed sampler window",
    )
    return sha256_json(
        {
            "prompt_token_sha256": audit.get("prompt_token_sha256"),
            "prompt_token_count": audit.get("prompt_token_count"),
            "forced_token_sha256": audit.get("forced_token_sha256"),
            "forced_token_count": forced,
            "sampled_token_count": sampled,
            "committed_sample_start": committed_start,
            "uncommitted_prefix_count": committed_start,
            "uncommitted_suffix_count": sampled - committed_start - forced,
            "status": audit.get("status"),
        }
    )


class ForcedAuditReader:
    """Incrementally fan out a fleet audit log to per-request waiters.

    The vLLM fleet owns one append-only file for its whole lifetime.  A supervisor
    run therefore starts at the file's current end and only consumes records that
    arrive after it starts.  One pump tails that run-local suffix for all concurrent
    requests; filesystem reads and JSON decoding stay off the aiohttp event loop.
    """

    poll_interval_s = 0.02

    def __init__(self, path: Path, *, start_at_end: bool = False):
        self.path = path
        try:
            metadata = path.stat()
        except FileNotFoundError:
            metadata = None
        self._identity = (
            (metadata.st_dev, metadata.st_ino) if metadata is not None else None
        )
        self._offset = metadata.st_size if metadata is not None and start_at_end else 0
        self._partial = b""
        self._seen: dict[str, dict[str, Any]] = {}
        self._waiters: dict[str, set[asyncio.Future[dict[str, Any]]]] = {}
        self._pump_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._failure: Exception | None = None

    def _read_new_records(self) -> list[dict[str, Any]]:
        """Read each newly appended byte once, retaining an incomplete final line."""

        try:
            with self.path.open("rb") as stream:
                metadata = os.fstat(stream.fileno())
                identity = (metadata.st_dev, metadata.st_ino)
                if self._identity is None:
                    self._identity = identity
                require(
                    identity == self._identity,
                    f"forced-token audit file changed during the run: {self.path}",
                )
                require(
                    metadata.st_size >= self._offset,
                    f"forced-token audit file was truncated during the run: {self.path}",
                )
                stream.seek(self._offset)
                chunk = stream.read()
                self._offset = stream.tell()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise ValidationError(f"cannot read forced-token audit {self.path}: {exc}") from exc

        if not chunk:
            return []
        lines = (self._partial + chunk).split(b"\n")
        self._partial = lines.pop()
        records: list[dict[str, Any]] = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"malformed appended forced-token audit record in {self.path}: {exc}"
                ) from exc
            require(
                isinstance(value, dict),
                f"forced-token audit record in {self.path} is not an object",
            )
            records.append(value)
        return records

    def _publish(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            require(
                record.get("schema_version") == "native-agent-replay.vllm-audit/v4",
                "invalid forced-token audit schema",
            )
            request_id = record.get("request_id")
            require(isinstance(request_id, str) and request_id, "invalid audit request ID")
            require(request_id not in self._seen, f"duplicate forced-token audit: {request_id}")
            self._seen[request_id] = record
            for future in self._waiters.pop(request_id, set()):
                if not future.done():
                    future.set_result(record)

    def refresh(self) -> None:
        """Synchronously ingest the next suffix; primarily useful in diagnostics."""

        self._publish(self._read_new_records())

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            active = self._waiters or (
                self._pump_task is not None and not self._pump_task.done()
            )
            if active:
                raise RuntimeError("forced-token audit reader used by multiple event loops")
            self._pump_task = None
        self._loop = loop
        return loop

    def _ensure_pump(self) -> None:
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(
                self._pump(),
                name="minireplay-forced-audit-pump",
            )

    async def _pump(self) -> None:
        try:
            while self._waiters:
                records = await asyncio.to_thread(self._read_new_records)
                self._publish(records)
                if self._waiters:
                    await asyncio.sleep(self.poll_interval_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - propagate one reader failure to every waiter
            self._failure = exc
            waiters = [future for futures in self._waiters.values() for future in futures]
            self._waiters.clear()
            for future in waiters:
                if not future.done():
                    future.set_exception(exc)
        finally:
            if self._pump_task is asyncio.current_task():
                self._pump_task = None

    async def _wait_for_record(
        self,
        request_id: str,
        *,
        timeout_s: float,
        missing: str,
    ) -> dict[str, Any]:
        if self._failure is not None:
            raise self._failure
        if record := self._seen.get(request_id):
            return record

        loop = self._bind_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._waiters.setdefault(request_id, set()).add(future)
        self._ensure_pump()
        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except TimeoutError as exc:
            raise MismatchError(missing) from exc
        finally:
            futures = self._waiters.get(request_id)
            if futures is not None:
                futures.discard(future)
                if not futures:
                    self._waiters.pop(request_id, None)

    async def wait(
        self,
        request_id: str,
        token_ids: list[int],
        *,
        prompt_token_ids: list[int] | None = None,
        committed_sample_start: int,
        sampled_token_count: int,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        record = await self._wait_for_record(
            request_id,
            timeout_s=timeout_s,
            missing=f"forced-token audit missing for {request_id}",
        )
        if (
            record.get("mode") != "force"
            or record.get("status") != "complete"
            or record.get("forced_count") != len(token_ids)
            or record.get("forced_token_count") != len(token_ids)
            or record.get("forced_token_sha256") != token_digest(token_ids)
            or record.get("forced_token_ids") != token_ids
            or not isinstance(record.get("sampled_token_count"), int)
            or record.get("sampled_token_count") != sampled_token_count
            or record.get("sampled_token_count") != len(record.get("sampled_token_ids", []))
            or record.get("sampled_token_sha256")
            != token_digest(record.get("sampled_token_ids", []))
            or record.get("committed_sample_start") != committed_sample_start
            or record.get("expected_sampled_token_count") != sampled_token_count
        ):
            raise MismatchError(f"forced-token audit mismatch for {request_id}")
        if prompt_token_ids is not None and (
            record.get("prompt_token_count") != len(prompt_token_ids)
            or record.get("prompt_token_sha256") != token_digest(prompt_token_ids)
            or record.get("prompt_token_ids") != prompt_token_ids
        ):
            raise MismatchError(f"prompt-token audit mismatch for {request_id}")
        return record

    async def wait_capture(
        self,
        request_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        record = await self._wait_for_record(
            request_id,
            timeout_s=timeout_s,
            missing=f"vLLM capture audit missing for {request_id}",
        )
        prompt = record.get("prompt_token_ids")
        sampled = record.get("sampled_token_ids")
        if (
            record.get("mode") != "capture"
            or record.get("status") != "capture_complete"
            or record.get("committed_sample_start") is not None
            or record.get("expected_sampled_token_count") is not None
            or not isinstance(prompt, list)
            or not isinstance(sampled, list)
            or not sampled
            or not all(isinstance(item, int) and item >= 0 for item in prompt + sampled)
            or record.get("prompt_token_sha256") != token_digest(prompt)
            or record.get("sampled_token_sha256") != token_digest(sampled)
            or not isinstance(record.get("sampled_token_count"), int)
            or record["sampled_token_count"] != len(sampled)
            or record.get("forced_token_ids") != []
            or record.get("forced_token_count") != 0
        ):
            raise MismatchError(f"vLLM capture audit mismatch for {request_id}")
        return record


def replay_audit_request_id(namespace: str | None, attempt_id: str) -> str:
    """A per-attempt engine request ID.

    Namespaced by run so two runs sharing one audit file cannot collide, and hashed
    so it satisfies the engine's request-ID charset without leaking a path.
    """

    seed = f"{namespace or 'default'}:{attempt_id}"
    return "nr-" + hashlib.sha256(seed.encode()).hexdigest()[:40]


def committed_sample_window(sampled_token_ids: list[int], committed_token_ids: list[int]) -> int:
    """Where the committed tokens sit inside the engine's own step sequence.

    One API call is not one decode step per committed token: speculative decoding and
    discarded steps mean the engine samples more than it commits. Forced replay has
    to line its tokens up against the engine's step counter, so recording locates the
    committed run inside the sampled run. An ambiguous match is fatal — guessing the
    offset would silently force the wrong steps.
    """

    width = len(committed_token_ids)
    require(width > 0, "engine committed no tokens")
    starts = [
        start
        for start in range(len(sampled_token_ids) - width + 1)
        if sampled_token_ids[start : start + width] == committed_token_ids
    ]
    require(
        len(starts) == 1,
        "engine sampler sequence has no unique contiguous committed-token window",
    )
    return starts[0]


def engine_evidence(audit: dict[str, Any], committed_token_ids: list[int]) -> dict[str, Any]:
    """The engine step window a forced replay must reproduce."""

    sampled_ids = audit.get("sampled_token_ids")
    require(isinstance(sampled_ids, list), "invalid engine sampled-token inventory")
    sampled = len(sampled_ids)
    committed = len(committed_token_ids)
    require(sampled >= committed, "engine sampled fewer tokens than it committed")
    if audit.get("mode") == "capture":
        committed_start = committed_sample_window(sampled_ids, committed_token_ids)
    else:
        committed_start = audit.get("committed_sample_start")
        require(
            isinstance(committed_start, int) and not isinstance(committed_start, bool),
            "forced-token audit omitted the committed sampler window",
        )
    suffix = sampled - committed_start - committed
    require(
        committed_start >= 0 and suffix >= 0,
        "engine committed sampler window is outside the sampled sequence",
    )
    return {
        "audit_schema": audit.get("schema_version"),
        "sampled_token_count": sampled,
        "committed_sample_start": committed_start,
        "committed_sample_count": committed,
        "uncommitted_prefix_count": committed_start,
        "uncommitted_suffix_count": suffix,
    }


def forced_upstream_body(
    recorded_request: dict[str, Any],
    expected: dict[str, Any],
    *,
    api: str,
    secret: str,
    audit_namespace: str | None,
) -> dict[str, Any]:
    """Build the upstream body that makes the engine commit the recorded tokens.

    The engine still runs prefill, logits and the sampling kernel for every step.
    Only the integer committed at each in-window step is replaced, after sampling.
    """

    ordered_json = expected.get("request_ordered_json")
    if ordered_json is None:
        # Backward compatibility for bundles recorded before ordered request
        # encodings were added. Such a bundle remains usable when its chat
        # template is insensitive to mapping order.
        body = copy.deepcopy(recorded_request)
    else:
        require(
            isinstance(ordered_json, str) and bool(ordered_json),
            "recorded ordered LLM request is invalid",
        )
        try:
            body = json.loads(ordered_json)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"recorded ordered LLM request is malformed: {exc}") from exc
        require(
            isinstance(body, dict) and body == recorded_request,
            "recorded ordered LLM request does not match its canonical request",
        )
    if api == "chat.completions":
        require(body.get("n", 1) == 1, "forced replay supports one completion per request")
        body["return_token_ids"] = True
    elif api == "responses":
        require(not body.get("stream"), "forced replay does not support streamed Responses")
        body["enable_response_messages"] = True
    else:
        raise ValidationError(f"unsupported LLM API: {api}")

    if int(expected.get("status_code", 200)) >= 400:
        return body

    token_ids = list(expected["response_token_ids"])
    require(bool(token_ids), "forced replay needs a non-empty committed token sequence")
    engine = expected.get("engine")
    require(
        isinstance(engine, dict),
        "this bundle carries no engine step window; re-record with a forced-capable "
        "vLLM so capture mode can observe it",
    )

    request_id = replay_audit_request_id(audit_namespace, str(expected["attempt_id"]))
    xargs = dict(body.get("vllm_xargs") or {})
    reserved = {
        "native_replay_request_id",
        "native_replay_prompt_token_ids",
        "native_replay_prompt_signature",
        "native_replay_token_ids",
        "native_replay_signature",
        "native_replay_committed_sample_start",
        "native_replay_sampled_token_count",
        "native_replay_sampler_signature",
    }
    require(
        not (reserved & xargs.keys()),
        "the recorded request already sets reserved forced-replay arguments",
    )
    prompt_ids = list(expected["prompt_token_ids"])
    xargs.update(
        {
            "native_replay_request_id": request_id,
            "native_replay_prompt_token_ids": prompt_ids,
            "native_replay_prompt_signature": sign_tokens(secret, request_id, prompt_ids),
            "native_replay_token_ids": token_ids,
            "native_replay_signature": sign_tokens(secret, request_id, token_ids),
            "native_replay_committed_sample_start": engine["committed_sample_start"],
            "native_replay_sampled_token_count": engine["sampled_token_count"],
            "native_replay_sampler_signature": sign_sampler_window(
                secret,
                request_id,
                engine["committed_sample_start"],
                engine["sampled_token_count"],
            ),
        }
    )
    body["vllm_xargs"] = xargs
    return body
