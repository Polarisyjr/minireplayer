"""The LLM lane.

Recording puts this in front of the real vLLM: it forwards every request, keeps the
exact response the framework saw, and captures the engine's token IDs so a later
stage can force them.

Replay has two modes:

* ``tool-only`` answers from the bundle without contacting vLLM at all. Only the
  tools execute for real; the LLM lane is a fixture. This is the cheap mode used to
  iterate on tool behaviour.
* ``full`` re-sends the recorded request to a real vLLM and commits the recorded
  tokens, so prefill, logits and sampling are genuinely executed.

Either way the framework receives the recorded response bytes, because chunk
boundaries and provider IDs are not reproducible and the framework's parser is
sensitive to both.
"""

from __future__ import annotations

import asyncio
import copy
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from .constants import LLM_SCHEMA, MAX_REQUEST_BYTES, SPAN_SCHEMA
from .errors import MismatchError, ValidationError, WorkloadComplete
from .forced import (
    ForcedAuditReader,
    engine_evidence,
    forced_upstream_body,
    replay_audit_request_id,
    sign_capture,
)
from .util import append_jsonl, monotonic_ns, require, sha256_json

# Response fields the engine regenerates per call. They are recorded as evidence
# but never compared, and the recorded values are what the framework receives.
_TOKEN_FIELDS = ("prompt_token_ids", "token_ids", "raw_message_tokens")

REPLAY_MODES = ("tool-only", "full")


@dataclass(frozen=True)
class RequestIdentity:
    actor_id: str
    session_id: str
    role: str
    target_id: str
    parent_span_id: str | None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.actor_id, self.session_id, self.role)


def _identity(request: web.Request) -> RequestIdentity:
    headers = request.headers
    return RequestIdentity(
        actor_id=headers.get("X-Native-Replay-Actor", "unknown"),
        session_id=headers.get("X-Native-Replay-Session", "unknown"),
        role=headers.get("X-Native-Replay-Role", "agent"),
        target_id=headers.get("X-Native-Replay-Target", "default"),
        parent_span_id=headers.get("X-Native-Replay-Parent-Span"),
    )


class SSEDecoder:
    """Minimal server-sent-events reader.

    Only enough to split an upstream stream into whole events, so recording can keep
    each event exactly as the framework received it.
    """

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer += chunk
        events: list[bytes] = []
        while True:
            for separator in (b"\n\n", b"\r\n\r\n"):
                index = self._buffer.find(separator)
                if index != -1:
                    events.append(self._buffer[:index])
                    self._buffer = self._buffer[index + len(separator) :]
                    break
            else:
                return events

    def finish(self) -> None:
        if self._buffer.strip():
            raise ValidationError("upstream SSE stream ended mid-event")


def sse_payload(event: bytes) -> str | None:
    for line in event.split(b"\n"):
        if line.startswith(b"data:"):
            return line[len(b"data:") :].strip().decode("utf-8")
    return None


def encode_sse(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode()


def strip_token_fields(value: Any) -> Any:
    """Remove engine token IDs from what the framework sees.

    They are requested only so the recording can carry them; a framework that never
    asked for them must not suddenly receive them.
    """

    if isinstance(value, dict):
        return {k: strip_token_fields(v) for k, v in value.items() if k not in _TOKEN_FIELDS}
    if isinstance(value, list):
        return [strip_token_fields(item) for item in value]
    return value


def collect_token_fields(value: Any) -> dict[str, list[int]]:
    prompt: list[int] = []
    output: list[int] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            raw_prompt = item.get("prompt_token_ids")
            if isinstance(raw_prompt, list) and not prompt:
                prompt.extend(int(t) for t in raw_prompt if isinstance(t, int))
            raw_output = item.get("token_ids")
            if isinstance(raw_output, list):
                output.extend(int(t) for t in raw_output if isinstance(t, int))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return {"prompt_token_ids": prompt, "response_token_ids": output}


class LLMStore:
    def __init__(
        self,
        *,
        mode: str,
        stage_dir: Path,
        upstreams: dict[str, str],
        bundle: Any | None = None,
        replay_mode: str = "tool-only",
        force_secret: str | None = None,
        audit_path: Path | None = None,
        audit_namespace: str | None = None,
    ) -> None:
        require(mode in {"record", "replay"}, f"invalid LLM store mode: {mode!r}")
        require(replay_mode in REPLAY_MODES, f"invalid replay mode: {replay_mode!r}")
        self.mode = mode
        self.stage_dir = stage_dir
        self.upstreams = upstreams
        self.bundle = bundle
        self.replay_mode = replay_mode
        self.force_secret = force_secret
        # The vLLM fleet appends every run to one serving-level audit log.  Anchor
        # this reader at the current EOF so this LLM store only consumes evidence
        # produced for its own run; request IDs provide the second isolation layer.
        self.audit = (
            ForcedAuditReader(audit_path, start_at_end=True)
            if audit_path is not None
            else None
        )
        self.audit_namespace = audit_namespace
        self.hard_failure: str | None = None
        self.client: aiohttp.ClientSession | None = None
        # `/v1/models` answers per target, kept because a framework may build its
        # clients by asking each endpoint what it serves (owl does, for every role,
        # before any call). Tool-only replay contacts no vLLM, so without the
        # recorded answer that discovery would fail on an endpoint the window never
        # exercised — a harness artefact, not framework drift.
        self.model_catalogue: dict[str, Any] = {}
        if bundle is not None:
            recorded = bundle.manifest.get("llm_models")
            if isinstance(recorded, dict):
                self.model_catalogue = dict(recorded)

        self._record_sequence: dict[tuple[str, str, str], int] = {}
        self._replay_sequence: dict[tuple[str, str, str], int] = {}
        self._expected: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        self._model_call_index: dict[str, str] = {}
        # Set by ReplayServices: whether the whole run (LLM *and* boundary) has
        # finished its recorded work. Exhausting one queue is only benign once the
        # rest of the run is done too.
        self.run_complete: Any = None
        # ReplayServices also provides the corresponding actor-level predicate for
        # concurrent prefixes whose actors reach the source cutoff at different
        # wall-clock times during replay.
        self.actor_complete: Any = None
        self._consumed = 0
        # A replay claim only reserves the source slot.  Completion is split into
        # evidence written and response delivered so the supervisor cannot tear
        # down the final engine request (or its framework-visible response) while
        # it is still in flight.
        self._completed: set[str] = set()
        self._delivered: set[str] = set()
        # Requests forwarded upstream but not yet answered. Whatever is still here
        # when the sweep closes the window is a tail the source never finished.
        self._inflight: dict[str, dict[str, Any]] = {}
        self._truncated: set[str] = set()
        self._truncated_elapsed: dict[str, int] = {}
        self._truncated_started: dict[str, int] = {}
        self.source_cutoff_at_ns: int | None = None

        if bundle is not None:
            for record in bundle.llm:
                key = (record["actor_id"], record["session_id"], record["role"])
                self._expected.setdefault(key, []).append(record)
            for queue in self._expected.values():
                queue.sort(key=lambda item: item["sequence"])
            # Cutoff tails are source-side diagnostic evidence, not replay slots.
            # Replay stops after the closed prefix and never sends these requests
            # to vLLM.

    # ---- dispatch trigger index ---------------------------------------------

    def attempt_for_model_call(self, model_call_id: Any) -> str | None:
        """Which LLM attempt emitted a given provider tool-call ID.

        A framework can invoke a tool the moment it finishes parsing a streamed
        tool call, so this index is what lets a dispatch name its causal parent
        without the adapter having to thread the attempt ID through.
        """

        if not isinstance(model_call_id, str):
            return None
        return self._model_call_index.get(model_call_id)

    def _index_model_calls(self, attempt_id: str, response: Any) -> None:
        def visit(item: Any) -> None:
            if isinstance(item, dict):
                call_id = item.get("id")
                if isinstance(call_id, str) and item.get("type") in {"function", "tool_call"}:
                    self._model_call_index.setdefault(call_id, attempt_id)
                if isinstance(item.get("tool_calls"), list):
                    for call in item["tool_calls"]:
                        if isinstance(call, dict) and isinstance(call.get("id"), str):
                            self._model_call_index.setdefault(call["id"], attempt_id)
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(response)

    # ---- claim ---------------------------------------------------------------

    def _claim(self, identity: RequestIdentity, body: dict[str, Any], api: str) -> dict[str, Any]:
        key = identity.key
        queue = self._expected.get(key, [])
        sequence = self._replay_sequence.get(key, 0)
        if sequence >= len(queue):
            whole_run_complete = self.run_complete is not None and self.run_complete()
            actor_complete = self.actor_complete is not None and self.actor_complete(
                identity.actor_id
            )
            # Independent framework branches sharing one actor can reach their
            # source-window boundary at different replay wall times. Once this
            # non-empty LLM lane has delivered every recorded response, its next
            # request is past the known prefix and must wait without inventing
            # work while sibling lanes finish. An empty lane, or one whose final
            # response is merely claimed/in flight, remains hard drift.
            lane_complete = bool(queue) and all(
                str(record["attempt_id"]) in self._delivered for record in queue
            )
            if whole_run_complete or actor_complete or lane_complete:
                raise WorkloadComplete(
                    f"LLM request for {key} arrived after the recorded window closed"
                )
            raise MismatchError(
                f"unexpected LLM request for {key}: the recording holds {len(queue)} "
                f"and all are consumed"
            )
        expected = queue[sequence]
        if expected["api"] != api:
            raise MismatchError(
                f"LLM API drift for {key} at sequence {sequence}: "
                f"expected={expected['api']} actual={api}"
            )
        observed = sha256_json(request_shape(body))
        if observed != expected["request_shape_sha256"]:
            raise MismatchError(
                f"LLM request drift for {key} at sequence {sequence}: "
                f"the framework built a structurally different request"
            )
        self._replay_sequence[key] = sequence + 1
        return expected

    # ---- HTTP ----------------------------------------------------------------

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        return await self._handle(request, "chat.completions")

    async def handle_responses(self, request: web.Request) -> web.StreamResponse:
        return await self._handle(request, "responses")

    async def handle_models(self, request: web.Request) -> web.StreamResponse:
        try:
            identity = _identity(request)
            if self.mode == "replay" and self.replay_mode == "tool-only":
                recorded = self.model_catalogue.get(identity.target_id)
                if recorded is None:
                    raise MismatchError(
                        f"replay asked target {identity.target_id!r} what it serves, but the "
                        "recording never did — re-record with this deployment's endpoints"
                    )
                return web.json_response(recorded)
            assert self.client is not None
            async with self.client.get(f"{self._upstream(identity)}/v1/models") as upstream:
                payload = await upstream.json()
                if self.mode == "record" and upstream.status == 200:
                    self.model_catalogue[identity.target_id] = payload
                return web.json_response(payload, status=upstream.status)
        except (ValidationError, MismatchError) as exc:
            self.hard_failure = str(exc)
            return web.json_response({"error": {"message": str(exc)}}, status=409)

    def _upstream(self, identity: RequestIdentity) -> str:
        url = self.upstreams.get(identity.target_id)
        if url is None:
            raise MismatchError(f"no vLLM upstream is configured for target {identity.target_id!r}")
        return url.rstrip("/")

    async def _handle(self, request: web.Request, api: str) -> web.StreamResponse:
        try:
            identity = _identity(request)
            raw = await request.read()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"malformed LLM request body: {exc}") from exc
            require(isinstance(body, dict), "LLM request body must be an object")

            if self.mode == "record":
                return await self._record(request, identity, body, api)
            try:
                expected = self._claim(identity, body, api)
            except WorkloadComplete:
                return await self._hold_past_window()
            if str(expected["attempt_id"]) in self._truncated:
                return await self._hold_truncated(expected)
            return await self._replay(request, identity, api, expected)
        except (ValidationError, MismatchError) as exc:
            self.hard_failure = str(exc)
            return web.json_response({"error": {"message": str(exc)}}, status=409)
        except Exception as exc:  # noqa: BLE001 - surfaced as a run-level failure
            self.hard_failure = f"LLM store failure: {exc}"
            return web.json_response({"error": {"message": self.hard_failure}}, status=502)

    # ---- record --------------------------------------------------------------

    async def _record(
        self,
        request: web.Request,
        identity: RequestIdentity,
        body: dict[str, Any],
        api: str,
    ) -> web.StreamResponse:
        assert self.client is not None
        attempt_id = f"llm-{secrets.token_hex(12)}"
        key = identity.key
        sequence = self._record_sequence.get(key, 0)
        self._record_sequence[key] = sequence + 1

        upstream_body = copy.deepcopy(body)
        # Ask the engine for its token IDs so this bundle is usable by the forced
        # decoding lane later without re-recording. They are stripped again before
        # the framework sees the response.
        if api == "chat.completions":
            upstream_body["return_token_ids"] = True
        else:
            upstream_body["enable_response_messages"] = True
        # Capture mode: the plugin observes (never alters) which sampler steps the
        # engine actually committed. Without that window a later forced replay has
        # no way to line its tokens up with the engine's own step counter.
        if self.force_secret is not None and self.audit is not None:
            request_id = replay_audit_request_id(self.audit_namespace, attempt_id)
            xargs = dict(upstream_body.get("vllm_xargs") or {})
            xargs["native_replay_capture_id"] = request_id
            xargs["native_replay_capture_signature"] = sign_capture(self.force_secret, request_id)
            upstream_body["vllm_xargs"] = xargs

        url = f"{self._upstream(identity)}/v1/" + (
            "chat/completions" if api == "chat.completions" else "responses"
        )
        started = monotonic_ns()
        streaming = bool(body.get("stream"))
        self._inflight[attempt_id] = {
            "cutoff_truncated": True,
            "attempt_id": attempt_id,
            "span_id": f"span-{secrets.token_hex(12)}",
            "actor_id": identity.actor_id,
            "session_id": identity.session_id,
            "role": identity.role,
            "target_id": identity.target_id,
            "api": api,
            "sequence": sequence,
            "request": body,
            "request_sha256": sha256_json(body),
            "request_shape_sha256": sha256_json(request_shape(body)),
            "started_at_ns": started,
        }

        async with self.client.post(url, json=upstream_body) as upstream:
            if streaming:
                return await self._record_stream(
                    request, upstream, identity, body, api, attempt_id, sequence, started
                )
            payload = await upstream.json()
            ended = monotonic_ns()
            tokens = collect_token_fields(payload)
            clean = strip_token_fields(payload)
            engine = await self._capture_engine(attempt_id, tokens["response_token_ids"])
            self._write_attempt(
                engine=engine,
                attempt_id=attempt_id,
                identity=identity,
                sequence=sequence,
                api=api,
                request_body=body,
                response=clean,
                stream=False,
                status_code=upstream.status,
                tokens=tokens,
                started=started,
                ended=ended,
            )
            self._index_model_calls(attempt_id, clean)
            response = web.json_response(clean, status=upstream.status)
            response.headers["X-Native-Replay-Attempt"] = attempt_id
            return response

    async def _record_stream(
        self,
        request: web.Request,
        upstream: aiohttp.ClientResponse,
        identity: RequestIdentity,
        body: dict[str, Any],
        api: str,
        attempt_id: str,
        sequence: int,
        started: int,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=upstream.status,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Native-Replay-Attempt": attempt_id,
            },
        )
        await response.prepare(request)
        decoder = SSEDecoder()
        chunks: list[dict[str, Any]] = []
        tokens: dict[str, list[int]] = {"prompt_token_ids": [], "response_token_ids": []}
        async for data in upstream.content.iter_any():
            for event in decoder.feed(data):
                payload = sse_payload(event)
                if payload is None:
                    continue
                if payload == "[DONE]":
                    chunks.append({"done": True})
                    await response.write(encode_sse("[DONE]"))
                    continue
                parsed = json.loads(payload)
                found = collect_token_fields(parsed)
                tokens["response_token_ids"].extend(found["response_token_ids"])
                if found["prompt_token_ids"] and not tokens["prompt_token_ids"]:
                    tokens["prompt_token_ids"] = found["prompt_token_ids"]
                clean = strip_token_fields(parsed)
                chunks.append({"done": False, "payload": clean})
                await response.write(encode_sse(json.dumps(clean, separators=(",", ":"))))
        decoder.finish()
        await response.write_eof()
        ended = monotonic_ns()
        self._write_attempt(
            attempt_id=attempt_id,
            identity=identity,
            sequence=sequence,
            api=api,
            request_body=body,
            response={"chunks": chunks},
            stream=True,
            status_code=upstream.status,
            tokens=tokens,
            started=started,
            ended=ended,
        )
        for chunk in chunks:
            if not chunk["done"]:
                self._index_model_calls(attempt_id, chunk["payload"])
        return response

    async def _capture_engine(self, attempt_id: str, committed: list[int]) -> dict[str, Any] | None:
        """Read back what the engine actually sampled for this call.

        Returned as evidence, and as the step window a forced replay will need. A
        recording made without a forced-capable engine simply has none, and `--mode
        full` then refuses that bundle rather than guessing.
        """

        if self.audit is None or self.force_secret is None or not committed:
            return None
        request_id = replay_audit_request_id(self.audit_namespace, attempt_id)
        try:
            record = await self.audit.wait_capture(request_id)
        except MismatchError as exc:
            raise MismatchError(f"capture audit for {attempt_id} did not arrive: {exc}") from exc
        return engine_evidence(record, committed)

    def _finish_inflight(self, attempt_id: str) -> None:
        self._inflight.pop(attempt_id, None)

    def _write_attempt(
        self,
        *,
        engine: dict[str, Any] | None = None,
        attempt_id: str,
        identity: RequestIdentity,
        sequence: int,
        api: str,
        request_body: dict[str, Any],
        response: Any,
        stream: bool,
        status_code: int,
        tokens: dict[str, list[int]],
        started: int,
        ended: int,
    ) -> None:
        self._finish_inflight(attempt_id)
        if self.source_cutoff_at_ns is not None and ended > self.source_cutoff_at_ns:
            return
        span_id = f"span-{secrets.token_hex(12)}"
        append_jsonl(
            self.stage_dir / "llm.jsonl",
            {
                "schema_version": LLM_SCHEMA,
                "attempt_id": attempt_id,
                "span_id": span_id,
                "actor_id": identity.actor_id,
                "session_id": identity.session_id,
                "role": identity.role,
                "target_id": identity.target_id,
                "sequence": sequence,
                "api": api,
                "request": request_body,
                # canonical_json sorts mapping keys when the JSONL is persisted.
                # That is normally semantically harmless, but chat templates may
                # iterate a tool's JSON Schema in insertion order, making the key
                # order part of the engine prompt tokens. Keep one opaque encoding
                # of the parsed request so full replay can reconstruct the exact
                # mapping order that recording forwarded upstream.
                "request_ordered_json": json.dumps(
                    request_body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "request_sha256": sha256_json(request_body),
                "request_shape_sha256": sha256_json(request_shape(request_body)),
                "response": response,
                "stream": stream,
                "status_code": status_code,
                "prompt_token_ids": tokens["prompt_token_ids"],
                "response_token_ids": tokens["response_token_ids"],
                "engine": engine,
                "started_at_ns": started,
                "ended_at_ns": ended,
            },
        )
        append_jsonl(
            self.stage_dir / "spans.jsonl",
            {
                "schema_version": SPAN_SCHEMA,
                "span_id": span_id,
                "parent_span_id": identity.parent_span_id,
                "actor_id": identity.actor_id,
                "kind": "llm",
                "name": f"llm:{identity.role}",
                "status": "ok" if status_code < 400 else "error",
                "started_at_ns": started,
                "ended_at_ns": ended,
            },
        )
        self._consumed += 1

    # ---- replay --------------------------------------------------------------

    async def _replay(
        self,
        request: web.Request,
        identity: RequestIdentity,
        api: str,
        expected: dict[str, Any],
    ) -> web.StreamResponse:
        started = monotonic_ns()
        if self.replay_mode == "full":
            await self._run_upstream(identity, api, expected)
        self._index_model_calls(str(expected["attempt_id"]), expected["response"])
        self._write_replay_attempt(expected, identity, started, monotonic_ns())
        if expected["stream"]:
            response = await self._replay_stream(request, expected)
        else:
            response = web.json_response(expected["response"], status=expected["status_code"])
            response.headers["X-Native-Replay-Attempt"] = str(expected["attempt_id"])
            # aiohttp normally writes a plain Response after the handler returns.
            # Do it here so delivery, not handler construction, is the completion
            # barrier observed by the supervisor.
            await response.prepare(request)
            await response.write_eof()
        attempt_id = str(expected["attempt_id"])
        self._delivered.add(attempt_id)
        self._consumed += 1
        return response

    async def _replay_stream(
        self,
        request: web.Request,
        expected: dict[str, Any],
    ) -> web.StreamResponse:
        """Re-emit the recorded events with their recorded boundaries.

        The engine may split the same token sequence differently on every run, and a
        framework's incremental parser can take a different path when it does, so the
        recorded chunking is replayed rather than re-derived.
        """

        response = web.StreamResponse(
            status=expected["status_code"],
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Native-Replay-Attempt": str(expected["attempt_id"]),
            },
        )
        await response.prepare(request)
        for chunk in expected["response"]["chunks"]:
            if chunk.get("done"):
                await response.write(encode_sse("[DONE]"))
                continue
            await response.write(encode_sse(json.dumps(chunk["payload"], separators=(",", ":"))))
        await response.write_eof()
        return response

    async def _run_upstream(
        self,
        identity: RequestIdentity,
        api: str,
        expected: dict[str, Any],
    ) -> None:
        """Make vLLM really execute this recorded request, committing recorded tokens.

        The engine performs the whole forward pass and the sampling kernel; only the
        integer committed at each in-window step is replaced, after sampling. The
        audit read-back is what proves it actually happened — without it a run could
        report GPU work it never did.
        """

        if self.force_secret is None or self.audit is None:
            raise MismatchError(
                "--mode full needs a forced-capable vLLM. Start one with "
                "`minireplay vllm-up` and set config.serving."
            )
        assert self.client is not None
        attempt_id = str(expected["attempt_id"])
        body = forced_upstream_body(
            expected["request"],
            expected,
            api=api,
            secret=self.force_secret,
            audit_namespace=self.audit_namespace,
        )
        url = f"{self._upstream(identity)}/v1/" + (
            "chat/completions" if api == "chat.completions" else "responses"
        )
        async with self.client.post(url, json=body) as upstream:
            payload = await upstream.json()
        if upstream.status != int(expected.get("status_code", 200)):
            raise MismatchError(
                f"forced replay of {attempt_id} returned HTTP {upstream.status}, "
                f"the recording saw {expected.get('status_code')}"
            )
        committed = list(expected["response_token_ids"])
        request_id = replay_audit_request_id(self.audit_namespace, attempt_id)
        record = await self.audit.wait(
            request_id,
            token_ids=committed,
            prompt_token_ids=list(expected["prompt_token_ids"]),
            committed_sample_start=expected["engine"]["committed_sample_start"],
            sampled_token_count=expected["engine"]["sampled_token_count"],
        )
        observed = engine_evidence(record, committed)
        if observed != expected["engine"]:
            raise MismatchError(
                f"engine step drift for {attempt_id}: "
                f"expected={expected['engine']} actual={observed}"
            )
        # The engine's own response is not handed to the framework: the recorded
        # bytes are, because chunk boundaries and provider IDs are not reproducible.
        del payload

    def _write_replay_attempt(
        self,
        expected: dict[str, Any],
        identity: RequestIdentity,
        started: int,
        ended: int,
    ) -> None:
        """Record that this slot was served, and how long serving it took.

        Replay keeps the recorded identity but this run's timings, so the ledger
        supports the same count and timeline checks as a recording. In tool-only
        mode the duration is near zero by construction; in full mode it is the real
        engine time.
        """

        record = dict(expected)
        record.update(
            {
                "started_at_ns": started,
                "ended_at_ns": ended,
                "replay_mode": self.replay_mode,
            }
        )
        append_jsonl(self.stage_dir / "llm.jsonl", record)
        append_jsonl(
            self.stage_dir / "spans.jsonl",
            {
                "schema_version": SPAN_SCHEMA,
                "span_id": str(expected["span_id"]),
                "parent_span_id": identity.parent_span_id,
                "actor_id": str(expected["actor_id"]),
                "kind": "llm",
                "name": f"llm:{expected['role']}",
                "status": "ok",
                "started_at_ns": started,
                "ended_at_ns": ended,
            },
        )
        self._completed.add(str(expected["attempt_id"]))

    async def _hold_past_window(self) -> web.StreamResponse:
        """Never respond. See errors.WorkloadComplete."""

        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def _hold_truncated(self, expected: dict[str, Any]) -> web.StreamResponse:
        attempt_id = str(expected["attempt_id"])
        self._truncated_started[attempt_id] = monotonic_ns()
        append_jsonl(
            self.stage_dir / "cutoff-tail-runtime.jsonl",
            {
                "schema_version": "minireplay.cutoff-tail-runtime/v1",
                "kind": "llm",
                "record_id": attempt_id,
                "actor_id": expected["actor_id"],
                "started_at_ns": monotonic_ns(),
            },
        )
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    # ---- cutoff and accounting ----------------------------------------------

    def freeze_source_cutoff(self, cutoff_at_ns: int) -> list[dict[str, Any]]:
        """Snapshot every request still awaiting the engine when the window closed.

        These are not failures. The source was cut mid-call; the bundle keeps them
        as diagnostics, while replay ends at the last closed call before them.
        """

        self.source_cutoff_at_ns = cutoff_at_ns
        tails: list[dict[str, Any]] = []
        for attempt_id, entry in list(self._inflight.items()):
            tail = dict(entry)
            tail["elapsed_ns"] = max(0, cutoff_at_ns - int(entry["started_at_ns"]))
            tails.append(tail)
            self._truncated.add(attempt_id)
            self._truncated_elapsed[attempt_id] = tail["elapsed_ns"]
        return tails

    def expected_complete(self) -> bool:
        if self.mode != "replay":
            return False
        for queue in self._expected.values():
            for record in queue:
                if str(record["attempt_id"]) not in self._delivered:
                    return False
        now = monotonic_ns()
        for attempt_id, required in self._truncated_elapsed.items():
            started = self._truncated_started.get(attempt_id)
            if started is None or (now - started) < required:
                return False
        return True

    def actor_expected_complete(self, actor_id: str) -> bool:
        """Whether one actor has consumed its LLM prefix and timed cutoff tails."""

        if self.mode != "replay":
            return False
        now = monotonic_ns()
        for key, queue in self._expected.items():
            if key[0] != actor_id:
                continue
            for record in queue:
                attempt_id = str(record["attempt_id"])
                if attempt_id not in self._truncated:
                    if attempt_id not in self._delivered:
                        return False
                    continue
                started = self._truncated_started.get(attempt_id)
                required = self._truncated_elapsed[attempt_id]
                if started is None or (now - started) < required:
                    return False
        return True

    def actor_expected_prefix_consumed(self, actor_id: str) -> bool:
        """Whether an actor entered its complete LLM prefix, including tails."""

        if self.mode != "replay":
            return False
        for key, queue in self._expected.items():
            if key[0] != actor_id:
                continue
            if self._replay_sequence.get(key, 0) < len(queue):
                return False
            for record in queue:
                attempt_id = str(record["attempt_id"])
                if attempt_id in self._truncated and attempt_id not in self._truncated_started:
                    return False
        return True

    def outstanding(self) -> dict[str, Any]:
        missing = {
            f"{key}": sum(
                str(record["attempt_id"]) not in self._delivered for record in queue
            )
            for key, queue in self._expected.items()
            if any(str(record["attempt_id"]) not in self._delivered for record in queue)
        }
        claimed_not_delivered = sorted(
            str(record["attempt_id"])
            for key, queue in self._expected.items()
            for record in queue[: self._replay_sequence.get(key, 0)]
            if str(record["attempt_id"]) not in self._delivered
        )
        return {
            "missing_llm": missing,
            "claimed_not_delivered": claimed_not_delivered,
            "evidence_not_delivered": sorted(self._completed - self._delivered),
            "consumed": self._consumed,
        }

    def assert_consumed(self) -> None:
        report = self.outstanding()
        if report["missing_llm"]:
            raise MismatchError(f"missing LLM attempts: {report['missing_llm']}")

    def application(self) -> web.Application:
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.router.add_post("/v1/chat/completions", self.handle_chat)
        app.router.add_post("/v1/responses", self.handle_responses)
        app.router.add_get("/v1/models", self.handle_models)
        return app


_SHAPE_STRUCTURAL = frozenset({"name", "role", "type"})


def request_shape(body: dict[str, Any]) -> dict[str, Any]:
    """A request's structure, with free text reduced to a size class.

    Prompts legitimately embed this run's paths and IDs, so comparing them verbatim
    would reject a correct replay. What must match is the shape: same message roles,
    same tool schemas, same sampling configuration, and free text of the same rough
    size. Byte-level prompt identity is established later by the forced-decoding
    lane, which compares the engine's own prompt token IDs.
    """

    payload_keys = {"messages", "input", "prompt"}
    return {
        "payload": {k: _shape(v) for k, v in body.items() if k in payload_keys},
        "configuration": {k: v for k, v in body.items() if k not in payload_keys},
    }


def _size_bucket(value: str) -> str:
    size = len(value)
    if size == 0:
        return "0"
    bound = 16
    while size > bound:
        bound *= 4
    return f"1-{bound}"


def _inline_data_shape(value: str) -> dict[str, str]:
    """Project an inline `data:` payload to its media type, with no size class.

    Owl drives a real Chromium and sends the live viewport to its VL endpoint, so
    these bytes are a fresh screenshot every run and their length is not
    reproducible — design §5 calls exactly that a browser internal. Keeping the
    media type keeps what is structural (an image is present here, and of what
    kind); keeping the length would make a replay valid only when a live page
    happened to render to a similar size.
    """

    header = value.split(",", 1)[0]
    media_type = header[len("data:") :].split(";", 1)[0] or "application/octet-stream"
    return {"kind": "inline-data", "media_type": media_type}


def _shape(value: Any, field: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _shape(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(item, field) for item in value]
    if isinstance(value, str) and field not in _SHAPE_STRUCTURAL:
        if value.startswith("data:"):
            return _inline_data_shape(value)
        return {"kind": "text", "size_bucket": _size_bucket(value)}
    return value
