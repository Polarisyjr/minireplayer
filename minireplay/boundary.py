"""The native operation boundary.

Every dispatch, tool, grader and shared-artifact operation crosses this ledger over
HTTP, so a framework process, a container and a Node plugin all speak the same
protocol.

Two rules shape the design:

* **Claim on entry, never on exit.** A slot is claimed when an operation starts,
  against a per-actor FIFO queue. Completion carries no comparison at all: by then
  the native work has already run, and its outcome is diagnostic evidence, not a
  gate. Ordering is therefore enforced without ever comparing completion times.

* **Cheap claims.** The claim compares precomputed digests, not structures. The
  structural validation ran once at bundle load. The comparison also happens before
  the operation's own clock starts, so it does not enter the measured duration.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from .constants import (
    ARTIFACT_SCHEMA,
    DISPATCH_SCHEMA,
    GRADER_SCHEMA,
    LEDGER_FILES,
    MAX_REQUEST_BYTES,
    SPAN_SCHEMA,
    TOOL_SCHEMA,
)
from .contracts import bind_typed_fields, run_path_map, to_physical
from .errors import MismatchError, ValidationError, WorkloadComplete
from .observation import project_result, validate_result_contract
from .util import append_jsonl, atomic_write_json, monotonic_ns, require, sha256_json

_ID_PREFIX = {"dispatch": "dispatch", "tool": "tool", "grader": "grader", "artifact": "artifact"}
_ID_FIELD = {
    "dispatch": "dispatch_id",
    "tool": "call_id",
    "grader": "attempt_id",
    "artifact": "event_id",
}
_SCHEMA = {
    "dispatch": DISPATCH_SCHEMA,
    "tool": TOOL_SCHEMA,
    "grader": GRADER_SCHEMA,
    "artifact": ARTIFACT_SCHEMA,
}


@dataclass
class Reservation:
    kind: str
    record_id: str
    span_id: str
    actor_id: str
    lane: str | None
    started_at_ns: int
    request: dict[str, Any]
    expected: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def claim_identity(kind: str, record: dict[str, Any]) -> tuple:
    """The small tuple a claim compares.

    Everything in it is either already a digest or a short interned string, so the
    comparison is O(1) regardless of how large the operation's arguments are.
    """

    if kind in {"dispatch", "tool"}:
        return (record["name"], record["arguments_sha256"])
    if kind == "grader":
        return (record["grader_kind"], record["implementation"])
    if kind == "artifact":
        return (record["logical_path"], record["operation"], record["version"])
    raise MismatchError(f"unknown boundary kind: {kind!r}")


def _issue_order(record: dict[str, Any]) -> int:
    """When the framework began this operation.

    Closed records carry `started_at_ns`; a cutoff tail never closed, so it carries
    the clock it began on as `source_started_at_ns`.
    """

    started = record.get("started_at_ns")
    if isinstance(started, int):
        return started
    source = record.get("source_started_at_ns")
    if isinstance(source, int):
        return source
    raise ValidationError(f"record has no start time to order by: {record.get('record_id')}")


class BoundaryLedger:
    def __init__(
        self,
        *,
        mode: str,
        stage_dir: Path,
        auth_token: str,
        adapter: str,
        run_root: Path,
        repo: Path,
        bundle: Any | None = None,
        llm_index: Any | None = None,
        fast_claim: bool = False,
    ) -> None:
        require(mode in {"record", "replay"}, f"invalid boundary mode: {mode!r}")
        self.mode = mode
        self.stage_dir = stage_dir
        self.auth_token = auth_token
        self.adapter = adapter
        self.run_root = run_root
        self.repo = repo
        self.path_map = run_path_map(run_root, repo)
        self.bundle = bundle
        self.llm_index = llm_index
        self.fast_claim = fast_claim
        self.hard_failure: str | None = None
        # Set by ReplayServices; see LLMStore.run_complete.
        self.run_complete: Any = None
        # A concurrent actor can reach its own recorded cutoff before the other
        # actors do. ReplayServices supplies the cross-ledger predicate so that
        # only actors which were still live at the source cutoff may stop here.
        self.actor_complete: Any = None
        # Non-zero while a handler is between its ledger append and its response, so
        # the supervisor can let writes drain before it reads the files.
        self.active_writes = 0

        self.active: dict[str, Reservation] = {}
        self._expected: dict[tuple, list[dict[str, Any]]] = {}
        self._cursor: dict[tuple, int] = {}
        self._completed: set[tuple[str, str]] = set()
        self._lane_of_record: dict[str, str | None] = {}

        self._truncated: set[tuple[str, str]] = set()
        self._truncated_elapsed: dict[tuple[str, str], int] = {}
        self._truncated_started: dict[tuple[str, str], int] = {}
        self.source_cutoff_at_ns: int | None = None

        if bundle is not None:
            self._load_expected(bundle)

    # ---- expectation loading -------------------------------------------------

    def _load_expected(self, bundle: Any) -> None:
        for kind in ("dispatch", "tool", "grader", "artifact"):
            for record in bundle.records(kind):
                lane = self._record_lane(kind, record)
                key = (kind, str(record["actor_id"]), lane)
                self._expected.setdefault(key, []).append(record)
        # Cutoff tails remain in the bundle as source diagnostics but are excluded
        # from replay. Only the closed causal prefix is claimable.
        for queue in self._expected.values():
            queue.sort(key=_issue_order)

    def _record_lane(self, kind: str, record: dict[str, Any]) -> str | None:
        """CORAL runs several concurrent OpenCode sessions inside one actor.

        Those sessions are independent causal chains, so they get their own queues;
        ordering between them is natural concurrency and must not be constrained.
        Every other adapter keeps one queue per actor.
        """

        if self.adapter != "coral" or kind not in {"dispatch", "tool"}:
            return None
        if kind == "dispatch":
            session = record.get("session_id")
            return str(session) if isinstance(session, str) and session else None
        return self._lane_of_record.get(str(record["dispatch_id"]))

    def _payload_lane(self, kind: str, payload: dict[str, Any]) -> str | None:
        if self.adapter != "coral" or kind not in {"dispatch", "tool"}:
            return None
        if kind == "dispatch":
            session = payload.get("session_id")
            return str(session) if isinstance(session, str) and session else None
        return self._lane_of_record.get(str(payload.get("dispatch_id")))

    # ---- claim ---------------------------------------------------------------

    def _claim(self, kind: str, actor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        lane = self._payload_lane(kind, payload)
        key = (kind, actor_id, lane)
        queue = self._expected.get(key, [])
        cursor = self._cursor.get(key, 0)
        if cursor >= len(queue):
            whole_run_complete = self.run_complete is not None and self.run_complete()
            actor_complete = self.actor_complete is not None and self.actor_complete(actor_id)
            if whole_run_complete or actor_complete:
                raise WorkloadComplete(
                    f"native {kind} for actor {actor_id} arrived after the recorded window closed"
                )
            raise MismatchError(
                f"unexpected native {kind} for actor {actor_id}"
                f"{f' lane {lane}' if lane else ''}: "
                f"the recording holds {len(queue)} and all are consumed"
            )
        expected = queue[cursor]
        if not self.fast_claim:
            observed = claim_identity(kind, self._claimable(kind, payload))
            recorded = claim_identity(kind, expected)
            if observed != recorded:
                raise MismatchError(
                    f"native {kind} invocation drift for actor {actor_id} "
                    f"lane {lane!r} at position {cursor}: "
                    f"expected={recorded!r} actual={observed!r}"
                )
        self._cursor[key] = cursor + 1
        return expected

    def _claimable(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        value = dict(payload)
        if kind in {"dispatch", "tool"}:
            value["arguments_sha256"] = sha256_json(self._logical(payload.get("arguments", {})))
        return value

    def _logical(self, arguments: Any) -> Any:
        """Reduce this run's directories to run-independent names."""

        return bind_typed_fields(self.adapter, arguments, self.path_map)

    # ---- start ---------------------------------------------------------------

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = payload.get("kind")
        require(kind in _ID_FIELD, f"invalid boundary kind: {kind!r}")
        actor_id = payload.get("actor_id")
        require(isinstance(actor_id, str) and bool(actor_id), "boundary actor_id is required")
        started = payload.get("started_at_ns", monotonic_ns())
        require(isinstance(started, int) and started >= 0, "invalid boundary start time")

        if kind == "dispatch":
            payload = self._resolve_trigger(payload)
        elif kind == "tool":
            validate_result_contract(payload.get("result_contract"))

        expected = self._claim(kind, actor_id, payload) if self.mode == "replay" else None
        if expected is not None:
            record_id = self._identity_of(str(kind), expected)[1]
            span_id = str(expected["span_id"])
        else:
            hint = payload.get("record_id_hint")
            record_id = (
                str(hint)
                if isinstance(hint, str) and hint
                else (f"{_ID_PREFIX[kind]}-{secrets.token_hex(12)}")
            )
            span_hint = payload.get("span_id_hint")
            span_id = (
                str(span_hint)
                if isinstance(span_hint, str) and span_hint
                else f"span-{secrets.token_hex(12)}"
            )

        lane = self._payload_lane(kind, payload)
        if kind == "dispatch":
            self._lane_of_record[record_id] = lane

        reservation_id = secrets.token_urlsafe(24)
        self.active[reservation_id] = Reservation(
            kind=str(kind),
            record_id=record_id,
            span_id=span_id,
            actor_id=actor_id,
            lane=lane,
            started_at_ns=started,
            request=payload,
            expected=expected,
        )
        identity = (str(kind), record_id)
        if identity in self._truncated:
            # Timed on the ledger's own clock, not the caller's. A tail must run for
            # the duration the source observed, and only a single clock can decide
            # that; a client's reported start comes from a different process.
            self._truncated_started[identity] = monotonic_ns()

        response = {"reservation_id": reservation_id, "record_id": record_id, "span_id": span_id}
        if expected is not None and kind == "dispatch":
            # The recorded arguments carry logical directory names. Expand them into
            # this run's directories so the native call is made with the identity the
            # recording proved, but against the workspace that actually exists now.
            response["execution_arguments"] = to_physical(
                self.adapter, expected["arguments"], self.run_root, self.repo
            )
        return response

    def _resolve_trigger(self, payload: dict[str, Any]) -> dict[str, Any]:
        origin = payload.get("origin")
        require(isinstance(origin, dict), "dispatch origin must be an object")
        origin = dict(origin)
        trigger = origin.get("trigger_id")
        if trigger == "auto" and self.llm_index is not None:
            model_call_id = origin.get("model_call_id")
            resolved = self.llm_index.attempt_for_model_call(model_call_id)
            if resolved is None:
                raise MismatchError(
                    f"dispatch names model call {model_call_id!r}, which no LLM attempt produced"
                )
            origin["trigger_id"] = resolved
        return {**payload, "origin": origin}

    # ---- complete ------------------------------------------------------------

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        reservation_id = payload.get("reservation_id")
        require(isinstance(reservation_id, str), "completion reservation_id is required")
        reservation = self.active.pop(reservation_id, None)
        require(reservation is not None, "unknown or already completed reservation")
        assert reservation is not None
        ended = payload.get("ended_at_ns", monotonic_ns())
        require(
            isinstance(ended, int) and ended >= reservation.started_at_ns,
            "invalid boundary end time",
        )

        identity = (reservation.kind, reservation.record_id)
        if identity in self._truncated:
            return self._hold_truncated(reservation, payload, ended)

        record = self._build_record(reservation, payload, ended)

        # Finished after the window closed: not part of the closed prefix, so it is
        # dropped rather than recorded.
        if (
            self.mode == "record"
            and self.source_cutoff_at_ns is not None
            and ended > self.source_cutoff_at_ns
        ):
            return self._completion_response(reservation, dropped=True)

        append_jsonl(self.stage_dir / LEDGER_FILES[reservation.kind], record)
        self._write_span(reservation, record, ended)
        self._completed.add(identity)
        return self._completion_response(reservation, dropped=False)

    def _completion_response(self, reservation: Reservation, *, dropped: bool) -> dict[str, Any]:
        response: dict[str, Any] = {"valid": True}
        if dropped:
            response["discarded_at_cutoff"] = True
        if reservation.kind != "tool":
            return response
        expected = reservation.expected
        if expected is None or dropped:
            response["result_replay_required"] = False
            return response
        response["framework_result"] = expected["result"]
        response["result_replay_required"] = True
        if expected.get("exception_raised") is True:
            # The native tool has already run. The recording's exception is what the
            # framework saw, and therefore what fixed its next control-flow decision,
            # so it is restored regardless of how this run's call actually ended.
            response["framework_exception"] = expected["result"]
        return response

    def _build_record(
        self,
        reservation: Reservation,
        payload: dict[str, Any],
        ended: int,
    ) -> dict[str, Any]:
        request = reservation.request
        common = {
            "schema_version": _SCHEMA[reservation.kind],
            _ID_FIELD[reservation.kind]: reservation.record_id,
            "span_id": reservation.span_id,
            "actor_id": reservation.actor_id,
            "process_role": request.get("process_role"),
            "started_at_ns": reservation.started_at_ns,
            "ended_at_ns": ended,
        }
        builder = {
            "dispatch": self._dispatch_record,
            "tool": self._tool_record,
            "grader": self._grader_record,
            "artifact": self._artifact_record,
        }[reservation.kind]
        return builder(reservation, payload, common)

    def _dispatch_record(
        self,
        reservation: Reservation,
        payload: dict[str, Any],
        common: dict[str, Any],
    ) -> dict[str, Any]:
        request = reservation.request
        native_arguments = request.get("arguments", {})
        arguments = self._logical(native_arguments)
        return {
            **common,
            "session_id": request.get("session_id"),
            "parser_identity": request.get("parser_identity"),
            "dispatcher_identity": request.get("dispatcher_identity"),
            "native_call_id": request.get("native_call_id"),
            "name": request.get("name"),
            "arguments": arguments,
            "arguments_sha256": sha256_json(arguments),
            "native_arguments": native_arguments,
            "origin": request.get("origin"),
            "status": payload.get("status"),
            "execution_call_id": payload.get("execution_call_id"),
        }

    def _tool_record(
        self,
        reservation: Reservation,
        payload: dict[str, Any],
        common: dict[str, Any],
    ) -> dict[str, Any]:
        request = reservation.request
        native_arguments = request.get("arguments", {})
        arguments = self._logical(native_arguments)
        contract = request["result_contract"]
        native_result = payload.get("result")
        projection = project_result(native_result, contract)
        expected = reservation.expected
        # Recording keeps what the framework saw; replay keeps the recorded value in
        # `result` and this run's real value in `native_result`, so one reader serves
        # both and the evidence stays honest about which is which.
        framework_result = native_result if expected is None else expected["result"]
        status = payload.get("status", "ok")
        exception_raised = bool(payload.get("exception_raised", False))
        if expected is not None and expected.get("exception_raised") is True:
            status = expected.get("status", "error")
            exception_raised = True
        return {
            **common,
            "dispatch_id": request.get("dispatch_id"),
            "name": request.get("name"),
            "implementation": request.get("implementation"),
            "arguments": arguments,
            "arguments_sha256": sha256_json(arguments),
            "native_arguments": native_arguments,
            "result_contract": contract,
            "semantic_timeout_s": request.get("semantic_timeout_s"),
            "result": framework_result,
            "native_result": native_result,
            "native_observations": projection.evidence,
            "status": status,
            "exception_raised": exception_raised,
            "native_execution": payload.get("native_execution", True),
            "cpu_seconds": payload.get("cpu_seconds", 0.0),
            "child_processes": payload.get("child_processes", []),
        }

    def _grader_record(
        self,
        reservation: Reservation,
        payload: dict[str, Any],
        common: dict[str, Any],
    ) -> dict[str, Any]:
        request = reservation.request
        return {
            **common,
            "implementation": request.get("implementation"),
            "grader_kind": request.get("grader_kind"),
            "trigger_id": request.get("trigger_id"),
            "status": payload.get("status", "ok"),
            "result": payload.get("result"),
            "llm_attempt_ids": payload.get("llm_attempt_ids", []),
            "tool_call_ids": payload.get("tool_call_ids", []),
            "artifact_versions": payload.get("artifact_versions", []),
            "cpu_seconds": payload.get("cpu_seconds", 0.0),
        }

    def _artifact_record(
        self,
        reservation: Reservation,
        payload: dict[str, Any],
        common: dict[str, Any],
    ) -> dict[str, Any]:
        request = reservation.request
        record = {
            **common,
            "logical_path": request.get("logical_path"),
            "physical_path": payload.get("physical_path"),
            "operation": request.get("operation"),
            "version": request.get("version"),
            "bytes_sha256": payload.get("bytes_sha256"),
            "size": payload.get("size"),
            "mode": payload.get("mode"),
            "triggered_by": payload.get("triggered_by", []),
            "read_from": payload.get("read_from"),
            "completed_at_ns": common["ended_at_ns"],
            "native_execution": payload.get("native_execution", True),
        }
        return record

    def _write_span(
        self,
        reservation: Reservation,
        record: dict[str, Any],
        ended: int,
    ) -> None:
        append_jsonl(
            self.stage_dir / "spans.jsonl",
            {
                "schema_version": SPAN_SCHEMA,
                "span_id": reservation.span_id,
                "parent_span_id": reservation.request.get("parent_span_id"),
                "actor_id": reservation.actor_id,
                "kind": reservation.kind,
                "name": reservation.request.get("name") or reservation.kind,
                "status": record.get("status", "ok"),
                "started_at_ns": reservation.started_at_ns,
                "ended_at_ns": ended,
            },
        )

    # ---- cutoff --------------------------------------------------------------

    def _hold_truncated(
        self,
        reservation: Reservation,
        payload: dict[str, Any],
        ended: int,
    ) -> dict[str, Any]:
        """A tail the source never finished.

        The native work ran for real and is measured, but the framework must never
        observe a result the source did not produce, so this reservation's HTTP
        response is withheld permanently. Withholding it is also what prevents the
        framework from refilling a task that does not exist in the recording.
        """

        append_jsonl(
            self.stage_dir / "cutoff-tail-runtime.jsonl",
            {
                "schema_version": "minireplay.cutoff-tail-runtime/v1",
                "kind": reservation.kind,
                "record_id": reservation.record_id,
                "actor_id": reservation.actor_id,
                "started_at_ns": reservation.started_at_ns,
                "native_completed_at_ns": ended,
                "status": payload.get("status", "ok"),
                "cpu_seconds": payload.get("cpu_seconds", 0.0),
            },
        )
        return {"valid": True, "_hold_for_cutoff": True}

    def freeze_source_cutoff(self, cutoff_at_ns: int) -> list[dict[str, Any]]:
        """Snapshot everything still running when the sweep closed the window."""

        self.source_cutoff_at_ns = cutoff_at_ns
        tails: list[dict[str, Any]] = []
        for reservation in list(self.active.values()):
            identity = (reservation.kind, reservation.record_id)
            self._truncated.add(identity)
            elapsed = max(0, cutoff_at_ns - reservation.started_at_ns)
            self._truncated_elapsed[identity] = elapsed
            arguments = self._logical(reservation.request.get("arguments", {}))
            tails.append(
                {
                    "cutoff_truncated": True,
                    "kind": reservation.kind,
                    _ID_FIELD[reservation.kind]: reservation.record_id,
                    "record_id": reservation.record_id,
                    "span_id": reservation.span_id,
                    "actor_id": reservation.actor_id,
                    "lane": reservation.lane,
                    "name": reservation.request.get("name"),
                    "implementation": reservation.request.get("implementation"),
                    "grader_kind": reservation.request.get("grader_kind"),
                    "logical_path": reservation.request.get("logical_path"),
                    "operation": reservation.request.get("operation"),
                    "version": reservation.request.get("version"),
                    "arguments": arguments,
                    "arguments_sha256": sha256_json(arguments),
                    "result_contract": reservation.request.get("result_contract"),
                    "source_started_at_ns": reservation.started_at_ns,
                    "elapsed_ns": elapsed,
                }
            )
        return tails

    def truncated_progress(self) -> dict[str, Any]:
        now = monotonic_ns()
        entries = []
        for identity, required in sorted(self._truncated_elapsed.items()):
            started = self._truncated_started.get(identity)
            entries.append(
                {
                    "kind": identity[0],
                    "record_id": identity[1],
                    "entered": started is not None,
                    "required_ns": required,
                    "elapsed_ns": (now - started) if started is not None else 0,
                }
            )
        return {"tails": entries}

    # ---- completion accounting ----------------------------------------------

    @staticmethod
    def _identity_of(kind: str, record: dict[str, Any]) -> tuple[str, str]:
        """Cutoff tails carry `record_id`; ledger records carry a per-kind id field."""

        value = record.get(_ID_FIELD[kind], record.get("record_id"))
        return (kind, str(value))

    def expected_complete(self) -> bool:
        """True once every recorded slot is consumed and every tail has served its time."""

        if self.mode != "replay":
            return False
        for key, queue in self._expected.items():
            for record in queue:
                identity = self._identity_of(key[0], record)
                if identity in self._truncated:
                    continue
                if identity not in self._completed:
                    return False
        now = monotonic_ns()
        for identity, required in self._truncated_elapsed.items():
            started = self._truncated_started.get(identity)
            if started is None or (now - started) < required:
                return False
        return True

    def actor_expected_complete(self, actor_id: str) -> bool:
        """Whether one actor has completed its closed operation prefix.

        This deliberately does not decide whether the actor was live at cutoff;
        ReplayServices combines it with the bundle's task-terminal evidence.
        """

        if self.mode != "replay":
            return False
        now = monotonic_ns()
        for key, queue in self._expected.items():
            if key[1] != actor_id:
                continue
            for record in queue:
                identity = self._identity_of(key[0], record)
                if identity in self._truncated:
                    started = self._truncated_started.get(identity)
                    required = self._truncated_elapsed[identity]
                    if started is None or (now - started) < required:
                        return False
                elif identity not in self._completed:
                    return False
        return True

    def actor_expected_prefix_consumed(self, actor_id: str) -> bool:
        """Whether an actor entered every operation in the closed prefix."""

        if self.mode != "replay":
            return False
        for key, queue in self._expected.items():
            if key[1] != actor_id:
                continue
            for record in queue:
                identity = self._identity_of(key[0], record)
                if identity in self._truncated:
                    if self._truncated_started.get(identity) is None:
                        return False
                elif identity not in self._completed:
                    return False
        return True

    def outstanding(self) -> dict[str, Any]:
        missing: list[str] = []
        for key, queue in self._expected.items():
            for record in queue:
                identity = self._identity_of(key[0], record)
                if identity in self._truncated or identity in self._completed:
                    continue
                missing.append(f"{identity[0]}:{identity[1]}")
        return {
            "missing": sorted(missing),
            "unexpected_active": sorted(
                f"{r.kind}:{r.record_id}"
                for r in self.active.values()
                if (r.kind, r.record_id) not in self._truncated
            ),
            "truncated": self.truncated_progress(),
        }

    def assert_consumed(self) -> None:
        report = self.outstanding()
        if report["missing"]:
            raise MismatchError(f"missing native operations: {report['missing']}")
        if report["unexpected_active"]:
            raise MismatchError(
                f"native operations still open at teardown: {report['unexpected_active']}"
            )

    def has_unexpected_active(self) -> bool:
        return any((r.kind, r.record_id) not in self._truncated for r in self.active.values())

    # ---- HTTP ----------------------------------------------------------------

    def _authorize(self, request: web.Request) -> None:
        header = request.headers.get("Authorization", "")
        if not secrets.compare_digest(header, f"Bearer {self.auth_token}"):
            raise web.HTTPUnauthorized(text="invalid boundary token")

    async def handle_start(self, request: web.Request) -> web.Response:
        self._authorize(request)
        payload = await request.json()
        try:
            return web.json_response(self.start(payload))
        except WorkloadComplete:
            # Never responds. See errors.WorkloadComplete.
            await asyncio.Event().wait()
            raise AssertionError("unreachable") from None
        except MismatchError as exc:
            return self._fail(exc, phase="start")

    async def handle_complete(self, request: web.Request) -> web.Response:
        self._authorize(request)
        payload = await request.json()
        self.active_writes += 1
        try:
            result = self.complete(payload)
        except MismatchError as exc:
            return self._fail(exc, phase="complete")
        finally:
            self.active_writes -= 1
        if result.pop("_hold_for_cutoff", False):
            # Never returns. See _hold_truncated.
            await asyncio.Event().wait()
        return web.json_response(result)

    async def handle_status(self, request: web.Request) -> web.Response:
        self._authorize(request)
        return web.json_response(
            {
                "mode": self.mode,
                "active": len(self.active),
                "hard_failure": self.hard_failure,
                **self.outstanding(),
            }
        )

    def _fail(self, exc: Exception, *, phase: str) -> web.Response:
        message = str(exc)
        if self.hard_failure is None:
            self.hard_failure = message
            failure_path = self.stage_dir / "first-failure.json"
            if not failure_path.exists():
                atomic_write_json(
                    failure_path,
                    {
                        "schema_version": "minireplay.failure/v1",
                        "phase": phase,
                        "message": message,
                    },
                )
        return web.json_response({"error": message}, status=409)

    def application(self) -> web.Application:
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.router.add_post("/v1/boundary/start", self.handle_start)
        app.router.add_post("/v1/boundary/complete", self.handle_complete)
        app.router.add_get("/v1/boundary/status", self.handle_status)
        return app
