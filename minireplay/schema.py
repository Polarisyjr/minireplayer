"""Record validation.

Scope is deliberately narrow: validate the fields replay actually consumes, so a
malformed bundle fails at load instead of halfway through a run. Fields that are
only evidence (timings, native results, child receipts) are checked for type but
not for content, because their content is diagnostic by design.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .constants import (
    ARTIFACT_SCHEMA,
    DISPATCH_SCHEMA,
    GRADER_SCHEMA,
    LLM_SCHEMA,
    MANIFEST_SCHEMA,
    SPAN_SCHEMA,
    TERMINAL_SCHEMA,
    TOOL_SCHEMA,
)
from .observation import validate_result_contract
from .util import require, sha256_json

_STATUS = {"ok", "error", "timeout"}


def _string(value: Any, context: str) -> str:
    require(isinstance(value, str) and bool(value), f"{context}: expected a non-empty string")
    return value


def _object(value: Any, context: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{context}: expected an object")
    return value


def _digest(value: Any, context: str) -> str:
    text = _string(value, context)
    require(
        len(text) == 64 and all(c in "0123456789abcdef" for c in text),
        f"{context}: expected a sha256 hex digest",
    )
    return text


def _timespan(value: dict[str, Any], context: str) -> None:
    for field in ("started_at_ns", "ended_at_ns"):
        require(
            isinstance(value.get(field), int) and not isinstance(value[field], bool),
            f"{context}.{field}: expected an integer",
        )
    require(
        value["ended_at_ns"] >= value["started_at_ns"],
        f"{context}: ended before it started",
    )


def validate_span(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == SPAN_SCHEMA, "span: unsupported schema")
    _string(value.get("span_id"), "span.span_id")
    _string(value.get("actor_id"), "span.actor_id")
    _string(value.get("kind"), "span.kind")
    parent = value.get("parent_span_id")
    require(parent is None or isinstance(parent, str), "span.parent_span_id: expected a string")
    _timespan(value, "span")


def validate_dispatch(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == DISPATCH_SCHEMA, "dispatch: unsupported schema")
    for field in ("dispatch_id", "span_id", "actor_id", "name"):
        _string(value.get(field), f"dispatch.{field}")
    _object(value.get("arguments"), "dispatch.arguments")
    require(
        value.get("arguments_sha256") == sha256_json(value["arguments"]),
        f"dispatch {value['dispatch_id']}: arguments digest does not match its arguments",
    )
    origin = _object(value.get("origin"), "dispatch.origin")
    _string(origin.get("kind"), "dispatch.origin.kind")
    require(
        value.get("status") in {"executed", "rejected", "failed-before-entry"},
        "dispatch.status: invalid resolution",
    )
    execution = value.get("execution_call_id")
    require(
        execution is None or isinstance(execution, str),
        "dispatch.execution_call_id: expected a string or null",
    )
    require(
        (value["status"] == "executed") == (execution is not None),
        f"dispatch {value['dispatch_id']}: status and execution_call_id disagree",
    )
    _timespan(value, "dispatch")


def validate_tool(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == TOOL_SCHEMA, "tool: unsupported schema")
    for field in ("call_id", "dispatch_id", "span_id", "actor_id", "name", "implementation"):
        _string(value.get(field), f"tool.{field}")
    _object(value.get("arguments"), "tool.arguments")
    require(
        value.get("arguments_sha256") == sha256_json(value["arguments"]),
        f"tool {value['call_id']}: arguments digest does not match its arguments",
    )
    validate_result_contract(value.get("result_contract"))
    require("result" in value, "tool: framework-visible result is missing")
    require("native_result" in value, "tool: native result evidence is missing")
    require(value.get("status") in _STATUS, "tool.status: invalid status")
    require(
        value.get("native_execution") is True,
        f"tool {value['call_id']}: native execution proof is missing",
    )
    require(
        isinstance(value.get("exception_raised"), bool),
        "tool.exception_raised: expected a boolean",
    )
    _timespan(value, "tool")


def validate_grader(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == GRADER_SCHEMA, "grader: unsupported schema")
    for field in ("attempt_id", "span_id", "actor_id", "implementation", "grader_kind"):
        _string(value.get(field), f"grader.{field}")
    require(value.get("status") in _STATUS, "grader.status: invalid status")
    _timespan(value, "grader")


def validate_artifact(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == ARTIFACT_SCHEMA, "artifact: unsupported schema")
    for field in ("event_id", "logical_path", "actor_id", "span_id"):
        _string(value.get(field), f"artifact.{field}")
    require(
        value.get("operation") in {"create", "write", "read"},
        "artifact.operation: invalid operation",
    )
    require(
        isinstance(value.get("version"), int) and value["version"] >= 1,
        "artifact.version: expected an integer >= 1",
    )
    _digest(value.get("bytes_sha256"), "artifact.bytes_sha256")
    require(
        isinstance(value.get("completed_at_ns"), int)
        and not isinstance(value["completed_at_ns"], bool),
        "artifact.completed_at_ns: expected an integer",
    )
    read_from = value.get("read_from")
    if value["operation"] == "read":
        _string(read_from, "artifact.read_from")
    else:
        require(read_from is None, "artifact: a producer event cannot have read_from")


def validate_llm(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == LLM_SCHEMA, "llm: unsupported schema")
    for field in ("attempt_id", "span_id", "actor_id", "session_id", "role", "target_id", "api"):
        _string(value.get(field), f"llm.{field}")
    require(
        isinstance(value.get("sequence"), int) and value["sequence"] >= 0,
        "llm.sequence: expected a non-negative integer",
    )
    _object(value.get("request"), "llm.request")
    require(
        value.get("request_sha256") == sha256_json(value["request"]),
        f"llm {value['attempt_id']}: request digest does not match its request",
    )
    _digest(value.get("request_shape_sha256"), "llm.request_shape_sha256")
    require("response" in value, "llm: response is missing")
    require(isinstance(value.get("stream"), bool), "llm.stream: expected a boolean")
    require(
        isinstance(value.get("status_code"), int) and not isinstance(value["status_code"], bool),
        "llm.status_code: expected an integer",
    )
    for field in ("prompt_token_ids", "response_token_ids"):
        tokens = value.get(field)
        require(isinstance(tokens, list), f"llm.{field}: expected a list")
        require(
            all(isinstance(t, int) and not isinstance(t, bool) for t in tokens),
            f"llm.{field}: expected integer token IDs",
        )


def validate_terminal(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == TERMINAL_SCHEMA, "terminal: unsupported schema")
    require(value.get("status") in {"success", "failure"}, "terminal.status: invalid status")
    require(
        isinstance(value.get("task_terminals"), list),
        "terminal.task_terminals: expected a list",
    )


def validate_manifest(value: dict[str, Any]) -> None:
    require(value.get("schema_version") == MANIFEST_SCHEMA, "manifest: unsupported schema")
    require(
        value.get("cutoff_policy", "evidence-only") == "evidence-only",
        "manifest.cutoff_policy: only evidence-only is supported",
    )
    _string(value.get("bundle_id"), "manifest.bundle_id")
    _string(value.get("adapter"), "manifest.adapter")
    _object(value.get("workload"), "manifest.workload")
    _object(value.get("counts"), "manifest.counts")
    actors = value.get("actors")
    require(isinstance(actors, list) and bool(actors), "manifest.actors: expected a non-empty list")
    for actor in actors:
        entry = _object(actor, "manifest.actors[]")
        _string(entry.get("actor_id"), "manifest.actors[].actor_id")
    lanes = value.get("lanes")
    require(isinstance(lanes, list), "manifest.lanes: expected a list")
    require(len(lanes) == len(actors), "manifest.lanes: expected exactly one lane per actor")
    actor_ids = {str(actor["actor_id"]) for actor in actors}
    lane_ids: set[str] = set()
    for lane in lanes:
        entry = _object(lane, "manifest.lanes[]")
        actor_id = _string(entry.get("actor_id"), "manifest.lanes[].actor_id")
        _string(entry.get("path"), "manifest.lanes[].path")
        _object(entry.get("counts"), "manifest.lanes[].counts")
        require(actor_id in actor_ids, f"manifest.lanes: undeclared actor {actor_id!r}")
        require(actor_id not in lane_ids, f"manifest.lanes: duplicate actor {actor_id!r}")
        lane_ids.add(actor_id)
    window = _object(value.get("window"), "manifest.window")
    for field in ("gate_at_ns", "terminal_at_ns"):
        require(
            isinstance(window.get(field), int) and not isinstance(window[field], bool),
            f"manifest.window.{field}: expected an integer",
        )
    require(
        window["terminal_at_ns"] >= window["gate_at_ns"],
        "manifest.window: terminal precedes the gate",
    )


def validate_causal_graph(
    *,
    spans: list[dict[str, Any]],
    dispatches: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """Check that the closed prefix really is closed.

    Cutoff pruning drops anything that depended on work which had not finished, so
    a surviving record must never point at a dropped one. A dangling reference here
    means the prune was wrong, not that the framework misbehaved.
    """

    span_ids = {span["span_id"] for span in spans}
    for span in spans:
        parent = span.get("parent_span_id")
        require(
            parent is None or parent in span_ids,
            f"span {span['span_id']}: parent {parent} is not in the bundle",
        )

    dispatch_ids = {dispatch["dispatch_id"] for dispatch in dispatches}
    tool_ids = {tool["call_id"] for tool in tools}
    for dispatch in dispatches:
        execution = dispatch.get("execution_call_id")
        require(
            execution is None or execution in tool_ids,
            f"dispatch {dispatch['dispatch_id']}: execution {execution} is not in the bundle",
        )
    for tool in tools:
        require(
            tool["dispatch_id"] in dispatch_ids,
            f"tool {tool['call_id']}: dispatch {tool['dispatch_id']} is not in the bundle",
        )

    owners: dict[str, str] = {}
    for dispatch in dispatches:
        execution = dispatch.get("execution_call_id")
        if execution is None:
            continue
        require(
            execution not in owners,
            f"tool {execution}: claimed by dispatches {owners.get(execution)} and "
            f"{dispatch['dispatch_id']}",
        )
        owners[execution] = dispatch["dispatch_id"]


def validate_artifact_graph(events: list[dict[str, Any]]) -> None:
    """Shared-artifact producer/consumer chain.

    A read must name a producer that already committed the exact bytes it read.
    This is the one place where an intermediate file change is a hard gate, because
    a shared artifact is how one actor's work becomes another actor's input.
    """

    versions: dict[str, int] = defaultdict(int)
    producers: dict[str, tuple[str, str, int]] = {}
    for event in sorted(events, key=lambda item: (item["completed_at_ns"], item["event_id"])):
        if event["operation"] == "read":
            source = event["read_from"]
            require(
                source in producers,
                f"artifact {event['event_id']}: read before its producer {source}",
            )
            path, digest, version = producers[source]
            require(
                path == event["logical_path"],
                "artifact: read path does not match its producer",
            )
            require(
                digest == event["bytes_sha256"],
                "artifact: read bytes differ from the producer",
            )
            require(version == event["version"], "artifact: read version differs from the producer")
            continue
        expected = versions[event["logical_path"]] + 1
        require(
            event["version"] == expected,
            f"artifact {event['logical_path']}: version {event['version']} is not monotonic "
            f"(expected {expected})",
        )
        versions[event["logical_path"]] = expected
        producers[event["event_id"]] = (
            event["logical_path"],
            event["bytes_sha256"],
            event["version"],
        )
