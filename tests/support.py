from __future__ import annotations

from pathlib import Path
from typing import Any

from minireplay.bundle import Bundle
from minireplay.constants import (
    ARTIFACT_SCHEMA,
    DISPATCH_SCHEMA,
    LLM_SCHEMA,
    MANIFEST_SCHEMA,
    SPAN_SCHEMA,
    TERMINAL_SCHEMA,
    TOOL_SCHEMA,
)
from minireplay.observation import exact_result_contract, recorded_output_result_contract
from minireplay.util import sha256_json

ZERO = "0" * 64


def dispatch(
    *,
    dispatch_id: str = "dispatch-0",
    actor_id: str = "actor-0",
    name: str = "shell",
    arguments: dict[str, Any] | None = None,
    execution_call_id: str | None = "tool-0",
    session_id: str | None = None,
    started: int = 100,
    ended: int = 200,
) -> dict[str, Any]:
    arguments = {"command": "echo hi"} if arguments is None else arguments
    return {
        "schema_version": DISPATCH_SCHEMA,
        "dispatch_id": dispatch_id,
        "span_id": f"span-{dispatch_id}",
        "actor_id": actor_id,
        "session_id": session_id,
        "process_role": "agent",
        "parser_identity": "parser",
        "dispatcher_identity": "dispatcher",
        "native_call_id": dispatch_id,
        "name": name,
        "arguments": arguments,
        "arguments_sha256": sha256_json(arguments),
        "origin": {"kind": "llm_structured", "trigger_id": "llm-0", "model_call_id": None},
        "status": "executed" if execution_call_id else "rejected",
        "execution_call_id": execution_call_id,
        "started_at_ns": started,
        "ended_at_ns": ended,
    }


def tool(
    *,
    call_id: str = "tool-0",
    dispatch_id: str | None = "dispatch-0",
    actor_id: str = "actor-0",
    name: str = "shell",
    causal_lane: str | None = None,
    arguments: dict[str, Any] | None = None,
    result: Any = None,
    contract: dict[str, Any] | None = None,
    exception_raised: bool = False,
    status: str = "ok",
    started: int = 110,
    ended: int = 190,
) -> dict[str, Any]:
    arguments = {"command": "echo hi"} if arguments is None else arguments
    result = {"output": "hi", "exit_code": 0} if result is None else result
    return {
        "schema_version": TOOL_SCHEMA,
        "call_id": call_id,
        "dispatch_id": dispatch_id,
        "causal_lane": causal_lane,
        "span_id": f"span-{call_id}",
        "actor_id": actor_id,
        "process_role": "agent",
        "name": name,
        "implementation": "native-shell",
        "arguments": arguments,
        "arguments_sha256": sha256_json(arguments),
        "result_contract": contract or recorded_output_result_contract("/output"),
        "semantic_timeout_s": None,
        "result": result,
        "native_result": result,
        "native_observations": [],
        "status": status,
        "exception_raised": exception_raised,
        "native_execution": True,
        "cpu_seconds": 0.0,
        "child_processes": [],
        "started_at_ns": started,
        "ended_at_ns": ended,
    }


def span(span_id: str, *, parent: str | None = None, actor_id: str = "actor-0") -> dict[str, Any]:
    return {
        "schema_version": SPAN_SCHEMA,
        "span_id": span_id,
        "parent_span_id": parent,
        "actor_id": actor_id,
        "kind": "tool",
        "name": "shell",
        "status": "ok",
        "started_at_ns": 100,
        "ended_at_ns": 200,
    }


def llm(
    *,
    attempt_id: str = "llm-0",
    actor_id: str = "actor-0",
    session_id: str = "actor-0",
    role: str = "agent",
    sequence: int = 0,
    request: dict[str, Any] | None = None,
    response: Any = None,
    stream: bool = False,
) -> dict[str, Any]:
    if request is None:
        request = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    response = {"id": "chatcmpl-1", "choices": []} if response is None else response
    from minireplay.llm_store import request_shape

    return {
        "schema_version": LLM_SCHEMA,
        "attempt_id": attempt_id,
        "span_id": f"span-{attempt_id}",
        "actor_id": actor_id,
        "session_id": session_id,
        "role": role,
        "target_id": "vllm-8000",
        "sequence": sequence,
        "api": "chat.completions",
        "request": request,
        "request_sha256": sha256_json(request),
        "request_shape_sha256": sha256_json(request_shape(request)),
        "response": response,
        "stream": stream,
        "status_code": 200,
        "prompt_token_ids": [1, 2, 3],
        "response_token_ids": [4, 5],
        "started_at_ns": 10,
        "ended_at_ns": 20,
    }


def artifact(
    *,
    event_id: str = "artifact-0",
    logical_path: str = "/attempts/a.json",
    operation: str = "create",
    version: int = 1,
    read_from: str | None = None,
    digest: str = ZERO,
    completed_at_ns: int = 100,
    actor_id: str = "actor-0",
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "event_id": event_id,
        "span_id": f"span-{event_id}",
        "actor_id": actor_id,
        "logical_path": logical_path,
        "physical_path": logical_path,
        "operation": operation,
        "version": version,
        "bytes_sha256": digest,
        "size": 1,
        "mode": 0o644,
        "triggered_by": [],
        "read_from": read_from,
        "completed_at_ns": completed_at_ns,
        "native_execution": True,
    }


def make_bundle(
    *,
    root: Path | None = None,
    dispatches: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    graders: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    llm_records: list[dict[str, Any]] | None = None,
    spans: list[dict[str, Any]] | None = None,
    cutoff_tails: dict[str, Any] | None = None,
    adapter: str = "mini-swe",
    actors: list[str] | None = None,
) -> Bundle:
    dispatches = [dispatch()] if dispatches is None else dispatches
    tools = [tool()] if tools is None else tools
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "bundle_id": "test",
        "adapter": adapter,
        "workload": {"framework": adapter, "concurrency": 1, "duration_s": 60, "seed": 42},
        "actors": [{"actor_id": a} for a in (actors or ["actor-0"])],
        "window": {"gate_at_ns": 0, "terminal_at_ns": 1000},
        "counts": {
            "llm": len(llm_records or []),
            "dispatch": len(dispatches),
            "tool": len(tools),
            "grader": len(graders or []),
            "artifact": len(artifacts or []),
        },
    }
    return Bundle(
        root=root or Path("/nonexistent"),
        manifest=manifest,
        llm=llm_records or [],
        dispatches=dispatches,
        tools=tools,
        graders=graders or [],
        artifacts=artifacts or [],
        spans=spans or [],
        cutoff_tails=cutoff_tails or {"operations": [], "llm_requests": []},
        terminal={
            "schema_version": TERMINAL_SCHEMA,
            "status": "success",
            "task_terminals": [],
        },
    )


__all__ = [
    "ZERO",
    "artifact",
    "dispatch",
    "exact_result_contract",
    "llm",
    "make_bundle",
    "span",
    "tool",
]
