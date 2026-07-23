from __future__ import annotations

import asyncio
from typing import Any

from minireplay.sdk import (
    composite_scope,
    current_context,
    remember_llm_attempt,
    reset_context,
    run_dispatch,
    run_tool,
    set_context,
)


class FakeBoundary:
    def __init__(self) -> None:
        self.sequence = 0
        self.open: dict[str, tuple[str, str]] = {}
        self.completed: list[tuple[str, dict[str, Any]]] = []
        self.started: list[tuple[str, dict[str, Any]]] = []

    def start(self, kind: str, **fields: Any) -> dict[str, str]:
        self.sequence += 1
        record_id = f"{kind}-{self.sequence}"
        reservation_id = f"reservation-{self.sequence}"
        self.open[reservation_id] = (kind, record_id)
        self.started.append((kind, fields))
        return {
            "reservation_id": reservation_id,
            "record_id": record_id,
            "span_id": f"span-{self.sequence}",
        }

    def complete(self, reservation_id: str, **fields: Any) -> dict[str, bool]:
        kind, record_id = self.open.pop(reservation_id)
        self.completed.append((f"{kind}:{record_id}", fields))
        return {"valid": True, "result_replay_required": False}


def test_nested_tools_do_not_become_sibling_dispatch_executions() -> None:
    """A dispatch owns its outer tool; primitives below it are descendants."""

    boundary = FakeBoundary()
    tokens = set_context(actor_id="actor", process_role="agent")
    try:
        def invoke_outer() -> str:
            return run_tool(
                name="outer",
                implementation="test.outer",
                arguments={},
                invoke=lambda: run_tool(
                    name="primitive",
                    implementation="test.primitive",
                    arguments={},
                    invoke=lambda: "nested-result",
                    client=boundary,
                ),
                client=boundary,
            )

        assert (
            run_dispatch(
                name="outer",
                arguments={},
                parser_identity="test.parser",
                dispatcher_identity="test.dispatcher",
                native_call_id="model-call-0",
                origin_kind="llm_structured",
                trigger_id="llm-0",
                invoke=invoke_outer,
                client=boundary,
            )
            == "nested-result"
        )
    finally:
        reset_context(tokens)

    dispatch_completions = [
        fields
        for identity, fields in boundary.completed
        if identity.startswith("dispatch:")
    ]
    assert len(dispatch_completions) == 1
    assert dispatch_completions[0]["status"] == "executed"
    assert dispatch_completions[0]["execution_call_id"] == "tool-2"


def test_composite_scope_records_only_its_replayable_primitives() -> None:
    """The orchestration envelope is not a dispatch or tool ledger entry."""

    boundary = FakeBoundary()
    tokens = set_context(actor_id="actor", process_role="agent")
    try:
        with composite_scope(name="browse_url", model_call_id="browser-call-0"):
            result = run_tool(
                name="browser_action",
                implementation="test.browser_action",
                arguments={"action_code": "click(1)"},
                invoke=lambda: "clicked",
                client=boundary,
            )
    finally:
        reset_context(tokens)

    assert result == "clicked"
    assert [kind for kind, _fields in boundary.started] == ["tool"]
    fields = boundary.started[0][1]
    assert fields["dispatch_id"] is None
    assert fields["causal_lane"] == "model-call:browser-call-0"


def test_owl_browse_url_bypasses_outer_dispatch_and_tool_ledgers() -> None:
    from minireplay.instrumentation.owl import _call_factory, _execute_tool_factory

    def browse_url() -> str:
        return "native result"

    browse_url.__module__ = "camel.toolkits.browser_toolkit"
    browse_url.__qualname__ = "AsyncBrowserToolkit.browse_url"

    class Tool:
        func = staticmethod(browse_url)
        openai_tool_schema = {"function": {"name": "browse_url"}}

    class Agent:
        _internal_tools = {"browse_url": Tool()}

    class Request:
        tool_name = "browse_url"
        tool_call_id = "browser-call-0"
        args = {"task_prompt": "inspect"}

    observed_lanes: list[str | None] = []

    def execute(_agent, _request):
        observed_lanes.append(current_context()["composite_lane"])
        return _call_factory(lambda _tool: "native result")(Tool())

    wrapped = _execute_tool_factory(execute)
    tokens = set_context(actor_id="actor", process_role="agent")
    try:
        remember_llm_attempt("llm-parent")
        assert wrapped(Agent(), Request()) == "native result"
        assert current_context()["composite_lane"] is None
    finally:
        reset_context(tokens)

    # A regular dispatch/tool wrapper would construct BoundaryClient and fail
    # without an endpoint. Passing here proves both outer ledgers were bypassed.
    assert observed_lanes == ["model-call:browser-call-0"]


def test_owl_composite_captures_open_visit_and_observe_primitives(monkeypatch) -> None:
    import minireplay.instrumentation.owl as owl

    captured: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    async def primitive(name, _implementation, arguments, invoke, **options):
        captured.append((name, arguments, options))
        return await invoke()

    monkeypatch.setattr(owl, "_primitive_tool_async", primitive)

    class Browser:
        headless = True

    async def native_init(_browser):
        return None

    async def native_visit(_browser, _url, **_kwargs):
        return None

    async def native_observe(_browser, _save_image=False):
        return object(), "/tmp/observation.png"

    async def exercise() -> None:
        browser = Browser()
        with composite_scope(name="browse_url", model_call_id="browser-call-0"):
            await owl._browser_init_factory(native_init)(browser)
            await owl._browser_visit_factory(native_visit)(
                browser,
                "https://example.test",
                timeout=123,
                max_retries=4,
            )
            await owl._browser_observe_factory(native_observe)(browser, True)

    tokens = set_context(actor_id="actor", process_role="agent")
    try:
        asyncio.run(exercise())
    finally:
        reset_context(tokens)

    assert [name for name, _arguments, _options in captured] == [
        "browser_open",
        "browser_visit_page",
        "browser_observe",
    ]
    assert captured[1][1] == {
        "url": "https://example.test",
        "timeout": 123,
        "max_retries": 4,
    }
    assert captured[2][2]["result_encoder"] is owl._browser_observe_result
    assert captured[2][2]["result_replayer"] is owl._keep_native_browser_observation
