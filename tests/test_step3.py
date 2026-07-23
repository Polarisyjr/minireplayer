"""Default Step3 export for source recordings."""

from __future__ import annotations

import json
from pathlib import Path

from minireplay.lane_record import (
    local_composite_scope_complete,
    local_composite_scope_start,
)
from minireplay.step3 import STEP3_SCHEMA, export_step3

S = 1_000_000_000


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_export_writes_step3_raw_text_png_and_cutoff_tails(tmp_path: Path) -> None:
    metadata = export_step3(
        output=tmp_path,
        run_id="source-c1",
        framework="mini-swe",
        records={
            "llm": [
                {
                    "attempt_id": "llm-0",
                    "actor_id": "actor-0",
                    "role": "agent",
                    "target_id": "vllm-8000",
                    "started_at_ns": 2 * S,
                    "ended_at_ns": 3 * S,
                    "prompt_token_ids": [1, 2],
                    "response_token_ids": [3],
                    "response": {"id": "chatcmpl-0"},
                }
            ],
            "tool": [
                {
                    "call_id": "tool-0",
                    "actor_id": "actor-0",
                    "name": "shell",
                    "status": "ok",
                    "started_at_ns": 4 * S,
                    "ended_at_ns": 5 * S,
                }
            ],
        },
        cutoff_tails={
            "llm_requests": [
                {
                    "attempt_id": "llm-tail",
                    "actor_id": "actor-0",
                    "role": "agent",
                    "target_id": "vllm-8000",
                    "started_at_ns": 6 * S,
                }
            ],
            "operations": [
                {
                    "kind": "tool",
                    "record_id": "tool-tail",
                    "actor_id": "actor-0",
                    "name": "shell",
                    "source_started_at_ns": 7 * S,
                },
                {
                    "kind": "dispatch",
                    "record_id": "dispatch-tail",
                    "actor_id": "actor-0",
                    "name": "shell",
                    "source_started_at_ns": 7 * S,
                },
            ],
        },
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=11 * S,
    )

    root = tmp_path / "step3"
    llm = rows(root / "raw/llm_spans.jsonl")
    tools = rows(root / "raw/tool_events.jsonl")
    assert metadata["schema_version"] == STEP3_SCHEMA
    assert metadata["counts"] == {
        "llm": 2,
        "tool": 2,
        "truncated_llm": 1,
        "truncated_tool": 1,
        "composite_scope": 0,
        "truncated_composite_scope": 0,
    }
    assert llm[-1]["timeline_kind"] == "truncated"
    assert tools[-1]["timeline_kind"] == "truncated"
    assert llm[-1]["ts_end"] == 110.0
    assert tools[-1]["ts_end"] == 110.0
    assert "dispatch-tail" not in (root / "raw/tool_events.jsonl").read_text()
    assert "[truncated]" in (root / "views/timeline.txt").read_text()
    assert (root / "views/timeline.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (root / "views/timeline.png").stat().st_size > 1024
    assert (root / "raw/engine_occupancy.jsonl").read_bytes() == b""
    assert (root / "raw/container_setup.jsonl").read_bytes() == b""
    assert (root / "raw/composite_scopes.jsonl").read_bytes() == b""


def test_render_uses_one_actor_lane_for_both_llm_and_tool(tmp_path: Path, monkeypatch) -> None:
    """The chart must not restore the old global LLM row."""

    import matplotlib.axes

    labels: list[str] = []
    original = matplotlib.axes.Axes.set_yticks

    def capture(self, ticks, labels_arg=None, *args, **kwargs):
        if labels_arg is not None:
            labels.extend(str(label) for label in labels_arg)
        return original(self, ticks, labels_arg, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_yticks", capture)
    export_step3(
        output=tmp_path,
        run_id="source-c2",
        framework="mini-swe",
        records={
            "llm": [
                {
                    "attempt_id": f"llm-{actor}",
                    "actor_id": actor,
                    "role": "agent",
                    "target_id": "vllm-8000",
                    "started_at_ns": started * S,
                    "ended_at_ns": (started + 1) * S,
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "response": {},
                }
                for actor, started in (("actor-a", 2), ("actor-b", 3))
            ],
            "tool": [
                {
                    "call_id": f"tool-{actor}",
                    "actor_id": actor,
                    "name": "shell",
                    "status": "ok",
                    "started_at_ns": (started + 1) * S,
                    "ended_at_ns": (started + 2) * S,
                }
                for actor, started in (("actor-a", 2), ("actor-b", 3))
            ],
        },
        cutoff_tails={"llm_requests": [], "operations": []},
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=6 * S,
    )

    assert labels == ["actor-b", "actor-a"]
    assert all(not label.startswith(("LLM:", "tool:")) for label in labels)


def test_composite_scope_is_an_unfilled_non_work_envelope(
    tmp_path: Path, monkeypatch
) -> None:
    import matplotlib.axes

    events = tmp_path / "lane-events"
    scope_id = local_composite_scope_start(
        root=events,
        actor_id="actor-0",
        session_id="actor-0",
        name="browse_url",
        causal_lane="model-call:browse-0",
        started_at_ns=2 * S,
    )
    local_composite_scope_complete(
        root=events,
        actor_id="actor-0",
        session_id="actor-0",
        scope_id=scope_id,
        ended_at_ns=5 * S,
        status="ok",
    )
    styles: list[dict] = []
    original = matplotlib.axes.Axes.broken_barh

    def capture(self, *args, **kwargs):
        styles.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "broken_barh", capture)
    metadata = export_step3(
        output=tmp_path,
        run_id="source-composite",
        framework="owl",
        records={
            "llm": [
                {
                    "attempt_id": "llm-0",
                    "actor_id": "actor-0",
                    "role": "browser_web",
                    "target_id": "vllm-8006",
                    "started_at_ns": 3 * S,
                    "ended_at_ns": 4 * S,
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "response": {},
                }
            ],
            "tool": [],
        },
        cutoff_tails={"llm_requests": [], "operations": []},
        scope_event_dir=events,
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=6 * S,
    )

    scopes = rows(tmp_path / "step3/raw/composite_scopes.jsonl")
    assert scopes == [
        {
            "causal_lane": "model-call:browse-0",
            "chain": "actor-0",
            "name": "browse_url",
            "scope_id": scope_id,
            "source": "minireplay-composite-scope",
            "timeline_kind": "scope",
            "ts_end": 104.0,
            "ts_start": 101.0,
        }
    ]
    assert metadata["counts"]["composite_scope"] == 1
    assert metadata["timeline"]["busy_s"] == 1.0
    assert any(
        style.get("facecolors") == "none" and style.get("linestyles") == "dashed"
        for style in styles
    )


def test_legacy_owl_browse_url_is_promoted_out_of_the_tool_lane(tmp_path: Path) -> None:
    metadata = export_step3(
        output=tmp_path,
        run_id="legacy-owl",
        framework="owl",
        records={
            "llm": [],
            "tool": [
                {
                    "call_id": "tool-browse",
                    "actor_id": "actor-0",
                    "name": "browse_url",
                    "causal_lane": "model-call:browse-0",
                    "status": "ok",
                    "started_at_ns": 2 * S,
                    "ended_at_ns": 5 * S,
                },
                {
                    "call_id": "tool-action",
                    "actor_id": "actor-0",
                    "name": "browser_action",
                    "status": "ok",
                    "started_at_ns": 3 * S,
                    "ended_at_ns": 4 * S,
                },
            ],
        },
        cutoff_tails={"llm_requests": [], "operations": []},
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=6 * S,
    )

    assert [row["tool"] for row in rows(tmp_path / "step3/raw/tool_events.jsonl")] == [
        "browser_action"
    ]
    assert [
        row["name"] for row in rows(tmp_path / "step3/raw/composite_scopes.jsonl")
    ] == ["browse_url"]
    assert metadata["counts"]["tool"] == 1
    assert metadata["counts"]["composite_scope"] == 1
    assert metadata["timeline"]["busy_s"] == 1.0
