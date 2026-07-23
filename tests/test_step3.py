"""Default Step3 export for source recordings."""

from __future__ import annotations

import json
from pathlib import Path

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
