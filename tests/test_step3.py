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
        "lane_termination": 0,
        "restart": 0,
        "heartbeat_restart": 0,
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
    assert (root / "raw/lane_terminations.jsonl").read_bytes() == b""
    assert (root / "raw/restart_events.jsonl").read_bytes() == b""


def test_cutoff_tail_ends_at_observed_elapsed_time_not_record_window(
    tmp_path: Path,
) -> None:
    export_step3(
        output=tmp_path,
        run_id="source-interrupted",
        framework="coral",
        records={"llm": [], "tool": []},
        cutoff_tails={
            "llm_requests": [
                {
                    "attempt_id": "llm-tail",
                    "actor_id": "agent-3",
                    "role": "coral-agent",
                    "target_id": "vllm-8002",
                    "started_at_ns": 6 * S,
                    "elapsed_ns": 2 * S,
                    "interrupted_at_ns": 8 * S,
                }
            ],
            "operations": [
                {
                    "kind": "tool",
                    "record_id": "tool-tail",
                    "actor_id": "agent-3",
                    "name": "bash",
                    "source_started_at_ns": 7 * S,
                    "elapsed_ns": S,
                    "lane_terminated_at_ns": 8 * S,
                }
            ],
        },
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=11 * S,
    )

    llm = rows(tmp_path / "step3/raw/llm_spans.jsonl")
    tools = rows(tmp_path / "step3/raw/tool_events.jsonl")
    assert llm[0]["ts_start"] == 105.0
    assert llm[0]["ts_end"] == 107.0
    assert llm[0]["e2e_s"] == 2.0
    assert tools[0]["ts_start"] == 106.0
    assert tools[0]["ts_end"] == 107.0


def test_coral_team_cutoff_marks_all_child_lanes_without_counting_work(
    tmp_path: Path,
) -> None:
    team_actor = "team-0"
    actors = [f"agent-{index}" for index in range(1, 5)]
    actor_lanes = {
        actor: {
            "concurrency_unit": "coral-team",
            "lane_kind": "agent",
            "parent_actor_id": team_actor,
            "team_slot": 0,
            "slot_generation": 0,
            "agent_id": actor,
            "agent_index": index,
            "source_task_id": "frontier_cs_algo/14",
        }
        for index, actor in enumerate(actors, start=1)
    }
    run_dir = tmp_path / "native-run"
    logs = run_dir / ".coral/public/logs"
    logs.mkdir(parents=True)
    (logs / "agent-1.0.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "coral",
                        "agent_id": "agent-1",
                        "source": "start",
                        "timestamp": "1970-01-01T00:01:41+00:00",
                    }
                ),
                json.dumps({"type": "step_finish", "timestamp": 103_000}),
            ]
        )
        + "\n"
    )
    (logs / "agent-1.1.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "coral",
                        "agent_id": "agent-1",
                        "source": "restart",
                        "timestamp": "1970-01-01T00:01:43.5+00:00",
                    }
                ),
                json.dumps({"type": "step_start", "timestamp": 104_000}),
            ]
        )
        + "\n"
    )
    metadata = export_step3(
        output=tmp_path,
        run_id="coral-team-cutoff",
        framework="coral",
        records={
            "llm": [
                {
                    "attempt_id": "llm-agent-3",
                    "actor_id": "agent-3",
                    "role": "coral-agent",
                    "target_id": "vllm-8002",
                    "started_at_ns": 6 * S,
                    "ended_at_ns": 7 * S,
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "response": {},
                }
            ],
            "tool": [],
        },
        cutoff_tails={"llm_requests": [], "operations": []},
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=11 * S,
        actor_lanes=actor_lanes,
        task_terminals=[
            {
                "actor_id": team_actor,
                "result": {
                    "termination_reason": "max_total_turns",
                    "replay_cutoff_at_epoch_ns": 107_500_000_000,
                    "run_dir": str(run_dir),
                },
            }
        ],
    )

    terminations = rows(tmp_path / "step3/raw/lane_terminations.jsonl")
    assert [row["chain"] for row in terminations] == actors
    assert {row["ts_terminated"] for row in terminations} == {107.5}
    assert {row["seconds_since_gate"] for row in terminations} == {7.5}
    assert {row["terminated_at_ns"] for row in terminations} == {8_500_000_000}
    assert {row["reason"] for row in terminations} == {"max_total_turns"}
    assert metadata["counts"]["lane_termination"] == 4
    assert metadata["counts"]["restart"] == 1
    assert metadata["timeline"]["busy_s"] == 1.0
    restarts = rows(tmp_path / "step3/raw/restart_events.jsonl")
    assert restarts == [
        {
            "agent_id": "agent-1",
            "chain": "agent-1",
            "end_seconds_since_gate": 4.0,
            "invocation_index": 1,
            "source": "coral-native-invocation-log",
            "source_prompt": "restart",
            "start_seconds_since_gate": 3.0,
            "team_chain": team_actor,
            "timeline_kind": "restart",
            "ts_end": 104.0,
            "ts_start": 103.0,
        }
    ]
    text = (tmp_path / "step3/views/timeline.txt").read_text()
    assert text.count("TEAM TERMINATED:max_total_turns") == 4
    assert "CONTROL:restart -> invocation-1" in text
    assert "no lane work after this marker" in text


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


def test_render_groups_coral_agents_by_team_slot(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.axes

    labels: list[str] = []
    original = matplotlib.axes.Axes.set_yticks

    def capture(self, ticks, labels_arg=None, *args, **kwargs):
        if labels_arg is not None:
            labels.extend(str(label) for label in labels_arg)
        return original(self, ticks, labels_arg, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_yticks", capture)
    actors = ("slot0-agent1", "slot0-agent2", "slot1-agent1")
    actor_lanes = {
        "slot0-agent1": {
            "concurrency_unit": "coral-team",
            "lane_kind": "agent",
            "team_slot": 0,
            "slot_generation": 0,
            "agent_id": "agent-1",
            "agent_index": 1,
            "source_task_id": "frontier_cs_algo/14",
        },
        "slot0-agent2": {
            "concurrency_unit": "coral-team",
            "lane_kind": "agent",
            "team_slot": 0,
            "slot_generation": 0,
            "agent_id": "agent-2",
            "agent_index": 2,
            "source_task_id": "frontier_cs_algo/14",
        },
        "slot1-agent1": {
            "concurrency_unit": "coral-team",
            "lane_kind": "agent",
            "team_slot": 1,
            "slot_generation": 0,
            "agent_id": "agent-1",
            "agent_index": 1,
            "source_task_id": "frontier_cs_algo/169",
        },
    }
    export_step3(
        output=tmp_path,
        run_id="coral-c2",
        framework="coral",
        records={
            "llm": [
                {
                    "attempt_id": f"llm-{actor}",
                    "actor_id": actor,
                    "role": "coral-agent",
                    "target_id": "vllm-8000",
                    "started_at_ns": (index + 2) * S,
                    "ended_at_ns": (index + 3) * S,
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "response": {},
                }
                for index, actor in enumerate(reversed(actors))
            ],
            "tool": [],
        },
        cutoff_tails={"llm_requests": [], "operations": []},
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=7 * S,
        actor_lanes=actor_lanes,
    )

    assert labels == [
        "slot-01/g00/agent-1 · 169",
        "slot-00/g00/agent-2 · 14",
        "slot-00/g00/agent-1 · 14",
    ]
    metadata = json.loads((tmp_path / "step3/metadata.json").read_text())
    assert metadata["actor_lanes"] == actor_lanes


def test_coral_subagent_work_stays_on_the_owning_agent_lane(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import matplotlib.axes

    labels: list[str] = []
    original = matplotlib.axes.Axes.set_yticks

    def capture(self, ticks, labels_arg=None, *args, **kwargs):
        if labels_arg is not None:
            labels.extend(str(label) for label in labels_arg)
        return original(self, ticks, labels_arg, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_yticks", capture)
    actor = "agent-3"
    root_session = f"{actor}/invocation-0/root-0"
    child_session = f"{root_session}/child-0"
    actor_lanes = {
        actor: {
            "concurrency_unit": "coral-team",
            "lane_kind": "agent",
            "team_slot": 0,
            "slot_generation": 0,
            "agent_id": "agent-3",
            "agent_index": 3,
            "source_task_id": "frontier_cs_algo/14",
        }
    }
    metadata = export_step3(
        output=tmp_path,
        run_id="coral-subagent",
        framework="coral",
        records={
            "llm": [
                {
                    "attempt_id": "llm-root",
                    "actor_id": actor,
                    "session_id": root_session,
                    "role": "coral-agent",
                    "target_id": "vllm-8002",
                    "started_at_ns": 2 * S,
                    "ended_at_ns": 3 * S,
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "response": {},
                },
                {
                    "attempt_id": "llm-child",
                    "actor_id": actor,
                    "session_id": child_session,
                    "role": "coral-subagent",
                    "target_id": "vllm-8002",
                    "started_at_ns": 4 * S,
                    "ended_at_ns": 5 * S,
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "response": {},
                },
            ],
            "dispatch": [
                {
                    "dispatch_id": "dispatch-child",
                    "actor_id": actor,
                    "session_id": child_session,
                }
            ],
            "tool": [
                {
                    "call_id": "tool-child",
                    "dispatch_id": "dispatch-child",
                    "actor_id": actor,
                    "name": "read",
                    "status": "ok",
                    "started_at_ns": 5 * S,
                    "ended_at_ns": 6 * S,
                }
            ],
        },
        cutoff_tails={
            "llm_requests": [],
            "operations": [
                {
                    "kind": "dispatch",
                    "record_id": "dispatch-task",
                    "dispatch_id": "dispatch-task",
                    "actor_id": actor,
                    "session_id": root_session,
                    "name": "task",
                    "source_started_at_ns": 3 * S,
                    "elapsed_ns": 7 * S,
                    "replay_entry": "enter-and-preserve-descendants",
                },
                {
                    "kind": "tool",
                    "record_id": "tool-task",
                    "call_id": "tool-task",
                    "dispatch_id": "dispatch-task",
                    "actor_id": actor,
                    "session_id": root_session,
                    "name": "task",
                    "source_started_at_ns": 3 * S,
                    "elapsed_ns": 7 * S,
                    "replay_entry": "enter-and-preserve-descendants",
                },
                {
                    "kind": "dispatch",
                    "record_id": "dispatch-bash-tail",
                    "dispatch_id": "dispatch-bash-tail",
                    "actor_id": actor,
                    "session_id": child_session,
                    "name": "bash",
                    "source_started_at_ns": 6 * S,
                    "elapsed_ns": 4 * S,
                    "replay_entry": "block-before-entry",
                },
                {
                    "kind": "tool",
                    "record_id": "tool-bash-tail",
                    "call_id": "tool-bash-tail",
                    "dispatch_id": "dispatch-bash-tail",
                    "actor_id": actor,
                    "session_id": child_session,
                    "name": "bash",
                    "source_started_at_ns": 6 * S,
                    "elapsed_ns": 4 * S,
                    "replay_entry": "block-before-entry",
                },
            ],
        },
        gate_at_ns=S,
        gate_at_epoch_ns=100 * S,
        terminal_at_ns=11 * S,
        actor_lanes=actor_lanes,
    )

    llm_rows = rows(tmp_path / "step3/raw/llm_spans.jsonl")
    tool_rows = rows(tmp_path / "step3/raw/tool_events.jsonl")
    scopes = rows(tmp_path / "step3/raw/composite_scopes.jsonl")
    assert llm_rows[1]["chain"] == actor
    assert llm_rows[1]["actor_chain"] == actor
    assert llm_rows[1]["session_id"] == child_session
    assert llm_rows[1]["work_scope"] == "subagent"
    assert tool_rows[0]["chain"] == actor
    assert tool_rows[0]["session_id"] == child_session
    assert tool_rows[0]["work_scope"] == "subagent"
    assert tool_rows[1]["chain"] == actor
    assert tool_rows[1]["session_id"] == child_session
    assert tool_rows[1]["tool"] == "bash"
    assert tool_rows[1]["timeline_kind"] == "truncated"
    assert tool_rows[1]["ts_end"] == 109.0
    assert scopes[0]["name"] == "task"
    assert scopes[0]["timeline_kind"] == "truncated"
    assert metadata["counts"]["tool"] == 2
    assert metadata["counts"]["composite_scope"] == 1
    assert labels == ["slot-00/g00/agent-3 · 14"]


def test_composite_scope_is_an_unfilled_non_work_envelope(tmp_path: Path, monkeypatch) -> None:
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
    assert [row["name"] for row in rows(tmp_path / "step3/raw/composite_scopes.jsonl")] == [
        "browse_url"
    ]
    assert metadata["counts"]["tool"] == 1
    assert metadata["counts"]["composite_scope"] == 1
    assert metadata["timeline"]["busy_s"] == 1.0
