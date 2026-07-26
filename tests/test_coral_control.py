from __future__ import annotations

import json

from minireplay.coral_control import (
    fixed_invocation_limits,
    grader_recorded,
    recorded_restart_controls,
)


def test_fixed_invocation_limits_include_closed_and_cutoff_llm_sessions() -> None:
    limits = fixed_invocation_limits(
        [
            {
                "actor_id": "agent-a",
                "session_id": "agent-a/invocation-0/root-0",
            },
            {
                "actor_id": "agent-a",
                "session_id": "agent-a/invocation-1/root-0",
            },
            {
                "actor_id": "agent-b",
                "session_id": "agent-b/invocation-0/root-0",
            },
            {
                "actor_id": "other",
                "session_id": "unrelated/invocation-99/root-0",
            },
        ],
        {
            "llm_requests": [
                {
                    "actor_id": "agent-a",
                    "session_id": "agent-a/invocation-2/root-0",
                }
            ]
        },
    )

    assert limits == {"agent-a": 3, "agent-b": 1}


def test_recorded_restart_controls_keep_exact_prompt_and_grader_trigger(tmp_path) -> None:
    logs = tmp_path / ".coral" / "public" / "logs"
    logs.mkdir(parents=True)
    for invocation, commit in ((1, "abcdef123456"), (2, "123456abcdef")):
        (logs / f"agent-1.{invocation}.log").write_text(
            json.dumps(
                {
                    "type": "coral",
                    "source": "heartbeat:reflect",
                    "prompt": f"## Eval Results\n\nCommit: {commit}\n\nReflect.",
                }
            )
            + "\n"
        )
    controls = recorded_restart_controls(
        actors=[
            {
                "actor_id": "actor-1",
                "lane": {
                    "lane_kind": "agent",
                    "agent_id": "agent-1",
                    "parent_actor_id": "team-0",
                },
            }
        ],
        task_terminals=[
            {
                "actor_id": "team-0",
                "result": {"run_dir": str(tmp_path)},
            }
        ],
        graders=[
            {
                "actor_id": "actor-1",
                "attempt_id": "grader-1",
                "trigger_id": "abcdef1234567890",
            },
            {
                "actor_id": "actor-1",
                "attempt_id": "grader-2",
                "trigger_id": "123456abcdef7890",
            },
        ],
        invocation_limits={"actor-1": 3},
    )

    assert [
        (control["invocation_index"], control["trigger_grader_attempt_id"])
        for control in controls
    ] == [(1, "grader-1"), (2, "grader-2")]
    assert controls[0]["prompt"].endswith("Reflect.")


def test_grader_recorded_reads_replay_stage(tmp_path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "graders.jsonl").write_text(
        json.dumps({"attempt_id": "grader-1"}) + "\n"
    )

    assert grader_recorded(tmp_path, "grader-1") is True
    assert grader_recorded(tmp_path, "grader-2") is False
    assert grader_recorded(tmp_path, None) is True
