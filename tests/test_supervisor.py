from __future__ import annotations

from pathlib import Path

from minireplay.supervisor import (
    _actors_from_stage,
    _apply_coral_task_cutoffs,
    _unexpected_failed_tasks,
)
from minireplay.util import atomic_write_json
from tests.support import make_bundle


def test_actor_inventory_includes_refill_actor_seen_only_at_cutoff(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    ready = tmp_path / "actor-ready"
    bindings = tmp_path / "lane-bindings"
    stage.mkdir()
    ready.mkdir()
    bindings.mkdir()
    atomic_write_json(
        ready / "initial.json",
        {
            "actor_id": "initial",
            "source_actor_id": "initial",
            "process_role": "agent",
        },
    )
    atomic_write_json(
        bindings / "binding.json",
        {
            "actor_id": "initial",
            "source_actor_id": "refill",
            "native_lane_key": "worker-0",
        },
    )

    actors = _actors_from_stage(
        stage,
        ready,
        {
            "llm_requests": [{"actor_id": "initial", "role": "agent"}],
            "operations": [],
        },
        bindings,
    )

    assert actors == [
        {
            "actor_id": "initial",
            "source_actor_id": "initial",
            "source_actor_ids": ["initial", "refill"],
            "process_role": "agent",
            "task": {"source_actor_id": "initial"},
        },
    ]


def test_only_cutoff_open_refill_timeout_is_an_expected_terminal_failure(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(actors=["lane-0"])
    bundle.manifest["actors"] = [
        {
            "actor_id": "lane-0",
            "source_actor_id": "initial",
            "source_actor_ids": ["initial", "refill"],
        }
    ]
    bundle.terminal["task_terminals"] = [
        {
            "actor_id": "lane-0",
            "status": "success",
            "task": {"source_actor_id": "initial"},
        }
    ]
    cutoff_timeout = {
        "actor_id": "lane-0",
        "status": "failure",
        "task": {"source_actor_id": "refill"},
        "result": {"error_type": "TimeoutError", "message": "timed out"},
    }

    assert (
        _unexpected_failed_tasks(
            mode="replay",
            reason="fixed-work-complete",
            bundle=bundle,
            task_terminals=[cutoff_timeout],
        )
        == []
    )

    for changed in (
        {**cutoff_timeout, "result": {"error_type": "RuntimeError"}},
        {**cutoff_timeout, "task": {"source_actor_id": "initial"}},
    ):
        assert _unexpected_failed_tasks(
            mode="replay",
            reason="fixed-work-complete",
            bundle=bundle,
            task_terminals=[changed],
        ) == ["lane-0"]


def test_coral_manager_cutoff_clamps_agent_tails_but_not_graders() -> None:
    gate_at_ns = 1_000_000_000
    gate_at_epoch_ns = 100_000_000_000
    cutoff_epoch_ns = 107_500_000_000
    tails = _apply_coral_task_cutoffs(
        adapter="coral",
        cutoff_tails={
            "llm_requests": [
                {
                    "actor_id": "agent-3",
                    "started_at_ns": 6_000_000_000,
                    "elapsed_ns": 5_000_000_000,
                }
            ],
            "operations": [
                {
                    "actor_id": "agent-2",
                    "process_role": "coral-opencode",
                    "source_started_at_ns": 7_000_000_000,
                    "elapsed_ns": 4_000_000_000,
                },
                {
                    "actor_id": "agent-2",
                    "process_role": "coral-grader",
                    "source_started_at_ns": 7_000_000_000,
                    "elapsed_ns": 4_000_000_000,
                },
            ],
        },
        task_terminals=[
            {
                "actor_id": "team-0",
                "result": {
                    "termination_reason": "max_total_turns",
                    "replay_cutoff_at_epoch_ns": cutoff_epoch_ns,
                },
            }
        ],
        actors=[
            {
                "actor_id": actor,
                "lane": {
                    "lane_kind": "agent",
                    "parent_actor_id": "team-0",
                },
            }
            for actor in ("agent-1", "agent-2", "agent-3", "agent-4")
        ],
        gate_at_ns=gate_at_ns,
        gate_at_epoch_ns=gate_at_epoch_ns,
    )

    expected_cutoff_ns = 8_500_000_000
    assert tails["llm_requests"][0]["elapsed_ns"] == 2_500_000_000
    assert tails["llm_requests"][0]["lane_terminated_at_ns"] == expected_cutoff_ns
    assert tails["operations"][0]["elapsed_ns"] == 1_500_000_000
    assert tails["operations"][0]["lane_termination_reason"] == "max_total_turns"
    assert "lane_terminated_at_ns" not in tails["operations"][1]
