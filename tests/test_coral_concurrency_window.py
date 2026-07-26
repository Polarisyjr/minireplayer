from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minireplay.bundle import _concurrency_window
from minireplay.comparison_plot import _display_actor, _work_actors
from minireplay.constants import MANIFEST_SCHEMA
from minireplay.errors import ValidationError
from minireplay.schema import validate_manifest
from minireplay.supervisor import _actors_from_stage


def _team_attempt(slot: int, generation: int, run_index: int, task: str) -> list[dict]:
    parent = f"team-{slot}-{generation}"
    common = {
        "framework": "coral",
        "concurrency_unit": "coral-team",
        "team_slot": slot,
        "slot_generation": generation,
        "run_index": run_index,
        "team_size": 4,
        "source_task_id": task,
    }
    return [
        {
            "actor_id": parent,
            "lane": {**common, "lane_kind": "team"},
        },
        *[
            {
                "actor_id": f"{parent}-agent-{index}",
                "lane": {
                    **common,
                    "lane_kind": "agent",
                    "agent_id": f"agent-{index}",
                    "agent_index": index,
                    "parent_actor_id": parent,
                },
            }
            for index in range(1, 5)
        ],
    ]


def test_coral_window_groups_four_agents_and_refills_one_whole_slot() -> None:
    actors = [
        *_team_attempt(0, 0, 0, "task-a"),
        *_team_attempt(1, 0, 1, "task-b"),
        *_team_attempt(0, 1, 2, "task-c"),
    ]

    window = _concurrency_window(
        adapter="coral",
        workload={"concurrency": 2, "coral_team_size": 4},
        actors=actors,
    )

    assert window is not None
    assert window["unit"] == "coral-team"
    assert window["size"] == 2
    assert window["team_size"] == 4
    assert window["target_agent_lanes"] == 8
    assert window["refill_unit"] == "whole-team"
    assert [len(slot["attempts"]) for slot in window["slots"]] == [2, 1]
    assert all(
        len(attempt["agent_actor_ids"]) == 4
        for slot in window["slots"]
        for attempt in slot["attempts"]
    )


def test_coral_window_rejects_partial_agent_refill_group() -> None:
    actors = _team_attempt(0, 0, 0, "task-a")[:-1]

    with pytest.raises(ValidationError, match="complete four-agent group"):
        _concurrency_window(
            adapter="coral",
            workload={"concurrency": 1, "coral_team_size": 4},
            actors=actors,
        )


def test_coral_manifest_and_plots_use_agent_children_not_parent_rows() -> None:
    actors = [
        *_team_attempt(0, 0, 0, "frontier_cs_algo/14"),
        *_team_attempt(1, 0, 1, "frontier_cs_algo/169"),
    ]
    window = _concurrency_window(
        adapter="coral",
        workload={"concurrency": 2, "coral_team_size": 4},
        actors=actors,
    )
    assert window is not None
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "bundle_id": "bundle",
        "adapter": "coral",
        "workload": {"concurrency": 2, "coral_team_size": 4},
        "counts": {},
        "actors": actors,
        "lanes": [
            {"actor_id": actor["actor_id"], "path": f"{index}.jsonl", "counts": {}}
            for index, actor in enumerate(actors)
        ],
        "window": {"gate_at_ns": 0, "terminal_at_ns": 1},
        "concurrency_window": window,
    }

    validate_manifest(manifest)
    bundle = SimpleNamespace(
        manifest=manifest,
        actor_ids=lambda: [actor["actor_id"] for actor in actors],
    )
    work_actors = _work_actors(bundle)
    lanes = {
        actor["actor_id"]: actor["lane"]
        for actor in actors
        if actor["lane"]["lane_kind"] == "agent"
    }

    assert len(work_actors) == 8
    assert all("-agent-" in actor for actor in work_actors)
    assert _display_actor(work_actors[0], lanes) == "S00/G00/A1  14"
    assert _display_actor(work_actors[-1], lanes) == "S01/G00/A4  169"


def test_actor_inventory_keeps_coral_parent_child_lane_metadata(tmp_path) -> None:
    ready = tmp_path / "ready"
    bindings = tmp_path / "bindings"
    stage = tmp_path / "stage"
    ready.mkdir()
    bindings.mkdir()
    stage.mkdir()
    parent_lane = _team_attempt(0, 0, 0, "task-a")[0]["lane"]
    child_lane = _team_attempt(0, 0, 0, "task-a")[1]["lane"]
    (ready / "team.json").write_text(
        json.dumps(
            {
                "actor_id": "team-0-0",
                "source_actor_id": "task-a",
                "process_role": "coral-team",
                "actor_metadata": parent_lane,
            }
        )
    )
    (bindings / "agent.json").write_text(
        json.dumps(
            {
                "actor_id": "team-0-0-agent-1",
                "source_actor_id": "task-a--agent-1",
                "actor_metadata": child_lane,
            }
        )
    )

    actors = _actors_from_stage(stage, ready, lane_binding_dir=bindings)
    by_actor = {actor["actor_id"]: actor for actor in actors}

    assert by_actor["team-0-0"]["lane"] == parent_lane
    assert by_actor["team-0-0-agent-1"]["lane"] == child_lane
    assert by_actor["team-0-0-agent-1"]["source_actor_ids"] == [
        "task-a--agent-1"
    ]
