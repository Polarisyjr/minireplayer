from __future__ import annotations

import json

from minireplay.instrumentation.gate import bind_task_lane
from minireplay.util import read_json


def test_native_slot_binds_refill_tasks_to_one_lane(tmp_path, monkeypatch) -> None:
    binding_dir = tmp_path / "bindings"
    monkeypatch.setenv("NATIVE_REPLAY_MODE", "record")
    monkeypatch.setenv("NATIVE_REPLAY_ACTOR_MAP", "{}")
    monkeypatch.setenv("NATIVE_REPLAY_LANE_BINDING_DIR", str(binding_dir))
    monkeypatch.delenv("NATIVE_REPLAY_ACTORS", raising=False)

    assert bind_task_lane("generic-task-a", "worker-slot-test") == "generic-task-a"
    assert bind_task_lane("generic-task-b", "worker-slot-test") == "generic-task-a"

    bindings = [read_json(path) for path in sorted(binding_dir.glob("*.json"))]
    assert [(item["source_actor_id"], item["actor_id"]) for item in bindings] == [
        ("generic-task-a", "generic-task-a"),
        ("generic-task-b", "generic-task-a"),
    ]


def test_replay_uses_recorded_task_lane_instead_of_fresh_slot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NATIVE_REPLAY_MODE", "replay")
    monkeypatch.setenv(
        "NATIVE_REPLAY_ACTOR_MAP",
        json.dumps({"generic-task-c": "lane-recorded", "generic-task-d": "lane-recorded"}),
    )
    monkeypatch.setenv("NATIVE_REPLAY_LANE_BINDING_DIR", str(tmp_path / "bindings"))
    monkeypatch.setenv("NATIVE_REPLAY_ACTORS", json.dumps(["lane-recorded"]))

    assert bind_task_lane("generic-task-c", "fresh-worker-7") == "lane-recorded"
    assert bind_task_lane("generic-task-d", "fresh-worker-2") == "lane-recorded"
