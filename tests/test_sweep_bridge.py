from __future__ import annotations

from minireplay.instrumentation.sweep_bridge import _queue_popen_factory


def test_coral_queue_bridge_preserves_team_slot_for_refill(monkeypatch) -> None:
    monkeypatch.setenv("NATIVE_REPLAY_CONCURRENCY", "8")
    captured: dict = {}

    def original(_self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    wrapped = _queue_popen_factory(original)
    command = [
        "bash",
        "/repo/scripts/coral/coral-vllm.sh",
        "/repo/examples/frontier_cs_algo/14/task.yaml",
        "agents.count=4",
        "workspace.run_dir=/runs/run-000008-14_task",
    ]
    environment = {
        "MULTIAGENT_CORAL_RUN_INDEX": "8",
        "MULTIAGENT_CORAL_TEAM_SLOT": "3",
        "MULTIAGENT_CORAL_SLOT_GENERATION": "1",
        "MULTIAGENT_CORAL_TEAM_SIZE": "4",
    }

    wrapped(object(), command, env=environment)

    child = captured["kwargs"]["env"]
    assert child["NATIVE_REPLAY_CORAL_TASK_ID"] == "refill-000008"
    assert child["NATIVE_REPLAY_CORAL_SOURCE_TASK_ID"] == "frontier_cs_algo/14"
    assert child["NATIVE_REPLAY_CORAL_RUN_INDEX"] == "8"
    assert child["NATIVE_REPLAY_CORAL_TEAM_SLOT"] == "3"
    assert child["NATIVE_REPLAY_CORAL_SLOT_GENERATION"] == "1"
    assert child["NATIVE_REPLAY_CORAL_TEAM_SIZE"] == "4"
