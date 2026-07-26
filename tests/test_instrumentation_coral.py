from __future__ import annotations

import json
from types import SimpleNamespace

from minireplay.instrumentation import coral
from minireplay.replay_control import mark_session_prefix_consumed
from minireplay.util import read_json


class _FakeState:
    def cover(self, *_args) -> None:
        pass

    def snapshot_registry(self, *_args, **_kwargs) -> None:
        pass


def test_coral_provider_uses_proxy_v1_url_exactly_once(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / ".opencode"
    config_dir.mkdir()
    config_path = config_dir / "opencode.json"
    config_path.write_text(
        json.dumps({"provider": {"vllm": {"options": {}}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("NATIVE_REPLAY_PROXY_URL", "http://127.0.0.1:4567/v1")
    monkeypatch.setattr(coral, "state", lambda: _FakeState())

    coral._inject_plugin(tmp_path, "vllm/model")

    value = json.loads(config_path.read_text(encoding="utf-8"))
    assert value["provider"]["vllm"]["options"]["baseURL"] == "http://127.0.0.1:4567/v1"


def test_coral_agent_target_is_resolved_before_actor_hashing(monkeypatch) -> None:
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TASK_ID", "frontier_cs_algo/14")
    monkeypatch.setenv("NATIVE_REPLAY_TARGET_MAP", "{}")
    monkeypatch.setenv(
        "NATIVE_REPLAY_ROLE_TARGETS",
        json.dumps(
            {
                "agent-1": "agent-1",
                "agent-2": "agent-2",
                "agent-3": "agent-3",
                "agent-4": "agent-4",
            }
        ),
    )
    monkeypatch.setenv("NATIVE_REPLAY_TARGET_ID", "agent-1")

    assert coral._agent_target("agent-4") == "agent-4"


def test_coral_restart_gets_a_distinct_invocation_session_prefix(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / ".coral_agent_id").write_text("agent-2")
    observed: list[dict[str, str]] = []

    def start(_runtime, worktree_path, model, gateway_url=None):
        assert worktree_path == tmp_path
        assert model == "vllm/model"
        observed.append(dict(coral._SPAWN_ENV.get() or {}))
        return SimpleNamespace()

    monkeypatch.setattr(coral, "_inject_plugin", lambda *_args: None)
    monkeypatch.setattr(coral, "_agent_actor", lambda _agent: "actor-2")
    monkeypatch.setattr(coral, "_agent_target", lambda _agent: "vllm-8001")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TASK_ID", "frontier_cs_algo/14")
    monkeypatch.setenv("NATIVE_REPLAY_PROXY_URL", "http://127.0.0.1:4567/v1")
    coral._INVOCATION_COUNTS.clear()
    wrapped = coral._runtime_start_factory(start)

    wrapped(SimpleNamespace(), tmp_path, "vllm/model")
    wrapped(SimpleNamespace(), tmp_path, "vllm/model")

    assert [value["NATIVE_REPLAY_INVOCATION_INDEX"] for value in observed] == ["0", "1"]
    assert [value["NATIVE_REPLAY_INVOCATION_ID"] for value in observed] == [
        "actor-2/invocation-0",
        "actor-2/invocation-1",
    ]


def test_coral_grader_declares_stable_pending_attempt_artifact(monkeypatch) -> None:
    observed: dict = {}
    attempt = SimpleNamespace(
        agent_id="agent-2",
        commit_hash="abc123",
        status="pending",
    )
    config = SimpleNamespace(
        grader=SimpleNamespace(entrypoint=None, timeout=30),
    )

    def fake_run_grader(**kwargs):
        observed.update(kwargs)
        return kwargs["invoke"]()

    monkeypatch.setattr(coral, "_agent_actor", lambda _agent: "actor-2")
    monkeypatch.setattr(coral, "set_context", lambda **_kwargs: ("tokens",))
    monkeypatch.setattr(coral, "reset_context", lambda _tokens: None)
    monkeypatch.setattr(coral, "run_grader", fake_run_grader)
    monkeypatch.setattr(
        coral,
        "_attempt_logical_path",
        lambda _coral_dir, _attempt, _actor: "/coral-attempts/actor-2/g7-tree.json",
    )
    wrapped = coral._grade_factory(lambda *_args: "graded")

    assert wrapped(attempt, "config.toml", "/physical/run/.coral", config) == "graded"
    logical = "/coral-attempts/actor-2/g7-tree.json"
    assert observed["artifact_versions"]() == [coral._artifact_id(logical, 1)]


def test_coral_team_announces_readiness_before_framework_start(monkeypatch) -> None:
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TASK_ID", "frontier_cs_algo/14")
    monkeypatch.setenv("NATIVE_REPLAY_ACTOR_MAP", "{}")
    monkeypatch.setenv("NATIVE_REPLAY_TARGET_MAP", "{}")
    monkeypatch.setenv("NATIVE_REPLAY_ROLE_TARGETS", "{}")
    monkeypatch.setenv("NATIVE_REPLAY_TARGET_ID", "agent-1")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_SOURCE_TASK_ID", "frontier_cs_algo/14")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_RUN_INDEX", "0")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TEAM_SLOT", "0")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_SLOT_GENERATION", "0")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TEAM_SIZE", "4")
    calls: list[tuple[tuple, dict]] = []
    tokens = ("context-token",)

    def ready(*args, **kwargs):
        calls.append((args, kwargs))
        return tokens

    monkeypatch.setattr(coral, "ready_and_wait", ready)
    manager = SimpleNamespace()
    wrapped = coral._start_all_factory(lambda self: "started")

    assert wrapped(manager) == "started"
    assert calls == [
        (
            ("frontier_cs_algo/14",),
            {
                "process_role": "coral-team",
                "session_id": "frontier_cs_algo/14",
                "llm_role": "coral-team",
                "target_id": "agent-1",
                "actor_metadata": {
                    "framework": "coral",
                    "lane_kind": "team",
                    "concurrency_unit": "coral-team",
                    "team_slot": 0,
                    "slot_generation": 0,
                    "run_index": 0,
                    "team_size": 4,
                    "source_task_id": "frontier_cs_algo/14",
                },
            },
        )
    ]
    assert manager._minireplay_task_tokens == tokens
    assert manager._minireplay_monitor_seen_override == set()


def test_coral_monitor_observes_attempts_created_during_agent_startup() -> None:
    manager = SimpleNamespace(_minireplay_monitor_seen_override=set())
    wrapped = coral._get_seen_attempts_factory(lambda _self: {"attempt-created-during-startup"})

    assert wrapped(manager) == set()
    assert not hasattr(manager, "_minireplay_monitor_seen_override")
    assert wrapped(manager) == {"attempt-created-during-startup"}


def test_coral_recorded_restart_interrupts_immediately(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[bool] = []
    actor = "actor-2"
    monkeypatch.setenv("NATIVE_REPLAY_MODE", "replay")
    monkeypatch.setenv("NATIVE_REPLAY_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TASK_ID", "frontier_cs_algo/14")
    monkeypatch.setattr(coral, "_agent_actor", lambda _agent: actor)
    coral._INVOCATION_COUNTS.clear()
    coral._INVOCATION_COUNTS["frontier_cs_algo/14--agent-2"] = 1
    mark_session_prefix_consumed(
        tmp_path,
        actor,
        f"{actor}/invocation-0/root-0",
    )
    wrapped = coral._request_interrupt_factory(
        lambda _self, *, at_turn_boundary: calls.append(at_turn_boundary) or "mode"
    )
    handle = SimpleNamespace(
        agent_id="agent-2",
        _minireplay_recorded_restart=True,
    )

    assert wrapped(handle, at_turn_boundary=True) == "mode"
    assert calls == [False]


def test_coral_live_replay_heartbeat_is_always_suppressed(monkeypatch) -> None:
    calls: list[tuple] = []
    handle = SimpleNamespace(agent_id="agent-2")
    manager = SimpleNamespace(handles=[handle])
    monkeypatch.setenv("NATIVE_REPLAY_MODE", "replay")
    wrapped = coral._schedule_interrupt_and_resume_factory(
        lambda *_args, **_kwargs: calls.append((_args, _kwargs)) or True
    )

    assert (
        wrapped(
            manager,
            0,
            "feedback",
            prompt_source="heartbeat:reflect",
        )
        is False
    )
    assert calls == []


def test_coral_recorded_restart_waits_for_prefix_and_trigger_grader(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple[int, str, str]] = []
    actor = "actor-2"
    session = f"{actor}/invocation-0/root-0"
    handle = SimpleNamespace(agent_id="agent-2")

    def schedule(idx, prompt, prompt_source=None):
        assert handle._minireplay_recorded_restart is True
        calls.append((idx, prompt, prompt_source))
        return True

    manager = SimpleNamespace(
        handles=[handle],
        _pending_resumes={},
        _schedule_interrupt_and_resume=schedule,
    )
    monkeypatch.setenv("NATIVE_REPLAY_MODE", "replay")
    monkeypatch.setenv("NATIVE_REPLAY_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TASK_ID", "frontier_cs_algo/14")
    monkeypatch.setenv(
        "NATIVE_REPLAY_CORAL_CONTROLS",
        json.dumps(
            [
                {
                    "actor_id": actor,
                    "invocation_index": 1,
                    "source": "heartbeat:reflect",
                    "prompt": "recorded feedback",
                    "trigger_grader_attempt_id": "grader-1",
                }
            ]
        ),
    )
    monkeypatch.setattr(coral, "_agent_actor", lambda _agent: actor)
    coral._INVOCATION_COUNTS.clear()
    coral._INVOCATION_COUNTS["frontier_cs_algo/14--agent-2"] = 1
    mark_session_prefix_consumed(tmp_path, actor, session)
    original_calls: list[bool] = []
    wrapped = coral._advance_pending_resumes_factory(
        lambda _self: original_calls.append(True)
    )

    wrapped(manager)
    assert calls == []

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "graders.jsonl").write_text(
        json.dumps({"attempt_id": "grader-1"}) + "\n"
    )
    wrapped(manager)

    assert calls == [(0, "recorded feedback", "minireplay-recorded:heartbeat:reflect")]
    assert original_calls == [True, True]
    assert not hasattr(handle, "_minireplay_recorded_restart")


def test_coral_agent_declares_parent_and_four_agent_team_slot(
    tmp_path,
    monkeypatch,
) -> None:
    binding_dir = tmp_path / "bindings"
    monkeypatch.setenv("NATIVE_REPLAY_MODE", "record")
    monkeypatch.setenv("NATIVE_REPLAY_ACTOR_MAP", "{}")
    monkeypatch.setenv("NATIVE_REPLAY_LANE_BINDING_DIR", str(binding_dir))
    monkeypatch.delenv("NATIVE_REPLAY_ACTORS", raising=False)
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TASK_ID", "frontier_cs_algo/999")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_SOURCE_TASK_ID", "frontier_cs_algo/999")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_RUN_INDEX", "9")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TEAM_SLOT", "2")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_SLOT_GENERATION", "1")
    monkeypatch.setenv("NATIVE_REPLAY_CORAL_TEAM_SIZE", "4")

    actor = coral._agent_actor("agent-4")

    bindings = [read_json(path) for path in binding_dir.glob("*.json")]
    assert len(bindings) == 1
    assert bindings[0]["actor_id"] == actor
    assert bindings[0]["source_actor_id"] == "frontier_cs_algo/999--agent-4"
    assert bindings[0]["actor_metadata"] == {
        "framework": "coral",
        "lane_kind": "agent",
        "concurrency_unit": "coral-team",
        "team_slot": 2,
        "slot_generation": 1,
        "run_index": 9,
        "team_size": 4,
        "source_task_id": "frontier_cs_algo/999",
        "agent_id": "agent-4",
        "agent_index": 4,
        "parent_actor_id": coral.resolve_actor("frontier_cs_algo/999"),
    }


def test_coral_replay_teardown_does_not_publish_a_false_task_failure(
    monkeypatch,
) -> None:
    manager = SimpleNamespace(
        _minireplay_task_tokens=("context-token",),
        _one_shot_terminal=False,
        _one_shot_failure=None,
    )
    terminals: list[dict] = []
    monkeypatch.setattr(coral, "report_task_terminal", lambda **value: terminals.append(value))
    monkeypatch.setattr(coral, "reset_context", lambda _tokens: None)
    wrapped = coral._monitor_factory(lambda self: "stopped")

    assert wrapped(manager) == "stopped"
    assert terminals == []
    assert manager._minireplay_task_tokens is None
