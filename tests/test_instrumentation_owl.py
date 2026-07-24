from __future__ import annotations

import minireplay.instrumentation.owl as owl


def test_process_pool_tasks_use_one_native_lane_per_worker(monkeypatch) -> None:
    submitted: list[tuple[object, ...]] = []

    def original(_executor, *args, **_kwargs):
        submitted.append(args)
        return object()

    wrapped = owl._submit_factory(original)

    def _proc_run_one():
        return None

    wrapped(object(), _proc_run_one, {"task_id": "source-task"})
    wrapped(object(), _proc_run_one, {"task_id": "refill-task"})

    assert submitted[0][0] is owl.gated_terminal_callable
    assert submitted[0][2] == "source-task"
    assert submitted[0][-1] == "process-worker"
    assert submitted[1][2] == "refill-task"
    assert submitted[1][-1] == "process-worker"


def test_serial_refills_use_one_native_lane(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    def gated(*args):
        calls.append(args)
        return "result"

    monkeypatch.setenv("NATIVE_REPLAY_SCOPE", "C1")
    monkeypatch.setattr(owl, "_TASK_SUBMISSIONS", iter([0, 1]))
    monkeypatch.setattr(owl, "_serial_source_actor", lambda: "source-task")
    monkeypatch.setattr(owl, "gated_terminal_callable", gated)
    wrapped = owl._serial_workforce_factory(lambda _workforce: "native")

    assert wrapped(object()) == "result"
    assert wrapped(object()) == "result"
    assert calls[0][1] == "source-task"
    assert calls[0][-1] == "serial-worker"
    assert calls[1][1] == "refill-000001"
    assert calls[1][-1] == "serial-worker"
