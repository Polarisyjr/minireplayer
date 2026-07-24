"""Cross-worker serialization for refill tasks bound to one causal lane."""

from __future__ import annotations

import threading

from minireplay.instrumentation.gate import _causal_lane_lock


def test_same_refill_lane_is_serialized(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NATIVE_REPLAY_LANE_BINDING_DIR", str(tmp_path))
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with _causal_lane_lock("actor-0", "process-worker"):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second() -> None:
        assert first_entered.wait(timeout=2)
        with _causal_lane_lock("actor-0", "process-worker"):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_entered.is_set()


def test_distinct_refill_lanes_do_not_block_each_other(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NATIVE_REPLAY_LANE_BINDING_DIR", str(tmp_path))
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()

    def hold(actor_id: str, entered: threading.Event) -> None:
        with _causal_lane_lock(actor_id, "process-worker"):
            entered.set()
            assert release.wait(timeout=2)

    threads = [
        threading.Thread(target=hold, args=("actor-0", first_entered)),
        threading.Thread(target=hold, args=("actor-1", second_entered)),
    ]
    for thread in threads:
        thread.start()
    assert first_entered.wait(timeout=2)
    assert second_entered.wait(timeout=2)
    release.set()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
