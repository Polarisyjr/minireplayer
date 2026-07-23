"""The sweep command must be the experiment's own, with output redirected only."""

from __future__ import annotations

from pathlib import Path

import pytest

from minireplay.config import RunConfig
from minireplay.launcher import native_command, sweep_command

REPO = Path("/repo")


def config(
    framework: str,
    concurrency: int = 1,
    duration_s: int = 1200,
    refill: bool = True,
    env: dict[str, str] | None = None,
) -> RunConfig:
    return RunConfig(
        framework=framework,
        repo=REPO,
        concurrency=concurrency,
        duration_s=duration_s,
        seed=42,
        targets={"vllm-8000": "http://127.0.0.1:8000"},
        refill=refill,
        env={} if env is None else env,
    )


@pytest.mark.parametrize("framework", ("mini-swe", "trae", "coral", "owl"))
@pytest.mark.parametrize("concurrency", (1, 8, 32))
def test_command_is_the_seeded_sweep_with_no_profiler(framework: str, concurrency: int) -> None:
    command = sweep_command(config(framework, concurrency))

    expected = ["bash", f"/repo/scripts/{framework}/sweep.sh"]
    if framework == "coral":
        expected.append("frontier_cs_algo")
    expected += [
        "-c",
        str(concurrency),
        "--refill",
        "--seed",
        "42",
        "-s",
        "none",
        "-d",
        "1200",
    ]
    assert list(command) == expected


def test_no_task_selection_flag_is_ever_passed() -> None:
    """Task choice belongs to the sweep's seeded order, never to the replayer."""

    for framework in ("mini-swe", "trae", "coral", "owl"):
        command = sweep_command(config(framework))
        assert "--idx" not in command
        assert "--instances" not in command


@pytest.mark.parametrize("concurrency", (1, 8, 32))
def test_refill_leaves_task_selection_to_the_native_sweep(concurrency: int) -> None:
    """Refill must draw the next seeded task rather than pinning a C-sized backlog.

    Every sweep defaults ``-n`` to the whole seeded pool. Pinning it to the
    concurrency level leaves the queue holding only the tasks already in flight, so a
    finishing task can only be replaced by itself — not a load model, and the
    workload override design §2 forbids.
    """

    for framework in ("mini-swe", "trae", "coral", "owl"):
        command = sweep_command(config(framework, concurrency))
        assert "-n" not in command
        assert "--num-tasks" not in command
        assert "--refill" in command
        assert "--no-refill" not in command


@pytest.mark.parametrize("concurrency", (1, 8, 32))
def test_no_refill_is_the_same_native_sweep_flag_for_every_framework(
    concurrency: int,
) -> None:
    for framework in ("mini-swe", "trae", "coral", "owl"):
        command = sweep_command(config(framework, concurrency, refill=False))
        assert "--no-refill" in command
        assert "--refill" not in command
        assert "-n" not in command
        assert "--num-tasks" not in command


def test_refill_is_part_of_every_frameworks_workload_identity() -> None:
    """A no-refill bundle must not be replayable with refill, or the reverse."""

    for framework in ("mini-swe", "trae", "coral", "owl"):
        with_refill = config(framework, 8).workload_identity()
        without_refill = config(framework, 8, refill=False).workload_identity()
        assert with_refill != without_refill
        assert with_refill["refill"] is True
        assert without_refill["refill"] is False


def test_only_output_paths_are_redirected() -> None:
    native = native_command(
        config("mini-swe"), Path("/run"), run_id="r-0", skip_vllm=False, record=True
    )
    assert native.environment["PHASE_EVENTS_PATH"] == "/run/sweep-phase-events.jsonl"
    assert native.environment["SWEEP_RESULTS_ROOT"] == "/run/sweep-results"
    assert native.environment["TMUX_TMPDIR"] == "/run/tmux"
    assert native.environment["SWEEP_RUN_ID"] == "r-0"
    assert "SWEEP_SKIP_VLLM" not in native.environment


def test_tool_only_replay_skips_the_vllm_preflight() -> None:
    """Tool-only replay answers the LLM lane itself, so no vLLM pool exists to check."""

    native = native_command(
        config("mini-swe"), Path("/run"), run_id="r-0", skip_vllm=True, record=False
    )
    assert native.environment["SWEEP_SKIP_VLLM"] == "1"
    assert native.environment["SWEEP_WARMUP"] == "0"
    assert native.environment["SWEEP_RANDOM_WARMUP"] == "0"


def test_record_warms_the_sampler_before_actor_traffic() -> None:
    native = native_command(
        config("mini-swe"), Path("/run"), run_id="r-0", skip_vllm=False, record=True
    )
    assert native.environment["SWEEP_WARMUP"] == "15"
    assert native.environment["SWEEP_RANDOM_WARMUP"] == "1"
    assert native.environment["SWEEP_WARMUP_CONCURRENCY"] == "8"


def test_full_replay_warms_the_sampler_before_actor_traffic() -> None:
    native = native_command(
        config("mini-swe"), Path("/run"), run_id="r-0", skip_vllm=False, record=False
    )
    assert native.environment["SWEEP_WARMUP"] == "15"
    assert native.environment["SWEEP_RANDOM_WARMUP"] == "1"


def test_configured_warmup_is_shared_by_record_and_full_replay() -> None:
    native = native_command(
        config(
            "mini-swe",
            env={"SWEEP_WARMUP": "10", "SWEEP_RANDOM_WARMUP": "1"},
        ),
        Path("/run"),
        run_id="r-0",
        skip_vllm=False,
        record=False,
    )

    assert native.environment["SWEEP_WARMUP"] == "10"
    assert native.environment["SWEEP_RANDOM_WARMUP"] == "1"


def test_tool_only_replay_cannot_reenable_warmup_from_config() -> None:
    native = native_command(
        config(
            "mini-swe",
            env={"SWEEP_WARMUP": "10", "SWEEP_RANDOM_WARMUP": "1"},
        ),
        Path("/run"),
        run_id="r-0",
        skip_vllm=True,
        record=False,
    )

    assert native.environment["SWEEP_WARMUP"] == "0"
    assert native.environment["SWEEP_RANDOM_WARMUP"] == "0"


def test_config_environment_can_override_record_warmup() -> None:
    native = native_command(
        config(
            "mini-swe",
            env={
                "SWEEP_WARMUP": "3",
                "SWEEP_RANDOM_WARMUP": "0",
                "SWEEP_WARMUP_CONCURRENCY": "4",
            },
        ),
        Path("/run"),
        run_id="r-0",
        skip_vllm=False,
        record=True,
    )

    assert native.environment["SWEEP_WARMUP"] == "3"
    assert native.environment["SWEEP_RANDOM_WARMUP"] == "0"
    assert native.environment["SWEEP_WARMUP_CONCURRENCY"] == "4"


def test_duration_is_passed_through_for_smoke_windows() -> None:
    assert "60" in sweep_command(config("mini-swe", duration_s=60))


def test_replay_can_extend_native_driver_without_changing_workload_identity() -> None:
    cfg = config("mini-swe", duration_s=45)
    native = native_command(
        cfg,
        Path("/run"),
        run_id="r-0",
        skip_vllm=False,
        record=False,
        duration_s=1545,
    )

    assert native.command[-2:] == ("-d", "1545")
    assert cfg.workload_identity()["duration_s"] == 45
