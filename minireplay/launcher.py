"""Build the native sweep command.

The sweep script owns task selection, concurrency, refill, the measurement window
and the ``sample_end`` boundary. This module only chooses the sweep invocation and
redirects its output paths; it never substitutes its own batch. Everything it can
change is passed through the environment, so the argv is exactly the sweep the
experiment already runs, plus ``-s none`` (attach no step1/step2 profiler, because
the replayer does its own measurement).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig

DEFAULT_WARMUP_S = 15

# A unix socket path is capped at ~108 bytes, and tmux puts its socket at
# ``$TMUX_TMPDIR/tmux-<uid>/default``. A run directory deep enough to overflow that
# fails inside the framework's own launcher ("error connecting to ... File name too
# long"), which is nowhere near whoever chose ``--out``, so keep the socket beside
# the run only while it fits.
_SOCKET_PATH_LIMIT = 100


def tmux_tmpdir(run_root: Path, run_id: str) -> Path:
    natural = run_root / "tmux"
    socket = natural / f"tmux-{os.getuid()}" / "default"
    if len(str(socket)) <= _SOCKET_PATH_LIMIT:
        return natural
    return Path(tempfile.gettempdir()) / f"minireplay-tmux-{run_id}"


@dataclass(frozen=True)
class NativeCommand:
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]


def sweep_command(config: RunConfig, *, duration_s: int | None = None) -> tuple[str, ...]:
    """Build the common native sweep command.

    ``config.duration_s`` is the source sampling window.  Replay may keep the
    native driver alive for longer because its boundary is fixed-work completion,
    not another wall-clock cutoff.
    """

    script = config.repo / "scripts" / config.framework / "sweep.sh"
    command = ["bash", str(script)]
    if config.framework == "coral":
        command.append(config.coral_dataset)
    command.extend(["-c", str(config.concurrency)])
    # Refill is one framework-independent workload dimension. Each native sweep
    # owns the implementation behind this common flag; the replayer never invents
    # a task list or scheduler.
    command.append("--refill" if config.refill else "--no-refill")
    command.extend(
        [
            "--seed",
            str(config.seed),
            "-s",
            "none",
            "-d",
            str(config.duration_s if duration_s is None else duration_s),
        ]
    )
    if config.framework == "coral":
        command.extend(
            [
                "--agent-turns",
                str(config.coral_agent_turns),
                "--global-turns",
                str(config.coral_global_turns),
                (
                    "--restart-exited"
                    if config.coral_restart_exited
                    else "--no-restart-exited"
                ),
            ]
        )
    return tuple(command)


def native_command(
    config: RunConfig,
    run_root: Path,
    *,
    run_id: str,
    skip_vllm: bool,
    record: bool,
    duration_s: int | None = None,
) -> NativeCommand:
    warm_vllm = not skip_vllm
    environment = {
        "PHASE_EVENTS_PATH": str(run_root / "sweep-phase-events.jsonl"),
        "SWEEP_RESULTS_ROOT": str(run_root / "sweep-results"),
        "SWEEP_RUN_ID": run_id,
        "SWEEP_MODE": "minireplay",
        "TMUX_TMPDIR": str(tmux_tmpdir(run_root, run_id)),
        # Recording and full replay warm the upstream directly before the actor
        # gate. The native sweep resets prefix/KV state both before and after this
        # traffic, so kernels and execution paths are hot without leaving warmup
        # prompts in the measured cache. Tool-only replay has no vLLM to warm.
        "SWEEP_WARMUP": str(DEFAULT_WARMUP_S) if warm_vllm else "0",
        "SWEEP_RANDOM_WARMUP": "1" if warm_vllm else "0",
        "SWEEP_WARMUP_CONCURRENCY": "8",
        "SWEEP_LABEL_TS": "0",
    }
    if skip_vllm:
        environment["SWEEP_SKIP_VLLM"] = "1"
    environment.update(config.env)
    # Config deployment overrides cannot accidentally make tool-only replay send
    # traffic to the vLLM endpoint it explicitly skips.
    if skip_vllm:
        environment["SWEEP_WARMUP"] = "0"
        environment["SWEEP_RANDOM_WARMUP"] = "0"
    return NativeCommand(
        command=sweep_command(config, duration_s=duration_s),
        cwd=config.repo,
        environment=environment,
    )
