from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from .patching import patch_method

_RUN = re.compile(r"^run-([0-9]{6})-")


def _coral_source_id(task: str) -> str:
    parts = Path(task).resolve().parts
    try:
        index = parts.index("examples")
    except ValueError as exc:
        raise RuntimeError(f"CORAL sweep task is outside examples: {task}") from exc
    relative = parts[index + 1 : -1]
    if not relative:
        raise RuntimeError(f"CORAL sweep task has no logical identity: {task}")
    return "/".join(relative)


def _run_index(command: list[str]) -> int:
    for value in command:
        if not value.startswith("workspace.run_dir="):
            continue
        name = Path(value.split("=", 1)[1]).name
        match = _RUN.match(name)
        if match is not None:
            return int(match.group(1))
    raise RuntimeError("CORAL sweep launch has no run index")


def _queue_popen_factory(original):
    def wrapped(self, *args: Any, **kwargs: Any) -> None:
        command = args[0] if args else kwargs.get("args")
        if (
            isinstance(command, (list, tuple))
            and len(command) >= 3
            and Path(str(command[1])).name == "coral-vllm.sh"
        ):
            values = [str(value) for value in command]
            run_index = _run_index(values)
            concurrency = int(os.environ["NATIVE_REPLAY_CONCURRENCY"])
            source = _coral_source_id(values[2])
            actor_source = source if run_index < concurrency else f"refill-{run_index:06d}"
            environment = dict(kwargs.get("env") or os.environ)
            environment.update(
                {
                    "NATIVE_REPLAY_CORAL_TASK_ID": actor_source,
                    "NATIVE_REPLAY_PROCESS_ROLE": "framework",
                }
            )
            kwargs["env"] = environment
        original(self, *args, **kwargs)

    return wrapped


def install_coral_queue_bridge() -> None:
    patch_method(subprocess.Popen, "__init__", _queue_popen_factory)
