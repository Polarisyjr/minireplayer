from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path


def _load_environment(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load exec-gate environment {path}: {exc}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise RuntimeError("exec-gate environment must be a string-to-string object")
    return value


def main() -> int:
    arguments = sys.argv[1:]
    if len(arguments) < 3 or arguments[0] != "--env-file" or "--" not in arguments:
        print(
            "usage: python -m minireplay.exec_gate --env-file PATH -- COMMAND ...",
            file=sys.stderr,
        )
        return 2
    separator = arguments.index("--")
    if separator != 2 or not arguments[separator + 1 :]:
        print(
            "native replay exec gate requires one environment file and a command", file=sys.stderr
        )
        return 2
    environment_file = Path(arguments[1])
    command = arguments[separator + 1 :]

    # The launcher intentionally starts this interpreter without replay
    # instrumentation.  SIGSTOP lets the supervisor attach cgroup/eBPF
    # collectors before this process reads the replay environment and execs
    # the native framework entrypoint.
    os.kill(os.getpid(), signal.SIGSTOP)
    environment = _load_environment(environment_file)
    os.environ.clear()
    os.environ.update(environment)
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
