from __future__ import annotations

import argparse
import os
from pathlib import Path

from .instrumentation.gate import ready_and_wait


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--ready-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a native command is required after --")
    os.environ["NATIVE_REPLAY_READY_DIR"] = str(args.ready_dir)
    os.environ["NATIVE_REPLAY_START_GATE"] = str(args.gate)
    os.environ["NATIVE_REPLAY_GATE_TIMEOUT_S"] = str(args.timeout_s)
    ready_and_wait(args.actor, process_role=args.role)
    # ContextVars do not survive exec; the canonical actor is available from
    # the ready file name and resolve_actor is deterministic.
    from .instrumentation.gate import resolve_actor

    os.environ["NATIVE_REPLAY_ACTOR_ID"] = resolve_actor(args.actor)
    os.environ["NATIVE_REPLAY_PROCESS_ROLE"] = args.role
    os.environ.setdefault("NATIVE_REPLAY_SESSION_ID", os.environ["NATIVE_REPLAY_ACTOR_ID"])
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
