from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from minireplay.constants import SUPPORTED_ADAPTERS
from minireplay.sdk import set_context

from .http import install_http_identity
from .process import install_process_capture
from .state import initialize


def is_framework_interpreter(configured: str, current: str) -> bool:
    return os.path.realpath(configured) == os.path.realpath(current)


def is_harness_process_role(role: str | None) -> bool:
    return role == "native-batch"


def is_inline_python_helper(argv0: str) -> bool:
    return argv0 in {"-", "-c"}


def is_multiprocessing_spawn_helper(argv: list[str]) -> bool:
    return argv and argv[0] == "-c" and "--multiprocessing-fork" in argv[1:]


def is_sweep_harness_script(argv0: str) -> bool:
    path = Path(argv0)
    parts = set(path.parts)
    return bool(
        {"step1_imbalance", "step2_uarch"} & parts
        or ("scripts" in parts and "lib" in parts)
    )


def install() -> None:
    adapter = os.environ.get("NATIVE_REPLAY_ADAPTER", "")
    if not adapter:
        return
    if adapter not in SUPPORTED_ADAPTERS:
        raise RuntimeError(f"unsupported native replay adapter: {adapter}")
    if adapter == "coral" and Path(sys.argv[0]).name == "run_queue.py":
        from .sweep_bridge import install_coral_queue_bridge

        install_coral_queue_bridge()
        return
    if is_sweep_harness_script(sys.argv[0]):
        return
    if is_harness_process_role(os.environ.get("NATIVE_REPLAY_PROCESS_ROLE")):
        return
    if is_inline_python_helper(sys.argv[0]) and not is_multiprocessing_spawn_helper(sys.argv):
        return
    if os.environ.get("NATIVE_REPLAY_TOOL_CHILD") == "1":
        return
    framework_python = os.environ.get("NATIVE_REPLAY_FRAMEWORK_PYTHON")
    if framework_python and not is_framework_interpreter(framework_python, sys.executable):
        return
    initialize(adapter)
    actor_id = os.environ.get("NATIVE_REPLAY_ACTOR_ID")
    if actor_id:
        set_context(
            actor_id=actor_id,
            process_role=os.environ.get("NATIVE_REPLAY_PROCESS_ROLE", "framework"),
            parent_span_id=os.environ.get("NATIVE_REPLAY_PARENT_SPAN_ID"),
            session_id=os.environ.get("NATIVE_REPLAY_SESSION_ID", actor_id),
            llm_role=os.environ.get("NATIVE_REPLAY_LLM_ROLE"),
            target_id=os.environ.get("NATIVE_REPLAY_TARGET_ID"),
        )
    install_http_identity()
    module = importlib.import_module(f"minireplay.instrumentation.{adapter.replace('-', '_')}")
    install_process_capture()
    try:
        module.install()
    except ImportError as exc:
        # Every interpreter that inherits PYTHONPATH lands here, including tools the
        # sweep merely shells out to (conda, for one). Failing is right — a silent
        # skip is how an uninstrumented run gets reported as a real one — but say
        # which knob confines instrumentation to the framework's own interpreter.
        raise RuntimeError(
            f"native replay adapter {adapter!r} cannot patch {sys.executable}: {exc}. "
            "If this interpreter is not the framework's, declare the one that is via "
            "NATIVE_REPLAY_FRAMEWORK_PYTHON (config.env)."
        ) from exc
