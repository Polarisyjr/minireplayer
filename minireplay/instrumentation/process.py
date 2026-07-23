from __future__ import annotations

import os
import subprocess

from minireplay.sdk import current_context, record_subprocess_launch
from minireplay.serialization import jsonable

from .patching import patch_method
from .state import state


def _popen_factory(original):
    def wrapped(self, *args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        context = current_context()
        if context["actor_id"] != "unknown":
            supplied = kwargs.get("env")
            environment = dict(os.environ if supplied is None else supplied)
            inherited = {
                "NATIVE_REPLAY_ACTOR_ID": str(context["actor_id"]),
                "NATIVE_REPLAY_PROCESS_ROLE": str(context["process_role"]),
                "NATIVE_REPLAY_SESSION_ID": str(context["session_id"]),
                "NATIVE_REPLAY_LLM_ROLE": str(context["llm_role"]),
                "NATIVE_REPLAY_TARGET_ID": str(context["target_id"]),
            }
            for name, value in inherited.items():
                environment.setdefault(name, value)
            if context["parent_span_id"] is not None:
                environment["NATIVE_REPLAY_PARENT_SPAN_ID"] = str(context["parent_span_id"])
            else:
                environment.pop("NATIVE_REPLAY_PARENT_SPAN_ID", None)
            if context["tool_call_id"] is not None:
                environment["NATIVE_REPLAY_TOOL_CHILD"] = "1"
            kwargs["env"] = environment
        original(self, *args, **kwargs)
        record_subprocess_launch(
            "subprocess.Popen",
            runtime_pid=self.pid,
            command=jsonable(command),
            cwd=jsonable(kwargs.get("cwd", os.getcwd())),
            shell=bool(kwargs.get("shell", False)),
            executable=jsonable(kwargs.get("executable")),
        )

    return wrapped


def install_process_capture() -> None:
    patch_method(subprocess.Popen, "__init__", _popen_factory)
    state().mark("python-subprocess-capture")
