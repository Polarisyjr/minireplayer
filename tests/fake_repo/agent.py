"""A stand-in agent.

It drives the same boundary protocol a real adapter drives — gate, LLM call,
dispatch, tool — but calls the SDK directly instead of being monkey-patched into a
framework. That keeps the integration test about the harness (supervisor, services,
gate, ledger, cutoff, bundle) rather than about any one framework's internals.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from minireplay.instrumentation.gate import ready_and_wait
from minireplay.observation import recorded_output_result_contract
from minireplay.sdk import (
    last_llm_attempt_id,
    llm_identity_headers,
    remember_llm_attempt,
    report_task_terminal,
    run_dispatch,
    run_tool,
)

TOOL_CALLS = int(os.environ.get("FAKE_AGENT_TOOL_CALLS", "2"))
# Lets a test make the agent call the same tool with different arguments, which is
# drift, as opposed to calling it more times, which is just an unfinished episode.
ARGUMENT_SALT = os.environ.get("FAKE_AGENT_ARGUMENT_SALT", "")


def call_llm(prompt: str) -> dict:
    body = json.dumps(
        {"model": "fake", "messages": [{"role": "user", "content": prompt}], "stream": False}
    ).encode()
    request = urllib.request.Request(
        f"{os.environ['NATIVE_REPLAY_PROXY_URL']}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", **llm_identity_headers()},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        remember_llm_attempt(response.headers.get("X-Native-Replay-Attempt"))
        return json.load(response)


def native_tool(index: int) -> dict:
    """Real work: touch the filesystem so the run has something to measure."""

    marker = os.path.join(os.environ["NATIVE_REPLAY_RUN_ROOT"], f"tool-{index}.txt")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(f"call {index} from pid {os.getpid()}\n")
    return {"output": f"wrote {marker}", "exit_code": 0, "pid": os.getpid()}


def main() -> int:
    source_actor_id = sys.argv[1]
    ready_and_wait(source_actor_id, process_role="fake-agent")

    for index in range(TOOL_CALLS):
        call_llm(f"step {index}")
        trigger = last_llm_attempt_id()
        assert trigger, "the proxy did not return an attempt id"

        def invoke(index: int = index):
            return run_tool(
                name="write_file",
                implementation="fake.write_file",
                arguments={"index": index, "salt": ARGUMENT_SALT},
                invoke=lambda: native_tool(index),
                result_contract=recorded_output_result_contract("/output"),
                result_replayer=lambda native, recorded: {**native, **recorded},
            )

        run_dispatch(
            name="write_file",
            arguments={"index": index, "salt": ARGUMENT_SALT},
            parser_identity="fake.parser",
            dispatcher_identity="fake.dispatcher",
            native_call_id=f"call-{index}",
            origin_kind="llm_structured",
            trigger_id=trigger,
            invoke=invoke,
        )

    report_task_terminal(
        result={"calls": TOOL_CALLS},
        status=os.environ.get("FAKE_AGENT_TERMINAL_STATUS", "success"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
