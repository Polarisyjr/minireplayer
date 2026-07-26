"""The ``llm-only`` replay mode.

The LLM lane runs for real against the engine while each tool is held for the
duration the source observed and then answered from the recording. That keeps the
engine's request arrival pattern comparable to a full replay without entering any
native tool implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from minireplay.config import load_config
from minireplay.constants import CONFIG_SCHEMA
from minireplay.errors import InfrastructureError, MismatchError
from minireplay.instrumentation import mini_swe
from minireplay.services import ReplayServices
from minireplay.supervisor import Supervisor
from tests.support import make_bundle, tool

FAKE_REPO = Path(__file__).parent / "fake_repo"


class LedgerClient:
    """Give a ledger the client-side call shape the adapters actually use."""

    def __init__(self, ledger: Any) -> None:
        self._ledger = ledger

    def complete(self, reservation_id: str, **fields: Any) -> dict[str, Any]:
        return self._ledger.complete({"reservation_id": reservation_id, **fields})


def services(tmp_path: Path, bundle, replay_mode: str) -> ReplayServices:
    return ReplayServices(
        mode="replay",
        stage_dir=tmp_path,
        auth_token="token",
        adapter="mini-swe",
        upstreams={"vllm-8000": "http://127.0.0.1:8000"},
        run_root=tmp_path,
        repo=tmp_path,
        bundle=bundle,
        replay_mode=replay_mode,
    )


def start_tool() -> dict[str, Any]:
    return {
        "kind": "tool",
        "actor_id": "actor-0",
        "process_role": "agent",
        "started_at_ns": 100,
        "dispatch_id": "dispatch-0",
        "name": "shell",
        "implementation": "native-shell",
        "arguments": {"command": "echo hi"},
        "result_contract": tool()["result_contract"],
    }


def test_llm_only_publishes_the_recorded_tool_duration(tmp_path: Path) -> None:
    bundle = make_bundle(tools=[tool(started=1_000, ended=1_500, status="timeout")])
    service = services(tmp_path, bundle, "llm-only")

    reservation = service.boundary.start(start_tool())

    assert reservation["simulated_execution"] == {"elapsed_ns": 500, "status": "timeout"}


@pytest.mark.parametrize("replay_mode", ["full", "tool-only"])
def test_other_modes_still_demand_native_tool_entry(tmp_path: Path, replay_mode: str) -> None:
    bundle = make_bundle(tools=[tool()])
    service = services(tmp_path, bundle, replay_mode)

    reservation = service.boundary.start(start_tool())

    assert "simulated_execution" not in reservation


def test_simulated_tool_completes_without_native_execution(tmp_path: Path) -> None:
    bundle = make_bundle(tools=[tool(started=1_000, ended=1_000)])
    service = services(tmp_path, bundle, "llm-only")
    reservation = service.boundary.start(start_tool())

    result = mini_swe._simulate_tool(
        LedgerClient(service.boundary),
        reservation,
        reservation["simulated_execution"],
    )

    assert result == {"output": "hi", "exit_code": 0}
    record = json.loads((tmp_path / "tools.jsonl").read_text().splitlines()[0])
    assert record["native_execution"] is False
    assert record["native_result"] is None
    assert record["result"] == {"output": "hi", "exit_code": 0}


def test_simulated_tool_refuses_a_recorded_exception(tmp_path: Path) -> None:
    bundle = make_bundle(
        tools=[
            tool(
                started=1_000,
                ended=1_000,
                status="error",
                exception_raised=True,
                result={"error_type": "RuntimeError", "message": "boom"},
            )
        ]
    )
    service = services(tmp_path, bundle, "llm-only")
    reservation = service.boundary.start(start_tool())

    with pytest.raises(MismatchError, match="cannot restore a recorded tool exception"):
        mini_swe._simulate_tool(
            LedgerClient(service.boundary),
            reservation,
            reservation["simulated_execution"],
        )


def test_llm_only_is_refused_for_an_adapter_that_cannot_simulate(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": CONFIG_SCHEMA,
                "framework": "coral",
                "repo": str(FAKE_REPO),
                "concurrency": 1,
                "duration_s": 5,
                "seed": 42,
                "targets": {"vllm-8000": "http://127.0.0.1:8000"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InfrastructureError, match="mini-swe only"):
        Supervisor(
            mode="replay",
            config=load_config(path),
            output=tmp_path / "replay",
            run_id="replay-llm-only",
            bundle=make_bundle(tools=[tool()]),
            replay_mode="llm-only",
        )
