"""Record then replay a stand-in framework, end to end.

This drives the real supervisor, services, gate, ledger, cutoff pruning, bundle
build and replay path. Only the framework is a stand-in, so it runs without a GPU,
a model or Docker while still covering everything the harness owns.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
from aiohttp import web

from minireplay.bundle import load_bundle
from minireplay.config import load_config
from minireplay.constants import CONFIG_SCHEMA
from minireplay.errors import InfrastructureError, MismatchError
from minireplay.supervisor import record_bundle, replay_bundle
from minireplay.util import read_json

FAKE_REPO = Path(__file__).parent / "fake_repo"


class FakeVLLM:
    """The smallest OpenAI-compatible responder a recording needs."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.url = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None

    async def _chat(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append(body)
        index = len(self.requests)
        return web.json_response(
            {
                "id": f"chatcmpl-{index}",
                "object": "chat.completion",
                "created": 1,
                "model": "fake",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"reply {index}"},
                        "finish_reason": "stop",
                        "token_ids": [100 + index, 200 + index],
                    }
                ],
                "prompt_token_ids": [1, 2, 3],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )

    def start(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        ready = threading.Event()

        async def boot() -> None:
            app = web.Application()
            app.router.add_post("/v1/chat/completions", self._chat)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            self._runner = runner
            port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
            self.url = f"http://127.0.0.1:{port}"

        def run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(boot())
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(timeout=10)

    def stop(self) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


@pytest.fixture
def vllm():
    server = FakeVLLM()
    server.start()
    yield server
    server.stop()


def write_config(tmp_path: Path, vllm_url: str, *, concurrency: int = 1, duration: int = 5) -> Path:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": CONFIG_SCHEMA,
                "framework": "mini-swe",
                "repo": str(FAKE_REPO),
                "concurrency": concurrency,
                "duration_s": duration,
                "seed": 42,
                "targets": {"vllm-8000": vllm_url},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.timeout(300)
def test_record_then_replay(tmp_path: Path, vllm) -> None:
    config_path = write_config(tmp_path, vllm.url)
    config = load_config(config_path)

    bundle = record_bundle(
        config=config,
        output=tmp_path / "source",
        bundle_output=tmp_path / "bundle",
        run_id="record-test",
    )

    counts = bundle.manifest["counts"]
    assert counts["tool"] == 2, f"expected two tool calls, got {counts}"
    assert counts["dispatch"] == 2
    assert counts["llm"] == 2
    assert len(bundle.manifest["actors"]) == 1

    # Every tool really ran: the stand-in writes a file per call.
    assert (tmp_path / "source" / "tool-0.txt").is_file()
    assert (tmp_path / "source" / "tool-1.txt").is_file()

    # The bundle reloads and revalidates from disk.
    assert load_bundle(tmp_path / "bundle").manifest == bundle.manifest

    source_metrics = read_json(tmp_path / "source" / "metrics.json")
    assert source_metrics["makespan_seconds"] > 0
    assert (tmp_path / "source" / "step3/raw/llm_spans.jsonl").is_file()
    assert (tmp_path / "source" / "step3/raw/tool_events.jsonl").is_file()
    assert (tmp_path / "source" / "step3/views/timeline.txt").is_file()
    assert (tmp_path / "source" / "step3/views/timeline.png").stat().st_size > 1024

    run = replay_bundle(
        config=config,
        bundle_dir=tmp_path / "bundle",
        output=tmp_path / "replay-0",
        replay_mode="tool-only",
        run_id="replay-test",
    )

    verdict = read_json(tmp_path / "replay-0" / "verdict.json")
    assert verdict["valid"] is True, verdict
    assert verdict["reason"] == "fixed-work-complete"

    # The tools executed for real again, in the replay's own directory.
    assert (tmp_path / "replay-0" / "tool-0.txt").is_file()
    assert (tmp_path / "replay-0" / "tool-1.txt").is_file()

    # Tool-only replay must not have contacted vLLM a second time.
    assert len(vllm.requests) == 2, "tool-only replay reached the upstream"

    replayed = list((tmp_path / "replay-0" / "stage" / "tools.jsonl").read_text().splitlines())
    assert len([line for line in replayed if line]) == 2
    assert run.metrics["makespan_seconds"] >= 0


@pytest.mark.timeout(300)
def test_replay_rejects_a_workload_change(tmp_path: Path, vllm) -> None:
    """A bundle only replays the workload it recorded."""

    config = load_config(write_config(tmp_path, vllm.url))
    record_bundle(
        config=config,
        output=tmp_path / "source",
        bundle_output=tmp_path / "bundle",
        run_id="record-test",
    )
    changed = load_config(write_config(tmp_path, vllm.url, concurrency=2))
    with pytest.raises(MismatchError, match="does not match the bundle's workload"):
        replay_bundle(
            config=changed,
            bundle_dir=tmp_path / "bundle",
            output=tmp_path / "replay-bad",
            run_id="replay-bad",
        )


@pytest.mark.timeout(300)
def test_replay_stops_at_the_recorded_amount_of_work(tmp_path: Path, vllm, monkeypatch) -> None:
    """A replay does the recorded work and no more, even if the framework wants to.

    A recording is cut at the sweep boundary, normally mid-episode, so a framework
    replaying it will still want to continue. That request is held rather than
    answered: the run is over, and the framework must not be able to observe an
    invented result or refill work the recording does not contain. This is what
    "fixed workload" means operationally.
    """

    config = load_config(write_config(tmp_path, vllm.url))
    monkeypatch.setenv("FAKE_AGENT_TOOL_CALLS", "2")
    bundle = record_bundle(
        config=config,
        output=tmp_path / "source",
        bundle_output=tmp_path / "bundle",
        run_id="record-test",
    )
    assert bundle.manifest["counts"]["tool"] == 2

    # The same framework, now willing to do half again as much work.
    monkeypatch.setenv("FAKE_AGENT_TOOL_CALLS", "3")
    replay_bundle(
        config=load_config(write_config(tmp_path, vllm.url)),
        bundle_dir=tmp_path / "bundle",
        output=tmp_path / "replay-more",
        run_id="replay-more",
    )

    verdict = read_json(tmp_path / "replay-more" / "verdict.json")
    assert verdict["valid"] is True, verdict

    # Exactly the recorded work ran. The third tool never executed.
    tools = [
        line
        for line in (tmp_path / "replay-more" / "stage" / "tools.jsonl").read_text().splitlines()
        if line
    ]
    assert len(tools) == 2
    assert (tmp_path / "replay-more" / "tool-1.txt").is_file()
    assert not (tmp_path / "replay-more" / "tool-2.txt").exists()


@pytest.mark.timeout(300)
def test_replay_rejects_drift_inside_the_window(tmp_path: Path, vllm, monkeypatch) -> None:
    """Doing something *different* within the recorded window is still a hard failure."""

    config = load_config(write_config(tmp_path, vllm.url))
    monkeypatch.setenv("FAKE_AGENT_TOOL_CALLS", "2")
    record_bundle(
        config=config,
        output=tmp_path / "source",
        bundle_output=tmp_path / "bundle",
        run_id="record-test",
    )

    monkeypatch.setenv("FAKE_AGENT_ARGUMENT_SALT", "changed")
    with pytest.raises(InfrastructureError) as error:
        replay_bundle(
            config=load_config(write_config(tmp_path, vllm.url)),
            bundle_dir=tmp_path / "bundle",
            output=tmp_path / "replay-drift",
            run_id="replay-drift",
        )
    assert "invocation drift" in str(error.value)
