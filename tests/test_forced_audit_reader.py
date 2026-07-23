from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from minireplay.forced import ForcedAuditReader, token_digest


def capture_record(request_id: str, *, prompt: list[int] | None = None) -> dict:
    prompt = prompt or [1, 2, 3]
    sampled = [10, 11]
    return {
        "schema_version": "native-agent-replay.vllm-audit/v4",
        "request_id": request_id,
        "mode": "capture",
        "prompt_token_ids": prompt,
        "prompt_token_count": len(prompt),
        "prompt_token_sha256": token_digest(prompt),
        "sampled_token_ids": sampled,
        "sampled_token_count": len(sampled),
        "sampled_token_sha256": token_digest(sampled),
        "forced_token_ids": [],
        "forced_token_count": 0,
        "forced_token_sha256": token_digest([]),
        "forced_count": 0,
        "committed_sample_start": None,
        "expected_sampled_token_count": None,
        "status": "capture_complete",
        "pid": 123,
    }


def encoded(record: dict) -> bytes:
    return json.dumps(record, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def append(path: Path, payload: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(payload)


async def test_run_local_reader_ignores_existing_fleet_history(tmp_path: Path) -> None:
    audit = tmp_path / "forced-audit.jsonl"
    audit.write_bytes(b"old fleet history does not even need to be parsed\n")
    reader = ForcedAuditReader(audit, start_at_end=True)

    append(audit, encoded(capture_record("current-run")))

    record = await reader.wait_capture("current-run", timeout_s=1)
    assert record["request_id"] == "current-run"
    assert set(reader._seen) == {"current-run"}


async def test_incremental_reader_retains_a_partial_final_line(tmp_path: Path) -> None:
    audit = tmp_path / "forced-audit.jsonl"
    audit.touch()
    reader = ForcedAuditReader(audit, start_at_end=True)
    payload = encoded(capture_record("split-record"))

    append(audit, payload[: len(payload) // 2])
    waiting = asyncio.create_task(reader.wait_capture("split-record", timeout_s=1))
    await asyncio.sleep(0.05)
    assert not waiting.done()

    append(audit, payload[len(payload) // 2 :])
    assert (await waiting)["request_id"] == "split-record"


async def test_one_pump_fans_out_one_incremental_read_to_many_waiters(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "forced-audit.jsonl"
    audit.touch()
    reader = ForcedAuditReader(audit, start_at_end=True)
    request_ids = [f"request-{index}" for index in range(32)]
    append(audit, b"".join(encoded(capture_record(request_id)) for request_id in request_ids))

    original = reader._read_new_records
    calls = 0

    def counted_read() -> list[dict]:
        nonlocal calls
        calls += 1
        return original()

    reader._read_new_records = counted_read
    records = await asyncio.gather(
        *(reader.wait_capture(request_id, timeout_s=1) for request_id in request_ids)
    )

    assert calls == 1
    assert {record["request_id"] for record in records} == set(request_ids)


async def test_audit_read_does_not_block_the_event_loop(tmp_path: Path) -> None:
    audit = tmp_path / "forced-audit.jsonl"
    audit.touch()
    reader = ForcedAuditReader(audit, start_at_end=True)
    append(audit, encoded(capture_record("off-loop")))

    original = reader._read_new_records
    started = threading.Event()
    release = threading.Event()

    def blocked_read() -> list[dict]:
        started.set()
        assert release.wait(timeout=1)
        return original()

    reader._read_new_records = blocked_read
    waiting = asyncio.create_task(reader.wait_capture("off-loop", timeout_s=1))
    assert await asyncio.to_thread(started.wait, 1)

    # This line can only run while blocked_read is waiting if the filesystem work
    # was moved off the aiohttp event loop.
    release.set()
    assert (await waiting)["request_id"] == "off-loop"
