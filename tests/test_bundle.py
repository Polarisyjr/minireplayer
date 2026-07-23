"""Bundle build/load round-trip and the validation that runs once at load."""

from __future__ import annotations

from pathlib import Path

import pytest

from minireplay.bundle import build_bundle, load_bundle
from minireplay.constants import TERMINAL_SCHEMA
from minireplay.errors import ValidationError
from minireplay.util import append_jsonl, sha256_json
from tests.support import ZERO, artifact, dispatch, llm, span, tool


def stage(tmp_path: Path, **records) -> Path:
    root = tmp_path / "stage"
    root.mkdir()
    for relative in ("llm.jsonl", "spans.jsonl", "dispatches.jsonl", "tools.jsonl",
                     "graders.jsonl", "artifacts.jsonl"):
        (root / relative).touch()
    for relative, entries in records.items():
        for entry in entries:
            append_jsonl(root / relative, entry)
    return root


def build(tmp_path: Path, stage_dir: Path, *, actors=None, name="bundle"):
    return build_bundle(
        stage_dir=stage_dir,
        output=tmp_path / name,
        bundle_id="test",
        adapter="mini-swe",
        workload={"framework": "mini-swe", "concurrency": 1, "duration_s": 60, "seed": 42},
        actors=actors or [{"actor_id": "actor-0", "source_actor_id": "inst-1"}],
        window={"gate_at_ns": 0, "terminal_at_ns": 1000},
        terminal={"schema_version": TERMINAL_SCHEMA, "status": "success", "task_terminals": []},
        cutoff_tails={"operations": [], "llm_requests": []},
    )


def test_round_trip(tmp_path: Path) -> None:
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [dispatch()],
            "tools.jsonl": [tool()],
            "llm.jsonl": [llm()],
            "spans.jsonl": [span("span-dispatch-0"), span("span-tool-0", parent="span-dispatch-0")],
        },
    )
    bundle = build(tmp_path, stage_dir)
    assert bundle.manifest["counts"] == {
        "llm": 1, "dispatch": 1, "tool": 1, "grader": 0, "artifact": 0,
    }
    assert bundle.actor_map() == {"inst-1": "actor-0"}
    assert bundle.manifest["schema_version"] == "minireplay.manifest/v2"
    assert len(bundle.manifest["lanes"]) == 1
    lane_path = bundle.root / bundle.manifest["lanes"][0]["path"]
    assert lane_path.is_file()
    assert not (bundle.root / "llm.jsonl").exists()
    assert not (bundle.root / "tools.jsonl").exists()
    assert not (bundle.root / "cutoff-tails.json").exists()

    reloaded = load_bundle(bundle.root)
    assert reloaded.manifest == bundle.manifest


def test_actor_map_maps_refill_tasks_to_their_worker_lane(tmp_path: Path) -> None:
    stage_dir = stage(tmp_path, **{"llm.jsonl": [llm()]})
    bundle = build(
        tmp_path,
        stage_dir,
        actors=[
            {
                "actor_id": "actor-0",
                "source_actor_id": "inst-1",
                "source_actor_ids": ["inst-1", "inst-9"],
            }
        ],
    )

    assert bundle.actor_map() == {"inst-1": "actor-0", "inst-9": "actor-0"}


def test_load_rejects_a_dangling_tool(tmp_path: Path) -> None:
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [],
            "tools.jsonl": [tool(dispatch_id="dispatch-missing")],
        },
    )
    with pytest.raises(ValidationError, match="is not in the bundle"):
        build(tmp_path, stage_dir)


def test_standalone_composite_primitive_needs_no_outer_dispatch(tmp_path: Path) -> None:
    primitive = tool(
        dispatch_id=None,
        causal_lane="model-call:browse-0",
        name="browser_action",
    )
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [],
            "tools.jsonl": [primitive],
            "spans.jsonl": [span(primitive["span_id"])],
        },
    )

    bundle = build(tmp_path, stage_dir)

    assert bundle.dispatches == []
    assert bundle.tools[0]["dispatch_id"] is None
    assert bundle.tools[0]["causal_lane"] == "model-call:browse-0"


def test_standalone_primitive_requires_a_causal_lane(tmp_path: Path) -> None:
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [],
            "tools.jsonl": [tool(dispatch_id=None)],
        },
    )
    with pytest.raises(ValidationError, match="requires a causal_lane"):
        build(tmp_path, stage_dir)


def test_load_rejects_a_tampered_argument_digest(tmp_path: Path) -> None:
    """The claim path trusts the digest, so the digest is verified here instead."""

    bad = tool()
    bad["arguments"] = {"command": "rm -rf /"}  # digest now describes the old value
    stage_dir = stage(tmp_path, **{"dispatches.jsonl": [dispatch()], "tools.jsonl": [bad]})
    with pytest.raises(ValidationError, match="arguments digest does not match"):
        build(tmp_path, stage_dir)


def test_load_rejects_an_undeclared_actor(tmp_path: Path) -> None:
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [dispatch(actor_id="ghost")],
            "tools.jsonl": [tool(actor_id="ghost")],
        },
    )
    with pytest.raises(ValidationError, match="undeclared actors"):
        build(tmp_path, stage_dir)


def test_load_rejects_two_dispatches_claiming_one_tool(tmp_path: Path) -> None:
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [
                dispatch(dispatch_id="d-0", execution_call_id="tool-0"),
                dispatch(dispatch_id="d-1", execution_call_id="tool-0"),
            ],
            "tools.jsonl": [tool(call_id="tool-0", dispatch_id="d-0")],
        },
    )
    with pytest.raises(ValidationError, match="claimed by dispatches"):
        build(tmp_path, stage_dir)


def test_artifact_read_must_match_its_producer(tmp_path: Path) -> None:
    """A shared artifact is how one actor's work reaches another; that link is a gate."""

    other = "1" * 64
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [],
            "tools.jsonl": [],
            "artifacts.jsonl": [
                artifact(event_id="a-write", operation="create", version=1, digest=ZERO,
                         completed_at_ns=100),
                artifact(event_id="a-read", operation="read", version=1, digest=other,
                         read_from="a-write", completed_at_ns=200),
            ],
        },
    )
    with pytest.raises(ValidationError, match="read bytes differ"):
        build(tmp_path, stage_dir)


def test_artifact_versions_must_be_monotonic(tmp_path: Path) -> None:
    stage_dir = stage(
        tmp_path,
        **{
            "dispatches.jsonl": [],
            "tools.jsonl": [],
            "artifacts.jsonl": [
                artifact(event_id="a-1", version=1, completed_at_ns=100),
                artifact(
                    event_id="a-3", operation="write", version=3, completed_at_ns=200
                ),
            ],
        },
    )
    with pytest.raises(ValidationError, match="not monotonic"):
        build(tmp_path, stage_dir)


def test_llm_request_digest_is_verified(tmp_path: Path) -> None:
    record = llm()
    record["request"] = {"model": "different"}
    stage_dir = stage(
        tmp_path, **{"dispatches.jsonl": [], "tools.jsonl": [], "llm.jsonl": [record]}
    )
    with pytest.raises(ValidationError, match="request digest does not match"):
        build(tmp_path, stage_dir)


def test_cutoff_tail_must_declare_its_duration(tmp_path: Path) -> None:
    stage_dir = stage(tmp_path, **{"dispatches.jsonl": [], "tools.jsonl": []})
    with pytest.raises(ValidationError, match="invalid elapsed_ns"):
        build_bundle(
            stage_dir=stage_dir,
            output=tmp_path / "b",
            bundle_id="test",
            adapter="mini-swe",
            workload={"framework": "mini-swe", "concurrency": 1, "duration_s": 60, "seed": 42},
            actors=[{"actor_id": "actor-0"}],
            window={"gate_at_ns": 0, "terminal_at_ns": 1000},
            terminal={"schema_version": TERMINAL_SCHEMA, "status": "success", "task_terminals": []},
            cutoff_tails={
                "operations": [
                    {"cutoff_truncated": True, "kind": "tool", "record_id": "t",
                     "actor_id": "actor-0"}
                ],
                "llm_requests": [],
            },
        )


def test_cutoff_tail_cannot_also_be_a_closed_llm_attempt(tmp_path: Path) -> None:
    record = llm()
    stage_dir = stage(
        tmp_path,
        **{"dispatches.jsonl": [], "tools.jsonl": [], "llm.jsonl": [record]},
    )
    tail = {
        "cutoff_truncated": True,
        "attempt_id": record["attempt_id"],
        "actor_id": record["actor_id"],
        "elapsed_ns": 10,
    }
    with pytest.raises(ValidationError, match="both closed and cutoff-truncated"):
        build_bundle(
            stage_dir=stage_dir,
            output=tmp_path / "b",
            bundle_id="test",
            adapter="mini-swe",
            workload={
                "framework": "mini-swe",
                "concurrency": 1,
                "duration_s": 60,
                "seed": 42,
            },
            actors=[{"actor_id": "actor-0"}],
            window={"gate_at_ns": 0, "terminal_at_ns": 1000},
            terminal={
                "schema_version": TERMINAL_SCHEMA,
                "status": "success",
                "task_terminals": [],
            },
            cutoff_tails={"operations": [], "llm_requests": [tail]},
        )


def test_digest_helper_is_the_one_the_claim_uses() -> None:
    """Guard the coupling: bundle-time digests and claim-time digests must agree."""

    from minireplay.boundary import claim_identity

    arguments = {"command": "echo hi", "cwd": "/w"}
    record = tool(arguments=arguments)
    assert claim_identity("tool", record) == ("shell", sha256_json(arguments))
