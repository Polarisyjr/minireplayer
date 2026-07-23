from __future__ import annotations

from pathlib import Path

from minireplay.supervisor import _actors_from_stage
from minireplay.util import atomic_write_json


def test_actor_inventory_includes_refill_actor_seen_only_at_cutoff(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    ready = tmp_path / "actor-ready"
    bindings = tmp_path / "lane-bindings"
    stage.mkdir()
    ready.mkdir()
    bindings.mkdir()
    atomic_write_json(
        ready / "initial.json",
        {
            "actor_id": "initial",
            "source_actor_id": "initial",
            "process_role": "agent",
        },
    )
    atomic_write_json(
        bindings / "binding.json",
        {
            "actor_id": "initial",
            "source_actor_id": "refill",
            "native_lane_key": "worker-0",
        },
    )

    actors = _actors_from_stage(
        stage,
        ready,
        {
            "llm_requests": [{"actor_id": "initial", "role": "agent"}],
            "operations": [],
        },
        bindings,
    )

    assert actors == [
        {
            "actor_id": "initial",
            "source_actor_id": "initial",
            "source_actor_ids": ["initial", "refill"],
            "process_role": "agent",
            "task": {"source_actor_id": "initial"},
        },
    ]
