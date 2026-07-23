"""Unified record/replay comparison plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import minireplay.cli as cli
import minireplay.comparison_plot as comparison_plot
from minireplay.constants import COMPARISON_SCHEMA
from minireplay.errors import ValidationError
from minireplay.util import atomic_write_json, read_json
from tests.support import llm, make_bundle, tool


def _write_run(
    root: Path,
    *,
    run_id: str,
    makespan: float,
    replay_mode: str,
    shift: float = 0.0,
    tool_name: str = "shell",
) -> None:
    root.mkdir()
    atomic_write_json(
        root / "metrics.json",
        {
            "run_id": run_id,
            "gate_at_ns": 0,
            "makespan_seconds": makespan,
            "busy_span_seconds": makespan - 0.1,
            "replay_mode": replay_mode,
            "operations": {
                "llm": {"count": 1},
                "dispatch": {"count": 0},
                "tool": {"count": 1},
                "grader": {"count": 0},
                "artifact": {"count": 0},
            },
        },
    )
    atomic_write_json(
        root / "timeline.json",
        {
            "spans": [
                {
                    "actor_id": "actor-0",
                    "lane": "llm",
                    "name": "llm",
                    "start_s": 0.1 + shift,
                    "end_s": 0.4 + shift,
                },
                {
                    "actor_id": "actor-0",
                    "lane": "tool",
                    "name": tool_name,
                    "start_s": 0.5 + shift,
                    "end_s": 0.8 + shift,
                },
            ]
        },
    )
    atomic_write_json(
        root / "verdict.json",
        {"valid": True, "reason": "fixed-work-complete"},
    )


def _bundle(tmp_path: Path):
    return make_bundle(
        root=tmp_path / "bundle",
        adapter="owl",
        actors=["actor-0", "actor-empty"],
        dispatches=[],
        tools=[
            tool(
                dispatch_id=None,
                causal_lane="model-call:0",
            )
        ],
        llm_records=[llm()],
        cutoff_tails={
            "operations": [],
            "llm_requests": [
                {
                    "actor_id": "actor-empty",
                    "started_at_ns": 900_000_000,
                }
            ],
        },
    )


def test_renders_any_number_of_replays_and_keeps_empty_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(comparison_plot, "load_bundle", lambda _path: bundle)
    source = tmp_path / "source"
    replays = [tmp_path / f"replay-{index}" for index in range(3)]
    _write_run(
        source,
        run_id="source",
        makespan=1.0,
        replay_mode="n/a",
    )
    for index, replay in enumerate(replays):
        _write_run(
            replay,
            run_id=f"replay-{index}",
            makespan=1.1 + index / 10,
            replay_mode="full",
            shift=index / 100,
        )

    result = comparison_plot.render_comparison(
        bundle_dir=bundle.root,
        source_dir=source,
        run_dirs=replays,
        output_dir=tmp_path / "plots",
        formats=("svg",),
    )

    assert Path(result["wallclock"]["svg"]).is_file()
    assert Path(result["timeline"]["svg"]).is_file()
    assert Path(result["csv"]).is_file()
    summary = read_json(Path(result["summary"]))
    assert summary["schema_version"] == COMPARISON_SCHEMA
    assert list(summary["batch"]) == ["record", "replay1", "replay2", "replay3"]
    assert summary["replay_spread"]["makespan_s"]["n"] == 3
    empty = next(row for row in summary["lanes"] if row["actor_id"] == "actor-empty")
    assert empty["record"]["wallclock_s"] == 0.0
    assert empty["replay3"]["llm_count"] == 0


def test_rejects_browse_url_as_a_filled_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(comparison_plot, "load_bundle", lambda _path: bundle)
    source = tmp_path / "source"
    replay = tmp_path / "replay"
    _write_run(
        source,
        run_id="source",
        makespan=1.0,
        replay_mode="n/a",
        tool_name="browse_url",
    )
    _write_run(
        replay,
        run_id="replay",
        makespan=1.1,
        replay_mode="full",
    )

    with pytest.raises(ValidationError, match="composite scope"):
        comparison_plot.render_comparison(
            bundle_dir=bundle.root,
            source_dir=source,
            run_dirs=[replay],
            output_dir=tmp_path / "plots",
        )


def test_cli_forwards_repeated_runs_and_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, Any] = {}

    def fake_render(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"summary": "done"}

    monkeypatch.setattr(comparison_plot, "render_comparison", fake_render)
    result = cli.main(
        [
            "plot-comparison",
            "--bundle",
            str(tmp_path / "bundle"),
            "--source",
            str(tmp_path / "source"),
            "--run",
            str(tmp_path / "r1"),
            "--run",
            str(tmp_path / "r2"),
            "--out",
            str(tmp_path / "plots"),
            "--format",
            "svg",
            "--format",
            "png",
        ]
    )

    assert result == 0
    assert observed["run_dirs"] == [tmp_path / "r1", tmp_path / "r2"]
    assert observed["formats"] == ["svg", "png"]
    assert '"summary": "done"' in capsys.readouterr().out
