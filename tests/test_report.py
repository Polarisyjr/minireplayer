"""Repeated replay report validity and timeline diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

import minireplay.report as report_module
from minireplay.report import build_report
from minireplay.util import atomic_write_json
from tests.support import make_bundle


def _write_run(root: Path, *, gaps: list[dict[str, object]]) -> None:
    root.mkdir()
    atomic_write_json(
        root / "metrics.json",
        {
            "run_id": root.name,
            "makespan_seconds": 10.0,
            "busy_span_seconds": 4.0,
            "operations": {
                "llm": {"count": 0},
                "dispatch": {"count": 1},
                "tool": {"count": 1},
                "grader": {"count": 0},
                "artifact": {"count": 0},
            },
        },
    )
    atomic_write_json(
        root / "verdict.json",
        {"valid": True, "reason": "fixed-work-complete"},
    )
    atomic_write_json(
        root / "timeline.json",
        {
            "window_s": 10.0,
            "coverage_fraction": 0.4,
            "unattributed_gap_seconds": sum(
                float(gap["duration_s"]) for gap in gaps
            ),
            "gaps": gaps,
        },
    )


def test_startup_and_tail_gaps_remain_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = make_bundle(root=tmp_path / "bundle")
    monkeypatch.setattr(report_module, "load_bundle", lambda _path: bundle)
    run = tmp_path / "replay"
    _write_run(
        run,
        gaps=[
            {
                "start_s": 0.0,
                "end_s": 6.0,
                "duration_s": 6.0,
                "attribution": "startup: framework had not reached its first call",
            },
            {
                "start_s": 9.0,
                "end_s": 10.0,
                "duration_s": 1.0,
                "attribution": "tail: no further activity",
            },
        ],
    )

    report = build_report(bundle_dir=bundle.root, run_dirs=[run])

    assert report["valid"] is True
    assert report["runs"][0]["timeline"]["unattributed_gap_seconds"] == 7.0
    assert report["runs"][0]["timeline"]["internal_gap_seconds"] == 0.0


def test_large_internal_gap_invalidates_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = make_bundle(root=tmp_path / "bundle")
    monkeypatch.setattr(report_module, "load_bundle", lambda _path: bundle)
    run = tmp_path / "replay"
    _write_run(
        run,
        gaps=[
            {
                "start_s": 2.0,
                "end_s": 8.0,
                "duration_s": 6.0,
                "attribution": "between llm and tool (uninstrumented framework work)",
            }
        ],
    )

    report = build_report(bundle_dir=bundle.root, run_dirs=[run])

    assert report["valid"] is False
    assert "uninstrumented gaps between recorded operations" in report["reasons"][0]
    assert report["runs"][0]["timeline"]["internal_gap_seconds"] == 6.0
