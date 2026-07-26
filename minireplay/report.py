"""Compare repeated replays of one bundle.

Section 7 asks for repeated replays and for the unstable ones to be explained, but
explicitly not for an automated attribution rule. So this reports spread and points
at the operations that carry it; deciding whether a given spread is a compilation
step, a network fetch or a real regression is left to whoever reads it.

A run is only checked for having done the recorded work. Being slower is never by
itself a reason to reject or re-run it.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .bundle import load_bundle
from .constants import REPORT_SCHEMA
from .util import read_json, require

# Reported, never enforced. It marks a metric worth a human's attention.
ATTENTION_SPREAD = 0.10


def _load_run(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    require(root.is_dir(), f"run directory does not exist: {root}")
    metrics = read_json(root / "metrics.json")
    verdict = read_json(root / "verdict.json")
    timeline = read_json(root / "timeline.json")
    return {
        "root": str(root),
        "run_id": metrics["run_id"],
        "valid": bool(verdict.get("valid")),
        "reason": verdict.get("reason"),
        "metrics": metrics,
        "timeline": timeline,
    }


def _spread(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    low, high = min(values), max(values)
    return {
        "n": len(values),
        "mean": round(mean, 3),
        "min": round(low, 3),
        "max": round(high, 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "relative_spread": round((high - low) / mean, 4) if mean > 0 else 0.0,
    }


def _series(runs: list[dict[str, Any]], *path: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        cursor: Any = run["metrics"]
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, (int, float)) and not isinstance(cursor, bool):
            values.append(float(cursor))
    return values


def _operation_spreads(runs: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = {
        kind
        for run in runs
        for kind in run["metrics"].get("operations", {})
    }
    result: dict[str, Any] = {}
    for kind in sorted(kinds):
        for field in ("count", "total_seconds", "cpu_seconds", "max_seconds"):
            values = _series(runs, "operations", kind, field)
            if values:
                result[f"{kind}.{field}"] = _spread(values)
    return result


def _internal_gap_seconds(timeline: dict[str, Any]) -> float:
    """Count only blanks bounded by instrumented work on both sides.

    Startup and tail gaps remain visible in the report, but they describe the
    framework outside its recorded causal work. Treating either as missing
    instrumentation makes short runs fail based on process startup or idle
    teardown time rather than on replay completeness.
    """

    window = timeline.get("window_s", 0.0)
    if not isinstance(window, (int, float)) or isinstance(window, bool):
        return 0.0
    total = 0.0
    for gap in timeline.get("gaps", []):
        start = gap.get("start_s")
        end = gap.get("end_s")
        duration = gap.get("duration_s")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (start, end, duration)
        ):
            continue
        if start > 0.001 and end < window - 0.001:
            total += duration
    return total


def build_report(
    *,
    bundle_dir: Path,
    run_dirs: list[Path],
    source_dir: Path | None = None,
) -> dict[str, Any]:
    bundle = load_bundle(bundle_dir)
    runs = [_load_run(path) for path in run_dirs]
    require(bool(runs), "report needs at least one replay run")

    reasons: list[str] = []
    for run in runs:
        if not run["valid"]:
            reasons.append(f"{run['run_id']}: run was invalid ({run['reason']})")
        operations = run["metrics"].get("operations", {})
        for kind, expected in bundle.manifest["counts"].items():
            observed = operations.get(kind, {}).get("count", 0)
            if observed != expected:
                reasons.append(
                    f"{run['run_id']}: replayed {observed} {kind} records, "
                    f"the bundle holds {expected}"
                )
        window = run["timeline"].get("window_s", 0.0)
        internal_gap = _internal_gap_seconds(run["timeline"])
        if window > 0 and internal_gap / window > 0.5:
            reasons.append(
                f"{run['run_id']}: {internal_gap:.1f}s of a {window:.1f}s window has "
                "uninstrumented gaps between recorded operations"
            )

    spreads = {
        # First to last instrumented activity. This is the number to compare: a
        # recording's makespan is its sweep window, which outlasts the work whenever
        # an agent finishes early.
        "busy_span_seconds": _spread(_series(runs, "busy_span_seconds")),
        "makespan_seconds": _spread(_series(runs, "makespan_seconds")),
        "framework_cpu_seconds": _spread(_series(runs, "framework_cgroup", "cpu_seconds")),
        "gpu_active_seconds": _spread(_series(runs, "gpu", "gpu_active_seconds")),
        "disk_read_bytes": _spread(_series(runs, "host_deltas", "disk_read_bytes")),
        "disk_write_bytes": _spread(_series(runs, "host_deltas", "disk_write_bytes")),
        "net_sent_bytes": _spread(_series(runs, "host_deltas", "net_sent_bytes")),
        "net_recv_bytes": _spread(_series(runs, "host_deltas", "net_recv_bytes")),
    }
    needs_attention = sorted(
        name
        for name, value in {**spreads, **_operation_spreads(runs)}.items()
        if value.get("n", 0) > 1 and value.get("relative_spread", 0.0) > ATTENTION_SPREAD
    )

    report = {
        "schema_version": REPORT_SCHEMA,
        "valid": not reasons,
        "reasons": reasons,
        "bundle": {
            "bundle_id": bundle.manifest["bundle_id"],
            "adapter": bundle.adapter,
            "workload": bundle.manifest["workload"],
            "counts": bundle.manifest["counts"],
            "actors": len(bundle.manifest["actors"]),
            "cutoff_policy": bundle.manifest.get("cutoff_policy", "evidence-only"),
            "cutoff_tails": {
                section: len(entries)
                for section, entries in bundle.cutoff_tails.items()
            },
        },
        "runs": [
            {
                "run_id": run["run_id"],
                "root": run["root"],
                "valid": run["valid"],
                "makespan_seconds": run["metrics"]["makespan_seconds"],
                "busy_span_seconds": run["metrics"].get("busy_span_seconds"),
                "replay_mode": run["metrics"].get("replay_mode"),
                "timeline": {
                    "coverage_fraction": run["timeline"].get("coverage_fraction"),
                    "unattributed_gap_seconds": run["timeline"].get(
                        "unattributed_gap_seconds"
                    ),
                    "internal_gap_seconds": round(
                        _internal_gap_seconds(run["timeline"]), 3
                    ),
                    "largest_gaps": sorted(
                        run["timeline"].get("gaps", []),
                        key=lambda gap: gap["duration_s"],
                        reverse=True,
                    )[:5],
                },
            }
            for run in runs
        ],
        "spreads": spreads,
        "operation_spreads": _operation_spreads(runs),
        "needs_attention": needs_attention,
        "note": (
            "relative_spread above "
            f"{ATTENTION_SPREAD:.0%} is flagged for inspection, not failed. "
            "Compilation, package fetches and network variance move these legitimately."
        ),
    }
    if source_dir is not None:
        source = _load_run(source_dir)
        source_timeline = source["timeline"]
        report["source"] = {
            "run_id": source["run_id"],
            "makespan_seconds": source["metrics"]["makespan_seconds"],
            "busy_span_seconds": source["metrics"].get("busy_span_seconds")
            or source_timeline.get("busy_span_s"),
            "idle_tail_seconds": source_timeline.get("unattributed_gap_seconds"),
            "note": (
                "recorder performance is not a baseline; shown for context only. "
                "Compare busy_span_seconds, not makespan_seconds: a recording's "
                "makespan is the sweep window, which keeps running after an agent "
                "finishes."
            ),
        }
    return report
