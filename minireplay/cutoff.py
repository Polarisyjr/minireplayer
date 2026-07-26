"""Close a recording at the sweep's sample boundary.

Dropping every record that ended after the cutoff is not enough. A tool can finish
inside the window while the dispatch that owns it finishes outside; keeping the tool
alone would leave the bundle referencing a record it does not contain.

So this prunes to a fixed point: a record survives only if everything it depends on
also survives. Composite cutoff operations are the exception. A built-in ``task``
may still be running when its already-completed child LLM and tool operations close.
Those descendants remain fixed work, with the open parent represented by its cutoff
tail rather than a closed ledger record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import atomic_write, atomic_write_json, canonical_json, iter_jsonl

_LEDGERS = {
    "dispatch": "dispatches.jsonl",
    "tool": "tools.jsonl",
    "grader": "graders.jsonl",
    "artifact": "artifacts.jsonl",
    "llm": "llm.jsonl",
}
_ID = {
    "dispatch": "dispatch_id",
    "tool": "call_id",
    "grader": "attempt_id",
    "artifact": "event_id",
    "llm": "attempt_id",
}


def _load(stage: Path, relative: str) -> list[dict[str, Any]]:
    path = stage / relative
    return list(iter_jsonl(path)) if path.is_file() else []


def _rewrite(stage: Path, relative: str, records: list[dict[str, Any]]) -> None:
    payload = b"".join(canonical_json(record) + b"\n" for record in records)
    atomic_write(stage / relative, payload)


def close_stage_at_cutoff(
    stage_dir: Path,
    *,
    cutoff_tails: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_dir = stage_dir.resolve()
    kept = {kind: _load(stage_dir, relative) for kind, relative in _LEDGERS.items()}
    spans = _load(stage_dir, "spans.jsonl")
    open_parent_span_ids = {
        str(record["span_id"])
        for record in (cutoff_tails or {}).get("operations", [])
        if record.get("replay_entry") == "enter-and-preserve-descendants"
        and isinstance(record.get("span_id"), str)
        and record["span_id"]
    }
    before = {kind: len(records) for kind, records in kept.items()}
    before["span"] = len(spans)

    while True:
        ids = {
            kind: {str(record[_ID[kind]]) for record in records} for kind, records in kept.items()
        }
        changed = False

        surviving = [
            tool
            for tool in kept["tool"]
            if tool.get("dispatch_id") is None or str(tool.get("dispatch_id")) in ids["dispatch"]
        ]
        if len(surviving) != len(kept["tool"]):
            kept["tool"] = surviving
            changed = True

        surviving = [
            dispatch
            for dispatch in kept["dispatch"]
            if dispatch.get("execution_call_id") is None
            or str(dispatch["execution_call_id"]) in ids["tool"]
        ]
        if len(surviving) != len(kept["dispatch"]):
            kept["dispatch"] = surviving
            changed = True

        surviving = [
            artifact
            for artifact in kept["artifact"]
            if artifact.get("read_from") is None or str(artifact["read_from"]) in ids["artifact"]
        ]
        if len(surviving) != len(kept["artifact"]):
            kept["artifact"] = surviving
            changed = True

        surviving = [
            grader
            for grader in kept["grader"]
            if all(str(value) in ids["llm"] for value in grader.get("llm_attempt_ids", []))
            and all(str(value) in ids["tool"] for value in grader.get("tool_call_ids", []))
            and all(str(value) in ids["artifact"] for value in grader.get("artifact_versions", []))
        ]
        if len(surviving) != len(kept["grader"]):
            kept["grader"] = surviving
            changed = True

        # Parent links live on spans because they cross LLM, tool and dispatch
        # kinds. Ordinary missing parents invalidate the descendant branch. A
        # composite cutoff parent remains a valid owner even though it has no
        # closed ledger record: replay enters that task and replays its closed
        # descendants before withholding the unfinished parent result.
        span_ids = {str(span["span_id"]) for span in spans} | open_parent_span_ids
        orphan_spans = {
            str(span["span_id"])
            for span in spans
            if span.get("parent_span_id") is not None
            and str(span["parent_span_id"]) not in span_ids
        }
        if orphan_spans:
            for kind, records in kept.items():
                surviving = [
                    record for record in records if str(record.get("span_id")) not in orphan_spans
                ]
                if len(surviving) != len(records):
                    kept[kind] = surviving
                    changed = True

        live = {
            str(record["span_id"])
            for records in kept.values()
            for record in records
            if record.get("span_id")
        }
        surviving_spans = [span for span in spans if str(span["span_id"]) in live]
        if len(surviving_spans) != len(spans):
            spans = surviving_spans
            changed = True

        if not changed:
            break

    for kind, relative in _LEDGERS.items():
        _rewrite(stage_dir, relative, kept[kind])
    _rewrite(stage_dir, "spans.jsonl", spans)

    after = {kind: len(records) for kind, records in kept.items()}
    after["span"] = len(spans)
    report = {
        "schema_version": "minireplay.cutoff-report/v1",
        "before": before,
        "after": after,
        "discarded": {kind: before[kind] - after[kind] for kind in before},
        "open_parent_span_ids": sorted(open_parent_span_ids),
    }
    atomic_write_json(stage_dir / "cutoff-report.json", report)
    return report
