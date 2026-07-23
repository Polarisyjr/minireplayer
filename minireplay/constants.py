from __future__ import annotations

CONFIG_SCHEMA = "minireplay.config/v1"
MANIFEST_SCHEMA = "minireplay.manifest/v2"
LANE_BUNDLE_EVENT_SCHEMA = "minireplay.lane-bundle-event/v1"
LLM_SCHEMA = "minireplay.llm-attempt/v1"
DISPATCH_SCHEMA = "minireplay.dispatch/v1"
TOOL_SCHEMA = "minireplay.tool-call/v1"
SPAN_SCHEMA = "minireplay.span/v1"
GRADER_SCHEMA = "minireplay.grader/v1"
ARTIFACT_SCHEMA = "minireplay.artifact/v1"
TERMINAL_SCHEMA = "minireplay.terminal/v1"
METRICS_SCHEMA = "minireplay.metrics/v1"
TIMELINE_SCHEMA = "minireplay.timeline/v1"
REPORT_SCHEMA = "minireplay.report/v1"
COMPARISON_SCHEMA = "minireplay.replay-comparison/v1"
RESULT_CONTRACT_SCHEMA = "native-agent-replay.result-contract/v2"
LANE_RECORD_EVENT_SCHEMA = "minireplay.lane-record-event/v1"

SUPPORTED_ADAPTERS = frozenset({"mini-swe", "trae", "coral", "owl"})

# aiohttp caps request bodies at 1 MiB by default, which is a property of this
# harness and not of the workload: owl's multimodal endpoint carries base64 frames
# and its histories grow with concurrency, so at C8 a legitimate request was
# rejected with "Content Too Large" and failed the run. The proxy has to accept
# whatever the framework would have sent to vLLM; the bound stays only so a runaway
# body cannot exhaust memory silently.
MAX_REQUEST_BYTES = 512 * 1024 * 1024

# Adapters call state().cover(...) with these; the vocabulary is kept so the
# ported adapters run unchanged, and the recorded entries stay as evidence.
INSTRUMENTATION_COVERAGE_KINDS = frozenset(
    {
        "dispatcher",
        "implementation",
        "parser",
        "plugin",
        "registry",
    }
)

# One JSONL per causal record kind. The ledger writes them during a run and the
# bundle stores them verbatim, so record and replay share one reader.
LEDGER_FILES = {
    "dispatch": "dispatches.jsonl",
    "tool": "tools.jsonl",
    "grader": "graders.jsonl",
    "artifact": "artifacts.jsonl",
}
LEDGER_ID_FIELD = {
    "dispatch": "dispatch_id",
    "tool": "call_id",
    "grader": "attempt_id",
    "artifact": "event_id",
}
LEDGER_SCHEMA = {
    "dispatch": DISPATCH_SCHEMA,
    "tool": TOOL_SCHEMA,
    "grader": GRADER_SCHEMA,
    "artifact": ARTIFACT_SCHEMA,
}

BUNDLE_FILES = (
    "manifest.json",
    "terminal.json",
)
