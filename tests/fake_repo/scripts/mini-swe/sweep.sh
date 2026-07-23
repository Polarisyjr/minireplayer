#!/bin/bash
# A stand-in for the real sweep: same argv contract, same phase events, same
# "launch C actors then close the window" shape. It exists so the supervisor,
# services, gate, ledger and bundle build can be exercised without a GPU,
# a model or Docker.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"

CONCURRENCY=1
DURATION=5
STEP=both
while [ $# -gt 0 ]; do
    case "$1" in
        -c|--concurrency) CONCURRENCY="$2"; shift 2 ;;
        -n|--num-tasks)   shift 2 ;;
        --seed)           shift 2 ;;
        -s|--step)        STEP="$2"; shift 2 ;;
        -d|--duration)    DURATION="$2"; shift 2 ;;
        *)                shift ;;
    esac
done
[ "$STEP" = none ] || { echo "fake sweep expects -s none, got $STEP" >&2; exit 2; }

phase() {
    python3 - "$PHASE_EVENTS_PATH" "$1" <<'PY'
import json, sys, time, pathlib
path, event = sys.argv[1:3]
pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"event": event, "ts_epoch": time.time()}) + "\n")
PY
}

phase sample_start
pids=()
for i in $(seq 0 $((CONCURRENCY - 1))); do
    # The stand-in agent drives the SDK directly, so it opts out of the adapter
    # bootstrap that would otherwise try to patch a framework that is not installed.
    NATIVE_REPLAY_ADAPTER= python3 "$HERE/agent.py" "task-$(printf '%02d' "$i")" &
    pids+=("$!")
done

deadline=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 0.2
done
phase sample_end
for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
wait || true
