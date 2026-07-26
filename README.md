# minireplayer

Sweep-derived agent workload replayer for CPU-harvesting experiments. The normative
contract is [`design.md`](design.md).

The workload is always resolved and executed by the `multiagent` sweep scripts. This
tool records what a framework did inside one sweep window, and replays that fixed
workload against a fresh workspace so two deployments can be compared on equal work.

During recording, each causal actor/session appends its own boundary events locally.
There is no record-side global dispatch reservation or first-wave barrier. After the
sweep closes, those lane logs are materialized into the validated ledgers and packed
as `bundle/lanes/*/events.jsonl`; the bundle root is only a run manifest and terminal
summary. Replay still validates each lane's next slot before native work starts.

Calls still active at source cutoff are retained as tail evidence. Ordinary tails
stop replay before native entry. A composite CORAL `task` tail is different: replay
enters the parent task so its already-completed subagent LLM/tools can run, then
withholds the unfinished parent result. An LLM response that completed before tool
dispatch is also retained; its tool call gets an explicit zero-duration
`pre_dispatch` marker and no tool record.

## What is fixed, and what is not

Fixed (a replay that deviates is invalid):

- which LLM calls happen, in which per-actor order, and what the framework sees back;
- which top-level native tools are invoked, with which arguments, in which per-actor
  order;
- the observation each tool returns to the framework, which is what fixes its next
  control-flow decision.

Not fixed (recorded as evidence, never compared):

- tool stdout, temporary paths, PIDs, child-process detail, browser internals;
- wall-clock durations, CPU, GPU, I/O — those are what the experiment measures.

## Replay modes

- `--mode tool-only` — tools execute for real; the LLM lane is answered from the
  bundle and vLLM is not contacted. Cheap, and enough to iterate on tool behaviour.
- `--mode full` — the recorded request also goes to a real vLLM, which performs real
  prefill, logits and sampling while committing the recorded tokens. Needs a fleet
  started by `minireplay vllm-up` and a bundle recorded against it; a run that asks
  for it otherwise fails rather than silently reporting GPU work it did not do.

## Workflow

```bash
cat > run.json <<'JSON'
{
  "schema_version": "minireplay.config/v1",
  "framework": "mini-swe",
  "repo": "/mnt/raid0/Jirong/HPCA/multiagent",
  "concurrency": 1,
  "refill": true,
  "duration_s": 180,
  "seed": 42,
  "targets": {"vllm-8000": "http://127.0.0.1:8000"},
  "gpu_ids": [0]
}
JSON

minireplay record --config run.json --out source-run/ --bundle bundle/
minireplay replay --config run.json --bundle bundle/ --out replay-0/
minireplay replay --config run.json --bundle bundle/ --out replay-1/
minireplay report --bundle bundle/ --run replay-0/ --run replay-1/ --source source-run/
minireplay plot-comparison \
  --bundle bundle/ --source source-run/ \
  --run replay-0/ --run replay-1/ --out comparison/
```

`record` runs the sweep with `-s none`, which attaches no step1/step2 profiler; the
window, refill and `sample_end` boundary are still the sweep's own. Step3 does not
go through the sweep's `-s` interface: every successful recording exports its
captured LLM/tool lanes (with cutoff tails marked as source-only diagnostics) to
`source-run/step3/raw/` and
renders `source-run/step3/views/timeline.{png,txt}` automatically.
For CORAL, manager-driven team termination is also exported as
`lane_terminations.jsonl` and marked on all four child lanes independently of the
later recording-window boundary. The post-termination hatch denotes no possible
lane work and is excluded from timeline coverage.
Native CORAL invocation transitions are exported separately as
`restart_events.jsonl`. They remain manager-driven control intervals rather than
replayed tools; invocation-scoped session identities ensure that later fixed
LLM/tool work cannot be consumed by the wrong pre-restart process.

Recording and full replay use the configured serving warmup before the actor gate.
The native sweep resets the prefix/KV cache before warmup and again after warmup,
so compilation and sampler startup are excluded without carrying warmup prompts
into the measured causal lanes. The formal default is a 15-second warmup followed
by a 180-second source window. Tool-only replay sends no serving traffic.

`refill` has the same meaning for every framework. When true, a completed task is
replaced with the next task in seeded order while the native pool has work. When
false, only the first `concurrency` tasks run and concurrency decays as they finish.
Because this changes the workload, it is part of bundle identity.

CORAL has a grouped concurrency window: `concurrency` counts four-agent teams, not
individual OpenCode processes. C8 therefore means 8 team slots and 32 agent lanes.
A refill replaces one complete four-agent team after that team has exited and been
cleaned up; it never fills individual agent positions. New bundles preserve this as
`concurrency_window` with team slot, generation, parent, and four child lane IDs.
CORAL defaults to 100 model turns per invocation, 10 invocations across the team,
and `restart_exited=true`; all three remain explicit workload-identity fields.

`report` compares repeated replays and flags metrics whose spread is worth a look. It
does not fail a run for being slow, and it does not attribute variance automatically
— compilation, package fetches and network access move these numbers legitimately.

`plot-comparison` validates the fixed operation counts and per-actor closed work,
then emits SVG/PNG batch wallclock and full causal-lane timelines plus per-lane
JSON/CSV. Repeat `--run` for any number of replays. Dispatch wrappers are omitted;
`browse_url` remains a composite scope rather than a filled tool, and recording
cutoff tails are shown only as transparent, hatched evidence.

## Checks

```bash
ruff check .
pytest -q
```
