# Progress

Working notes. `design.md` is the contract; this file is status only.

## Status

**Both stages work and are proven on the real mini-swe stack, and stage 1 now also
runs end to end on the real owl stack.**

⚠️ **All bundles below predate bug 13** (the launcher pinned the queued backlog to the
concurrency level, so refill re-ran finished tasks instead of drawing new ones). The
mechanics they demonstrate hold; the *workload* they ran is not the one the sweep
defaults to. Re-record before quoting any of these numbers as the workload.

- Stage 1 (`--mode tool-only`): mini-swe C1 and C8 plus the cutoff-tail path; owl C1.
- Stage 2 (`--mode full`): forced decoding, with the engine really performing
  prefill, logits and sampling while committing the recorded tokens. mini-swe only.

| | |
| --- | --- |
| Package | ~9.3k lines vs replayer-new's 17,586 (**~53%**) |
| Tests | 83 passing (unit + end-to-end), `ruff` clean |
| Interface | 6 commands: `record` / `replay` / `report` / `validate` / `vllm-up` / `vllm-down` |

## Real runs (mini-swe, Qwen3-Coder-30B TP8, real Docker containers)

### C1, 60s window

Recording: 27 LLM calls, 26 dispatches, 26 tool executions, 1 actor.
GPU genuinely worked — 41.3s active, 74% mean utilisation, 79.9 GB peak.

| metric | source | replay-0 | replay-1 |
| --- | --- | --- | --- |
| makespan | 49.6s | 8.850s | 8.855s |
| framework CPU | 1.89s | 1.47s | 1.47s |
| timeline coverage | 96.3% | 96.0% | 96.7% |
| unattributed gaps | 0 | 0 | 0 |

Replay-to-replay makespan spread **0.06%**. Replay is shorter than the recording
because tool-only drops the 39.1s of LLM time and keeps the 8.5s of real tool work.
`report` correctly flagged `llm.total_seconds` for inspection (a tiny number has
large relative spread) without failing the run.

### C8, 120s window — the concurrency case

Recording: **8 actors, 282 LLM calls, 280 dispatches, 280 tool executions**,
113.7s makespan, 18.7s CPU, 96.1s GPU active, 98.96% timeline coverage.

Replay consumed **exactly** the recorded workload, with per-actor tool counts all
correct:

```
sphinx-10449 39 | django-11299 40 | django-12143 25 | django-14672 40
django-14493 35 | xarray-6938  25 | django-15916 36 | django-11551 40   = 280
```

This is the case design §8 singles out ("并发 slot 不按完成顺序猜测"): eight actors
whose completion order is genuinely nondeterministic, matched without ever comparing
completion times.

### Cutoff tails, 20s window

Deliberately cut mid-episode: 1 LLM tail captured (1.31s elapsed). Replay re-entered
the tail, held it for its recorded duration, never answered it, and finished valid.

## Real runs (owl GAIA workforce, 6×Qwen3-Coder-30B + 1×Qwen2.5-VL-7B, 7 endpoints)

### C1, 90s window

Recording: **2 actors, 30 LLM calls across 4 native roles, 19 dispatches, 19 tool
executions**, 1 LLM cutoff tail, 0 instrumentation failures. GPU genuinely worked —
65.4s active. Timeline coverage 94.95%.

| metric | source | replay-0 | replay-1 |
| --- | --- | --- | --- |
| busy span | 81.379s | 2.385s | 2.362s |
| makespan | 83.583s | 4.621s | 4.571s |
| framework CPU | 15.00s | 16.89s | 16.60s |
| GPU active | 65.4s | 0.0s | 0.0s |
| dispatch / tool / llm | 19 / 19 / 30 | 19 / 19 / 30 | 19 / 19 / 30 |

Both replays ended `fixed-work-complete` and consumed **exactly** the recorded
workload. Replay-to-replay spread: busy span 0.97%, makespan 1.09%, framework CPU
1.7%. `report` flagged `disk_*` and `net_sent_bytes` for inspection — host-wide
counters that other containers share, as on mini-swe.

Owl is the case where per-endpoint routing matters: all 12 of owl's model
call-sites resolved through the proxy to the right one of the 7 targets, and the
recorded `llm_role` is owl's own role name (`coordinator` / `task` / `reasoning` /
`answerer`), not a single lane.

### Reading the CPU number on this host — a trap

Replay reports ~16.9s of framework CPU inside a 4.6s makespan, which looks wrong
until you attribute it. Per-PID sampling says it is almost entirely two of the 19
tool calls: `execute_code` bodies that `import numpy, pandas`. Measured directly in
owl's environment that import costs **7.3s of CPU in 0.42s of wall (1700%)** because
the BLAS/OpenMP runtime starts a thread per core, and this box has **96 cores**;
`OMP_NUM_THREADS=1` drops the same import to 0.39s of CPU. Roughly 82% of the total
is system time, i.e. thread startup and page faults, not the agent computing.

Two consequences. First, `framework_cgroup.cpu_seconds` is not comparable to
`busy_span_seconds` — one is summed across 96 possible cores and the whole process
tree, the other is a wall-clock union of instrumented spans. Second, the *useful*
comparison is user CPU: **3.04s recorded vs 3.07s replayed**, with total CPU barely
moving while wall-clock falls 18×. That is the evidence the tools really re-executed;
the work did not shrink, the waiting did.

### C8 fire-once — and the one thing that does not replay

C8 is where owl gets dirty, which is the point of running it: the recording covers
6 distinct tools (`browse_url` 5, `extract_document_content` 11, `search_duckduckgo`
9, `search_wiki` 6, `execute_code` 14, `return_json_response` 19) across 8 actors,
206 LLM calls, 64 dispatches. The recording is valid, and so is replay-0.

Replay-0's 202.5s of "tail" is **correct**, not a gap: the window closed on 8
operation tails (longest 278.9s) and 5 LLM tails, and design §6 requires replay to
re-enter each and hold it for its recorded duration.

**Replay-1 failed** — `LLM request drift for (840bfca7…, browser_web) at sequence 6:
the framework built a structurally different request`. The cause is not drift in the
framework's control flow:

- 110 of the 206 LLM calls are browser-internal (`browser_planning` 57,
  `browser_web` 53), made *inside* `browse_url` while the real Chromium drives a
  live page;
- each `browser_web` request embeds a live screenshot — 225 of them across the
  recording, 39 KB to 426 KB;
- the shape gate reduces free text to a power-of-4 size bucket, so a replay is valid
  exactly when the fresh screenshot happens to land in the same bucket as the
  recorded one. Replay-0 did. Replay-1 did not.

So this is a coin flip on live page bytes, and it is also what produced bug 15.

**This is a conflict inside design, not a bug in the code.** §4 makes the request's
structured work projection a hard gate for *every* LLM call; §5 says owl's Chromium
must genuinely run but that browser internals are diagnostics only. The browser's own
VL calls are both at once. Worth noting that the framework still follows the recorded
trajectory either way — those internal calls receive their *recorded* responses, so
only the request side is unreproducible.

**Decided: (a).** An inline `data:` payload now projects to its media type with no
size class, so a PNG turning into a JPEG still fails the gate and a dropped image
part still fails, but a screenshot's byte length no longer decides validity. Pinned
by `test_inline_image_payloads_do_not_carry_a_size_class`. Note this invalidates any
bundle containing images: `request_shape_sha256` is computed at record time.

### Two cutoff-tail bugs that only a long browser workload reaches

**Bug 16 — a tail was queued last instead of where it started.** `_load_expected`
appended tails to the end of their queue, reasoning that this "is where it was when
the window closed". That is wrong: a tail is *unfinished* at the cutoff, but it still
*began* somewhere, and a long one begins early. owl's `extract_document_content` ran
277s of a 300s window, so it started second while eight quicker operations began
after it and closed before it — yet it was queued eleventh. Replay, which reaches it
far sooner without the LLM waits, was told it had drifted. Queues are claimed on
entry (the module docstring says so), so they must be in issue order: closed records
carry `started_at_ns`, tails `source_started_at_ns`, and `_issue_order` refuses to
guess when neither is present. Two tests pin the rule in both directions. The LLM
lane was always right here — it sorts by `sequence` — so bug 5's fix had only ever
been half applied.

**Bug 17 — a nested tool tail is not frozen with its parent dispatch.** Open, found
immediately after 16. The C8 census reads `{'dispatch': 8, 'tool': 2}` with tail
names `{'browse_url': 9, 'browser_action': 1}`: every in-flight `browse_url` left a
dispatch tail but almost none left the tool tail nested inside it. Replay then claims
the LLM response, claims the dispatch tail, and the tool underneath finds no slot —
reported as `native tool invocation drift`, which is what every remaining C8 replay
failure has been.

### owl's tool granularity — what a slot is

owl already draws the line this adapter needs, and the adapter was ignoring it: a
step whose complete input sits in its arguments is a replayable primitive
(`camel.utils.replay_capture.run_tool_primitive`, `record_browser_primitive`), and
a step backed by a model is an LLM call. `browse_url` is not one operation but a
loop of them:

```
browse_url
 ├─ browser_open                     primitive
 └─ loop:  async_observe
 │          ├─ screenshot            primitive  (browser_observe)
 │          └─ web_agent.step()      LLM slot   (role browser_web)
 │        async_act(action_code)     primitive  (browser_action)
 └─ browser_close                    primitive
```

Recording only the outer `FunctionTool` put all of that inside one opaque slot —
**64 tool records where owl's own step3 lane had 176** for the same run, with the
per-step arguments, ordering and outcomes invisible.

Three patch points close it, all against methods verified to exist on the installed
owl (`async_act` and `_close_browser_primitive` are on `AsyncBrowserToolkit`, not on
`AsyncBaseBrowser` as the enclosing-method search first suggested):

| patch | primitives it covers | recorded observation |
| --- | --- | --- |
| `replay_capture.run_tool_primitive` | document extraction ×2, video download, frame extraction | each call's own return |
| `AsyncBrowserToolkit.async_act` | `browser_action` | `(success, info)` |
| `AsyncBrowserToolkit._close_browser_primitive` | `browser_close` | — |

`run_tool_primitive` is owl's own wrapper and takes the callable, so one patch covers
both the document and video chains without the adapter knowing which toolkit raised
them. `async_act` earns a real slot rather than bookkeeping because its
`(success, info)` goes into `trajectory_info` → `self.history` → the next prompt, so
the recorded outcome is what fixes the browser's onward control flow.

**Not covered by L1, by design:** `browser_open` and `browser_observe` are *segments*
inside `browse_url` and `async_observe`, not wrappable methods, and the observation
owl records for the screenshot is only `{"screenshot_path": …}` — a path, not the
bytes. Handing a recorded path to a fresh run fixes nothing. Those two belong to L2,
where the screenshot becomes a bundle artifact; only then does the VL request carry
recorded bytes, and only then can the image projection below be deleted outright.

### A second way the browser breaks a recording

Re-recording C8 under the new rule died after 19s: `LLM store failure: Server
disconnected`. The fleet was healthy; the VL container's own log has the reason —
`ValueError: Input length (33214) exceeds model's maximum context length (32768)`.
A slightly larger live screenshot pushed the prompt past `qwen2.5-vl-7b.yaml`'s
`max_model_len`, and because the request was streaming, vLLM failed after headers
were sent and dropped the connection.

Once in 90 minutes of running, so it is a coin flip rather than a wall — but it kills
a whole recording when it lands. Two separable problems:

1. **Serving.** 32768 is marginal for owl's browser workload. This is a property of
   the native stack, not of the replayer: an ordinary owl run hits the same ceiling.
   Raising the VL endpoint's `max_model_len` would make it rarer.
2. **This harness.** An upstream error currently fails the *run*, where natively owl
   would receive the error and handle it — its own error path is visible in the C8
   logs. Design §5 already settles the principle for tools ("Source 中由 native
   implementation 直接抛出的异常也是录制 observation"); the LLM lane has no
   equivalent, so any upstream fault is a harness failure rather than a recorded
   observation that replay re-raises. That gap is worth closing on its own merits —
   it is not specific to screenshots.

**Superseded options** (kept for the reasoning):

- **(a) drop image payloads from the shape projection** — an `image_url` becomes
  `{"kind": "image"}` with no size class. Everything structural stays gated: message
  count and roles, tool schemas, sampling configuration, order, identity. This treats
  a screenshot's byte length as exactly what §5 calls a browser internal.
- **(b) claim browser-internal roles by position only**, leaving the shape gate
  untouched elsewhere. Narrower, but it needs a list of which roles are
  browser-internal, which is owl-specific knowledge in a general lane.
- **(c) leave it** and accept that any owl workload reaching `browse_url` replays
  only by luck.

### Independent corroboration of the tool ledger (step3 lane)

The owl checkout writes its own step 3 tool lane whenever `STEP3_TOOL_LOG` is set —
`camel/toolkits/function_tool.py` plus `camel/utils/replay_capture.py`, a code path
with no connection to this adapter — and `scripts/owl/start.sh` already forwards the
variable into its tmux session. So putting it in `config.env` costs nothing and gives
a second, independent observation of the same run to check the ledger against.

Done once, on a recording: **15 tool executions on both sides, identical name
sequence** (9 `execute_code`, 6 `return_json_response`), per-call durations agreeing
to **0.1–2.7 ms**. The sign is always the same — minireplay's span is the longer one,
because it wraps `FunctionTool.__call__` from outside while owl's hook sits inside, so
the difference *is* this harness's own overhead: **14.8 ms across 15 tools**.

In this particular recording both refills re-ran the *same* GAIA task, so step3's
`task_id` chain label could not tell them apart while minireplay's `refill-000001`
could. That is not how steady mode behaves — it was bug 13 below, and with the queued
backlog restored, refill draws the next distinct task and `task_id` separates them on
its own. Pair the two lanes **by time order** anyway: the labels are only equivalent
until the seeded pool wraps.

The LLM lane is available the same way and is not yet done: the fleet already exports
per-request OTLP spans (`start_vllm_multi.sh` turns `--otlp-traces-endpoint` on by
default, and each container carries `OTEL_RESOURCE_ATTRIBUTES=vllm.port=…`), so
running step3's receiver during a recording would give an engine-side ground truth to
check `llm.jsonl` against, split by endpoint. In tool-only replay that lane should be
*empty* — itself a checkable claim — and under `--mode full` it is the independent
evidence that the engine really ran.

### What owl needs in `config.env`

Two entries, both using mechanisms that already existed:

```json
"env": {
  "NATIVE_REPLAY_FRAMEWORK_PYTHON": "/mnt/raid0/Jirong/miniconda3/envs/owl/bin/python",
  "PYTHONPATH": "/mnt/raid0/Jirong/HPCA/multiagent/frameworks/owl"
}
```

`FRAMEWORK_PYTHON` confines instrumentation to owl's interpreter. Without it every
Python the sweep shells out to installs the adapter, and owl's sweep shells out to
`conda` — whose own launcher then died importing `camel`, taking the sweep with it
(silent `exit 70`, empty log). `PYTHONPATH` is needed because owl's `camel/` and
`utils/` are plain directories in its checkout, and `sitecustomize` runs before
`sys.path[0]` is set to the script's directory.

## Bugs found and fixed by real running

The unit tests did not catch these; running did.

1. **`json.dumps([])` is the truthy string `"[]"`** — an empty actor inventory
   rejected every actor. Recording now omits the variable entirely.
2. **A dead sweep was not noticed**, so the gate wait hung for its full 600s.
3. **`TMUX_TMPDIR` was declared but never created**, so mini-swe's launcher failed.
4. **`PROCESS_ROLE=native-batch` disables instrumentation and is inherited.** Since
   the sweep is launched directly (no wrapper), this would have silently produced an
   uninstrumented run.
5. **LLM cutoff tails were never captured** (`freeze_source_cutoff` returned `[]`),
   and **neither LLM nor operation tails were added to the replay claim queues**, so
   a tail could never be claimed. Both violate design §6.
6. **Post-window work was treated as drift.** A recording is cut mid-episode, so a
   correct replay reaches the end with the framework still wanting to continue. That
   request is now held forever (`errors.WorkloadComplete`); drift *inside* the window
   is still a hard failure.
7. **`config.env` could replace `PYTHONPATH`**, silently disabling instrumentation.
   It now appends.
8. **Tail duration used the caller's clock**, not the ledger's.

Then four more, all found by running owl for the first time. Every one of them was
invisible to the static patch-point check below, which is the point: that check
proves the hooks *attach*, not that the run *works*.

9. **The owl adapter read a model-role attribute nothing sets.** It looked for
   `_minireplay_role`; owl's own launcher tags each model `_agent_replay_role`
   (`make_model` in `run_gaia_workforce_vllm_flex.py`), which is the only place the
   native role survives to the model object. A renamed copy fails *every* LLM call.
   replayer-new has the same bug under a third name, which is how it got here.
10. **`identity_bindings` was consumed but never produced.** Recording wrote each
    actor's owl worker identities to `runtime-identities/`, replay read them from
    `manifest.identity_bindings` — and `build_bundle` never put them there. The
    failure mode was maximally confusing: `KeyError` per task, owl counts a failed
    task as a finished one and refills, and the run dies as *actor drift* several
    steps downstream ("unexpected LLM request for refill-000003").
11. **`/v1/models` answered an empty list in tool-only replay.** owl asks each
    endpoint what it serves before building any client, so discovery raised
    `IndexError` on an endpoint the window never exercised. The recording now keeps
    each target's answer in the bundle and replay serves it back.
12. **The tmux socket path could exceed the ~108-byte `sun_path` limit.** A deep
    enough `--out` made the framework launcher fail with `error connecting to ...
    (File name too long)`, buried in a framework log. `TMUX_TMPDIR` now falls back
    to a short run-owned directory, removed at teardown.
13. **The launcher pinned the queued backlog to the concurrency level** — it passed
    `-n <concurrency>` where every sweep defaults `-n` to the whole seeded pool.
    That is a workload override, which design §2 forbids outright, and it quietly
    changed what refill *means*: with the queue holding only the tasks already in
    flight, a finishing task could only be replaced by itself. At owl C1 the pool
    collapsed from 165 tasks to 1, so both "refill" actors re-ran the same GAIA
    question instead of drawing the seeded order's next task. `-n` is no longer
    passed; C1 is still the seeded order's first task and C8 still its first eight,
    because that is simply what W in-flight slots draw first. Two tests now pin it:
    the exact argv, and a standing assertion that neither `-n` nor `--num-tasks` is
    ever passed for any framework or concurrency.
    **Every bundle recorded before this is a different workload and needs re-recording.**

    The interface is now framework-independent: `config.refill` selects the load
    model and is part of every framework's workload identity. The launcher passes
    the common `--refill` / `--no-refill` sweep flag. Under no-refill the shared
    sweep driver samples exactly the seeded order's first C tasks; under refill it
    leaves the native sweep's full backlog intact. Owl's old `--no-steady` remains
    only as a compatibility alias, and old config `load_model` values are accepted
    as input-only aliases.
14. **owl's serial gate could only ever run in replay.** At C1 with `--no-steady`,
    owl takes the stock `run_workforce_with_retry` path rather than the process
    pool, and the adapter's gate for it began by requiring
    `NATIVE_REPLAY_ACTOR_MAP` to hold exactly one actor. Recording resolves no task
    list — an actor names itself at the gate — so the map is empty and every
    recording died with `Owl C1 serial workload must declare exactly one actor`.
    Same shape as bug 10: code written against replay's preconditions, never
    exercised by a recording. The serial path now names its actor from the task
    index when there is no inventory to pin against, which both sides derive
    identically (`idx-61`).
15. **The proxy rejected a legitimate request as "Content Too Large."** aiohttp caps
    request bodies at 1 MiB by default and both services took that default, so at C8
    — where owl's histories are longer and its multimodal endpoint carries base64
    frames — a request the framework would have sent to vLLM was refused by the
    harness instead, failing the recording outright. C1 never came close, which is
    why it took a concurrency step to surface. The cap is now 512 MiB: high enough
    that the proxy is transparent to anything the workload does, low enough that a
    runaway body still cannot silently exhaust memory.

## Adapter patch points verified against the installed frameworks

All 28 targets present; the ported instrumentation matches what is installed.

| framework | patch points | note |
| --- | --- | --- |
| mini-swe | 7/7 | mini-swe-agent 2.4.5 |
| trae | 6/6 | |
| coral | 6/6 | |
| owl | 9/9 | needs `PYTHONPATH`; import order matters (`camel.toolkits` first) |

The check imports each framework in its own environment and confirms every target
attribute is still there — it does not run anything, and owl showed how little that
proves: 9/9 hooks attached and the adapter still could not complete one LLM call
(bugs 9-12 above). Treat this table as "nothing was renamed", not as readiness.

mini-swe (C1, C8) and owl (C1) are proven end to end; trae and coral are not.
**Two landmines of the bug-10 kind are already visible for them**, both reading an
environment variable nothing sets — `coral.py:405` needs
`NATIVE_REPLAY_OPENCODE_IDENTITY` and `trae.py:405` needs `LITELLM_BASE_URL`.
coral's is in `install()` itself, so it will fail on the first instrumented process.

## Pre-existing environment problem, now fixed

`multiagent` was moved from `/mnt/raid0/Jirong/multiagent` to
`/mnt/raid0/Jirong/HPCA/multiagent`, which left every editable install pointing at a
path that no longer exists — `import minisweagent` / `trae_agent` / `coral` failed in
their own environments, for any use of these checkouts, not just the replayer.

The per-config `env.PYTHONPATH` workaround has been replaced by repointing the `.pth`
files themselves (one line each; the original is kept beside it as `.bak-<timestamp>`):

| import | `.pth` |
| --- | --- |
| `minisweagent` | `envs/sweagent/…/__editable__.mini_swe_agent-2.4.5.pth` |
| `trae_agent` | `frameworks/trae-agent/.venv` (py3.12) `_editable_impl_trae_agent.pth` |
| `coral` | `frameworks/CORAL/.venv` (py3.13) `_editable_impl_coral.pth` |
| `frontier_cs` | same venv, `_editable_impl_frontier_cs.pth` — coral's task data |

All four import cleanly now, and no `.pth` anywhere in these environments still names
the old path. Note that coral and trae install into venvs inside their own checkouts,
not into the conda env of the same name — checking the wrong interpreter reports a
missing module that is in fact installed.

**owl is not covered by this**, because owl never had an editable install: `camel/` and
`utils/` are plain directories in the owl checkout, so owl resolves imports from its
repo root. Its `needs PYTHONPATH` note above is about that layout, not about the move.

Unrelated, but easy to confuse with the above: `supervisor` always injects its own
`PYTHONPATH` (`bootstrap` + package root) to carry `sitecustomize`, and appends rather
than replaces whatever `config.env` asks for.

## Changes to `multiagent` (three files, all purely additive)

`scripts/lib/sweep_common.sh` — existing `-s 1|2|both` unchanged:

- `SWEEP_STEP=none`: attach no step1/step2 profiler. The window was already driven by
  the driver's own `sleep`, not by the monitors, so this needed no restructuring.
- `SWEEP_SKIP_VLLM=1`: skip prefix-cache reset and serving preflight, for tool-only
  replay whose LLM endpoint is served from a bundle.

Verified `-s both` still behaves identically and `-s bogus` is still rejected.

`serving/scripts/start_vllm_multi.sh` — two new optional variables, empty by default
so ordinary runs are unaffected:

- `VLLM_EXTRA_ENV` — space-separated `NAME=VALUE` added as `-e` to every instance;
- `VLLM_EXTRA_MOUNTS` — space-separated `/host:/container` added as `-v`.

`scripts/owl/sweep.sh` — one line: `SWEEP_WARMUP=30` became `: "${SWEEP_WARMUP:=30}"`,
the same default-if-unset idiom `sweep_common.sh` already uses for its own knobs.

The warmup exists so the step1/step2 monitors settle before tasks launch. Under
`SWEEP_STEP=none` nothing is attached, so there is nothing to settle, and with
`SWEEP_RANDOM_WARMUP=0` it sends no traffic either — it was 30s of pure sleep. The
launcher had been exporting `SWEEP_WARMUP=0` since the beginning; the hard assignment
silently discarded it, which is the same failure shape as bug 7.

Verified all four paths: no env and no flag still gives 30s; `SWEEP_WARMUP=0` in the
environment now gives 0s; an explicit `-w 7` still beats the environment; `-s both`
and the `-s bogus` rejection are unchanged. **The other three sweeps hard-assign it
too** (mini-swe 10s, trae 30s, coral 30s) — same one-line fix when each is run.

Measured, re-recording and replaying twice afterwards (all three `valid`): the gap
between `monitors_skipped` and `framework_workload_start` went from **30.09s to
0.09s**, record wall from ~127s to **97s**, and replay wall from ~50s to **20s / 19s**
— a 3× faster iteration on the loop you actually repeat.

## Stage 2: forced decoding — working

`minireplay-vllm:v0.19.0` is built from the running `vllm/vllm-openai:v0.19.0`
(a new tag; nothing existing was touched).

### Real run, mini-swe C1 / 60s

Recording ran in **capture mode**: the plugin observes (never alters) which sampler
steps the engine actually committed, and that window goes into the bundle. Every one
of the 18 calls got one — e.g. the engine sampled 114 steps and committed 113 tokens,
so a forced replay must skip the 1-token uncommitted suffix.

| metric | source | replay-0 | replay-1 |
| --- | --- | --- | --- |
| **busy span** (first→last activity) | **36.28s** | **36.64s** | **36.72s** |
| makespan (gate→terminal) | 54.84s | 36.68s | 36.81s |
| idle tail | 18.52s | 0 | 0 |
| **GPU active** | **29.4s** | **27.1s** | **27.5s** |
| GPU mean utilisation | 48.2% | 69.8% | — |
| framework CPU | 2.13s | 1.27s | 1.33s |

**Compare `busy_span_seconds`, not `makespan_seconds`.** A recording's makespan is
the sweep's window, which keeps running after the agent finishes — here the agent
submitted its patch and the window idled for another 18.5s. A replay ends when the
recorded work is consumed, so it has no such tail. On the comparable number the
replay is within **1.0%** of the recording, and per lane: LLM 29.04→29.73s (+2.4%),
tool 6.68→6.65s (−0.4%), dispatch 6.75→6.72s (−0.4%).

**GPU active is the point.** In tool-only replay it is 0.0s; here the engine really
runs. Replay-to-replay spread: makespan 0.35%, GPU 1.5%.

The audit file shows 18 `capture`/`capture_complete` records from the recording and
18 `force`/`complete` from the replay, each with `forced_count=113`,
`sampled_token_count=114`, `committed_sample_start=0` — the engine forced exactly the
recorded tokens at exactly the recorded step window.

`report` flagged `disk_*`/`net_*` for inspection: those are host-wide counters and
other containers share this box. Flagged, not failed — which is the intended
behaviour.

### How it is wired (decision: option **c**)

Serving topology stays in `multiagent/serving/scripts/start_vllm_multi.sh`.
`minireplay vllm-up` calls it and adds only what forced decoding needs:

```bash
minireplay vllm-up   --config run.json     # patched image + secret + audit mount
minireplay record    --config run.json --out src/ --bundle b/
minireplay replay    --config run.json --bundle b/ --out r0/ --mode full
minireplay vllm-down --config run.json
```

`vllm-up` then verifies the container is genuinely forced-capable (both source
patches, the secret, the audit path variable, a writable audit file) and fails if
not. A `--mode full` run against a bundle with no engine window is refused rather
than silently downgraded.

## Superseded: earlier notes on stage 2

`minireplay-vllm:v0.19.0` is built from the running `vllm/vllm-openai:v0.19.0`
(a new tag; nothing existing was touched). Verified inside the image:

- both source patches applied (post-sampling commitment hook, discard mask publish);
- entry point `native_replay_forced` registered;
- `ForcedSequenceProcessor.apply` is a no-op, so prefill/logits/sampling all really run.

`tests/test_forced_protocol.py` pins that the proxy and engine sign byte-identical
payloads — they are separate implementations on purpose, so something must check they
still agree.

### Open decision — who starts vLLM?

Forced decoding needs vLLM started with the patched image, a shared
`NATIVE_REPLAY_FORCE_SECRET`, and a writable audit path. Recording also needs
*capture mode* to learn each call's engine step window, which current bundles lack —
so **stage-2 bundles require re-recording** (expected: design §8 treats an
instrumentation change as a re-record trigger).

Two ways to wire it:

- **(a) minireplayer owns the vLLM lifecycle**, as replayer-new did. Self-contained,
  but duplicates `multiagent/serving/scripts/start_vllm_multi.sh`, which already
  handles CDI-vs-nvidia runtime, cpusets, page-cache warmup and multi-instance
  topology. That is the kind of complexity this rewrite exists to remove.
- **(b) the operator starts vLLM; minireplayer validates and refuses to run
  `--mode full` unless the plugin and secret are present.** Smaller, and keeps one
  owner for serving topology.

I lean **(b)**. Until it is decided, `--mode full` fails loudly rather than silently
reporting GPU work it did not do.

## Not done

- trae and coral real runs. Their patch points are present but nothing has been
  executed, and the two unset environment variables above say what will break first.
  Each needs its own vLLM topology (trae TP4+TP2+TP2, coral 4×TP2), so each costs a
  fleet restart.
- owl beyond C1: C8, and `--mode full`. The owl bundle is capture-mode capable (it
  was recorded against the patched image), so forced decoding needs no re-record.
- C32, and the formal 1200s window.
- Forced decoding at C8+ (only mini-swe C1 has been run in `--mode full`).

## Housekeeping

`minireplay-vllm:v0.19.0` is currently running as **owl's topology** — six
`vllm-Qwen3-Coder-30B-{0..5}` on GPUs 0-5 (ports 8000-8005) plus
`vllm-Qwen2.5-VL-7B-6` on GPU 6 (port 8006), leaving GPU 7 for owl's in-process
whisper. Going back to mini-swe means another fleet restart.

`config.serving.gpu_mode` is `nvidia` because this daemon has no `features.cdi` and
the serving script warns that enabling it would bounce other users' containers.

```bash
minireplay vllm-down --config <config>
```

Working configs: `/tmp/mini-c1-forced.json` (mini-swe C1, forced-capable),
`/tmp/mini-c8-smoke.json` (mini-swe C8, tool-only), `/tmp/owl-c1.json` (owl C1,
forced-capable). The owl one declares seven `targets` (`vllm-8000`…`vllm-8006`) and
the same config pair `scripts/owl/start_vllm.sh` uses, so `vllm-up` reproduces owl's
native topology rather than a second description of it.
