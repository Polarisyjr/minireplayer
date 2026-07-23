# Owl autoresearch log

持续实验日志。`OWL-STATUS.md` 是上一轮人工交接，保持只读；`design.md` 是规范。本文件只
记录本轮 autoresearch 的假设、实际运行、失败证据、修复和下一步，不用成功结果覆盖失败。

## 目标

1. 在当前统一 causal-lane、`evidence-only` cutoff 语义下重新录制 Owl。
2. Replay 必须使用 full 模式：LLM 进入 forced-capable vLLM，native tool 真实执行。
3. Full replay 统一执行 `reset prefix/KV -> warmup -> reset prefix/KV -> gate`。
4. 先拿到 C1 record + 两次 clean full replay，再提升到 C8；每次失败先归因再复测。
5. 保留 source、bundle、失败 run、有效 replay、report、审计和关键日志。

## 基线

- 2026-07-23：MiniSWE C1/C8/C32 full replay 已验证；C32 两次有效 replay 为
  34.107s / 34.365s，380 LLM + 376 native tools，cutoff tail replay entry 为 0。
- Git 首版已推送：`main@7f219bd` (`Polarisyjr/minireplayer`)。
- 提交前门禁：`ruff check .`；`120 passed`。
- 上一轮 Owl 状态：C1 tool-only 已通过；C8 卡在旧语义下的嵌套 cutoff tail。当前
  cutoff tails 已改为 source-only evidence，因此必须用新录制重新判断，不能复用旧结论。

## 实验日志

### E00 — 发现当前 Owl 运行资产（完成）

假设：旧 scratchpad 中仍有可复用的 7-endpoint target/role mapping 和 fire-once 配置；若
不存在，则从 `multiagent/scripts/owl` 与 serving YAML 重建最小 C1 配置。

结果：

- 当前是 MiniSWE TP8 fleet：单个 `vllm-Qwen3-Coder-30B-0` 占满 GPU 0--7，只有
  8000 端口；开始 Owl 前必须切换拓扑。
- Owl Python 为 `/mnt/raid0/Jirong/miniconda3/envs/owl/bin/python`，核心 patch point
  import 探针全部通过。import 时 ffmpeg 报 `libavdevice.so.62` 缺失；C1 index 61 是无
  附件纯文本题，不受影响，但把该问题列为 C8 的 video 工具风险。
- 找回 `/tmp/owl-c1.json`；它带有已失效 scratchpad 的 `STEP3_TOOL_LOG`，只作为参数
  参考，不直接运行。旧 `run_scope.sh` / `crosscheck.py` 未在 `/tmp` 找到。
- dry-run 固定 workload：valid/all、seed 42、fire-once 的 C1 是 index 61，task id
  `6f37996b-2ac7-44b0-8e68-6d28256631b4`，pool size 165。
- 新实验根为 `results/owl-autoresearch-20260723`。三个 C1 config 具有相同 workload
  identity，分别把 Owl 自身 tool lane 写到 source/replay1/replay2 独立文件。
- 使用 180s source watchdog；fire-once 完成可提前结束。record 和 full replay 都统一
  `SWEEP_WARMUP=10`，原生 sweep 在 warmup 前后各 reset 一次 prefix/KV cache。

### E01 — 切换并验证 Owl forced-capable fleet（完成）

结果：

- `minireplay vllm-down` 正常关闭 MiniSWE TP8；GPU 0--7 均回到 4 MiB 基线。
- `minireplay vllm-up` 启动 7 个 forced-capable 容器，audit 为
  `results/owl-autoresearch-20260723/.state/audit/forced-audit.jsonl`。
- 8000--8005 均返回 `Qwen3-Coder-30B`，8006 返回 `Qwen3-VL-8B`；所有端点 HTTP 200。
- 服务是顺序启动，健康数按 3/7 -> 4/7 -> 5/7 -> 6/7 -> 7/7 增长，没有容器退出。
  GPU 0--5 分别承载一个 Coder，GPU 6 承载 VL；GPU 7 留给 Owl 本地 whisper。
- vLLM 日志中的 OTLP `172.17.0.1:4317` 不可用只影响旧 step3 receiver，不影响
  minireplayer 自己的 forced audit；本轮独立工具证据由 `STEP3_TOOL_LOG` 直接记录。

### E02 — C1 新语义录制（完成）

命令：

```bash
.venv/bin/python -m minireplay record \
  --config results/owl-autoresearch-20260723/configs/c1-record.json \
  --out results/owl-autoresearch-20260723/c1-source \
  --bundle results/owl-autoresearch-20260723/c1-bundle \
  --run-id owl-full-c1-source-20260723
```

验收：source valid；固定 actor/index 61；LLM/tool/dispatch 闭合前缀非空；诊断 tail 不进入
bundle replay ledger；Owl 自身 tool lane 与 minireplayer 顶层 tool 顺序可核对。

第一次尝试（失败，保留为 `c1-source`）：actor gate 前原生 sweep rc=1。证据
`random_warmup.json` 显示 8000 的 8 个 worker 成功、8006 成功，而 8001--8005 的
首批请求都在 10s 内超时。端点和容器没有退出；这是新 fleet 后启动端点的首次请求编译
尚未完成，不是 Owl workload 或 gate 漂移。失败 warmup 已触发各端点执行路径；重试仍由
sweep 先 reset，再 warmup，成功后第二次 reset，因此不会继承失败 run 的 prefix cache。

第二次尝试写入 `c1-source-r2`，bundle 仍使用尚不存在的正式路径 `c1-bundle`。

第二次内容验收通过：bundle `558ea35e6f9fec84` valid，task accuracy=1，闭合前缀为
19 LLM + 11 dispatch + 11 tool，0 cutoff tails，instrumentation failure=0；Owl 自身
step3 lane 为 11 个 start/end 配对，与 minireplayer 数量一致。发现新的测量问题：task 与
busy span 都在约 46s 完成，但 source makespan 为 175.890s。根因是 common sweep 先无条件
`sleep $SWEEP_DURATION`，之后才运行 early-stop；同时 Owl 的 `pgrep -f` 会匹配留在 tmux
里的保活 shell。

修复（`multiagent` 工作树，未覆盖其既有改动）：

- `sweep_common.sh` 在测量窗口内轮询 fire-once framework；只有观察到 alive 后再消失才
  提前发 `sample_end`，避免把慢启动当完成；profiler 仍有 5s flush。
- Owl 的存活探针改为检查 tmux pane shell 的直接子进程。Python/tee pipeline 运行时有
  子进程，`=== DONE ===; exec bash` 后没有，不再由命令行文本猜测。
- `bash -n`、8s 合成 early-stop 探针（3s 返回）和 Owl dry-run 都通过。

第三次重录使用 `c1-source-r3` / `c1-bundle-r3`，独立 Owl tool lane，目标是 source
makespan 与 terminal/busy span 同量级。`c1-source-r2` 与 `c1-bundle` 继续作为修复前证据。

第三次结果：通过。bundle id `558ea35e6f9fec84`，valid，accuracy=1，16 LLM +
9 dispatch + 9 tool，0 tails，Owl tool lane 为 9 个完整 start/end 对。source
`makespan=35.986s`、`busy_span=33.614s`、GPU active=27.841s；native log 明确出现
`framework finished before the 180s window`。第二、三次 bundle id 相同是当前实现按 workload
identity 命名，不代表 ledger 内容相同；正式 replay 固定使用 `c1-bundle-r3` 路径。

### E03 — C1 full replay #1（完成）

使用 `c1-replay1.json`；期望消费 16 LLM + 9 dispatch + 9 tool，LLM forced audit 必须非空，
reason 必须是 `fixed-work-complete`，并核对 Owl 独立 tool lane 的 9 次真实执行。

第一次尝试（失败，保留为 `c1-replay1`）：第一个 task-role forced request 已由 port 8001
完成并留下 `status=complete` audit；并发的 coordinator sequence 0 在 port 8000 fatal，错误
为 prompt token drift，首差异位于 `return_json_response` 的 JSON Schema 键顺序。source
ledger prompt tokens 与 capture audit 逐 token 相同，排除了 capture 关联错误。

根因是通用持久化层：录制转发的是保留 Python dict 插入顺序的 request；`append_jsonl`
为了 canonical evidence 递归 `sort_keys=True`，bundle 再加载后得到字母序 dict。Qwen chat
template 会按 JSON Schema 插入顺序渲染工具定义，所以语义相同的 request 产生了不同 prompt
tokens。修复没有给 Owl 加例外：每条新 LLM record 增加 `request_ordered_json` opaque string；
full replay 从该字段恢复原始 mapping order，并校验它与 canonical `request` 在 JSON 语义上
相等。旧 bundle 无该字段时保持兼容。

定向门禁：ruff clean；forced/order、LLM store、端到端 supervisor 共 `19 passed`。下一步跑
全套测试，重启被故意 fail-fast 拉死的 port 8000，然后必须重新录制带 ordered request 的
bundle；旧 `c1-bundle-r3` 只保留为失败证据。

全套门禁通过：ruff clean，`121 passed`。fleet 正常 down/up 后 8000--8006 再次全部
HTTP 200。第四次录制使用 `c1-source-r4` / `c1-bundle-r4` 与独立 tool lane；这是第一个
携带 `request_ordered_json` 的 Owl bundle。

第四次在 gate 前失败并保留：冷 fleet 上每端点并发 8 的 10s warmup 形成单卡排队，多个
worker 超时，但所有容器仍健康。C1 setting 改为 `SWEEP_WARMUP_CONCURRENCY=1`；仍是 10s
随机流量，仍执行 warmup 前后各一次 reset，并同步用于 source/replay1/replay2。这更匹配
C1 每个角色端点的实际并发。第五次写入 `c1-source-r5` / `c1-bundle-r5`。

第五次录制通过：valid、accuracy=1、17 LLM + 10 dispatch + 10 tool、0 tails；source
`makespan=53.933s`、`busy_span=51.084s`、GPU active=42.235s。10 个工具中一次
`extract_excel_content('table.xlsx')` 按原生语义返回 FileNotFoundError，Owl 自身 lane 也
记录同一次失败，任务随后正常恢复并答对，因此它是 workload observation，不是基础设施
失败。17/17 LLM 都携带 ordered request。对 coordinator sequence 0 构造实际 forced body
并调用 port 8000 `/render`，得到 777 个 prompt tokens，与 source **逐 token 相同**。

E03 第二次尝试使用 `c1-bundle-r5`、输出 `c1-replay1-r2` 与独立 tool lane。

E03 第二次尝试通过：valid，reason=`fixed-work-complete`，17 LLM + 10 dispatch +
10 tool，0 tails/gap，instrumentation failure=0；17/17 forced audit 均 `complete`，每条
prompt tokens 和 forced committed tokens 都与 source 精确相等。角色分布为 answerer 1、
coordinator 4、reasoning 10、task 2。Owl 独立 lane 有 10 个 start/end 对，并复现同一个
`extract_excel_content` 原生错误。`makespan=51.323s`、`busy_span=51.416s`、GPU
active=39.448s。replay 在 ledger 完成时停住，早于 Owl 写最终 task terminal，符合 fixed-work
边界，且不是遗漏（所有三个 ledger 和 audit 已消费完）。

### E04 — C1 full replay #2（完成）

使用同一个 `c1-bundle-r5` 和相同 warmup/reset setting，输出 `c1-replay2`；验收条件与
E03 相同。完成后生成 source + 两次 full replay report。

结果：通过。valid，reason=`fixed-work-complete`，17 LLM + 10 dispatch + 10 tool，
0 tails/gap；17/17 forced audit 都是 `complete`，prompt 与 committed output token 均与
source 精确一致。Owl 独立工具 lane 有 10 个完整调用。`makespan=51.264s`、
`busy_span=51.374s`、GPU active=40.166s。

`c1-full-report.json` valid 且 reasons 为空。两次 replay 的 makespan 为 51.323s / 51.264s，
relative spread=0.12%；busy span spread=0.08%；GPU active spread=1.8%。报告只将 host
disk/network/framework CPU 的环境波动标为需要观察，不影响回放有效性。正式 C1 存档为：

- source：`c1-source-r5`
- bundle：`c1-bundle-r5`
- replay：`c1-replay1-r2`、`c1-replay2`
- report：`c1-full-report.json`
- Owl native tool evidence：`evidence/c1-*.owl-tools.jsonl`

### E05 — C8 录制与两次 full replay（完成）

沿用旧 Owl fire-once workload identity：concurrency=8、duration=300s、seed=42、
refill=false；7 个 endpoint 和角色映射不变。每次 source/replay 都执行 10s random warmup，
warmup 前后 reset prefix/KV，C8 warmup concurrency=8。source、两次 replay 各写独立 Owl
native tool lane。先 dry-run 固定八个任务，再录制新 causal-lane/evidence-only bundle；任何
失败 run 都原地保留并在本节追加归因。

dry-run 固定 index 为 `[61,91,111,42,146,135,89,11]`；启动前 8000--8006 全部健康。
正式 source 写入 `c8-source` / `c8-bundle`，run id `owl-full-c8-source-20260723`。

录制通过：8/8 actor terminal status 都是 success；bundle `4bde07237fb062e8` valid，闭合
前缀为 228 LLM + 71 dispatch + 71 tool，228/228 LLM 均有 `request_ordered_json`。
source `makespan=256.843s`、`busy_span=253.558s`、GPU active=181.605s。Owl 独立日志中
71 个顶层 tool start 都有对应 end；此外浏览器内部 primitive 使用 end-only 事件记录，不能
误当成缺 start 的顶层调用。

source 有 0 个 LLM tail 和 10 个 operation tail：它们全是长时间未返回的浏览/文档
dispatch（8 个 `browse_url`、2 个 `extract_document_content`）。它们只存在于
`cutoff-tails.json` 和 manifest lane 计数中，未进入
71 个闭合 replay ledger；这正是 `evidence-only` 边界语义，不能把它们当成第 72 个工具或
在 replay 末尾续跑。接下来 replay #1 必须只消费 228/71/71 并以
`fixed-work-complete` 结束。

replay #1 第一次尝试保留为 `c8-replay1`，在约 30s fail-fast：actor
`e2d69698-...` 的闭合 dispatch cursor 期望第二个 `return_json_response`，实际先到的是
source cutoff tail `browse_url`。这不是输出或参数 drift：source 中该 browse branch 从约
9.36s 一直挂到 cutoff，同时同 actor 的其他并发 subtask 后来仍产生闭合 LLM/tool。当前
boundary 只按 `(kind, actor, adapter-specific lane)` 排队，Owl 的 lane 为 None，错误地给
并发分支强加总序；同时只有在整个 actor 闭合前缀消费完以后才能挡住 tail，所以挡得太晚。

修复方向必须通用：dispatch 用显式 `causal_lane`，缺省由 session + LLM model-call/trigger
派生，tool 继承其 dispatch lane；bundle 的 cutoff tail 也保留同一 causal lane。replay 若
命中该 lane 的 source-only tail，应在 native entry 前永久 hold 该分支，但不消费/执行 tail，
其他并发 lane 继续消费闭合前缀。不能增加 Owl adapter 条件，也不能退回 actor-wide fast
claim。修复后要重录，因为旧 tail 没保存 origin/causal-lane 元数据。

通用实现与门禁完成：新格式显式持久化 `causal_lane`；未显式提供时 dispatch 由
model-call（无 model-call 时为 session + trigger）派生，tool 通过 dispatch id 继承；tail
命中在 boundary start 时转成 source-window hold。老 bundle 保留只读兼容路径（老 CORAL
session lane 或 actor FIFO），新路径没有 Owl 条件。新增反序并发分支与 interleaved cutoff
分支测试；ruff clean，完整 `123 passed`。重新录制写入 `c8-source-r2` / `c8-bundle-r2`，
原 source/bundle/replay1 继续保留为发现该问题的证据。

第二次录制通过：`c8-source-r2` / `c8-bundle-r2` valid，8/8 task terminal success，闭合
前缀 263 LLM + 79 dispatch + 79 tool；263/263 ordered request，79/79 dispatch 和
79/79 tool 都有 causal lane，且所有 tool lane 与父 dispatch 完全一致。source
`makespan=293.620s`、`busy_span=290.327s`、GPU active=225.988s。Owl native log 的
79 个顶层 start 全部配对 end。

新 source 同样有 0 LLM tail、10 dispatch tail（9 browse + 1 document）；10/10 都有非空
且互异的 causal lane。用真实 bundle 构造不带显式 lane、只带原始 session/origin 的 tail
entry probe，boundary 正确派生相同 lane 并抛出 source-cutoff hold，证明 replay 不依赖
adapter 注入私有字段。replay #1 第二次尝试使用 `c8-bundle-r2`，输出 `c8-replay1-r2`。

replay #1 第二次尝试保留为 `c8-replay1-r2`：79/79 dispatch/tool 和 130 个非浏览 LLM
均已消费，但 Owl 提前结束，bundle 尚缺 133 个 `browser_planning/browser_web` LLM。先前把
它解释为“父 tail 的不可达子 LLM”不够准确；回看 raw lane event 后确认，10 个 dispatch
tail 都是 instrumentation 制造的假 tail：对应的 10 个外层 orchestration tool 以及 69 个
内部 primitive 在 source cutoff 前已经真实完成。SDK 的 dispatch execution capture 却把
outer tool 和 nested primitives 都列为同一个 framework dispatch 的直接执行，
`_dispatch_resolution` 因长度大于 1 抛 `one framework dispatch entered multiple native tool
executions`，而异常发生在 invoke 返回后的 completion 路径外，最终留下未闭合 dispatch。

通用修复：dispatch 只把 `_tool_call` 为空时进入的最外层 tool 记为 direct execution；处于
outer tool context 的 primitive 仍各自进入 tool ledger，但作为后代，不再污染 dispatch 的
一对一 resolution。没有 Owl/tool-name 判断。新增 outer + nested primitive SDK 测试，确保
dispatch 的 `execution_call_id` 只指 outer tool；ruff clean，完整 `124 passed`。需要第三次
录制 `c8-source-r3` / `c8-bundle-r3`：预期上述 10 dispatch 正常闭合，内部 133 LLM 和
69 primitive 也一起成为可达 fixed work；若八个 task 在 300s 内结束，应没有这 10 个假
tail。

第三次录制保留为 `c8-source-r3`，它在 300s 边界前被 native sweep rc=1 中止，未封包。
这次 workload 本身没有 `multiple native tool executions`（原生日志 0 次），证明 SDK 修复
有效；失败来自上一轮 early-stop shell 修复的 deadline 分支：
`[ "$now" -lt "$deadline" ] || return` 的 bare `return` 继承了 `[` 的 status=1，在
`set -e` 下恰好于完整窗口结束时退出 setting，来不及发 sample_end。改为 `return 0`；提前
完成路径不变。

r3 的未物化 raw stage 也提供了 cutoff-closure 证据：207/207 个 browser LLM span 都有非空
parent，指向 outer tool/primitive；r2 在 `close_stage_at_cutoff` 后之所以全变成 parent=null，
是旧逻辑把已删父 span 的 child **re-root** 了。统一 evidence-only 语义要求父 operation 若在
cutoff 未闭合，子 LLM/tool 不能成为独立 fixed work。因此 cutoff fixed-point 改为按 span
parent 跨 record kind 删除整条不可达后代链，不再 re-root；新增 cutoff tool 下 child LLM
删除测试。这样真实 300s tail 会保留为 source evidence，但其内部未形成闭合因果前缀的工作
不会被 replay。

完整门禁在上述两项修复后为 ruff clean、`125 passed`；deadline 合成 probe 返回
`deadline-ok`。第四次录制写入 `c8-source-r4` / `c8-bundle-r4`。

r4 通过并形成真正的 300s cutoff：bundle `4bde07237fb062e8` valid，3/8 task 在窗口内
terminal success，其余 5 条 lane 在 sample_end 截断。source `makespan=292.853s`、
`busy_span=290.654s`、GPU active=263.882s。raw closed stage 有 334 LLM + 69 dispatch +
186 tool；causal fixed-point 删除 cutoff parent 下 27 LLM + 13 tool 后，正式闭合前缀为
307 LLM + 69 dispatch + 173 tool。另有 4 LLM tails 和 9 operation tails（4 dispatch +
5 tool）仅作 source evidence，9/9 operation tail 都有 causal lane。

307/307 LLM 有 ordered request；69 dispatch 的 causal lane 互异，173/173 tool 都与父
dispatch lane 一致。原生日志没有 `multiple native tool executions`。Owl 独立日志有 52
个顶层可直接执行 tool start/end 对，另有 244 个 orchestration-internal/primitive end 事件；
这和 minireplayer 的 173 个 fixed tool 不是同一层计数，后者已由因果图验证。replay #1
第三次尝试使用 `c8-bundle-r4`，输出 `c8-replay1-r3`。

replay #1 第三次尝试保留为 `c8-replay1-r3`：约 28s 时 port 8006 的首个
`browser_web` forced request 返回 HTTP 500。此前已有 50 条并发请求完成；失败请求本身
prompt 长 2831、source committed output 192 tokens。容器没有 OOM；engine fatal 的精确
原因是 `native replay unresolved output placeholder`。vLLM 0.19 的 async scheduling 会在
长多模态 prefill 后将尚未完成 GPU-to-CPU 回填的输出暂存为尾部 `-1`，旧 processor 将合法
异步占位误判为 commit drift。

通用修复把 output 分为已回填实 token 前缀与连续 `-1` 尾占位：实 token 必须逐项等于
forced 前缀，占位后不得再出现实 token，logical output 不得超过当前 sampler 位置一格；
因此仍然 fail-closed，不会放宽 token identity。ruff clean，完整 `130 passed`。重建同名
vLLM image 并重启 fleet 后，用失败的同一条 VL request 独立验证：HTTP 200，返回 192
tokens；audit=`complete`，forced=192/192、sampler=193/193、prompt=2831 精确一致。
replay #1 第四次尝试继续固定 `c8-bundle-r4`，写入 `c8-replay1-r4`，不会覆盖前三次现场。

第四次在 actor gate 前失败并保留：刚重建/重启的六个 Coder endpoint 首轮
concurrency=8 random warmup 触发 CUDA graph/shape 冷编译，8000--8005 各有 3--5 个 worker
超过 warmup client timeout；VL 8006 全部成功，所有晚到 Coder 请求也最终返回 200，容器无
异常。这次没有消费任何 replay ledger。保持实验 setting 不变，利用这轮完成冷预热后，第五
次写入 `c8-replay1-r5`。

replay #1 第五次通过：verdict valid，reason=`fixed-work-complete`，307 LLM + 69 dispatch +
173 tool 全部消费，replay cutoff tails 为空，unattributed gap=0；307 条本轮 forced audit 全为
`force/complete`，request id 全唯一，prompt/forced/sampler count 全精确。source 与 replay 的
LLM、dispatch、tool **逐 actor 计数全部一致**。Owl 独立工具日志有 52 个 direct start，52/52
都有 end；其余 215 条是 orchestration/internal end-only evidence，所有 end success=true。
`makespan=285.098s`、`busy_span=285.071s`、GPU active=246.797s。接下来 replay #2 沿用
同一个 `c8-bundle-r4`、同一 10s concurrency=8 warmup/reset setting。

replay #2 第一次尝试保留为 `c8-replay2`，它暴露了 run-level completion 竞态：verdict 一度
写成 fixed-work-complete，但 stage 只有 306/307 LLM，最后一条 actor `023e9d44-...` 请求的
engine audit 是 incomplete（forced 18/55、sampler 18/56）。根因是 LLM queue 在 request
claim 时就前移到末尾；supervisor 将“所有 slot 已 claim”误当成“所有 request 已完成”，在
最后一条仍生成时 teardown。dispatch/tool 均完整，这不是 lane drift。

通用完成协议现区分 claimed、evidence-written、response-delivered：LLM 必须完成 upstream
forced audit、落 stage evidence，并把 recorded response 写到 framework socket 后才计入
run/actor complete；operation 同样要在 complete response 写出后才计入 complete。超出 source
prefix 的下一条仍可在 claim/start 入口 hold，不会因此多执行。新增 LLM 与 operation
delivery-barrier 测试，并更新 diagnostic-tail 断言；ruff clean，完整 `131 passed`。第二次
replay #2 写入 `c8-replay2-r2`，旧的 306/307 run 不覆盖。

replay #2 第二次通过：verdict valid，reason=`fixed-work-complete`，307 LLM + 69 dispatch +
173 tool；起始 audit 行 2083，本轮新增恰好 307 条，全部 `force/complete`、request id 唯一，
prompt/forced/sampler identity 全精确。source 与 replay 的三类 ledger 逐 actor 计数完全一致；
Owl native tool evidence 仍为 52 个 direct start 全配对、267 个 end 全 success。`makespan=
282.618s`、`busy_span=282.591s`、GPU active=245.153s，证明 delivery barrier 修复了第一
次的 306/307 竞态。

正式 `c8-full-report.json` valid、reasons 为空。replay1/replay2 makespan 为 285.098s /
282.618s，relative spread=0.87%；busy span spread=0.87%；GPU active spread=0.67%。所有
operation count spread=0，LLM total time spread=0.74%，tool total time spread=0.51%。唯一
`needs_attention` 是 host disk read bytes（82.06% spread），属于环境 I/O 计数，不参与
work identity，也没有造成 timeline gap（两轮 coverage=0.9999、gap=0）。source 的 4 LLM
和 9 operation cutoff tails 只存在于 source evidence，两个 replay 的 cutoff-tails 都为空。

逐 actor fixed-work wallclock（actor ID 前 8 位）也没有出现 count 抵消；每行的 LLM / dispatch /
tool 数在 source、replay1、replay2 三者完全一致：

| actor | LLM/D/T | source s | replay1 s | replay2 s | replay spread |
|---|---:|---:|---:|---:|---:|
| `023e9d44` | 61/13/35 | 287.449 | 284.232 | 280.387 | 1.36% |
| `6f37996b` | 15/8/8 | 48.095 | 49.119 | 48.539 | 1.19% |
| `7dd30055` | 20/12/12 | 61.350 | 61.959 | 61.710 | 0.40% |
| `840bfca7` | 58/5/32 | 243.515 | 240.120 | 244.725 | 1.90% |
| `ded28325` | 11/5/5 | 18.182 | 18.611 | 18.919 | 1.65% |
| `e2d69698` | 34/9/20 | 164.036 | 170.404 | 163.705 | 4.01% |
| `e961a717` | 52/6/29 | 287.858 | 285.064 | 280.681 | 1.55% |
| `f46b4380` | 56/11/32 | 290.650 | 281.681 | 282.588 | 0.32% |

正式 C8 存档为：

- source：`c8-source-r4`
- bundle：`c8-bundle-r4`（bundle id `4bde07237fb062e8`）
- replay：`c8-replay1-r5`、`c8-replay2-r2`
- report：`c8-full-report.json`
- Owl native tool evidence：`evidence/c8-replay1-r5.owl-tools.jsonl`、
  `evidence/c8-replay2-r2.owl-tools.jsonl`

所有中间失败存档均保留：初始 actor-FIFO/cutoff 失败、causal-lane 后的 nested execution
失败、300s deadline shell 失败、VL async placeholder engine fatal、冷 fleet warmup timeout，
以及 completion barrier 前的 306/307 replay。没有覆盖或删除任一现场。

### E06 — 当前结论

C1 与 C8 都完成一录两次 full replay，LLM 确实进入 vLLM forward/sampling 并由 engine audit
逐 token 证明，工具链真实执行并独立留有 Owl evidence。C8 还实际覆盖了 evidence-only
cutoff、并发 causal lane、outer+nested primitive、多模态 async scheduling 和最后一条
in-flight completion 等 C1 不会触发的边界。正式结果目录在
`results/owl-autoresearch-20260723/`；本文件是试错日志，原 `OWL-STATUS.md` 未修改。

### E07 — browser_web 输出上限与 C8 重录

在 `browse_url` 改为仅作透明 composite scope、内部 browser primitive/VLM 分别记录之后，
`c8-source-r7` 暴露一个 `browser_web` Qwen3-VL 请求从 gate 后 42.607s 一直未返回到
293.540s cutoff。该请求携带两张截图且未发送 `max_tokens`；同一 8006 实例在此期间仍完成
79 个后续请求，因此不是整个服务阻塞。

请求侧配置现仅对 `roles.browser_web.extra_config` 设置 `max_tokens: 4096`，不影响共用 8006
的 image/video 角色。全新重启 6×Coder + 1×VL fleet 后，先做 30s 编译预热，再按正式
reset → 10s warmup → reset → gate 流程录制 `c8-source-r8` / `c8-bundle-r8`。

r8 valid，window=293.391s，闭合前缀为 252 LLM + 61 dispatch + 231 tool；cutoff evidence
为 3 个 `browser_planning` LLM tail 和 1 个 `browser_action` tail，没有 `browser_web`
tail。71/71 个完成的 `browser_web` 请求都携带 `max_tokens=4096` 且 `finish_reason=stop`；
延迟 median=2.977s、P95=3.824s、max=4.508s，输出最大 253 tokens。r7 中长挂的 actor
`ded28325-...` 在 r8 于约 21s terminal success，但它这次没有走 browser/VL 分支，因此这轮
证明配置下发和当前 workload 轨迹正常，不能把“同一异常请求已由 token cap 修好”作为强因果
结论。

### E08 — C8 r8 full replay 与同参 cutoff tail

第一次 full replay `c8-r8-replay1` 在 239 条 LLM evidence / 240 条 forced audit 后
fail-closed：actor `840bfca7-...` 的外层 `web` 队列已消费 6 个录制响应，却发起第 7 个
请求。初步按“独立 lane 提前耗尽”处理后，第二次 `c8-r8-replay1-r2` 越过原失败点，但停在
246 LLM + 61 dispatch + 225 tool；缺口恰好全属于该 actor（3 个 `browser_planning`、
3 个 `browser_web`、6 个 browser primitive），证明只挂起额外请求不能恢复已中断的闭合
工作。

最终按 source lane-event、LLM 内容和 tool 参数逐项对齐，根因是第三个透明 `browse_url`
scope 中，第一条闭合 `browser_action` 与 source cutoff 时最后一条未闭合 action 参数完全
相同，均为 `fill_input_id(24, 'Carolyn Collins Petersen')`。boundary 原先先按调用 identity
匹配 cutoff tail、再匹配同一 causal lane 尚未消费的 closed queue，因此把第一条闭合工作
错认成最后的 tail 并提前 hold；30s 后 Owl 报 browse timeout，外层才产生第 7 个 `web`
请求。这不是 `browse_url` 工具重放问题：它始终只是 composite scope，真正参与 ledger 的是
内部 LLM 和 browser primitive。

通用修复将同一 lane 的匹配顺序改为：先严格消费全部 closed fixed prefix，queue 耗尽后才
允许匹配 evidence-only cutoff tail；若存在已知 tail 但下一调用 identity 不同，则
fail-closed 为 `cutoff-tail drift`，不能被“lane 已完成”规则掩盖。新增“closed action 与
cutoff tail 同名同参”以及错误 tail identity 回归测试。

第三次 `c8-r8-replay1-r3` 正式通过：verdict valid，reason=`fixed-work-complete`，
252 LLM + 61 dispatch + 231 tool 与 source 的 ID 集合及逐 actor 计数完全一致；actor
`840bfca7-...` 从失败现场的 44/44 补齐到 50 LLM / 50 tool。审计基线 3425 后新增恰好
252 条，全部 `force/complete`、request id 唯一，`expected_sampled_token_count` 与实际
sampler count 全部一致。replay cutoff tails 为空，`makespan=293.313s`、
`busy_span=293.257s`、timeline coverage=0.9998、gap=0。

replay Step3 图仅绘制 252 个真实 LLM span 和 231 个真实内部/顶层 tool interval；
`browse_url` 没有实心条，也不计入工作量。最终门禁为 ruff clean、`134 passed`。

### E09 — C8 r8 第二次 clean replay 与三运行对比图

第二次 replay 前向 8000--8006 逐 endpoint 发送 30s、concurrency=8 的生成预热流量；
6 个 Qwen3-Coder-30B endpoint 和 1 个 Qwen3-VL-8B endpoint 的 8 个 worker 均至少成功
完成请求，随后 7 个 `/v1/models` 检查全部正常。正式 replay 仍独立执行
reset → 10s warmup → reset → gate，输出 `c8-r8-replay2`，Owl tool evidence 写入
`evidence/c8-r8-replay2.owl-tools.jsonl`。

`c8-r8-replay2` valid，reason=`fixed-work-complete`，297.734s 完成 252 LLM +
61 dispatch + 231 tool；三类 source ID 集合完全一致，replay cutoff tails 为空。forced
audit 基线 3677 后新增恰好 252 条，全部 `force/complete`、request id 唯一，
`expected_sampled_token_count` 与实际 sampler count 全部一致。两次 clean replay 的
makespan 为 293.313s / 297.734s，差 4.421s（1.507%）；busy span 为
293.257s / 297.710s。

当时新增三运行绘图入口，按 C32 图的结构同时绘制 Record、Full replay 1、
Full replay 2；该入口现已统一收进 `minireplay plot-comparison`：

- `c8-r8-record-two-replays-batch-and-per-lane-wallclock.{svg,png}`
- `c8-r8-record-two-replays-full-chain-lanes-timeline.{svg,png}`
- `c8-r8-record-two-replays-per-lane.{json,csv}`
- `c8-r8-two-replays-report.json`

图中只填充真实 LLM 和 native tool；dispatch wrapper 省略以避免重复计算，`browse_url`
composite scope 不作为工具绘制。source 的 3 条 LLM tail 和 1 条 tool tail 仅在 Record
栏显示为红色斜线，两个 replay 均不执行这些 evidence-only tails。

### E10 — Owl C32 / 120s source recording

沿用 r8 fleet、角色映射和 `browser_web max_tokens=4096`，配置 C32、duration=120s、
seed=42、refill=false；正式录制仍执行 reset → 10s warmup → reset → gate。输出
`c32-source-r1` / `c32-bundle-r1`，bundle id `0a2c9e71af7ea3a0`。

录制 valid，reason=`sweep-sample-end`，32 个 actor 的闭合前缀为 431 LLM +
198 dispatch + 363 native tool。sweep 的 `sample_start` 到 `sample_end` 为约 120.73s；
32 actor ready barrier 在 sample start 后约 8.69s 打开，因此从 source gate 计的
makespan/busy span 为 112.042s / 112.034s，这与 C8 的 300s 配置显示约 293s 是同一
时间口径，不是 duration 被缩短。

source cutoff evidence 为 16 LLM tails（10 browser_planning、4 browser_web、1 reasoning、
1 web）和 7 operation tails（1 dispatch、6 tool）。53 条闭合 `browser_web` 请求全部携带
`max_tokens=4096` 且 `finish_reason=stop`。`browse_url` 仍只有 24 个 composite scope
outline，闭合 tool 中 `browse_url` 数量为 0；Step3 实心工作只来自真实 LLM/native tool。
一条 `video_download` 因 YouTube HTTP 403 以 error result 闭合并被正常录入固定前缀。

### E11 — Owl C32 两次 clean full replay 与三运行分析图

使用同一个 `c32-bundle-r1`，分别输出 `c32-r1-replay1` / `c32-r1-replay2`，两次均执行
reset → 10s warmup → reset → gate。两次 verdict 均为 valid，
reason=`fixed-work-complete`，各完成 431 LLM + 198 dispatch + 363 native tool；
LLM attempt id、dispatch id 和 tool call id 与 source 集合完全一致，replay cutoff tails
均为空。

Replay 1/2 makespan 为 118.142s / 119.094s，busy span 为 118.123s / 119.052s；
Replay 2 比 Replay 1 慢 0.952s（0.806%）。每次各新增 431 条 forced audit，全部
`force/complete`、request id 唯一，`expected_sampled_token_count` 与实际 sampler count
一致。两次 timeline 均无 >=2s gap，报告 valid。

使用正式的 `minireplay plot-comparison` 功能从 bundle 自动读取 workload、actor 数和
fixed-work counts；`--run` 可重复，因此同一入口支持任意 replay 数量。本次 C32 产物由
以下命令重新生成：

```bash
minireplay plot-comparison \
  --bundle results/owl-autoresearch-20260723/c32-bundle-r1 \
  --source results/owl-autoresearch-20260723/c32-source-r1 \
  --run results/owl-autoresearch-20260723/c32-r1-replay1 \
  --run results/owl-autoresearch-20260723/c32-r1-replay2 \
  --out results/owl-autoresearch-20260723 \
  --prefix c32-r1-record-two-replays --label "Owl C32"
```

输出：

- `c32-r1-record-two-replays-batch-and-per-lane-wallclock.{svg,png}`
- `c32-r1-record-two-replays-full-chain-lanes-timeline.{svg,png}`
- `c32-r1-record-two-replays-per-lane.{json,csv}`
- `c32-r1-two-replays-report.json`

32 lane 的 Replay 2−Replay 1 wallclock delta 均值/中位数为 -0.601s / -0.167s，
P95 absolute delta=5.803s，最大 absolute delta=11.825s。最大项来自 actor
`e8cb5b03-...`：Replay 1 的 document extraction/native fetch 明显更慢；反向较大的
`0ff53813-...` 则是 Replay 2 的第二组 document extraction 约 15.8s，体现真实 native
network/document 路径的缓存与服务方波动。尽管单 lane 有移动，batch spread 仅 0.8%。
