# CORAL autoresearch log

持续实验日志。规范仍以 `design.md` 为准，执行清单见 `CORAL-CHECKLIST.md`。本文件记录本轮
假设、命令、运行结果、失败证据、修复和下一步；失败 run 不删除、不覆盖。

## 目标与边界

1. 参考已完成的 MiniSWE 与 Owl 流程，为 CORAL 拿到可审计的 record 和 clean full replay。
2. 2026-07-24 当前正式默认改为 warmup=15s、duration=180s、
   `agents.max_turns=100`、`agents.max_total_turns=10`、
   `agents.restart_exited=true`；后文 E01--E08 的 3/8/false/60s 是上一轮历史实验参数，
   不再代表当前默认。
3. 在正式录制前用多轮 smoke 分离三个概念：
   - `agents.max_turns`：一次 agent runtime invocation 内允许的 model turns；
   - `agents.max_total_turns`：整个 CORAL team 启动的 agent invocation 共享预算；
   - `agents.restart_exited`：单次 invocation 退出后，manager 是否续起/恢复。
4. 优先做配置和 minireplayer 外围接入改动；除非证据表明是通用正确性问题，不改 CORAL
   framework 业务逻辑。

## 基线

- 2026-07-23：MiniSWE C1/C8/C32 full replay 已完成。
- 2026-07-23：Owl C1/C8/C32 与 refill 路径已完成；当前 minireplayer
  `main@51a6355`，工作树在本轮开始时干净。
- `multiagent` 工作树已有多项与本任务无关的修改/未跟踪文件；本轮不得覆盖它们。
- minireplayer 已有 CORAL instrumentation，包含 agent runtime、grader、attempt artifact、
  task terminal 和 team turn/restart summary。
- 当前 native CORAL sweep 仍通过 `run_queue.py` 固定传入
  `agents.restart_exited=false`，只把 `-m/--max-turns` 传成 per-agent
  `agents.max_turns`；minireplayer 的 `RunConfig` 尚无 CORAL global-turn/restart 参数。

## 初始静态读码结论（待 smoke 验证）

- `agents.max_turns` 默认 200，被传给每次 OpenCode runtime；它不是 team 级共享计数。
- manager 的 `_turn_count()` 定义为初始 `agents.count` 加所有 restart 次数，因此
  `agents.max_total_turns` 统计的是 agent invocation 数，不是 OpenCode 内部 model turns。
- `restart_exited=false` 时，agent 退出后不会自动恢复；有限 team 在所有初始 agent 都退出后
  terminal。若 agents.count=4，它通常只消耗 4 个 invocation，即使 global budget=8。
- `restart_exited=true` 时，dead agent、heartbeat 或 timeout 可以创建新 invocation；global
  budget 限制这些 invocation 的总数。达到预算后 manager 会停止仍存活的 team。
- 因而 global=8 只有配合 restart=true 才可能被完整消耗；但它的终止时点可能在第 8 个
  invocation 刚启动后立即截断，这一点必须以 smoke 的 manifest、agent log 和边界事件验证。

## 实验日志

### E00 — 建立研究资产与静态定位（完成）

已建立本日志和 checklist。实验前状态：

- minireplayer 工作树原本干净；本轮只新增本文和 checklist。
- multiagent 工作树已有与本任务无关的修改和未跟踪文件，本轮不覆盖。
- 8 张 A100 均为 4 MiB 空闲，8000--8006 未监听；没有活动 CORAL/OpenCode/vLLM。
- 有一个 13 小时前 Trae tmux 保活 shell，但没有 GPU/端口子进程，本轮不清理它。
- CORAL 自身 manager async-resume 定向测试 `5 passed`。

### E01 — 合成 turn / exit smoke（完成）

可复现实验与输出位于
`results/coral-autoresearch-20260723/evidence/manager_turn_smoke.py` 和
`manager-turn-smoke-results.json`。

结果：

1. 直接运行 OpenCode 的 `_tee_and_limit`，分别设 `max_turns=1` 与 3。它精确在第 1/3
   个 `step_finish` 写出 `*.turn-limit.json`，随后 SIGKILL 独立 agent process group；
   两次都没有读取或改变 manager global 计数。这确认 agent turn 是一次 invocation 内的
   model step 硬上限。
2. 模拟 4 个初始 invocation 均已结束，`restart_exited=false, max_total_turns=8`：
   restart 为 0，最终 `_turn_count=4`，team 正常 one-shot terminal；global 8 没有耗尽。
3. 相同初态，`restart_exited=true, max_total_turns=8`：manager 各续起 agent-1..4 一次，
   `_turn_count=8` 后立即 `stop_all(immediate=True)`；停止瞬间四个新 invocation 全部 alive，
   并写 `termination_reason=max_total_turns` 与 cutoff 时间。

因此 user-facing 的 “global turn=8” 实际不是 8 个 model turn，而是 8 次 team-wide agent
invocation（4 个初始 + 4 个 restart/heartbeat resume）。`restart=false` 不可能在四 agent
默认拓扑上耗尽 8；`restart=true` 会耗尽 8，但当前 manager 会把最后一波刚启动的
invocation 当 cutoff tail 立即停掉。下一步仍需在真实 OpenCode + vLLM 上复核，不能只凭 fake
handle 决定正式参数。

### E02 — 真实 turn / exit smoke（完成）

在 forced-capable 4×TP2 Qwen3-Coder-30B fleet 上，对同一条
`frontier_cs_algo/14` 依次运行：

- A：agent-turn=1，global=8，restart=false；
- B：agent-turn=1，global=8，restart=true；
- B2：重复 B；
- C：agent-turn=3，global=8，restart=true。

四轮均核对 OpenCode log、turn-limit marker、`replay_recording.json`、restart 日志和进程
退出。结果：

| smoke | restart | agent max | invocation count | restart count | 完成的 model turns | cutoff |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | false | 1 | 4 | 0 | 4（每 agent 1） | 无，one-shot terminal |
| B | true | 1 | 8 | 4 | 6（初始 4 + agent-3/4 restart 各 1） | global=8 立即 cutoff |
| B2 | true | 1 | 8 | 4 | 4（四个 restart 均 0） | global=8 立即 cutoff |
| C | true | 3 | 8 | 4 | 12（初始每 agent 3，restart 均 0） | global=8 立即 cutoff |

B 与相同参数 B2 的 fixed model work 不同，原因不是模型输出：global 预算只在 manager
monitor tick 中按“启动 invocation”计数。B 中较快的 agent-3/4 已完成 restart invocation，
而较慢的 agent-1/2 初始 invocation 尚未结束；等最后两个 restart 启动、总数到 8 时才统一
cutoff。B2 四个初始 invocation 更同步，最后一波全部在首个 model turn 前被停。

结论：`restart_exited=true + global=8` 固定的是 process invocation 数，不固定实际 model
work；同参重复已观察到 4 与 6 model turns，给 record/replay 引入没有收益的调度敏感分支。
正式 autoresearch 选择：

- `coral_restart_exited=false`：保持 CORAL 本来为 replay-recording 定义的 finite one-shot；
- `coral_global_turns=8`：按用户要求保留并写入 workload identity/manager config，但在
  restart=false 下安全地只消耗四个初始 invocation；
- `coral_agent_turns=3`：每个初始 agent 精确三个 model turns，共最多 12，足以覆盖
  LLM/tool 路径，同时保持短跑；
- `duration_s=60`：比 Owl 180/300s 短，且覆盖真实 C smoke 的完成时间。

这不需要改 CORAL framework 业务代码。只给 minireplayer config/launcher 与
`multiagent/scripts/coral/{sweep.sh,run_queue.py}` 增加了显式参数透传；默认仍是历史
`agent=200/global=0/restart=false`。

### E03 — CORAL minireplayer 接入预检（完成）

dry-run 已确认 seed=42 的 nested task 顺序固定，且下列三项从 RunConfig 一直透传到
CORAL manager：

- `coral_agent_turns=3` → `agents.max_turns=3`；
- `coral_global_turns=8` → `agents.max_total_turns=8`；
- `coral_restart_exited=false` → `agents.restart_exited=false`。

同时修复了几个已有 CORAL port 遗留：instrumentation 的插件文件名、proxy URL 重复
`/v1`、OpenCode implementation identity，以及 team actor 在 gate 前没有 ready。改动都在
minireplayer 接线层；CORAL framework 业务逻辑未改。

### E04 — C1 录制迭代与正式双回放（完成）

C1 前四轮失败样本全部保留。它们依次暴露并帮助修复：

1. CORAL team 没有进入通用 ready/gate；
2. OpenCode base URL 形成 `/v1/v1`；
3. streamed tool-call ID 在字节交给 framework 后才建索引，dispatch 会抢先到达；
4. role target 在 actor hash 后解析，四个 agent 被错误路由到同一 endpoint；
5. streaming source 没有保存 engine audit；
6. full replay 发起 forced streaming request 后未 drain SSE，engine generation 提前取消；
7. fixed-work replay 主动 teardown 被误报成 native task failure。

正式 source 为 `c1-source-r5`，bundle 为 `c1-bundle-r5`：

- bundle ID `b35290e022c7e000`；
- 5 actors，16 LLM、12 dispatch、12 tool，artifact/grader 均为 0；
- 每个真实 agent target 精确 4 个 LLM；
- source task terminal success，0 cutoff tail，bundle validation 通过。

两次 clean full replay：

| run | verdict | LLM / dispatch / tool | forced audit | makespan | busy span |
| --- | --- | --- | --- | ---: | ---: |
| `c1-replay1-r3` | valid | 16 / 12 / 12 | 16 force、16 complete、0 error | 11.658s | 5.714s |
| `c1-replay2` | valid | 16 / 12 / 12 | 16 force、16 complete、0 error | 11.460s | 5.937s |

两轮每条 audit 均满足 `sampled_token_count = forced_token_count + 1`。busy-span
relative spread 3.83%，makespan spread 1.71%；report valid，comparison PNG/SVG 已目检。

### E05 — C8 并发失败与唯一 framework 修复（完成）

首次 C8 source 同时启动八个 queue worker，3 个 task 在创建 workspace 时失败。根因不是
replay：多个 run 对同一 `task_dir/latest` 执行 `unlink`/`symlink`，存在 TOCTOU race。

这是 CORAL workspace 的通用并发正确性问题，因此保留本轮唯一 framework 代码改动：

- 每个 writer 创建唯一临时 symlink；
- 用 `os.replace` 原子发布 `latest`；
- finally 清理临时链接。

16 个并发更新的定向测试稳定通过；CORAL workspace suite `10 passed`。没有修改 manager
turn、调度或 agent 业务逻辑。

本轮还在 minireplayer 收紧两项失败语义：

- 任一已发布 task terminal 为 failure，整个 run 必须 failure；
- 默认 record run ID 增加随机后缀，重试相同 workload 不会复用 framework 输出目录。

### E06 — 内置 `task`/subagent 因果覆盖探针（完成，有显式安全门禁）

C8 的若干失败 source（r3--r7、r9）出现了 Qwen 自主选择 OpenCode 内置 `task` 工具。
它可绕过普通 `tool.execute.before`，若 parent operation 不在 bundle，child LLM/tool 会在
因果闭包中被删除，旧校验仍可能让一个静默缺工作的 bundle 通过。

处理分两层：

1. OpenCode plugin 同时观察 `message.part.updated` 的 pending/running/error 状态，以共享
   start promise 幂等创建 dispatch/tool reservation；r9 debug 证明 parent task 在 child
   session 创建前已成功 start。
2. bundle validator 递归提取 streamed/non-streamed provider response 中的 tool-call ID，
   要求每个 CORAL `(actor, call_id)` 都有 dispatch 或 cutoff evidence。缺一条即拒绝封包。

r9 的 parent task 落在极短 turn/进程终止边界上，最终没有形成闭合 dispatch；validator
按预期拒绝该 source。这里没有为了挽救随机边界样本去修改 CORAL agent 生命周期，也没有
伪造 parent completion。正式策略是：短跑中选择因果闭合的 source；任何再现此边界的样本
都 fail closed，绝不静默少 replay subagent work。

### E07 — C8 正式 source 与双回放（完成）

正式 source 为 `c8-source-r8`，bundle 为 `c8-bundle-r8`：

- bundle ID `947522f31c103703`；
- workload：C8、duration=60s、agent-turn=3、global-turn=8、
  restart=false、refill=false、seed=42；
- 40 actors，121 LLM、120 dispatch、120 tool，artifact/grader 均为 0；
- 8 个 source task terminal 全 success；
- 0 operation/LLM cutoff tail，因果闭包 discarded=0；
- tool-call coverage gate 与完整 bundle validation 均通过。

该 source 没有模型生成的内置 `task` 调用；它用于正式 replay。E06 的异常样本和 gate
测试仍保留，负责覆盖随机出现 `task` 时的 fail-closed 行为。

两次 clean full replay：

| run | verdict | LLM / dispatch / tool | forced audit | makespan | busy span |
| --- | --- | --- | --- | ---: | ---: |
| `c8-replay1-r8` | valid | 121 / 120 / 120 | 121 force、121 complete、0 error | 39.857s | 32.374s |
| `c8-replay2-r8` | valid | 121 / 120 / 120 | 121 force、121 complete、0 error | 39.572s | 32.413s |

两轮每条 audit 都有唯一 request ID，且
`sampled_token_count = forced_token_count + 1`。两轮均以 `fixed-work-complete`
结束，无失败 task terminal；framework polling 尚未来得及发布的第八个 terminal 不被伪造，
因为 fixed LLM/tool slots 已全部闭合。

对比结果：

- busy-span relative spread：0.1204%；
- makespan relative spread：0.7176%；
- replay2 - replay1：-0.285s（-0.715%）；
- per-lane wallclock 最大绝对差：0.9372s；
- 两份 report 均 valid，内部未归因 gap 为 0；startup gap 保留为诊断而不当作漏采；
- wallclock 与 full causal-lane timeline 的 PNG/SVG 均已生成并目检。

### E08 — 最终回归与结论（完成）

回归：

- minireplayer：`ruff check .` clean，`pytest -q` 为 `162 passed`；
- OpenCode plugin：Bun build 成功；
- CORAL framework workspace：`10 passed`；
- multiagent sweep/queue helper：`18 passed`；
- `sweep.sh` 与测试 wrapper：`bash -n` clean；
- 三个工作树的 `git diff --check` clean。
- 专用 4×TP2 fleet 已通过 `vllm-down` 停止；8000--8003 无监听，8 张 A100 均回到
  4 MiB / 0% utilization。

最终参数建议：

- `restart_exited=false`。它使 finite one-shot 每个 team 只启动初始四个 invocation，
  避免最后一波 restart 刚启动就被 global=8 截断，以及相同参数 4/6 model-turn 的调度漂移。
- `global_turns=8` 继续保留在 workload identity 和 manager 配置中，满足实验约束；在
  restart=false 下它是上界而非必须耗尽的配额。
- `agent_turns=3` 控制每次 OpenCode invocation 的 `step_finish` 硬上限，与 global invocation
  budget 完全独立。
- `duration=60s` 足以得到 C1/C8 闭合 source，并比 MiniSWE/Owl 正式窗口更短。

结果根目录为 `results/coral-autoresearch-20260723/`。正式交付以 r5 C1 和 r8 C8 为准；
r3--r7/r9 等失败目录作为 autoresearch 证据保留，不覆盖、不冒充有效结果。

### E09 — 2026-07-24 默认参数与 task 部分完成语义修正（进行中）

当前调试目录为 `results/coral-defaults-20260724/`。C1 `c1-source-r4` 在 native team
启动第 10 个 invocation 后按现有 manager 语义执行 `max_total_turns` cutoff；此行为暂不改。
该样本暴露两个此前没有落实完整的边界：

1. 两个 parent agent 的 LLM 已经完整结束并吐出 OpenCode `task` tool-call ID，但
   minireplayer 没有捕获 dispatch。OpenCode 数据库中的两个 task part 为 `running`，
   manager cutoff 时父调用没有闭合。LLM 完成独立于后续 dispatch，因此不能拒绝或删除
   该 LLM；dispatch 未入场意味着没有 tool。
2. `task` 不是原子 tool，而是包含 child session 的复合 subagent 调用。如果父 task 被杀，
   source 中已经完成的 child LLM、dispatch、tool 都是 fixed work，必须保存；只有仍未完成
   的 child work 和父 result 属于 cutoff。旧因果闭包会因 parent span 未闭合而整支删除，
   这是错误语义。

修正后的记录/replay contract：

- 对“LLM 已闭合、dispatch 未开始”生成显式零时长 `pre_dispatch` marker；bundle 保留 LLM、
  不生成 tool，replay 若走到该 dispatch 则在 native entry 前阻断。
- 对 cutoff 时 active 的 `task` dispatch/tool 标记
  `enter-and-preserve-descendants`。因果闭包把该 cutoff parent span 视为有效 owner，保留
  所有闭合 child work。Replay 允许父 task 入场以创建 child session 并消费这些固定 slots，
  但永久扣住父 completion；未闭合 child tail 不消费，也不触发额外 refill。
- OpenCode plugin 除 pending/running part event 外，在 `session.created` 时从 OpenCode
  message store 反查 running task，补齐绕过 `tool.execute.before` 的内置 task parent，
  并在 child 首次 LLM 前绑定 parent span。

定向回归现为 60 passed。用 r4 原始 stage 离线重建的
`c1-bundle-r4-fixed` 已通过 bundle validation；两条异常调用均准确呈现为：
`completed LLM + pre_dispatch(block-before-entry) + no tool`，而不是伪造 completion 或
丢弃 LLM。确定性的 integrated test 另行覆盖了“父 task 在 cutoff 时 open、闭合 child
LLM/dispatch/tool 被保留、下一条未闭合 child LLM 留作 tail、replay 允许父入场并消费
闭合 child slot”的完整形态。

`c1-source-r6` 进一步暴露了两个 cutoff 时钟混在图里的问题。source gate 为
`1784871038926806015ns`，CORAL manager 在
`1784871159266759489ns` 因 `max_total_turns` 杀掉整组 agent，故真实 team cutoff 是
gate 后精确 `120.339953474s`；录制 sample window 则继续到约 180s。旧 Step3 错把
agent-3 的未完成 LLM 延长到 sample window 末尾，并且其余三条当时没有 active LLM/tool
的 lane 没有任何终止证据，视觉上像是原因不明的空白。

现已把两类证据分开：

- cutoff tail 的结束时间取实际 `elapsed_ns`，并由更早的 stream interruption 或 manager
  lane cutoff 上界收紧，不再默认延长到 source terminal；
- manager terminal 保存 epoch cutoff 与 reason，并投影为同一 team 下四条 agent lane
  的独立 `lane_termination` 事件；PNG 在精确位置画红色短标记，之后用浅色斜线明确标注
  “team terminated; no work after”。该区域不是 LLM/tool，不计入 busy/coverage。

r6 重建后的四条 lane 都在 `120.339953474s` 标记终止；agent-3 的 LLM 从
`114.776534783s` 截至该点，保留 `5.563418691s`，而不是延伸到 180s。下一步使用含原生
manager terminal 字段的新 source 做全量回归；通过后在全新的 CORAL 结果根目录完成 C1
record + 两次 full replay，再决定是否进入 C8。

同一张旧图还把不同 OpenCode invocation 压平到了单条 actor lane。原生 Step3 会从
`agent-N.<sequence>.log` 恢复 invocation 间的 restart control window；r6 实际有 6 次，
全部由 `heartbeat:reflect` 触发（agent-1/4 各 1 次，agent-2/3 各 2 次），四个初始
invocation 加起来正好是 manager 的 global=10。

Replay 对 restart 采用“原生执行、严格校验边界”，而不是把 restart 伪造成 tool：

- CORAL manager 仍负责自然退出、resume/restart；
- 每个 agent 的每次 runtime start 获得单调 `invocation_index`，OpenCode root/child
  session identity 都带 `actor/invocation-N` 前缀；因此 source invocation N+1 的
  LLM/tool 不可能被 replay invocation N 顺序吞掉；
- Step3 的 `restart_events.jsonl` 与青色/紫色控制条单独显示 heartbeat/ordinary restart，
  不计入 LLM/tool busy；
- cutoff 前只启动但没有闭合 work 的最后 invocation 仅保留 lifecycle/cutoff evidence，
  不伪造 LLM 或 tool。

后续审计发现，前述 r4 离线 bundle 和基于它完成的 r9 replay 仍然不成立，不能计入正式
结果。r4 的原始 Step3 ledger 实际包含 agent-3 子 session 的 8 条闭合 LLM 和 7 条闭合
tool；旧 closure 却把“缺失的 parent `task` dispatch”一律解释成
`pre_dispatch(block-before-entry)`，随之删除了整个已完成 child graph。bundle 内只剩
70 LLM / 80 tool，而 closure 前是 78 LLM / 87 tool。r9 只是忠实回放了这个已经缩水的
bundle，不能证明部分完成语义正确。

现在把两种形态严格分开：

- LLM 吐出 tool-call、dispatch 未入场且没有 child session：保留闭合 LLM，生成
  `pre_dispatch(block-before-entry)` 证据，不生成 tool；
- 缺失的是内置 `task` parent，但已观察到带 parent span 的 child session：补出
  `task` dispatch/cutoff owner，使用 `enter-and-preserve-descendants`，把 task 结束延伸到
  team cutoff；所有已闭合 child LLM/tool 保留，未闭合 parent result 和 child tail 不伪造。

原 Step3 对 subagent 的 lane 归属是正确的：每个 CORAL agent 保持一条 lane，child
session 不是新的 CORAL 并发 lane，其 LLM/tool 合并回 owning agent 行。一次把 child
拆成独立可视 lane 的调试修改属于误改，已撤回；raw row 继续保存完整
`actor/invocation-N/root-N/child-N` session 与 `work_scope=subagent`，parent `task` 只作为
同一行上的虚线 composite scope。

对 formal r4 的 OpenCode store 做第二次审计后，撤回“47--165s 是空段”的判断。child
最后一条 LLM 在约 47s 生成 `call_fb6685e6ea6345a3ad23f2c0` 后，OpenCode 已创建
`bash` running part、完成 `./solution` permission/path 解析，并实际留下
42,784,787-byte tool-output（从 `walk 1` 持续输出到 team cutoff）。旧 recorder 却把
同一 call 合成为 `pre_dispatch, elapsed_ns=0`，所以漏掉的不是 idle，而是一条约
47--165s 的 active truncated bash tool。这是 recorder/instrumentation capture bug，
不是 Step3 的 subagent lane 处理问题，也不是 native CORAL/OpenCode 没有 dispatch。

当前 plugin 已从每个 ordinary/tool `message.part.updated` running event 幂等创建
dispatch+tool reservation，并与 `tool.execute.before` 共享 start promise；因此即使
`tool.execute.after` 永远不到达，source freeze 也能保存 active tool tail。新增可执行
Bun 回归验证 ordinary running bash 会成为 cutoff reservation source。r4 图仅依据其
OpenCode DB、log 和 tool-output 离线恢复该红色 truncated span，不将旧 source/bundle
冒充为新 contract 下的有效录制。

进一步审计 CORAL manager 后确认，grader daemon 默认有四个 worker，四个 agent 各有独立
grader；“同一个 3 秒 manager poll tick 看见多个 grader 完成”是 native control-plane
现象，不是 recorded work 的执行顺序。Replay 不再根据当次运行中 grader 的到达批次重新
推导 heartbeat restart，也不采用逐个 drip grader 的调度补丁。录制 bundle 现在保存每个
固定 invocation 边界的 control record：agent、invocation index、原始 prompt/source 和
触发它的 grader attempt。Replay 先消费该 invocation 的全部固定 LLM/tool prefix，并确认
对应 grader 已记录，再按录制的 prompt 启动下一 invocation；live heartbeat restart 被
抑制。这样 restart 是录制下来的控制语义，而不是由 replay 时线程/轮询时序重新创造。

修正后全量 minireplayer 回归为 `205 passed`，`ruff check .` clean。新的正式目录为
`results/coral-formal-20260724/`；必须重新录制 C1 并完成两次 replay，旧 r4/r9 只保留为
诊断证据。C1 通过前不启动 C8。

### E10 — r10 三条 active cutoff tool 的原生存储恢复

再次按 lane 审计 formal r10 后，撤回 agent-1/2/3 cutoff 前空白的判断。三个 OpenCode
SQLite store 在 team cutoff 后都保留了唯一的 `running` bash part，且命令均为
`coral eval`：

- agent-1：`83.841--100.345s`；
- agent-2：`94.807--100.345s`；
- agent-3：`97.130--100.345s`。

旧 `cutoff-tails.json` 虽然包含三个 call，却把它们都降成
`kind=dispatch, elapsed_ns=0`，Step3 因而漏掉了三条 active tool tail。agent-4 在
`95.378--100.342s` 只有 heartbeat restart control，cutoff 时没有 active LLM/tool，
所以不应伪造红色 work span。

`evidence/rebuild_r10_timeline.py` 现在逐一只读三个原生 DB，要求每个 agent 恰有一个
`running coral eval`，使用 DB 的 native start timestamp 和 manager 的精确
`terminated_at_epoch_ns` 重建同一 agent lane 上的红色 truncated bash。旧漏画版本保存在
`timeline-missing-agent1-tail-retracted.png`，当前 `timeline.png` 已目检包含三条 tail。

同时发现 OpenCode 会把插件模块的每个 export 都当作 plugin factory。此前为测试导出的
`eventStartArguments` 等 helper 会被传入 plugin context 并触发
`part.state.status` load error。helper 已改为模块内部函数，插件只保留一个 default
export，并对 part state 做空值保护；Bun/Step3 定向回归为 `11 passed`。

按最新实验指令，`c1-source-r13` 录制结束后只做 bundle validation，不再 replay/C8。
bundle `adc8e45387252bb4` valid，包含 62 LLM、56 dispatch、56 tool、5 grader 和
10 artifact。随后已执行 `vllm-down`：8000--8003 无监听，8 张 GPU 均为
4 MiB / 0% utilization。
