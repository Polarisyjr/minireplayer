# Sweep-Derived Agent Replayer

## 1. 目标

本工具记录四个 agent framework （miniswe,coral,owl,trae） 的工作轨迹，使其能在 fresh vLLM 和 fresh writable workspace 中 replay：LLM 调用、committed token、顶层 tool
调用及返回给 framework 的 observation。

vLLM 和 tool 都必须真实执行--但是要给一个tool-only的replay接口允许只重放tool；禁止用 sleep、stub 或录制耗时模拟负载。

## 2. Workload 真源

支持串行或并行。可以使用multiagent/scripts/<framework>/sweep接口启动 （*注意这个不需要挂载step1和step2的monitor*）

所有 framework 使用同一个 `refill` workload 参数：`true` 时某个任务结束后按 seeded
order 补入下一个任务，`false` 时只运行 seeded order 的前 C 个任务且并发度随完成自然下降。
该参数必须进入 bundle workload identity，record 与 replay 必须一致。

coral是跑frointer-cs-algo这个数据集

CORAL 的 C 不能解释为扁平 agent 数。它单独维护 C 个 team/task slot；每个 slot 同时拥有
固定 4 条 OpenCode agent 子 lane，因此 C8 的目标窗口是 8 个 team、32 条 agent lane。
CORAL refill 的原子单位是整个 team：只有某个 task 的 4-agent 进程组全部结束并完成清理后，
该 team slot 才能进入下一代 task；禁止按单 agent 补位。录制必须保存
`team_slot / slot_generation / parent team / agent-1..4` 层级，bundle 必须拒绝不完整 team。

如有必要可以复用已有代码简单wrap新的启动器。不得另造 task list、queue、prompt、scheduler、completion condition 或 workload override。

如果使用sweep规范，C1 是 seeded order 的第一个任务。C8 是同一 order 的前八个任务按 sweep 语义并发，不是八条 C1 的拼接。任务在 cutoff 时未完成是正常实验结果，不得换任务或要求自然完成。

开发 smoke 可以通过同一 sweep 的原生 `--duration` 参数缩短窗口；正式默认是先 warmup 15 秒，
再录制 180 秒。CORAL 默认每个 invocation 最多 100 个 model turn、team 全局最多 10 个
invocation，并允许 exited agent 重启；这三个参数都属于 workload identity。

Sweep 脚本和 framework checkout 必须已经具备本地模型、dataset、image 与依赖。
Replayer 不下载、不替换，也不自造 workload。

## 3. Bundle

Bundle 保存：

- sweep 命令、checkout revision、解析后的任务、并发语义和 serving topology；
- source 窗口内已经闭合的 LLM、dispatch、tool 和 grader causal slots；
- 原始 LLM request、prompt token、committed output token 和 response/chunks；
- 顶层 tool 参数、实际 native outcome，以及返回给 framework 的 observation；
- source cutoff 时仍 active 的 operation，保存为 `truncated` 边界证据，包括原始请求、
  native implementation 和从 operation start 到 cutoff 的已执行时长；
- 已经闭合的 LLM 独立成立：它吐出 tool-call 后尚未进入 dispatch 就被杀时，保留 LLM，
  记录零时长 `pre_dispatch` 边界，并且不存在 tool；
- CORAL 内置 `task` 是复合 subagent 父调用。父 task 在 cutoff 时未完成，不得删除它下面
  已经闭合的 child LLM/dispatch/tool；未完成的 child 只作为 tail。
- 使用类似step3的渲染方法补出timeline

Bundle 的物理布局按 actor causal lane 分片：每个 lane 独立保存其 LLM、dispatch、tool、
grader、artifact、span 和 cutoff tail，根目录 manifest 只保存 workload、actor/lane 索引及
计数。内存加载时可以重建全局诊断视图，但 replay 的领取顺序只在 lane 内成立，不以跨 lane
完成顺序构造全局 barrier。

## 4. LLM Replay

每个 LLM slot 由 actor、session、role、causal parent 和 actor-local ordinal 领取。
Replay 将 bundle 中的原始 request 发给 vLLM，让引擎真实完成 prefill、logits 和 sampling，
在 commitment boundary 提交录制 token，并把录制 response/chunks 交给 framework。

Hard gate：API、target、request 的结构化工作投影、prompt token、sample step、committed
token、causal parent 和 slot 消费。自然 sampled candidate、response ID、时间戳、JSON key
order 和日志文本不是 hard gate。

## 5. Tool Replay

Tool 请求必须由录制 LLM output 经 framework 原生 parser/dispatcher 产生。每个 slot：

1. 真实进入 native implementation；
2. 记录实际 status、耗时、CPU与实际结果；
3. 向 framework 返回录制 observation，固定后续控制流。

Hard gate 只包含顶层 slot 的数量/顺序/身份和 native implementation 确实进入。实际
outcome class、stdout 文本、DOM、远端响应、临时路径、PID、child process 细节、文件树
和实际 result 内容只作诊断。不得为这些噪声添加逐个正则补丁。

Source 中由 native implementation 直接抛出的异常也是录制 observation。Replay 仍先执行
同一 native tool，再向 framework 恢复 source 的异常类型与消息，以固定后续控制流；实际
native run 是否产生同类异常只进入诊断证据，不单独使 replay 无效。

因此本工具固定的是 controller-visible trajectory 与顶层 native tool invocation，不声称
不同 replay 的 tool 内部 CPU 指令、浏览器请求、网络工作或微观状态逐次完全相同。
Owl 的 Chromium 必须真实启动并执行 `browse_url`，但 browser 内部只作诊断。

## 6. 并发与窗口

Record 热路径中每个 actor/session 直接追加自己的 start/complete 事件，不经过共享的
dispatch `/start` 服务。LLM proxy 与 replay boundary 分属独立 event loop；一个 lane 的同步
请求、落盘或工具耗时不得阻塞其他 lane。录制结束后再离线物化和校验 canonical ledger。

Source 完整执行 sweep，`sample_end` 是录制边界。边界时间由 sweep 事件冻结。已经完成的
task 全部保留，framework 按原生 sweep 继续 refill。边界时仍 active 的 LLM、dispatch、tool
或 grader 不伪造成完成，也不整段丢弃，而是保存为 `cutoff_truncated` 边界证据。

Replay 消费 cutoff 前已经闭合的因果前缀。普通 tail 和 `pre_dispatch` tail 都在 LLM 或
native implementation 入场前停止。唯一的入场例外是仍 active 的复合 CORAL `task`：
replay 必须进入该父 task，重建 child session，消费 source 中已经闭合的 child
LLM/dispatch/tool，然后永久扣住父 task completion；source 未完成的 child tail 不领取，
父 result 不伪造。这既保留成功执行的子工作，也不会允许 source 中不存在的后续 refill。

`llm-only` 是 `tool-only` 的镜像模式：LLM lane 与 `full` 完全一致地打真实 vLLM，
而每个 tool 不进入 native implementation，只按 source 观察到的时长占住该 lane，然后返回
bundle 里录制的 observation。占时长是必需的——立即返回会改变 engine 看到的请求到达模式、
batching 和 cache 压力，测的就是另一个 workload。该模式下 tool ledger 记 `native_execution:
false`，它不满足"每个顶层 tool 真实进入 native implementation"，因此不能当作 full replay
的有效性结论使用。当前只对 mini-swe 实现；其他 framework 直接拒绝该模式。

Replay 使用 sweep 的相同 launcher、task order、并发和 refill。它在全部闭合 slots 完成后
结束；普通 `truncated` 尾段只说明 source 在哪里被切断。复合 task tail 只领取父入场与
source 已观察到的 elapsed window，不产生 parent completion。录制后检查 timeline，不应出现
明显无法归因的空白。如果 sweep 自身先到 cutoff、出现未知 slot、缺少 slot 或控制流分叉，
run 无效。

CORAL 的并发遥测和 timeline 以 team slot 分组：每代 team 下恰好四条 agent lane。图中的
32 条 agent lane 不能被误报为 C32；它们是 C8 的 `8 × 4` 孙 lane。开启 refill 时，新一代
仍画在其 team slot 分组内，并在前一代四条 lane 全部退出后开始。CORAL manager 可能先于
180s sample window 因 global turn budget 杀掉 team；这个 team cutoff 必须以 manager
发布的 epoch 时间投影到同组全部四条 lane。未完成 LLM/tool tail 只画到这个更早的时间，
无 active work 的 lane 也必须显示 termination marker；marker 之后是“不可能再有 work”的
终止区，不是 cutoff operation，也不计入 busy/coverage。

同一 team generation 内，单条 agent lane 可以经历多次 OpenCode invocation。Restart 是
原生控制流边界，不是 replayable tool：Step3 从 CORAL invocation log 单独画 restart
window，且不计入 work busy。Record 给每次 runtime start 分配单调 invocation index，并把
它写入 root/child session identity；full replay 仍由 CORAL manager 原生 restart，但只有
相同 invocation generation 才能消费对应的 LLM/tool。这样既保留 restart/resume 造成的
真实间隔，也禁止把多个进程的事件静默展平成一个持续会话。

Record 和 full replay 在 actor gate 前执行同一套 serving warmup。顺序固定为
`reset prefix/KV -> warmup -> reset prefix/KV -> open gate`，从而排除首个 measured LLM 的
kernel/sampler 冷启动，同时不把 warmup prompt 留在测量 cache 中。Tool-only replay 不访问 vLLM。


每个 run 使用独立目录、fresh vLLM、独立 tmux socket 和 writable state。清理只能作用于本 run 拥有的进程、容器和 cgroup。

## 7. 准入与指标

每个 framework/C1 或 C8 bundle 用一次 source recording 和一次 fresh replay 准入。
Replay 必须满足：

- LLM/dispatch/tool/grader slots 恰好消费；
- prompt 与 committed token 一致；
- 每个顶层 tool 真实进入 native implementation；
- 向 framework 回灌的 observation 与 bundle 一致；
- 普通 cutoff / `pre_dispatch` 尾段未进入 LLM 或 native implementation；
- 复合 task tail 只允许父 task 入场，其闭合 child slots 恰好消费、父 result 未返回，
  未完成 child tail 未领取，且没有产生额外 refill；
- 无未知调用、instrumentation failure、vLLM 泄漏或 run-owned 资源残留。

报告 replay 的 makespan、CPU、GPU、I/O、网络、operation timing 和实际 tool outcome。
Recorder 的性能不作为 baseline；应多次replayer检验是否性能一致。注意关于网络的、编译error的等因素可能会导致出现较大偏移，除此之外多次重放应该比较稳定。如有不稳定的需要你（也就是agent）智能检查归因，这个没法提前写归因脚本。tool 内部诊断漂移不使 run 无效，但必须可见。准入不使用事后挑选的性能阈值，也不因较慢而选择性重跑。

## 8. 变更与测试


最低测试覆盖：四个 framework 的 C1/C8 LLM commitment；顶层 tool 真实执行并回灌 observation；fresh tool-only replay 输出完整性能指标，多次replay后除了本身就有抖动的工具外性能大致一致；并发 slot 不按完成顺序猜测；teardown 后无本 run 残留。

之后继续 完整replay （把vllm的部分加进来形成完整lane）；如没问题再加C32
