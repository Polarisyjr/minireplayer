# CORAL autoresearch checklist

持续更新；详细证据与每次失败现场写入 `CORAL-AUTORESEARCH.md`。完成项只在有可复核产物后
勾选，不用后续成功覆盖早期失败。

## 约束

- [x] 建立持续 goal。
- [x] 固定正式实验的 shared/global turn budget 为 8。
- [x] duration 使用短窗口；先 smoke，再决定正式短窗口的精确秒数。
- [x] 优先修改 minireplayer 配置、launcher 或 instrumentation；避免 CORAL framework
  业务逻辑改动。
- [x] 记录实验前两个工作树的已有改动，避免覆盖无关用户工作。

## Turn / exit smoke

- [x] 静态确认 `agents.max_turns` 的计数单位与退出行为。
- [x] 静态确认 `agents.max_total_turns=8` 的计数单位、容量判定和终止时点。
- [x] 静态确认 `agents.restart_exited=false` 的有限运行语义。
- [x] 静态确认 `agents.restart_exited=true` 与 global budget=8 的组合语义。
- [x] 合成 smoke A：低 agent-turn、restart=false，核对 invocation、restart、terminal 和截止。
- [x] 合成 smoke B：低 agent-turn、restart=true、global=8，核对 invocation、restart、terminal
  和截止。
- [x] 合成 smoke C：改变 agent-turn、保持 global=8，分离“单次会话 model turn”与“团队
  invocation turn”。
- [x] 真实 smoke A/B/B2/C：在 OpenCode + vLLM 上复核合成结果。
- [x] 根据证据选定正式 `restart_exited=false`、agent-turn=3、global-turn=8。

## CORAL record / replay

- [x] 建立独立 `results/coral-autoresearch-20260723/`，保存 config、evidence 和失败 run。
- [x] dry-run 验证 seed=42、固定任务顺序及 agent/global/restart 参数透传。
- [x] C1 短窗口 record 有效并封包：bundle `b35290e022c7e000`，
  16 LLM / 12 dispatch / 12 tool，0 cutoff tail。
- [x] C1 两次 clean full replay；每轮 16 个 forced request 及全部 ledger 精确闭合。
- [x] 生成并目检 C1 report 与 comparison plot。
- [x] C8 短窗口 record 有效并封包：bundle `947522f31c103703`，
  40 actors、121 LLM / 120 dispatch / 120 tool，0 cutoff tail。
- [x] C8 两次 clean full replay；每轮 121 个 forced request 及全部 ledger 精确闭合。
- [x] 生成并目检 C8 report 与 comparison plot。
- [x] 增加 CORAL provider tool-call → dispatch/cutoff evidence 完整性门禁；不完整
  `task`/subagent 样本不得封包。

## 最小改动与回归

- [x] framework 仅保留一个通用正确性修复：并发 run 原子更新 `latest` symlink。
- [x] CORAL workspace 并发测试：10 passed。
- [x] sweep 参数与 queue 测试：18 passed；shell syntax clean。
- [x] OpenCode plugin Bun build 通过。
- [x] minireplayer `ruff check .` clean，`pytest -q` 162 passed。
- [x] 更新本文和 `CORAL-AUTORESEARCH.md` 的最终结论。
- [x] 停止本轮 4×TP2 vLLM fleet；8000--8003 无监听，8 张 GPU 回到 4 MiB。
