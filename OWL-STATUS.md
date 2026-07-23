# owl 接入现状

本轮工作的交接说明。规范以 `design.md` 为准，详细的 bug 记录在 `PROGRESS.md`。

## 一句话

C1 已完全跑通并经三方独立核对；**C8 尚未拿到两次干净重放**，卡在一个已定位、待修的
bug 17 上。

## 状态

| 项 | 状态 |
| --- | --- |
| owl VL endpoint 换 Qwen3-VL-8B（131072 上下文） | ✅ fleet 已起，forced-capable |
| C1 fire-once：录制 + 两次重放 | ✅ 全 valid，gap 0，抖动 1.1% |
| C1 三方交叉核对（我们 / owl 自身 lane / vLLM 引擎 span） | ✅ 全部一致 |
| L1：owl model-free primitive 变成一等 tool slot | ⚠️ 已实现，**未经 C8 验证** |
| C8 fire-once：两次干净重放 | ❌ 卡在 bug 17 |
| L2：截图/视频存进 bundle 的 artifact lane | ⬜ 未开始 |
| LLM 上游错误记成 observation | ⬜ 未开始 |
| 换种子复验 | ⬜ 未开始 |

测试 93 通过，`ruff` 干净。

## C1 的三方核对结果

三个互相独立的观测源，全部吻合：

```
tools   minireplay=14  owl 自身 step3 lane=14   顺序完全一致
        逐个时长差 −0.1 ~ +2.1ms，合计 +14.2ms  ← 这就是本 harness 的包裹开销
llm     minireplay=24  vLLM 引擎 span=24
        引擎侧按 endpoint  {answerer:1, coord:6, reasoning:14, task:3}
        账本侧按 role      {answerer:1, coordinator:6, reasoning:14, task:3}
两次重放 engine span 均为 0 —— tool-only 确实没碰 vLLM（原来是假设，现在是被检验的事实）
```

## 本轮发现的 bug

| # | 问题 | 状态 |
| --- | --- | --- |
| 13 | launcher 把 `-n` 钉死成并发数，refill 只能重跑刚做完的任务 | ✅ 已修，统一 `refill` 进 workload identity |
| 14 | owl 串行路径的 gate 只能在 replay 跑，录制必挂 | ✅ 已修 |
| 15 | 代理 1 MiB 请求体上限，C8 的截图请求被自己人拒了 | ✅ 已修（512 MiB） |
| 16 | 截断尾段被排到队尾，而不是它真正发起的位置 | ✅ 已修，2 个测试双向钉住 |
| 17 | **嵌套的 tool 尾段没有随父 dispatch 一起冻结** | ❌ 待修 ← 下一步 |

另有一个 design 内部冲突（§4 硬门禁 vs §5「browser 内部只作诊断」），已按决定处理：
内联 `data:` 负载只保留 media_type，丢掉字节长度。L2 完成后这条让步应当删除。

## 下一步：修 bug 17

证据（C8 录制的尾段普查）：

```
kind 分布 : {'dispatch': 8, 'tool': 2}
name 分布 : {'browse_url': 9, 'browser_action': 1}
```

每个在飞的 `browse_url` 都留下了 dispatch 尾段，却几乎都没留下嵌套在里面的 tool 尾段。
重放于是：认领 LLM 响应 ✓ → 认领 dispatch 尾段 ✓ → 里层 tool 找不到槽位 → 判漂移。

修法方向在 `boundary.freeze_source_cutoff`：一个 dispatch 尾段若含嵌套的 tool 预约，
两者必须一起冻结。修完重跑 C8。

## 顺序建议

**C8 拿到两次干净重放之前，不要动 L2。** artifact lane 会再改一次 bundle 结构，叠在
未验证的基础上，下次失败就会有两个可疑源头。

## 复现命令

```bash
SP=<scratchpad>
# 起 fleet（6×qwen3-coder-30b + 1×qwen3-vl-8b）
.venv/bin/python -m minireplay vllm-up --config $SP/owl-c1-fo.json
# 一个 scope 全流程：录制 + 两次重放 + report，并抓 step3 两条 lane
bash $SP/run_scope.sh c1-fo $SP/owl-c1-fo.json $SP/lanes/step3-tools-c1.jsonl
# 三方核对
.venv/bin/python $SP/crosscheck.py $SP/c1-fo-bundle \
    $SP/lanes/c1-fo-src.tools.jsonl $SP/lanes/c1-fo-src.spans.jsonl
```

step3 的两条独立观测：owl 自身的 tool lane 只需在 `config.env` 里设 `STEP3_TOOL_LOG`；
LLM lane 靠 step3 的 OTLP receiver 容器（fleet 默认就在导出 span）。

## 对 `multiagent` 的改动（均为纯增量）

- `scripts/lib/sweep_common.sh`：`SWEEP_STEP=none`、`SWEEP_SKIP_VLLM=1`
- `serving/scripts/start_vllm_multi.sh`：`VLLM_EXTRA_ENV`、`VLLM_EXTRA_MOUNTS`
- `scripts/owl/sweep.sh`：`SWEEP_WARMUP` 改为可被环境覆盖（省掉每轮 30s 空转）
- `serving/configs/qwen3-vl-8b.yaml`（新增）、`scripts/owl/start_vllm.sh` 指向它

## 已知坑

- 起 fleet 时若换了模型名，`start_vllm_multi.sh` 按**容器名**停旧容器，旧名的容器会继续
  占着端口；失败的 `Created` 容器同样占端口。换模型后要先 `docker rm -f` 旧的。
- 这台机器 96 核，`framework_cgroup.cpu_seconds` 会远大于 `busy_span_seconds`，主要来自
  tool 子进程里 `import numpy, pandas` 时 BLAS 按核起线程（一次 7.3s CPU / 0.42s 墙钟）。
  真正可比的是 user CPU。详见 `PROGRESS.md`。
