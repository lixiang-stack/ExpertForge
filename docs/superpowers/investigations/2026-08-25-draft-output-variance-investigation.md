# Gate-First 后 se-1/se-3 Token 增幅调查 — 单次抽样方差

Date: 2026-08-25

Status: Concluded

References:

- [2026-08-25-critique-gate-first-design.md](../specs/2026-08-25-critique-gate-first-design.md)
- PR #23（gate-first 实现）：https://github.com/lixiang-stack/ExpertForge/pull/23
- 实测结果：`evaluation/results/2026-08-25-compare.json`；对照 `evaluation/results/2026-08-24-compare.json`

---

## 1. 现象

Gate-first 合并后实测 `compare --ids se-129 se-1 se-2 se-3 se-4`：

```
se-1   base 4883 → orch 21910  (+348.7%)   Q 5.0 → 5.0
se-2   base 21307 → orch 22602 (+6.1%)
se-3   base 15758 → orch 28839 (+83.0%)
se-4   base 10834 → orch 12845 (+18.6%)
overall +63.31%，质量增益 0
```

se-1/se-3 增幅远超预期（spec 预估通过路径 ≈ +20–50%）。

## 2. 取证与排除

**2.1 调用次数证明全部走了早退路径。** 结果文件中 4 个 case 的 orch `llm_calls` 均为 **2**（draft + 内部 judge）；critique 流水线至少需 7 次调用。gate-first 行为正确，没有 case 进入 critics/revise，管线开销只剩一次很小的 judge 调用。

**2.2 增幅来自 draft 单次调用的 output tokens：**

| case | base_out | orch draft_out | 倍率 |
|------|----------|----------------|------|
| se-1 | 4,339 | 17,015 | 3.9× |
| se-2 | 20,763 | 19,524 | 0.94× |
| se-3 | 15,210 | 25,249 | 1.66× |
| se-4 | 10,345 | **8,329** | 0.81× |

**2.3 结构性差异已排除。** `_draft` 与基线 `Strategy.process` 消息结构逐字相同（`system=context`、`user=question`，均不传 temperature/thinking/max_tokens 覆盖；orchestrator.py `_draft` vs strategy.py `process/build_messages`）。二者是同一 prompt、同一采样参数的独立抽样。

**2.4 同题跨运行对比坐证方差。** 同一问题在两次运行间的输出长度：

| case | run1 base total(08-24) | run2 base_out(08-25) | run2 orch draft_out |
|------|------------------------|----------------------|---------------------|
| se-1 | 16,907 | 4,339 | 17,015 |
| se-3 | 21,101 | 15,210 | 25,249 |
| se-4 | 6,898 | 10,345 | 8,329 |

同题 base 自身相差 4 倍（se-1），se-4 甚至出现 orch draft 比 base 更短。

## 3. 根因

**温度 0.3 + thinking 开启 + 无 `max_tokens` 上限时，单次补全的输出长度高度发散**——这是该采样配置下的预期属性，不是编排缺陷，也不是模型缺陷。se-1 的 +348% 是「base 侧恰好抽到短样本（4.3k）、orch 侧恰好抽到长样本（17k）」的配对噪声；上一轮运行中长样本出现在 base 侧（16.9k）。旧 critique 拓扑曾把抽到的长度无条件乘 N 倍放大；gate-first 移除了这种乘法，剩余波动即基线固有。

## 4. 测量学结论

n=1 配对 compare 在此方差下不具备单例结论力："+348%" 与 "-19%" 都可能是噪声；此前各轮 compare 的 per-case Tok%（含 ≤300% 成本门判定）一直带有同等量级的误差棒。质量侧 5.0 饱和问题另见 [judge variance investigation](2026-08-22-judge-variance-investigation.md)。

## 5. 可选后续（按代价从小到大）

1. **测量侧**：compare 对每 case 跑 n≥3 取中位数——消除配对噪声，不改生产行为（多 run 统计本就是本文件前身遗留项）。
2. **测量侧**：compare 两侧统一 `temperature=0`——降低但不消除思考长度方差。
3. **生产侧**：draft 加 `max_tokens` 硬上限（参数已存在，P2 措施 C 引入）——压住尾部，但打破与 baseline 的「同等推理预算」约定，截断需要降级路径（可复用 `LLMError → 返回` 模板），且只约束 orch 侧会重新引入不对称。

## 6. 结论

Gate-first 达成设计目标（编排开销 ≈ 一次 judge call）；残余成本波动为单次抽样方差，属测量方法问题而非功能问题。建议合并 PR #23 后以方案 1（n≥3 中位数）重测再评估是否需要方案 3。
