# 排查记录：critique 拓扑复测 se-129 质量仍低于 baseline

日期：2026-08-22 ｜ 分支：critique-topology ｜ 关联：PR #22、[重构 spec](2026-08-21-refactor-orchestration.md)

## 1. 问题

critique 拓扑上线后运行 `uv run python -m agent.evaluation compare --ids se-129`：

```
se-129    Base Q=5.00   Orch Q=4.83   Gain=-0.17 (-3.40%)   Base Tok=6121   Orch Tok=41102 (+571%)
```

启用 orchestration 后质量仍低于 baseline，token 多消耗 571%。

## 2. 排查过程

### 2.1 执行路径核查（trace：.observability/trace-2026-08-22.jsonl）

管线按设计执行，共 8 次 LLM 调用：

```
draft(in489/out8980, thinking开) → planner(4视角) → critics×4(合并22个issue)
→ reviser(out5015) → evaluator(全5分, passed) → 返回
```

### 2.2 答案人工核查

- 旧 map-reduce 的四类矛盾（部署模式 / RPO 分层 / 事务方案 / 架构风格）**全部消失**
- 假设显式标注（"设计基准"、"工程目标非硬性保证"）✓
- 唯一候选瑕疵：§3.3 复式记账示例两行 INSERT 共享 `biz_flow_no`，而注释备选约束 `UNIQUE(biz_flow_no)` 会使第二行插入失败——自相矛盾型硬伤（收尾句只背书了坏的选项）

### 2.3 关键证据：同一答案被同一 judge 模型评了两次

| 评分方 | 模型 | 对象 | 结果 |
|---|---|---|---|
| 管线内 evaluator | deepseek-v4-pro | revise 后终稿 | 全 5 分 |
| compare 外部 judge | deepseek-v4-pro | **完全相同的文本** | correctness 4 |

同答案、同模型、分数差 1 个维度 → 怀疑 judge 采样方差。

### 2.4 实验 1：重复评分（judge temperature 对比）

同一对保存的答案（base22 / orch22），各重复评分 3 次：

| temp | base | orch |
|---|---|---|
| 0.3 | 5.00 / 5.00 / 5.00 | 5.00 / **4.83** / 5.00 |
| 0.0 | 5.00 / 5.00 / 5.00 | **5.00 / 5.00 / 5.00** |

- temp=0.3 下 orch 的波动与 compare 报告的 -0.17 完全同型 → **假信号被直接重现**
- temp=0.0 下完全稳定 → **假信号被消除**

### 2.5 实验 2：rationale 探针（temp=0.3 ×6 次，要求给出最弱维度及理由）

- 本轮零掉分；合并全部采样，orch 在 temp=0.3 下掉频率 ≈12%/次
- judge 批判焦点为 `practical_usefulness`（缺具体配置/代码示例）与 `hallucination`（RPO/RTO 量化目标是否充分背书），**从未提及 §3.3**
- 抓到 judge 自相矛盾：对同一句"RPO 目标：5 秒（工程目标非硬性保证）"，#4/#6 判为无依据断言、#5 判为已恰当标注假设

## 3. 结论

1. **不存在真实质量回退**。稳定测量（temp=0）下 orch = base = 5.0；critique 拓扑达成设计目标（矛盾消除 + 质量追平 baseline）。
2. **-0.17 是 judge 采样噪声**。temp=0.3 时单次评分有 ~12% 概率偶发严格扣 1 个维度分；temp=0 即消除。
3. **扣分机制是 judge 对"量化目标是否充分背书"的随机解读**，而非答案客观缺陷（答案已显式标注假设）。
4. §3.3 的 UNIQUE 约束矛盾与本次扣分无关，但属客观瑕疵，列为独立小修项。

## 4. 改进落地

- Judge 评分默认 `temperature=0.0`（commit 76a419c，含单元测试）；生成端保持生产设置不变
- 已识别可观测性缺口：critic findings 未持久化（reviser 决策事件仅记 `{"issues": 22}` 计数），建议后续将 issue description 写入 trace decision

## 5. 遗留事项

- [ ] n=1 单 case，需多 case compare 做统计验证
- [ ] 成本裁剪（+571% tokens）：critics 去 expert_policy 上下文、按 intent 配置视角数等（spec §4 P2）
- [ ] §3.3 示例与约束矛盾的小修（critic prompt 增加"示例与 schema 一致性"检查维度）
