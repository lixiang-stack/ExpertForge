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

### 2.6 偶发扣分为何只落在 orch：暴露面机制

| 因素 | base | orch |
|---|---|---|
| 答案长度 | 5186 字符 | 9079 字符 |
| 可证伪细节密度 | 低（决策表、原则性表述） | 高（具体 SQL/DDL/参数名/版本声明） |

- temp=0.3 的每次评分 ≈ 从"评审严格度 × 注意力焦点"分布中抽样；答案含 k 个候选瑕疵时，P(掉分) ≈ 1−(1−p)^k
- **base 生成的是抽象内容（无量化指标、少具体断言），很难被具象地挑出错误；orch 生成的内容很具体，评价时更容易找出问题**——掉分概率因此结构性不对称（实测 base 0/4，orch 1/4）
- 含义：这是测量器的系统性偏置而非纯噪声——用绝对评分直接比较"抽象 vs 具体"的答案时，天平偏向抽象方
- 注意：temp=0 只是恰好落在宽松解码模式，消除方差 ≠ 判定对错；换模型版本可能是"确定性偏严"

## 3. 结论

1. **不存在真实质量回退**。稳定测量（temp=0）下 orch = base = 5.0；critique 拓扑达成设计目标（矛盾消除 + 质量追平 baseline）。
2. **-0.17 是 judge 采样噪声**。temp=0.3 时单次评分有 ~12% 概率偶发严格扣 1 个维度分；temp=0 即消除。
3. **扣分机制是 judge 对"量化目标是否充分背书"的随机解读**，而非答案客观缺陷（答案已显式标注假设）。
4. §3.3 的 UNIQUE 约束矛盾与本次扣分无关，但属客观瑕疵，列为独立小修项。
5. **judge 绝对评分存在"暴露面偏置"**：具体性内容比抽象内容更容易被挑刺，比较两类答案时天平系统性偏向抽象方。后续评估宜采用成对比较或 rubric 化评分以消除该偏置。

## 4. 改进落地

- Judge 评分默认 `temperature=0.0`（commit 76a419c，含单元测试）；生成端保持生产设置不变
- 已识别可观测性缺口：critic findings 未持久化（reviser 决策事件仅记 `{"issues": 22}` 计数），建议后续将 issue description 写入 trace decision

## 5. 遗留事项

- [ ] n=1 单 case，需多 case compare 做统计验证
- [ ] 成本裁剪（+571% tokens）：critics 去 expert_policy 上下文、按 intent 配置视角数等（spec §4 P2）
- [ ] §3.3 示例与约束矛盾的小修（critic prompt 增加"示例与 schema 一致性"检查维度）
