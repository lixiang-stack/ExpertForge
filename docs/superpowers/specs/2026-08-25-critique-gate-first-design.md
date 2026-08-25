# Critique 拓扑 P3 — Gate-First（evaluate-first）设计

Date: 2026-08-25

Status: Approved (2026-08-25)

References:

- [2026-08-24-critique-cost-trim-design.md](2026-08-24-critique-cost-trim-design.md)（P2，A/B/C 已落地于 PR #22）
- 实测：`compare --ids se-1 se-2 se-3 se-4`（2026-08-24 结果文件）

---

## 1. Problem

P2 裁剪（critics 去 context、按 intent 限视角、revise 预算）落地后，实测仍不达标：

```
se-1  base 16907 → orch 40595  (+140%)   Q 5.0 → 5.0
se-2  base 21603 → orch 42648  (+97%)    Q 5.0 → 5.0
se-3  base 21101 → orch 49869  (+136%)   Q 5.0 → 5.0
se-4  base  6898 → orch 33763  (+389%)   Q 5.0 → 5.0
overall +150.9%，质量增益 0
```

两个结构性根因：

1. **成本无条件，价值有条件**。critique 流水线固定支付 planner + N×critic（各重读全文）+ revise（重写全文），成本 ≈ 答案长度 × 常数；但只有 draft 本身有缺陷时才产生价值。实测中 draft 从不需要修补。A/B/C 压低了常数，未改变「无条件」结构。
2. **测量饱和**。全部维度 5.0/5.0，`min_dimension_score=3` 下一切通过——当前基准测不出质量差异（gain≡0 是测量无分辨率，不是 orch 无用的证明）。se-4 增幅最大恰因其 base 最小。

## 2. Goals

1. `evaluator.enabled=true` 时把 critique 全套机制改为**条件执行**：draft 先经 judge 评分，达标直接返回（≈ base + 1 次 judge call）；不达标才进入 perspectives/critics/revise。
2. 失败路径的 judge 调用次数与现有尾部回路完全一致；`enabled=false` 行为逐字节保留。
3. 零新配置键；map_reduce 不动；domains are data 不破坏。

## 3. Non-goals

- 不改 judge prompt、维度、温度与 `Judge.score` 失败语义。
- 不动 map_reduce 的 evaluator/re-aggregate 回路。
- 不在本设计内解决测量饱和（更难 case / 更严 rubric 属 evaluation 数据集工作，另立）。
- 不做「单次 self-check 替代 critics」（方案 C，留作失败率实测偏高后的第二刀）。

## 4. 设计

### 4.1 现状流（P2 后）

```
draft → plan_perspectives → critics×N → consolidate → revise? → _evaluate_loop(judge 尾部)
```

### 4.2 Gate-First 流

`_run_critique` 在 `policy.evaluator.enabled` 时，将 **draft 直接作为 `_evaluate_loop` 的初始答案**，并按轮次分派 `improve`：

```
round 0: judge(draft)
           ├─ 全维度 ≥ min_dimension_score ⇒ return draft          （早退）
           ├─ scorecard is None ⇒ return draft                     （沿用 _evaluate_loop 现有语义）
           └─ 存在低分维度 ⇒ improve(round_no=0)：
                plan_perspectives（intent 上限）→ critics → consolidate
                  ├─ issues 非空 ⇒ 预算化 revise(draft, issues)     （沿用 P2 预算规则）
                  └─ issues 为空 ⇒ 返回原 draft
round 1..max_rounds: judge(improved) ⇒ 通过返回 / 耗尽返回
round ≥ 1 的 improve：judge-feedback revise（现状代码不变）
```

实现形态：新增一个 critique-pass 辅助方法封装「perspectives → critics → consolidate → 条件 revise」，供 round 0 的 `improve` 与 `enabled=false` 的遗留路径共用；`_evaluate_loop` 本体不改。

### 4.3 成本模型

| 路径 | LLM 调用 | 相对 base 开销 |
|------|----------|----------------|
| draft 达标（预期主路径） | draft + judge | ≈ +20–50%（judge 只读 question+answer，不含 thinking tokens） |
| draft 不达标 | draft + judge + planner + N×critic + revise + judge | 与现状相同 +1 judge |
| judge 失败/解析失败 | draft + judge(败) | 同早退路径 |

### 4.4 语义与边界

- **阈值**：沿用 `EvaluatorPolicy.min_dimension_score`（software_engineering 保持 3）。调高即提高门槛、增加进入 critique 的比例——作为按 domain 的成本/严格度旋钮写入文档，本设计不改值。
- **gate 失败但 critics 无 issues**：improve 返回原 draft，round 1 对同一 draft 重判；大概率再次失败后耗尽轮次返回。多花 1 次 judge + N×critic，可接受并记录为已知行为。
- **critics 全部失败（LLMError）**：沿用现有降级链（warning 日志、空 issues）→ revise 跳过 → 重判原 draft。
- **revise LLMError**：沿用现有 catch → 返回 draft 进入重判。
- **调用顺序可观测性**：trace 中 `orchestration.evaluator` 相位出现在 critic 相位之前（gate 在前），属预期变化。

## 5. 配置与兼容

```yaml
# orchestration.yaml —— 不新增任何键
topology: critique
evaluator:
  enabled: true     # true ⇒ gate-first（语义变更点，见 changelog）
  min_dimension_score: 3
  max_rounds: 1
```

- `enabled=false`：无条件 critique、无 judge——行为与 P2 完全一致（回归网：现有全部 `enabled=false` critique 测试不改一字通过）。
- map_reduce 路径零改动；`Judge`、`_evaluate_loop`、预算规则（P2 措施 C）、视角上限（P2 措施 B）、critics 无 context（P2 措施 A）全部复用。

## 6. 实现范围

| 文件 | 变更 |
|------|------|
| `agent/orchestrator.py` | `_run_critique` 重排：draft 作为初始答案进 `_evaluate_loop`；抽取 critique-pass 辅助方法；round 0 improve = critique pass |
| `tests/unit/test_orchestrator.py` | 新增 gate 流测试（见 §7）；现有 enabled=false 测试保持不动 |
| `AGENTS.md` | Architecture 一行更新 critique 流描述 |
| 观测性 / llm / config / domain_config | **零改动** |

## 7. 测试（FakeClient，无需新设施）

1. **早退**：[draft, PASS] → 返回 draft；恰好 2 次调用；无 planner/critic 调用。
2. **失败进全套**：[draft, LOW, perspectives, issues…, revised, PASS] → 返回 revised；断言调用顺序 evaluator 判分先于 critic。
3. **空 issues 耗尽**：[draft, LOW, perspectives, EMPTY, EMPTY, LOW] → 返回 draft，共 6 次调用（max_rounds=1）。
4. **gate 解析失败视为通过**：[draft, "not json"] → 返回 draft，2 次调用。
5. **max_rounds=0**：[draft, LOW] → 仅 1 次 judge，返回 draft，不进 critique。
6. **回归**：`enabled=false` 的现有 5 条 critique 测试不改通过；map_reduce 测试全绿。
7. **预算联动**：失败路径 revise 调用带 P2 预算 `max_tokens`（复用 BudgetRecordingClient）。

## 8. 验证

- 单测：`uv run pytest -q` 全绿。
- 实测门（需 API key）：`uv run python -m agent.evaluation compare --ids se-129 se-1 se-2 se-3 se-4`
  - 质量门：orch Q ≥ base Q；
  - 成本门：通过路径 case 增幅 ≤ ~50%（对比现状 +97~389%）；
  - 若某 case 频繁触发失败路径导致超标，记录失败率——那是启用第二刀（方案 C）或调阈值的依据，不在本设计内处理。

## 9. Acceptance Criteria

1. `evaluator.enabled=true` 的 critique 拓扑执行 gate-first 流；draft 达标时零 critic/planner/revise 调用。
2. judge 失败（LLMError / scorecard=None）视为通过，返回 draft。
3. `evaluator.enabled=false` 与 map_reduce 行为零回归；无新配置键；非法配置行为不变（无新解析逻辑）。
4. 实测 compare 满足 §8 双门（或文档化失败率缺口与后续刀位）。
