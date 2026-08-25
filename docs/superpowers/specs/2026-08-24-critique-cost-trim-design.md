# Critique 拓扑 P2 — 成本裁剪设计

Date: 2026-08-24

Status: Approved (2026-08-24)

References:

- [2026-08-21-refactor-orchestration.md](2026-08-21-refactor-orchestration.md) §4 P2
- [2026-08-22-judge-variance-investigation.md](../investigations/2026-08-22-judge-variance-investigation.md) §3–§5
- PR #22（critique topology）：https://github.com/lixiang-stack/ExpertForge/pull/22 —— 本设计基于该代码状态开发

---

## 1. Problem

critique 拓扑（draft → parallel critics → conditional revise）已消除 map-reduce 的多作者矛盾，且在稳定测量（judge temperature=0）下 se-129 质量追平 baseline（orch Q = base Q = 5.0）。

但成本不可接受：

```
se-129    Base Tok=6121   Orch Tok=41102   (+571%)
```

原 refactor spec §5 将「token 增幅 ≤300%」视为可接受线，超标留给 P2。触发条件已满足：**P1 质量达标，成本不可接受**。

2026-08-24 多 case 复测（`compare --ids se-1 se-2 se-3 se-4`，均为 architecture_design / complex / planning）同样显示质量增益为 0 而 token 增幅持续存在——成本问题普遍成立，非 se-129 孤例。曾评估过「draft → cheap 检查 → 达标早退」的 gate 方案，本设计不采用；A/B/C 为已选定路线。

## 2. Goals

1. 在保持 orch Q ≥ base Q（judge temp=0）且不重现旧四类矛盾的前提下，把 se-129 类 complex/`architecture_design` 的 token 增幅压到 **≤300%**（理想落点 ~150–250%）。
2. 三项裁剪可独立落地、可逐步验证：critics 去 strategy context、按 intent 配置视角数、revise 输出预算控制。
3. 配置遵循 domains are data；map_reduce 路径不动。

## 3. Non-goals

- 不关闭 draft thinking（那是 RC1 修复，质量下限）。
- 不静态写死视角列表（只约束数量；内容仍由 planner 自适应）。
- 不改 `expert_policy.md` 正文、judge、evaluator 回路语义。
- 不在本设计内做多 case 统计验证（仍属 investigation 遗留项）。

## 4. 成本解剖（se-129 / critique，2026-08-22）

```
draft(in≈489/out≈8980, thinking 开)
→ planner(4 视角)
→ critics×4（合并 22 issues；每次重读完整 draft）
→ reviser(out≈5015；终稿长于 draft）
→ evaluator
合计 ≈41k（base ≈6k）
```

| 环节 | 成本性质 | 裁剪杠杆 |
|------|----------|----------|
| draft | 质量下限，与 baseline 同构 | **不裁** |
| planner | 小；携带完整 strategy context | 本轮不裁；仅消费视角数配置 |
| critics×N | **主导 in-tokens**：N × (strategy context + question + 全文 draft) | 去 context + 降 N |
| revise | 再读 draft + 大量 issues → out 膨胀 | 输出预算 + 外科修订约束 |
| evaluator | 测量/回路开销 | 不动 |

落地顺序（影响从大到小）：**B（视角数）→ A（critic context）→ C（revise 预算）**；每步跑一次 compare（命令与门槛见 §10）。

## 5. 措施 A — critics 去掉 expert_policy / strategy context

### 5.1 现状

`Orchestrator._critic` 的 system prompt 以 `{context}` 开头，其中：

```
context = Strategy.build_system_prompt()
        = expert_policy.md + "\n\n" + strategy prompt
```

每个 critic 都重复挂载这份**生成端**身份文案（planning 策略下约 2k 字符）。

### 5.2 改动

- `_CRITIC_SYSTEM_TEMPLATE` **删除** `{context}` 前缀；仅保留 reviewer 角色、focus、缺陷检查清单与 JSON schema 约束。
- `_draft` / `_revise` / `_plan_perspectives` **继续**使用完整 `context`（作者与规划仍需专家身份与 Uncertainty Policy）。
- `expert_policy.md` 文件本身不改。

### 5.3 理由

Critic 是缺陷探测器，不是作者。矛盾 / 无依据断言 / 技术错误已由 critic 清单覆盖；再挂 expert_policy 只增加 N 倍 in-tokens，不增加可验证信号。

### 5.4 预估与验收

- **预估节省**：`len(context)` tokens × N。se-129 上约 500–700 × 4 ≈ **2–3k**（次要杠杆，零质量风险）。
- **验收**：critic 调用的 system 消息不含 expert_policy 特征句（如 "Senior Software Engineering Expert"）；单元测试断言 `_critic` 的 `messages[0]` 不含 policy 文本。

## 6. 措施 B — 按 intent 配置视角数

### 6.1 现状

`_PERSPECTIVES_PROMPT` 硬编码 `Plan 2-4 distinct review perspectives`；planner 在 se-129 上稳定产出 4 个视角 → 4 次全文 draft 重读 + 22 条 issues 灌进 revise。

### 6.2 配置（domains are data）

```yaml
# orchestration.yaml
topology: critique
max_workers: 4                 # 并发上限，语义不变
critique:
  default_max_perspectives: 3  # 全局缺省（比原上限 4 更省）
  max_perspectives_by_intent:
    architecture_design: 3     # 耦合综合型：够覆盖，不放满 4
    troubleshooting: 2
    code_task: 2
```

### 6.3 行为

- `OrchestrationPolicy` 增加可选嵌套 `critique`；非法值 / 未知 intent key → `ConfigError`。
- 无 `critique` 块时：**直接生效新缺省上限 3（已决策，2026-08-24）**；需要旧行为（上限 4）的 domain 显式配置即可。
- `_plan_perspectives` 按 intent 解析上限 `N`，prompt 改为 `Plan exactly N distinct review perspectives`（或 `Plan up to N`，实现选一种并写死测试）。
- Planner **仍自适应选择视角内容与 role**；配置只约束数量，不静态写死视角列表。
- `max_workers` 只做线程池上限，与 `max_perspectives` 解耦。

### 6.4 推荐初值

`architecture_design: 3`——覆盖一致性 / 可行性 / 合规（或成本）三类主风险，去掉第四个边际视角。若 A+B+C 后仍 >300%，再降到 2。

### 6.5 预估与验收

- **预估节省**：每少 1 个 critic ≈ 省一次「全文 draft 重读 + issue 输出」（se-129 量级约 **4–6k**/视角），并减少灌进 revise 的 issue 数。4→3 ≈ 4–6k；4→2 ≈ 8–12k。
- **验收**：配置 `architecture_design: 2` 时 perspectives 长度 ≤ 2；intent 未列出时走 `default_max_perspectives`。

## 7. 措施 C — revise 输出预算控制

### 7.1 现状

`_revise` 无 `max_tokens`；se-129 上 out≈5015，终稿 9079 字符 vs draft/base ~5k 字符——评审在「修复」之外还在扩写，既烧 tokens 又放大 judge 暴露面偏置（investigation §2.6）。

`LLMClient.chat_completion` 当前不支持 `max_tokens`。

### 7.2 改动（硬预算 + 软指令）

1. **`LLMClient.chat_completion` 增加可选 `max_tokens: int | None = None`**，传入时写入 API；其它调用点保持不传（行为不变）。
2. **`_revise` 计算预算**：

   ```
   max_tokens = clamp(
     ceil(draft_completion_tokens * revise_token_ratio),
     revise_min_tokens,
     revise_max_tokens,
   )
   ```

   若拿不到 draft 的 completion 计数，则退化为 `revise_max_tokens` 绝对上限。

3. **prompt 约束**写入 `_REVISE_SYSTEM_TEMPLATE`：只做定向修补，禁止扩写新章节；终稿长度应与 draft 同量级；保留所有未点名的正确内容。

4. **配置**：

```yaml
critique:
  # ... perspectives 同上 ...
  revise_token_ratio: 1.1
  revise_min_tokens: 1024
  revise_max_tokens: 6000
```

### 7.3 可选增强

`_consolidate` 后按 severity 排序，只把 `high` + 前 K 条 `medium` 交给 revise（配置 `revise_max_issues`，例如 8），避免 22 条 issues 把 reviser 拖进重写模式。缺省可关闭。

### 7.4 预估与验收

- **预估节省**：revise out 5015→≤~3500–4000，并抑制终稿膨胀；约 **1–3k** 直接节省，另有测量稳定性收益。
- **验收**：`_revise` 调用带 `max_tokens=`；单元测试用 FakeClient 断言传入值；超长 draft 时不超过 `revise_max_tokens`。

## 8. 合计预估与风险

| 方案组合 | 粗估 Orch Tok（相对 se-129 41k） | 相对 base 6k |
|----------|----------------------------------|--------------|
| 仅 A | ~38–39k | ~+530% |
| A + B(4→3) | ~33–35k | ~+440–470% |
| A + B(4→2) + C | ~24–28k | ~+290–360%（压线或达标） |
| A + B(4→3) + C + issue 顶栏 | ~26–30k | ~+320–390%（接近） |

数字为量级估计，以 se-129 compare 实测为准。若 A+B+C 后仍 >300%，下一刀是 **issue 顶栏** 与 **architecture_design 视角数再降到 2**，而不是动 draft thinking。

**风险与护栏**：

- 视角过少 → 漏检耦合矛盾：靠质量门（矛盾清单人工抽检 + orch Q ≥ base Q）拦截。
- 去掉 critic context → 评审口吻漂移：可接受；缺陷类型由固定清单约束。
- revise 硬截断 → 答案中断：靠 ratio/min 留余量，prompt 要求完整成稿；若截断可观测且需降级，再补「截断 → 返回 draft」（现有 `LLMError → draft` 可作模板）。

## 9. 实现范围

| 文件 | 变更 |
|------|------|
| `domain/software_engineering/orchestration.yaml` | 增加 `critique:` 块 |
| `agent/config.py` | `CritiquePolicy` + `OrchestrationPolicy.critique` |
| `agent/domain_config.py` | 解析/校验 `critique` 字段 |
| `agent/orchestrator.py` | critic 去 context；perspectives 按 intent 取 N；revise 传 max_tokens + prompt |
| `agent/llm.py` | `chat_completion(..., max_tokens=None)` |
| `tests/unit/test_orchestrator.py` / `test_config.py` / llm 相关测试 | 覆盖 A/B/C |
| map_reduce 路径 | **不动** |

## 10. 验证

- **命令**：`uv run python -m agent.evaluation compare --ids se-129 se-1 se-2 se-3 se-4`（topology=critique vs baseline）。
- **质量门**：orch Q ≥ base Q（judge temperature=0）；人工核对无部署模式 / RPO 分层 / 事务方案 / 架构风格四类矛盾。
- **成本门**：se-129 token 增幅 **≤300%**；se-1..se-4 增幅不高于现状且质量增益 ≥ 0。
- **节奏**：B → A → C 分步启用，每步保留 compare 结果。

## 11. 测试

- `_critic` system 不含 expert_policy。
- perspectives 数量受 intent 配置上限约束；缺省 / 非法配置行为符合 §6.3。
- `_revise` 传入预期 `max_tokens`；省略时请求体无该字段。
- 现有 critique / map_reduce 单测与 `uv run pytest tests/unit -q` 全绿。

## 12. Acceptance Criteria

1. 配置可表达 per-intent 视角上限与 revise 预算，非法配置报 `ConfigError`。
2. Critics 不再挂载 strategy/`expert_policy` context；draft/revise/planner 仍挂载。
3. se-129 compare：质量门通过且 token 增幅 ≤300%（或文档化「需再降视角数/启用 issue 顶栏」的实测缺口）。
4. map_reduce 行为与测试无回归。
