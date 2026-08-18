# Orchestration Policy + Evaluator / Optimizer — Design

**Date:** 2026-08-18
**Status:** Design agreed; pending user review
**Source requirements:** `draft_v2.md` §16 (Evaluator / Optimizer) and §17 (Orchestration Policy)

## 1. Goal

Two P1 orchestration upgrades over the current pipeline
(Planner → parallel Workers → Aggregator):

1. **Orchestration Policy (§17):** replace the `complexity_gate`-driven
   "complexity == complex → always orchestrate" decision with a config-driven,
   per-domain policy combining `enabled`, `min_complexity`, and `intents`.
   The orchestrator's own config (`max_workers`) moves out of `config.json`
   into the same policy file.
2. **Evaluator / Optimizer (§16):** after aggregation, an Evaluator judges the
   answer and, on failure, an Optimizer re-aggregates the worker outputs with
   feedback until the answer passes or `max_rounds` is exhausted.

## 2. Decisions (agreed with user)

| Decision | Choice |
|----------|--------|
| Spec scope | One combined spec for §16 + §17. |
| Policy placement | Per-domain `orchestration.yaml` under `domain/<name>/` (mirrors `complexity.yaml`). |
| Backward compatibility | None. `strategies.yaml` deleted; `orchestration.yaml` is required (missing → `ConfigError`). Only the `enabled` flag decides. |
| Orchestrate formula | `enabled AND complexity_rank ≥ min_complexity AND intent ∈ intents`. `intents` is a required non-empty list. |
| Strategy ids | Derived from `prompts/*.md` filenames. `unsupported_complex.md` is deleted (dead code since the orchestrator MVP). |
| Default strategy | Removed. `intent_mapping` must cover every intent (load-time `ConfigError` otherwise); router uses `intent_mapping[intent]` with no fallback. |
| Worker timeout | Deleted globally. All LLM calls are bounded by `config.timeout` only; `worker_timeout` / `OrchestratorConfig` / `effective_timeout` removed. |
| `max_workers` | Lives only in `orchestration.yaml`; the `config.json` `orchestrator` block is deleted. |
| Evaluator | Fully reuses `agent/evaluation/judge.py` (6-dimension scorecard); runs on every orchestrated task when `evaluator.enabled`. No separate enable list — the draft's §16 list is subsumed by the orchestration policy (orchestration already requires `complex` + listed intents). |
| Evaluator model | `config.evaluation.judge_model` (fallback `config.model`). |
| Pass / Fail | All 6 dimensions ≥ `evaluator.min_dimension_score` (default 3). Judge parse failure or `LLMError` → treat as Pass (never degrade a valid answer, never trigger a pointless Optimizer loop). |
| Optimizer | Re-aggregate only (same model as the aggregator) with the previous draft + judge feedback (low-scoring dimension names). Workers are not re-run. |
| Optimizer rounds | `evaluator.max_rounds` (default 1). After exhaustion, return the last version (best effort, no infinite loop). `LLMError` in any optimizer step → return the current answer. |
| Per-strategy model | Removed (unused by any domain). Model routing stays `model_low` / `model_high` / `model`. |

## 3. Component changes

### 3.1 `domain/software_engineering/orchestration.yaml` — policy content

```yaml
enabled: true
min_complexity: complex
intents:
  - architecture_design
  - performance_analysis
  - troubleshooting
  - code_review
max_workers: 4
evaluator:
  enabled: true
  min_dimension_score: 3
  max_rounds: 1
```

### 3.2 `agent/config.py` — data model + loader

- New dataclasses:

  ```python
  @dataclass
  class EvaluatorPolicy:
      enabled: bool = True
      min_dimension_score: int = 3
      max_rounds: int = 1

  @dataclass
  class OrchestrationPolicy:
      enabled: bool = True
      min_complexity: str = "complex"
      intents: list[str] = field(default_factory=list)
      max_workers: int = 4
      evaluator: EvaluatorPolicy = field(default_factory=EvaluatorPolicy)
  ```

- `DomainConfig` gains `orchestration: OrchestrationPolicy` (required) and
  `strategies` becomes `list[str]` (strategy ids derived from `prompts/*.md`
  filenames). `StrategyDef`, `default_strategy`, and `OrchestratorConfig` are
  deleted.
- `load_domain_config`:
  - Reads `orchestration.yaml` — required; missing, non-mapping, empty `intents`,
    unknown `min_complexity`, unknown intent ids, or invalid `evaluator` values →
    `ConfigError`.
  - Validates `intent_mapping` covers every intent in `intents.yaml` →
    `ConfigError` on gaps.
  - Derives strategy ids from the `prompts/` directory; loads each
    `prompts/<sid>.md`. The `unsupported_complex.md` load is removed.
- `load_config`: the top-level `orchestrator` block parsing is removed.
- `effective_timeout` is deleted; callers use `config.timeout` directly.

### 3.3 `agent/router.py` — orchestration decision

```python
policy = self.domain.orchestration
orchestrate = (
    policy.enabled
    and complexity_rank(result.complexity) >= complexity_rank(policy.min_complexity)
    and result.intent in policy.intents
)
```

- `complexity_rank`: simple < medium < complex.
- `strategy = self.domain.intent_mapping[intent]` — no fallback.

### 3.4 `agent/worker_pool.py`

- `run_workers` drops the `timeout` parameter and the per-worker wall-clock
  `future.result(timeout=...)` guard. All LLM calls are bounded by the client
  `config.timeout`.

### 3.5 `agent/orchestrator.py` — Evaluator / Optimizer loop

`run()` after aggregation (only when `policy.evaluator.enabled`):

```
aggregated = _aggregate(...)
if not policy.evaluator.enabled:
    return aggregated
judge = Judge(self.client, judge_model)
for _ in range(max_rounds + 1):
    scorecard = judge.score(question, aggregated)      # reference=None
    if scorecard is None:                              # parse failure / LLMError
        return aggregated                              # treat as Pass
    if all(score >= policy.evaluator.min_dimension_score for score in scorecard.values()):
        return aggregated                              # Pass → Final
    if round == max_rounds:
        return aggregated                              # exhaustion → last version
    aggregated = _reaggregate(question, results, aggregated, feedback)  # Fail → Optimizer
```

- `judge_model = (config.evaluation.judge_model if config.evaluation else None) or config.model`.
- `_reaggregate` re-invokes the aggregator with the worker results **plus** the
  previous draft and the judge feedback — the dimension names that scored below
  the threshold (e.g. `correctness: 2/5, hallucination: 2/5`). The system prompt
  adds: "The previous draft scored low on these dimensions: <feedback>. Improve
  accordingly." Same model as the original aggregation.
- Worker output is never re-run; the optimizer is aggregation-only.
- Any `LLMError` inside `_reaggregate` → return the previous `aggregated`
  (no crash, no loop).

### 3.6 `agent/observability/patch.py`

- `_wrap_aggregate` decision record appends `{"evaluated": bool, "evaluator_model": ...}`.
- New evaluator decision record: `{"phase": "evaluator", "scorecard": {...6 dims}, "passed": bool}`.
- New optimizer decision record per round: `{"phase": "optimizer", "round": n, "feedback": [...], "passed": bool}`.
- Existing trace/phase machinery, thread-safe counters, and `contextvars`
  propagation are unchanged.

### 3.7 Deletions

- `domain/software_engineering/strategies.yaml`
- `domain/software_engineering/prompts/unsupported_complex.md`
- `config.example.json` `orchestrator` block (and any `orchestrator` block in
  `config.json`)
- `OrchestratorConfig`, `StrategyDef`, `default_strategy`, `effective_timeout`

## 4. Data flow

```
Planner → parallel Workers → Aggregator → Evaluator
                                              │
                                 ┌────────────┴────────────┐
                                 │ Pass                    │ Fail
                                 ▼                         ▼
                               Final                  Optimizer (re-aggregate
                                                       + feedback)
                                                               │
                                                               ▼
                                                          Evaluator (repeat,
                                                        ≤ max_rounds)
```

## 5. Error handling

| Failure | Behavior |
|---------|----------|
| `orchestration.yaml` missing / invalid | `ConfigError` at load. |
| `intent_mapping` misses an intent | `ConfigError` at load. |
| Judge parse failure / `LLMError` | Treated as Pass; answer returned unchanged. |
| Re-aggregation `LLMError` | Previous draft returned. |
| `max_rounds` exhausted | Last version returned. |
| Worker failures (existing §14 behavior) | Partial aggregation unchanged. |

## 6. Testing

- `tests/test_config.py`:
  - `orchestration.yaml` parses into `OrchestrationPolicy`; missing → `ConfigError`;
    unknown `min_complexity` / unknown intent / invalid `evaluator` → `ConfigError`.
  - `intent_mapping` coverage validation (gap → `ConfigError`).
  - Strategy ids derived from `prompts/` directory.
  - Removed assertions: `unsupported_complex.md` requirement, `default_strategy`,
    `orchestrator` block in `config.json`.
- `tests/test_router.py`:
  - Decision formula: all three conditions; `enabled: false` never orchestrates;
    complexity below `min_complexity` does not; intent not in `intents` does not;
    `complexity_gate` logic removed.
- `tests/test_orchestrator.py`:
  - Evaluator runs after aggregation (Judge invoked with question + aggregated
    answer); all-dimensions-above-threshold → Pass, no optimizer call.
  - A dimension below threshold → optimizer re-aggregates (aggregator invoked again
    with previous draft + feedback dimension names), judge re-run, then Pass.
  - `max_rounds` exhausted while still failing → last version returned, loop count
    capped.
  - Judge parse failure / `LLMError` → treated as Pass.
  - Optimizer `LLMError` → previous draft returned.
  - `evaluator.enabled: false` → no judge call.
- `tests/test_worker_pool.py`: `run_workers` without a timeout parameter.
- `tests/test_observability_patch.py` / `tests/test_chat.py`: updated decision
  records and `config.timeout` usage.
- `tests/test_evaluation_judge.py`: unchanged (Judge reused, not rebuilt).
- `tests/test_evaluation_runner.py`: `effective_timeout` tests replaced with
  `config.timeout` behavior.

## 7. Out of scope

- Re-running workers inside the Optimizer.
- More than `max_rounds` evaluator iterations / dynamic round budgeting.
- Reintroducing a per-worker wall-clock timeout or per-strategy model overrides.
- A generic policy loader shared across `orchestration.yaml` / `complexity.yaml`
  / `intents.yaml`.
- Non-software-engineering domains.

## 8. Success criteria

1. `uv run pytest -q` all green.
2. Orchestration is decided solely by `orchestration.yaml`
   (`enabled AND complexity_rank ≥ min_complexity AND intent ∈ intents`);
   `complexity_gate` is gone.
3. Orchestrated tasks run the Evaluator (Judge) by default; low-scoring answers
   are re-aggregated with feedback up to `max_rounds`.
4. A failing/hung judge never degrades a valid answer; the Optimizer never loops
   past `max_rounds`.
5. `config.timeout` bounds every LLM call; no orchestrator config remains in
   `config.json`.
6. `strategies.yaml`, `unsupported_complex.md`, `default_strategy`, and
   `OrchestratorConfig` are gone.