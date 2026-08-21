# Orchestration Experiment: Single-Agent vs Orchestrated Pipeline Comparison

Date: 2026-08-21

Status: Draft

Reference: `draft_v3.md` §9

---

## 1. Problem

The system supports two execution paths: a single-agent (direct strategy prompt) path and an orchestrated multi-agent pipeline (planner → workers → aggregator → evaluator). The orchestrated pipeline is gated by `orchestration.yaml` based on intent + complexity. However, there is no tooling to **measure** whether orchestration actually produces better answers than the single-agent path, and if so, at what cost.

Without this data, routing decisions are guesswork: which intents and complexities benefit from orchestration? How much quality gain does each extra LLM call buy? The system needs a controlled experiment tool to answer these questions.

## 2. Goals

1. Build a `compare` subcommand that runs each `full_expert` case through two execution modes:
   - **Baseline:** single strategy-prompt LLM call (direct answer)
   - **Orchestrated:** full production pipeline (planner → workers → aggregator → evaluator)
2. Record per-case and aggregate metrics: answer quality, LLM calls, tokens, latency
3. Compute `Quality Gain`, `Additional Tokens`, `Cost Efficiency` per case and grouped by intent/complexity
4. Produce a comparison report (JSON + console summary)
5. All changes isolated to a new module `agent/evaluation/compare.py` + CLI — zero production code changes

## 3. Non-goals

- Not running the experiment and interpreting results (that's a separate activity)
- Not updating orchestration.yaml or routing policy based on results
- Not adding pricing/cost in dollars (tokens only)
- Not adding required-points coverage or expert-behavior rubrics (existing 6-dim judge is sufficient)
- Not modifying Orchestrator, Chat, Router, or any production code path

## 4. Execution Modes

### 4.1 Baseline

A single LLM call using the case's expected strategy prompt as the system context:

```
strategy prompt → single LLM call → answer
```

Implementation: `Strategy.process(client, question, [], model=config.model)` — exactly one call.

### 4.2 Orchestrated

The full production multi-agent pipeline:

```
strategy context → planner → workers (2-4 parallel) → aggregator → evaluator (internal judge) → (optional reaggregate) → answer
```

Implementation: `Orchestrator.run(question, route, model=config.model)` with a synthetic `RouteResult(orchestrate=True)`. The orchestrator's internal evaluator runs the same judge as production (controlled by `orchestration.yaml` policy). All internal LLM calls (planner, workers, aggregate, internal evaluator judge, reaggregate) are counted in the orchestrated cost.

### 4.3 Model

Both modes use `config.model` (the same model). This isolates the execution mode as the only variable — not model quality.

## 5. Architecture

### 5.1 New module: `agent/evaluation/compare.py`

```
CompareCaseResult
  └─ case: EvalCase
  └─ baseline: ModeRun          # single-call result
  └─ orchestrated: ModeRun      # multi-agent pipeline result
  └─ quality_gain: float|None   # orch_quality - base_quality
  └─ quality_gain_pct: float|None
  └─ additional_tokens: int
  └─ token_increase_pct: float|None
  └─ cost_efficiency: float|None  # quality_gain / additional_tokens

ModeRun
  └─ answer: str|None
  └─ scorecard: dict|None
  └─ quality: float|None          # mean of 6 judge dims
  └─ llm_calls: int
  └─ in_tokens, out_tokens, total_tokens, cache_tokens: int
  └─ latency_ms: float
  └─ error: str|None
```

### 5.2 Core function: `run_compare`

```python
def run_compare(
    config: AgentConfig,
    domain: DomainConfig,
    cases: list[EvalCase],
    client: LLMClient,
    judge_client: LLMClient | None = None,
) -> list[CompareCaseResult]:
```

Flow per case:
1. Create `RecordingClient` wrappers for each mode (baseline_recorder, orch_recorder) and a separate judge_recorder
2. Build synthetic `RouteResult` for baseline: `RouteResult(in_domain=True, strategy=expected, intent=expected, complexity=expected, orchestrate=False)`
3. Build synthetic `RouteResult` for orchestrated: `RouteResult(in_domain=True, strategy=expected, intent=expected, complexity=expected, orchestrate=True)`
4. **Baseline:** `registry[strategy].process(client=baseline_recorder, question, history=[], model=config.model)`
5. **Orchestrated:** `Orchestrator(orch_recorder, config, domain).run(question, route, config.model)`
6. **Judge both:** `Judge(judge_recorder, judge_model).score(question, answer, reference=case.reference)`
7. Compute `quality_gain = orch_quality - base_quality`, `additional_tokens = orch_total - base_total`, `cost_efficiency = quality_gain / additional_tokens`
8. Guard: if `additional_tokens <= 0`, cost_efficiency is None. If `base_quality == 0`, quality_gain_pct is None. If either mode has no quality, quality_gain is None.

### 5.3 Cost accounting

| Source | Counted in baseline cost | Counted in orchestrated cost |
|--------|-------------------------|------------------------------|
| Strategy processor call | Yes | — |
| Planner call | — | Yes |
| Worker calls | — | Yes |
| Aggregator call | — | Yes |
| Internal evaluator judge call | — | Yes |
| Reaggregate call (if any) | — | Yes |
| External quality judge call | No | No |

The external quality judge is measurement overhead, applied equally to both modes. It is NOT counted in either mode's cost so the "additional cost" delta reflects only the pipeline difference.

## 6. Report Format

### 6.1 JSON output

Written to `evaluation/results/compare-<label>-<timestamp>.json`:

```json
{
  "kind": "compare",
  "label": "experiment-1",
  "model": "gpt-4o",
  "judge_model": "gpt-4o-mini",
  "n_cases": 5,
  "n_compared": 5,
  "cases": [
    {
      "id": "se-052",
      "intent": "troubleshooting",
      "complexity": "complex",
      "strategy": "analysis",
      "baseline": {
        "quality": 3.8,
        "scorecard": {"correctness": 4, "relevance": 4, "completeness": 3, "technical_depth": 4, "practical_usefulness": 4, "hallucination": 4},
        "llm_calls": 1,
        "total_tokens": 150,
        "in_tokens": 100,
        "out_tokens": 50,
        "cache_tokens": 30,
        "latency_ms": 1200,
        "error": null
      },
      "orchestrated": {
        "quality": 4.3,
        "scorecard": {...},
        "llm_calls": 5,
        "total_tokens": 1000,
        "in_tokens": 700,
        "out_tokens": 300,
        "cache_tokens": 200,
        "latency_ms": 8500,
        "error": null
      },
      "quality_gain": 0.5,
      "quality_gain_pct": 13.2,
      "additional_tokens": 850,
      "token_increase_pct": 566.7,
      "cost_efficiency": 0.00059
    }
  ],
  "aggregates": {
    "overall": {
      "mean_quality_gain": 0.42,
      "sum_additional_tokens": 4250,
      "cost_efficiency": 0.00049,
      "n_compared": 5
    },
    "by_intent": {
      "troubleshooting": {
        "mean_quality_gain": 0.50,
        "sum_additional_tokens": 850,
        "cost_efficiency": 0.00059,
        "n_compared": 1
      },
      "architecture_design": {
        "mean_quality_gain": 0.35,
        "sum_additional_tokens": 2400,
        "cost_efficiency": 0.00015,
        "n_compared": 2
      },
      "code_task": {
        "mean_quality_gain": 0.40,
        "sum_additional_tokens": 1000,
        "cost_efficiency": 0.00040,
        "n_compared": 2
      }
    },
    "by_complexity": {
      "complex": {
        "mean_quality_gain": 0.42,
        "sum_additional_tokens": 4250,
        "cost_efficiency": 0.00049,
        "n_compared": 5
      }
    }
  }
}
```

### 6.2 Console summary

```
Compare: baseline vs orchestrated (5 cases)
-------------------------------------------
Case         Intent               Base Q  Orch Q  Gain  Gain%   Base Tok  Orch Tok  +Tok    Tok%    Cost Eff
se-052       troubleshooting      3.8     4.3     +0.5  +13.2%   150      1000      +850   +566.7%  0.00059
se-071       architecture_design  3.6     3.9     +0.3   +8.3%   200      1400     +1200   +600.0%  0.00025
...
───────────────────────────────────────────────────────────
Overall                          3.7     4.1     +0.4  +11.4%   1800     6050     +4250   +236.1%  0.00049

By intent:
  troubleshooting:      0.50 gain / +850 tokens / 0.00059 eff (1 case)
  architecture_design:  0.35 gain / +2400 tokens / 0.00015 eff (2 cases)
  code_task:            0.40 gain / +1000 tokens / 0.00040 eff (2 cases)

By complexity:
  complex:              0.42 gain / +4250 tokens / 0.00049 eff (5 cases)
```

## 7. CLI

New subcommand under `agent/evaluation`:

```
uv run python -m agent.evaluation compare [--label X] [--config PATH] [--results-dir DIR] [--ids se-052 se-071]
```

- Defaults: `--label "compare"`, runs all `full_expert` cases in the dataset.
- `--ids`: optional filter to specific case IDs (subset of full_expert).
- No `--tier` flag — always runs full_expert (the only meaningful tier for this experiment).
- Reuses the same `--config` / `--dataset` / `--results-dir` flags as `run`.
- Reuses the same judge client construction logic as `run` (shared helper).

## 8. Error Handling

- Per-case: if a mode's LLM call throws, record `error` in that mode's slot, set `answer=None`, `quality=None`.
- If either mode lacks quality, the case's `quality_gain` is `None` and excluded from aggregate `mean_quality_gain`.
- If both modes fail, same — excluded from quality aggregates, but costs still recorded.
- One failing case does not abort the rest (same pattern as `run_evaluation`).
- Judge parsing failure → scorecard None → quality None → excluded.

## 9. Testing

Add `tests/unit/test_evaluation_compare.py` with `FakeClient` (same pattern as `test_evaluation_runner.py`):

| Test | What it verifies |
|------|-----------------|
| `test_compare_baseline_single_call` | Baseline makes exactly 1 LLM call |
| `test_compare_orchestrated_multiple_calls` | Orchestrated makes 5+ calls (planner, 2 workers, aggregate, internal evaluator) |
| `test_compare_quality_gain` | quality_gain = orch_quality - base_quality from fake scorecards |
| `test_compare_additional_tokens` | additional_tokens = orch_total - base_total |
| `test_compare_cost_efficiency` | cost_efficiency = quality_gain / additional_tokens |
| `test_compare_quality_pct` | quality_gain_pct = (gain / base_quality) * 100 |
| `test_compare_token_pct` | token_increase_pct = (additional / base_tokens) * 100 |
| `test_compare_error_per_mode` | One mode fails → quality=None, quality_gain=None, other mode's data still recorded |
| `test_compare_both_modes_fail` | Both modes fail → quality_gain=None, excluded from aggregates |
| `test_compare_zero_quality_guard` | baseline quality=0 → quality_gain_pct=None |
| `test_compare_zero_tokens_guard` | additional_tokens=0 → cost_efficiency=None |
| `test_compare_aggregate_by_intent` | Per-intent aggregate contains correct n_compared, mean quality_gain, sum tokens |
| `test_compare_aggregate_by_complexity` | Per-complexity aggregate contains correct values |
| `test_compare_aggregate_overall` | Overall aggregate contains correct values |
| `test_compare_cli_args` | CLI arg parsing works (--ids, --label, --config, --results-dir) |
| `test_compare_cli_defaults` | Default selects all full_expert cases |

## 10. Implementation Plan

The implementation is structured as a single task with steps:

1. Create `agent/evaluation/compare.py` with `ModeRun`, `CompareCaseResult` dataclasses and `run_compare` function
2. Implement `ModeRun.from_answers` helper to aggregate recorder calls + judge scorecard into a `ModeRun` struct
3. Implement `run_compare` flow: per case, construct synthetic routes, run both modes, judge, compute deltas
4. Implement report serialization (JSON + console summary)
5. Add `compare` subcommand to `agent/evaluation/__main__.py` (reuse judge client helper from `run`)
6. Add `tests/unit/test_evaluation_compare.py` with unit tests using `FakeClient`
7. Run full test suite to verify no regressions

## 11. Acceptance Criteria

1. `uv run python -m agent.evaluation compare --help` shows the compare subcommand with flags
2. `uv run python -m agent.evaluation compare` runs all full_expert cases, prints a comparison table, and writes a JSON report
3. `uv run python -m agent.evaluation compare --ids se-052` runs only the specified case
4. The JSON report contains per-case modes, quality_gain, additional_tokens, cost_efficiency, and aggregates by intent and complexity
5. `uv run pytest tests/unit/test_evaluation_compare.py -q` passes all tests
6. `uv run pytest -q` passes (no regressions)