# Evaluation Framework (P0) Design

## 1. Goal

建立 ExpertForge 的基础评估体系，使后续 Prompt、Classification、Routing 和 Strategy 优化均可以通过数据验证。

P0 验收标准（来自 `draft_v2.md` §4.8）：

- 能批量执行 Golden Dataset。
- 能输出 Classification Accuracy。
- 能输出 Routing Accuracy。
- 能输出 Answer Quality。
- 能输出 Token / Latency。
- Prompt 修改后可以进行 A/B Evaluation。
- Evaluation 结果可用于比较不同模型和不同 Prompt。

## 2. Architecture

### 2.1 Module layout

```
evaluation/                          # NEW top-level data dir (grows across domains)
├── datasets/
│   └── software_engineering.yaml    # the golden dataset (committed)
└── results/                         # gitignored; per-run JSON results
    └── 2026-08-14-<label>.json

agent/evaluation/                    # NEW subsystem (sibling to agent/observability/)
├── __init__.py                      # facade / wiring
├── __main__.py                      # `python -m agent.evaluation`
├── dataset.py                       # load + validate YAML cases
├── runner.py                        # drive pipeline per case, collect metrics
├── judge.py                         # LLM-as-judge scoring
├── metrics.py                       # accuracy/cost aggregations + breakdowns
├── report.py                        # JSON write + terminal summary
└── diff.py                          # compare two result files
```

### 2.2 Decoupling from observability

Evaluation reads **only**:

1. Pipeline return values — `RouteResult` from `Router.route()`, `ChatResponse` from `Chat.respond()`.
2. Its own evaluator-owned client wrapper that records token/latency from `LLMClient._usage_local`.

It never reads `TraceStore` / observability event stream. Observability can be enabled, disabled, or replaced without affecting evaluation.

### 2.3 Small change to `agent/chat.py`

`Chat.respond(question, *, route=None)` — evaluation passes the route it already computed, avoiding a second classification LLM call. When `route` is provided, `respond` skips `self.router.route(question)` and uses the given `RouteResult`. Default `None` preserves current behavior; `Chat`, `repl`, `agent_cli` unchanged.

### 2.4 Client wrapper

`agent/evaluation/runner.py` owns a thin wrapper around `LLMClient` that records per-call usage and latency into a per-case collector. It reads `client._usage_local.usage` (the same thread-local populated by `LLMClient.chat_completion`) for prompt/completion/total/cache tokens and times each call itself. This is independent of `agent/observability/client.py`.

## 3. Dataset

### 3.1 Location

One YAML file per domain under `evaluation/datasets/`. Initial: `evaluation/datasets/software_engineering.yaml`. The loader is path-agnostic (`--dataset`), so the layout can evolve without code changes.

### 3.2 Case schema

```yaml
domain: software_engineering          # the domain being evaluated
cases:
  - id: se-001
    question: "What is dependency injection?"
    category: knowledge               # knowledge | problem_solving | evaluation | generation | boundary
    expected:
      domain: software_engineering
      intent: concept_explain
      complexity: simple              # simple | medium | complex
      strategy: teaching
      orchestrate: false              # expected orchestrator usage for this case
    answer_quality: true              # default true; judge this case's answer
    reference: "..."                  # optional; ground truth for the judge
```

- `expected.domain` is the expected domain for this case. For in-domain cases it equals the dataset's `domain`. For out-of-domain boundary cases it is `"other"` (the pipeline should reject).
- `expected.strategy` is the expected route strategy; `expected.orchestrate` is whether the pipeline should run the Orchestrator.
- `answer_quality: true` (default) means the runner executes the full pipeline (real answer generation) and invokes the judge. `--skip-quality` runs classification/routing-only (cheap).
- `category` drives breakdown reporting, not metrics.

### 3.3 Coverage (initial `software_engineering` dataset)

At least cover:

- Knowledge: FAQ, concept_explain, tutorial, learning_guide
- Problem Solving: troubleshooting, performance_analysis, architecture_design
- Evaluation: comparison, code_review
- Generation: generate_code
- Boundary cases (marked `category: boundary`): SE/non-SE, FAQ/concept_explain, tutorial/learning_guide, troubleshooting/performance_analysis, comparison/architecture_design, medium/complex.

Initial dataset size: ~40-60 cases.

## 4. Metrics

### 4.1 Classification accuracy

- `domain_accuracy` — fraction of cases where predicted `in_domain` matches the expected in-domain status (`expected.domain == dataset.domain`).
- `intent_accuracy` — fraction where predicted intent equals expected intent (only in-domain cases).
- `complexity_accuracy` — fraction where predicted complexity equals expected complexity (only in-domain cases).
- Per-intent breakdown: `intent_accuracy` grouped by expected intent.
- Out-of-domain cases (`expected.domain: "other"`) contribute to `domain_accuracy` only (predicted `in_domain` should be false); they are excluded from intent/complexity accuracy.

### 4.2 Routing accuracy

- `strategy_accuracy` — fraction where routed strategy equals `expected.strategy`. For out-of-domain cases (`expected.domain: "other"`) the expected strategy is `reject`.
- `orchestration_accuracy` — fraction where the case actually orchestrated matches `expected.orchestrate` (out-of-domain cases expect `false`).
- `model_routing_accuracy` — fraction where the model used for the answer call matches `resolve_model(config, domain, route, config.model)` (the model the pipeline *should* have selected). Out-of-domain cases have no answer call and are excluded.

### 4.3 Answer quality (LLM-as-judge)

Judge scores each answer on six dimensions, 1-5:

- Correctness
- Relevance
- Completeness
- Technical Depth
- Practical Usefulness
- Hallucination / Unsupported Claims

Aggregated as per-dimension mean over answer-quality cases. Judge uses a separate `judge_model` from config (see §6). For complex cases the judge prompt includes the optional `reference` as ground truth. Out-of-domain cases (`expected.domain: "other"`) produce no answer and are excluded from answer quality.

### 4.4 Cost / latency

Per case and aggregate:

- `input_tokens`, `output_tokens`, `total_tokens`, `cache_tokens`
- `llm_calls`
- `latency_ms`

Breakdown by routing path: `simple → direct`, `medium → strategy`, `complex → orchestrator`.

## 5. Runner flow

For each case, sequentially:

1. `route = router.route(question)` — records classification + routing decisions (one LLM call).
2. If `answer_quality` (and not `--skip-quality`): `response = chat.respond(question, route=route)` — full pipeline; records answer + tokens/latency.
3. If judged: `scorecard = judge.score(question, response.text, reference)`.
4. Per-case result collected into the run record.

The evaluator-owned client wrapper records token/latency for every LLM call (classification, strategy, orchestration).

## 6. Config

Extend `AgentConfig` with an optional `evaluation` block:

- `evaluation: { judge_model: str | None, results_dir: "evaluation/results" }` — evaluation-only settings. `judge_model` is the model used by the LLM-as-judge (falls back to `model` if omitted); it is consumed only by the evaluation subsystem, so it lives inside `evaluation` rather than at the top level.

`python -m agent.evaluation` loads the same `config.json` (path overridable via `--config`). Model, domain_dir, model tiers all come from the existing agent config — evaluation measures the *real* agent configuration.

## 7. CLI

```bash
# Run the golden dataset (all metrics; full pipeline for answer_quality cases)
python -m agent.evaluation run [--dataset evaluation/datasets/software_engineering.yaml]
                               [--label my-run]
                               [--skip-quality]
                               [--config path/to/config.json]

# Compare two runs
python -m agent.evaluation diff <run-a.json> <run-b.json>
```

Output:
- Terminal summary: accuracy numbers, answer-quality means, cost/latency aggregate + routing-path breakdown.
- JSON result file: `evaluation/results/2026-08-14-<label>.json` (timestamped, machine-readable, diffable).

`diff` prints per-metric comparison of two result files (accuracy deltas, answer-quality deltas, token/latency deltas).

## 8. cache_tokens

Add cache-token capture to `LLMClient.chat_completion` in `agent/llm.py`:

- Read `resp.usage.prompt_tokens_details.cached_tokens` if present, else 0.
- Store alongside the existing thread-local usage (e.g. `_usage_local.cache_tokens`).
- Add tests in `tests/test_llm.py`.

This is the one change to core business code; it is additive and default-safe (0 when the provider does not report cache tokens). Observability's `TracedLLMClient` is not required to change for P0.

## 9. Testing

- `tests/test_evaluation_dataset.py` — dataset loading, validation errors (missing fields, unknown intent/strategy/complexity, domain mismatch).
- `tests/test_evaluation_runner.py` — runner drives the pipeline with a FakeClient (like existing modules); per-case collection; `route=` reuse avoids double classification.
- `tests/test_evaluation_metrics.py` — accuracy math, per-intent breakdown, routing-path cost breakdown.
- `tests/test_evaluation_judge.py` — judge prompt building, JSON scorecard parsing, fallback when the judge output is unparseable.
- `tests/test_evaluation_diff.py` — diff of two result dicts.
- `tests/test_evaluation_report.py` — terminal summary + JSON write.
- `tests/test_llm.py` — cache_tokens capture.
- Real-API smoke path mirrors `tests/test_smoke.py` (skips without `AGENT_API_KEY`): run a tiny dataset slice end-to-end and assert a result file is produced.

All unit tests pass without an API key (`uv run pytest -q`).

## 10. Out of scope (P0)

- Concurrency / parallel case execution (sequential for P0).
- HTML report rendering (JSON + terminal only).
- Auto-optimization / regression gates on metrics.
- Observability integration (deliberately decoupled).
- Cache-token capture inside `TracedLLMClient` / observability reports.