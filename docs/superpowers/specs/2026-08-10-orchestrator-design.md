# Orchestrator Design (Complex Task Pipeline — MVP)

**Date:** 2026-08-10
**Status:** Design agreed; pending user review
**Source requirement:** `draft_v1.md` sections 5.1, 13, 18, 19 (Execution); `draft.md` section 十

## 1. Goal

Replace the `COMPLEX_UNSUPPORTED` placeholder with a real **Orchestrator** for complex
tasks: when a strategy has `complexity_gate: true` and the classifier returns
`complex`, route the question through an MVP pipeline **Planner → Workers →
Aggregator** that produces a final answer, instead of returning the
`unsupported_complex` prompt.

Per `draft_v1.md` §19, the MVP excludes Evaluator/Optimizer (multi-round
Evaluator, auto Prompt Optimization) and multi-tier Planner. Those are marked with
`TODO` comments for later iterations.

Out of scope: Evaluator/Optimizer phases, concurrent worker execution,
multi-tier planning, complex memory. These are `TODO`-noted, not implemented.

## 2. Routing change (`agent/router.py`)

`RouteResult` gains `orchestrate: bool = False`.

`route()` no longer rewrites the strategy to `COMPLEX_UNSUPPORTED`. Instead, when
the complexity gate triggers, it keeps the original strategy (so its prompt context
is preserved) and sets `orchestrate=True`:

```python
orchestrate = False
strategy_def = self.domain.strategies.get(strategy)
if strategy_def and strategy_def.complexity_gate and result.complexity == "complex":
    orchestrate = True
return RouteResult(
    in_domain=True,
    strategy=strategy,
    intent=intent_id,
    complexity=result.complexity,
    orchestrate=orchestrate,
)
```

- The `COMPLEX_UNSUPPORTED` constant is deleted; `unsupported_complex` is no longer
  produced by routing.
- Reject / non-complex paths are unchanged.

## 3. New module: `agent/orchestrator.py`

```python
class Orchestrator:
    def __init__(self, client, config, domain)
    def run(self, question: str, route: RouteResult, model: str) -> str
```

The `run()` method executes three stages in sequence, all using the same `model`
(the caller resolves it via `resolve_model`, which for `complex` returns
`model_high`).

### 3.1 Planner

Single LLM call with structured output (`json_schema`):

```json
{
  "type": "object",
  "properties": {
    "tasks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title":       {"type": "string"},
          "instruction": {"type": "string"}
        },
        "required": ["title", "instruction"]
      }
    }
  },
  "required": ["tasks"]
}
```

Planner prompt = domain description + the strategy's prompt template + the user
question + a rule to decompose into 2-4 focused sub-tasks. It asks the model to
split the complex task into sub-tasks it can each answer as a standalone
LLM call.

**Failure/degradation:** if the planner output fails JSON parse or validation (no
`tasks`, empty tasks, malformed items), `run()` degrades to a **single direct
answer** using the strategy's prompt in one LLM call (same shape as the existing
processor path). It never fabricates a fake plan or returns a broken result.

### 3.2 Workers

For each sub-task in order (sequential; concurrency is a `TODO`), one LLM call:

- system: strategy prompt template + `"Sub-task: {instruction}"`
- user: the original user question

Each worker's output is collected into a list. A worker whose output is empty
string contributes an empty entry (Aggregator decides how to handle it).

### 3.3 Aggregator

Single LLM call that receives the original question, the strategy prompt, and all
worker outputs (labeled by `title`), and produces the final answer string.

## 4. Chat integration (`agent/chat.py`)

```python
if route.orchestrate:
    model = resolve_model(self.config, self.domain, route, self.config.model)
    answer = self.orchestrator.run(question, route, model)
    self.history.append((question, answer))
    return ChatResponse(kind="answer", text=answer)
```

- `Chat` constructs an `Orchestrator(self.client, self.config, self.domain)` once in
  `__init__`.
- The `COMPLEX_UNSUPPORTED` branch in `respond()` is removed (no more
  `unsupported_complex` responses).
- Reject branch unchanged.

## 5. Error handling

- **Planner parse/validation failure** → degrade to a single direct answer (one LLM
  call via the strategy prompt). No fake plan.
- **LLMError in any stage** → propagates to the caller (`Chat`), matching existing
  processor behavior (errors are not swallowed).

## 6. Model routing

`Chat` resolves the model once via `resolve_model(self.config, self.domain, route,
self.config.model)` and passes it to `run()`. For `complex`, that is `model_high`
(per the Model Routing spec). All three stages share it.

## 7. Testing

New `tests/test_orchestrator.py`:

- **Normal path:** a FakeClient returns planner JSON (`{"tasks": [{"title": "t1",
  "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}`), two worker
  outputs, then the aggregated answer. Assert call count and order (planner, worker
  1, worker 2, aggregator), and that `model` passed to every call is `model_high`.
- **Planner invalid JSON** → degrades to a single direct-answer call; assert exactly
  1 LLM call and its model.
- **Planner valid JSON but empty `tasks`** → same degradation.
- **Worker returns empty string** → aggregator still called with the collected
  outputs (including the empty entry).

Update `tests/test_router.py`:

- `test_route_complex_gated_to_unsupported` becomes `test_route_complex_gated_sets_orchestrate`:
  strategy stays `"debugging"`, `orchestrate is True`.
- Other cases assert `orchestrate is False`.

Update `tests/test_chat.py`:

- `test_respond_unsupported_complex` becomes `test_respond_orchestrates_complex`:
  FakeClient returns the classification JSON, planner JSON, two worker outputs,
  and the final answer; assert `kind == "answer"`, `text == final answer`, and
  `history` updated.
- The `_domain()` fixture in `tests/test_chat.py` constructs `DomainConfig` directly
  (not via `load_domain_config`), so its `prompts` dict is a plain dict. **Decision:**
  keep `"unsupported_complex": "Needs orchestrator."` in the fixture — it is harmless
  and avoids churn. Only the `test_respond_unsupported_complex` test is replaced.

`tests/test_config.py`: `load_domain_config` (config.py:223) reads
`unsupported_complex.md` eagerly and `test_load_domain_config_missing_prompt`
(tests/test_config.py:322) asserts a missing `unsupported_complex.md` raises
`ConfigError`. **Decision:** keep `unsupported_complex.md` loading and the
`prompts["unsupported_complex"]` key for backward compatibility of existing domain
dirs; `Chat` simply never returns it. No change to `load_domain_config` or its
tests. The `unsupported_complex` key stays in the prompts dict but is unused at
runtime.

Run `uv run pytest -q` (expect all green).

## 8. Docs

- Keep `domain/software_engineering/prompts/unsupported_complex.md` (do not delete):
  `load_domain_config` reads it eagerly and a test asserts it is required. It is
  simply unused at runtime after this change.
- Update `README.md`: replace any mention of complex tasks being unsupported with a
  short Orchestrator description.
- Add this spec; then an implementation plan under `docs/superpowers/plans/`.

## 9. Success criteria

1. `uv run pytest -q` all green.
2. `complex` + gated strategy → Orchestrator (Planner → Workers → Aggregator), not
   the `unsupported` reply.
3. Strategy prompt context flows into Planner, Workers, and Aggregator.
4. Orchestrator stages use `model_high` (complex tier).
5. Planner failure degrades to a single direct answer — never a fake plan or crash.
6. Non-complex behavior unchanged (reject, direct processors, strategy routing).
7. `TODO` comments mark future Evaluator/Optimizer and concurrent workers.