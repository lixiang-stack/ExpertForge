# Model Routing Design (Complexity → Model Tier)

**Date:** 2026-08-10
**Status:** Design agreed; pending user review
**Source requirement:** `draft_v1.md` sections 5.1, 12, 19 (Routing), 12.1, 12.2

## 1. Goal

Bring the answer-generation model selection in line with `draft_v1.md` §12 and §19:
route a user question to a **low-end (cheap) model** when it is `simple`, and to a
**high-end (capable) model** when it is `medium` or `complex`, while preserving the
existing per-strategy model override.

The classification call keeps using `classifier_model` (unchanged). The orchestrator
for complex tasks (§13) is a separate, later iteration and stays out of scope here.

## 2. Configuration

`AgentConfig` gains two optional fields:

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "classifier_model": "deepseek-v4-flash",
  "model_low": "deepseek-v4-flash",
  "model_high": "deepseek-reasoner"
}
```

- `model_low` — low-end tier. When absent, falls back to `model`.
- `model_high` — high-end tier. When absent, falls back to `model`.
- `model` — the default fallback tier (existing field, unchanged).
- `classifier_model` — classification model (existing field, unchanged).

`agent/config.py` reads `model_low`/`model_high` and stores `None` when the value is
missing or an empty string. No new required fields: existing `config.json` files keep
working.

`config.example.json` gains both fields with illustrative values and a short comment.

## 3. New module: `agent/model_router.py`

Pure function:

```python
def resolve_model(
    config: AgentConfig,
    domain: DomainConfig,
    route: RouteResult,
    default: str,
) -> str
```

Resolution order (first match wins):

1. `domain.strategies[route.strategy].model` — explicit strategy-level override wins.
   (Existing priority retained.)
2. `route.complexity == "simple"` → `config.model_low or default`.
3. Any other complexity (`medium`, `complex`, `None`) → `config.model_high or default`.
4. Fallback to `default`.

`default` is passed in by `Chat` as `config.model` so the resolver stays a pure
function with no hidden config reads.

## 4. Router integration (agent/chat.py)

Replace the current per-strategy model resolution block:

```python
model = self.config.model
strategy_def = self.domain.strategies.get(route.strategy)
if strategy_def and strategy_def.model:
    model = strategy_def.model
```

with:

```python
model = resolve_model(self.config, self.domain, route, self.config.model)
```

Behavior for existing configs that already set `strategy.model` is unchanged.
`agent/router.py` is untouched; `RouteResult` is untouched.

## 5. Testing

New `tests/test_model_router.py`:

- `simple` + `model_low` set → returns `model_low`.
- `simple` + `model_low` missing → falls back to `default`.
- `medium`, `complex`, `complexity=None` → `model_high` when set.
- `medium`/`complex` + `model_high` missing → falls back to `default`.
- `strategy.model` set → wins over both complexity tiers.

Update `tests/test_chat.py`: FakeClient records the model argument; assert
`resolve_model` output flows into `processor.process`. Existing behavioral
assertions unchanged.

`tests/test_config.py`: add loading cases for `model_low`/`model_high` (present,
missing, empty string → None).

Run `uv run pytest -q` (all green).

## 6. Docs

- Update `README.md` config table with `model_low`/`model_high`.
- Add this spec (done).

## 7. Success criteria

1. `uv run pytest -q` all green.
2. `simple` → low-end tier; `medium`/`complex` → high-end tier, with `model` fallback.
3. Strategy-level `model` override retains its priority.
4. Existing `config.json` files (without the new fields) work unchanged.
5. Classification still uses `classifier_model`; classification behavior unchanged.