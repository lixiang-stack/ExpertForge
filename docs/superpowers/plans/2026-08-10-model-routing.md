# Model Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route answer generation to a low-end model for `simple` questions and a high-end model for `medium`/`complex` questions, preserving per-strategy model override, and derive the classifier model as `model_low or model` instead of a standalone config entry.

**Architecture:** `AgentConfig` gains optional `model_low`/`model_high` (fall back to `model`). A new pure module `agent/model_router.py` exposes `resolve_model(config, domain, route, default)` with precedence: `strategy.model` → complexity tier (`simple`→low, else→high) → `default`. `agent/chat.py` calls it where it currently resolves `model` itself. `classifier_model` becomes derived in `load_config` (`model_low or model`). `Router`, `RouteResult`, processors, repl, agent_cli are untouched.

**Tech Stack:** Python 3.10+, pyyaml, pytest, uv (all already in use).

## Global Constraints

- `git` working tree starts on `main`; do NOT modify `domain/software_engineering/*` files.
- `uv run pytest -q` must pass (currently 62 tests) after every task.
- `AgentConfig` field order in `load_config` construction: `base_url, model, classifier_model, domain_dir, model_low, model_high` — `classifier_model` remains required (non-None) on the dataclass; `model_low`/`model_high` default to `None`.
- `resolve_model(config: AgentConfig, domain: DomainConfig, route: RouteResult, default: str) -> str` — pure function; precedence: (1) `strategy.model`, (2) `model_low`/`model_high` by complexity, (3) `default`.
- Legacy explicit `classifier_model` key in config JSON is IGNORED (no error, no effect).
- `RouteResult` and `Router.route()` are unchanged. `Chat`, `repl`, `agent_cli`, processors unchanged in code.
- All example-domain `tests/*` construct `AgentConfig` with explicit keyword args — adding fields with defaults is safe.

---

### Task 1: Config support for `model_low` / `model_high` and derived `classifier_model`

**Files:**
- Modify: `agent/config.py` (dataclass + `load_config`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `AgentConfig` and `load_config`.
- Produces:
  - `AgentConfig` fields: `model_low: str | None = None`, `model_high: str | None = None`.
  - `load_config` sets `classifier_model = model_low or model`, `model_low`/`model_high` as `None` when missing/empty.

- [ ] **Step 1: Write the failing tests** — in `tests/test_config.py`, update `test_load_config_basic` (remove the legacy `classifier_model` key and its assertion; the derived value replaces it) and add:

```python
def test_load_config_model_tiers(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "model_low": "low-a",
        "model_high": "high-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.model_low == "low-a"
    assert cfg.model_high == "high-a"


def test_model_tiers_empty_string_become_none(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "model_low": "",
        "model_high": "",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.model_low is None
    assert cfg.model_high is None


def test_model_tiers_absent_become_none(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.model_low is None
    assert cfg.model_high is None


def test_classifier_model_derives_from_model_low(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "model_low": "low-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "low-a"


def test_legacy_classifier_model_entry_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "classifier_model": "legacy-a",
        "model_low": "low-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.classifier_model == "low-a"  # legacy key ignored
```

Also update `test_load_config_basic` — drop `"classifier_model": "classifier-a"` from the input dict and change its assertion to `assert cfg.classifier_model == "model-a"` (no `model_low` present → derives from `model`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `AgentConfig` has no attributes `model_low`/`model_high`.

- [ ] **Step 3: Implement** — in `agent/config.py`, extend the dataclass:

```python
@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_dir: str
    model_low: str | None = None
    model_high: str | None = None
```

In `load_config`, replace the `classifier_model = raw.get("classifier_model") or model` line with:

```python
    model_low = raw.get("model_low")
    model_high = raw.get("model_high")
    model_low = model_low if isinstance(model_low, str) and model_low else None
    model_high = model_high if isinstance(model_high, str) and model_high else None

    classifier_model = model_low or model
```

and add `model_low=model_low, model_high=model_high` to the returned `AgentConfig(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: add model_low/model_high config tiers; derive classifier_model"
```

---

### Task 2: New `agent/model_router.py` with `resolve_model`

**Files:**
- Create: `agent/model_router.py`
- Create: `tests/test_model_router.py`
- Test: `tests/test_model_router.py`

**Interfaces:**
- Consumes: `AgentConfig`/`DomainConfig` from `agent/config.py`; `RouteResult` from `agent/router.py`.
- Produces: `resolve_model(config: AgentConfig, domain: DomainConfig, route: RouteResult, default: str) -> str`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_model_router.py`:

```python
from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.model_router import resolve_model
from agent.router import RouteResult


def _config(model_low=None, model_high=None):
    return AgentConfig(
        base_url="https://x", model="m", classifier_model="c",
        domain_dir="d", model_low=model_low, model_high=model_high,
    )


def _domain(strategy_model=None):
    strategies = {
        "direct": StrategyDef("direct", model=strategy_model),
        "teaching": StrategyDef("teaching"),
    }
    return DomainConfig(
        name="sw",
        description="d",
        out_of_domain_reply="Out.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies=strategies,
        prompts={},
    )


def _route(strategy="direct", complexity=None):
    return RouteResult(
        in_domain=True, strategy=strategy,
        intent="faq", complexity=complexity,
    )


def test_simple_uses_model_low():
    domain = _domain()
    result = resolve_model(_config("low-a", "high-a"), domain, _route(complexity="simple"), "default")
    assert result == "low-a"


def test_simple_missing_model_low_falls_back_to_default():
    domain = _domain()
    result = resolve_model(_config(), _domain(), _route(complexity="simple"), "default")
    assert result == "default"


def test_medium_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _domain(), _route(complexity="medium"), "default")
    assert result == "high-a"


def test_complex_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _domain(), _route(complexity="complex"), "default")
    assert result == "high-a"


def test_none_complexity_uses_model_high():
    result = resolve_model(_config("low-a", "high-a"), _domain(), _route(complexity=None), "default")
    assert result == "high-a"


def test_medium_missing_model_high_falls_back_to_default():
    result = resolve_model(_config("low-a", None), _domain(), _route(complexity="medium"), "default")
    assert result == "default"


def test_strategy_model_overrides_complexity():
    result = resolve_model(
        _config("low-a", "high-a"), _domain(strategy_model="strat-a"),
        _route(complexity="simple"), "default",
    )
    assert result == "strat-a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_router.py -v`
Expected: `ModuleNotFoundError: No module named 'agent.model_router'`.

- [ ] **Step 3: Implement** — create `agent/model_router.py`:

```python
from __future__ import annotations

from .config import AgentConfig, DomainConfig
from .router import RouteResult


def resolve_model(
    config: AgentConfig,
    domain: DomainConfig,
    route: RouteResult,
    default: str,
) -> str:
    strategy_def = domain.strategies.get(route.strategy)
    if strategy_def and strategy_def.model:
        return strategy_def.model
    if route.complexity == "simple":
        return config.model_low or default
    return config.model_high or default
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/model_router.py tests/test_model_router.py
git commit -m "feat: add resolve_model complexity-based model routing"
```

---

### Task 3: Wire `resolve_model` into `Chat`

**Files:**
- Modify: `agent/chat.py:22,40-43`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `resolve_model` from Task 2.
- Produces: unchanged `respond(question) -> ChatResponse`; answer calls now use the resolved model.

- [ ] **Step 1: Write the failing test** — in `tests/test_chat.py`, update `FakeClient` to record the last model, and add a routing assertion:

```python
class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.models = []

    def chat_completion(
        self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None
    ):
        self.models.append(model)
        return self.responses.pop(0)
```

Add:

```python
def test_respond_uses_complexity_routed_model():
    client = FakeClient([
        '{"in_domain": true, "intent": "faq", "complexity": "simple", "reason": "ok"}',
        "the answer",
    ])
    chat = Chat(client, AgentConfig(base_url="https://x", model="m", classifier_model="cm",
                                    domain_dir="d", model_low="low-a", model_high="high-a"),
                _domain())
    resp = chat.respond("what is defer")
    assert resp.kind == "answer"
    # first call: classification (model=cm); second call: generator (model=low)
    assert client.models == ["cm", "low-a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat.py -v`
Expected: FAIL — `client.models` shows the generator call used the default model instead of `model_low` (Chat still uses its own if/else).

- [ ] **Step 3: Implement** — in `agent/chat.py`, add the import and replace the model resolution:

```python
from .model_router import resolve_model
```

Replace (lines 40-43):

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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chat.py -v`
Expected: PASS.

- [ ] **Step 5: Full regression**

Run: `uv run pytest -q`
Expected: all pass (62+ total).

- [ ] **Step 6: Commit**

```bash
git add agent/chat.py tests/test_chat.py
git commit -m "feat: route generator model by complexity via resolve_model"
```

---

### Task 4: Example config, README, and final review

**Files:**
- Modify: `config.example.json`
- Modify: `README.md:24-28`
- Test: none new (docs/config only)

**Interfaces:**
- Consumes: nothing new.

- [ ] **Step 1: Update `config.example.json`** — replace `classifier_model` with the low/high tiers:

```json
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "model_low": "deepseek-v4-flash",
  "model_high": "deepseek-pro",
  "domain_dir": "domain/software_engineering"
}
```

- [ ] **Step 2: Update `README.md` bullets (lines 24-28)**

Replace the `model` and `classifier_model` bullets:

```markdown
- `model`: the base model, used for answers unless overridden below.
- `model_low`: low-cost model tier for `simple` questions (falls back to `model`).
- `model_high`: high-capability model tier for `medium`/`complex` questions (falls back to `model`).
```

`classifier_model` is no longer configured — classification uses `model_low` (falling back to `model`).

- [ ] **Step 3: Update the Domain directory section** — the `strategies.yaml` bullet may already mention a per-strategy `model`; verify no other README references to `classifier_model` remain.

Run: `rg -n "classifier_model" README.md agent/ tests/`
Expected: only `agent/config.py` (dataclass + `load_config` derivation) should reference `classifier_model`; no test/README/dir references to a legacy key.

- [ ] **Step 4: Full verification**

```bash
uv run pytest -q
env -u AGENT_API_KEY uv run python -m agent --ask "What is Go defer?"
```

Expected: all tests pass; the no-key smoke prints the `AGENT_API_KEY` error and exits 1.

- [ ] **Step 5: Commit**

```bash
git add config.example.json README.md
git commit -m "docs: model tiers in example config and README"
```

- [ ] **Step 6: Review the plan against the spec**

  - §2 configuration fields `model_low`/`model_high`, fallback to `model`, legacy `classifier_model` ignored → Task 1, Task 4.
  - §3 `resolve_model` pure function and precedence → Task 2.
  - §4 chat.py integration, strategy.model priority retained → Task 3.
  - §5 classifier_model derivation `model_low or model`, router unchanged → Task 1.
  - §6 testing cases (`test_model_router.py`, `test_chat.py`, `test_config.py`) → Tasks 1-3.
  - §7 success criteria 1-6 → covered by the four tasks.