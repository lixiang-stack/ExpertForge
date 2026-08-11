# Domain-Agnostic Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the strategy/processor layer from software-engineering so any expert domain (declared purely in config data) works with zero code changes.

**Architecture:** Delete the `agent/processors/` package (5 SE-hardcoded subclasses + whitelist registry + hardcoded `DEFAULT_STRATEGY`). Replace with a single domain-agnostic `agent/strategy.py` module that builds a generic `Strategy` from `domain.strategies` and self-contained prompt files. The default strategy becomes a `default: true` flag in `strategies.yaml`, parsed and validated at config load.

**Tech Stack:** Python 3, PyYAML, pytest (all already in the repo).

## Global Constraints

- All tactic prompt/copy and error messages stay in English (existing convention).
- Behavior of the software-engineering domain must be unchanged: same strategies, same answer structures, same orchestration.
- `unsupported_complex.md` remains loaded eagerly (backward compatibility) and unused at runtime.
- No new dependencies; `uv run pytest -q` must stay green.
- Prompt files are **fully self-contained**: no `{name}`/`{description}`/`{structure}` placeholders anywhere; `Strategy.build_system_prompt()` returns the prompt text verbatim.
- The `DomainConfig` dataclass gains a `default_strategy: str` field (positional field list changes — see Task 2).

---

### Task 1: Create `agent/strategy.py` (generic Strategy + build_registry)

**Files:**
- Create: `agent/strategy.py`

**Interfaces:**
- Produces: `class Strategy` with `__init__(strategy_id: str, prompt_template: str)`, `build_system_prompt() -> str` (returns the template verbatim), `build_messages(history, question, *, max_turns=20) -> list[dict]`, `process(client, question, history, *, model=None) -> str`; and `build_registry(domain: DomainConfig) -> dict[str, Strategy]` that iterates `domain.strategies`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_strategy.py`:

```python
from agent.config import DomainConfig, IntentDef, StrategyDef
from agent.strategy import Strategy, build_registry


def _prompts():
    return {
        "direct": "Direct answer prompt.",
        "teaching": "Teaching prompt.",
    }


def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out.",
        intents={},
        intent_mapping={},
        strategies={"direct": StrategyDef("direct", default=True),
                    "teaching": StrategyDef("teaching", complexity_gate=True)},
        default_strategy="direct",
        prompts=_prompts(),
    )


class FakeClient:
    def __init__(self, text="answer"):
        self.text = text
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False):
        self.calls.append((messages, model))
        return self.text


def test_build_registry_builds_each_strategy():
    registry = build_registry(_domain())
    assert set(registry) == {"direct", "teaching"}
    assert isinstance(registry["teaching"], Strategy)
    assert registry["direct"].build_system_prompt() == "Direct answer prompt."


def test_build_system_prompt_returns_template_verbatim():
    p = Strategy("direct", "You are an agent in the X domain.\n- Approach\n- Code snippet")
    prompt = p.build_system_prompt()
    assert prompt == "You are an agent in the X domain.\n- Approach\n- Code snippet"
    assert "{name}" not in prompt
    assert "{description}" not in prompt
    assert "{structure}" not in prompt


def test_process_single_call_returns_string():
    client = FakeClient("answer")
    p = Strategy("direct", "You are an agent in the X domain.")
    out = p.process(client, "q", [("旧问", "旧答")])
    assert out == "answer"
    assert len(client.calls) == 1
    messages, model = client.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "旧问"}
    assert messages[2] == {"role": "assistant", "content": "旧答"}
    assert messages[-1]["content"] == "q"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.strategy'`

- [ ] **Step 3: Write minimal implementation**

Create `agent/strategy.py`:

```python
from __future__ import annotations

from .config import DomainConfig


class Strategy:
    def __init__(self, strategy_id: str, prompt_template: str):
        self.strategy_id = strategy_id
        self.prompt_template = prompt_template

    def build_system_prompt(self) -> str:
        return self.prompt_template

    def build_messages(
        self,
        history: list[tuple[str, str]],
        question: str,
        *,
        max_turns: int = 20,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.build_system_prompt()}]
        for user_text, assistant_text in history[-max_turns:]:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": question})
        return messages

    def process(self, client, question: str, history: list[tuple[str, str]], *, model: str | None = None) -> str:
        return client.chat_completion(self.build_messages(history, question), model=model)


def build_registry(domain: DomainConfig) -> dict[str, Strategy]:
    return {
        sid: Strategy(sid, domain.prompts[sid])
        for sid in domain.strategies
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_strategy.py -v`
Expected: PASS (note: needs Task 2's `DomainConfig.default_strategy` field to exist; if it fails on `default_strategy`, proceed — Task 2 adds it, then re-run.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_strategy.py agent/strategy.py
git commit -m "feat: add generic domain-agnostic Strategy module"
```

---

### Task 2: Update the config schema (`config.py`) — `StrategyDef.default`, `DomainConfig.default_strategy`, validation

**Files:**
- Modify: `agent/config.py:104-119` (`StrategyDef`), `agent/config.py:111-119` (`DomainConfig`), `agent/config.py:195-223` (strategies parsing)
- Modify: `tests/test_config.py` (fixtures + new validation tests)

**Interfaces:**
- Consumes: nothing.
- Produces: `StrategyDef` gains field `default: bool = False`; `DomainConfig` gains field `default_strategy: str`; `load_domain_config` raises `ConfigError` when not exactly one strategy has `default: true`, and sets `DomainConfig.default_strategy` to that id.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def _write_domain_with_default(tmp_path, strategies_yaml):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "x", "description": "d",
    }), encoding="utf-8")
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text(strategies_yaml, encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    return str(base)


def test_load_domain_config_resolves_default_strategy(tmp_path):
    domain = load_domain_config(_write_domain_with_default(
        tmp_path, "direct:\n  default: true\n"))
    assert domain.default_strategy == "direct"
    assert domain.strategies["direct"].default is True


def test_load_domain_config_zero_defaults_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_write_domain_with_default(tmp_path, "direct:\n"))


def test_load_domain_config_multiple_defaults_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_write_domain_with_default(
            tmp_path, "direct:\n  default: true\nteaching:\n  default: true\n"))
```

Also update the existing fixture `_write_domain` (tests/test_config.py:214-241) so its `strategies.yaml` has a `default: true` on `direct`, and its prompt files drop the `{structure}`/`{name}` placeholders (they are now fully self-contained):

```python
    (base / "strategies.yaml").write_text(
        "teaching:\n  complexity_gate: true\ndirect:\n  model: model-direct\n  default: true\n",
        encoding="utf-8",
    )
    (base / "prompts" / "teaching.md").write_text(
        "teach self-contained", encoding="utf-8"
    )
    (base / "prompts" / "direct.md").write_text(
        "direct self-contained", encoding="utf-8"
    )
```

And update `test_load_domain_config_basic` (line 244-257) to assert `domain.default_strategy == "direct"` and adjust the prompt assertions (`"teach {name}" in domain.prompts["teaching"]` becomes `"teach self-contained" in domain.prompts["teaching"]`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'default'` and `AttributeError: 'DomainConfig' object has no attribute 'default_strategy'`.

- [ ] **Step 3: Write minimal implementation**

In `agent/config.py`, change the `StrategyDef` dataclass (line 104-109):

```python
@dataclass
class StrategyDef:
    id: str
    model: str | None = None
    complexity_gate: bool = False
    default: bool = False
```

Change `DomainConfig` (line 111-119) to add `default_strategy` before `prompts`:

```python
@dataclass
class DomainConfig:
    name: str
    description: str
    out_of_domain_reply: str
    intents: dict[str, IntentDef]
    intent_mapping: dict[str, str]
    strategies: dict[str, StrategyDef]
    default_strategy: str
    prompts: dict[str, str]
```

Update the strategies parsing loop (lines 200-210) to read `default`:

```python
    configured_default = None
    for sid, item in strategies_data.items():
        if isinstance(item, dict):
            model = item.get("model")
            strategies[sid] = StrategyDef(
                id=sid,
                model=model if isinstance(model, str) and model else None,
                complexity_gate=bool(item.get("complexity_gate", False)),
                default=bool(item.get("default", False)),
            )
            if item.get("default"):
                configured_default = sid
        else:
            strategies[sid] = StrategyDef(id=sid)
```

After the mapping-validation loop (after line 217), add default resolution and validation:

```python
    if configured_default is None:
        raise ConfigError(
            f"Exactly one strategy in {base / 'strategies.yaml'} must have default: true"
        )
    if sum(1 for sdef in strategies.values() if sdef.default) != 1:
        raise ConfigError(
            f"Only one strategy in {base / 'strategies.yaml'} may have default: true"
        )
```

Then add `default_strategy=configured_default` to the returned `DomainConfig(...)` (line 225-232).

- [ ] **Step 4: Run tests to verify they pass and fix any cascade**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS. Then run `uv run pytest tests/test_strategy.py -v` — must now PASS too.

- [ ] **Step 5: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: make default strategy config-driven with validation"
```

---

### Task 3: Wire `Router` to use `domain.default_strategy`

**Files:**
- Modify: `agent/router.py:9,36`
- Modify: `tests/test_router.py` (fixtures)

**Interfaces:**
- Consumes: `DomainConfig.default_strategy` (from Task 2).
- Produces: `Router.route()` falls back to `domain.default_strategy` for unmapped intents; `DEFAULT_STRATEGY` constant removed.

- [ ] **Step 1: Write the failing test**

In `tests/test_router.py`, update the `_domain()` fixture so `direct` is the default, and rename `test_route_unknown_intent_defaults_to_direct` → `test_route_unknown_intent_falls_back_to_default` with an assertion against the flagged default (change the fixture's default strategy to `teaching` to prove it is not hardcoded to `direct`):

```python
def _domain(**overrides):
    default = {
        "name": "软件工程",
        "description": "sw",
        "out_of_domain_reply": "Out.",
        "intents": {
            "concept_explain": IntentDef("concept_explain", "explain"),
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
            "architecture_design": IntentDef("architecture_design", "arch"),
        },
        "intent_mapping": {
            "concept_explain": "teaching",
            "faq": "direct",
            "troubleshooting": "debugging",
            "architecture_design": "analysis",
        },
        "strategies": {
            "teaching": StrategyDef("teaching", complexity_gate=True, default=True),
            "direct": StrategyDef("direct"),
            "debugging": StrategyDef("debugging", complexity_gate=True),
            "analysis": StrategyDef("analysis", complexity_gate=True),
        },
        "default_strategy": "teaching",
        "prompts": {},
    }
    default.update(overrides)
    return DomainConfig(**default)
```

Replace the old `test_route_unknown_intent_defaults_to_direct`:

```python
def test_route_unknown_intent_falls_back_to_default():
    client = FakeClient([_combined(True, "bogus", "simple")])  # intent bogus → validation sets None
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "teaching"
    assert result.orchestrate is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_router.py::test_route_unknown_intent_falls_back_to_default -v`
Expected: FAIL — strategy is `"direct"` because `DEFAULT_STRATEGY` is hardcoded, but expected `"teaching"`.

- [ ] **Step 3: Write minimal implementation**

In `agent/router.py`, replace `DEFAULT_STRATEGY = "direct"` and its use:

- Delete `DEFAULT_STRATEGY = "direct"` (line 9).
- Change line 36 from:
  ```python
  strategy = self.domain.intent_mapping.get(intent_id, DEFAULT_STRATEGY)
  ```
  to:
  ```python
  strategy = self.domain.intent_mapping.get(intent_id, self.domain.default_strategy)
  ```

- [ ] **Step 4: Run the full router + looping tests to verify and fix cascades**

Run: `uv run pytest tests/test_router.py -v`
Expected: PASS — including `test_route_unknown_intent_falls_back_to_default` (strategy `teaching`), `test_route_in_domain_maps_strategy_and_keeps_fields` (`faq`→`direct` unchanged), and `test_route_complex_ungated_strategy_stays` (`faq` still maps to un-gated `direct`, `orchestrate is False`).

Then run `uv run pytest tests/test_chat.py tests/test_repl.py tests/test_orchestrator.py -v`:
Expected: this may FAIL because those files construct `DomainConfig` directly and now need the `default_strategy` field — that field is fixed in Task 4, so **defer** those failures; only `test_router.py` must PASS now.

- [ ] **Step 5: Commit**

```bash
git add agent/router.py tests/test_router.py
git commit -m "feat: route unmapped intents to config-driven default strategy"
```

---

### Task 4: Update callers (`chat.py`, `orchestrator.py`) and delete `agent/processors/`

**Files:**
- Modify: `agent/chat.py:9,25,36-38`
- Modify: `agent/orchestrator.py:8,74,77`
- Delete: `agent/processors/` (entire package)
- Modify: `tests/test_chat.py`, `tests/test_repl.py`, `tests/test_orchestrator.py` (fixtures + imports), delete `tests/test_processors.py`

**Interfaces:**
- Consumes: `build_registry` from `agent.strategy`; `DomainConfig.default_strategy` field.
- Produces: `Chat.respond()` and `Orchestrator` no longer reference `agent.processors`; the `"No processor for strategy"` branch is removed.

- [ ] **Step 1: Write the failing test (update fixtures)**

Update `tests/test_chat.py` fixture (lines 5-24) to add `default_strategy` and drop `{structure}`:

```python
def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies={
            "direct": StrategyDef("direct", default=True),
            "debugging": StrategyDef("debugging", complexity_gate=True),
        },
        default_strategy="direct",
        prompts={
            "direct": "Direct answer prompt.",
            "debugging": "Debugging prompt.",
            "unsupported_complex": "Needs orchestrator.",
        },
    )
```

Update `tests/test_repl.py` fixture (lines 10-22) the same way (`default_strategy="direct"`, self-contained prompts):

```python
def _domain():
    return DomainConfig(
        name="软件工程",
        description="sw",
        out_of_domain_reply="Out of domain.",
        intents={"faq": IntentDef("faq", "quick")},
        intent_mapping={"faq": "direct"},
        strategies={"direct": StrategyDef("direct", default=True)},
        default_strategy="direct",
        prompts={
            "direct": "Direct answer prompt.",
            "unsupported_complex": "unsupported",
        },
    )
```

Update `tests/test_orchestrator.py` fixture (lines 6-17) the same way (`default_strategy="debugging"`, self-contained prompt):

```python
def _domain():
    return DomainConfig(
        name="sw",
        description="software engineering",
        out_of_domain_reply="Out.",
        intents={"troubleshooting": IntentDef("troubleshooting", "debug")},
        intent_mapping={"troubleshooting": "debugging"},
        strategies={"debugging": StrategyDef("debugging", complexity_gate=True, default=True)},
        default_strategy="debugging",
        prompts={
            "debugging": "Debugging system prompt.",
        },
    )
```

Delete `tests/test_processors.py` (replaced by `test_strategy.py` from Task 1).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat.py -v`
Expected: FAIL with `TypeError: DomainConfig.__init__() missing 1 required positional argument: 'default_strategy'` (from `chat.py` constructing processors via the old import).

- [ ] **Step 3: Write minimal implementation**

In `agent/chat.py`:
- Change line 9 import from `from .processors.registry import build_registry` to `from .strategy import build_registry`.
- Delete lines 36-38 (the `"No processor for strategy"` branch), so `respond()` becomes:

```python
    def respond(self, question: str) -> ChatResponse:
        route = self.router.route(question)
        if not route.in_domain:
            text = self.domain.out_of_domain_reply
            if route.reject_reason:
                text += f" ({route.reject_reason})"
            return ChatResponse(kind="reject", text=text)
        processor = self.processors[route.strategy]
        model = resolve_model(self.config, self.domain, route, self.config.model)
        if route.orchestrate:
            answer = self.orchestrator.run(question, route, model)
        else:
            answer = processor.process(self.client, question, self.history, model=model)
        self.history.append((question, answer))
        return ChatResponse(kind="answer", text=answer)
```

In `agent/orchestrator.py`:
- Change line 8 import from `from .processors.registry import build_registry` to `from .strategy import build_registry`. No other change.

Delete the package: `rm -r agent/processors`.

- [ ] **Step 4: Run the full suite to verify and fix cascade**

Run: `uv run pytest tests/test_chat.py tests/test_repl.py tests/test_orchestrator.py -v`
Expected: PASS. Then run `uv run pytest -q` — fix any remaining fixture/import errors (e.g. other direct `DomainConfig` constructions needing `default_strategy`).

- [ ] **Step 5: Commit**

```bash
git add -A agent/chat.py agent/orchestrator.py tests/test_chat.py tests/test_repl.py tests/test_orchestrator.py
git rm -r agent/processors tests/test_processors.py
git commit -m "refactor: switch to generic Strategy, remove SE-hardcoded processors"
```

---

### Task 5: Migrate the software-engineering domain config

**Files:**
- Modify: `domain/software_engineering/strategies.yaml`
- Modify: `domain/software_engineering/prompts/{direct,teaching,debugging,analysis,code_snippet}.md`

**Interfaces:**
- Consumes: new fully self-contained prompt format (no placeholders at all); `default: true` flag.
- Produces: migrated SE domain config matching the new data model.
- Note: each prompt file embeds the domain's literal name ("Software Engineering") and description ("Covers software design, development, testing, operations, and performance optimization.") from `domain/software_engineering/domain.json`.

- [ ] **Step 1: Update `strategies.yaml` to add the default flag**

Write `domain/software_engineering/strategies.yaml`:

```yaml
direct:
  default: true
  complexity_gate: false
teaching:
  complexity_gate: true
debugging:
  complexity_gate: true
analysis:
  complexity_gate: true
code_snippet:
  complexity_gate: true
```

- [ ] **Step 2: Migrate each prompt file to be fully self-contained**

`direct.md`:

```markdown
You are an expert Agent in the Software Engineering domain.

Covers software design, development, testing, operations, and performance optimization.

Answering requirements:
- Answer authoritatively and professionally.
- Adjust the structure of your answer to fit each question; do not force a fixed template.
- Only answer questions within this domain.
```

`teaching.md`:

```markdown
You are an expert Agent in the Software Engineering domain.

Covers software design, development, testing, operations, and performance optimization.

Answer in this structure:
- Concept
- Why it is designed this way
- How it works
- Concrete example
- Common misconceptions
- Summary

Answering requirements:
- Answer authoritatively and professionally.
- Explain the topic thoroughly and insightfully.
- Only answer questions within this domain.
```

`debugging.md`:

```markdown
You are an expert Agent in the Software Engineering domain.

Covers software design, development, testing, operations, and performance optimization.

Answer in this structure:
- Problem analysis
- Possible causes
- Verification steps
- Fix suggestions
- Best practices

Answering requirements:
- Answer authoritatively and professionally.
- Be systematic: analyze before proposing fixes.
- Only answer questions within this domain.
```

`analysis.md`:

```markdown
You are an expert Agent in the Software Engineering domain.

Covers software design, development, testing, operations, and performance optimization.

Answer in this structure:
- Comparison dimensions
- Key differences
- Trade-offs
- Recommendation

Answering requirements:
- Answer authoritatively and professionally.
- Compare objectively and point out trade-offs.
- Only answer questions within this domain.
```

`code_snippet.md`:

```markdown
You are an expert Agent in the Software Engineering domain.

Covers software design, development, testing, operations, and performance optimization.

Answer in this structure:
- Approach
- Code snippet
- Key points and caveats
- How to extend or adapt it

Answering requirements:
- Answer authoritatively and professionally.
- Produce short, idiomatic code fragments focused on the question.
- Keep the snippet self-contained and explain the reasoning inline.
- Only answer questions within this domain.
```

- [ ] **Step 3: Verify the migrated domain loads**

Run:
```bash
uv run python -c "from agent.config import load_domain_config; d=load_domain_config('domain/software_engineering'); assert d.default_strategy=='direct'; assert '{name}' not in d.prompts['teaching'] and '{description}' not in d.prompts['teaching']; print('default:', d.default_strategy); print('teaching structure ok:', 'Concept' in d.prompts['teaching'])"
```
Expected: prints `default: direct` and `teaching structure ok: True`.

- [ ] **Step 4: Commit**

```bash
git add domain/software_engineering/strategies.yaml domain/software_engineering/prompts
git commit -m "refactor: migrate SE domain to self-contained prompts and default flag"
```

---

### Task 6: Add the dogfood (non-SE domain) test fixture and full-path test

**Files:**
- Create: `tests/test_domain_agnostic.py`

**Interfaces:**
- Consumes: `load_domain_config` (via a temp `finance` domain dir), `Chat`, and a `FakeClient`.
- Produces: end-to-end proof that a fully custom strategy id (no `direct`/`teaching`/etc.) routes and answers with zero code changes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_domain_agnostic.py`:

```python
import json

from agent.chat import Chat
from agent.config import AgentConfig, load_domain_config


def _write_finance_domain(tmp_path):
    base = tmp_path / "finance"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(json.dumps({
        "name": "Finance Advice",
        "description": "Personal finance, investment, and risk guidance.",
        "out_of_domain_reply": "Out of finance domain.",
    }), encoding="utf-8")
    (base / "intents.yaml").write_text(
        "- id: portfolio_review\n  description: review an investment portfolio\n"
        "- id: risk_check\n  description: assess financial risk\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text(
        "portfolio_review: advise\nrisk_check: risk_assessment\n", encoding="utf-8"
    )
    (base / "strategies.yaml").write_text(
        "advise:\n  default: true\n  complexity_gate: true\n"
        "risk_assessment:\n  complexity_gate: true\n",
        encoding="utf-8",
    )
    (base / "prompts" / "advise.md").write_text(
        "You are a finance advisor in the Finance Advice domain.\n\n"
        "Personal finance, investment, and risk guidance.\n\n"
        "Structure:\n- Summary\n- Options\n- Recommendation\n- Risks\n",
        encoding="utf-8",
    )
    (base / "prompts" / "risk_assessment.md").write_text(
        "You are a risk assessor in the Finance Advice domain.\n\n"
        "Personal finance, investment, and risk guidance.\n\n"
        "Structure:\n- Risk factors\n- Likelihood\n- Mitigations\n",
        encoding="utf-8",
    )
    (base / "prompts" / "unsupported_complex.md").write_text("unsupported", encoding="utf-8")
    return str(base)


def _config(domain_dir):
    return AgentConfig(base_url="https://x", model="m", classifier_model="m", domain_dir=domain_dir)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_custom_strategy_answers_without_code_changes(tmp_path):
    domain = load_domain_config(_write_finance_domain(tmp_path))
    assert domain.default_strategy == "advise"
    client = FakeClient([
        '{"in_domain": true, "intent": "portfolio_review", "complexity": "simple", "reason": "ok"}',
        "the finance advice",
    ])
    chat = Chat(client, _config(domain_dir=str(tmp_path / "finance")), domain)
    resp = chat.respond("Should I diversify into bonds?")
    assert resp.kind == "answer"
    assert resp.text == "the finance advice"


def test_custom_strategy_orchestrates_complex(tmp_path):
    domain = load_domain_config(_write_finance_domain(tmp_path))
    client = FakeClient([
        '{"in_domain": true, "intent": "risk_check", "complexity": "complex", "reason": "ok"}',
        '{"tasks": [{"title": "r1", "instruction": "identify risks"}]}',
        "risk worker output",
        "final risk answer",
    ])
    chat = Chat(client, _config(domain_dir=str(tmp_path / "finance")), domain)
    resp = chat.respond("Assess my retirement risk profile")
    assert resp.kind == "answer"
    assert resp.text == "final risk answer"
```

- [ ] **Step 2: Run test to verify it passes (this guards the whole refactor)**

Run: `uv run pytest tests/test_domain_agnostic.py -v`
Expected: PASS. This is the regression guard — before this refactor it would FAIL with `chat.py` "No processor for strategy" (custom ids `advise`/`risk_assessment` were absent from `PROCESSOR_CLASSES` and `DEFAULT_STRATEGY` was hardcoded to `direct`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_domain_agnostic.py
git commit -m "test: add dogfood non-SE domain proving domain-agnostic strategies"
```

---

### Task 7: Update `README.md`

**Files:**
- Modify: `README.md:42-56` (domain directory section)

**Interfaces:**
- Consumes: the new data model.

- [ ] **Step 1: Review the current README domain section then rewrite it**

Read `README.md` lines 42-56, then replace that block with a version that describes strategies as data (remove any mention of fixed processor classes), including the `default: true` flag. Concretely, restate the directory layout:

```markdown
Each expert domain lives in its own directory, e.g. `domain/software_engineering/`:

- `domain.json`: domain name, description, and out-of-domain reply.
- `intents.yaml`: the intents the classifier can detect.
- `intent_mapping.yaml`: maps each intent to a strategy.
- `strategies.yaml`: strategy definitions — per-strategy optional `model`,
  `complexity_gate`, and exactly one `default: true` marker (the strategy used for
  unmapped intents).
- `prompts/*.md`: one fully self-contained system prompt per strategy. Each file
  embeds the strategy's answer structure and domain context directly, with **no**
  placeholders (`{name}`/`{description}`/`{structure}`).

Strategies are fully data-driven: swapping in a new expert domain means writing a new
domain directory — no code changes. Complex questions on a gated strategy run through
an Orchestrator pipeline (Planner → Workers → Aggregator) that builds on the strategy
prompt.
```

- [ ] **Step 2: Verify no stale references remain**

Run (code-level removed concepts + README):
```bash
grep -rn "PROCESSOR_CLASSES\|DEFAULT_STRATEGY\|No processor for strategy\|agent.processors\|{structure}" README.md agent/ tests/ --include=*.py
```
Expected: no matches.

Run (no placeholders left in any domain prompt file):
```bash
grep -rn "{name}\|{description}\|{structure}" domain/ --include=*.md
```
Expected: no matches. (Note: `{name}`/`{description}`/`{question}` remain in the **in-code** prompt templates of `agent/classification.py` and `agent/orchestrator.py` — those are separate from the domain prompt files and are intentionally out of scope for this change.)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for data-driven strategies"
```

---

### Task 8: Final verification and cleanup

**Files:**
- None (verification only).

**Interfaces:**
- Consumes: all prior tasks.

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Confirm the success criteria from the spec**

Run each check and confirm the output:
```bash
uv run pytest -q
grep -rn "PROCESSOR_CLASSES\|DEFAULT_STRATEGY\|No processor for strategy\|agent.processors\|{structure}" README.md agent/ tests/ --include=*.py | grep -v __pycache__ || echo "none"
grep -rn "{name}\|{description}\|{structure}" domain/ --include=*.md || echo "no placeholders in domain prompts"
ls agent/processors 2>&1 || echo "agent/processors removed"
uv run python -c "from agent.config import load_domain_config; d=load_domain_config('domain/software_engineering'); print('SE default strategy:', d.default_strategy)"
```
Expected: tests green; `none` for the first grep; `no placeholders in domain prompts` for the domain grep; `agent/processors removed`; SE default is `direct`.

- [ ] **Step 3: Commit any remaining changes (e.g. if the grep in Task 7 Step 2 missed a file)**

```bash
git add -A
git commit -m "chore: final domain-agnostic strategy cleanup" 2>/dev/null || echo "nothing to commit"
```
