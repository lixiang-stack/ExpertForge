# Strategy Routing Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the four post-review design changes to the completed strategy-routing branch: rename `coding` → `code_snippet`, move `code_review` → `analysis`, remove the Clarification phase entirely, and make classification a single-call structured-output JSON request.

**Architecture:** No new modules. Edits land across `llm.py`, `classifier.py`, `config.py`, `router.py`, `chat.py`, `repl.py`, `agent_cli.py`, the example `domain/software_engineering/` files, and their tests. `Chat` drops its pending-clarification state machine; the router's `RouteResult` loses `needs_clarification`. Each task keeps the full suite green.

**Tech Stack:** Python 3.10+, openai SDK, pyyaml, pytest, uv (already in use).

## Global Constraints

- `uv run pytest` must pass fully after every task (currently 63 tests).
- `Chat.respond(question)` — no `allow_clarification` parameter; `ChatResponse.kind` ∈ {reject, unsupported, error, answer}.
- `RouteResult` fields: `in_domain`, `strategy`, `intent`, `complexity`, `reject_reason` (no `needs_clarification`).
- `LLMClient.chat_completion(messages, *, model, temperature, disable_thinking, json_mode)`.
- Example domain strategies: `direct`, `teaching`, `debugging`, `analysis`, `code_snippet`.
- `load_domain_config` requires prompts for each declared strategy plus `unsupported_complex.md`; no `clarify.md`.
- Every FakeClient test double accepts `json_mode=False`.
- Run all commands from the worktree root: `/Users/haoli/github/ExpertForge/.worktrees/strategy-routing`.

---

### Task 1: `json_mode` in the LLM client

**Files:**
- Modify: `agent/llm.py:17-38`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `chat_completion(messages, *, model=None, temperature=0.3, disable_thinking=False, json_mode=False)`; when `json_mode=True` passes `response_format={"type": "json_object"}` to the SDK, alongside any `extra_body` for `disable_thinking`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_llm.py`:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_json_mode_passes_response_format(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "{}"
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}], json_mode=True)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


@patch("agent.llm.OpenAI")
def test_chat_completion_json_mode_off_by_default(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: `test_chat_completion_json_mode_passes_response_format` FAILS (`TypeError: chat_completion() got an unexpected keyword argument 'json_mode'`).

- [ ] **Step 3: Implement**

In `agent/llm.py`, change the `chat_completion` signature and body:

```python
    def chat_completion(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        disable_thinking: bool = False,
        json_mode: bool = False,
    ) -> str:
        try:
            kwargs = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if disable_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            return content or ""
        except OpenAIError as e:
            raise LLMError(f"LLM API call failed: {e}") from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent/llm.py tests/test_llm.py
git commit -m "feat: support json_mode in chat completion"
```

---

### Task 2: Single-call structured classification

**Files:**
- Modify: `agent/classifier.py`
- Test: `tests/test_classifier.py`
- Modify: test fakes so they accept `json_mode` (see Step 2)

**Interfaces:**
- Consumes: `LLMClient.chat_completion(..., json_mode=True)` from Task 1.
- Produces (unchanged signatures): `classify_question(client, question, name, description, *, model=None) → Classification`; `classify_intent(client, question, name, description, intents, *, model=None) → IntentClassification`; `classify_complexity(client, question, name, description, *, model=None) → ComplexityClassification`. Private `_classify_json(client, prompt, parser, *, model=None)` now issues **one** call.
- Removes: `_STRICT_REMINDER` and the `for strict in (False, True)` retry loop.

- [ ] **Step 1: Rewrite the failing tests** — replace `tests/test_classifier.py` with:

```python
import pytest

from agent.classifier import (
    Classification,
    classify_complexity,
    classify_intent,
    classify_question,
)
from agent.llm import LLMError


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
        self.calls.append((messages, model, disable_thinking, json_mode))
        return self.responses.pop(0)


def test_classify_in_domain():
    client = FakeClient(['{"in_domain": true, "reason": "in software engineering"}'])
    result = classify_question(client, "What is microservices?", "软件工程", "software engineering")
    assert isinstance(result, Classification)
    assert result.in_domain is True
    assert result.reason == "in software engineering"
    assert len(client.calls) == 1
    assert client.calls[0][2] is True
    assert client.calls[0][3] is True


def test_classify_out_of_domain():
    client = FakeClient(['{"in_domain": false, "reason": "unrelated"}'])
    result = classify_question(client, "What is the weather?", "软件工程", "software engineering")
    assert result.in_domain is False
    assert len(client.calls) == 1


def test_single_call_parse_failure_falls_back_reject():
    client = FakeClient(["garbage"])
    result = classify_question(client, "xxx", "软件工程", "software engineering")
    assert result.in_domain is False
    assert "Unreliable" in result.reason
    assert len(client.calls) == 1


def test_string_bool_not_coerced_to_true():
    client = FakeClient(['{"in_domain": "false", "reason": "unrelated"}'])
    result = classify_question(client, "What is the weather?", "软件工程", "software engineering")
    assert result.in_domain is False
    assert len(client.calls) == 1


def test_propagates_llm_error():
    class FailingClient:
        def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
            raise LLMError("boom")

    with pytest.raises(LLMError):
        classify_question(FailingClient(), "q", "软件工程", "software engineering")


def test_classify_intent_success():
    client = FakeClient(['{"intent": "concept_explain", "reason": "why"}'])
    result = classify_intent(client, "why is interface like this", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == "concept_explain"
    assert result.reason == "why"
    assert len(client.calls) == 1
    assert client.calls[0][3] is True


def test_classify_intent_unknown_falls_back_empty():
    client = FakeClient(['{"intent": "bogus", "reason": "x"}'])
    result = classify_intent(client, "q", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == ""
    assert "Unreliable" in result.reason
    assert len(client.calls) == 1


def test_classify_intent_unreliable_falls_back_empty():
    client = FakeClient(["garbage"])
    result = classify_intent(client, "q", "软件工程", "sw", ["concept_explain", "faq"])
    assert result.intent_id == ""
    assert "Unreliable" in result.reason
    assert len(client.calls) == 1


def test_classify_complexity_success():
    client = FakeClient(['{"complexity": "complex", "reason": "big scope"}'])
    result = classify_complexity(client, "design a redis cluster", "软件工程", "sw")
    assert result.level == "complex"
    assert len(client.calls) == 1


def test_classify_complexity_invalid_level_defaults_medium():
    client = FakeClient(['{"complexity": "huge", "reason": "x"}'])
    result = classify_complexity(client, "q", "软件工程", "sw")
    assert result.level == "medium"
    assert len(client.calls) == 1


def test_classify_complexity_unreliable_defaults_medium():
    client = FakeClient(["garbage"])
    result = classify_complexity(client, "q", "软件工程", "sw")
    assert result.level == "medium"
```

- [ ] **Step 2: Update cross-tool FakeClients** — in ALL of `tests/test_chat.py`, `tests/test_router.py`, `tests/test_repl.py` give `chat_completion` the extra `json_mode=False` parameter (they are exercised end-to-end through the classifier). Also fix `tests/test_agent_cli.py` (both fake classes, lines ~60 and ~82). Keep the return-body identical.

Exact change in each file:

```python
    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False):
```

(Only the `json_mode` parameter is added; keep `return` bodies unchanged.)

- [ ] **Step 3: Run tests to verify failures**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAILURES — current tests expecting retry behavior (2 calls, "Reminder" in content) conflict.

- [ ] **Step 4: Implement** — replace the retry logic in `agent/classifier.py`. Delete `_STRICT_REMINDER` and rewrite `_classify_json`:

```python
def _classify_json(client, prompt: str, parser, *, model: str | None = None):
    text = client.chat_completion(
        [{"role": "system", "content": prompt}],
        model=model,
        disable_thinking=True,
        json_mode=True,
    )
    return parser(text)
```

The three `classify_*` functions and the `_parse_*` functions stay otherwise identical. Their `_xxx_PROMPT` constants keep the existing "Output ONLY a single JSON object and nothing else." instruction.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py tests/test_llm.py -v`
Expected: all pass. Then run full suite: `uv run pytest -q` — all green (the fakes from Step 2 now accept `json_mode`).

- [ ] **Step 6: Commit**

```bash
git add agent/classifier.py tests/test_classifier.py tests/test_chat.py tests/test_router.py tests/test_repl.py tests/test_agent_cli.py
git commit -m "refactor: single-call structured-output classification"
```

---

### Task 3: Remove the Clarification phase

**Files:**
- Modify: `agent/config.py`, `agent/router.py`, `agent/chat.py`, `agent/repl.py`, `agent/agent_cli.py`
- Modify: `domain/software_engineering/intents.yaml`
- Delete: `domain/software_engineering/prompts/clarify.md`
- Test: `tests/test_config.py`, `tests/test_router.py`, `tests/test_chat.py`, `tests/test_repl.py`, `tests/test_agent_cli.py`

**Interfaces:**
- Consumes: existing `RouteResult` from Task 2's unchanged router.
- Produces: `RouteResult` with no `needs_clarification`; `IntentDef(id, description)` (no `needs_clarification`); `Chat.respond(question) -> ChatResponse` (no `allow_clarification`); prompts dict keyed only by strategies + `unsupported_complex`.

- [ ] **Step 1: Rewrite the failing tests**

1. In `tests/test_router.py`:
   - Remove the `needs_clarification=False` assert in `test_route_in_domain_simple_strategy` and in `test_route_complex_gated_to_unsupported` (`result.needs_clarification is False`).
   - Remove `test_route_needs_clarification` (whole test), and the `troubleshooting` intent entry in `_domain` (or just drop the `needs_clarification=True` arg). Keep `troubleshooting` mapped to `debugging` if you keep the intent.

2. In `tests/test_chat.py`:
   - In `_domain()`: change `IntentDef("troubleshooting", "debug", needs_clarification=True)` → `IntentDef("troubleshooting", "debug")`, and remove the `"clarify": ...` entry from `prompts`.
   - Delete `test_respond_clarification_then_answer` and `test_respond_skips_clarification_when_disallowed` entirely.

3. In `tests/test_config.py`:
   - In helper `_write_domain`: remove `needs_clarification: true` from the `faq` intent line and drop the `(base / "prompts" / "clarify.md").write_text("clarify", ...)` line.
   - In `test_load_domain_config_basic`, remove the `domain.intents["faq"].needs_clarification is True` assert, the `domain.intents["concept_explain"].needs_clarification is False` assert (line ~176-177), and the `assert "clarify" in domain.prompts["clarify"]` line (~183).
   - In `test_load_domain_config_out_of_domain_reply_default`, `test_load_domain_config_bad_yaml`, `test_load_domain_config_mapping_unknown_intent`, and `test_load_domain_config_out_of_domain_reply_default`: remove the `"prompts" / "clarify.md"` write lines (they only existed to satisfy the old loader; extra unmasked files are harmless but keep the fixtures clean).
   - Rewrite `test_load_domain_config_missing_prompt` so the *new* required file is the missing one — it currently triggers `ConfigError` only because `clarify.md` is absent. Change it to omit `unsupported_complex.md` instead (keep `faq: direct` mapping, `direct.md` prompt present):

```python
def test_load_domain_config_missing_prompt(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("- id: faq\n  description: q\n", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("faq: direct\n", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d {structure}", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_domain_config(str(base))
```

4. In `tests/test_repl.py`:
   - Remove `"clarify": "clarify",` from the `_domain()` prompts dict.

5. In `tests/test_agent_cli.py`:
   - `_write_domain` (the CLI helper) writes `clarify.md`; remove that write (the loader no longer reads it).

- [ ] **Step 2: Run suite to verify failures**

Run: `uv run pytest tests/test_chat.py tests/test_router.py tests/test_repl.py tests/test_agent_cli.py tests/test_config.py -q`
Expected: failures — `IntentDef` still requires `needs_clarification`, and `load_domain_config` still reads `clarify.md`.

- [ ] **Step 3: Implement**

**`agent/config.py`** — in `IntentDef`, delete the field:

```python
@dataclass
class IntentDef:
    id: str
    description: str
```

In `load_domain_config`, the intent loop becomes:

```python
    for item in intents_data:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ConfigError(f"Invalid intent entry in {base / 'intents.yaml'}: {item}")
        iid = item["id"]
        intents[iid] = IntentDef(
            id=iid,
            description=item.get("description") or "",
        )
```

Remove the prompt-loading for `clarify`:

```python
    prompts: dict[str, str] = {}
    prompt_dir = base / "prompts"
    for sid in strategies:
        prompts[sid] = _read_prompt(prompt_dir / f"{sid}.md")
    prompts["unsupported_complex"] = _read_prompt(prompt_dir / "unsupported_complex.md")
```

**`agent/router.py`** — remove `needs_clarification` from the dataclass and from `route()`:

```python
@dataclass
class RouteResult:
    in_domain: bool
    strategy: str
    intent: str | None = None
    complexity: str | None = None
    reject_reason: str = ""
```

In `route()`, delete the lines that compute `needs_clarification`:

```python
        strategy = self.domain.intent_mapping.get(intent_result.intent_id, DEFAULT_STRATEGY)
        strategy_def = self.domain.strategies.get(strategy)
        if strategy_def and strategy_def.complexity_gate and complexity_result.level == "complex":
            strategy = COMPLEX_UNSUPPORTED

        return RouteResult(
            in_domain=True,
            strategy=strategy,
            intent=intent_result.intent_id or None,
            complexity=complexity_result.level,
        )
```

(The `needs_clarification` computation is gone; `classify_intent` still receives `list(self.domain.intents)`.)

**`agent/chat.py`** — replace the whole file body, keeping the dataclass:

```python
from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig, DomainConfig
from .llm import LLMClient
from .processors.registry import build_registry
from .router import COMPLEX_UNSUPPORTED, Router


@dataclass
class ChatResponse:
    kind: str
    text: str


class Chat:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain
        self.router = Router(client, config, domain)
        self.processors = build_registry(domain)
        self.history: list[tuple[str, str]] = []

    def respond(self, question: str) -> ChatResponse:
        route = self.router.route(question)
        if not route.in_domain:
            text = self.domain.out_of_domain_reply
            if route.reject_reason:
                text += f" ({route.reject_reason})"
            return ChatResponse(kind="reject", text=text)
        if route.strategy == COMPLEX_UNSUPPORTED:
            return ChatResponse(
                kind="unsupported", text=self.domain.prompts["unsupported_complex"]
            )
        processor = self.processors.get(route.strategy)
        if processor is None:
            return ChatResponse(kind="error", text=f"No processor for strategy '{route.strategy}'")
        model = self.config.model
        strategy_def = self.domain.strategies.get(route.strategy)
        if strategy_def and strategy_def.model:
            model = strategy_def.model
        answer = processor.process(self.client, question, self.history, model=model)
        self.history.append((question, answer))
        return ChatResponse(kind="answer", text=answer)
```

**`agent/repl.py`** — remove the clarification branch, so the loop is:

```python
        try:
            response = chat.respond(question)
            print("expert > " + response.text)
        except LLMError as e:
            print(f"[error] {e}")
```

**`agent/agent_cli.py`** — change the `--ask` branch: `Chat(...).respond(ask)` (remove `allow_clarification=False`).

**`domain/software_engineering/intents.yaml`** — remove the `needs_clarification: true|alse` line from every entry (all 11 intents become just `id` + `description`).

**Delete** `domain/software_engineering/prompts/clarify.md`:

```bash
git rm domain/software_engineering/prompts/clarify.md
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent/config.py agent/router.py agent/chat.py agent/repl.py agent/agent_cli.py domain/software_engineering/intents.yaml tests/
git commit -m "feat: remove clarification phase (single-pass routing)"
```

---

### Task 4: Rename `coding` → `code_snippet`, move `code_review` → `analysis`

**Files:**
- Rename: `agent/processors/coding.py` → `agent/processors/code_snippet.py`
- Modify: `agent/processors/registry.py`
- Rename: `domain/software_engineering/prompts/coding.md` → `code_snippet.md`
- Modify: `domain/software_engineering/strategies.yaml`, `domain/software_engineering/intent_mapping.yaml`
- Test: `tests/test_processors.py`, `tests/test_config.py` (prompt keys), `agent` references in `tests`

**Interfaces:**
- Consumes: `Processor` base class in `agent/processors/base.py` (unchanged).
- Produces: `CodeSnippetProcessor` with `strategy_id = "code_snippet"`, registered in `PROCESSOR_CLASSES`; `generate_code: code_snippet` and `code_review: analysis` in the example mapping.

- [ ] **Step 1: Write the failing tests** — in `tests/test_processors.py`, replace `CodingProcessor` references

1. `from agent.processors.coding import CodingProcessor` → `from agent.processors.code_snippet import CodeSnippetProcessor`
2. In `_prompts()`, key `"coding": "Code ..."` → `"code_snippet": "Code ..."`
3. `test_coding_structure` → `test_code_snippet_structure`, instantiate `CodeSnippetProcessor`, assert `"Approach" in p.build_system_prompt()`.
4. `test_build_registry` expected set → `{"direct", "teaching", "debugging", "analysis", "code_snippet"}`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_processors.py -v`
Expected: import/module errors — `agent/processors/coding.py` no longer exists.

- [ ] **Step 3: Rename and implement**

```bash
git mv agent/processors/coding.py agent/processors/code_snippet.py
```

In `agent/processors/code_snippet.py`, update:

```python
class CodeSnippetProcessor(Processor):
    strategy_id = "code_snippet"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Approach\n"
            "- Code snippet\n"
            "- Key points and caveats\n"
            "- How to extend or adapt it"
        )
```

In `agent/processors/registry.py`, change import and add to `PROCESSOR_CLASSES`:

```python
from .code_snippet import CodeSnippetProcessor

PROCESSOR_CLASSES = {
    "direct": DirectAnswerProcessor,
    "teaching": TeachingProcessor,
    "debugging": DebuggingProcessor,
    "analysis": AnalysisProcessor,
    "code_snippet": CodeSnippetProcessor,
}
```

Rename and reword the prompt file:

```bash
git mv domain/software_engineering/prompts/coding.md domain/software_engineering/prompts/code_snippet.md
```

New content in to `code_snippet.md`:

```markdown
You are an expert Agent in the {name} domain.

{description}

{structure}

Answering requirements:
- Answer authoritatively and professionally.
- Produce short, idiomatic code fragments focused on the question.
- Keep the snippet self-contained and explain the reasoning inline.
- Only answer questions within this domain.
```

In `domain/software_engineering/strategies.yaml` change `coding:` → `code_snippet:` (keep `model: null`, `complexity_gate: true`).

In `domain/software_engineering/intent_mapping.yaml`:

```yaml
concept_explain: teaching
tutorial: teaching
learning_guide: teaching
faq: direct
summarization: direct
troubleshooting: debugging
comparison: analysis
performance_analysis: analysis
architecture_design: analysis
generate_code: code_snippet
code_review: analysis
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent/processors/ domain/software_engineering/ tests/test_processors.py
git commit -m "refactor: code_snippet strategy, code_review mapped to analysis"
```

---

### Task 5: Docs, full regression, smoke

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-09-strategy-routing-revision.md` (this file, if discovered during execution)

**Interfaces:** none (documents).

- [ ] **Step 1: Update `README.md`**

- In the intro, change "five strategy processors" → "five strategy processors" stays, but update the list in the Domain directory section from `direct`/`teaching`/`debugging`/`analysis`/`coding` to `direct`/`teaching`/`debugging`/`analysis`/`code_snippet`, and drop the reference to `clarify.md` (keep `unsupported_complex.md`).
- Fix the same `coding`→`code_snippet` reference in the "prompts/*.md" bullet.

- [ ] **Step 2: Full regression**

Run: `uv run pytest -v`
Expected: all pass (63+ tests; count locally).

Also run the no-key smoke to confirm the chain still reaches the API-key check:

```bash
env -u AGENT_API_KEY uv run python -m agent --ask "What is Go defer?"; echo "exit=$?"
```

Expected: stdout/stderr contains `Config error: AGENT_API_KEY environment variable is not set. ...` and exit code 1.

- [ ] **Step 3: Review the doc-to-code consistency**

Check `docs/superpowers/specs/2026-08-07-strategy-routing-design.md` still matches the working tree (it was already updated in the spec revision). No code changed here.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: post-review README updates"
```

- [ ] **Step 5: Report**

Report to the coordinator: list commits, final test count, and any deviations found.

---