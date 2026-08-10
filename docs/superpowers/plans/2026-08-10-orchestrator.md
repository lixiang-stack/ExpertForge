# Orchestrator (Complex Task Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `COMPLEX_UNSUPPORTED` placeholder with a real Orchestrator pipeline (Planner → Workers → Aggregator) that produces a final answer for complex tasks, preserving the strategy prompt context and using the `model_high` tier.

**Architecture:** `RouteResult` gains an `orchestrate: bool` flag; `Router.route()` stops rewriting the strategy to `COMPLEX_UNSUPPORTED` and instead keeps the original strategy while setting `orchestrate=True` on a gated complex hit. A new `agent/orchestrator.py` owns the pipeline: Planner (structured JSON via `json_schema`), sequential Workers, and an Aggregator, all sharing the resolved model. `Chat.respond()` dispatches to the Orchestrator when `route.orchestrate` is set.

**Tech Stack:** Python 3.10+, openai SDK, pyyaml, pytest, uv (all already in use).

## Global Constraints

- `git` working tree starts on `main`; do NOT modify `domain/software_engineering/*` files except as explicitly listed (only README/doc references change; prompt files stay).
- `uv run pytest -q` must pass (currently 78 tests) after every task.
- `RouteResult` fields: `in_domain: bool`, `strategy: str`, `intent: str | None = None`, `complexity: str | None = None`, `reject_reason: str = ""`, `orchestrate: bool = False`. The `COMPLEX_UNSUPPORTED` constant is deleted.
- `Orchestrator(client, config, domain)` with `run(self, question: str, route: RouteResult, model: str) -> str`. It reuses `build_registry(domain)` (from `agent/processors/registry.py`) to obtain the strategy's `Processor` and call `build_system_prompt()` for stage context. Do not modify `Processor` or the registry.
- Orchestrator stages all use the `model` argument passed to `run()`. `Chat` passes `resolve_model(self.config, self.domain, route, self.config.model)`.
- `unsupported_complex` prompt handling in `load_domain_config` (agent/config.py:223) and its tests are UNCHANGED — the file stays, `Chat` just never returns it.
- `LLMClient.chat_completion(messages, *, model, temperature, disable_thinking, json_mode, json_schema)` — existing signature; `json_schema` sends `response_format` JSON-schema; the orchestrator uses it for the Planner call.
- All `tests/*` FakeClients use the signature `chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None)`.

---

### Task 1: `RouteResult.orchestrate` + router change

**Files:**
- Modify: `agent/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: existing `ClassificationService`, `DomainConfig`, `RouteResult`.
- Produces: `RouteResult` with new `orchestrate: bool = False`; `COMPLEX_UNSUPPORTED` constant removed; `Router.route()` returns the original strategy with `orchestrate=True` on gated complex hits.

- [ ] **Step 1: Update the failing tests** — in `tests/test_router.py`, replace the `COMPLEX_UNSUPPORTED` import and the two complex cases:

```python
from agent.router import DEFAULT_STRATEGY, Router  # COMPLEX_UNSUPPORTED removed
```

Replace:

```python
def test_route_complex_gated_to_unsupported():
    client = FakeClient([_combined(True, "architecture_design", "complex")])
    result = Router(client, _config(), _domain()).route("design a big system")
    assert result.strategy == COMPLEX_UNSUPPORTED
```

with:

```python
def test_route_complex_gated_sets_orchestrate():
    client = FakeClient([_combined(True, "architecture_design", "complex")])
    result = Router(client, _config(), _domain()).route("design a big system")
    assert result.strategy == "analysis"
    assert result.orchestrate is True
```

And update `test_route_complex_ungated_strategy_stays` to also assert `orchestrate is False`:

```python
def test_route_complex_ungated_strategy_stays():
    client = FakeClient([_combined(True, "faq", "complex")])
    result = Router(client, _config(), _domain()).route("q")
    assert result.strategy == "direct"
    assert result.orchestrate is False
```

Also add `assert result.orchestrate is False` to `test_route_in_domain_maps_strategy_and_keeps_fields` and `test_route_unknown_intent_defaults_to_direct`.

Also update `tests/test_chat.py::test_respond_unsupported_complex` — since the router no longer produces `complex_unsupported`, this test's expected behavior changes. Until the Orchestrator exists (Task 2), complex gated tasks temporarily fall back to the strategy's processor, so this test becomes a processor-fallback test:

```python
def test_respond_complex_gated_falls_back_to_processor():
    chat = Chat(FakeClient([
        '{"in_domain": true, "intent": "troubleshooting", "complexity": "complex", "reason": "ok"}',
        "debug answer",
    ]), _config(), _domain())
    resp = chat.respond("huge debugging task")
    assert resp.kind == "answer"
    assert resp.text == "debug answer"
    assert chat.history == [("huge debugging task", "debug answer")]
```

Note: `test_chat.py`'s `_domain()` fixture has `strategies={"debugging": StrategyDef("debugging", complexity_gate=True)}` and prompts include `"debugging"`, so the fallback goes through the `DebuggingProcessor`. This test will be replaced again in Task 3 with the orchestrator version.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_router.py tests/test_chat.py -v`
Expected: FAIL — `RouteResult` has no attribute `orchestrate`; `COMPLEX_UNSUPPORTED` import error in both files.

- [ ] **Step 3: Implement** — in `agent/router.py`:

Delete the line `COMPLEX_UNSUPPORTED = "complex_unsupported"`. Change the `RouteResult` dataclass to:

```python
@dataclass
class RouteResult:
    in_domain: bool
    strategy: str
    intent: str | None = None
    complexity: str | None = None
    reject_reason: str = ""
    orchestrate: bool = False
```

Change `route()`'s tail to:

```python
        strategy = self.domain.intent_mapping.get(intent_id, DEFAULT_STRATEGY)
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

**Also in `agent/chat.py`** — the temporary fallback keeps the suite green until the Orchestrator exists (Task 2). Remove `COMPLEX_UNSUPPORTED` from the router import, and replace the `COMPLEX_UNSUPPORTED` branch with an `orchestrate` branch that currently routes through the existing processor (this branch is switched to `orchestrator.run` in Task 3):

```python
from .router import Router
```

Replace in `respond()`:

```python
        if route.orchestrate:
            processor = self.processors.get(route.strategy)
            if processor is None:
                return ChatResponse(kind="error", text=f"No processor for strategy '{route.strategy}'")
            model = resolve_model(self.config, self.domain, route, self.config.model)
            answer = processor.process(self.client, question, self.history, model=model)
            self.history.append((question, answer))
            return ChatResponse(kind="answer", text=answer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_router.py tests/test_chat.py -v`
Expected: all pass.

- [ ] **Step 5: Check for other `COMPLEX_UNSUPPORTED` / `unsupported` references**

Run: `rg -n "COMPLEX_UNSUPPORTED|complex_unsupported|kind=\"unsupported\"" agent/ tests/`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add agent/router.py agent/chat.py tests/test_router.py tests/test_chat.py
git commit -m "feat: add RouteResult.orchestrate flag, route complex to strategy processor fallback"
```

---

### Task 2: New `agent/orchestrator.py`

**Files:**
- Create: `agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `DomainConfig`/`AgentConfig` from `agent/config.py`; `RouteResult` from `agent/router.py`; `LLMClient` from `agent/llm.py`; `build_registry` from `agent/processors/registry.py`.
- Produces:
  - `class Orchestrator` with `__init__(self, client, config, domain)` and `run(self, question: str, route: RouteResult, model: str) -> str`.
  - Internal methods: `_strategy_context(strategy) -> str`, `_plan(question, strategy, context, model) -> list[tuple[str, str]] | None`, `_worker(question, task, context, model) -> str`, `_aggregate(question, strategy, context, tasks, outputs, model) -> str`, `_direct_answer(question, strategy, context, model) -> str`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_orchestrator.py`:

```python
import json

from agent.config import AgentConfig, DomainConfig, IntentDef, StrategyDef
from agent.orchestrator import Orchestrator
from agent.router import RouteResult


def _domain():
    return DomainConfig(
        name="sw",
        description="software engineering",
        out_of_domain_reply="Out.",
        intents={"troubleshooting": IntentDef("troubleshooting", "debug")},
        intent_mapping={"troubleshooting": "debugging"},
        strategies={"debugging": StrategyDef("debugging", complexity_gate=True)},
        prompts={
            "debugging": "Debug {name} {description} {structure}",
        },
    )


def _config():
    return AgentConfig(
        base_url="https://x", model="m", classifier_model="cm", domain_dir="d",
        model_low="low-a", model_high="high-a",
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        return self.responses.pop(0)


def _route():
    return RouteResult(
        in_domain=True, strategy="debugging", intent="troubleshooting",
        complexity="complex", orchestrate=True,
    )


def test_run_normal_path_planner_workers_aggregator():
    client = FakeClient([
        '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}',
        "worker1 output",
        "worker2 output",
        "final answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    assert len(client.calls) == 4
    # planner call uses json_schema; workers + aggregator do not
    planner_messages, planner_model, planner_dt, planner_jm, planner_schema = client.calls[0]
    assert planner_schema is not None
    assert "tasks" in planner_schema["properties"]
    assert planner_dt is True
    for _, model, dt, jm, schema in client.calls:
        assert model == "high-a"


def test_run_planner_invalid_json_degrades_to_direct():
    client = FakeClient([
        "not json",
        "direct answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "direct answer"
    assert len(client.calls) == 2


def test_run_planner_empty_tasks_degrades_to_direct():
    client = FakeClient([
        '{"tasks": []}',
        "direct answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "direct answer"
    assert len(client.calls) == 2


def test_run_worker_empty_output_still_aggregates():
    client = FakeClient([
        '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}',
        "worker1 output",
        "",
        "final answer",
    ])
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final answer"
    assert len(client.calls) == 4
    # aggregator user message contains both worker outputs including the empty one
    agg_messages = client.calls[3][0]
    agg_user = agg_messages[-1]["content"]
    assert "worker1 output" in agg_user
```

Add `import json` at the top (unused in the current fixtures but harmless; keep it for future helpers).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: `ModuleNotFoundError: No module named 'agent.orchestrator'`.

- [ ] **Step 3: Implement** — create `agent/orchestrator.py`:

```python
from __future__ import annotations

import json
import re

from .config import AgentConfig, DomainConfig
from .llm import LLMClient
from .processors.registry import build_registry
from .router import RouteResult


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _planner_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["title", "instruction"],
                },
            }
        },
        "required": ["tasks"],
    }


class Orchestrator:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain
        self._processors = build_registry(domain)

    def _strategy_context(self, strategy: str) -> str:
        proc = self._processors.get(strategy)
        if proc is not None:
            return proc.build_system_prompt()
        return self.domain.prompts.get(strategy, "")

    def run(self, question: str, route: RouteResult, model: str) -> str:
        context = self._strategy_context(route.strategy)
        tasks = self._plan(question, route.strategy, context, model)
        # TODO: add Evaluator / Optimizer phases after aggregation (future)
        if tasks is None:
            return self._direct_answer(question, route.strategy, context, model)
        outputs = [
            self._worker(question, task, context, model)
            for task in tasks  # TODO: parallelize worker execution (future)
        ]
        return self._aggregate(question, route.strategy, context, tasks, outputs, model)

    def _plan(
        self, question: str, strategy: str, context: str, model: str
    ) -> list[tuple[str, str]] | None:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a planning agent for an expert domain named {self.domain.name}.\n\n"
                    f"{self.domain.description}\n\n"
                    f"Task context:\n{context}\n\n"
                    "Rules:\n"
                    "- Decompose the user's complex task into 2-4 focused sub-tasks.\n"
                    "- Each sub-task must be answerable by a single standalone LLM call.\n"
                    "- Output ONLY a single JSON object: "
                    '{"tasks": [{"title": "...", "instruction": "..."}]}\n\n'
                    f"User question: {question}"
                ),
            }
        ]
        text = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_planner_schema()
        )
        data = _parse_json(text)
        if not data or not isinstance(data.get("tasks"), list):
            return None
        tasks: list[tuple[str, str]] = []
        for item in data["tasks"]:
            if not isinstance(item, dict):
                return None
            title = item.get("title")
            instruction = item.get("instruction")
            if not isinstance(title, str) or not isinstance(instruction, str):
                return None
            tasks.append((title, instruction))
        return tasks or None

    def _worker(self, question: str, task: tuple[str, str], context: str, model: str) -> str:
        title, instruction = task
        messages = [
            {
                "role": "system",
                "content": f"{context}\n\nSub-task: {instruction}",
            },
            {"role": "user", "content": question},
        ]
        return self.client.chat_completion(messages, model=model)

    def _aggregate(
        self,
        question: str,
        strategy: str,
        context: str,
        tasks: list[tuple[str, str]],
        outputs: list[str],
        model: str,
    ) -> str:
        sections = []
        for (title, _instruction), output in zip(tasks, outputs):
            sections.append(f"Sub-task: {title}\n{output}")
        user_content = (
            f"User question: {question}\n\n"
            f"Sub-task results:\n\n" + "\n\n".join(sections)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{context}\n\n"
                    "You are synthesizing sub-task results into one coherent final "
                    "answer to the user's original question."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(messages, model=model)

    def _direct_answer(self, question: str, strategy: str, context: str, model: str) -> str:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        return self.client.chat_completion(messages, model=model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: all pass.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -q`
Expected: all pass (78 + 4 new = 82).

- [ ] **Step 6: Commit**

```bash
git add agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add Orchestrator pipeline (planner/workers/aggregator) with degradation"
```

---

### Task 3: Wire Orchestrator into `Chat`

**Files:**
- Modify: `agent/chat.py`
- Test: `tests/test_chat.py`

**Interfaces:**
- Consumes: `Orchestrator` from `agent/orchestrator.py` (Task 2); `resolve_model` from `agent/model_router.py`.
- Produces: `Chat.respond()` dispatches to the Orchestrator when `route.orchestrate` is True; no more `unsupported` responses.

- [ ] **Step 1: Write the failing test** — in `tests/test_chat.py`, replace `test_respond_complex_gated_falls_back_to_processor` (the temporary Task 1 test) with the orchestrator version:

```python
def test_respond_orchestrates_complex():
    client = FakeClient([
        '{"in_domain": true, "intent": "troubleshooting", "complexity": "complex", "reason": "ok"}',
        '{"tasks": [{"title": "t1", "instruction": "i1"}, {"title": "t2", "instruction": "i2"}]}',
        "worker1 output",
        "worker2 output",
        "final answer",
    ])
    chat = Chat(client, _config(), _domain())
    resp = chat.respond("huge debugging task")
    assert resp.kind == "answer"
    assert resp.text == "final answer"
    assert chat.history == [("huge debugging task", "final answer")]
```

Note: the `_domain()` fixture already declares `strategies={"debugging": StrategyDef("debugging", complexity_gate=True)}` and prompts include `"debugging"`, so `build_registry` builds a `DebuggingProcessor` for the orchestrator's `_strategy_context`. Keep the `"unsupported_complex"` key in the fixture (unused, harmless).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_chat.py -v`
Expected: FAIL — `Chat` still routes `route.orchestrate` through the processor fallback (Task 1 state); the FakeClient for the orchestrator path consumes the planner/worker/aggregator responses as processor output, so `resp.text` won't equal `"final answer"`.

- [ ] **Step 3: Implement** — in `agent/chat.py`:

Add the import:

```python
        self.orchestrator = Orchestrator(client, config, domain)
```

Replace the `COMPLEX_UNSUPPORTED` branch in `respond()`:

```python
        if route.orchestrate:
            model = resolve_model(self.config, self.domain, route, self.config.model)
            answer = self.orchestrator.run(question, route, model)
            self.history.append((question, answer))
            return ChatResponse(kind="answer", text=answer)
```

The final file body (excluding imports/docstring) becomes:

```python
class Chat:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain
        self.router = Router(client, config, domain)
        self.processors = build_registry(domain)
        self.orchestrator = Orchestrator(client, config, domain)
        self.history: list[tuple[str, str]] = []

    def respond(self, question: str) -> ChatResponse:
        route = self.router.route(question)
        if not route.in_domain:
            text = self.domain.out_of_domain_reply
            if route.reject_reason:
                text += f" ({route.reject_reason})"
            return ChatResponse(kind="reject", text=text)
        if route.orchestrate:
            model = resolve_model(self.config, self.domain, route, self.config.model)
            answer = self.orchestrator.run(question, route, model)
            self.history.append((question, answer))
            return ChatResponse(kind="answer", text=answer)
        processor = self.processors.get(route.strategy)
        if processor is None:
            return ChatResponse(kind="error", text=f"No processor for strategy '{route.strategy}'")
        model = resolve_model(self.config, self.domain, route, self.config.model)
        answer = processor.process(self.client, question, self.history, model=model)
        self.history.append((question, answer))
        return ChatResponse(kind="answer", text=answer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_chat.py -v`
Expected: all pass.

- [ ] **Step 5: Run full suite and confirm no stale references**

Run: `uv run pytest -q`
Expected: all pass (82; `test_respond_complex_gated_falls_back_to_processor` replaced by `test_respond_orchestrates_complex`).

Run: `rg -n "COMPLEX_UNSUPPORTED|complex_unsupported|kind=\"unsupported\"" agent/ tests/`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add agent/chat.py tests/test_chat.py
git commit -m "feat: dispatch complex tasks through Orchestrator in Chat"
```

---

### Task 4: README + final verification

**Files:**
- Modify: `README.md`
- Test: none new (docs only)

**Interfaces:**
- Consumes: nothing new.

- [ ] **Step 1: Update README** — locate any text saying complex tasks are unsupported or referencing the orchestrator as future work. Replace with a short Orchestrator description. If the README lists `unsupported_complex.md` in the Domain directory section, keep the file mention (it still exists and is loaded) but add a note that complex tasks now run through the Orchestrator pipeline (Planner → Workers → Aggregator) using the strategy prompt and `model_high`.

- [ ] **Step 2: Full verification**

```bash
uv run pytest -q
env -u AGENT_API_KEY uv run python -m agent --ask "What is Go defer?"
```

Expected: all tests pass (82); the no-key smoke prints the `AGENT_API_KEY` error and exits 1.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe orchestrator pipeline in README"
```

- [ ] **Step 4: Review the plan against the spec**

  - §2 `RouteResult.orchestrate`, `COMPLEX_UNSUPPORTED` deleted, strategy preserved → Task 1.
  - §3 Planner (json_schema) / Workers (sequential) / Aggregator, degradation on planner failure → Task 2.
  - §3 TODO comments for Evaluator/Optimizer and concurrency → Task 2 (`run` method).
  - §4 Chat integration, orchestrator constructed in `__init__`, unsupported branch removed → Task 3.
  - §5 error handling: planner parse/validation → direct answer; LLMError propagates → Task 2.
  - §6 model routing via `resolve_model`, all stages share it → Tasks 2, 3.
  - §7 tests (orchestrator, router, chat), `unsupported_complex` loading unchanged → Tasks 1-3.
  - §8 docs: README + spec/plan; keep `unsupported_complex.md` → Task 4.
  - §9 success criteria 1-7 → covered across Tasks 1-4.
