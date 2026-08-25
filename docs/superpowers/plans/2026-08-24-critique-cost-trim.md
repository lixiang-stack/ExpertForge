# Critique Topology P2 Cost Trim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut critique-topology token overhead (~+571% vs baseline on se-129) to ≤300% via three independent measures: critics drop strategy context (A), per-intent perspective caps (B), revise output budgets (C).

**Architecture:** All three measures modify the critique path of `Orchestrator._run_critique` only. Measure B adds a `CritiquePolicy` dataclass parsed from an optional `critique:` block in `orchestration.yaml` (domains are data). Measure C threads a `max_tokens` parameter through `LLMClient.chat_completion` and computes a budget in `_revise` from the draft's completion-token count. The observability layer (`agent/observability/patch.py`) monkey-patches `_plan_perspectives`, `_critic`, `_revise`, `_draft` — every signature change here MUST update the corresponding wrapper in the same task.

**Tech Stack:** Python ≥ 3.10, pytest, PyYAML, OpenAI SDK client wrapper.

**Spec:** `docs/superpowers/specs/2026-08-24-critique-cost-trim-design.md` (Approved). Base: PR #22 critique topology (already on local `main`).

## Global Constraints

- Every module starts with `from __future__ import annotations`.
- The `map_reduce` path (`_run_map_reduce`, `_plan`, `_worker`, `_aggregate`, `_direct_answer`, `_reaggregate`) is **untouched**.
- `expert_policy.md` files are untouched; draft keeps thinking enabled (no `disable_thinking` on `_draft`).
- Behavior is config-driven: perspective counts come only from `CritiquePolicy` defaults / `orchestration.yaml`; never hardcode perspective lists.
- Invalid config raises `ConfigError` (never silent fallback) — matches existing `domain_config.py` validation style (`isinstance` + range checks, no bool special-casing).
- Observability wrappers degrade to warnings on failure but their signatures must match the patched methods exactly (they use explicit parameter lists, not `*args/**kwargs`).
- Test fakes in `tests/unit/test_orchestrator.py` record 5-tuples `(messages, model, disable_thinking, json_mode, json_schema)` — do NOT change tuple shape; extend signatures only.
- Run tests with `uv run pytest -q` (pytest.ini restricts to `tests/unit`). One file: `uv run pytest tests/unit/test_orchestrator.py -q`.
- Commits follow repo style: `feat: ...` / `docs: ...`, one commit per task.

---

### Task 1: `CritiquePolicy` config + domain_config parsing

**Files:**
- Modify: `agent/config.py` (add `CritiquePolicy` near `EvaluatorPolicy` ~line 271; add field to `OrchestrationPolicy` ~line 278)
- Modify: `agent/domain_config.py` (`_parse_orchestration` ~line 99-142; `DOMAIN_FILE_CONTRACT` line 19)
- Test: `tests/unit/test_config.py` (append near topology tests ~line 1102)

**Interfaces:**
- Consumes: existing `OrchestrationPolicy`, `ConfigError`, `_parse_orchestration(base, intents)`.
- Produces: `CritiquePolicy(default_max_perspectives: int = 3, max_perspectives_by_intent: dict[str, int] = <factory dict>, revise_token_ratio: float = 1.1, revise_min_tokens: int = 1024, revise_max_tokens: int = 6000)`; `OrchestrationPolicy.critique: CritiquePolicy | None = None` (last field). Later tasks construct `CritiquePolicy()` directly and read these attribute names.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_config.py` (the `_write_domain(tmp_path)` helper writes intents `concept_explain` + `faq`; reuse it like the topology tests do):

```python
_CRITIQUE_YAML_FULL = (
    "topology: critique\n"
    "critique:\n"
    "  default_max_perspectives: 2\n"
    "  max_perspectives_by_intent:\n"
    "    faq: 1\n"
    "  revise_token_ratio: 1.5\n"
    "  revise_min_tokens: 512\n"
    "  revise_max_tokens: 4096\n"
)


def _domain_with_yaml(tmp_path, yaml_text):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "orchestration.yaml").write_text(yaml_text, encoding="utf-8")
    (base / "domain.json").write_text(
        json.dumps({"name": "软件工程", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text(
        "- id: concept_explain\n  description: explain\n"
        "- id: faq\n  description: quick question\n",
        encoding="utf-8",
    )
    (base / "intent_mapping.yaml").write_text(
        "concept_explain: teaching\nfaq: direct\n", encoding="utf-8"
    )
    (base / "prompts" / "teaching.md").write_text("teach", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("direct", encoding="utf-8")
    return str(base)


def test_load_domain_config_critique_absent_is_none(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    assert domain.orchestration.critique is None


def test_load_domain_config_critique_block_parsed(tmp_path):
    domain = load_domain_config(_domain_with_yaml(tmp_path, ORCHESTRATION_YAML + _CRITIQUE_YAML_FULL))
    c = domain.orchestration.critique
    assert c.default_max_perspectives == 2
    assert c.max_perspectives_by_intent == {"faq": 1}
    assert c.revise_token_ratio == 1.5
    assert c.revise_min_tokens == 512
    assert c.revise_max_tokens == 4096


def test_load_domain_config_critique_partial_block_uses_defaults(tmp_path):
    domain = load_domain_config(_domain_with_yaml(
        tmp_path, ORCHESTRATION_YAML + "critique:\n  default_max_perspectives: 2\n"))
    c = domain.orchestration.critique
    assert c.default_max_perspectives == 2
    assert c.max_perspectives_by_intent == {}
    assert c.revise_token_ratio == 1.1
    assert c.revise_min_tokens == 1024
    assert c.revise_max_tokens == 6000


def test_load_domain_config_critique_non_mapping_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_domain_with_yaml(tmp_path, ORCHESTRATION_YAML + "critique: bogus\n"))


def test_load_domain_config_critique_bad_default_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_domain_with_yaml(
            tmp_path, ORCHESTRATION_YAML + "critique:\n  default_max_perspectives: 0\n"))


def test_load_domain_config_critique_unknown_intent_key_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_domain_with_yaml(
            tmp_path,
            ORCHESTRATION_YAML
            + "critique:\n  max_perspectives_by_intent:\n    no_such_intent: 2\n"))


def test_load_domain_config_critique_bad_ratio_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_domain_with_yaml(
            tmp_path, ORCHESTRATION_YAML + "critique:\n  revise_token_ratio: 0\n"))


def test_load_domain_config_critique_min_gt_max_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_domain_config(_domain_with_yaml(
            tmp_path,
            ORCHESTRATION_YAML
            + "critique:\n  revise_min_tokens: 9000\n  revise_max_tokens: 6000\n"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py -q -k critique`
Expected: FAIL — `AttributeError: 'OrchestrationPolicy' object has no attribute 'critique'` (or similar).

- [ ] **Step 3: Implement `CritiquePolicy` in `agent/config.py`**

Add after the `EvaluatorPolicy` class (line ~275):

```python
@dataclass
class CritiquePolicy:
    default_max_perspectives: int = 3
    max_perspectives_by_intent: dict[str, int] = field(default_factory=dict)
    revise_token_ratio: float = 1.1
    revise_min_tokens: int = 1024
    revise_max_tokens: int = 6000
```

Extend `OrchestrationPolicy` (keep field order, add last):

```python
@dataclass
class OrchestrationPolicy:
    enabled: bool = True
    min_complexity: str = "complex"
    intents: list[str] = field(default_factory=list)
    max_workers: int = 4
    topology: str = "map_reduce"
    evaluator: EvaluatorPolicy = field(default_factory=EvaluatorPolicy)
    critique: CritiquePolicy | None = None
```

- [ ] **Step 4: Parse/validate in `agent/domain_config.py`**

Import `CritiquePolicy` in the import block from `.config`. Update `DOMAIN_FILE_CONTRACT` string: change `"(enabled, min_complexity, intents, max_workers, topology, evaluator), "` to `"(enabled, min_complexity, intents, max_workers, topology, evaluator, optional critique), "`.

Inside `_parse_orchestration`, after the evaluator parsing (after line 130) and before `return OrchestrationPolicy(...)`:

```python
    critique_data = orch_data.get("critique")
    critique = None
    if critique_data is not None:
        if not isinstance(critique_data, dict):
            raise ConfigError(f"orchestration.yaml 'critique' must be a mapping: {orch_path}")
        default_max = critique_data.get("default_max_perspectives", 3)
        if not isinstance(default_max, int) or default_max <= 0:
            raise ConfigError(
                f"orchestration.yaml 'default_max_perspectives' must be a positive int: {orch_path}"
            )
        by_intent_raw = critique_data.get("max_perspectives_by_intent", {})
        if not isinstance(by_intent_raw, dict):
            raise ConfigError(
                f"orchestration.yaml 'max_perspectives_by_intent' must be a mapping: {orch_path}"
            )
        by_intent: dict[str, int] = {}
        for key, value in by_intent_raw.items():
            if not isinstance(key, str) or key not in intents:
                raise ConfigError(
                    f"orchestration.yaml 'max_perspectives_by_intent' references "
                    f"unknown intent {key!r}: {orch_path}"
                )
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(
                    f"orchestration.yaml 'max_perspectives_by_intent[{key}]' "
                    f"must be a positive int: {orch_path}"
                )
            by_intent[key] = value
        ratio = critique_data.get("revise_token_ratio", 1.1)
        if not isinstance(ratio, (int, float)) or ratio <= 0:
            raise ConfigError(
                f"orchestration.yaml 'revise_token_ratio' must be a positive number: {orch_path}"
            )
        min_tokens = critique_data.get("revise_min_tokens", 1024)
        max_tokens = critique_data.get("revise_max_tokens", 6000)
        for name, value in (("revise_min_tokens", min_tokens), ("revise_max_tokens", max_tokens)):
            if not isinstance(value, int) or value <= 0:
                raise ConfigError(f"orchestration.yaml '{name}' must be a positive int: {orch_path}")
        if min_tokens > max_tokens:
            raise ConfigError(
                f"orchestration.yaml 'revise_min_tokens' must be "
                f"<= 'revise_max_tokens': {orch_path}"
            )
        critique = CritiquePolicy(
            default_max_perspectives=default_max,
            max_perspectives_by_intent=by_intent,
            revise_token_ratio=ratio,
            revise_min_tokens=min_tokens,
            revise_max_tokens=max_tokens,
        )
```

And pass `critique=critique,` as the last argument of the `return OrchestrationPolicy(...)` constructor.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py -q`
Expected: PASS (all, including pre-existing).

- [ ] **Step 6: Commit**

```bash
git add agent/config.py agent/domain_config.py tests/unit/test_config.py
git commit -m "feat: CritiquePolicy config block for critique cost trimming"
```

---

### Task 2: Measure B — perspective cap by intent

**Files:**
- Modify: `agent/orchestrator.py` (`run` line 198, `_run_critique` line 232, `_plan_perspectives` line 278, `_PERSPECTIVES_PROMPT` line 123; add import of `CritiquePolicy`)
- Modify: `agent/observability/patch.py` (`_wrap_plan_perspectives` line 300)
- Modify: `domain/software_engineering/orchestration.yaml` (append `critique:` block)
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `CritiquePolicy` from Task 1; `OrchestrationPolicy.critique | None`.
- Produces: `_plan_perspectives(self, question, strategy, context, model, *, max_perspectives: int) -> list[WorkerTask] | None` (new keyword-only arg); `_run_critique(self, question, strategy, context, model, policy, intent)` (new trailing positional arg); `Orchestrator._resolve_max_perspectives(self, intent: str) -> int`. The resolved N appears in the planner system prompt as `Plan exactly {N} distinct review perspectives`.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_orchestrator.py`:

```python
from agent.config import CritiquePolicy

_FOUR_PERSPECTIVES_JSON = (
    '{"perspectives": ['
    '{"title": "p1", "focus": "f1", "role": "R1"},'
    '{"title": "p2", "focus": "f2", "role": "R2"},'
    '{"title": "p3", "focus": "f3", "role": "R3"},'
    '{"title": "p4", "focus": "f4", "role": "R4"}]}'
)


def test_plan_perspectives_prompt_pins_n_and_truncates():
    client = FakeClient([_FOUR_PERSPECTIVES_JSON])
    orch = Orchestrator(client, _config(), _critique_domain())
    result = orch._plan_perspectives("q", "debugging", "ctx", "high-a", max_perspectives=3)
    assert len(result) == 3
    assert "exactly 3 distinct review perspectives" in client.calls[0][0][0]["content"]


def test_run_critique_default_cap_is_three_without_critique_block():
    client = FakeClient(["draft answer", _CRITIQUE_PLAN_JSON, _ISSUES_EMPTY, _ISSUES_EMPTY])
    Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert "exactly 3 distinct review perspectives" in client.calls[1][0][0]["content"]


def test_run_critique_uses_intent_override_from_critique_block():
    domain = _critique_domain(evaluator=EvaluatorPolicy(enabled=False))
    domain.orchestration.critique = CritiquePolicy(
        default_max_perspectives=3, max_perspectives_by_intent={"troubleshooting": 2})
    client = FakeClient(["draft answer", _CRITIQUE_PLAN_JSON, _ISSUES_EMPTY, _ISSUES_EMPTY])
    Orchestrator(client, _config(), domain).run("huge task", _route(), "high-a")
    assert "exactly 2 distinct review perspectives" in client.calls[1][0][0]["content"]
```

(`_route()` has `intent="troubleshooting"`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator.py -q -k "perspect"`
Expected: FAIL with `TypeError: _plan_perspectives() got an unexpected keyword argument 'max_perspectives'` and assertion failures on "exactly 3".

- [ ] **Step 3: Implement in `agent/orchestrator.py`**

a) Extend the import: `from .config import AgentConfig, CritiquePolicy, DomainConfig, resolve_judge_model`.

b) Change `_PERSPECTIVES_PROMPT` rule line:

```python
_RULES = """Rules:
- Plan exactly {n} distinct review perspectives for checking a draft expert answer.
```

Concretely, replace the template's rule bullet `- Plan 2-4 distinct review perspectives for checking a draft expert answer.` with `- Plan exactly {n} distinct review perspectives for checking a draft expert answer.` (the `{n}` placeholder joins the existing `.format` call).

c) Add resolver method to `Orchestrator` (after `_strategy_context`):

```python
    def _resolve_max_perspectives(self, intent: str) -> int:
        policy = self.domain.orchestration
        critique = policy.critique if policy and policy.critique else CritiquePolicy()
        return critique.max_perspectives_by_intent.get(intent, critique.default_max_perspectives)
```

(The `or CritiquePolicy()` implements the approved decision: no `critique:` block ⇒ new default cap 3 applies.)

d) Change signatures and call sites:

```python
    def run(self, question: str, route: RouteResult, model: str) -> str:
        ...
        if topology == "critique":
            return self._run_critique(question, route.strategy, context, model, policy, route.intent)
        return self._run_map_reduce(question, route, context, model, policy)

    def _run_critique(self, question: str, strategy: str, context: str, model: str, policy, intent: str) -> str:
        draft = self._draft(question, strategy, context, model)
        max_perspectives = self._resolve_max_perspectives(intent)
        perspectives = self._plan_perspectives(
            question, strategy, context, model, max_perspectives=max_perspectives,
        )
        ...  # rest unchanged
```

e) `_plan_perspectives` gets keyword-only param, formats `{n}`, and hard-truncates regardless of model output:

```python
    def _plan_perspectives(
        self, question: str, strategy: str, context: str, model: str, *,
        max_perspectives: int,
    ) -> list[WorkerTask] | None:
        prompt = _PERSPECTIVES_PROMPT.format(
            name=self.domain.name,
            description=self.domain.description,
            context=context,
            n=max_perspectives,
        )
        ...  # request/parsing identical
        return perspectives[:max_perspectives] or None
```

- [ ] **Step 4: Update observability wrapper in `agent/observability/patch.py`**

Replace `_wrap_plan_perspectives` (line ~300):

```python
    def _wrap_plan_perspectives(self, original, key):
        def wrapper(orch, question, strategy, context, model, *, max_perspectives=None):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, model,
                                max_perspectives=max_perspectives)
            with phase(inst._phase(key)):
                perspectives = original(orch, question, strategy, context, model,
                                        max_perspectives=max_perspectives)
                tid = current_trace_id()
                if tid:
                    data = {"degraded": True} if perspectives is None else {
                        "tasks": [{"title": t.title, "instruction": t.instruction,
                                   "role": t.role} for t in perspectives]}
                    data["max_perspectives"] = max_perspectives
                    inst._record_decision(tid, inst._phase(key), data)
                return perspectives
        return wrapper
```

- [ ] **Step 5: Configure the software_engineering domain**

`domain/software_engineering/orchestration.yaml` becomes:

```yaml
enabled: true
min_complexity: complex
intents:
  - architecture_design
  - troubleshooting
  - code_task
max_workers: 4
topology: critique
evaluator:
  enabled: true
  min_dimension_score: 3
  max_rounds: 1
critique:
  default_max_perspectives: 3
  max_perspectives_by_intent:
    architecture_design: 3
    troubleshooting: 2
    code_task: 2
  revise_token_ratio: 1.1
  revise_min_tokens: 1024
  revise_max_tokens: 6000
```

Sanity-check parse: `uv run python -c "from agent.domain_config import load_domain_config; d = load_domain_config('domain/software_engineering'); print(d.orchestration.critique)"` → prints the CritiquePolicy values above.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_orchestrator.py tests/unit/test_config.py tests/unit/test_observability_patch.py tests/unit/test_chat.py tests/unit/test_router.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent/orchestrator.py agent/observability/patch.py domain/software_engineering/orchestration.yaml tests/unit/test_orchestrator.py
git commit -m "feat: cap critique perspectives by intent config (measure B)"
```

---

### Task 3: Measure A — critics drop strategy/expert-policy context

**Files:**
- Modify: `agent/orchestrator.py` (`_CRITIC_SYSTEM_TEMPLATE` line 139; `_critic` line 310; call site in `_run_critique` line 239)
- Modify: `agent/observability/patch.py` (`_wrap_critic` line 316, and its entry in the `factories`/`targets` lists stay unchanged)
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_critic(self, question, perspective: WorkerTask, draft: str, model: str) -> str` — the `context` parameter is REMOVED. The observability `_wrap_critic` wrapper signature changes identically.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_orchestrator.py`:

```python
def test_critic_system_excludes_strategy_context():
    client = FakeClient([_ISSUES_EMPTY])
    orch = Orchestrator(client, _config(), _domain())
    orch._critic("q", WorkerTask("t1", "check coherence", "Coherence"), "draft text", "high-a")
    system = client.calls[0][0][0]["content"]
    assert "Debugging system prompt." not in system
    assert "You are a reviewer" in system
    assert "Coherence" in system
    assert "check coherence" in system


def test_run_critique_only_draft_planner_revise_carry_context():
    client = FakeClient([
        "draft answer",
        _CRITIQUE_PLAN_JSON,
        _ISSUES_EMPTY,
        _ISSUES_EMPTY,
        "revised answer",
    ])
    Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    roles = ["draft", "planner", "critic", "critic", "revise"]
    for (messages, *_rest), role in zip(client.calls, roles):
        carries = "Debugging system prompt." in messages[0]["content"]
        if role in {"draft", "planner", "revise"}:
            assert carries, role
        else:
            assert not carries, role
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator.py -q -k "context"`
Expected: FAIL — critic systems currently contain the context.

- [ ] **Step 3: Implement in `agent/orchestrator.py`**

a) Template loses its `{context}` prefix:

```python
_CRITIC_SYSTEM_TEMPLATE = """You are a reviewer. Your review perspective: {role}
Focus: {instruction}

Review the draft answer to the user's question from this perspective only.
Report only real defects:
- Internal contradictions or inconsistent decisions across sections
- Unsupported claims presented as fact (assumptions must stay flagged as assumptions)
- Technical errors
- Missing reasoning where the question demands it

Do NOT rewrite the answer. Output ONLY a single JSON object:
{{"issues": [{{"severity": "high|medium|low", "description": "...", "suggestion": "..."}}]}}
If there are no defects, output {{"issues": []}}.
"""
```

b) `_critic` drops the `context` parameter:

```python
    def _critic(self, question: str, perspective: WorkerTask, draft: str, model: str) -> str:
        system = _CRITIC_SYSTEM_TEMPLATE.format(
            role=perspective.role, instruction=perspective.instruction,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"User question:\n{question}\n\nDraft answer:\n{draft}"},
        ]
        return self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_issues_schema()
        ).text
```

c) Call site in `_run_critique`:

```python
                lambda p: self._critic(question, p, draft, model),
```

- [ ] **Step 4: Update observability wrapper in `agent/observability/patch.py`**

```python
    def _wrap_critic(self, original, key):
        def wrapper(orch, question, perspective, draft, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, perspective, draft, model)
            base = inst._phase(key)
            n = inst._next_worker(current_trace_id() or "")
            with phase(f"{base}.{n}"):
                tid = current_trace_id()
                try:
                    result = original(orch, question, perspective, draft, model)
                except Exception as e:  # noqa: BLE001 - record failure, then re-raise; business decides
                    if tid:
                        inst._record_decision(tid, f"{base}.{n}", {
                            "task": perspective.title, "role": perspective.role, "error": str(e)})
                    raise
                if tid:
                    inst._record_decision(tid, f"{base}.{n}", {
                        "task": perspective.title, "role": perspective.role})
                return result
        return wrapper
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_orchestrator.py tests/unit/test_observability_patch.py -q`
Expected: PASS (existing `CriticFailingClient` still matches critics via `"You are a reviewer"`).

- [ ] **Step 6: Commit**

```bash
git add agent/orchestrator.py agent/observability/patch.py tests/unit/test_orchestrator.py
git commit -m "feat: critics no longer mount expert_policy/strategy context (measure A)"
```

---

### Task 4: `LLMClient.chat_completion` accepts `max_tokens`

**Files:**
- Modify: `agent/llm.py` (`chat_completion` signature line 41; kwargs assembly line 60)
- Test: `tests/unit/test_llm.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `chat_completion(messages, *, model=None, temperature=0.3, disable_thinking=False, json_mode=False, json_schema=None, max_tokens: int | None = None) -> ChatResult`. When `None` the request body has NO `max_tokens` key; when set, `kwargs["max_tokens"] = max_tokens`. `RecordingClient` (evaluation/runner.py) forwards via `**kwargs` — no change needed there.

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_llm.py`:

```python
@patch("agent.llm.OpenAI")
def test_chat_completion_omits_max_tokens_by_default(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}])

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in kwargs


@patch("agent.llm.OpenAI")
def test_chat_completion_forwards_max_tokens(mock_openai):
    resp = MagicMock()
    resp.choices[0].message.content = "x"
    resp.model = "model-a"
    resp.usage = None
    mock_openai.return_value.chat.completions.create.return_value = resp

    client = LLMClient("https://api.example.com/v1", "key", "model-a")
    client.chat_completion([{"role": "user", "content": "hi"}], max_tokens=2048)

    kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 2048
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_llm.py -q -k max_tokens`
Expected: FAIL with `TypeError: unexpected keyword argument 'max_tokens'`.

- [ ] **Step 3: Implement in `agent/llm.py`**

Add the parameter and write it conditionally:

```python
    def chat_completion(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        disable_thinking: bool = False,
        json_mode: bool = False,
        json_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        ...
            kwargs = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_llm.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/llm.py tests/unit/test_llm.py
git commit -m "feat: optional max_tokens on LLMClient.chat_completion"
```

---

### Task 5: Measure C — revise output budget + surgical-revision prompt

**Files:**
- Modify: `agent/orchestrator.py` (`_REVISE_SYSTEM_TEMPLATE` line 156; `_draft` line 269; `_run_critique` lines 232-254 incl. `improve`; `_revise` line 330; add `import math` and `CritiquePolicy` import already added in Task 2)
- Modify: `agent/observability/patch.py` (`_wrap_revise` line 338)
- Test: `tests/unit/test_orchestrator.py` (extend the four fake clients' signatures; add recording subclasses)

**Interfaces:**
- Consumes: Task 1 `CritiquePolicy` fields (`revise_token_ratio`, `revise_min_tokens`, `revise_max_tokens`); Task 4 `max_tokens` kwarg.
- Produces:
  - `_draft(...) -> ChatResult` (was `-> str`; callers use `.text`).
  - `_revise(self, question, strategy, context, draft, issues, model, draft_completion_tokens: int | None = None) -> str` (new trailing keyword-friendly arg, default `None`).
  - Budget rule (exact): if `draft_completion_tokens` is truthy → `max_tokens = max(min_tokens, min(ceil(tokens * ratio), max_tokens_cap))`; otherwise → `max_tokens = revise_max_tokens`.

- [ ] **Step 1: Extend the test fakes (signatures only) and add recording subclasses**

In `tests/unit/test_orchestrator.py`, add `, max_tokens=None` as the LAST parameter to `chat_completion` in ALL FOUR fake classes (`FakeClient` line 37, `RaisingClient` line 48, `CallRaisingClient` line 72, `CriticFailingClient` line 341). Do NOT change what they append to `self.calls`. Example for `FakeClient`:

```python
    def chat_completion(self, messages, model=None, temperature=0.3, disable_thinking=False,
                        json_mode=False, json_schema=None, max_tokens=None):
        self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
        return ChatResult(text=self.responses.pop(0), model=model or "m")
```

Then append the new helpers and tests:

```python
class BudgetRecordingClient(FakeClient):
    """Records max_tokens per call; fakes a fixed completion_tokens on every response."""

    def __init__(self, responses, completion_tokens=0):
        super().__init__(responses)
        self.completion_tokens = completion_tokens
        self.max_tokens_seen = []

    def chat_completion(self, messages, model=None, temperature=0.3, disable_thinking=False,
                        json_mode=False, json_schema=None, max_tokens=None):
        self.max_tokens_seen.append(max_tokens)
        text = self.responses.pop(0)
        return ChatResult(text=text, model=model or "m",
                          completion_tokens=self.completion_tokens)


def _budget_client(completion_tokens):
    return BudgetRecordingClient(
        ["draft answer", _CRITIQUE_PLAN_JSON, _ISSUES_JSON, _ISSUES_EMPTY, "revised answer"],
        completion_tokens=completion_tokens,
    )


def test_revise_budget_applies_ratio_within_bounds():
    client = _budget_client(completion_tokens=2000)  # ceil(2000*1.1)=2200
    Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert client.max_tokens_seen == [None, None, None, None, 2200]


def test_revise_budget_floors_at_min_tokens():
    client = _budget_client(completion_tokens=500)  # ceil(550)=550 -> floor 1024
    Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert client.max_tokens_seen[-1] == 1024


def test_revise_budget_caps_at_max_tokens():
    client = _budget_client(completion_tokens=8000)  # ceil(8800) -> cap 6000
    Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert client.max_tokens_seen[-1] == 6000


def test_revise_budget_defaults_to_cap_without_usage_count():
    client = _budget_client(completion_tokens=0)  # unavailable count -> absolute cap
    Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert client.max_tokens_seen[-1] == 6000


def test_revise_prompt_contains_surgical_constraints():
    client = _budget_client(completion_tokens=2000)
    Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    system = client.calls[4][0][0]["content"]
    assert "You authored the draft" in system
    assert "Do NOT expand the answer" in system
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator.py -q -k "budget or surgical"`
Expected: FAIL — `TypeError` on `max_tokens` kwarg reaching `FakeClient` before signature extension is complete; then `assert [None,...,None] == [...,2200]` once signatures accept but orchestrator doesn't send budgets.

- [ ] **Step 3: Implement in `agent/orchestrator.py`**

a) Add `import math` to the imports; ensure `CritiquePolicy` is imported (added in Task 2).

b) `_draft` returns the full `ChatResult` (thinking-parity comment preserved):

```python
    def _draft(self, question: str, strategy: str, context: str, model: str):
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        # No disable_thinking: keep the client default so the draft has the
        # same reasoning budget as the single-call baseline (Strategy.process).
        return self.client.chat_completion(messages, model=model)
```

c) `_REVISE_SYSTEM_TEMPLATE` gains surgical constraints:

```python
_REVISE_SYSTEM_TEMPLATE = """{context}

You authored the draft answer below. Reviewers found issues in it. Produce an
improved final version that resolves every issue while keeping the overall
structure and all correct content. State important assumptions explicitly;
never present invented numbers or facts as established requirements.

Resolve ONLY the listed issues with targeted edits. Do NOT expand the answer
with new sections; keep the final version on the same scale as the draft.
Preserve all correct content that no issue implicates.
"""
```

d) `_run_critique` extracts draft text + token count and passes both through:

```python
    def _run_critique(self, question: str, strategy: str, context: str, model: str, policy, intent: str) -> str:
        draft_result = self._draft(question, strategy, context, model)
        draft = draft_result.text
        draft_completion_tokens = draft_result.completion_tokens or None
        max_perspectives = self._resolve_max_perspectives(intent)
        perspectives = self._plan_perspectives(
            question, strategy, context, model, max_perspectives=max_perspectives,
        )
        issues: list[Issue] = []
        if perspectives:
            results = run_workers(
                perspectives,
                lambda p: self._critic(question, p, draft, model),
                max_workers=policy.max_workers if policy else 4,
            )
            for r in results:
                if r.error:
                    logger.warning(
                        "critic failure", task=r.task.title, role=r.task.role, error=r.error
                    )
            issues = self._consolidate(results)
        answer = draft
        if issues:
            try:
                answer = self._revise(
                    question, strategy, context, draft, issues, model,
                    draft_completion_tokens=draft_completion_tokens,
                )
            except LLMError:
                logger.warning("revise failure, returning draft")
                answer = draft
        if not (policy and policy.evaluator.enabled):
            return answer

        def improve(previous: str, feedback: list[str], round_no: int) -> str:
            judge_issues = [
                Issue(severity="high", description=f"Judge scored too low - {f}", suggestion="")
                for f in feedback
            ]
            return self._revise(question, strategy, context, previous, judge_issues, model)

        return self._evaluate_loop(
            question, strategy, context, answer, model, policy.evaluator, improve=improve,
        )
```

(The judge-feedback `improve` passes no token count — the previous answer's length is unknown, so the budget degrades to the configured cap.)

e) `_revise` computes and sends the budget:

```python
    def _revise(self, question: str, strategy: str, context: str, draft: str, issues: list[Issue],
                model: str, draft_completion_tokens: int | None = None) -> str:
        policy = self.domain.orchestration
        critique = policy.critique if policy and policy.critique else CritiquePolicy()
        if draft_completion_tokens:
            raw = math.ceil(draft_completion_tokens * critique.revise_token_ratio)
            max_tokens = max(critique.revise_min_tokens, min(raw, critique.revise_max_tokens))
        else:
            max_tokens = critique.revise_max_tokens
        lines = "\n".join(
            f"- [{i.severity}] {i.description}" + (f" Suggestion: {i.suggestion}" if i.suggestion else "")
            for i in issues
        )
        system = _REVISE_SYSTEM_TEMPLATE.format(context=context)
        user_content = (
            f"User question:\n{question}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Reviewer issues to resolve:\n{lines}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(
            messages, model=model, disable_thinking=True, max_tokens=max_tokens,
        ).text
```

- [ ] **Step 4: Update observability wrapper in `agent/observability/patch.py`**

```python
    def _wrap_revise(self, original, key):
        def wrapper(orch, question, strategy, context, draft, issues, model,
                    draft_completion_tokens=None):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, draft, issues, model,
                                draft_completion_tokens=draft_completion_tokens)
            with phase(inst._phase(key)):
                answer = original(orch, question, strategy, context, draft, issues, model,
                                  draft_completion_tokens=draft_completion_tokens)
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, inst._phase(key), {
                        "issues": len(issues),
                        "draft_completion_tokens": draft_completion_tokens})
                return answer
        return wrapper
```

Note: `_wrap_direct` (used for `_draft`) forwards args positionally and returns whatever the original returns, so `_draft` returning `ChatResult` needs NO wrapper change.

- [ ] **Step 5: Run the orchestrator + observability suites**

Run: `uv run pytest tests/unit/test_orchestrator.py tests/unit/test_observability_patch.py tests/unit/test_evaluation_compare.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/orchestrator.py agent/observability/patch.py tests/unit/test_orchestrator.py
git commit -m "feat: bounded revise output via max_tokens budget (measure C)"
```

---

### Task 6: Full unit suite green + live verification handoff

**Files:**
- No source changes expected. If anything fails, fix before committing.

- [ ] **Step 1: Run the whole hermetic unit suite**

Run: `uv run pytest -q`
Expected: all PASS (baseline was 371+; now includes the new tests from Tasks 1–5). map_reduce coverage (planner/workers/aggregator/reaggregate tests) must be untouched-green.

- [ ] **Step 2: Verify domain-agnostic guarantee**

Run: `uv run pytest tests/unit/test_domain_agnostic.py -q`
Expected: PASS — temp domains built without a `critique:` block still load (optional block, default cap 3 applies at runtime).

- [ ] **Step 3: Commit (only if fixes were needed)**

```bash
git status   # confirm clean tree for tracked sources
```

If Step 1 required fixes: stage exactly those files and `git commit -m "fix: address review fallout from critique cost trim"`. Otherwise no commit.

- [ ] **Step 4: Live compare verification (manual, needs API keys)**

This step needs `AGENT_API_KEY` (and `AGENT_JUDGE_API_KEY`); auto-skipped otherwise. Run AFTER merging/deploying, per spec §10:

```bash
uv run python -m agent.evaluation compare --ids se-129 se-1 se-2 se-3 se-4
```

Gates (spec §10): orchestrated quality ≥ baseline quality on every case; se-129 token increase ≤300%; se-1..se-4 increase not worse than current state. Expected rollout order B → A → C was implemented in one branch here; if the cost gate fails, next levers per spec §8 are issue top-K (§7.3) and dropping `architecture_design` perspectives to 2 — do NOT touch draft thinking.

---

## Self-Review Notes

- Spec coverage: §5 (A) → Task 3; §6 (B) → Tasks 1+2 (incl. §6.3 default-3 decision via `or CritiquePolicy()`, truncation guarantee, per-intent override, yaml block); §7.1–7.2 (C) → Tasks 4+5 (ratio/min/max clamp, unavailable-count fallback to cap, prompt constraints); §11 tests → distributed per task; §12 AC#1→Task 1, AC#2→Task 3, AC#3→Task 6 live gates, AC#4→Task 6 Step 1. §7.3 issue top-K intentionally excluded (spec: optional, next knife).
- Signature-change audit: `_plan_perspectives`, `_critic`, `_revise`, `_draft` all have matching updates in `agent/observability/patch.py` within their tasks (`_wrap_direct` needs none for `_draft`). Test fakes extended in Task 5 Step 1 BEFORE orchestrator sends `max_tokens`.
- Type consistency: `CritiquePolicy` attribute names identical across Task 1 (definition), Task 2 (`default_max_perspectives`, `max_perspectives_by_intent`), Task 5 (`revise_*`). `max_perspectives` keyword-only everywhere it appears.
