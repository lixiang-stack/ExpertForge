# Critique Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace map-reduce orchestration with a single-author "critique" topology (draft → parallel critics → conditional revise) for coupled-synthesis tasks, fixing quality regression (spec: `docs/superpowers/specs/2026-08-21-refactor-orchestration.md`).

**Architecture:** New `topology` field in `OrchestrationPolicy` selects between the existing `map_reduce` path (default, untouched behavior) and a new `critique` path in `Orchestrator`. Critique reuses `run_workers` for parallel critics and the existing evaluator loop via an injected `improve` callable. Observability gains four new phase mappings; the report groups critics like workers.

**Tech Stack:** Python ≥ 3.10, pytest, PyYAML domain configs, no new dependencies.

## Global Constraints

- Every module starts with `from __future__ import annotations`.
- Logging only via `get_logger("<component>")` + structlog-style kwargs (`logger.info("event", key=value)`).
- Observability/logging must never break business logic: failure paths degrade to `warnings.warn`, never raise.
- Unit suite is hermetic (`uv run pytest -q`, testpaths=tests/unit); never call real LLMs in unit tests.
- No lint/typecheck tooling configured — verification is the unit suite.
- Domains are data: adding a domain requires zero code changes (`tests/unit/test_domain_agnostic.py` enforces).
- The `map_reduce` path remains the default and its behavior/tests stay intact.
- Draft calls must NOT pass `disable_thinking` (parity with baseline `Strategy.process`); planner/critic/revise calls MUST pass `disable_thinking=True`.

---

### Task 1: `topology` field in OrchestrationPolicy

**Files:**
- Modify: `agent/config.py:278-284` (OrchestrationPolicy dataclass)
- Modify: `agent/domain_config.py:22-23` (docstring) and `agent/domain_config.py:99-136` (`_parse_orchestration`)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `OrchestrationPolicy.topology: str = "map_reduce"`; `_parse_orchestration` accepts `topology: critique | map_reduce` in `orchestration.yaml`, raises `ConfigError` on other values.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config.py` (the file already imports `load_domain_config`, `ConfigError`, `pytest`, and defines `ORCHESTRATION_YAML` + `_write_domain`):

```python
def test_load_domain_config_topology_defaults_to_map_reduce(tmp_path):
    domain = load_domain_config(_write_domain(tmp_path))
    assert domain.orchestration.topology == "map_reduce"


def test_load_domain_config_topology_critique(tmp_path):
    base = tmp_path / "domain"
    _write_domain(tmp_path)
    (base / "orchestration.yaml").write_text(
        ORCHESTRATION_YAML + "topology: critique\n", encoding="utf-8"
    )
    domain = load_domain_config(str(base))
    assert domain.orchestration.topology == "critique"


def test_load_domain_config_topology_invalid_raises(tmp_path):
    base = tmp_path / "domain"
    _write_domain(tmp_path)
    (base / "orchestration.yaml").write_text(
        ORCHESTRATION_YAML + "topology: bogus\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_domain_config(str(base))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py -q -k topology`
Expected: FAIL — `AttributeError: 'OrchestrationPolicy' object has no attribute 'topology'` (first test), then ConfigError not raised (third test).

- [ ] **Step 3: Implement**

In `agent/config.py`, add the field to `OrchestrationPolicy`:

```python
@dataclass
class OrchestrationPolicy:
    enabled: bool = True
    min_complexity: str = "complex"
    intents: list[str] = field(default_factory=list)
    max_workers: int = 4
    topology: str = "map_reduce"
    evaluator: EvaluatorPolicy = field(default_factory=EvaluatorPolicy)
```

In `agent/domain_config.py`, inside `_parse_orchestration`, after the `max_workers` validation and before the `ev = orch_data.get("evaluator")` line, add:

```python
    topology = orch_data.get("topology", "map_reduce")
    if topology not in ("map_reduce", "critique"):
        raise ConfigError(
            f"orchestration.yaml 'topology' must be 'map_reduce' or 'critique': {orch_path}"
        )
```

and add `topology=topology,` to the `OrchestrationPolicy(...)` construction (after `max_workers=max_workers,`).

Update the module docstring listing of `orchestration.yaml` keys (line ~23) from `(enabled, min_complexity, intents, max_workers, evaluator)` to `(enabled, min_complexity, intents, max_workers, topology, evaluator)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py -q`
Expected: PASS (all, including pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add agent/config.py agent/domain_config.py tests/unit/test_config.py
git commit -m "feat: topology field in orchestration config (map_reduce|critique)"
```

---

### Task 2: Refactor `_evaluate_loop` to accept an `improve` callable

Behavior-preserving refactor. This decouples the judge loop from worker results so Task 4 can reuse it for the critique topology.

**Files:**
- Modify: `agent/orchestrator.py:85-135` (`run` tail and `_evaluate_loop`)
- Test: `tests/unit/test_orchestrator.py` (existing tests are the safety net)

**Interfaces:**
- Produces: `_evaluate_loop(self, question: str, strategy: str, context: str, answer: str, model: str, evaluator, improve) -> str` where `improve(previous: str, feedback: list[str], round_no: int) -> str`. Later tasks rely on exactly this signature.

- [ ] **Step 1: Refactor `_evaluate_loop`**

Replace the current `_evaluate_loop` (which takes `results` and calls `_reaggregate` directly) with:

```python
    def _evaluate_loop(
        self, question: str, strategy: str, context: str,
        answer: str, model: str, evaluator,
        improve,
    ) -> str:
        judge = Judge(self.client, self._judge_name())
        threshold = evaluator.min_dimension_score
        for round_no in range(evaluator.max_rounds + 1):
            scorecard = self._evaluate(judge, question, answer)
            if scorecard is None:
                return answer
            if all(score >= threshold for score in scorecard.values()):
                return answer
            if round_no == evaluator.max_rounds:
                return answer
            feedback = [f"{dim}: {score}/5" for dim, score in scorecard.items() if score < threshold]
            try:
                answer = improve(answer, feedback, round_no)
            except LLMError:
                return answer
        return answer
```

Update the call site at the end of `run`:

```python
        if not (policy and policy.evaluator.enabled):
            return answer
        return self._evaluate_loop(
            question, route.strategy, context, answer, model, policy.evaluator,
            improve=lambda previous, feedback, round_no: self._reaggregate(
                question, route.strategy, context, results, previous, feedback, round_no, model),
        )
```

- [ ] **Step 2: Run the orchestrator suite to verify no behavior change**

Run: `uv run pytest tests/unit/test_orchestrator.py -q`
Expected: PASS — all existing tests, including `test_run_evaluator_fail_optimizes_once` and `test_run_reaggregate_llm_error_returns_previous`.

- [ ] **Step 3: Commit**

```bash
git add agent/orchestrator.py
git commit -m "refactor: _evaluate_loop takes injected improve callable"
```

---

### Task 3: Critique topology core (evaluator disabled paths)

**Files:**
- Modify: `agent/orchestrator.py`
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `WorkerTask`/`WorkerResult`/`run_workers` from `agent.worker_pool`; `parse_json` from `agent.parsing`; `LLMError`; `OrchestrationPolicy.topology` (Task 1); `_evaluate_loop(..., improve)` signature (Task 2).
- Produces:
  - `@dataclass Issue(severity: str, description: str, suggestion: str)`
  - `Orchestrator._draft(question, strategy, context, model) -> str`
  - `Orchestrator._plan_perspectives(question, strategy, context, model) -> list[WorkerTask] | None`
  - `Orchestrator._critic(question, perspective: WorkerTask, context, draft, model) -> str` (raw JSON text)
  - `Orchestrator._consolidate(results: list[WorkerResult]) -> list[Issue]`
  - `Orchestrator._revise(question, strategy, context, draft, issues: list[Issue], model) -> str`
  - `Orchestrator._run_critique(question, strategy, context, model, policy) -> str`
  - `Orchestrator.run` dispatches on `policy.topology` (`critique` → `_run_critique`, else `_run_map_reduce`)

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_orchestrator.py`:

```python
_CRITIQUE_PLAN_JSON = (
    '{"perspectives": [{"title": "consistency", "focus": "internal contradictions", "role": "Consistency Reviewer"},'
    ' {"title": "feasibility", "focus": "operational feasibility", "role": "Feasibility Reviewer"}]}'
)
_ISSUES_JSON = '{"issues": [{"severity": "high", "description": "contradictory deployment modes", "suggestion": "pick one"}]}'
_ISSUES_EMPTY = '{"issues": []}'


def _critique_domain(evaluator=None):
    return DomainConfig(
        name="sw",
        description="software engineering",
        out_of_domain_reply="Out.",
        intents={"troubleshooting": IntentDef("troubleshooting", "debug")},
        intent_mapping={"troubleshooting": "debugging"},
        strategies=["debugging"],
        prompts={"debugging": "Debugging system prompt."},
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4, topology="critique",
            evaluator=evaluator or EvaluatorPolicy(enabled=True, min_dimension_score=3, max_rounds=1),
        ),
    )


class CriticFailingClient(FakeClient):
    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        if "You are a reviewer" in messages[0]["content"]:
            self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
            raise LLMError("critic boom")
        return super().chat_completion(messages, model=model, disable_thinking=disable_thinking,
                                       json_mode=json_mode, json_schema=json_schema)


def test_run_critique_full_flow_revises_on_issues():
    client = FakeClient([
        "draft answer",
        _CRITIQUE_PLAN_JSON,
        _ISSUES_JSON,
        _ISSUES_EMPTY,
        "revised answer",
    ])
    result = Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert result == "revised answer"
    assert len(client.calls) == 5
    # draft keeps client-default thinking behaviour (parity with Strategy.process)
    assert client.calls[0][2] is False
    assert client.calls[0][0][0]["content"] == "Debugging system prompt."
    # perspectives planner call uses json_schema intent and disables thinking
    _, _, p_dt, p_jm, p_schema = client.calls[1]
    assert p_schema is not None and p_jm is False and p_dt is True
    # critic calls disable thinking and express json_schema intent
    critic_calls = [c for c in client.calls[2:4] if "You are a reviewer" in c[0][0]["content"]]
    assert len(critic_calls) == 2
    for _, _, dt, _, schema in critic_calls:
        assert dt is True and schema is not None
    # revise is a single-author call over draft + findings
    assert "You authored the draft" in client.calls[4][0][0]["content"]
    revise_user = client.calls[4][0][-1]["content"]
    assert "Draft answer:\ndraft answer" in revise_user
    assert "contradictory deployment modes" in revise_user


def test_run_critique_no_issues_skips_revise():
    client = FakeClient(["draft answer", _CRITIQUE_PLAN_JSON, _ISSUES_EMPTY, _ISSUES_EMPTY])
    result = Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 4
    assert not any("You authored the draft" in c[0][0]["content"] for c in client.calls)


def test_run_critique_perspectives_invalid_returns_draft():
    client = FakeClient(["draft answer", "not json"])
    result = Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 2


def test_run_critique_all_critics_fail_returns_draft():
    client = CriticFailingClient(["draft answer", _CRITIQUE_PLAN_JSON])
    result = Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 4


def test_run_critique_critic_bad_json_yields_no_issues_from_that_critic():
    client = FakeClient([
        "draft answer",
        _CRITIQUE_PLAN_JSON,
        "not json",
        _ISSUES_JSON,
        "revised answer",
    ])
    result = Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert result == "revised answer"
    revise_calls = [c for c in client.calls if "You authored the draft" in c[0][0]["content"]]
    assert len(revise_calls) == 1
    assert "contradictory deployment modes" in revise_calls[0][0][-1]["content"]


def test_run_critique_revise_llm_error_returns_draft():
    client = CallRaisingClient(
        ["draft answer", _CRITIQUE_PLAN_JSON, _ISSUES_JSON, _ISSUES_EMPTY, "unused"],
        raise_on_call=4,
    )
    result = Orchestrator(client, _config(), _critique_domain(evaluator=EvaluatorPolicy(enabled=False))).run(
        "huge task", _route(), "high-a")
    assert result == "draft answer"
```

Note on ordering: the two critic calls run concurrently inside `run_workers`, but they always occupy `client.calls[2]` and `client.calls[3]` because the draft and perspectives calls complete before workers start, and revise starts only after both critics finish. Tests that need to tell critics apart match on system-prompt content, never on index order.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator.py -q -k critique`
Expected: FAIL — all six tests fail because the topology dispatch does not exist yet (`run` executes the map-reduce path, so e.g. the second response `"not json"` goes to `_plan` and the result is a direct answer instead of `"revised answer"` / `"draft answer"`).

- [ ] **Step 3: Implement**

In `agent/orchestrator.py`:

Add import at top:

```python
from dataclasses import dataclass
```

Add after the imports / near `WorkerTask` usage:

```python
@dataclass
class Issue:
    severity: str
    description: str
    suggestion: str
```

Add prompts next to `_PLANNER_PROMPT`:

```python
_PERSPECTIVES_PROMPT = """You are a review planner for an expert domain named {name}.

{description}

Task context:
{context}

Rules:
- Plan 2-4 distinct review perspectives for checking a draft expert answer.
- Each perspective must be verifiable by reading the user's question and the
  draft alone (e.g. Consistency & Coherence, Feasibility & Operations,
  Compliance & Security, Cost).
- Assign each perspective a distinct role name that defines its focused responsibility.
- Output ONLY a single JSON object: {{"perspectives": [{{"title": "...", "focus": "...", "role": "..."}}]}}
"""

_CRITIC_SYSTEM_TEMPLATE = """{context}

You are a reviewer. Your review perspective: {role}
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

_REVISE_SYSTEM_TEMPLATE = """{context}

You authored the draft answer below. Reviewers found issues in it. Produce an
improved final version that resolves every issue while keeping the overall
structure and all correct content. State important assumptions explicitly;
never present invented numbers or facts as established requirements.
"""
```

Add schema helpers next to `_planner_schema`:

```python
def _perspectives_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "perspectives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "focus": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["title", "focus", "role"],
                },
            }
        },
        "required": ["perspectives"],
    }


def _issues_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                        "description": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["severity", "description", "suggestion"],
                },
            }
        },
        "required": ["issues"],
    }


def _parse_issues(text: str | None) -> list[Issue]:
    data = parse_json(text) if text else None
    if not data or not isinstance(data.get("issues"), list):
        return []
    issues: list[Issue] = []
    for item in data["issues"]:
        if not isinstance(item, dict):
            continue
        description = item.get("description")
        if not isinstance(description, str) or not description:
            continue
        severity = item.get("severity")
        if not isinstance(severity, str) or not severity:
            severity = "medium"
        suggestion = item.get("suggestion")
        if not isinstance(suggestion, str):
            suggestion = ""
        issues.append(Issue(severity=severity, description=description, suggestion=suggestion))
    return issues
```

Restructure `Orchestrator.run` into a dispatcher plus two private flows. Replace the body of `run` (keep the log line) with:

```python
    def run(self, question: str, route: RouteResult, model: str) -> str:
        logger.info("orchestration start", strategy=route.strategy, model=model)
        context = self._strategy_context(route.strategy)
        policy = self.domain.orchestration
        topology = policy.topology if policy else "map_reduce"
        if topology == "critique":
            return self._run_critique(question, route.strategy, context, model, policy)
        return self._run_map_reduce(question, route, context, model, policy)

    def _run_map_reduce(self, question: str, route: RouteResult, context: str, model: str, policy) -> str:
        tasks = self._plan(question, route.strategy, context, model)
        if tasks is None:
            return self._direct_answer(question, route.strategy, context, model)
        results = run_workers(
            tasks,
            lambda task: self._worker(question, task, context, model),
            max_workers=policy.max_workers if policy else 4,
        )
        for r in results:
            if r.error:
                logger.warning(
                    "worker failure", task=r.task.title, role=r.task.role, error=r.error
                )
        if all(r.error for r in results):
            return self._direct_answer(question, route.strategy, context, model)
        answer = self._aggregate(question, route.strategy, context, results, model)
        if not (policy and policy.evaluator.enabled):
            return answer
        return self._evaluate_loop(
            question, route.strategy, context, answer, model, policy.evaluator,
            improve=lambda previous, feedback, round_no: self._reaggregate(
                question, route.strategy, context, results, previous, feedback, round_no, model),
        )
```

Add the critique methods to the class (in this task `_run_critique` ends after the revise step; Task 4 inserts the evaluator block before the final return):

```python
    def _run_critique(self, question: str, strategy: str, context: str, model: str, policy) -> str:
        draft = self._draft(question, strategy, context, model)
        perspectives = self._plan_perspectives(question, strategy, context, model)
        issues: list[Issue] = []
        if perspectives:
            results = run_workers(
                perspectives,
                lambda p: self._critic(question, p, context, draft, model),
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
                answer = self._revise(question, strategy, context, draft, issues, model)
            except LLMError:
                logger.warning("revise failure, returning draft")
                answer = draft
        return answer

    def _draft(self, question: str, strategy: str, context: str, model: str) -> str:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        # No disable_thinking: keep the client default so the draft has the
        # same reasoning budget as the single-call baseline (Strategy.process).
        return self.client.chat_completion(messages, model=model).text

    def _plan_perspectives(
        self, question: str, strategy: str, context: str, model: str
    ) -> list[WorkerTask] | None:
        prompt = _PERSPECTIVES_PROMPT.format(
            name=self.domain.name,
            description=self.domain.description,
            context=context,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        result = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_perspectives_schema()
        )
        data = parse_json(result.text)
        if not data or not isinstance(data.get("perspectives"), list):
            return None
        perspectives: list[WorkerTask] = []
        for item in data["perspectives"]:
            if not isinstance(item, dict):
                return None
            title = item.get("title")
            focus = item.get("focus")
            if not isinstance(title, str) or not isinstance(focus, str):
                return None
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = title
            perspectives.append(WorkerTask(title=title, instruction=focus, role=role))
        return perspectives or None

    def _critic(self, question: str, perspective: WorkerTask, context: str, draft: str, model: str) -> str:
        system = _CRITIC_SYSTEM_TEMPLATE.format(
            context=context, role=perspective.role, instruction=perspective.instruction,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"User question:\n{question}\n\nDraft answer:\n{draft}"},
        ]
        return self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_issues_schema()
        ).text

    def _consolidate(self, results: list[WorkerResult]) -> list[Issue]:
        issues: list[Issue] = []
        for r in results:
            if r.error or r.text is None:
                continue
            issues.extend(_parse_issues(r.text))
        return issues

    def _revise(self, question: str, strategy: str, context: str, draft: str, issues: list[Issue], model: str) -> str:
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
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_orchestrator.py -q`
Expected: PASS — all new critique tests and all pre-existing map-reduce tests.

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat: critique topology (draft -> parallel critics -> conditional revise)"
```

---

### Task 4: Wire the evaluator loop into critique

When the internal judge scores below threshold, the improvement path for critique is another single-author revise seeded with the judge's dimension feedback (instead of map-reduce's re-aggregate).

**Files:**
- Modify: `agent/orchestrator.py` (`_run_critique` tail)
- Test: `tests/unit/test_orchestrator.py`

**Interfaces:**
- Consumes: `_evaluate_loop(..., improve)` (Task 2), `_revise` (Task 3), `Issue`.
- Produces: critique honors `policy.evaluator` identically to map-reduce (same judge, same threshold, same max_rounds semantics).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_orchestrator.py`:

```python
def test_run_critique_evaluator_low_score_revises_with_judge_feedback():
    client = FakeClient([
        "draft answer",
        _CRITIQUE_PLAN_JSON,
        _ISSUES_EMPTY,
        _ISSUES_EMPTY,
        _SCORECARD_LOW,
        "judge-improved answer",
        _SCORECARD_PASS,
    ])
    result = Orchestrator(client, _config(), _critique_domain()).run("huge task", _route(), "high-a")
    assert result == "judge-improved answer"
    assert len(client.calls) == 7
    # 5th call is the judge scoring the draft
    assert client.calls[4][0][-1]["content"] == "huge task"
    # 6th call is a revise seeded with the judge feedback
    assert "You authored the draft" in client.calls[5][0][0]["content"]
    revise_user = client.calls[5][0][-1]["content"]
    assert "correctness: 2/5" in revise_user
    assert "Draft answer:\ndraft answer" in revise_user


def test_run_critique_evaluator_passes_returns_answer_unchanged():
    client = FakeClient([
        "draft answer",
        _CRITIQUE_PLAN_JSON,
        _ISSUES_EMPTY,
        _ISSUES_EMPTY,
        _SCORECARD_PASS,
    ])
    result = Orchestrator(client, _config(), _critique_domain()).run("huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator.py -q -k "critique and evaluator"`
Expected: FAIL — first test returns `"draft answer"` instead of `"judge-improved answer"` (evaluator block missing in `_run_critique`).

- [ ] **Step 3: Implement**

In `_run_critique`, replace the final `return answer` with:

```python
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_orchestrator.py -q`
Expected: PASS — all tests.

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat: critique topology honors evaluator loop via revise"
```

---

### Task 5: Observability phases for critique + report grouping

**Files:**
- Modify: `agent/observability/patch.py` (DEFAULT_PHASES ~line 46, factories dict ~line 95, wrapper factories, targets ~line 294)
- Modify: `agent/observability/report_data.py` (`_decision_type` ~line 138, `_stage_title` ~line 201, `group_stages` ~line 246)
- Test: `tests/unit/test_observability_patch.py`, `tests/unit/test_report_data.py`

**Interfaces:**
- Consumes: `Orchestrator._draft/_plan_perspectives/_critic/_revise` signatures from Task 3.
- Produces: phase names `orchestration.draft`, `orchestration.planner` (reused for perspectives), `orchestration.critic.N`, `orchestration.reviser`; critic decisions render as worker-type steps grouped under one `orchestration.critic` stage.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_observability_patch.py`, extend the `_domain_complex` helper signature to accept topology (keep default so existing callers are unchanged):

```python
def _domain_complex(evaluator=None, topology="map_reduce"):
    return DomainConfig(
        name="sw", description="desc", out_of_domain_reply="Out.",
        intents={
            "faq": IntentDef("faq", "quick"),
            "troubleshooting": IntentDef("troubleshooting", "debug"),
        },
        intent_mapping={"faq": "direct", "troubleshooting": "debugging"},
        strategies=["direct", "debugging"],
        orchestration=OrchestrationPolicy(
            enabled=True, min_complexity="complex", intents=["troubleshooting"],
            max_workers=4, topology=topology,
            evaluator=evaluator or EvaluatorPolicy(enabled=False),
        ),
        prompts={
            "direct": "Direct prompt.",
            "debugging": "Debugging prompt.",
        },
    )
```

Add constants next to `_PLAN`:

```python
_CRITIQUE_PLAN = ('{"perspectives": [{"title": "consistency", "focus": "contradictions", "role": "R1"},'
                  ' {"title": "feasibility", "focus": "operations", "role": "R2"}]}')
_ISSUES_EMPTY = '{"issues": []}'
```

Add the test:

```python
def test_install_wraps_critique_phases(tmp_path):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY_COMPLEX, "draft answer", _CRITIQUE_PLAN, _ISSUES_EMPTY, _ISSUES_EMPTY])
    chat = Chat(inner, _config(), _domain_complex(topology="critique"))
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("huge debugging task")

    assert resp.text == "draft answer"
    assert "orchestration.draft" in inner.seen_phases
    assert "orchestration.planner" in inner.seen_phases
    assert "orchestration.critic.1" in inner.seen_phases
    assert "orchestration.critic.2" in inner.seen_phases
    events, _ = read_events(tmp_path / "obs")
    critic_decisions = [
        e for e in events
        if e["type"] == "decision" and e["phase"].startswith("orchestration.critic")
    ]
    assert {e["data"]["task"] for e in critic_decisions} == {"consistency", "feasibility"}
```

In `tests/unit/test_report_data.py`, extend the import from `agent.observability.report_data` to include `build_timeline` and `group_stages`, then add:

```python
def test_build_timeline_critic_decision_type():
    tl = build_timeline([{"type": "decision", "trace_id": "a", "phase": "orchestration.critic.1",
                          "ts": 1, "data": {"task": "consistency"}}])
    steps = tl["a"]
    assert steps[0].kind == "decision"
    assert steps[0].detail["type"] == "worker"
    assert steps[0].detail["data"]["task"] == "consistency"


def test_group_stages_groups_critics_like_workers():
    events = [
        {"type": "trace_start", "trace_id": "a", "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.draft", "model": "m",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "decision", "trace_id": "a", "phase": "orchestration.critic.1", "ts": 20,
         "data": {"task": "consistency", "role": "R1"}},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.critic.1", "model": "m",
         "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
         "latency_ms": 50, "status": "ok", "ts": 30},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.critic.2", "model": "m",
         "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
         "latency_ms": 50, "status": "ok", "ts": 40},
    ]
    stages = group_stages(build_timeline(events))["a"]
    assert [s.title for s in stages] == ["orchestration.draft", "orchestration.critic"]
    critic_stage = stages[1]
    assert [w.number for w in critic_stage.workers] == [1, 2]
    assert critic_stage.workers[0].task_title == "consistency"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_observability_patch.py::test_install_wraps_critique_phases tests/unit/test_report_data.py -q`
Expected: FAIL — `orchestration.draft` not in seen_phases (no patch entry); report tests fail on stage titles/grouping.

- [ ] **Step 3: Implement patch.py**

In `DEFAULT_PHASES` add:

```python
    "Orchestrator._draft": "orchestration.draft",
    "Orchestrator._plan_perspectives": "orchestration.planner",
    "Orchestrator._critic": "orchestration.critic",
    "Orchestrator._revise": "orchestration.reviser",
```

In the `factories` dict inside `_wrap` add:

```python
            "Orchestrator._draft": self._wrap_direct,
            "Orchestrator._plan_perspectives": self._wrap_plan_perspectives,
            "Orchestrator._critic": self._wrap_critic,
            "Orchestrator._revise": self._wrap_revise,
```

(`_wrap_direct` is reused for `_draft`: identical `(orch, question, strategy, context, model)` signature.)

Add three wrapper factories next to the existing ones:

```python
    def _wrap_plan_perspectives(self, original, key):
        def wrapper(orch, question, strategy, context, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, model)  # passthrough: real business call
            with phase(inst._phase(key)):
                perspectives = original(orch, question, strategy, context, model)  # <-- real business call
                tid = current_trace_id()
                if tid:
                    data = {"degraded": True} if perspectives is None else {
                        "tasks": [{"title": t.title, "instruction": t.instruction,
                                   "role": t.role} for t in perspectives]}
                    inst._record_decision(tid, inst._phase(key), data)
                return perspectives
        return wrapper

    def _wrap_critic(self, original, key):
        def wrapper(orch, question, perspective, context, draft, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, perspective, context, draft, model)  # passthrough: real business call
            base = inst._phase(key)
            n = inst._next_worker(current_trace_id() or "")
            with phase(f"{base}.{n}"):
                tid = current_trace_id()
                try:
                    result = original(orch, question, perspective, context, draft, model)  # <-- real business call
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

    def _wrap_revise(self, original, key):
        def wrapper(orch, question, strategy, context, draft, issues, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, draft, issues, model)  # passthrough: real business call
            with phase(inst._phase(key)):
                answer = original(orch, question, strategy, context, draft, issues, model)  # <-- real business call
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, inst._phase(key), {"issues": len(issues)})
                return answer
        return wrapper
```

In `apply()` targets list add:

```python
            ("Orchestrator._draft", Orchestrator, "_draft"),
            ("Orchestrator._plan_perspectives", Orchestrator, "_plan_perspectives"),
            ("Orchestrator._critic", Orchestrator, "_critic"),
            ("Orchestrator._revise", Orchestrator, "_revise"),
```

- [ ] **Step 4: Implement report_data.py**

In `_decision_type`, after the `orchestration.worker` check add:

```python
    if phase.startswith("orchestration.critic"):
        return "worker"
```

In `_stage_title`, after the `orchestration.worker.` check add:

```python
    if ph.startswith("orchestration.critic"):
        return "orchestration.critic"
```

In `group_stages`, change the worker-group condition to include critics:

```python
            if title in ("orchestration.worker", "orchestration.critic"):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_observability_patch.py tests/unit/test_report_data.py tests/unit/test_tracing.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/observability/patch.py agent/observability/report_data.py tests/unit/test_observability_patch.py tests/unit/test_report_data.py
git commit -m "feat: observability phases and report grouping for critique topology"
```

---

### Task 6: Enable critique for software_engineering + full regression

**Files:**
- Modify: `domain/software_engineering/orchestration.yaml`
- Test: full unit suite incl. `tests/unit/test_domain_agnostic.py`

**Interfaces:**
- Consumes: everything above. No code changes — domains are data.

- [ ] **Step 1: Flip the domain config**

Edit `domain/software_engineering/orchestration.yaml` to:

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
```

- [ ] **Step 2: Run the full unit suite**

Run: `uv run pytest -q`
Expected: PASS — including `test_domain_agnostic.py` (its synthetic domain writes its own `orchestration.yaml` without `topology`, exercising the default) and all map-reduce orchestrator tests.

- [ ] **Step 3: Commit**

```bash
git add domain/software_engineering/orchestration.yaml
git commit -m "feat: enable critique topology for software_engineering domain"
```

---

## Manual live verification (spec §5 — not part of hermetic tasks)

Requires `AGENT_API_KEY` and `AGENT_JUDGE_API_KEY` (exits 1 with `Config error:` if the judge key is missing). After Task 6:

```bash
# critique vs baseline on the regression case
uv run python -m agent.evaluation compare --ids se-129

# A/B old vs new topology: temporarily set topology: map_reduce, rerun, restore
uv run python -m agent.evaluation compare --ids se-129

# broader sample
uv run python -m agent.evaluation compare   # curated full_expert cases
```

Success criteria (spec §5): orch Q ≥ base Q (gain ≥ 0); no recurrence of the four contradiction classes (deployment mode / RPO tiers / transaction scheme / architecture style); token increase ≤300% reported honestly. Inspect answers in `evaluation/results/*.json` (`cases[].orchestrated.answer`) for the contradiction checklist.
