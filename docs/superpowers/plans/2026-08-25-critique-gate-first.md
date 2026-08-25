# Critique Gate-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the critique topology conditional: when `evaluator.enabled`, the draft is judged FIRST — passing drafts return immediately (≈ base cost), only failing scorecards trigger planner/critics/revise.

**Architecture:** Two commits. Task 1 is a behavior-preserving extraction of the perspectives→critics→consolidate→conditional-revise sequence into `_critique_pass` (all 391 tests pass unmodified). Task 2 rewires `_run_critique`'s `evaluator.enabled=true` branch to feed the raw draft into the existing `_evaluate_loop`, with round-0 `improve` running the full critique pass and rounds ≥1 keeping the existing judge-feedback revise. Task 3 updates docs.

**Tech Stack:** Python ≥ 3.10, pytest. No new dependencies, no new config keys, no signature changes outside `agent/orchestrator.py`.

**Spec:** `docs/superpowers/specs/2026-08-25-critique-gate-first-design.md` (Approved).

## Global Constraints

- Only `agent/orchestrator.py`, `tests/unit/test_orchestrator.py`, and `AGENTS.md` may be modified across all tasks. `agent/observability/patch.py`, `agent/llm.py`, `agent/config.py`, `agent/domain_config.py` are ZERO-change files.
- The map_reduce path (`_run_map_reduce`, `_plan`, `_worker`, `_aggregate`, `_direct_answer`, `_reaggregate`) is untouched.
- `_evaluate_loop`, `_evaluate`, `_revise`, `_critic`, `_plan_perspectives`, `_draft`, `_resolve_max_perspectives` bodies are NOT edited in this plan — Task 1 only MOVES lines into `_critique_pass`; Task 2 only edits `_run_critique`.
- `evaluator.enabled=false` behavior must stay byte-equivalent: every existing test using `_critique_domain(evaluator=EvaluatorPolicy(enabled=False))` passes WITHOUT modification.
- Gate semantics (exact): scorecard None ⇒ treated as pass (return draft); all dimensions ≥ `min_dimension_score` ⇒ return draft; otherwise improve runs. Round 0 improve = critique pass; rounds ≥1 improve = judge-feedback revise (existing code).
- Test fakes record 5-tuples `(messages, model, disable_thinking, json_mode, json_schema)` — never change tuple shape.
- Run tests with `uv run pytest -q`; targeted: `uv run pytest tests/unit/test_orchestrator.py -q`. Commit style: `feat:` / `refactor:` / `docs:`, one commit per task.

---

### Task 1: Extract `_critique_pass` (pure refactor, zero behavior change)

**Files:**
- Modify: `agent/orchestrator.py:240-283` (`_run_critique`)
- Test: none added — this task's gate is the UNMODIFIED existing suite

**Interfaces:**
- Consumes: existing `_plan_perspectives(question, strategy, context, model, *, max_perspectives)`, `_critic(question, perspective, draft, model)`, `_consolidate(results)`, `_revise(question, strategy, context, draft, issues, model, draft_completion_tokens=None)`, `_resolve_max_perspectives(intent)`, `run_workers(tasks, fn, max_workers=n)`.
- Produces: `_critique_pass(self, question: str, strategy: str, context: str, draft: str, model: str, policy, intent: str, draft_completion_tokens: int | None = None) -> str` — Task 2 calls it with exactly these names.

- [ ] **Step 1: Record the green baseline**

Run: `uv run pytest -q`
Expected: 391 passed (or current count) — note the number; it must not change in this task.

- [ ] **Step 2: Replace `_run_critique` with the extracted form**

Replace `agent/orchestrator.py` lines 240-283 with exactly:

```python
    def _run_critique(self, question: str, strategy: str, context: str, model: str, policy, intent: str) -> str:
        draft_result = self._draft(question, strategy, context, model)
        draft = draft_result.text
        draft_completion_tokens = draft_result.completion_tokens or None
        answer = self._critique_pass(
            question, strategy, context, draft, model, policy, intent,
            draft_completion_tokens=draft_completion_tokens,
        )
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

    def _critique_pass(
        self, question: str, strategy: str, context: str, draft: str, model: str,
        policy, intent: str, draft_completion_tokens: int | None = None,
    ) -> str:
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
        if not issues:
            return draft
        try:
            return self._revise(
                question, strategy, context, draft, issues, model,
                draft_completion_tokens=draft_completion_tokens,
            )
        except LLMError:
            logger.warning("revise failure, returning draft")
            return draft
```

(The moved lines are verbatim from the old `_run_critique` body; the only new logic is `if not issues: return draft` replacing the `answer = draft` / `if issues:` dance — semantically identical.)

- [ ] **Step 3: Verify zero regressions**

Run: `uv run pytest -q`
Expected: SAME count as Step 1, all passed. If anything fails, the extraction altered behavior — fix before committing.

- [ ] **Step 4: Commit**

```bash
git add agent/orchestrator.py
git commit -m "refactor: extract critique pipeline into Orchestrator._critique_pass"
```

---

### Task 2: Gate-first wiring for `evaluator.enabled=true`

**Files:**
- Modify: `agent/orchestrator.py:240-256` (`_run_critique` only)
- Test: `tests/unit/test_orchestrator.py` (rewrite 2 tests, add 7)

**Interfaces:**
- Consumes: `_critique_pass` from Task 1; `_evaluate_loop` (existing, unchanged); `BudgetRecordingClient` with `.max_tokens_seen` list (exists since the P2 cost-trim).
- Produces: gated `_run_critique`. No downstream consumers besides `Orchestrator.run`.

- [ ] **Step 1: Rewrite the two obsolete enabled=true tests and add the new gate tests**

In `tests/unit/test_orchestrator.py` DELETE `test_run_critique_evaluator_low_score_revises_with_judge_feedback` and `test_run_critique_evaluator_passes_returns_answer_unchanged` (they encode tail-judge ordering), then append:

```python
_JUDGE_MARKER = "strict evaluator"


def _gate_call_indices(client):
    judge = [i for i, c in enumerate(client.calls) if _JUDGE_MARKER in c[0][0]["content"]]
    critic = [i for i, c in enumerate(client.calls) if "You are a reviewer" in c[0][0]["content"]]
    return judge, critic


def test_run_critique_gate_pass_returns_draft_early():
    client = FakeClient(["draft answer", _SCORECARD_PASS])
    result = Orchestrator(client, _config(), _critique_domain()).run("huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 2
    judge, critic = _gate_call_indices(client)
    assert judge == [1] and critic == []


def test_run_critique_gate_fail_runs_critique_then_revise_then_rejudges():
    client = FakeClient([
        "draft answer",        # 0 draft
        _SCORECARD_LOW,        # 1 gate judge on draft -> fail
        _CRITIQUE_PLAN_JSON,   # 2 improve(0): perspectives
        _ISSUES_JSON,          # 3 critic consistency
        _ISSUES_EMPTY,         # 4 critic feasibility
        "revised answer",      # 5 budgeted revise
        _SCORECARD_PASS,       # 6 round-1 judge on revised -> pass
    ])
    result = Orchestrator(client, _config(), _critique_domain()).run("huge task", _route(), "high-a")
    assert result == "revised answer"
    judge, critic = _gate_call_indices(client)
    assert judge == [1, 6]
    assert critic == [3, 4]
    revise_user = client.calls[5][0][-1]["content"]
    assert "contradictory deployment modes" in revise_user
    assert "Draft answer:\ndraft answer" in revise_user


def test_run_critique_gate_round1_low_revises_with_judge_feedback():
    # max_rounds must be >= 2: with the frozen _evaluate_loop, improve is only
    # called for round_no < max_rounds, so judge-feedback revise (rounds >= 1)
    # is unreachable at max_rounds=1.
    domain = _critique_domain(
        evaluator=EvaluatorPolicy(enabled=True, min_dimension_score=3, max_rounds=2))
    client = FakeClient([
        "draft answer",        # 0 draft
        _SCORECARD_LOW,        # 1 gate fails
        _CRITIQUE_PLAN_JSON,   # 2 improve(0): critics find nothing
        _ISSUES_EMPTY,         # 3
        _ISSUES_EMPTY,         # 4
        _SCORECARD_LOW,        # 5 round-1 judge still low
        "judge-improved",      # 6 improve(1): judge-feedback revise
        _SCORECARD_PASS,       # 7 round-2 judge -> pass
    ])
    result = Orchestrator(client, _config(), domain).run("huge task", _route(), "high-a")
    assert result == "judge-improved"
    assert len(client.calls) == 8
    feedback_user = client.calls[6][0][-1]["content"]
    assert "correctness: 2/5" in feedback_user


def test_run_critique_gate_fail_empty_issues_exhausts_rounds_returns_draft():
    client = FakeClient([
        "draft answer",        # 0 draft
        _SCORECARD_LOW,        # 1 gate fails
        _CRITIQUE_PLAN_JSON,   # 2 perspectives
        _ISSUES_EMPTY,         # 3 critics find nothing -> improve returns draft
        _ISSUES_EMPTY,         # 4
        _SCORECARD_LOW,        # 5 round-1 judges same draft -> exhausted
    ])
    result = Orchestrator(client, _config(), _critique_domain()).run("huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 6


def test_run_critique_gate_judge_parse_failure_treated_as_pass():
    client = FakeClient(["draft answer", "not json"])
    result = Orchestrator(client, _config(), _critique_domain()).run("huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 2


def test_run_critique_gate_zero_max_rounds_single_judge_no_critique():
    domain = _critique_domain(
        evaluator=EvaluatorPolicy(enabled=True, min_dimension_score=3, max_rounds=0))
    client = FakeClient(["draft answer", _SCORECARD_LOW])
    result = Orchestrator(client, _config(), domain).run("huge task", _route(), "high-a")
    assert result == "draft answer"
    assert len(client.calls) == 2
    _, critic = _gate_call_indices(client)
    assert critic == []


def test_run_critique_gate_fail_revise_carries_p2_budget():
    client = BudgetRecordingClient(
        ["draft answer", _SCORECARD_LOW, _CRITIQUE_PLAN_JSON, _ISSUES_JSON, _ISSUES_EMPTY,
         "revised answer", _SCORECARD_PASS],
        completion_tokens=2000,
    )
    Orchestrator(client, _config(), _critique_domain()).run("huge task", _route(), "high-a")
    assert client.max_tokens_seen[5] == 2200  # ceil(2000 * 1.1), inside [1024, 6000]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator.py -q -k gate`
Expected: FAIL — current code runs critique BEFORE judging, so call counts/order do not match (e.g. gate-pass case makes 5 calls, not 2).

- [ ] **Step 3: Rewire `_run_critique`**

Replace the `_run_critique` body (Task 1 version, lines 240-256) with exactly:

```python
    def _run_critique(self, question: str, strategy: str, context: str, model: str, policy, intent: str) -> str:
        draft_result = self._draft(question, strategy, context, model)
        draft = draft_result.text
        draft_completion_tokens = draft_result.completion_tokens or None
        if not (policy and policy.evaluator.enabled):
            return self._critique_pass(
                question, strategy, context, draft, model, policy, intent,
                draft_completion_tokens=draft_completion_tokens,
            )

        def improve(previous: str, feedback: list[str], round_no: int) -> str:
            if round_no == 0:
                return self._critique_pass(
                    question, strategy, context, previous, model, policy, intent,
                    draft_completion_tokens=draft_completion_tokens,
                )
            judge_issues = [
                Issue(severity="high", description=f"Judge scored too low - {f}", suggestion="")
                for f in feedback
            ]
            return self._revise(question, strategy, context, previous, judge_issues, model)

        return self._evaluate_loop(
            question, strategy, context, draft, model, policy.evaluator, improve=improve,
        )
```

(`_evaluate_loop` already treats `scorecard is None` as pass and stops after `max_rounds` — the gate semantics fall out of the existing loop.)

- [ ] **Step 4: Verify orchestrator suite**

Run: `uv run pytest tests/unit/test_orchestrator.py -q`
Expected: ALL PASS — the 6 rewritten/new gate tests plus every pre-existing `enabled=false` critique test and map_reduce test, none of which were modified.

- [ ] **Step 5: Full suite**

Run: `uv run pytest -q`
Expected: all PASS (391 baseline − 2 deleted + 7 added = 396).

- [ ] **Step 6: Commit**

```bash
git add agent/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat: gate-first critique topology (judge the draft before critics)"
```

---

### Task 3: Docs + verification handoff

**Files:**
- Modify: `AGENTS.md` (Architecture section, Flow bullet)

**Interfaces:** none.

- [ ] **Step 1: Update the Flow description in `AGENTS.md`**

Find the bullet starting `- **Flow**:` and append one sentence at its end (inside the same bullet):

```
Critique topology is gate-first when `evaluator.enabled`: the draft is judged
before critics run, so passing drafts return at ~baseline cost.
```

- [ ] **Step 2: Full unit suite**

Run: `uv run pytest -q && uv run pytest tests/unit/test_domain_agnostic.py -q`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: describe gate-first critique flow in AGENTS.md"
```

- [ ] **Step 4: Live verification (manual, needs API keys — report to human, do not block)**

Per spec §8: `uv run python -m agent.evaluation compare --ids se-129 se-1 se-2 se-3 se-4`.
Gates: orch Q ≥ base Q everywhere; passing-path token increase ≲50% (vs current +97~389%). High gate-failure rate on some case = input for the next knife (spec §3 方案 C / threshold tuning), not a defect of this change.

---

## Self-Review Notes

- Spec §4.2 flow ↔ Task 2 Step 3 (round 0 = critique pass, rounds ≥1 = judge feedback); §4.4 edges ↔ tests (empty-issues exhaustion, parse-failure pass, zero rounds); §5 compatibility ↔ Task 1's unmodified-suite gate + Task 2 Step 4 assertion; §7 items 1-7 ↔ test names: gate_pass(1), gate_fail(2)+round1(7→rewritten as round1_low), empty exhaust(3), parse failure(4), zero rounds(5), budget(7); §7 item 6 (regression) ↔ Task 1 Step 3 + Task 2 Step 4. §8 ↔ Task 3 Step 4.
- Placeholder scan: none — all code verbatim.
- Type consistency: `_critique_pass` signature identical between Task 1 definition and Task 2 call sites (`question, strategy, context, draft, model, policy, intent, draft_completion_tokens=`); `_JUDGE_MARKER`/`_gate_call_indices` defined once and used only in Task 2 tests appended together.
