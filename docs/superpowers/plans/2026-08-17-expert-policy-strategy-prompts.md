# Software Engineering Expert Policy + Strategy Prompt 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a single `domain/software_engineering/expert_policy.md` defining the Software Engineering Expert, prepend it to every strategy system prompt at runtime, and refactor the 5 strategy prompts to carry only their strategy-specific behavior.

**Architecture:** `expert_policy.md` is loaded once at domain config load into `DomainConfig.expert_policy` (missing file → `""`). `Strategy.build_system_prompt()` returns `expert_policy + "\n\n" + prompt_template` when the policy is non-empty, else the template verbatim. Orchestrator workers/aggregator inherit the policy automatically via `build_system_prompt()`; the Planner template is untouched.

**Tech Stack:** Python 3.13, pytest. Config loading follows the existing `intents.yaml`/`complexity.yaml` pattern in `agent/config.py`.

## Global Constraints

- System prompt at runtime = `expert_policy + "\n\n" + strategy_prompt`; no placeholders in prompt files.
- Domains without `expert_policy.md` must load unchanged and behave exactly as today (`expert_policy = ""`).
- `unsupported_complex.md` is NOT a strategy — it must NOT get the policy prepended; leave the file unchanged.
- Orchestrator Planner template (`_PLANNER_PROMPT` in `agent/orchestrator.py`) is unchanged.
- The 5 strategy prompt files drop the shared "You are an expert Agent in the Software Engineering domain. Covers software design..." identity block and the generic "Answer authoritatively and professionally" boilerplate.
- All prompt content (policy + strategies) is written in English.
- No changes to classification, routing, complexity policy, or `domain.json`.

---

## File Structure

- `domain/software_engineering/expert_policy.md` — new: Expert Identity, Engineering Principles, Context Awareness, Uncertainty Policy.
- `domain/software_engineering/prompts/{direct,teaching,debugging,analysis,code_snippet}.md` — refactored: strategy-specific behavior only.
- `agent/config.py` — `DomainConfig.expert_policy: str = ""`; loader reads `base/expert_policy.md`.
- `agent/strategy.py` — `Strategy.expert_policy`; prepend in `build_system_prompt()`; `build_registry` passes `domain.expert_policy`.
- `evaluation/datasets/software_engineering/{debugging,analysis,code_snippet}.yaml` — new answer-quality cases.
- Tests: `tests/test_config.py`, `tests/test_strategy.py`, `tests/test_evaluation_dataset.py`.

---

### Task 1: `expert_policy` field + loader in `agent/config.py`

**Files:**
- Modify: `agent/config.py` (`DomainConfig` ~line 172; `load_domain_config` return ~line 342)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: existing `_read_prompt` (or plain `Path.read_text`), `ConfigError` in `agent/config.py`.
- Produces:
  - `DomainConfig.expert_policy: str = ""`
  - `load_domain_config(domain_dir)` reads `domain_dir/expert_policy.md`; missing file → `expert_policy = ""` (no error).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py` (after the complexity loader tests):

```python
def test_load_domain_config_expert_policy(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    (base / "expert_policy.md").write_text(
        "You are a Senior Software Engineering Expert.", encoding="utf-8"
    )
    domain = load_domain_config(str(base))
    assert domain.expert_policy == "You are a Senior Software Engineering Expert."


def test_load_domain_config_expert_policy_missing_is_empty(tmp_path):
    base = tmp_path / "domain"
    (base / "prompts").mkdir(parents=True)
    (base / "domain.json").write_text(
        json.dumps({"name": "x", "description": "d"}), encoding="utf-8"
    )
    (base / "intents.yaml").write_text("", encoding="utf-8")
    (base / "intent_mapping.yaml").write_text("", encoding="utf-8")
    (base / "strategies.yaml").write_text("direct:\n  default: true\n", encoding="utf-8")
    (base / "prompts" / "direct.md").write_text("d", encoding="utf-8")
    (base / "prompts" / "unsupported_complex.md").write_text("u", encoding="utf-8")
    domain = load_domain_config(str(base))
    assert domain.expert_policy == ""
```

Note: `json` and `load_domain_config` are already imported in the test file's header.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -k expert_policy -v`
Expected: FAIL — `DomainConfig.expert_policy` attribute missing.

- [ ] **Step 3: Add the field**

In `agent/config.py`, add to `DomainConfig` (after `complexity`):

```python
    complexity: ComplexityPolicy | None = None
    expert_policy: str = ""
```

- [ ] **Step 4: Add the loader logic**

In `load_domain_config`, before the `return DomainConfig(...)` statement, add:

```python
    expert_policy = ""
    expert_policy_path = base / "expert_policy.md"
    if expert_policy_path.is_file():
        expert_policy = expert_policy_path.read_text(encoding="utf-8")
```

Add `expert_policy=expert_policy,` to the returned `DomainConfig(...)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -k expert_policy -v`
Expected: PASS (2 tests). Then run `pytest tests/test_config.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 6: Commit**

```bash
git add agent/config.py tests/test_config.py
git commit -m "feat: load expert_policy into domain config"
```

---

### Task 2: Runtime prepend in `agent/strategy.py`

**Files:**
- Modify: `agent/strategy.py` (`Strategy.__init__`, `build_system_prompt`, `build_registry`)
- Test: `tests/test_strategy.py`

**Interfaces:**
- Consumes: `DomainConfig.expert_policy` (Task 1).
- Produces:
  - `Strategy(strategy_id: str, prompt_template: str, expert_policy: str = "")`
  - `Strategy.build_system_prompt() -> str` returns `expert_policy + "\n\n" + prompt_template` when policy non-empty, else `prompt_template`.
  - `build_registry(domain)` passes `domain.expert_policy` to each `Strategy`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_strategy.py`:

```python
def test_build_system_prompt_prepends_expert_policy():
    p = Strategy("direct", "Strategy text.", expert_policy="POLICY")
    assert p.build_system_prompt() == "POLICY\n\nStrategy text."


def test_build_registry_passes_expert_policy():
    domain = _domain()
    domain.expert_policy = "POLICY"
    registry = build_registry(domain)
    assert registry["direct"].build_system_prompt() == "POLICY\n\nDirect answer prompt."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_strategy.py -k expert_policy -v`
Expected: FAIL — `TypeError: Strategy.__init__() got an unexpected keyword argument 'expert_policy'`.

- [ ] **Step 3: Implement**

In `agent/strategy.py`:

```python
class Strategy:
    def __init__(self, strategy_id: str, prompt_template: str, expert_policy: str = ""):
        self.strategy_id = strategy_id
        self.prompt_template = prompt_template
        self.expert_policy = expert_policy

    def build_system_prompt(self) -> str:
        if self.expert_policy:
            return self.expert_policy + "\n\n" + self.prompt_template
        return self.prompt_template
```

And in `build_registry`:

```python
def build_registry(domain: DomainConfig) -> dict[str, Strategy]:
    return {
        sid: Strategy(sid, domain.prompts[sid], expert_policy=domain.expert_policy)
        for sid in domain.strategies
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategy.py -v`
Expected: PASS (all new + existing, including `test_build_system_prompt_returns_template_verbatim` — empty policy keeps the verbatim behavior).

- [ ] **Step 5: Commit**

```bash
git add agent/strategy.py tests/test_strategy.py
git commit -m "feat: prepend expert policy to strategy system prompts"
```

---

### Task 3: Create `domain/software_engineering/expert_policy.md`

**Files:**
- Create: `domain/software_engineering/expert_policy.md`

**Interfaces:**
- Consumes: schema from Task 1 (plain text file at `base/expert_policy.md`).
- Produces: the policy text prepended by `Strategy.build_system_prompt()` (Task 2).

- [ ] **Step 1: Write the file**

Create `domain/software_engineering/expert_policy.md`:

```markdown
# Software Engineering Expert Policy

## Expert Identity

You are a Senior Software Engineering Expert. Your answers focus on:
- Technical correctness
- Practical feasibility
- Context
- Trade-offs
- Long-term maintenance cost

## Engineering Principles

1. Correctness matters more than verbosity.
2. Prefer the simplest solution that meets the requirements.
3. State important assumptions explicitly.
4. Distinguish technical facts from personal recommendations.
5. Never describe an option as unconditionally best.
6. When multiple options exist, explain the trade-offs.
7. Adapt recommendations to actual constraints.

## Context Awareness

When relevant to the question, consider the relevant dimensions among:
language, framework, runtime, deployment, scale, latency, consistency,
reliability, security, operability, maintenance.

Discuss only the dimensions that matter for the question — not all of them on
every answer.

## Uncertainty Policy

Never fabricate:
- API behavior
- Version-specific features
- Benchmarks
- Implementation details
- A definitive root cause without sufficient evidence

When evidence is insufficient, follow:
Missing Evidence → Hypotheses → Verification Steps
```

- [ ] **Step 2: Verify the loader picks it up**

Run: `python -c "from agent.config import load_domain_config; d=load_domain_config('domain/software_engineering'); print('Senior Software Engineering Expert' in d.expert_policy); print(len(d.expert_policy))"`
Expected: prints `True` and a positive length.

- [ ] **Step 3: Commit**

```bash
git add domain/software_engineering/expert_policy.md
git commit -m "feat: software engineering expert policy"
```

---

### Task 4: Refactor the 5 strategy prompts

**Files:**
- Modify: `domain/software_engineering/prompts/direct.md`
- Modify: `domain/software_engineering/prompts/teaching.md`
- Modify: `domain/software_engineering/prompts/debugging.md`
- Modify: `domain/software_engineering/prompts/analysis.md`
- Modify: `domain/software_engineering/prompts/code_snippet.md`

**Interfaces:**
- Consumes: `expert_policy` (Task 3) now carries the domain identity; these files must NOT repeat it.
- Produces: the "Strategy Policy" half of the runtime system prompt.

- [ ] **Step 1: Rewrite `direct.md`**

Overwrite `domain/software_engineering/prompts/direct.md`:

```markdown
# Direct Strategy

Answer the user's question directly.

Requirements:
- Understand the user's real goal.
- Choose the answer depth based on the question.
- Do not force a fixed template.
- State assumptions when necessary.
- Provide code and examples when necessary.
- When multiple options exist, explain the trade-offs.
- When uncertain, state the uncertainty explicitly.
```

- [ ] **Step 2: Rewrite `teaching.md`**

Overwrite `domain/software_engineering/prompts/teaching.md`:

```markdown
# Teaching Strategy

Explain the topic with a structure chosen for the learning goal — not a fixed
template.

Principles:
- Simple questions: answer concisely.
- Complex concepts: explain layer by layer.
- Use analogies when they help.
- Use code when it helps.
- Point out common misconceptions when relevant.
- Never add content just to satisfy a template.
```

- [ ] **Step 3: Rewrite `debugging.md`**

Overwrite `domain/software_engineering/prompts/debugging.md`:

```markdown
# Debugging Strategy

Debug systematically. A list of possible causes is NOT a completed debugging
analysis.

Work from observed symptoms through to the root cause:

Observed Symptoms → Facts / Evidence → Hypotheses → Discriminating Tests →
Root Cause → Fix → Prevention

When the root cause cannot be determined, state clearly:
- Most likely hypothesis
- Evidence for it
- How to verify it
- Alternative hypothesis
```

- [ ] **Step 4: Rewrite `analysis.md`**

Overwrite `domain/software_engineering/prompts/analysis.md`:

```markdown
# Analysis Strategy

Structure your analysis as:

Decision → Evaluation Criteria → Alternatives → Trade-offs → Risks →
Recommendation → When the recommendation changes

Explicitly state the conditions under which the recommendation would change.
```

- [ ] **Step 5: Rewrite `code_snippet.md`**

Overwrite `domain/software_engineering/prompts/code_snippet.md`:

```markdown
# Code Snippet Strategy

Requirements:
- Prefer the minimal complete solution.
- Never drop error handling to make the code shorter.
- Never ignore resource release.
- Never ignore necessary concurrency safety.
- Never ignore necessary input validation.
- If the code is a teaching example, state which production concerns were
  omitted.
```

- [ ] **Step 6: Verify the domain loads and prompts compose**

Run: `python -c "from agent.config import load_domain_config; from agent.strategy import build_registry; d=load_domain_config('domain/software_engineering'); r=build_registry(d); p=r['debugging'].build_system_prompt(); assert p.startswith('# Software Engineering Expert Policy'); assert 'Observed Symptoms' in p; assert 'You are an expert Agent' not in p; print('ok')"`
Expected: prints `ok`.

Also verify `unsupported_complex` is untouched:
Run: `python -c "from agent.config import load_domain_config; d=load_domain_config('domain/software_engineering'); print('orchestrator pipeline' in d.prompts['unsupported_complex'])"`
Expected: prints `True` (original text still present).

- [ ] **Step 7: Commit**

```bash
git add domain/software_engineering/prompts/
git commit -m "feat: refactor software engineering strategy prompts"
```

---

### Task 5: Evaluation dataset cases for the new policy behaviors

**Files:**
- Modify: `evaluation/datasets/software_engineering/debugging.yaml`
- Modify: `evaluation/datasets/software_engineering/analysis.yaml`
- Modify: `evaluation/datasets/software_engineering/code_snippet.yaml`
- Test: `tests/test_evaluation_dataset.py`

**Interfaces:**
- Consumes: dataset schema in `agent/evaluation/dataset.py` (`cases`, `id`, `question`, `expected`, `answer_quality`).
- Produces: three answer-quality cases exercising §11/§12/§13 policy behaviors.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluation_dataset.py`:

```python
def test_policy_behavior_cases_present():
    suites = load_suites("evaluation/datasets/software_engineering")
    cases = {c.id: c for s in suites for c in s.cases}
    assert cases["se-054"].expected_strategy == "debugging"
    assert "no logs" in cases["se-054"].question
    assert cases["se-082"].expected_strategy == "analysis"
    assert "monolith" in cases["se-082"].question
    assert cases["se-102"].expected_strategy == "code_snippet"
    assert "close the file" in cases["se-102"].question
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation_dataset.py::test_policy_behavior_cases_present -v`
Expected: FAIL — KeyError `se-054`.

- [ ] **Step 3: Add the cases**

Append to `evaluation/datasets/software_engineering/debugging.yaml`:

```yaml
  - id: se-054
    question: "My service returns 500 errors sporadically in production, but I have no logs or stack traces yet. What is the root cause?"
    expected: {domain: software_engineering, intent: troubleshooting, complexity: medium, strategy: debugging, orchestrate: false}
    answer_quality: true
```

Append to `evaluation/datasets/software_engineering/analysis.yaml`:

```yaml
  - id: se-082
    question: "Should a startup migrate from a monolith to microservices now, or stay with the monolith?"
    expected: {domain: software_engineering, intent: architecture_design, complexity: medium, strategy: analysis, orchestrate: false}
    answer_quality: true
```

Append to `evaluation/datasets/software_engineering/code_snippet.yaml`:

```yaml
  - id: se-102
    question: "Write a Python function that reads a file, processes each line, writes results to an output file, and closes the file properly even when an error occurs."
    expected: {domain: software_engineering, intent: generate_code, complexity: medium, strategy: code_snippet, orchestrate: false}
    answer_quality: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation_dataset.py::test_policy_behavior_cases_present -v`
Expected: PASS. Then run `pytest tests/test_evaluation_dataset.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
git add evaluation/datasets/software_engineering/debugging.yaml evaluation/datasets/software_engineering/analysis.yaml evaluation/datasets/software_engineering/code_snippet.yaml tests/test_evaluation_dataset.py
git commit -m "feat: evaluation cases for expert policy behaviors"
```

---

### Task 6: Full regression

**Files:** none new.

**Interfaces:** consumes all prior tasks.

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all tests, including domain-agnostic, evaluation CLI, and smoke tests).

- [ ] **Step 2: Verify the software-engineering domain end-to-end**

Run: `python -c "from agent.config import load_domain_config; from agent.strategy import build_registry; d=load_domain_config('domain/software_engineering'); [print(sid, 'len=', len(r.build_system_prompt())) for sid, r in build_registry(d).items()]"`
Expected: prints each strategy id with a prompt length well above the bare strategy text (policy prepended).

- [ ] **Step 3: Record evaluation baseline (manual, needs API key)**

Follow README §Baseline tracking:

```bash
uv run python -m agent.evaluation run --label expert-policy-strategy-prompts
uv run python -m agent.evaluation baseline evaluation/results/<printed-path>
```

Commit the updated `evaluation/results/baseline.json` together with the change that produced it.

- [ ] **Step 4: Commit any stragglers**

```bash
git status --short
git add -A
git commit -m "chore: expert policy + strategy prompt regression"
```
(If nothing changed, skip the commit.)