# Software Engineering Expert Policy + Strategy Prompt 重构 — Design

Date: 2026-08-17
Status: Draft

## 1. Goal

Per `draft_v2.md` §7–13: introduce a single `expert_policy.md` that defines
what makes a good Software Engineering Expert, remove the repeated domain
identity from the strategy prompts, and refactor each strategy prompt so it
carries only its strategy-specific behavior. The final system prompt becomes:

```text
Expert Policy
      +
Strategy Policy
      +
User Task
```

## 2. Approach

**Approach A** (chosen): runtime composition. `expert_policy.md` is loaded once
at domain config load into `DomainConfig.expert_policy`. `Strategy.build_system_prompt()`
returns `expert_policy + "\n\n" + strategy_prompt` when the policy is non-empty,
and the prompt verbatim when empty. This keeps a single source of truth for the
expert identity (§8.1), requires no placeholders, and is backward compatible:
domains without `expert_policy.md` behave exactly as today.

Rationale:
- Runtime prepend chosen over static embedding (duplicates the policy into each
  of the 5 files, contradicts the "no repeated Domain Identity" goal) and over a
  `{expert_policy}` placeholder (conflicts with the README's self-contained,
  no-placeholder prompt design).
- The orchestrator workers/aggregator reuse the strategy prompt via
  `build_system_prompt()` (`agent/orchestrator.py:76-78`), so they inherit the
  policy automatically. The Planner's separate template (`_PLANNER_PROMPT`) is
  intentionally left unchanged: it is a mechanical decomposition role, and the
  policy already reaches it indirectly via the strategy `context`.
- Rejected adding a generic "policy bundle" loader across domains: only the
  software_engineering domain needs it now; defer shared extraction until real
  duplication.

## 3. Component Changes

### 3.1 `domain/software_engineering/expert_policy.md` — new file

English, four sections aligned with §7.2:

- **Expert Identity**: act as a Senior Software Engineering Expert; prioritize
  technical correctness, practical feasibility, context, trade-offs, and
  long-term maintenance cost.
- **Engineering Principles**: the 7 rules from §7.2 (correctness over verbosity,
  simplest sufficient solution, state important assumptions, distinguish facts
  from recommendations, never describe an option as unconditionally best, explain
  trade-offs when multiple options exist, adapt advice to actual constraints).
- **Context Awareness**: consider the relevant dimensions from §7.2
  (language/framework/runtime/deployment/scale/latency/consistency/reliability/
  security/operability/maintenance) only when the question calls for them — not
  every dimension on every answer.
- **Uncertainty Policy**: forbidden fabrications (§7.2) and the
  `Missing Evidence → Hypotheses → Verification Steps` flow when evidence is
  insufficient.

### 3.2 `agent/config.py` — data model + loader

- `DomainConfig` gains `expert_policy: str = ""` (backward compatible default).
- `load_domain_config` reads `base / "expert_policy.md"`:
  - file present → `expert_policy = path.read_text(encoding="utf-8")`
  - file missing → `expert_policy = ""` (no error)

### 3.3 `agent/strategy.py` — runtime composition

- `Strategy.__init__` gains `expert_policy: str = ""`.
- `build_system_prompt()`:
  ```python
  def build_system_prompt(self) -> str:
      if self.expert_policy:
          return self.expert_policy + "\n\n" + self.prompt_template
      return self.prompt_template
  ```
- `build_registry` passes `domain.expert_policy` to each `Strategy`.

### 3.4 Strategy prompt refactors (`domain/software_engineering/prompts/*.md`)

Each file drops the shared "You are an expert Agent in the Software Engineering
domain. Covers software design..." identity and keeps only strategy-specific
behavior. The domain scope lives in `domain.json` + `expert_policy.md`.

- **direct.md** (§9): understand the user's real goal; decide answer depth from
  the question; no fixed template; state assumptions when necessary; give code
  and examples when necessary; explain trade-offs when multiple options exist;
  state uncertainty explicitly when uncertain.
- **teaching.md** (§10): choose the explanation structure dynamically per the
  learning goal; simple questions answered concisely; complex concepts explained
  layer by layer; analogies/code/misconceptions only when they help; never add
  irrelevant content just to satisfy a template.
- **debugging.md** (§11): progress from symptoms to root cause through
  `Observed Symptoms → Facts/Evidence → Hypotheses → Discriminating Tests →
  Root Cause → Fix → Prevention`; a "possible causes" list is not a completed
  debugging analysis; when the root cause cannot be determined, state the most
  likely hypothesis, its evidence, how to verify it, and the alternative.
- **analysis.md** (§12): structure as `Decision → Evaluation Criteria →
  Alternatives → Trade-offs → Risks → Recommendation → When the recommendation
  changes`; explicitly state the conditions under which the recommendation
  would change.
- **code_snippet.md** (§13): prefer the minimal complete solution without
  dropping error handling, resource release, necessary concurrency safety, or
  necessary input validation for brevity; if the code is a teaching example,
  state which production handling was omitted.

`unsupported_complex.md` is unchanged (not a strategy, no policy prepend).

### 3.5 Evaluation datasets

Add answer-quality cases exercising the new policy behaviors to
`evaluation/datasets/software_engineering/` (following the existing `se-1xx`
sequence, each with `answer_quality: true`):

| Suite | New case | Validates |
|-------|----------|-----------|
| debugging | A question with only symptoms and no logs/evidence | Root cause not asserted without evidence; hypotheses + verification steps offered |
| analysis | A trade-off decision question whose answer depends on scale/constraints | "When the recommendation changes" section present |
| code_snippet | A code task requiring error handling and resource release (e.g. file/connection handling) | Error handling / resource release retained, not dropped for brevity |

The datasets already cover direct/teaching; existing cases remain valid for
regression. After the prompt refactor, re-run evaluation and record a baseline
(`run` + `baseline`) per README §Baseline tracking.

### 3.6 Tests

- `test_config.py`: `expert_policy.md` present → loaded; missing → `""`.
- `test_strategy.py`: `build_system_prompt()` prepends the policy when set
  (assert `policy in prompt`, `prompt.startswith(policy)`); returns the template
  verbatim when `expert_policy=""` (existing assertion keeps passing).
- `build_registry` passes the policy (assert via a constructed domain).
- Regression: full existing unit suite passes.

## 4. Acceptance Criteria (§7.2 / §23 Phase 1-2)

- `expert_policy.md` exists with all four §7.2 sections. ✓
- Strategy prompts no longer repeat the domain identity. ✓
- System prompt at runtime = policy + strategy prompt (verify via
  `build_system_prompt()`).
- Domains without `expert_policy.md` load and behave unchanged.
- Debugging/analysis answers follow the new reasoning structures (§11/§12).
- Evaluation datasets cover the new policy behaviors; baseline recorded after
  the change.

## 5. Out of Scope

- Orchestrator Planner prompt changes (§14–17 are separate P1 items).
- Worker parallelization, Evaluator/Optimizer, role specialization.
- Other domain directories.
- Multi-turn classification, question rewrite, domain/strategy decoupling
  beyond this refactor (§21 P2).
- Model routing changes.