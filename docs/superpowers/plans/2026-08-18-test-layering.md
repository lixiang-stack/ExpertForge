# Test Layering (Unit vs Live API Tests) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the test suite into two physical tiers — `tests/unit/` (hermetic, run by default) and `live/` (real LLM API, explicit opt-in) — so `uv run pytest -q` never makes a live API call regardless of `AGENT_API_KEY`.

**Architecture:** Rehome existing tests by tier via `git mv`; extract the two pure live-config helper functions out of the live smoke test into a neutral `tests/helpers.py` module; constrain default collection with pytest's `testpaths`; keep `AGENT_API_KEY`-based `skipif` on the live tests as a double-safety.

**Tech Stack:** Python, pytest (>=8.0).

## Global Constraints

- Execute on branch `fix/gemini-smoke-compat` in worktree `.worktrees/gemini-smoke-compat` (HEAD `5ce36a6`). All file edits and git/uv commands run inside that directory.
- Run tests with `uv run pytest` from the worktree root.
- Spec: `docs/superpowers/specs/2026-08-18-test-layering-design.md` (in the main repo, committed `ac3c8f5`).
- The default suite (`uv run pytest -q`) must NEVER call a live API, even when `AGENT_API_KEY` is set.
- Live tests keep their existing `skipif(os.environ.get("AGENT_API_KEY") is None)` gate.
- `tests/helpers.py` contains ONLY `resolve_live_config_src(root: Path) -> dict` and `absolutize_domain_dir(config: dict, root: Path) -> dict`, extracted verbatim from the current `tests/test_smoke.py` (lines 27-43).
- Do not edit archived plan/spec documents or the main repo checkout.

---

### Task 1: Rehome live tests to `live/` and extract shared helpers

**Files:**
- Move: `tests/test_smoke.py` → `live/test_smoke.py`
- Move: `tests/test_integration.py` → `live/test_integration.py`
- Create: `live/__init__.py`
- Create: `tests/helpers.py`
- Modify: `live/test_smoke.py` (remove local helper defs; import from `tests.helpers`; update docstring)
- Modify: `live/test_integration.py` (update docstring)
- Modify: `tests/test_smoke_config.py:6` (import from `tests.helpers` instead of `tests.test_smoke`)

**Interfaces:**
- Consumes: current `tests/test_smoke.py` lines 27-43 (the two helper functions).
- Produces: `tests.helpers.resolve_live_config_src`, `tests.helpers.absolutize_domain_dir` — imported by `live/test_smoke.py` and `tests/test_smoke_config.py`.

- [ ] **Step 1: Move the two live test files and add package markers**

```bash
mkdir -p live
git mv tests/test_smoke.py live/test_smoke.py
git mv tests/test_integration.py live/test_integration.py
touch live/__init__.py
```

- [ ] **Step 2: Write the failing import (RED)**

Edit `tests/test_smoke_config.py:6` so the import becomes:

```python
from tests.helpers import absolutize_domain_dir, resolve_live_config_src
```

- [ ] **Step 3: Verify the test fails**

Run: `uv run pytest tests/test_smoke_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.helpers'`

- [ ] **Step 4: Create `tests/helpers.py`**

```python
import json
from pathlib import Path


def resolve_live_config_src(root: Path) -> dict:
    """Return the preferred live-test config: the user's config.json when it
    exists, otherwise config.example.json, both resolved under ``root``."""
    for name in ("config.json", "config.example.json"):
        path = root / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No config.json or config.example.json under {root}")


def absolutize_domain_dir(config: dict, root: Path) -> dict:
    """Return a copy of ``config`` with ``domain_dir`` resolved against ``root``."""
    merged = dict(config)
    domain_dir = merged.get("domain_dir")
    path = Path(domain_dir) if domain_dir else root
    merged["domain_dir"] = str(path if path.is_absolute() else root / path)
    return merged
```

- [ ] **Step 5: Point `live/test_smoke.py` at the shared helpers**

In `live/test_smoke.py`:
1. Delete the local definitions of `resolve_live_config_src` and `absolutize_domain_dir`.
2. Add at the top, after the existing imports:

```python
from tests.helpers import absolutize_domain_dir, resolve_live_config_src
```

3. Update the module docstring: replace the run command line

```python
    uv run pytest tests/test_smoke.py -v
```

with

```python
    uv run pytest live -v
```

- [ ] **Step 6: Update `live/test_integration.py` docstring**

Replace the run command line

```python
    uv run pytest tests/test_integration.py -v
```

with

```python
    uv run pytest live -v
```

- [ ] **Step 7: Verify green (GREEN)**

Run: `uv run pytest tests/test_smoke_config.py -q`
Expected: PASS (6 passed)

Run: `uv run pytest -q`
Expected: PASS (302 passed — the existing `pytest.ini` `testpaths = tests` already excludes `live/`, so the moved live tests are not collected)

Run: `uv run pytest live -q`
Expected: 5 skipped (no `AGENT_API_KEY`)

- [ ] **Step 8: Commit**

```bash
git add -A live tests/helpers.py tests/test_smoke_config.py
git commit -m "refactor: rehome live API tests to live/ and extract shared config helpers"
```

---

### Task 2: Move hermetic unit tests to `tests/unit/`

**Files:**
- Move: all 29 hermetic `tests/test_*.py` → `tests/unit/` (every file EXCEPT `test_smoke.py` and `test_integration.py`, which moved in Task 1)
- Create: `tests/unit/__init__.py`
- Modify: `tests/unit/test_evaluation_dataset.py:124` (`parents[1]` → `parents[2]`)

**Interfaces:**
- Consumes: nothing from other tasks; pure file rehome.
- Produces: the `tests/unit/` package that Task 3's `testpaths` will point at.

- [ ] **Step 1: Move the unit test files**

```bash
mkdir -p tests/unit
touch tests/unit/__init__.py
git mv tests/test_agent_cli.py tests/unit/
git mv tests/test_capabilities.py tests/unit/
git mv tests/test_chat.py tests/unit/
git mv tests/test_classification.py tests/unit/
git mv tests/test_config.py tests/unit/
git mv tests/test_domain_agnostic.py tests/unit/
git mv tests/test_evaluation_cli.py tests/unit/
git mv tests/test_evaluation_dataset.py tests/unit/
git mv tests/test_evaluation_diff.py tests/unit/
git mv tests/test_evaluation_judge.py tests/unit/
git mv tests/test_evaluation_metrics.py tests/unit/
git mv tests/test_evaluation_report.py tests/unit/
git mv tests/test_evaluation_runner.py tests/unit/
git mv tests/test_llm.py tests/unit/
git mv tests/test_model_router.py tests/unit/
git mv tests/test_negotiate.py tests/unit/
git mv tests/test_observability_client.py tests/unit/
git mv tests/test_observability_install.py tests/unit/
git mv tests/test_observability_patch.py tests/unit/
git mv tests/test_orchestrator.py tests/unit/
git mv tests/test_parsing.py tests/unit/
git mv tests/test_repl.py tests/unit/
git mv tests/test_report.py tests/unit/
git mv tests/test_report_data.py tests/unit/
git mv tests/test_router.py tests/unit/
git mv tests/test_smoke_config.py tests/unit/
git mv tests/test_strategy.py tests/unit/
git mv tests/test_tracing.py tests/unit/
git mv tests/test_worker_pool.py tests/unit/
```

- [ ] **Step 2: Fix the repo-root reference that shifts one level**

In `tests/unit/test_evaluation_dataset.py:124`, change:

```python
    repo = Path(__file__).resolve().parents[1]
```

to:

```python
    repo = Path(__file__).resolve().parents[2]
```

- [ ] **Step 3: Verify the suite still passes**

Run: `uv run pytest -q`
Expected: PASS (302 passed — the existing `pytest.ini` `testpaths = tests` collects `tests/` recursively, including the new `tests/unit/`)

Run: `uv run pytest tests/unit/test_evaluation_dataset.py::test_load_committed_software_engineering_suites -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add -A tests/unit
git commit -m "refactor: move hermetic unit tests into tests/unit/"
```

---

### Task 3: Configure pytest — `testpaths`, `pythonpath`, markers

**Files:**
- Modify: `pytest.ini` (replace the existing `[pytest]` block — `testpaths` currently `tests`, no markers)

**Why `pytest.ini`, not `pyproject.toml`:** the repo's root `pytest.ini` (from the scaffold commit) already declares `testpaths = tests`. pytest reads only the FIRST config file it finds (precedence: `pytest.ini` > `pyproject.toml` > `tox.ini` > `setup.cfg`), so adding `[tool.pytest.ini_options]` to `pyproject.toml` would be silently ignored. The live tests are already excluded from the default run by the existing `testpaths = tests`; this task narrows it to `tests/unit` and adds the `live` marker.

**Interfaces:**
- Consumes: `tests/unit/` (Task 2) and `live/` (Task 1).
- Produces: the default-run restriction that makes `uv run pytest -q` hermetic-only and registers the `live` marker.

- [ ] **Step 1: Record the pre-change default-run behavior**

Run: `uv run pytest -q`
Expected: PASS (302 passed, 0 skipped) — `testpaths = tests` already excludes `live/`. The remaining gap: `testpaths` is not scoped to `tests/unit`, and the `live` marker is not registered.

- [ ] **Step 2: Update `pytest.ini`**

Replace the contents of `pytest.ini`:

```ini
[pytest]
testpaths = tests/unit
pythonpath = .
markers =
    live: exercises the real LLM API (requires AGENT_API_KEY)
```

- [ ] **Step 3: Verify the default run is hermetic-only**

Run: `uv run pytest -q`
Expected: PASS — **302 passed, 0 skipped** (`testpaths = tests/unit` limits collection to the unit tier; `live/` is not collected)

- [ ] **Step 4: Verify the live tier is importable and skippable**

Run: `uv run pytest live -q`
Expected: 5 skipped (live tests collected, `AGENT_API_KEY` unset, imports resolve via `pythonpath = .`)

Run: `uv run pytest live/test_smoke.py -q`
Expected: 3 skipped — this proves `from tests.helpers import ...` and `from agent import ...` resolve from the `live/` package

- [ ] **Step 5: Commit**

```bash
git add pytest.ini
git commit -m "chore: scope default pytest run to tests/unit and register live marker"
```

---

### Task 4: Update README test documentation

**Files:**
- Modify: `README.md:185-214` (the "Smoke tests" and "Integration tests" sections)

**Interfaces:**
- Consumes: the final layout from Tasks 1-3.
- Produces: accurate user-facing run instructions.

- [ ] **Step 1: Replace the test documentation**

Replace the two sections at `README.md:185-203`:

```markdown
### Smoke tests

`tests/test_smoke.py` exercises the real agent end-to-end against the API:
a single-shot `--ask` question returns a non-empty answer, and an
out-of-domain question returns the rejection reply.

```bash
uv run pytest tests/test_smoke.py -v
```

### Integration tests

`tests/test_integration.py` exercises deeper pipeline paths against the API:
a complex gated question runs through the Orchestrator (Planner → Workers →
Aggregator), and a medium question flows through a strategy.

```bash
uv run pytest tests/test_integration.py -v
```
```

with:

```markdown
### Unit tests

`tests/unit/` is the hermetic unit-test tier. It never calls a live API and
is the default run:

```bash
uv run pytest -q
```

### Live tests

`live/` exercises the real agent against the LLM API: `test_smoke.py` runs a
single-shot `--ask` question and an out-of-domain question end-to-end;
`test_integration.py` runs a complex gated question through the Orchestrator
(Planner → Workers → Aggregator) and a medium question through a strategy.

Live tests run only when you opt in explicitly and `AGENT_API_KEY` is set:

```bash
uv run pytest live -v
```
```

- [ ] **Step 2: Update the API-key paragraph**

In the "API key security" section at `README.md:207-208`, replace:

```markdown
The smoke and integration tests need a real `AGENT_API_KEY` and skip
automatically when it is not set.
```

with:

```markdown
The live tests need a real `AGENT_API_KEY` and skip automatically when it is
not set.
```

- [ ] **Step 3: Verify no stale test paths remain in docs/code**

Run: `rg -n "tests/test_smoke|tests/test_integration" README.md tests/ live/`
Expected: no matches

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document two-tier test layout (unit vs live)"
```

---

### Task 5: Regression verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Verify default suite is hermetic and green**

Run: `uv run pytest -q`
Expected: 302 passed, 0 skipped

- [ ] **Step 2: Verify live tier gating**

Run: `uv run pytest live -q`
Expected: 5 skipped (no `AGENT_API_KEY`)

- [ ] **Step 3: Verify no stale references anywhere**

Run: `rg -n "tests/test_smoke|tests/test_integration" tests/ live/ README.md pytest.ini`
Expected: no matches

- [ ] **Step 4: Verify the tree is clean**

Run: `git status --short`
Expected: clean (no leftover files in `tests/` other than `__init__.py` and `helpers.py`; `tests/unit/` and `live/` populated; `pytest.ini` updated)

- [ ] **Step 5: Commit (only if cleanup surfaced)**

If Step 3 or Step 4 found residual references or stray files, fix them, then:

```bash
git add -A
git commit -m "fix: remove residual test-path references"
```

Otherwise: no commit.