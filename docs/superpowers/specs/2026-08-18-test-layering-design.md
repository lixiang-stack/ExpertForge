# Test Layering: Unit vs Live API Tests

Date: 2026-08-18

## Problem

Running `uv run pytest -q` from the repo root collects `tests/test_smoke.py` and
`tests/test_integration.py`, which gate on `AGENT_API_KEY` (skip only when unset).
When the key IS present in the environment (e.g. sourced from `/tmp/ef_api_key.sh`),
the default test run makes real Gemini API calls — exhausting the free-tier quota
and failing with `429 RESOURCE_EXHAUSTED`. The default test suite must NEVER hit a
live API, regardless of environment.

## Decision

Split the test suite into two physical tiers with pytest `testpaths`:

- **Unit tier** — hermetic, deterministic, no external services. Default run.
- **Live tier** — exercises the real LLM API; requires `AGENT_API_KEY`; explicit run only.

Layout:

```
tests/
  __init__.py              # existing
  helpers.py               # NEW: live-config helpers (pure functions, shared)
  unit/                    # unit tier — default `uv run pytest -q` runs ONLY this
    __init__.py            # NEW
    test_*.py              # all existing hermetic test files move here
live/                      # live tier — explicit `uv run pytest live -v`
  __init__.py              # NEW
  test_smoke.py            # moved from tests/
  test_integration.py      # moved from tests/
```

`pytest.ini` (existing root config from the scaffold; `testpaths` is already `tests`):

```ini
[pytest]
testpaths = tests/unit
pythonpath = .
markers =
    live: exercises the real LLM API (requires AGENT_API_KEY)
```

> Note: config lives in `pytest.ini`, not `pyproject.toml` — pytest reads only the
> first config file it finds (`pytest.ini` takes precedence over `pyproject.toml`),
> so a `[tool.pytest.ini_options]` block would be silently ignored.

## Rationale for this layout

- `testpaths = ["tests/unit"]` restricts the DEFAULT collection to the unit tier.
  The live tier is a sibling directory, so it is excluded without `--ignore`
  (an `--ignore`-based approach would also apply when running the live dir
  explicitly — a footgun).
- `tests/test_integration.py` is a misnomer: it calls the real API. It is a live
  test and is re-homed accordingly. The repo has no hermetic "integration" tier
  today; a formal `tests/integration/` layer is deliberately not created (YAGNI).
- `tests/helpers.py` holds the two pure config helpers previously defined inside
  `tests/test_smoke.py` (`resolve_live_config_src`, `absolutize_domain_dir`).
  This removes the "unit test imports a live test module" coupling
  (`tests/test_smoke_config.py` did `from tests.test_smoke import ...`).
  The unit test now imports them from `tests.helpers`; the live tests import the
  same helpers.
- `pythonpath = ["."]` puts the repo root on `sys.path` so `from agent import ...`
  and `from tests.helpers import ...` work regardless of collection directory.
- `tests/unit/__init__.py` and `live/__init__.py` keep the package-consistent
  import style already used by `tests/__init__.py`.

## Required changes

1. `git mv` all hermetic `tests/test_*.py` (29 files, excluding smoke and
   integration) into `tests/unit/`.
2. `git mv tests/test_smoke.py` → `live/test_smoke.py`;
   `git mv tests/test_integration.py` → `live/test_integration.py`.
3. Create `tests/helpers.py` with `resolve_live_config_src` and
   `absolutize_domain_dir` (extracted verbatim from `tests/test_smoke.py`).
4. `live/test_smoke.py`: drop its local helper definitions, import them from
   `tests.helpers`; update the module docstring run command to
   `uv run pytest live -v`; keep the `AGENT_API_KEY` skipif.
5. `live/test_integration.py`: update the module docstring run command to
   `uv run pytest live -v`; keep the `AGENT_API_KEY` skipif.
6. `tests/unit/test_smoke_config.py`: import the helpers from `tests.helpers`
   instead of `tests.test_smoke`.
7. `tests/unit/test_evaluation_dataset.py:124`: `Path(__file__).resolve().parents[1]`
   → `parents[2]` (repo root is now two levels up from a `tests/unit/` file).
8. Add `tests/unit/__init__.py` and `live/__init__.py`.
9. `pyproject.toml`: add `[tool.pytest.ini_options]` with `testpaths`,
   `pythonpath`, `markers` as above.
10. README.md: replace the current smoke/integration test paragraphs with a
    two-tier description and the new run commands.

## Verification

- `uv run pytest -q` → 302 passed, 0 skipped, and NO API calls even with
  `AGENT_API_KEY` set.
- `uv run pytest live -q` → 5 tests (3 smoke + 2 integration), all skipped when
  `AGENT_API_KEY` is unset.
- `uv run pytest tests/unit/test_smoke_config.py -q` → passes (helpers import
  resolved).
- `rg -n "tests/test_smoke|tests/test_integration" tests/ live/` → no stale path
  references in code; README updated.

## Out of scope

- The main repo checkout (pre-refactor snapshot) is unchanged; this lands on the
  `fix/gemini-smoke-compat` worktree branch as part of the combined PR #15.
- No `tests/integration/` tier is created.
- Archived plan/spec documents are not edited.