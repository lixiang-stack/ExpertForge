# Orchestrator Parallelism + Worker Roles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run orchestrator workers concurrently (bounded, timed-out, partial-failure tolerant) and give each worker a distinct `role` from the planner.

**Architecture:** A new sync `agent/worker_pool.py` provides `WorkerTask`/`WorkerResult`/`run_workers()` (ThreadPoolExecutor + per-worker wall-clock timeout + never-raises-on-failure). The orchestrator's planner emits an optional-role task (`role` defaults to `title`), runs workers through the pool, and aggregates partial results with failure markers. Observability gets a thread-local phase stack (parallel workers must not corrupt each other's phase) and records roles + failures.

**Tech Stack:** Python ≥3.10, stdlib `concurrent.futures`/`threading`, pytest (`uv run pytest -q`).

## Global Constraints

- `max_workers` default `4`; `worker_timeout` default `120` (seconds) — verbatim from the design spec.
- `run_workers` never raises on worker failure; results come back in input order.
- A single failed worker must not fail the task; only all-workers-failed degrades to a direct answer.
- Missing/empty `role` in a parsed task defaults to the task's `title`.
- Worker system prompt format: `{context}\n\nRole: {role}\nSub-task: {instruction}`.
- Aggregator failure-marker format: `[worker failed: {error}]`.
- Commit message style: `feat: ...` / `docs: ...` (matches repo history).
- Test command: `uv run pytest -q` (all green at end of every task).
- The project sync stack stays sync — no async refactor.

---
### Task 1: `agent/worker_pool.py` — WorkerTask, WorkerResult, run_workers

**Files:**
- Create: `agent/worker_pool.py`
- Test: `tests/test_worker_pool.py`

**Interfaces:**
- Produces: `WorkerTask(title: str, instruction: str, role: str)`, `WorkerResult(task: WorkerTask, text: str | None = None, error: str | None = None)`, and `run_workers(tasks: list[WorkerTask], run_one, *, max_workers: int = 4, timeout: float = 120.0) -> list[WorkerResult]`.
- Consumes: nothing (standalone; no LLM, no other project modules).

- [ ] **Step 1: Write the failing test**

Create `tests/test_worker_pool.py`:

```python
import threading
import time

from agent.worker_pool import WorkerResult, WorkerTask, run_workers


def _tasks(n=4):
    return [WorkerTask(title=f"t{i}", instruction=f"i{i}", role=f"r{i}") for i in range(n)]


def test_run_workers_returns_results_in_input_order():
    results = run_workers(_tasks(3), lambda t: f"out:{t.title}", max_workers=2, timeout=5.0)
    assert [r.task.title for r in results] == ["t0", "t1", "t2"]
    assert all(r.error is None for r in results)
    assert [r.text for r in results] == ["out:t0", "out:t1", "out:t2"]


def test_run_workers_caps_concurrency():
    lock = threading.Lock()
    active = 0
    max_active = 0

    def run_one(task):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return task.title

    results = run_workers(_tasks(6), run_one, max_workers=2, timeout=5.0)
    assert max_active <= 2
    assert len(results) == 6
    assert all(r.error is None for r in results)


def test_run_workers_worker_timeout_marks_failure():
    results = run_workers(_tasks(1), lambda t: time.sleep(1.0) or "late", timeout=0.05)
    assert results[0].text is None
    assert results[0].error == "timeout"


def test_run_workers_exception_captured_not_raised():
    def run_one(task):
        raise ValueError("boom")

    results = run_workers(_tasks(2), run_one, max_workers=2, timeout=5.0)
    assert [r.error for r in results] == ["boom", "boom"]
    assert all(r.text is None for r in results)


def test_run_workers_all_failed_returns_all_errors():
    def run_one(task):
        raise RuntimeError("x")

    results = run_workers(_tasks(2), run_one, timeout=5.0)
    assert [r.error for r in results] == ["x", "x"]
    assert all(r.text is None for r in results)


def test_run_workers_empty_tasks():
    assert run_workers([], lambda t: "x", timeout=5.0) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_worker_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.worker_pool'`

- [ ] **Step 3: Write minimal implementation**

Create `agent/worker_pool.py`:

```python
"""Parallel worker execution with bounded concurrency and per-worker timeout."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass


@dataclass
class WorkerTask:
    title: str
    instruction: str
    role: str


@dataclass
class WorkerResult:
    task: WorkerTask
    text: str | None = None
    error: str | None = None


def run_workers(
    tasks: list[WorkerTask],
    run_one,
    *,
    max_workers: int = 4,
    timeout: float = 120.0,
) -> list[WorkerResult]:
    """Run run_one(task) per task concurrently, capped at max_workers, with a
    per-worker wall-clock timeout. Never raises on worker failure; results are
    returned in input order. `run_one` is injected so this module needs no LLM."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {i: executor.submit(run_one, task) for i, task in enumerate(tasks)}
        results: list[WorkerResult] = []
        for i, task in enumerate(tasks):
            try:
                text = futures[i].result(timeout=timeout)
            except FutureTimeoutError:
                results.append(WorkerResult(task=task, text=None, error="timeout"))
            except Exception as e:  # noqa: BLE001 - worker failure is captured, not propagated
                results.append(WorkerResult(task=task, text=None, error=str(e)))
            else:
                results.append(WorkerResult(task=task, text=text))
    finally:
        # wait=False so a hung worker beyond the timeout never blocks the pipeline;
        # cancel_futures=True drops queued-but-not-started tasks.
        executor.shutdown(wait=False, cancel_futures=True)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_worker_pool.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add agent/worker_pool.py tests/test_worker_pool.py
git commit -m "feat: parallel worker pool with timeout and partial-failure capture"
```

---
### Task 2: Config — `OrchestratorConfig` + parsing

**Files:**
- Modify: `agent/config.py`
- Modify: `config.example.json`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `OrchestratorConfig(max_workers: int = 4, worker_timeout: float = 120.0)`; `AgentConfig.orchestrator: OrchestratorConfig | None = None`.
- Consumes: nothing new. Task 4 uses `config.orchestrator`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py` (uses the existing `_write_config` helper):

```python
def test_load_config_orchestrator_parsed(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "orchestrator": {"max_workers": 8, "worker_timeout": 90},
    })
    cfg = load_config(path)
    assert cfg.orchestrator is not None
    assert cfg.orchestrator.max_workers == 8
    assert cfg.orchestrator.worker_timeout == 90


def test_load_config_orchestrator_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
    })
    cfg = load_config(path)
    assert cfg.orchestrator is None


def test_load_config_orchestrator_invalid_values_default(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "orchestrator": {"max_workers": 0, "worker_timeout": -5},
    })
    cfg = load_config(path)
    assert cfg.orchestrator.max_workers == 4
    assert cfg.orchestrator.worker_timeout == 120.0


def test_load_config_orchestrator_non_dict_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "k")
    path = _write_config(tmp_path, {
        "base_url": "https://api.example.com/v1",
        "model": "model-a",
        "domain_dir": "domain/software_engineering",
        "orchestrator": "nope",
    })
    cfg = load_config(path)
    assert cfg.orchestrator is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -k orchestrator -v`
Expected: FAIL — `AttributeError: 'AgentConfig' object has no attribute 'orchestrator'`

- [ ] **Step 3: Write minimal implementation**

In `agent/config.py`:

Add the dataclass next to `AgentConfig` (before it, so the field can reference it):

```python
@dataclass
class OrchestratorConfig:
    max_workers: int = 4
    worker_timeout: float = 120.0
```

Add the field to `AgentConfig`:

```python
@dataclass
class AgentConfig:
    base_url: str
    model: str
    classifier_model: str
    domain_dir: str
    model_low: str | None = None
    model_high: str | None = None
    observability: ObservabilityConfig | None = None
    evaluation: EvaluationConfig | None = None
    orchestrator: OrchestratorConfig | None = None
```

In `load_config`, after the `evaluation` parsing block and before the final
`return AgentConfig(...)`:

```python
    raw_orch = raw.get("orchestrator")
    orchestrator = None
    if isinstance(raw_orch, dict):
        max_workers = raw_orch.get("max_workers")
        worker_timeout = raw_orch.get("worker_timeout")
        orchestrator = OrchestratorConfig(
            max_workers=max_workers if isinstance(max_workers, int) and max_workers > 0 else 4,
            worker_timeout=worker_timeout
            if isinstance(worker_timeout, (int, float)) and worker_timeout > 0
            else 120.0,
        )
```

Add `orchestrator=orchestrator` to the `return AgentConfig(...)` call.

Update `config.example.json` — add before `"observability"`:

```json
  "orchestrator": {
    "max_workers": 4,
    "worker_timeout": 120
  },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add agent/config.py config.example.json tests/test_config.py
git commit -m "feat: orchestrator config block with max_workers and worker_timeout"
```

---
### Task 3: Observability — thread-local phase stacks

**Files:**
- Modify: `agent/observability/tracing.py`
- Test: `tests/test_tracing.py`

**Interfaces:**
- Produces: `_Span.phases` becomes a per-OS-thread stack (parallel workers no longer corrupt each other's `current_phase()`); `phase()`, `current_phase()`, `trace_span()` public behavior unchanged.
- Consumes: nothing. Required before Task 4 runs workers in parallel with observability installed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing.py` (needs `from agent.observability.tracing import _Span` added to the existing import at the top):

```python
def test_span_phases_are_thread_local():
    span = _Span(trace_id="x")
    results = {}

    def worker(name):
        span.phases.append(name)
        time.sleep(0.05)
        results[name] = span.phases[-1]
        span.phases.pop()

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results == {"A": "A", "B": "B"}
```

The `import threading` at the top of `tests/test_tracing.py` already exists (line 2); add `time` to it (`import threading` → `import threading` stays, add `import time` as its own line).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_span_phases_are_thread_local -v`
Expected: FAIL — with a shared list, the interleaved `append`/`pop` across the two threads makes at least one thread read the other's phase (assert fails), or the shared-list dataclass field makes both see `"B"`.

- [ ] **Step 3: Write minimal implementation**

In `agent/observability/tracing.py`, replace the `_Span` dataclass:

```python
@dataclass
class _Span:
    """Live per-trace state of the current execution context.

    `phases` is a per-OS-thread stack (via `threading.local`): each worker
    thread keeps its own nesting path, so parallel workers never corrupt each
    other's `current_phase()`. Everything else about the phase model is
    unchanged — the stack holds the chain from outermost phase to innermost
    one for the current thread only.
    """

    trace_id: str
    _local: threading.local = field(default_factory=threading.local)

    @property
    def phases(self) -> list[str]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        return stack
```

`phase()` and `current_phase()` keep working unchanged (`span.phases.append(...)`,
`span.phases.pop()`, `span.phases[-1]` now hit the thread-local stack).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -q`
Expected: all pass (existing `test_span_stack_ctx_and_phase` etc. unaffected).

- [ ] **Step 5: Commit**

```bash
git add agent/observability/tracing.py tests/test_tracing.py
git commit -m "fix: thread-local observability phase stacks for parallel workers"
```

---
### Task 4: Orchestrator — planner role + parallel run + role/failure aggregation

**Files:**
- Modify: `agent/orchestrator.py`
- Modify: `agent/observability/patch.py` (records must follow the new `WorkerTask` shape)
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_chat.py` (fixture gains roles)
- Test: `tests/test_observability_patch.py` (role records + failure record)

**Interfaces:**
- Consumes: `WorkerTask`, `WorkerResult`, `run_workers` from `agent/worker_pool.py`; `OrchestratorConfig` from `agent/config.py`.
- Produces: `Orchestrator._plan(...) -> list[WorkerTask] | None`; `Orchestrator._worker(question, task: WorkerTask, context, model) -> str`; `Orchestrator._aggregate(question, strategy, context, results: list[WorkerResult], model) -> str`. `Orchestrator.run(question, route, model) -> str` unchanged signature.

- [ ] **Step 1: Write the failing test**

Update `tests/test_orchestrator.py`:

1. Imports: add `from agent.llm import ChatResult, LLMError` and
   `from agent.worker_pool import WorkerResult, WorkerTask`.

2. Add `RaisingClient` after `FakeClient`:

```python
class RaisingClient(FakeClient):
    def __init__(self, responses, raise_on):
        super().__init__(responses)
        self.raise_on = set(raise_on)

    def chat_completion(self, messages, model=None, disable_thinking=False, json_mode=False, json_schema=None):
        if len(self.calls) in self.raise_on:
            self.calls.append((messages, model, disable_thinking, json_mode, json_schema))
            raise LLMError("worker boom")
        return super().chat_completion(messages, model=model, disable_thinking=disable_thinking,
                                       json_mode=json_mode, json_schema=json_schema)
```

3. Update the planner JSON fixtures to include `role`:
   - `test_run_normal_path_planner_workers_aggregator` line 48:
     `'{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}, {"title": "t2", "instruction": "i2", "role": "R2"}]}'`
   - `test_run_worker_empty_output_still_aggregates` line 107: same JSON as above.
   - `test_run_planner_uses_json_object_main_path` line 125:
     `'{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}]}'`

4. Add new tests:

```python
def test_worker_prompt_includes_role():
    client = FakeClient(["w1"])
    orch = Orchestrator(client, _config(), _domain())
    orch._worker("q", WorkerTask("t1", "i1", "Architecture"), "ctx", "high-a")
    assert len(client.calls) == 1
    sys_content = client.calls[0][0][0]["content"]
    assert "Role: Architecture" in sys_content
    assert "Sub-task: i1" in sys_content


def test_plan_role_defaults_to_title():
    client = FakeClient(['{"tasks": [{"title": "t1", "instruction": "i1"}]}'])
    orch = Orchestrator(client, _config(), _domain())
    tasks = orch._plan("q", "debugging", "ctx", "high-a")
    assert tasks == [WorkerTask("t1", "i1", "t1")]


def test_plan_planner_prompt_mentions_roles():
    client = FakeClient(['{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}]}'])
    orch = Orchestrator(client, _config(), _domain())
    orch._plan("q", "debugging", "ctx", "high-a")
    planner_sys = client.calls[0][0][0]["content"]
    assert "distinct analysis role" in planner_sys


def test_aggregate_includes_role_labels_and_failure_marker():
    client = FakeClient(["final"])
    orch = Orchestrator(client, _config(), _domain())
    results = [
        WorkerResult(WorkerTask("t1", "i1", "R1"), text="good output"),
        WorkerResult(WorkerTask("t2", "i2", "R2"), text=None, error="timeout"),
    ]
    answer = orch._aggregate("q", "debugging", "ctx", results, "high-a")
    assert answer == "final"
    user_content = client.calls[0][0][-1]["content"]
    assert "Sub-task (R1): t1" in user_content
    assert "good output" in user_content
    assert "Sub-task (R2): t2" in user_content
    assert "[worker failed: timeout]" in user_content


def test_run_partial_worker_failure_aggregates_partial():
    client = RaisingClient([
        '{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}, {"title": "t2", "instruction": "i2", "role": "R2"}]}',
        "w1", "final",
    ], raise_on={1})
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "final"
    assert len(client.calls) == 4
    agg_user = client.calls[3][0][-1]["content"]
    assert "w1" in agg_user
    assert "[worker failed:" in agg_user


def test_run_all_workers_fail_degrades_to_direct():
    client = RaisingClient([
        '{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}, {"title": "t2", "instruction": "i2", "role": "R2"}]}',
        "direct answer",
    ], raise_on={1, 2})
    result = Orchestrator(client, _config(), _domain()).run("huge task", _route(), "high-a")
    assert result == "direct answer"
    assert len(client.calls) == 4
```

> Note: worker→output mapping is nondeterministic under parallel execution, so
> parallel-path tests assert on aggregates (call counts, presence of strings),
> never on which worker produced which output. Deterministic mapping is asserted
> via the direct `_worker`/`_plan`/`_aggregate` unit tests above.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL — planner schema/parse do not know `role`; `_worker`/`_aggregate` signatures unchanged; `run` has no `run_workers`.

- [ ] **Step 3: Write minimal implementation**

In `agent/orchestrator.py`:

1. Imports: `from .config import AgentConfig, DomainConfig, OrchestratorConfig` and
   `from .worker_pool import WorkerResult, WorkerTask, run_workers`.

2. `_planner_schema()` — add `role`:

```python
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
                        "role": {"type": "string"},
                    },
                    "required": ["title", "instruction", "role"],
                },
            }
        },
        "required": ["tasks"],
    }
```

3. `_PLANNER_PROMPT` — add a role rule after the decomposition rule:

```python
Rules:
- Decompose the user's complex task into 2-4 focused sub-tasks.
- Each sub-task must be answerable by a single standalone LLM call.
- Assign each sub-task a distinct analysis role (e.g. Architecture, Scalability,
  Reliability / Failure Modes, Operations) that defines its focused responsibility.
- Output ONLY a single JSON object: {{"tasks": [{{"title": "...", "instruction": "...", "role": "..."}}]}}
```

4. `_PLANNER_DEGRADED_INSTRUCTION` — match the new shape:

```python
{
  "tasks": [
    {"title": "<short sub-task title>", "instruction": "<standalone sub-task instruction>", "role": "<analysis role>"}
  ]
}
```

5. `_plan` — build `WorkerTask`s, default role to title:

```python
    def _plan(
        self, question: str, strategy: str, context: str, model: str
    ) -> list[WorkerTask] | None:
        prompt = _PLANNER_PROMPT.format(
            name=self.domain.name,
            description=self.domain.description,
            context=context,
            question=question,
        )
        messages = [{"role": "system", "content": prompt}]
        result = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_mode=True
        )
        data = _parse_json(result.text)
        if not data or not isinstance(data.get("tasks"), list):
            return None
        tasks: list[WorkerTask] = []
        for item in data["tasks"]:
            if not isinstance(item, dict):
                return None
            title = item.get("title")
            instruction = item.get("instruction")
            if not isinstance(title, str) or not isinstance(instruction, str):
                return None
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = title
            tasks.append(WorkerTask(title=title, instruction=instruction, role=role))
        return tasks or None
```

6. `run()` — parallel workers via the pool:

```python
    def run(self, question: str, route: RouteResult, model: str) -> str:
        context = self._strategy_context(route.strategy)
        tasks = self._plan(question, route.strategy, context, model)
        # TODO: add Evaluator / Optimizer phases after aggregation (future)
        if tasks is None:
            return self._direct_answer(question, route.strategy, context, model)
        oc = self.config.orchestrator or OrchestratorConfig()
        results = run_workers(
            tasks,
            lambda task: self._worker(question, task, context, model),
            max_workers=oc.max_workers,
            timeout=oc.worker_timeout,
        )
        if all(r.error for r in results):
            return self._direct_answer(question, route.strategy, context, model)
        return self._aggregate(question, route.strategy, context, results, model)
```

7. `_worker` — role in the system prompt:

```python
    def _worker(self, question: str, task: WorkerTask, context: str, model: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"{context}\n\nRole: {task.role}\nSub-task: {task.instruction}",
            },
            {"role": "user", "content": question},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text
```

8. `_aggregate` — role labels + failure markers:

```python
    def _aggregate(
        self,
        question: str,
        strategy: str,
        context: str,
        results: list[WorkerResult],
        model: str,
    ) -> str:
        sections = []
        for r in results:
            label = f"Sub-task ({r.task.role}): {r.task.title}"
            if r.error:
                sections.append(f"{label}\n[worker failed: {r.error}]")
            else:
                sections.append(f"{label}\n{r.text}")
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
                    "answer to the user's original question. Some sub-task results "
                    "may be missing due to worker failure; produce the best answer "
                    "from what is available."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text
```

In `agent/observability/patch.py` (must follow the new shapes or the suite breaks):

1. `_wrap_plan` — record roles:

```python
                if tid:
                    data = {"degraded": True} if tasks is None else {
                        "tasks": [{"title": t.title, "instruction": t.instruction,
                                   "role": t.role} for t in tasks]}
                    inst._record_decision(tid, inst._phase(key), data)
```

2. `_wrap_worker` — use `task.title`/`task.role` and record failures:

```python
    def _wrap_worker(self, original, key):
        def wrapper(orch, question, task, context, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, task, context, model)
            base = inst._phase(key)
            n = inst._next_worker(current_trace_id() or "")
            with phase(f"{base}.{n}"):
                tid = current_trace_id()
                try:
                    result = original(orch, question, task, context, model)
                except Exception as e:  # noqa: BLE001 - record failure, then re-raise; business decides
                    if tid:
                        inst._record_decision(tid, f"{base}.{n}", {
                            "task": task.title, "role": task.role, "error": str(e)})
                    raise
                if tid:
                    inst._record_decision(tid, f"{base}.{n}", {
                        "task": task.title, "role": task.role})
                return result
        return wrapper
```

Update `tests/test_chat.py` line 70 planner JSON to include roles:

```python
        '{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}, {"title": "t2", "instruction": "i2", "role": "R2"}]}',
```

(`tests/test_domain_agnostic.py` and `tests/test_observability_patch.py` fixtures without `role` keep working — role defaults to `title`; `tests/test_report.py` / `tests/test_report_data.py` use hand-crafted events, unaffected.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: all pass, including the new role/failure/parallel tests.

Run: `uv run pytest -q`
Expected: all green. If `test_observability_patch.py` is flaky on which worker
produced which output, remember its assertions are phase-presence-only, so it
must stay green; if it is not, stop and check the thread-local fix from Task 3.

- [ ] **Step 5: Commit**

```bash
git add agent/orchestrator.py agent/observability/patch.py tests/test_orchestrator.py tests/test_chat.py
git commit -m "feat: parallel workers with roles and partial-failure aggregation"
```

---
### Task 5: Observability tests — role + failure records

**Files:**
- Test: `tests/test_observability_patch.py`

**Interfaces:**
- Consumes: `Orchestrator`/`patch` behavior from Task 4.
- Produces: regression coverage proving the planner decision records `role` and a failed worker records `error`.

- [ ] **Step 1: Write the failing test**

Update `tests/test_observability_patch.py`:

1. Imports: add `import threading`, and `from agent.llm import ChatResult, LLMError`.

2. Update `_PLAN` (line 33) to include roles:

```python
_PLAN = '{"tasks": [{"title": "t1", "instruction": "i1", "role": "R1"}, {"title": "t2", "instruction": "i2", "role": "R2"}]}'
```

3. Add `RaisingInner` after `FakeInner`:

```python
class RaisingInner(FakeInner):
    def __init__(self, responses, raise_on_call):
        super().__init__(responses)
        self.raise_on_call = raise_on_call
        self._count = 0
        self._lock = threading.Lock()

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs):
        with self._lock:
            self._count += 1
            should_raise = self._count == self.raise_on_call
        if should_raise:
            raise LLMError("worker boom")
        return super().chat_completion(messages, model=model, temperature=temperature, **kwargs)
```

4. Add tests:

```python
def test_orchestration_planner_decision_records_roles(tmp_path):
    store = _store(tmp_path)
    inner = FakeInner([_CLASSIFY_COMPLEX, _PLAN, "w1", "w2", "final"])
    chat = Chat(inner, _config(), _domain_complex())
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("huge debugging task")
    assert resp.text == "final"
    events, _ = read_events(tmp_path / "obs")
    planner = [e for e in events if e["type"] == "decision" and e["phase"] == "orchestration.planner"]
    assert len(planner) == 1
    assert [t["role"] for t in planner[0]["data"]["tasks"]] == ["R1", "R2"]


def test_orchestration_worker_failure_recorded(tmp_path):
    store = _store(tmp_path)
    inner = RaisingInner([_CLASSIFY_COMPLEX, _PLAN, "w1", "final"], raise_on_call=3)
    chat = Chat(inner, _config(), _domain_complex())
    patch_mod.Installed(store, {}).apply()
    resp = chat.respond("huge debugging task")
    assert resp.text == "final"  # partial failure still yields an answer
    events, _ = read_events(tmp_path / "obs")
    worker_decisions = [
        e for e in events
        if e["type"] == "decision" and e["phase"].startswith("orchestration.worker.")
    ]
    errors = [e["data"].get("error") for e in worker_decisions]
    assert len(errors) == 2
    assert errors.count(None) == 1 and errors.count("worker boom") == 1
```

> Call indexing for the failure test: `chat.respond` → classify (call 1),
> planner (call 2), two parallel workers (calls 3 & 4), aggregator (call 5).
> `raise_on_call=3` makes exactly one worker raise regardless of scheduling;
> which worker it is is nondeterministic, so the test asserts on the event set.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_observability_patch.py::test_orchestration_planner_decision_records_roles tests/test_observability_patch.py::test_orchestration_worker_failure_recorded -v`
Expected: FAIL — the planner record has no `role` key, and `_wrap_worker` records neither role nor `error`.

- [ ] **Step 3: Implement**

No production-code changes needed — the behavior was implemented in Task 4.
The tests pass against it. (If they do not, revisit Task 4 Step 3.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_observability_patch.py -q`
Expected: all pass.

- [ ] **Step 5: Run the full suite and commit**

Run: `uv run pytest -q`
Expected: all green.

```bash
git add tests/test_observability_patch.py
git commit -m "test: observability records planner roles and worker failures"
```

---
## Self-Review Notes

- **Spec coverage:** §14 (concurrency → Task 1/4, max workers + timeout → Task 2/4,
  partial failure → Task 4, aggregator partial handling → Task 4, all-failed degrade
  → Task 4); §15 (planner role in schema+prompt → Task 4, role in worker prompt →
  Task 4, role default to title → Task 4, role labels in aggregation → Task 4);
  observability (§6 of spec) → Tasks 3/4/5.
- **Placeholder scan:** every code step contains real code; no TBD/TODO-later steps.
- **Type consistency:** `WorkerTask(title, instruction, role)` / `WorkerResult(task, text, error)` /
  `run_workers(tasks, run_one, *, max_workers, timeout)` / `OrchestratorConfig(max_workers, worker_timeout)`
  are used identically in Tasks 1, 2, and 4. `_plan` returns `list[WorkerTask] | None`; `_worker`
  takes `WorkerTask`; `_aggregate` takes `list[WorkerResult]` — the patch.py wrappers match.
- **Flakiness guard:** parallel-path tests assert on aggregates/counts/presence, never on
  specific worker→output mapping; deterministic mapping is covered by direct unit calls.