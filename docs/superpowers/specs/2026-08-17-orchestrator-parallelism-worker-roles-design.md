# Orchestrator Parallelism + Worker Role Specialization — Design

**Date:** 2026-08-17
**Status:** Design agreed; pending user review
**Source requirements:** `draft_v2.md` §14 (Orchestrator 并行化) and §15 (Worker Role Specialization)

## 1. Goal

Two P1 orchestrator upgrades over the MVP pipeline (Planner → Workers → Aggregator):

1. **Parallelism (§14):** run workers concurrently (Planner → parallel Workers → Aggregator)
   with a configurable max-worker limit, per-worker timeout, and partial-failure
   tolerance — a single worker failure must not fail the whole task, and the
   Aggregator must handle partial worker failure.
2. **Role specialization (§15):** the Planner emits a distinct `role` per worker
   (e.g. Architecture, Scalability, Reliability/Failure Modes, Operations) so each
   worker has a clear analysis responsibility.

## 2. Decisions (agreed with user)

| Decision | Choice |
|----------|--------|
| Concurrency mechanism | `concurrent.futures.ThreadPoolExecutor`; sync stack unchanged (no async refactor). |
| Implementation structure | New isolated `agent/worker_pool.py` module (Approach B), not in-place in the orchestrator. |
| Worker failure handling | Failed/timed-out workers get a short marker in the aggregator input; Aggregator produces a partial answer. Only if **all** workers fail does `run()` degrade to a direct answer. |
| `role` field | Required in the planner schema + prompt. At parse time, a missing/empty `role` defaults to the task's `title` (graceful, plan not discarded). |
| Config placement | Global `orchestrator` block in `config.json` → `AgentConfig`. |
| Timeout semantics | Per-worker wall-clock timeout, default **120s**. |
| Defaults | `max_workers: 4`, `worker_timeout: 120` (seconds). |

## 3. New module: `agent/worker_pool.py`

No LLM dependency; fully synchronous.

```python
@dataclass
class WorkerTask:
    title: str
    instruction: str
    role: str

@dataclass
class WorkerResult:
    task: WorkerTask
    text: str | None = None      # None when failed / timed out
    error: str | None = None     # short reason: "timeout", "LLMError: ..."

def run_workers(tasks: list[WorkerTask], run_one, *,
                max_workers: int = 4, timeout: float = 120.0) -> list[WorkerResult]:
    """Run run_one(task) per task in a ThreadPoolExecutor capped at max_workers,
    with a per-worker wall-clock timeout. Never raises on worker failure;
    returns results in input task order."""
```

- `run_one(task) -> str` is injected by the orchestrator (a closure calling the
  LLM), keeping the module testable standalone with a stub that needs no LLM.
- `ThreadPoolExecutor(max_workers=...)` + `future.result(timeout=...)`; on
  `TimeoutError` or any `Exception`, the worker is recorded as failed
  (`text=None`, `error="timeout"` / `str(e)`) and never propagated.
- Results returned in input order regardless of completion order.
- `contextvars` are captured at `submit()` and propagate into worker threads
  automatically, so observability `phase()` tracing keeps working unchanged.
- When `max_workers < len(tasks)`, extra tasks queue in the pool and execute in
  batches — no worker is skipped.

## 4. Orchestrator changes (`agent/orchestrator.py`)

### 4.1 Planner

- `_planner_schema()` item becomes `{"title", "instruction", "role"}`, all three
  `required`.
- `_PLANNER_PROMPT` gains a rule: "Assign each sub-task a distinct analysis role
  (e.g. Architecture, Scalability, Reliability/Failure Modes, Operations) that
  defines its focused responsibility."
- Parsing returns `list[WorkerTask]`; `role` missing/empty → default to `title`.
- Degradation unchanged: planner output that fails entirely (no/invalid/malformed
  `tasks`) → `None` → direct answer.

### 4.2 run()

```python
tasks = self._plan(...)                       # list[WorkerTask]
if tasks is None:
    return self._direct_answer(...)
results = run_workers(
    tasks,
    lambda t: self._worker(question, t, context, model),
    max_workers=cfg.max_workers,
    timeout=cfg.worker_timeout,
)
if all(r.error for r in results):
    return self._direct_answer(...)           # all workers failed -> degrade
return self._aggregate(question, strategy, context, results, model)
```

### 4.3 Worker

`_worker(question, task: WorkerTask, context, model) -> str`. System prompt:

```
{context}

Role: {role}
Sub-task: {instruction}
```

### 4.4 Aggregator

`_aggregate(question, strategy, context, results, model)` where `results` is
`list[WorkerResult]`:

- Successful worker → `Sub-task ({role}): {title}\n{output}`.
- Failed worker → `Sub-task ({role}): {title}\n[worker failed: {error}]`.
- System prompt gains one line: "Some sub-task results may be missing due to
  worker failure; produce the best answer from what is available."

## 5. Config changes (`agent/config.py`)

- New `@dataclass OrchestratorConfig(max_workers: int = 4, worker_timeout: float = 120.0)`.
- `AgentConfig` gains `orchestrator: OrchestratorConfig | None = None`.
- `load_config` parses the optional top-level `"orchestrator"` block leniently:
  missing block → defaults; invalid/negative/zero `max_workers` → default;
  `worker_timeout` float seconds, invalid → default. Consistent with existing
  `model_low`/`model_high` handling.
- `config.example.json` documents the block:
  ```json
  "orchestrator": { "max_workers": 4, "worker_timeout": 120 }
  ```
- `Orchestrator.__init__` reads `config.orchestrator or OrchestratorConfig()`.

## 6. Observability changes (`agent/observability/patch.py`)

- `_wrap_plan` decision record becomes
  `{"tasks": [{"title": ..., "instruction": ..., "role": ...} ...]}`.
- `_wrap_worker` decision record includes `{"task": task.title, "role": task.role}`,
  and records failure when the result is a failed worker (`{"error": ...}`).
- The per-worker counter is already thread-safe (locked `_next_worker`); no other
  patching changes. Parallel workers share the existing trace/phase machinery.

## 7. Testing

New `tests/test_worker_pool.py` (standalone stub `run_one`, no LLM):
- concurrency across N tasks with `max_workers` cap (submitted concurrently, all
  results returned);
- per-worker timeout: slow stub → `error == "timeout"`, `text is None`;
- exception in `run_one` → failed result, not raised;
- all-failed → all results carry errors;
- input-order preservation;
- queuing when `max_workers < len(tasks)`.

`tests/test_orchestrator.py`:
- planner JSON now includes `role`; worker system prompt contains `Role: ...`;
- missing/empty `role` → defaults to `title`;
- partial failure: one worker raises → aggregator receives the `[worker failed: ...]`
  marker while the other output is present; still a final answer;
- all-workers-fail → direct-answer degradation;
- existing degradation tests (invalid JSON, empty/malformed tasks) unchanged.

`tests/test_chat.py`, `tests/test_observability_patch.py`: updated only where the
planner JSON / decision record shape changes.

`tests/test_config.py`: orchestrator block parsed; missing/invalid → defaults.

## 8. Out of scope (separate P1 items, `draft_v2.md` §16+)

- Evaluator / Optimizer phases.
- Orchestration Policy (§17).
- Strategy prompt content changes beyond the planner role assignment.
- Async / asyncio refactor of the sync stack.

## 9. Success criteria

1. `uv run pytest -q` all green.
2. Workers run concurrently under `max_workers`; each has a `worker_timeout`.
3. One failed/timed-out worker → partial answer from the Aggregator (with a
   failure marker); the task does not fail.
4. All workers failed → degrade to a direct answer, never a crash or broken output.
5. Planner emits a `role` per worker; worker system prompts carry `Role: ...`;
   missing role defaults to `title`.
6. Observability records worker roles and failures.
7. Non-orchestrated behavior unchanged (reject, direct processors, strategy routing).