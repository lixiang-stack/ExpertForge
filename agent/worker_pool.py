"""Parallel worker execution with bounded concurrency and per-worker timeout."""

from __future__ import annotations

import contextvars
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
    returned in input order. `run_one` is injected so this module needs no LLM.

    Each work item runs within its own copy of the caller's contextvars context
    (the same approach asyncio's executor/task machinery uses): ThreadPoolExecutor
    threads start with an empty context, so this keeps contextvars (e.g. an
    observability span/trace_id) visible inside the worker threads. A single
    Context cannot be entered by more than one thread at once, so each submitted
    work item gets a fresh copy rather than sharing one."""
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            i: executor.submit(
                lambda t=task, c=contextvars.copy_context(): c.run(run_one, t)
            )
            for i, task in enumerate(tasks)
        }
        results: list[WorkerResult] = []
        for i, task in enumerate(tasks):
            try:
                text = futures[i].result(timeout=timeout)
            except FutureTimeoutError:
                results.append(WorkerResult(task=task, text=None, error="timeout"))
            except Exception as e:  # noqa: BLE001 - worker failure is captured, not propagated
                results.append(WorkerResult(task=task, text=None, error=str(e) or repr(e)))
            else:
                results.append(WorkerResult(task=task, text=text))
    finally:
        # wait=False so a hung worker beyond the timeout never blocks the pipeline;
        # cancel_futures=True drops queued-but-not-started tasks.
        executor.shutdown(wait=False, cancel_futures=True)
    return results
