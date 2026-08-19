import contextvars
import threading
import time

from agent.worker_pool import WorkerTask, run_workers


def _tasks(n=4):
    return [WorkerTask(title=f"t{i}", instruction=f"i{i}", role=f"r{i}") for i in range(n)]


def test_run_workers_returns_results_in_input_order():
    results = run_workers(_tasks(3), lambda t: f"out:{t.title}", max_workers=2)
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

    results = run_workers(_tasks(6), run_one, max_workers=2)
    assert max_active <= 2
    assert len(results) == 6
    assert all(r.error is None for r in results)


def test_run_workers_exception_captured_not_raised():
    def run_one(task):
        raise ValueError("boom")

    results = run_workers(_tasks(2), run_one, max_workers=2)
    assert [r.error for r in results] == ["boom", "boom"]
    assert all(r.text is None for r in results)


def test_run_workers_all_failed_returns_all_errors():
    def run_one(task):
        raise RuntimeError("x")

    results = run_workers(_tasks(2), run_one)
    assert [r.error for r in results] == ["x", "x"]
    assert all(r.text is None for r in results)


def test_run_workers_propagates_caller_context():
    cv = contextvars.ContextVar("probe", default="unset")
    cv.set("from-caller")

    def run_one(task):
        return cv.get()

    results = run_workers(_tasks(1), run_one)
    assert results[0].text == "from-caller"


def test_run_workers_empty_tasks():
    assert run_workers([], lambda t: "x") == []


def test_run_workers_rejects_timeout_parameter():
    import pytest
    with pytest.raises(TypeError):
        run_workers(_tasks(1), lambda t: "x", max_workers=1, timeout=1.0)


def test_run_workers_empty_exception_message_uses_repr():
    def run_one(task):
        raise ValueError()

    results = run_workers(_tasks(1), run_one)
    assert results[0].text is None
    assert results[0].error == "ValueError()"
