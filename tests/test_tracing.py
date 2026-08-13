import json
import threading
import warnings
from pathlib import Path

from agent.observability.tracing import (
    TraceStore,
    current_phase,
    current_trace_id,
    format_trace_summary,
    phase,
    read_events,
    trace_span,
)


def _write_snapshot(tmp_path, lines):
    day_dir = tmp_path / "observability"
    day_dir.mkdir(parents=True, exist_ok=True)
    p = day_dir / "trace-2026-08-11.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return day_dir


def test_read_events_ok_and_bad_lines(tmp_path):
    day_dir = _write_snapshot(tmp_path, [
        json.dumps({"type": "llm_call", "trace_id": "a", "phase": "classification"}),
        "{this is not json",
        json.dumps({"type": "llm_call", "trace_id": "b", "phase": "route"}),
    ])
    events, bad = read_events(day_dir, day="2026-08-11")
    assert bad == 1
    assert [e["phase"] for e in events] == ["classification", "route"]


def test_read_events_nonexistent_day_returns_empty(tmp_path):
    events, bad = read_events(tmp_path / "observability", day="2099-01-01")
    assert events == []
    assert bad == 0


def test_trace_store_writes_and_reads_back(tmp_path):
    store = TraceStore(tmp_path / "obs")
    store.write({"type": "trace_start", "trace_id": "abc", "question": "q"})
    store.write({"type": "llm_call", "trace_id": "abc", "phase": "classification",
                 "prompt_tokens": 3, "completion_tokens": 4})
    store.close()
    events, bad = read_events(tmp_path / "obs")
    assert bad == 0
    assert len(events) == 2
    assert events[0]["type"] == "trace_start"


def test_trace_store_write_after_close_reopens_file(tmp_path):
    store = TraceStore(tmp_path / "obs")
    store.write({"type": "trace_start", "trace_id": "abc", "question": "q"})
    store.close()
    store.write({"type": "llm_call", "trace_id": "abc", "phase": "route"})
    store.close()
    events, bad = read_events(tmp_path / "obs")
    assert bad == 0
    assert len(events) == 2
    assert [e["type"] for e in events] == ["trace_start", "llm_call"]


def test_trace_store_write_never_raises(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / "obs")
    monkeypatch.setattr(store, "_current_file", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store.write({"type": "llm_call"})  # _current_file returns None -> warn path
    assert any("observability" in str(w.message) for w in caught)


def test_trace_store_trace_llm_calls_accumulates_in_memory(tmp_path):
    store = TraceStore(tmp_path / "obs")
    store.write({"type": "trace_start", "trace_id": "abc"})
    store.write({"type": "llm_call", "trace_id": "abc", "phase": "classification"})
    store.write({"type": "llm_call", "trace_id": "abc", "phase": "route"})
    store.write({"type": "llm_call", "trace_id": "zzz", "phase": "classification"})
    calls = store.trace_llm_calls("abc")
    assert [c["phase"] for c in calls] == ["classification", "route"]


def test_span_stack_ctx_and_phase():
    with trace_span() as tid:
        assert tid == current_trace_id()
        assert current_phase() is None
        with phase("classification"):
            assert current_phase() == "classification"
        assert current_phase() is None
    assert current_trace_id() is None
    assert current_phase() is None


def test_phase_noop_without_span():
    with phase("classification"):
        assert current_phase() is None


def test_format_trace_summary():
    calls = [
        {"phase": "classification", "prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 300},
        {"phase": "strategy.direct", "prompt_tokens": 123, "completion_tokens": 45, "latency_ms": 2100},
    ]
    line = format_trace_summary("abc", calls)
    assert line == "[trace abc] in=133 out=50 2.4s"


def test_format_trace_summary_missing_usage_zero():
    line = format_trace_summary("abc", [{"phase": "route", "prompt_tokens": None,
                                          "completion_tokens": None, "latency_ms": None}])
    assert line == "[trace abc] in=0 out=0 0.0s"


def test_read_events_corrupt_non_utf8_day_file_does_not_raise(tmp_path):
    day_dir = tmp_path / "observability"
    day_dir.mkdir(parents=True, exist_ok=True)
    p = day_dir / "trace-2026-08-11.jsonl"
    p.write_bytes(b'{"type": "llm_call", "trace_id": "a"}\n\xff\xfe\n{"type": "llm_call", "trace_id": "b"}\n')
    events, bad = read_events(day_dir, day="2026-08-11")
    assert bad == 1
    assert [e["trace_id"] for e in events] == ["a", "b"]