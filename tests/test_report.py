import json
from pathlib import Path

import pytest

from agent.observability.report import build_html_report, main, summarize_traces
from agent.observability.report_data import TraceSummary


def _events():
    return [
        {"type": "trace_start", "trace_id": "a", "question": "q1", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "decision", "trace_id": "a", "phase": "classification", "ts": 5,
         "data": {"in_domain": True, "intent": "question", "complexity": "low",
                  "reason": "self explanatory"}},
        {"type": "llm_call", "trace_id": "a", "phase": "classification", "model": "m",
         "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
         "latency_ms": 100, "status": "ok", "ts": 10},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.direct", "model": "m",
         "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30,
         "latency_ms": 200, "status": "ok", "ts": 20},
        {"type": "trace_end", "trace_id": "a", "answer_len": 50, "total_llm_calls": 2,
         "total_tokens": 45, "total_latency_ms": 300.0, "reject": False,
         "phase": "trace", "ts": 300},
    ]


def _worker_events():
    return [
        {"type": "trace_start", "trace_id": "b", "question": "q2", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "decision", "trace_id": "b", "phase": "orchestration.worker.1", "ts": 2,
         "data": {"task": "task alpha"}},
        {"type": "llm_call", "trace_id": "b", "phase": "orchestration.worker.1", "model": "m",
         "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
         "latency_ms": 50, "status": "ok", "ts": 3},
        {"type": "decision", "trace_id": "b", "phase": "orchestration.worker.2", "ts": 4,
         "data": {"task": "task beta"}},
        {"type": "llm_call", "trace_id": "b", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11,
         "latency_ms": 60, "status": "ok", "ts": 5},
        {"type": "trace_end", "trace_id": "b", "answer_len": 30, "total_llm_calls": 2,
         "total_tokens": 18, "total_latency_ms": 110.0, "reject": False,
         "phase": "trace", "ts": 6},
    ]


def test_summarize_traces_reexported():
    rows = summarize_traces(_events())
    assert isinstance(rows[0], TraceSummary)
    assert rows[0].in_tokens == 30
    assert rows[0].out_tokens == 15


def test_html_has_summary_and_labels():
    html = build_html_report(_events())
    for token in ("<html", "traces", "LLM calls", "q1", "strategy.direct", "<svg",
                  "classification", "time="):
        assert token in html


def test_html_decision_payload_json():
    html = build_html_report(_events())
    assert "<pre class='json'>" in html
    assert "&quot;intent&quot;: &quot;question&quot;" in html
    assert "&quot;complexity&quot;: &quot;low&quot;" in html
    assert "intent=question" not in html


def test_html_no_list_numbers():
    html = build_html_report(_events())
    assert "<ol" not in html
    assert "class='stages'" in html


def test_html_phase_latency_merges_workers_and_strategies():
    events = [
        {"type": "trace_start", "trace_id": "a", "question": "q", "domain": "sw",
         "phase": "trace", "ts": 1},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.1", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 30, "status": "ok", "ts": 2},
        {"type": "llm_call", "trace_id": "a", "phase": "orchestration.worker.2", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 70, "status": "ok", "ts": 3},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.teaching", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 100, "status": "ok", "ts": 4},
        {"type": "llm_call", "trace_id": "a", "phase": "strategy.analysis", "model": "m",
         "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
         "latency_ms": 200, "status": "ok", "ts": 5},
        {"type": "trace_end", "trace_id": "a", "answer_len": 1, "total_llm_calls": 4,
         "total_tokens": 8, "total_latency_ms": 400.0, "reject": False,
         "phase": "trace", "ts": 6},
    ]
    html = build_html_report(events)
    assert "100 ms" in html   # orchestration.worker = 30 + 70
    assert "300 ms" in html   # strategy = 100 + 200
    assert "30 ms" not in html
    assert "70 ms" not in html


def test_html_worker_stage_nested():
    html = build_html_report(_worker_events())
    assert "orchestration.worker" in html
    assert "worker1: task alpha" in html
    assert "worker2: task beta" in html
    assert html.count("class=\"worker\"") == 2


def test_html_has_time_labels():
    html = build_html_report(_events())
    assert "time=0.1s" in html  # classification call 100ms
    assert "time=0.2s" in html  # strategy.direct call 200ms
    assert "time=0.3s" in html  # trace summary header (300ms)
    assert "time=0.3s total" in html  # top summary strip


def test_html_details_collapsed_by_default():
    html = build_html_report(_events())
    assert "<details>" in html
    assert "<details open" not in html
    html2 = build_html_report(_events(), default_collapsed=False)
    assert "<details open" in html2


def test_html_result_header_on_own_line():
    html = build_html_report(_events())
    assert "<b>result</b>" in html
    # the content follows in a nested block-level <ul>, so it renders on its own
    # line below the header (block elements stack vertically in the browser)
    assert "<b>result</b><ul><li class=\"result\">" in html
    assert "answer_len=50 reject=False, total in=30 out=15 tokens, time=0.3s" in html


def test_html_pre_json_wraps_long_lines():
    html = build_html_report(_events())
    assert "white-space:pre-wrap" in html
    assert "overflow-wrap:anywhere" in html


def test_html_escapes_user_content():
    events = _events()
    events[0]["question"] = "<script>alert('x')</script>"
    events[1]["data"]["reason"] = "</b><script>"
    html = build_html_report(events)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_main_writes_report_and_prints_path(tmp_path, capsys):
    day_file = tmp_path / "trace-2026-08-11.jsonl"
    day_file.write_text("\n".join(json.dumps(e) for e in _events()) + "\n", encoding="utf-8")
    code = main(["report", "--data-dir", str(tmp_path), "--day", "2026-08-11"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Report written to" in out
    assert (tmp_path / "report.html").exists()
    assert "<html" in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_build_cli_report_removed():
    with pytest.raises(ImportError):
        from agent.observability.report import build_cli_report


def test_main_write_failure_degrades(tmp_path, monkeypatch):
    day_file = tmp_path / "trace-2026-08-11.jsonl"
    day_file.write_text("\n".join(json.dumps(e) for e in _events()) + "\n", encoding="utf-8")

    def _boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)
    with pytest.warns(UserWarning, match="failed to write HTML report"):
        code = main(["report", "--data-dir", str(tmp_path), "--day", "2026-08-11"])
    assert code == 1
    assert not (tmp_path / "report.html").exists()
