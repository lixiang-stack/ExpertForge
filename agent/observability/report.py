"""Observability report CLI.

Commands:
  report              generate a self-contained HTML report from trace JSONL
    --data-dir DIR    trace JSONL directory (default: .observability)
    --day YYYY-MM-DD  filter to a single day

Example:
  uv run python -m agent.observability report
  uv run python -m agent.observability report --day 2026-08-11
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import warnings
from pathlib import Path

from .report_data import Stage, build_timeline, group_stages, model_stats, summarize_traces, total_stats
from .tracing import read_events


def _short(s, n=28):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _call_row_html(step) -> str:
    d = step.detail
    badge = "error" if d.get("status") == "error" else "ok"
    err = f' <span class="err">{html.escape(str(d.get("error") or ""))}</span>' if d.get("error") else ""
    secs = float(d.get("latency_ms") or 0) / 1000
    return (f'<li class="call">model={html.escape(str(d.get("model") or ""))} '
            f'in={d.get("in_tokens")} out={d.get("out_tokens")} time={secs:.1f}s [{badge}]{err}</li>')


def _decision_json_html(data: dict) -> str:
    text = json.dumps(data if isinstance(data, dict) else {}, indent=2, ensure_ascii=False)
    return "<pre class='json'>" + html.escape(text) + "</pre>"


def _stage_li(stage: Stage, summary) -> str:
    title = stage.title
    header = f'<b>{html.escape(title)}</b>'
    if title == "result":
        result_step = next((s for s in stage.steps if s.kind == "result"), None)
        answer_len = (result_step.detail or {}).get("answer_len") if result_step else ""
        content = (f'<li class="result">answer_len={answer_len} reject={summary.reject}, '
                   f'total in={summary.in_tokens} out={summary.out_tokens} tokens, '
                   f'time={summary.total_latency_ms / 1000:.1f}s</li>')
        return f'<li class="stage">{header}<ul>{content}</ul></li>'
    body: list[str] = []
    if stage.workers:
        for w in stage.workers:
            calls = "".join(_call_row_html(s) for s in w.steps)
            body.append(
                f'<li class="worker"><b>worker{w.number}: {html.escape(w.task_title)}</b>'
                f'<ul>{calls}</ul></li>'
            )
    else:
        for s in stage.steps:
            if s.kind == "decision":
                data = s.detail.get("data") or {}
                if title == "classification" or title == "route":
                    body.append(f'<li class="decision">{_decision_json_html(data)}</li>')
                elif title == "orchestration.planner":
                    if data.get("degraded"):
                        body.append('<li class="decision">planning degraded</li>')
                    else:
                        tasks = data.get("tasks")
                        if isinstance(tasks, list):
                            for i, t in enumerate(tasks):
                                if isinstance(t, dict):
                                    body.append(
                                        f'<li class="decision">task{i + 1}: '
                                        f'{html.escape(str(t.get("title") or ""))}</li>')
                else:
                    body.append(f'<li class="decision">{_decision_json_html(data)}</li>')
            elif s.kind == "llm_call":
                body.append(_call_row_html(s))
    if body:
        return f'<li class="stage">{header}<ul>{"".join(body)}</ul></li>'
    return f'<li class="stage">{header}</li>'


def _trace_body_html(stages: list[Stage], summary) -> str:
    if not stages:
        return "<p>No steps recorded.</p>"
    return "<ul class='stages'>" + "".join(_stage_li(s, summary) for s in stages) + "</ul>"


def _label_chart(items: list[tuple[str, float]], title: str, caption: str, *, unit: str) -> str:
    """Labeled horizontal-bar SVG with visible text labels and value annotations."""
    header = f'<h3>{html.escape(title)}</h3>'
    if not items:
        return header + f'<p class="caption">{html.escape(caption)} — no data</p>'
    mx = max(v for _, v in items) or 1
    boxes = []
    for i, (label, value) in enumerate(items):
        y = 14 + i * 18
        w = max(2, int(value / mx * 400))
        boxes.append(
            f'<text x="4" y="{y}" font-size="11">{_short(html.escape(label), 30)}</text>'
            f'<rect x="250" y="{y - 9}" width="{w}" height="11" fill="#4a90d9"/>'
            f'<text x="656" y="{y}" font-size="11">{value:g} {html.escape(unit)}</text>'
        )
    height = 14 + len(items) * 18
    svg = f'<svg width="720" height="{height}" xmlns="http://www.w3.org/2000/svg">{"".join(boxes)}</svg>'
    return header + svg + f'<p class="caption">{html.escape(caption)}</p>'


def build_html_report(events: list[dict], *, default_collapsed: bool = True) -> str:
    summaries = summarize_traces(events)
    total = total_stats(events)
    timeline = build_timeline(events)
    grouped = group_stages(timeline)
    open_attr = "" if default_collapsed else " open"

    trend_items = [(f"{_short(r.question)}", r.total_tokens) for r in summaries]
    model_items = [(m.model, m.in_tokens + m.out_tokens) for m in model_stats(events)]
    def _chart_phase(ph: str) -> str:
        if ph.startswith("orchestration.worker."):
            return "orchestration.worker"
        if ph.startswith("strategy."):
            return "strategy"
        return ph

    phase_lat: dict[str, float] = {}
    for steps in timeline.values():
        for s in steps:
            if s.kind == "llm_call":
                ph = _chart_phase(s.phase or "?")
                phase_lat[ph] = phase_lat.get(ph, 0.0) + float(s.detail.get("latency_ms") or 0)
    phase_items = sorted(phase_lat.items(), key=lambda kv: kv[1], reverse=True)

    cards = []
    for r in summaries:
        stages = grouped.get(r.trace_id, [])
        cards.append(
            f'<details{open_attr}><summary>{html.escape(r.trace_id)} — '
            f'{html.escape(_short(r.question))} in={r.in_tokens} out={r.out_tokens} '
            f'total={r.total_tokens} time={r.total_latency_ms / 1000:.1f}s'
            f'</summary>{_trace_body_html(stages, r)}</details>'
        )

    meta = (f'{total["traces"]} traces, {total["llm_calls"]} LLM calls, '
            f'in={total["in_tokens"]} out={total["out_tokens"]} total={total["total_tokens"]} tokens, '
            f'time={total["total_latency_ms"] / 1000:.1f}s total')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ExpertForge observability</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1024px}}
details{{margin:.5rem 0;border:1px solid #ddd;padding:.5rem;border-radius:4px}}
summary{{cursor:pointer}}ul{{margin:.3rem 0 0;list-style:none;padding-left:0}}li{{margin:.15rem 0}}ul ul{{padding-left:1.2rem}}
.caption{{color:#666;font-size:.85rem;margin:.2rem 0 1rem}}
.decision{{color:#1a5276}}.call{{color:#333}}.err{{color:#c0392b}}.result{{color:#7d6608}}
pre.json{{white-space:pre-wrap;overflow-wrap:anywhere;margin:.2rem 0}}
</style></head><body>
<h1>ExpertForge observability report</h1>
<p>{meta}</p>
{_label_chart(trend_items, "Token usage by trace", "Total tokens consumed per question.", unit="tokens")}
{_label_chart(model_items, "Model distribution", "Total in+out tokens per model.", unit="tokens")}
{_label_chart(phase_items, "Phase latency", "Summed latency per phase.", unit="ms")}
<h2>Traces</h2>
{''.join(cards)}
</body></html>
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "report":
        argv = argv[1:]
    parser = argparse.ArgumentParser(prog="agent.observability.report",
                                     description="Generate observability HTML report")
    parser.add_argument("--data-dir", default=".observability", help="trace JSONL directory")
    parser.add_argument("--day", default=None, help="YYYY-MM-DD filter")
    args = parser.parse_args(argv)
    events, bad = read_events(args.data_dir, day=args.day)
    if bad:
        print(f"note: {bad} unreadable trace line(s) skipped", file=sys.stderr)
    try:
        report = build_html_report(events)
        path = Path(args.data_dir) / "report.html"
        path.write_text(report, encoding="utf-8")
        print(f"Report written to {path}")
    except Exception as e:  # noqa: BLE001 - degrade, never crash
        warnings.warn(f"observability: failed to write HTML report: {e}")
        return 1
    return 0
