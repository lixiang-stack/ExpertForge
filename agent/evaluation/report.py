from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _case_record(r) -> dict:
    return {
        "id": r.case.id,
        "question": r.case.question,
        "category": r.case.category,
        "expected_domain": r.case.expected_domain,
        "expected_intent": r.case.expected_intent,
        "expected_complexity": r.case.expected_complexity,
        "expected_strategy": r.case.expected_strategy,
        "expected_orchestrate": r.case.expected_orchestrate,
        "in_domain": r.in_domain,
        "intent": r.intent,
        "complexity": r.complexity,
        "strategy": r.strategy,
        "orchestrate": r.orchestrate,
        "answer": r.answer,
        "actual_model": r.actual_model,
        "expected_model": r.expected_model,
        "scorecard": r.scorecard,
        "llm_calls": r.llm_calls,
        "in_tokens": r.in_tokens,
        "out_tokens": r.out_tokens,
        "total_tokens": r.total_tokens,
        "cache_tokens": r.cache_tokens,
        "latency_ms": r.latency_ms,
    }


def serialize_results(
    cases,
    metrics,
    *,
    domain: str,
    label: str,
    model: str,
    judge_model: str | None,
    skip_quality: bool,
    dataset_path: str,
) -> dict:
    return {
        "domain": domain,
        "label": label,
        "model": model,
        "judge_model": judge_model,
        "skip_quality": skip_quality,
        "dataset": dataset_path,
        "metrics": metrics,
        "cases": [_case_record(r) for r in cases],
    }


def write_result(results_dir: str, record: dict, *, label: str) -> str:
    base = Path(results_dir)
    base.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    path = base / f"{day}-{label}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _fmt_accuracy(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _fmt_cost(cost: dict) -> str:
    return (
        f"llm_calls={cost.get('llm_calls', 0)} "
        f"in={cost.get('in_tokens', 0)} out={cost.get('out_tokens', 0)} "
        f"total={cost.get('total_tokens', 0)} cache={cost.get('cache_tokens', 0)} "
        f"latency={cost.get('latency_ms', 0)}ms"
    )


def format_summary(record: dict) -> str:
    m = record["metrics"]
    cls = m["classification"]
    routing = m["routing"]
    aq = m["answer_quality"]
    cost = m["cost"]
    lines = [
        f"Evaluation run: {record['label']}  (domain={record['domain']}, "
        f"cases={m['n_cases']}, model={record['model']}, "
        f"judge_model={record['judge_model'] or record['model']})",
        "",
        "Classification:",
        f"  domain_accuracy     {_fmt_accuracy(cls['domain_accuracy'])}",
        f"  intent_accuracy     {_fmt_accuracy(cls['intent_accuracy'])}",
        f"  complexity_accuracy {_fmt_accuracy(cls['complexity_accuracy'])}",
    ]
    if cls["per_intent"]:
        lines.append("  per_intent:")
        for iid, acc in cls["per_intent"].items():
            lines.append(f"    {iid}: {_fmt_accuracy(acc)}")
    lines += [
        "",
        "Routing:",
        f"  strategy_accuracy        {_fmt_accuracy(routing['strategy_accuracy'])}",
        f"  orchestration_accuracy   {_fmt_accuracy(routing['orchestration_accuracy'])}",
        f"  model_routing_accuracy   {_fmt_accuracy(routing['model_routing_accuracy'])}",
        "",
        "Answer quality (judged cases):",
    ]
    if aq:
        for dim, mean in aq.items():
            lines.append(f"  {dim}: {mean}")
    else:
        lines.append("  (none)")
    lines += ["", "Cost / latency (total):", f"  {_fmt_cost(cost)}", "  by_path:"]
    for path, pcost in cost["by_path"].items():
        lines.append(f"    {path}: {_fmt_cost(pcost)}")
    return "\n".join(lines) + "\n"
