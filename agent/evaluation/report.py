from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _case_record(r) -> dict:
    return {
        "id": r.case.id,
        "suite": r.suite,
        "tier": r.tier,
        "question": r.case.question,
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
        "error": r.error,
    }


def serialize_results(
    cases,
    metrics,
    metrics_by_tier,
    *,
    domain: str,
    label: str,
    model: str,
    judge_model: str | None,
    tiers: list[str],
    smoke_only: bool,
    dataset_path: str,
    failed_cases: list[dict],
) -> dict:
    return {
        "domain": domain,
        "label": label,
        "model": model,
        "judge_model": judge_model,
        "smoke_only": smoke_only,
        "dataset": dataset_path,
        "tiers": tiers,
        "metrics": metrics,
        "metrics_by_tier": metrics_by_tier,
        "failed_cases": failed_cases,
        "cases": [_case_record(r) for r in cases],
    }


def write_result(results_dir: str, record: dict, *, label: str) -> str:
    base = Path(results_dir)
    base.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    path = base / f"{day}-{label}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def slim_record(record: dict) -> dict:
    """Return the result record without the per-case ``cases`` list.

    Used for the committed baseline: metrics and metadata only, so git history
    of the baseline file reflects iteration effect without case-level noise.
    """
    return {k: v for k, v in record.items() if k != "cases"}


def write_baseline(path: str, record: dict) -> str:
    base = Path(path).parent
    base.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
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
    if record.get("smoke_only"):
        selection = "smoke"
    else:
        selection = "tiers: " + ",".join(record.get("tiers", []))
    lines = [
        f"Evaluation run: {record['label']}  (domain={record['domain']}, "
        f"cases={m['n_cases']}, model={record['model']}, "
        f"judge_model={record['judge_model'] or record['model']}, selection={selection})",
    ]
    failed = record.get("failed_cases") or []
    if failed:
        lines.append(f"Failed cases: {len(failed)}")
    lines += [
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
    if cls.get("per_complexity") or {}:
        lines.append("  per_complexity:")
        for level, acc in cls["per_complexity"].items():
            lines.append(f"    {level}: {_fmt_accuracy(acc)}")
    lines += [
        "",
        "Routing:",
        f"  strategy_accuracy        {_fmt_accuracy(routing['strategy_accuracy'])}",
    ]
    if routing.get("per_strategy"):
        lines.append("  per_strategy:")
        for sid, acc in routing["per_strategy"].items():
            lines.append(f"    {sid}: {_fmt_accuracy(acc)}")
    lines += [
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
    lines += ["", "Per-tier:"]
    for tname in record.get("tiers", []):
        tm = record["metrics_by_tier"].get(tname, {})
        lines.append(
            f"  {tname}: n={tm.get('n_cases', 0)} "
            f"domain={_fmt_accuracy(tm.get('classification', {}).get('domain_accuracy'))} "
            f"intent={_fmt_accuracy(tm.get('classification', {}).get('intent_accuracy'))} "
            f"strategy={_fmt_accuracy(tm.get('routing', {}).get('strategy_accuracy'))} "
            f"{_fmt_cost(tm.get('cost', {}))}"
        )
    if failed:
        lines += ["", "Failed cases:"]
        for fc in failed:
            lines.append(f"  {fc['id']} [{fc['tier']}]: " + "; ".join(fc["reasons"]))
    return "\n".join(lines) + "\n"
