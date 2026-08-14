from __future__ import annotations

import json


def load_result(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Result file must contain a JSON object: {path}")
    return data


def _num(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _diff_value(a, b) -> str:
    if a is None or b is None:
        return f"{_num(a)} -> {_num(b)}"
    return f"{_num(a)} -> {_num(b)} (delta {b - a:+.2f})"


def diff_runs(record_a: dict, record_b: dict) -> str:
    ma = record_a["metrics"]
    mb = record_b["metrics"]
    lines = [
        f"Comparing {record_a['label']} (model={record_a.get('model')}) -> "
        f"{record_b['label']} (model={record_b.get('model')})",
        "",
        "Classification:",
    ]
    clsa = ma["classification"]
    clsb = mb["classification"]
    for key in ("domain_accuracy", "intent_accuracy", "complexity_accuracy"):
        lines.append(f"  {key}: {_diff_value(clsa[key], clsb[key])}")
    lines += ["", "Routing:"]
    ra = ma["routing"]
    rb = mb["routing"]
    for key in ("strategy_accuracy", "orchestration_accuracy", "model_routing_accuracy"):
        lines.append(f"  {key}: {_diff_value(ra[key], rb[key])}")
    lines += ["", "Answer quality:"]
    aqa = ma["answer_quality"]
    aqb = mb["answer_quality"]
    for dim in sorted(set(aqa) | set(aqb)):
        lines.append(f"  {dim}: {_diff_value(aqa.get(dim), aqb.get(dim))}")
    lines += ["", "Cost:"]
    ca = ma["cost"]
    cb = mb["cost"]
    for key in ("llm_calls", "in_tokens", "out_tokens", "total_tokens",
                "cache_tokens", "latency_ms"):
        lines.append(f"  {key}: {_diff_value(ca[key], cb[key])}")
    return "\n".join(lines) + "\n"
