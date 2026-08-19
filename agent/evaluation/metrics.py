from __future__ import annotations

from agent.evaluation.dataset import Suite, TIERS, is_in_domain
from agent.evaluation.judge import JUDGE_DIMENSIONS
from agent.evaluation.runner import CaseResult

_COST_KEYS = ("llm_calls", "in_tokens", "out_tokens", "total_tokens",
              "cache_tokens", "latency_ms")


def _accuracy(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return round(correct / total, 4)


def _zero_cost() -> dict:
    return {k: 0 for k in _COST_KEYS}


def _add_cost(acc: dict, r: CaseResult) -> None:
    for k in _COST_KEYS:
        acc[k] += getattr(r, k, 0)


def compute_metrics(suite: Suite, results: list[CaseResult]) -> dict:
    n = len(results)
    domain_correct = 0
    intent_total = 0
    intent_correct = 0
    complexity_total = 0
    complexity_correct = 0
    strategy_correct = 0
    orchestration_correct = 0
    model_total = 0
    model_correct = 0
    per_intent: dict[str, list[bool]] = {}
    per_intent_order: list[str] = []
    per_complexity: dict[str, list[bool]] = {}
    per_complexity_order: list[str] = []
    per_strategy: dict[str, list[bool]] = {}
    per_strategy_order: list[str] = []
    judged: list[dict] = []
    total_cost = _zero_cost()
    by_path: dict[str, dict] = {}

    for r in results:
        c = r.case
        expected_in = is_in_domain(c, suite)
        if r.in_domain == expected_in:
            domain_correct += 1
        if r.strategy == c.expected_strategy:
            strategy_correct += 1
            per_strategy.setdefault(c.expected_strategy, []).append(True)
        else:
            per_strategy.setdefault(c.expected_strategy, []).append(False)
        if c.expected_strategy not in per_strategy_order:
            per_strategy_order.append(c.expected_strategy)
        if r.orchestrate == c.expected_orchestrate:
            orchestration_correct += 1
        if r.actual_model is not None and r.expected_model is not None:
            model_total += 1
            if r.actual_model == r.expected_model:
                model_correct += 1
        if expected_in:
            intent_total += 1
            if r.intent == c.expected_intent:
                intent_correct += 1
                per_intent.setdefault(c.expected_intent, []).append(True)
            else:
                per_intent.setdefault(c.expected_intent, []).append(False)
            if c.expected_intent not in per_intent_order:
                per_intent_order.append(c.expected_intent)
            complexity_total += 1
            complexity_ok = r.complexity == c.expected_complexity
            if complexity_ok:
                complexity_correct += 1
            if complexity_ok:
                per_complexity.setdefault(c.expected_complexity, []).append(True)
            else:
                per_complexity.setdefault(c.expected_complexity, []).append(False)
            if c.expected_complexity not in per_complexity_order:
                per_complexity_order.append(c.expected_complexity)
            path = c.expected_complexity
            if path not in by_path:
                by_path[path] = _zero_cost()
            _add_cost(by_path[path], r)
        if r.scorecard is not None:
            judged.append(r.scorecard)
        _add_cost(total_cost, r)

    total_cost["latency_ms"] = round(total_cost["latency_ms"], 1)
    for path in by_path:
        by_path[path]["latency_ms"] = round(by_path[path]["latency_ms"], 1)

    per_intent_accuracy = {}
    for iid in per_intent_order:
        marks = per_intent[iid]
        per_intent_accuracy[iid] = _accuracy(sum(marks), len(marks))

    per_complexity_accuracy = {}
    for level in per_complexity_order:
        marks = per_complexity[level]
        per_complexity_accuracy[level] = _accuracy(sum(marks), len(marks))

    per_strategy_accuracy = {}
    for sid in per_strategy_order:
        marks = per_strategy[sid]
        per_strategy_accuracy[sid] = _accuracy(sum(marks), len(marks))

    answer_quality = {}
    if judged:
        for dim in JUDGE_DIMENSIONS:
            answer_quality[dim] = round(
                sum(j[dim] for j in judged) / len(judged), 2
            )

    return {
        "n_cases": n,
        "n_failed": sum(1 for r in results if r.error is not None),
        "classification": {
            "domain_accuracy": _accuracy(domain_correct, n),
            "intent_accuracy": _accuracy(intent_correct, intent_total),
            "complexity_accuracy": _accuracy(complexity_correct, complexity_total),
            "per_intent": per_intent_accuracy,
            "per_complexity": per_complexity_accuracy,
        },
        "routing": {
            "strategy_accuracy": _accuracy(strategy_correct, n),
            "per_strategy": per_strategy_accuracy,
            "orchestration_accuracy": _accuracy(orchestration_correct, n),
            "model_routing_accuracy": _accuracy(model_correct, model_total),
        },
        "answer_quality": answer_quality,
        "cost": {"by_path": by_path, **total_cost},
    }


def compute_metrics_by_tier(cases, results, *, domain: str) -> dict[str, dict]:
    by_tier: dict[str, dict] = {}
    for tier in TIERS:
        tier_cases = [c for c in cases if c.tier == tier]
        tier_ids = {c.id for c in tier_cases}
        tier_results = [r for r in results if r.case.id in tier_ids]
        by_tier[tier] = compute_metrics(
            Suite(name=tier, domain=domain, cases=tier_cases), tier_results
        )
    return by_tier


def case_failures(case, result, domain: str) -> list[str]:
    if result.error:
        return [f"error: {result.error}"]
    reasons = []
    expected_in = case.expected_domain == domain
    if result.in_domain != expected_in:
        reasons.append("domain mismatch")
    if expected_in:
        if result.intent != case.expected_intent:
            reasons.append(f"intent mismatch (expected {case.expected_intent}, got {result.intent})")
        if result.complexity != case.expected_complexity:
            reasons.append("complexity mismatch")
    if result.strategy != case.expected_strategy:
        reasons.append("strategy mismatch")
    if result.orchestrate != case.expected_orchestrate:
        reasons.append("orchestration mismatch")
    if (result.actual_model is not None and result.expected_model is not None
            and result.actual_model != result.expected_model):
        reasons.append("model routing mismatch")
    if result.scorecard is not None and any(v < 3 for v in result.scorecard.values()):
        reasons.append("judge score below threshold (<3)")
    return reasons


def failed_cases(results, domain: str) -> list[dict]:
    out = []
    for r in results:
        reasons = case_failures(r.case, r, domain)
        if reasons:
            out.append({
                "id": r.case.id,
                "tier": r.case.tier,
                "suite": r.suite,
                "question": r.case.question,
                "reasons": reasons,
            })
    return out
