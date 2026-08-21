from __future__ import annotations

from dataclasses import dataclass, field

from agent.config import AgentConfig, DomainConfig, resolve_judge_model
from agent.evaluation.dataset import EvalCase
from agent.evaluation.judge import Judge
from agent.evaluation.runner import RecordingClient
from agent.llm import LLMClient
from agent.loggers import get_logger
from agent.observability.tracing import trace_span
from agent.orchestrator import Orchestrator
from agent.router import RouteResult
from agent.strategy import build_registry

logger = get_logger("compare")

def _sum_calls(calls: list[dict]) -> dict:
    return {
        "llm_calls": len(calls),
        "in_tokens": sum(c["prompt_tokens"] for c in calls),
        "out_tokens": sum(c["completion_tokens"] for c in calls),
        "total_tokens": sum(c["total_tokens"] for c in calls),
        "cache_tokens": sum(c["cache_tokens"] for c in calls),
        "latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
    }


@dataclass
class ModeRun:
    answer: str | None = None
    scorecard: dict | None = None
    quality: float | None = None
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class CompareCaseResult:
    case: EvalCase
    baseline: ModeRun = field(default_factory=ModeRun)
    orchestrated: ModeRun = field(default_factory=ModeRun)
    quality_gain: float | None = None
    quality_gain_pct: float | None = None
    additional_tokens: int = 0
    token_increase_pct: float | None = None
    cost_efficiency: float | None = None


def _mode_run_from_recorder(
    calls: list[dict], scorecard: dict | None, error: str | None,
) -> ModeRun:
    costs = _sum_calls(calls)
    quality = None
    if scorecard is not None:
        scores = [v for v in scorecard.values() if isinstance(v, (int, float))]
        quality = round(sum(scores) / len(scores), 2) if scores else None
    return ModeRun(scorecard=scorecard, quality=quality, error=error, **costs)


def _compute_deltas(baseline: ModeRun, orchestrated: ModeRun) -> dict:
    quality_gain = None
    quality_gain_pct = None
    additional_tokens = 0
    token_increase_pct = None
    cost_efficiency = None

    if baseline.quality is not None and orchestrated.quality is not None:
        quality_gain = round(orchestrated.quality - baseline.quality, 2)
        if baseline.quality != 0:
            quality_gain_pct = round((quality_gain / baseline.quality) * 100, 2)

    additional_tokens = orchestrated.total_tokens - baseline.total_tokens
    if baseline.total_tokens > 0:
        token_increase_pct = round(
            (additional_tokens / baseline.total_tokens) * 100, 2
        )
    if quality_gain is not None and additional_tokens > 0:
        cost_efficiency = round(quality_gain / additional_tokens, 6)

    return {
        "quality_gain": quality_gain,
        "quality_gain_pct": quality_gain_pct,
        "additional_tokens": additional_tokens,
        "token_increase_pct": token_increase_pct,
        "cost_efficiency": cost_efficiency,
    }


def run_compare(
    config: AgentConfig,
    domain: DomainConfig,
    cases: list[EvalCase],
    client: LLMClient,
    judge_client: LLMClient | None = None,
) -> list[CompareCaseResult]:
    registry = build_registry(domain)
    judge_model = resolve_judge_model(config)
    judge = Judge(judge_client if judge_client is not None else client, judge_model)
    results: list[CompareCaseResult] = []

    for case in cases:
        strategy_id = case.expected_strategy
        base_recorder = RecordingClient(client)
        orch_recorder = RecordingClient(client)

        base_error = None
        base_answer = None
        base_scorecard = None

        with trace_span():
            try:
                base_answer = registry[strategy_id].process(
                    base_recorder, case.question, [], model=config.model,
                )
            except Exception as e:
                base_error = f"{type(e).__name__}: {e}"
                logger.warning("compare baseline error", case=case.id, error=base_error)

        orch_error = None
        orch_answer = None
        orch_scorecard = None

        with trace_span():
            try:
                route = RouteResult(
                    in_domain=True,
                    strategy=strategy_id,
                    intent=case.expected_intent,
                    complexity=case.expected_complexity,
                    orchestrate=True,
                )
                orch = Orchestrator(orch_recorder, config, domain)
                orch_answer = orch.run(case.question, route, config.model)
            except Exception as e:
                orch_error = f"{type(e).__name__}: {e}"
                logger.warning("compare orchestrated error", case=case.id, error=orch_error)

        if base_answer is not None:
            base_scorecard = judge.score(
                case.question, base_answer, reference=case.reference,
            )
        if orch_answer is not None:
            orch_scorecard = judge.score(
                case.question, orch_answer, reference=case.reference,
            )

        baseline = _mode_run_from_recorder(
            base_recorder.calls, base_scorecard, base_error,
        )
        baseline.answer = base_answer

        orchestrated = _mode_run_from_recorder(
            orch_recorder.calls, orch_scorecard, orch_error,
        )
        orchestrated.answer = orch_answer

        deltas = _compute_deltas(baseline, orchestrated)
        results.append(CompareCaseResult(
            case=case, baseline=baseline, orchestrated=orchestrated, **deltas,
        ))

    return results


def _compute_aggregates(results: list[CompareCaseResult]) -> dict:
    gain_values = [r.quality_gain for r in results if r.quality_gain is not None]
    n_compared = len(gain_values)
    if n_compared == 0:
        overall = {"mean_quality_gain": None, "sum_additional_tokens": 0,
                    "cost_efficiency": None, "n_compared": 0}
        return {"overall": overall, "by_intent": {}, "by_complexity": {}}

    total_gain = sum(gain_values)
    mean_gain = round(total_gain / n_compared, 2)
    total_additional = sum(r.additional_tokens for r in results if r.quality_gain is not None)
    cost_eff = round(mean_gain / total_additional, 6) if total_additional > 0 else None

    overall = {
        "mean_quality_gain": mean_gain,
        "sum_additional_tokens": total_additional,
        "cost_efficiency": cost_eff,
        "n_compared": n_compared,
    }

    by_intent: dict[str, dict] = {}
    by_complexity: dict[str, dict] = {}

    for r in results:
        if r.quality_gain is None:
            continue
        intent = r.case.expected_intent or "unknown"
        complexity = r.case.expected_complexity or "unknown"

        for bucket, key in [(by_intent, intent), (by_complexity, complexity)]:
            if key not in bucket:
                bucket[key] = {"gains": [], "tokens": 0, "n": 0}
            bucket[key]["gains"].append(r.quality_gain)
            bucket[key]["tokens"] += r.additional_tokens
            bucket[key]["n"] += 1

    def _fmt_bucket(data: dict) -> dict:
        mean_g = round(sum(data["gains"]) / data["n"], 2) if data["n"] else None
        ce = round(mean_g / data["tokens"], 6) if mean_g is not None and data["tokens"] > 0 else None
        return {"mean_quality_gain": mean_g, "sum_additional_tokens": data["tokens"],
                "cost_efficiency": ce, "n_compared": data["n"]}

    return {
        "overall": overall,
        "by_intent": {k: _fmt_bucket(v) for k, v in sorted(by_intent.items())},
        "by_complexity": {k: _fmt_bucket(v) for k, v in sorted(by_complexity.items())},
    }


def serialize_compare_result(
    results: list[CompareCaseResult],
    *,
    domain: str,
    label: str,
    model: str,
    judge_model: str | None,
) -> dict:
    aggregates = _compute_aggregates(results)

    def _mode_record(m: ModeRun) -> dict:
        return {
            "answer": m.answer,
            "scorecard": m.scorecard,
            "quality": m.quality,
            "llm_calls": m.llm_calls,
            "in_tokens": m.in_tokens,
            "out_tokens": m.out_tokens,
            "total_tokens": m.total_tokens,
            "cache_tokens": m.cache_tokens,
            "latency_ms": m.latency_ms,
            "error": m.error,
        }

    cases = []
    for r in results:
        c = r.case
        cases.append({
            "id": c.id,
            "intent": c.expected_intent,
            "complexity": c.expected_complexity,
            "strategy": c.expected_strategy,
            "baseline": _mode_record(r.baseline),
            "orchestrated": _mode_record(r.orchestrated),
            "quality_gain": r.quality_gain,
            "quality_gain_pct": r.quality_gain_pct,
            "additional_tokens": r.additional_tokens,
            "token_increase_pct": r.token_increase_pct,
            "cost_efficiency": r.cost_efficiency,
        })

    return {
        "kind": "compare",
        "label": label,
        "model": model,
        "judge_model": judge_model,
        "domain": domain,
        "n_cases": len(results),
        "n_compared": aggregates["overall"]["n_compared"],
        "cases": cases,
        "aggregates": aggregates,
    }


def _fmt_val(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) < 0.01:
            return f"{v:.6f}"
        return f"{v:.2f}"
    return str(v)


def _fmt_signed(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        sign = "+" if v >= 0 else ""
        if abs(v) < 0.01:
            return f"{sign}{v:.6f}"
        return f"{sign}{v:.2f}"
    return str(v)


def format_compare_summary(
    results: list[CompareCaseResult], label: str, model: str, judge_model: str | None,
) -> str:
    lines = []
    lines.append(f"Compare: baseline vs orchestrated ({len(results)} cases)")
    lines.append("-" * 90)
    header = (
        f"{'Case':<12} {'Intent':<20} {'Base Q':<7} {'Orch Q':<7} "
        f"{'Gain':<7} {'Gain%':<7} {'Base Tok':<9} {'Orch Tok':<9} "
        f"{'+Tok':<7} {'Tok%':<7} {'Cost Eff':<10}"
    )
    lines.append(header)
    lines.append("-" * 90)

    for r in results:
        b = r.baseline
        o = r.orchestrated
        lines.append(
            f"{r.case.id:<12} {(r.case.expected_intent or ''):<20} "
            f"{_fmt_val(b.quality):<7} {_fmt_val(o.quality):<7} "
            f"{_fmt_signed(r.quality_gain):<7} {_fmt_val(r.quality_gain_pct):<7} "
            f"{b.total_tokens:<9} {o.total_tokens:<9} "
            f"{_fmt_signed(r.additional_tokens):<7} {_fmt_val(r.token_increase_pct):<7} "
            f"{_fmt_val(r.cost_efficiency):<10}"
        )

    lines.append("-" * 90)
    agg = _compute_aggregates(results)
    oa = agg["overall"]
    base_total = sum(r.baseline.total_tokens for r in results)
    orch_total = sum(r.orchestrated.total_tokens for r in results)
    base_qualities = [r.baseline.quality for r in results if r.baseline.quality is not None]
    orch_qualities = [r.orchestrated.quality for r in results if r.orchestrated.quality is not None]
    mean_base_q = round(sum(base_qualities) / len(base_qualities), 2) if base_qualities else 0
    mean_orch_q = round(sum(orch_qualities) / len(orch_qualities), 2) if orch_qualities else 0
    total_gain_pct = round(((mean_orch_q - mean_base_q) / mean_base_q) * 100, 2) if mean_base_q else None
    total_token_pct = round((oa['sum_additional_tokens'] / base_total) * 100, 2) if base_total else None
    lines.append(
        f"{'Overall':<12} {'':<20} "
        f"{_fmt_val(mean_base_q):<7} {_fmt_val(mean_orch_q):<7} "
        f"{_fmt_signed(oa['mean_quality_gain']):<7} {_fmt_val(total_gain_pct):<7} "
        f"{base_total:<9} {orch_total:<9} "
        f"{_fmt_signed(oa['sum_additional_tokens']):<7} {_fmt_val(total_token_pct):<7} "
        f"{_fmt_val(oa['cost_efficiency']):<10}"
    )

    if agg["by_intent"]:
        lines.append("")
        lines.append("By intent:")
        for intent, data in agg["by_intent"].items():
            lines.append(
                f"  {intent:<20} {_fmt_val(data['mean_quality_gain']):<8} "
                f"gain / +{data['sum_additional_tokens']:<6} tokens / "
                f"{_fmt_val(data['cost_efficiency']):<8} eff ({data['n_compared']} cases)"
            )

    if agg["by_complexity"]:
        lines.append("")
        lines.append("By complexity:")
        for complexity, data in agg["by_complexity"].items():
            lines.append(
                f"  {complexity:<12} {_fmt_val(data['mean_quality_gain']):<8} "
                f"gain / +{data['sum_additional_tokens']:<6} tokens / "
                f"{_fmt_val(data['cost_efficiency']):<8} eff ({data['n_compared']} cases)"
            )

    return "\n".join(lines)