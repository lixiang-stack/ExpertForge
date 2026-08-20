from __future__ import annotations

import time
from dataclasses import dataclass

from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig, resolve_judge_model
from agent.llm import ChatResult, LLMClient
from agent.loggers import get_logger
from agent.model_router import resolve_model
from agent.router import Router

from .dataset import EvalCase, FULL_EXPERT, Suite
from .judge import Judge

logger = get_logger("evaluation")


class RecordingClient:
    """Thin LLMClient wrapper that records per-call usage and latency.

    Reads token usage from the returned ChatResult; completely independent
    of observability.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.calls: list[dict] = []

    def reset(self) -> None:
        self.calls = []

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> ChatResult:
        started = time.perf_counter()
        result = self._inner.chat_completion(
            messages, model=model, temperature=temperature, **kwargs
        )
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        self.calls.append({
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "cache_tokens": result.cache_tokens,
            "latency_ms": elapsed,
        })
        return result


@dataclass
class CaseResult:
    case: EvalCase
    in_domain: bool
    intent: str | None
    complexity: str | None
    strategy: str
    orchestrate: bool
    answer: str | None
    actual_model: str | None
    expected_model: str | None
    scorecard: dict | None
    suite: str = ""
    tier: str = ""
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


def _sum_calls(calls: list[dict]) -> dict:
    return {
        "llm_calls": len(calls),
        "in_tokens": sum(c["prompt_tokens"] for c in calls),
        "out_tokens": sum(c["completion_tokens"] for c in calls),
        "total_tokens": sum(c["total_tokens"] for c in calls),
        "cache_tokens": sum(c["cache_tokens"] for c in calls),
        "latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
    }


def run_evaluation(
    config: AgentConfig,
    domain: DomainConfig,
    suite: Suite,
    client: LLMClient,
    judge_client: LLMClient | None = None,
) -> list[CaseResult]:
    """Run every dataset case through the real pipeline and record per-case results.

    Flow per case:
      1. Reset the recorder so each case's costs are isolated.
      2. Build a fresh ``Chat`` (no history leaks between cases).
      3. ``Router.route`` -- classification + strategy selection. This is always
         the first LLM call of the case.
      4. ``resolve_model`` -- the *expected* model for this route, the
         ground-truth comparison target.
      5. Quality phase (only if ``case.tier == "full_expert"``):
         ``chat.respond`` runs the full pipeline, then the judge scores it.
      6. Aggregate all recorded calls; append a ``CaseResult``.

    Cost accounting notes:
      - ``recorder.calls`` accumulates *all* LLM calls since ``reset()``: the
        router's classification call, the answer-pipeline calls, and the judge
        call are summed into one case's token totals (intentional: full
        pipeline cost per case, which is why ``llm_calls`` is typically >= 3).
      - When ``judge_client`` is given, the judge's calls are recorded on a
        separate recorder (``judge_recorder``) but still summed into the same
        per-case totals.
      - The router's classification call is always made; the answer-pipeline
        and judge calls happen only for ``full_expert`` tier cases.
    """
    recorder = RecordingClient(client)
    judge_recorder = RecordingClient(judge_client) if judge_client is not None else None
    router = Router(recorder, config, domain)
    judge = Judge(judge_recorder if judge_recorder is not None else recorder,
                  resolve_judge_model(config))
    results: list[CaseResult] = []
    for case in suite.cases:
        recorder.reset()
        if judge_recorder is not None:
            judge_recorder.reset()
        route = None
        expected_model = None
        answer = None
        scorecard = None
        actual_model = None
        error = None
        try:
            chat = Chat(recorder, config, domain)  # fresh history per case
            route = router.route(case.question)
            expected_model = resolve_model(config, route, config.model)
            if case.tier == FULL_EXPERT:
                resp = chat.respond(case.question, route=route)
                answer = resp.text
                # actual_model is captured from the last recorded LLM call AFTER
                # `chat.respond` but BEFORE the judge runs, so it reflects the model
                # that produced the answer, not the judge's model. When the route is
                # out-of-domain `chat.respond` returns the reject reply without making
                # an LLM call, so `calls[-1]` would be the router's classification
                # call instead -- not the answer model. Guard on `route.in_domain` so
                # actual_model stays None in that case.
                actual_model = (recorder.calls[-1]["model"] if route.in_domain and recorder.calls
                                else None)
                scorecard = judge.score(case.question, answer, reference=case.reference)
        except Exception as e:  # noqa: BLE001 -- one failing case must not abort the run
            error = f"{type(e).__name__}: {e}"
            logger.warning("eval case error", case=case.id, error=error)
        calls = recorder.calls + (judge_recorder.calls if judge_recorder is not None else [])
        costs = _sum_calls(calls)
        results.append(CaseResult(
            case=case,
            in_domain=bool(route and route.in_domain),
            intent=route.intent if route else None,
            complexity=route.complexity if route else None,
            strategy=route.strategy if route else None,
            orchestrate=bool(route and route.orchestrate),
            answer=answer,
            actual_model=actual_model,
            expected_model=expected_model,
            scorecard=scorecard,
            suite=suite.name,
            tier=case.tier,
            error=error,
            **costs,
        ))
    return results
