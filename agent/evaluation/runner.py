from __future__ import annotations

import time
from dataclasses import dataclass

from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig
from agent.llm import LLMClient
from agent.model_router import resolve_model
from agent.router import Router

from .dataset import EvalCase, Suite
from .judge import Judge


class RecordingClient:
    """Thin LLMClient wrapper that records per-call usage and latency.

    Reads the same thread-local `_usage_local` that `LLMClient` populates;
    completely independent of observability.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return self._inner.model

    def reset(self) -> None:
        self.calls = []

    def _usage(self):
        usage = getattr(self._inner, "_usage_local", None)
        return getattr(usage, "usage", None)

    def chat_completion(self, messages, *, model=None, temperature=0.3, **kwargs) -> str:
        started = time.perf_counter()
        text = self._inner.chat_completion(
            messages, model=model, temperature=temperature, **kwargs
        )
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        u = self._usage()
        usage_local = getattr(self._inner, "_usage_local", None)
        self.calls.append({
            "model": model or self._inner.model,
            "prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) if u else 0,
            "total_tokens": getattr(u, "total_tokens", 0) if u else 0,
            "cache_tokens": getattr(usage_local, "cache_tokens", 0) if usage_local else 0,
            "latency_ms": elapsed,
        })
        return text

    def chat_completion_stream(self, messages, *, model=None, temperature=0.7, **kwargs):
        yield from self._inner.chat_completion_stream(
            messages, model=model, temperature=temperature, **kwargs
        )


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
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0
    latency_ms: float = 0.0


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
    *,
    skip_quality: bool = False,
) -> list[CaseResult]:
    """Run every dataset case through the real pipeline and record per-case results.

    Flow per case:
      1. Reset the recorder so each case's costs are isolated.
      2. Build a fresh ``Chat`` (no history leaks between cases).
      3. ``Router.route`` -- classification + strategy selection. This is always
         the first LLM call of the case, even when quality scoring is skipped.
      4. ``resolve_model`` -- the *expected* model for this route, the
         ground-truth comparison target.
      5. Quality phase (only if ``case.answer_quality`` and not ``skip_quality``):
         ``chat.respond`` runs the full pipeline, then the judge scores it.
      6. Aggregate all recorded calls; append a ``CaseResult``.

    Cost accounting notes:
      - ``recorder.calls`` accumulates *all* LLM calls since ``reset()``: the
        router's classification call, the answer-pipeline calls, and the judge
        call are summed into one case's token totals (intentional: full
        pipeline cost per case, which is why ``llm_calls`` is typically >= 3).
      - ``skip_quality`` still runs the router, so it does not reduce the case
        to zero LLM calls.
    """
    recorder = RecordingClient(client)
    router = Router(recorder, config, domain)
    judge = Judge(recorder,
                  config.evaluation.judge_model if config.evaluation else config.model)
    results: list[CaseResult] = []
    for case in suite.cases:
        recorder.reset()
        chat = Chat(recorder, config, domain)  # fresh history per case
        route = router.route(case.question)
        expected_model = resolve_model(config, domain, route, config.model)
        answer = None
        scorecard = None
        actual_model = None
        if case.answer_quality and not skip_quality:
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
        costs = _sum_calls(recorder.calls)
        results.append(CaseResult(
            case=case,
            in_domain=route.in_domain,
            intent=route.intent,
            complexity=route.complexity,
            strategy=route.strategy,
            orchestrate=route.orchestrate,
            answer=answer,
            actual_model=actual_model,
            expected_model=expected_model,
            scorecard=scorecard,
            suite=suite.name,
            **costs,
        ))
    return results
