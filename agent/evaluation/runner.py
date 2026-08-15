from __future__ import annotations

import time
from dataclasses import dataclass

from agent.chat import Chat
from agent.config import AgentConfig, DomainConfig
from agent.llm import ChatResult, LLMClient
from agent.model_router import resolve_model
from agent.router import Router

from .dataset import EvalCase, Dataset
from .judge import Judge


class RecordingClient:
    """Thin LLMClient wrapper that records per-call usage and latency.

    Reads token usage from the returned ChatResult; completely independent
    of observability.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return self._inner.model

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
    dataset: Dataset,
    client: LLMClient,
    *,
    skip_quality: bool = False,
) -> list[CaseResult]:
    recorder = RecordingClient(client)
    router = Router(recorder, config, domain)
    judge = Judge(recorder,
                  config.evaluation.judge_model if config.evaluation else config.model)
    results: list[CaseResult] = []
    for case in dataset.cases:
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
            actual_model = recorder.calls[-1]["model"] if recorder.calls else None
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
            **costs,
        ))
    return results
