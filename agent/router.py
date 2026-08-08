from __future__ import annotations

from dataclasses import dataclass

from .classifier import classify_complexity, classify_intent, classify_question
from .config import AgentConfig, DomainConfig
from .llm import LLMClient

DEFAULT_STRATEGY = "direct"
COMPLEX_UNSUPPORTED = "complex_unsupported"


@dataclass
class RouteResult:
    in_domain: bool
    strategy: str
    intent: str | None = None
    complexity: str | None = None
    needs_clarification: bool = False
    reject_reason: str = ""


class Router:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain

    def route(self, question: str) -> RouteResult:
        domain_result = classify_question(
            self.client,
            question,
            self.domain.name,
            self.domain.description,
            model=self.config.classifier_model,
        )
        if not domain_result.in_domain:
            return RouteResult(
                in_domain=False, strategy="reject", reject_reason=domain_result.reason
            )

        intent_result = classify_intent(
            self.client,
            question,
            self.domain.name,
            self.domain.description,
            list(self.domain.intents),
            model=self.config.classifier_model,
        )
        complexity_result = classify_complexity(
            self.client,
            question,
            self.domain.name,
            self.domain.description,
            model=self.config.classifier_model,
        )

        strategy = self.domain.intent_mapping.get(intent_result.intent_id, DEFAULT_STRATEGY)
        strategy_def = self.domain.strategies.get(strategy)
        if strategy_def and strategy_def.complexity_gate and complexity_result.level == "complex":
            strategy = COMPLEX_UNSUPPORTED

        needs_clarification = False
        if strategy != COMPLEX_UNSUPPORTED:
            intent_def = self.domain.intents.get(intent_result.intent_id)
            needs_clarification = bool(intent_def and intent_def.needs_clarification)

        return RouteResult(
            in_domain=True,
            strategy=strategy,
            intent=intent_result.intent_id or None,
            complexity=complexity_result.level,
            needs_clarification=needs_clarification,
        )
