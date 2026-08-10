from __future__ import annotations

from dataclasses import dataclass

from .classification import ClassificationService
from .config import AgentConfig, DomainConfig
from .llm import LLMClient

DEFAULT_STRATEGY = "direct"


@dataclass
class RouteResult:
    in_domain: bool
    strategy: str
    intent: str | None = None
    complexity: str | None = None
    reject_reason: str = ""
    orchestrate: bool = False


class Router:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain
        self.classifier = ClassificationService(client, domain)

    def route(self, question: str) -> RouteResult:
        result = self.classifier.classify(question, model=self.config.classifier_model)
        if not result.in_domain:
            return RouteResult(
                in_domain=False, strategy="reject", reject_reason=result.reason
            )
        intent_id = result.intent
        strategy = self.domain.intent_mapping.get(intent_id, DEFAULT_STRATEGY)
        orchestrate = False
        strategy_def = self.domain.strategies.get(strategy)
        if strategy_def and strategy_def.complexity_gate and result.complexity == "complex":
            orchestrate = True
        return RouteResult(
            in_domain=True,
            strategy=strategy,
            intent=intent_id,
            complexity=result.complexity,
            orchestrate=orchestrate,
        )