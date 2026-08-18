from __future__ import annotations

from dataclasses import dataclass

from .classification import ClassificationService
from .config import COMPLEXITY_LEVELS, AgentConfig, DomainConfig
from .llm import LLMClient


_COMPLEXITY_RANK = {level: i for i, level in enumerate(COMPLEXITY_LEVELS)}


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
        if not intent_id or intent_id not in self.domain.intent_mapping:
            return RouteResult(
                in_domain=False, strategy="reject",
                reject_reason=f"Unknown intent: {intent_id}",
            )
        strategy = self.domain.intent_mapping[intent_id]
        orchestrate = False
        policy = self.domain.orchestration
        if policy is not None:
            orchestrate = (
                policy.enabled
                and _COMPLEXITY_RANK.get(result.complexity, -1)
                >= _COMPLEXITY_RANK.get(policy.min_complexity, 0)
                and result.intent in policy.intents
            )
        return RouteResult(
            in_domain=True,
            strategy=strategy,
            intent=intent_id,
            complexity=result.complexity,
            orchestrate=orchestrate,
        )