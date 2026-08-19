from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig, DomainConfig
from .llm import LLMClient
from .loggers import get_logger
from .model_router import resolve_model
from .orchestrator import Orchestrator
from .strategy import build_registry
from .router import RouteResult, Router

logger = get_logger("chat")


@dataclass
class ChatResponse:
    kind: str
    text: str


class Chat:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain
        self.router = Router(client, config, domain)
        self.processors = build_registry(domain)
        self.orchestrator = Orchestrator(client, config, domain)
        self.history: list[tuple[str, str]] = []

    def respond(self, question: str, *, route: RouteResult | None = None) -> ChatResponse:
        if route is None:
            route = self.router.route(question)
        if not route.in_domain:
            text = self.domain.out_of_domain_reply
            if route.reject_reason:
                text += f" ({route.reject_reason})"
            return ChatResponse(kind="reject", text=text)
        processor = self.processors[route.strategy]
        model = resolve_model(self.config, route, self.config.model)
        try:
            if route.orchestrate:
                answer = self.orchestrator.run(question, route, model)
            else:
                answer = processor.process(self.client, question, self.history, model=model)
        except Exception:
            logger.exception(
                "answer generation failed", strategy=route.strategy, model=model
            )
            raise
        self.history.append((question, answer))
        logger.info("answer generated", strategy=route.strategy, model=model)
        return ChatResponse(kind="answer", text=answer)
