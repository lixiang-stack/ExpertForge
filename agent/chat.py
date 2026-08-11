from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig, DomainConfig
from .llm import LLMClient
from .model_router import resolve_model
from .orchestrator import Orchestrator
from .strategy import build_registry
from .router import Router


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

    def respond(self, question: str) -> ChatResponse:
        route = self.router.route(question)
        if not route.in_domain:
            text = self.domain.out_of_domain_reply
            if route.reject_reason:
                text += f" ({route.reject_reason})"
            return ChatResponse(kind="reject", text=text)
        processor = self.processors[route.strategy]
        model = resolve_model(self.config, self.domain, route, self.config.model)
        if route.orchestrate:
            answer = self.orchestrator.run(question, route, model)
        else:
            answer = processor.process(self.client, question, self.history, model=model)
        self.history.append((question, answer))
        return ChatResponse(kind="answer", text=answer)
