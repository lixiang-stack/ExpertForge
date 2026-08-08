from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig, DomainConfig
from .llm import LLMClient
from .processors.registry import build_registry
from .router import COMPLEX_UNSUPPORTED, Router


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
        self.history: list[tuple[str, str]] = []
        self._pending: str | None = None

    def respond(self, question: str, *, allow_clarification: bool = True) -> ChatResponse:
        route = self.router.route(question)
        if not route.in_domain:
            text = self.domain.out_of_domain_reply
            if route.reject_reason:
                text += f" ({route.reject_reason})"
            return ChatResponse(kind="reject", text=text)
        if route.strategy == COMPLEX_UNSUPPORTED:
            return ChatResponse(
                kind="unsupported", text=self.domain.prompts["unsupported_complex"]
            )
        if route.needs_clarification and allow_clarification:
            self._pending = question
            return ChatResponse(
                kind="clarification", text=self._ask_clarification(question, route)
            )
        processor = self.processors.get(route.strategy)
        if processor is None:
            return ChatResponse(kind="error", text=f"No processor for strategy '{route.strategy}'")
        model = self.config.model
        strategy_def = self.domain.strategies.get(route.strategy)
        if strategy_def and strategy_def.model:
            model = strategy_def.model
        answer = processor.process(self.client, question, self.history, model=model)
        self.history.append((question, answer))
        return ChatResponse(kind="answer", text=answer)

    def answer_clarification(self, supplementary: str) -> ChatResponse:
        pending = self._pending
        self._pending = None
        if pending is None:
            return ChatResponse(kind="answer", text="")
        merged = pending + "\n\nAdditional context: " + supplementary
        return self.respond(merged, allow_clarification=False)

    def _ask_clarification(self, question: str, route) -> str:
        prompt = self.domain.prompts["clarify"].format(
            question=question,
            intent=route.intent or "unknown",
            complexity=route.complexity or "unknown",
        )
        return self.client.chat_completion(
            [{"role": "system", "content": prompt}],
            model=self.config.classifier_model,
            disable_thinking=True,
        )
