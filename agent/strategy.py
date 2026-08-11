from __future__ import annotations

from .config import DomainConfig


class Strategy:
    def __init__(self, strategy_id: str, prompt_template: str):
        self.strategy_id = strategy_id
        self.prompt_template = prompt_template

    def build_system_prompt(self) -> str:
        return self.prompt_template

    def build_messages(
        self,
        history: list[tuple[str, str]],
        question: str,
        *,
        max_turns: int = 20,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.build_system_prompt()}]
        for user_text, assistant_text in history[-max_turns:]:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": question})
        return messages

    def process(self, client, question: str, history: list[tuple[str, str]], *, model: str | None = None) -> str:
        return client.chat_completion(self.build_messages(history, question), model=model)


def build_registry(domain: DomainConfig) -> dict[str, Strategy]:
    return {
        sid: Strategy(sid, domain.prompts[sid])
        for sid in domain.strategies
    }
