from __future__ import annotations


class Processor:
    strategy_id = "base"

    def __init__(self, prompt_template: str, domain_name: str, domain_description: str):
        self.prompt_template = prompt_template
        self.domain_name = domain_name
        self.domain_description = domain_description

    @property
    def structure(self) -> str:
        return ""

    def build_system_prompt(self) -> str:
        template = self.prompt_template.replace("{structure}", self.structure)
        return template.format(name=self.domain_name, description=self.domain_description)

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
