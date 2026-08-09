from __future__ import annotations

from .chat import Chat
from .config import AgentConfig, DomainConfig
from .llm import LLMClient, LLMError


def run_repl(client: LLMClient, config: AgentConfig, domain: DomainConfig) -> None:
    chat = Chat(client, config, domain)
    print(f"ExpertForge | Domain: {domain.name} | Type exit or quit to leave")

    while True:
        try:
            question = input("you > ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        try:
            response = chat.respond(question)
            print("expert > " + response.text)
        except LLMError as e:
            print(f"[error] {e}")
