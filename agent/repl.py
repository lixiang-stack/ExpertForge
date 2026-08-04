from __future__ import annotations

from .classifier import classify_question
from .config import AgentConfig
from .generator import build_messages, build_system_prompt
from .llm import LLMClient, LLMError


def run_repl(client: LLMClient, config: AgentConfig) -> None:
    system_prompt = build_system_prompt(config.domain_name, config.domain_description)
    history: list[tuple[str, str]] = []
    print(f"ExpertForge | Domain: {config.domain_name} | Type exit or quit to leave")

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
            result = classify_question(
                client,
                question,
                config.domain_name,
                config.domain_description,
                model=config.classifier_model,
            )
        except LLMError as e:
            print(f"[error] {e}")
            continue

        if not result.in_domain:
            print(config.out_of_domain_reply)
            if result.reason:
                print(f"({result.reason})")
            continue

        messages = build_messages(system_prompt, history, question)
        print("expert > ", end="", flush=True)
        answer_parts: list[str] = []
        try:
            for chunk in client.chat_completion_stream(messages, model=config.model):
                print(chunk, end="", flush=True)
                answer_parts.append(chunk)
            print()
            history.append((question, "".join(answer_parts)))
        except LLMError as e:
            print()
            print(f"[error] {e}")
