from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """You are an expert Agent in the {name} domain.

{description}

Answering requirements:
- Answer authoritatively and professionally.
- Explain the user's question from multiple angles, being thorough and insightful.
- Adjust the structure of your answer to fit each question; do not force a fixed template.
- Only answer questions within this domain.
"""


def build_system_prompt(domain_name: str, domain_description: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(name=domain_name, description=domain_description)


def build_messages(
    system_prompt: str,
    history: list[tuple[str, str]],
    question: str,
    *,
    max_turns: int = 20,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for user_text, assistant_text in history[-max_turns:]:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": question})
    return messages
