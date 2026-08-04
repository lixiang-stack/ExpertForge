from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm import LLMClient


@dataclass
class Classification:
    in_domain: bool
    reason: str


_CLASSIFY_PROMPT = """You are a domain boundary judge. Given an expert domain, decide whether the user's question belongs to that domain.

Expert domain name: {name}
Expert domain description: {description}

Rules:
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"in_domain": true or false, "reason": "one-sentence justification"}}

User question: {question}
"""


def _build_prompt(name: str, description: str, question: str, strict: bool = False) -> str:
    prompt = _CLASSIFY_PROMPT.format(name=name, description=description, question=question)
    if strict:
        prompt += "\nReminder: output ONLY the JSON object above and no other text."
    return prompt


def _parse_classification(text: str) -> Classification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "in_domain" not in data:
        return None
    if not isinstance(data["in_domain"], bool):
        return None
    reason = data.get("reason")
    return Classification(
        in_domain=data["in_domain"],
        reason=reason if isinstance(reason, str) else "",
    )


def classify_question(
    client: LLMClient,
    question: str,
    domain_name: str,
    domain_description: str,
    *,
    model: str | None = None,
) -> Classification:
    for strict in (False, True):
        prompt = _build_prompt(domain_name, domain_description, question, strict=strict)
        text = client.chat_completion(
            [{"role": "system", "content": prompt}], model=model, disable_thinking=True
        )
        result = _parse_classification(text)
        if result is not None:
            return result
    return Classification(
        in_domain=False, reason="Unreliable classification: classifier output could not be parsed"
    )
