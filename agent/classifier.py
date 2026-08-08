from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm import LLMClient


@dataclass
class Classification:
    in_domain: bool
    reason: str


@dataclass
class IntentClassification:
    intent_id: str
    reason: str


@dataclass
class ComplexityClassification:
    level: str
    reason: str


_CLASSIFY_PROMPT = """You are a domain boundary judge. Given an expert domain, decide whether the user's question belongs to that domain.

Expert domain name: {name}
Expert domain description: {description}

Rules:
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"in_domain": true or false, "reason": "one-sentence justification"}}

User question: {question}
"""

_INTENT_PROMPT = """You are an intent judge for an expert agent in the {name} domain.

Domain description: {description}

Available intents:
{intents}

Rules:
- Choose the single intent that best matches the user's goal.
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"intent": "one of the intent ids above", "reason": "one-sentence justification"}}

User question: {question}
"""

_COMPLEXITY_PROMPT = """You are a task complexity judge for an expert agent in the {name} domain.

Domain description: {description}

Complexity levels:
- simple: answerable in a short, direct response
- medium: requires some structured explanation
- complex: large scope, multiple steps or subsystems

Rules:
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"complexity": "simple" or "medium" or "complex", "reason": "one-sentence justification"}}

User question: {question}
"""

_STRICT_REMINDER = "\nReminder: output ONLY the JSON object above and no other text."


def _build_classify_prompt(name: str, description: str, question: str) -> str:
    return _CLASSIFY_PROMPT.format(name=name, description=description, question=question)


def _classify_json(client, prompt: str, parser, *, model: str | None = None):
    for strict in (False, True):
        text = client.chat_completion(
            [{"role": "system", "content": prompt + (_STRICT_REMINDER if strict else "")}],
            model=model,
            disable_thinking=True,
        )
        parsed = parser(text)
        if parsed is not None:
            return parsed
    return None


def _parse_classification(text: str) -> Classification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("in_domain"), bool):
        return None
    reason = data.get("reason")
    return Classification(
        in_domain=data["in_domain"],
        reason=reason if isinstance(reason, str) else "",
    )


def _parse_intent(text: str, allowed: set[str]) -> IntentClassification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("intent"), str):
        return None
    if data["intent"] not in allowed:
        return None
    reason = data.get("reason")
    return IntentClassification(
        intent_id=data["intent"],
        reason=reason if isinstance(reason, str) else "",
    )


def _parse_complexity(text: str) -> ComplexityClassification | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("complexity"), str):
        return None
    if data["complexity"] not in {"simple", "medium", "complex"}:
        return None
    reason = data.get("reason")
    return ComplexityClassification(
        level=data["complexity"],
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
    prompt = _build_classify_prompt(domain_name, domain_description, question)
    result = _classify_json(client, prompt, _parse_classification, model=model)
    if result is not None:
        return result
    return Classification(
        in_domain=False, reason="Unreliable classification: classifier output could not be parsed"
    )


def classify_intent(
    client: LLMClient,
    question: str,
    domain_name: str,
    domain_description: str,
    intents: list[str],
    *,
    model: str | None = None,
) -> IntentClassification:
    prompt = _INTENT_PROMPT.format(
        name=domain_name,
        description=domain_description,
        intents="\n".join(f"- {i}" for i in intents),
        question=question,
    )
    result = _classify_json(client, prompt, lambda t: _parse_intent(t, set(intents)), model=model)
    if result is not None:
        return result
    return IntentClassification(
        intent_id="", reason="Unreliable classification: classifier output could not be parsed"
    )


def classify_complexity(
    client: LLMClient,
    question: str,
    domain_name: str,
    domain_description: str,
    *,
    model: str | None = None,
) -> ComplexityClassification:
    prompt = _COMPLEXITY_PROMPT.format(
        name=domain_name, description=domain_description, question=question
    )
    result = _classify_json(client, prompt, _parse_complexity, model=model)
    if result is not None:
        return result
    return ComplexityClassification(
        level="medium", reason="Unreliable classification: classifier output could not be parsed"
    )
