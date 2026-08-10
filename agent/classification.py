from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import DomainConfig
from .llm import LLMClient, LLMError

COMPLEXITY_LEVELS = ("simple", "medium", "complex")


@dataclass
class ClassificationResult:
    in_domain: bool
    intent: str | None
    complexity: str | None
    reason: str


def build_classification_schema(intent_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "in_domain": {"type": "boolean"},
            "intent": {"type": ["string", "null"], "enum": intent_ids + [None]},
            "complexity": {
                "type": ["string", "null"],
                "enum": list(COMPLEXITY_LEVELS) + [None],
            },
            "reason": {"type": "string"},
        },
        "required": ["in_domain", "intent", "complexity", "reason"],
    }


_CLASSIFICATION_PROMPT = """You are a domain and task classifier for an expert domain named {name}.

Domain description: {description}

Available intents:
{intents}

Rules:
- Decide whether the question belongs to the domain above.
- If in_domain is false, set intent and complexity to null.
- If in_domain is true, choose the single intent that best matches the user's goal
  from the listed intents. Do not invent a new intent.
- Also judge task complexity as one of {complexity_levels}:
  simple (short direct answer), medium (needs structured explanation),
  complex (large scope, multiple steps or subsystems).
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"in_domain": true|false, "intent": "<intent id or null>", "complexity": "<simple|medium|complex or null>", "reason": "one-sentence justification"}}

User question: {question}
"""


def build_classification_prompt(
    name: str,
    description: str,
    intent_items: list[tuple[str, str]],
    question: str,
) -> str:
    intents = "\n".join(f"- {iid}: {desc}" for iid, desc in intent_items)
    return _CLASSIFICATION_PROMPT.format(
        name=name,
        description=description,
        intents=intents,
        complexity_levels=", ".join(COMPLEXITY_LEVELS),
        question=question,
    )


def _parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate_classification(data: dict | None, intent_ids: list[str]) -> ClassificationResult:
    if not data or not isinstance(data.get("in_domain"), bool):
        return ClassificationResult(
            in_domain=False,
            intent=None,
            complexity=None,
            reason="Unreliable classification: classifier output could not be parsed",
        )
    in_domain = data["in_domain"]
    intent = data.get("intent")
    complexity = data.get("complexity")
    if in_domain:
        if intent not in intent_ids:
            intent = None
        if complexity not in COMPLEXITY_LEVELS:
            complexity = "medium"
    else:
        # Out of domain: intent/complexity are meaningless downstream; leave as-is.
        pass
    reason = data.get("reason")
    return ClassificationResult(
        in_domain=in_domain,
        intent=intent if isinstance(intent, str) else None,
        complexity=complexity if isinstance(complexity, str) else None,
        reason=reason if isinstance(reason, str) else "",
    )


_DEGRADED_INSTRUCTION = """

Answer in JSON only, using exactly this structure:
{
  "in_domain": true or false,
  "intent": "one of the available intent ids, or null when out of domain",
  "complexity": "simple" | "medium" | "complex" or null when out of domain,
  "reason": "one-sentence justification"
}
"""


class ClassificationService:
    def __init__(self, client: LLMClient, domain: DomainConfig):
        self.client = client
        self.domain = domain

    def classify(self, question: str, *, model: str | None = None) -> ClassificationResult:
        intent_ids = list(self.domain.intents)
        intent_items = [
            (iid, idef.description) for iid, idef in self.domain.intents.items()
        ]
        schema = build_classification_schema(intent_ids)
        prompt = build_classification_prompt(
            self.domain.name, self.domain.description, intent_items, question
        )
        messages = [{"role": "system", "content": prompt}]
        # TODO: prefer json_schema structured output once the model/provider supports it.
        # The current provider rejects response_format=json_schema, so json_object
        # (json_mode) is the main path for now. The json_schema path below is kept
        # (commented out) for when a provider with json_schema support is used.
        #
        # try:
        #     text = self.client.chat_completion(
        #         messages,
        #         model=model,
        #         disable_thinking=True,
        #         json_schema=schema,
        #     )
        # except LLMError:
        #     # Provider rejected json_schema (capability issue): degrade once.
        #     degraded_messages = [
        #         {"role": "system", "content": prompt + _DEGRADED_INSTRUCTION}
        #     ]
        #     text = self.client.chat_completion(
        #         degraded_messages,
        #         model=model,
        #         disable_thinking=True,
        #         json_mode=True,
        #     )
        text = self.client.chat_completion(
            messages,
            model=model,
            disable_thinking=True,
            json_mode=True,
        )
        return validate_classification(_parse(text), intent_ids)