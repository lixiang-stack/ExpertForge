from __future__ import annotations

from dataclasses import dataclass

from .config import COMPLEXITY_LEVELS, ComplexityPolicy, DomainConfig, IntentDef
from .llm import LLMClient
from .parsing import parse_json


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
- Also judge task complexity as one of:
  {complexity_section}
- Output ONLY a single JSON object and nothing else.
- JSON format: {{"in_domain": true|false, "intent": "<intent id or null>", "complexity": "<simple|medium|complex or null>", "reason": "one-sentence justification"}}
"""


def build_complexity_section(policy: ComplexityPolicy | None) -> str:
    if policy is None:
        return (
            "simple (short direct answer), medium (needs structured explanation), "
            "complex (large scope, multiple steps or subsystems)"
        )
    blocks: list[str] = []
    for level in policy.levels:
        lines = [f"- {level.level}: {level.description}"]
        for dim in level.dimensions:
            lines.append(f"  {dim}")
        if level.positive_examples:
            lines.append("  Positive examples:")
            lines.extend(f"    - {ex}" for ex in level.positive_examples)
        if level.negative_examples:
            lines.append("  Negative examples:")
            lines.extend(f"    - {ex}" for ex in level.negative_examples)
        for b in level.boundaries:
            lines.append(f"  Boundary: {b}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_classification_prompt(
    name: str,
    description: str,
    intents: list[IntentDef],
    complexity: ComplexityPolicy | None = None,
) -> str:
    lines: list[str] = []
    for idef in intents:
        header = f"- {idef.id}: {idef.description}"
        if not (idef.positive_examples or idef.negative_examples or idef.boundaries):
            lines.append(header)
            continue
        lines.append(header)
        if idef.positive_examples:
            lines.append("  Positive examples:")
            lines.extend(f"    - {ex}" for ex in idef.positive_examples)
        if idef.negative_examples:
            lines.append("  Negative examples:")
            lines.extend(f"    - {ex}" for ex in idef.negative_examples)
        for b in idef.boundaries:
            lines.append(f"  Boundary: {b}")
    intents_block = "\n".join(lines)
    return _CLASSIFICATION_PROMPT.format(
        name=name,
        description=description,
        intents=intents_block,
        complexity_section=build_complexity_section(complexity),
    )


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


class ClassificationService:
    def __init__(self, client: LLMClient, domain: DomainConfig):
        self.client = client
        self.domain = domain

    def classify(self, question: str, *, model: str | None = None) -> ClassificationResult:
        intent_ids = list(self.domain.intents)
        schema = build_classification_schema(intent_ids)
        prompt = build_classification_prompt(
            self.domain.name, self.domain.description,
            list(self.domain.intents.values()),
            complexity=self.domain.complexity,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        result = self.client.chat_completion(
            messages,
            model=model,
            disable_thinking=True,
            json_schema=schema,
        )
        return validate_classification(parse_json(result.text), intent_ids)