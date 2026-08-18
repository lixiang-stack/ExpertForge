from __future__ import annotations

from agent.llm import LLMClient, LLMError
from agent.parsing import parse_json

JUDGE_DIMENSIONS = (
    "correctness",
    "relevance",
    "completeness",
    "technical_depth",
    "practical_usefulness",
    "hallucination",
)

_JUDGE_PROMPT = """You are a strict evaluator of technical answers.

Agent answer:
{answer}
{reference_block}
Score the answer on each dimension from 1 (worst) to 5 (best):
- correctness: factual accuracy and technical truth
- relevance: how directly it addresses the question
- completeness: whether all important aspects are covered
- technical_depth: depth and sophistication of the explanation
- practical_usefulness: how actionable and useful the answer is
- hallucination: 1 = many unsupported claims, 5 = no unsupported claims

Output ONLY a single JSON object:
{{"correctness": 1-5, "relevance": 1-5, "completeness": 1-5, "technical_depth": 1-5, "practical_usefulness": 1-5, "hallucination": 1-5}}
"""


def build_judge_prompt(answer: str, *, reference: str | None = None) -> str:
    reference_block = (
        f"\nGround truth reference:\n{reference}" if reference else "\nNo reference provided."
    )
    return _JUDGE_PROMPT.format(
        answer=answer,
        reference_block=reference_block,
    )


def parse_scorecard(text: str | None) -> dict | None:
    data = parse_json(text) if text else None
    if data is None:
        return None
    for dim in JUDGE_DIMENSIONS:
        value = data.get(dim)
        if not isinstance(value, int) or not 1 <= value <= 5:
            return None
    return data


class Judge:
    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def score(self, question: str, answer: str, *, reference: str | None = None) -> dict | None:
        prompt = build_judge_prompt(answer, reference=reference)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        try:
            result = self.client.chat_completion(
                messages,
                model=self.model,
                disable_thinking=True,
                json_mode=True,
            )
        except LLMError:
            return None
        return parse_scorecard(result.text)
