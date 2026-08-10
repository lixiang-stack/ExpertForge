from __future__ import annotations

import json
import re

from .config import AgentConfig, DomainConfig
from .llm import LLMClient, LLMError
from .processors.registry import build_registry
from .router import RouteResult


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _planner_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["title", "instruction"],
                },
            }
        },
        "required": ["tasks"],
    }


_PLANNER_PROMPT = """You are a planning agent for an expert domain named {name}.

{description}

Task context:
{context}

Rules:
- Decompose the user's complex task into 2-4 focused sub-tasks.
- Each sub-task must be answerable by a single standalone LLM call.
- Output ONLY a single JSON object: {{"tasks": [{{"title": "...", "instruction": "..."}}]}}

User question: {question}
"""

_PLANNER_DEGRADED_INSTRUCTION = """

Answer in JSON only, using exactly this structure:
{
  "tasks": [
    {"title": "<short sub-task title>", "instruction": "<standalone sub-task instruction>"}
  ]
}
"""


class Orchestrator:
    def __init__(self, client: LLMClient, config: AgentConfig, domain: DomainConfig):
        self.client = client
        self.config = config
        self.domain = domain
        self._processors = build_registry(domain)

    def _strategy_context(self, strategy: str) -> str:
        proc = self._processors[strategy]
        return proc.build_system_prompt()

    def run(self, question: str, route: RouteResult, model: str) -> str:
        context = self._strategy_context(route.strategy)
        tasks = self._plan(question, route.strategy, context, model)
        # TODO: add Evaluator / Optimizer phases after aggregation (future)
        if tasks is None:
            return self._direct_answer(question, route.strategy, context, model)
        outputs = [
            self._worker(question, task, context, model)
            for task in tasks  # TODO: parallelize worker execution (future)
        ]
        return self._aggregate(question, route.strategy, context, tasks, outputs, model)

    def _plan(
        self, question: str, strategy: str, context: str, model: str
    ) -> list[tuple[str, str]] | None:
        prompt = _PLANNER_PROMPT.format(
            name=self.domain.name,
            description=self.domain.description,
            context=context,
            question=question,
        )
        messages = [{"role": "system", "content": prompt}]
        try:
            text = self.client.chat_completion(
                messages, model=model, disable_thinking=True, json_schema=_planner_schema()
            )
        except LLMError:
            # Provider rejected json_schema (capability issue): degrade once.
            degraded_messages = [
                {"role": "system", "content": prompt + _PLANNER_DEGRADED_INSTRUCTION}
            ]
            text = self.client.chat_completion(
                degraded_messages, model=model, disable_thinking=True, json_mode=True
            )
        data = _parse_json(text)
        if not data or not isinstance(data.get("tasks"), list):
            return None
        tasks: list[tuple[str, str]] = []
        for item in data["tasks"]:
            if not isinstance(item, dict):
                return None
            title = item.get("title")
            instruction = item.get("instruction")
            if not isinstance(title, str) or not isinstance(instruction, str):
                return None
            tasks.append((title, instruction))
        return tasks or None

    def _worker(self, question: str, task: tuple[str, str], context: str, model: str) -> str:
        _title, instruction = task
        messages = [
            {
                "role": "system",
                "content": f"{context}\n\nSub-task: {instruction}",
            },
            {"role": "user", "content": question},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True)

    def _aggregate(
        self,
        question: str,
        strategy: str,
        context: str,
        tasks: list[tuple[str, str]],
        outputs: list[str],
        model: str,
    ) -> str:
        sections = []
        for (title, _instruction), output in zip(tasks, outputs):
            sections.append(f"Sub-task: {title}\n{output}")
        user_content = (
            f"User question: {question}\n\n"
            f"Sub-task results:\n\n" + "\n\n".join(sections)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{context}\n\n"
                    "You are synthesizing sub-task results into one coherent final "
                    "answer to the user's original question."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True)

    def _direct_answer(self, question: str, strategy: str, context: str, model: str) -> str:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True)
