from __future__ import annotations

import json
import re

from .config import AgentConfig, DomainConfig, OrchestratorConfig
from .llm import LLMClient, LLMError
from .strategy import build_registry
from .router import RouteResult
from .worker_pool import WorkerResult, WorkerTask, run_workers


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
                        "role": {"type": "string"},
                    },
                    "required": ["title", "instruction", "role"],
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
- Assign each sub-task a distinct analysis role (e.g. Architecture, Scalability,
  Reliability / Failure Modes, Operations) that defines its focused responsibility.
- Output ONLY a single JSON object: {{"tasks": [{{"title": "...", "instruction": "...", "role": "..."}}]}}

User question: {question}
"""

_PLANNER_DEGRADED_INSTRUCTION = """

Answer in JSON only, using exactly this structure:
{
  "tasks": [
    {"title": "<short sub-task title>", "instruction": "<standalone sub-task instruction>", "role": "<analysis role>"}
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
        oc = self.config.orchestrator or OrchestratorConfig()
        results = run_workers(
            tasks,
            lambda task: self._worker(question, task, context, model),
            max_workers=oc.max_workers,
            timeout=oc.worker_timeout,
        )
        if all(r.error for r in results):
            return self._direct_answer(question, route.strategy, context, model)
        return self._aggregate(question, route.strategy, context, results, model)

    def _plan(
        self, question: str, strategy: str, context: str, model: str
    ) -> list[WorkerTask] | None:
        prompt = _PLANNER_PROMPT.format(
            name=self.domain.name,
            description=self.domain.description,
            context=context,
            question=question,
        )
        messages = [{"role": "system", "content": prompt}]
        # TODO: prefer json_schema structured output once the model/provider supports it.
        # The current provider rejects response_format=json_schema, so json_object
        # (json_mode) is the main path for now. The json_schema path below is kept
        # (commented out) for when a provider with json_schema support is used.
        #
        # try:
        #     text = self.client.chat_completion(
        #         messages, model=model, disable_thinking=True, json_schema=_planner_schema()
        #     )
        # except LLMError:
        #     # Provider rejected json_schema (capability issue): degrade once.
        #     degraded_messages = [
        #         {"role": "system", "content": prompt + _PLANNER_DEGRADED_INSTRUCTION}
        #     ]
        #     text = self.client.chat_completion(
        #         degraded_messages, model=model, disable_thinking=True, json_mode=True
        #     )
        result = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_mode=True
        )
        data = _parse_json(result.text)
        if not data or not isinstance(data.get("tasks"), list):
            return None
        tasks: list[WorkerTask] = []
        for item in data["tasks"]:
            if not isinstance(item, dict):
                return None
            title = item.get("title")
            instruction = item.get("instruction")
            if not isinstance(title, str) or not isinstance(instruction, str):
                return None
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = title
            tasks.append(WorkerTask(title=title, instruction=instruction, role=role))
        return tasks or None

    def _worker(self, question: str, task: WorkerTask, context: str, model: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"{context}\n\nRole: {task.role}\nSub-task: {task.instruction}",
            },
            {"role": "user", "content": question},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text

    def _aggregate(
        self,
        question: str,
        strategy: str,
        context: str,
        results: list[WorkerResult],
        model: str,
    ) -> str:
        sections = []
        for r in results:
            label = f"Sub-task ({r.task.role}): {r.task.title}"
            if r.error:
                sections.append(f"{label}\n[worker failed: {r.error}]")
            else:
                sections.append(f"{label}\n{r.text}")
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
                    "answer to the user's original question. Some sub-task results "
                    "may be missing due to worker failure; produce the best answer "
                    "from what is available."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text

    def _direct_answer(self, question: str, strategy: str, context: str, model: str) -> str:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text
