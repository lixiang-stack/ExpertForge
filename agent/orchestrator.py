from __future__ import annotations

from .config import AgentConfig, DomainConfig, resolve_judge_model
from .evaluation.judge import Judge
from .llm import LLMClient, LLMError
from .loggers import get_logger
from .parsing import parse_json
from .strategy import build_registry
from .router import RouteResult
from .worker_pool import WorkerResult, WorkerTask, run_workers

logger = get_logger("orchestrator")


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
"""


_REAGGREGATE_SYSTEM_PROMPT = """{context}

You are synthesizing sub-task results into one coherent final answer to the
user's original question. Some sub-task results may be missing due to worker
failure; produce the best answer from what is available.

A previous draft scored too low on these judge dimensions; produce an
improved draft that addresses them:
{feedback_lines}
"""

_REAGGREGATE_USER_TEMPLATE = """User question: {question}

Sub-task results:

{sub_task_sections}

Previous draft:
{previous}
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
        logger.info("orchestration start", strategy=route.strategy, model=model)
        context = self._strategy_context(route.strategy)
        tasks = self._plan(question, route.strategy, context, model)
        if tasks is None:
            return self._direct_answer(question, route.strategy, context, model)
        policy = self.domain.orchestration
        results = run_workers(
            tasks,
            lambda task: self._worker(question, task, context, model),
            max_workers=policy.max_workers if policy else 4,
        )
        for r in results:
            if r.error:
                logger.warning(
                    "worker failure", task=r.task.title, role=r.task.role, error=r.error
                )
        if all(r.error for r in results):
            return self._direct_answer(question, route.strategy, context, model)
        answer = self._aggregate(question, route.strategy, context, results, model)
        if not (policy and policy.evaluator.enabled):
            return answer
        return self._evaluate_loop(
            question, route.strategy, context, results, answer, model, policy.evaluator
        )

    def _judge_name(self) -> str:
        return resolve_judge_model(self.config)

    def _evaluate_loop(
        self, question: str, strategy: str, context: str,
        results: list[WorkerResult], answer: str, model: str, evaluator,
    ) -> str:
        judge = Judge(self.client, self._judge_name())
        threshold = evaluator.min_dimension_score
        for round_no in range(evaluator.max_rounds + 1):
            scorecard = self._evaluate(judge, question, answer)
            if scorecard is None:
                return answer
            if all(score >= threshold for score in scorecard.values()):
                return answer
            if round_no == evaluator.max_rounds:
                return answer
            feedback = [f"{dim}: {score}/5" for dim, score in scorecard.items() if score < threshold]
            try:
                answer = self._reaggregate(
                    question, strategy, context, results, answer, feedback, round_no, model
                )
            except LLMError:
                return answer
        return answer

    def _evaluate(self, judge: Judge, question: str, answer: str) -> dict | None:
        return judge.score(question, answer)

    def _reaggregate(
        self, question: str, strategy: str, context: str,
        results: list[WorkerResult], previous: str, feedback: list[str],
        round_no: int, model: str,
    ) -> str:
        sections = []
        for r in results:
            label = f"Sub-task ({r.task.role}): {r.task.title}"
            if r.error:
                sections.append(f"{label}\n[worker failed: {r.error}]")
            else:
                sections.append(f"{label}\n{r.text}")
        feedback_lines = "\n".join(f"- {f}" for f in feedback)
        system = _REAGGREGATE_SYSTEM_PROMPT.format(
            context=context, feedback_lines=feedback_lines,
        )
        user_content = _REAGGREGATE_USER_TEMPLATE.format(
            question=question,
            sub_task_sections="\n\n".join(sections),
            previous=previous,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text

    def _plan(
        self, question: str, strategy: str, context: str, model: str
    ) -> list[WorkerTask] | None:
        prompt = _PLANNER_PROMPT.format(
            name=self.domain.name,
            description=self.domain.description,
            context=context,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        result = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_planner_schema()
        )
        data = parse_json(result.text)
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
