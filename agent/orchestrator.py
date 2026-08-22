from __future__ import annotations

from dataclasses import dataclass

from .config import AgentConfig, DomainConfig, resolve_judge_model
from .evaluation.judge import Judge
from .llm import LLMClient, LLMError
from .loggers import get_logger
from .parsing import parse_json
from .strategy import build_registry
from .router import RouteResult
from .worker_pool import WorkerResult, WorkerTask, run_workers

logger = get_logger("orchestrator")


@dataclass
class Issue:
    severity: str
    description: str
    suggestion: str


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


def _perspectives_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "perspectives": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "focus": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["title", "focus", "role"],
                },
            }
        },
        "required": ["perspectives"],
    }


def _issues_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                        "description": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["severity", "description", "suggestion"],
                },
            }
        },
        "required": ["issues"],
    }


def _parse_issues(text: str | None) -> list[Issue]:
    data = parse_json(text) if text else None
    if not data or not isinstance(data.get("issues"), list):
        return []
    issues: list[Issue] = []
    for item in data["issues"]:
        if not isinstance(item, dict):
            continue
        description = item.get("description")
        if not isinstance(description, str) or not description:
            continue
        severity = item.get("severity")
        if not isinstance(severity, str) or not severity:
            severity = "medium"
        suggestion = item.get("suggestion")
        if not isinstance(suggestion, str):
            suggestion = ""
        issues.append(Issue(severity=severity, description=description, suggestion=suggestion))
    return issues


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

_PERSPECTIVES_PROMPT = """You are a review planner for an expert domain named {name}.

{description}

Task context:
{context}

Rules:
- Plan 2-4 distinct review perspectives for checking a draft expert answer.
- Each perspective must be verifiable by reading the user's question and the
  draft alone (e.g. Consistency & Coherence, Feasibility & Operations,
  Compliance & Security, Cost).
- Assign each perspective a distinct role name that defines its focused responsibility.
- Output ONLY a single JSON object: {{"perspectives": [{{"title": "...", "focus": "...", "role": "..."}}]}}
"""

_CRITIC_SYSTEM_TEMPLATE = """{context}

You are a reviewer. Your review perspective: {role}
Focus: {instruction}

Review the draft answer to the user's question from this perspective only.
Report only real defects:
- Internal contradictions or inconsistent decisions across sections
- Unsupported claims presented as fact (assumptions must stay flagged as assumptions)
- Technical errors
- Missing reasoning where the question demands it

Do NOT rewrite the answer. Output ONLY a single JSON object:
{{"issues": [{{"severity": "high|medium|low", "description": "...", "suggestion": "..."}}]}}
If there are no defects, output {{"issues": []}}.
"""

_REVISE_SYSTEM_TEMPLATE = """{context}

You authored the draft answer below. Reviewers found issues in it. Produce an
improved final version that resolves every issue while keeping the overall
structure and all correct content. State important assumptions explicitly;
never present invented numbers or facts as established requirements.
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
        policy = self.domain.orchestration
        topology = policy.topology if policy else "map_reduce"
        if topology == "critique":
            return self._run_critique(question, route.strategy, context, model, policy)
        return self._run_map_reduce(question, route, context, model, policy)

    def _run_map_reduce(self, question: str, route: RouteResult, context: str, model: str, policy) -> str:
        tasks = self._plan(question, route.strategy, context, model)
        if tasks is None:
            return self._direct_answer(question, route.strategy, context, model)
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
            question, route.strategy, context, answer, model, policy.evaluator,
            improve=lambda previous, feedback, round_no: self._reaggregate(
                question, route.strategy, context, results, previous, feedback, round_no, model),
        )

    def _run_critique(self, question: str, strategy: str, context: str, model: str, policy) -> str:
        draft = self._draft(question, strategy, context, model)
        perspectives = self._plan_perspectives(question, strategy, context, model)
        issues: list[Issue] = []
        if perspectives:
            results = run_workers(
                perspectives,
                lambda p: self._critic(question, p, context, draft, model),
                max_workers=policy.max_workers if policy else 4,
            )
            for r in results:
                if r.error:
                    logger.warning(
                        "critic failure", task=r.task.title, role=r.task.role, error=r.error
                    )
            issues = self._consolidate(results)
        answer = draft
        if issues:
            try:
                answer = self._revise(question, strategy, context, draft, issues, model)
            except LLMError:
                logger.warning("revise failure, returning draft")
                answer = draft
        if not (policy and policy.evaluator.enabled):
            return answer

        def improve(previous: str, feedback: list[str], round_no: int) -> str:
            judge_issues = [
                Issue(severity="high", description=f"Judge scored too low - {f}", suggestion="")
                for f in feedback
            ]
            return self._revise(question, strategy, context, previous, judge_issues, model)

        return self._evaluate_loop(
            question, strategy, context, answer, model, policy.evaluator, improve=improve,
        )

    def _draft(self, question: str, strategy: str, context: str, model: str) -> str:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
        ]
        # No disable_thinking: keep the client default so the draft has the
        # same reasoning budget as the single-call baseline (Strategy.process).
        return self.client.chat_completion(messages, model=model).text

    def _plan_perspectives(
        self, question: str, strategy: str, context: str, model: str
    ) -> list[WorkerTask] | None:
        prompt = _PERSPECTIVES_PROMPT.format(
            name=self.domain.name,
            description=self.domain.description,
            context=context,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
        result = self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_perspectives_schema()
        )
        data = parse_json(result.text)
        if not data or not isinstance(data.get("perspectives"), list):
            return None
        perspectives: list[WorkerTask] = []
        for item in data["perspectives"]:
            if not isinstance(item, dict):
                return None
            title = item.get("title")
            focus = item.get("focus")
            if not isinstance(title, str) or not isinstance(focus, str):
                return None
            role = item.get("role")
            if not isinstance(role, str) or not role:
                role = title
            perspectives.append(WorkerTask(title=title, instruction=focus, role=role))
        return perspectives or None

    def _critic(self, question: str, perspective: WorkerTask, context: str, draft: str, model: str) -> str:
        system = _CRITIC_SYSTEM_TEMPLATE.format(
            context=context, role=perspective.role, instruction=perspective.instruction,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"User question:\n{question}\n\nDraft answer:\n{draft}"},
        ]
        return self.client.chat_completion(
            messages, model=model, disable_thinking=True, json_schema=_issues_schema()
        ).text

    def _consolidate(self, results: list[WorkerResult]) -> list[Issue]:
        issues: list[Issue] = []
        for r in results:
            if r.error or r.text is None:
                continue
            issues.extend(_parse_issues(r.text))
        return issues

    def _revise(self, question: str, strategy: str, context: str, draft: str, issues: list[Issue], model: str) -> str:
        lines = "\n".join(
            f"- [{i.severity}] {i.description}" + (f" Suggestion: {i.suggestion}" if i.suggestion else "")
            for i in issues
        )
        system = _REVISE_SYSTEM_TEMPLATE.format(context=context)
        user_content = (
            f"User question:\n{question}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Reviewer issues to resolve:\n{lines}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return self.client.chat_completion(messages, model=model, disable_thinking=True).text

    def _judge_name(self) -> str:
        return resolve_judge_model(self.config)

    def _evaluate_loop(
        self, question: str, strategy: str, context: str,
        answer: str, model: str, evaluator,
        improve,
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
                answer = improve(answer, feedback, round_no)
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
