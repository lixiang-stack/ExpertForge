from __future__ import annotations

import functools
import threading
import warnings
from dataclasses import dataclass, field

from agent.chat import Chat
from agent.classification import ClassificationService
from agent.orchestrator import Orchestrator
from agent.router import Router
from agent.strategy import Strategy

from .tracing import (
    TraceStore,
    current_trace_id,
    format_trace_summary,
    now_millis,
    phase,
    trace_span,
)


DEFAULT_PHASES: dict[str, str] = {
    "Chat.respond": "trace",
    "ClassificationService.classify": "classification",
    "Router.route": "route",
    "Strategy.process": "strategy",
    "Orchestrator._plan": "orchestration.planner",
    "Orchestrator._worker": "orchestration.worker",
    "Orchestrator._aggregate": "orchestration.aggregate",
    "Orchestrator._direct_answer": "orchestration.direct",
}

# The active Installed (or None). Wrappers become transparent passthroughs when
# None, so class-level patching is safe even across tests/modules that never
# install observability.
_ACTIVE: "Installed | None" = None

_PATCH_MARKER = "__observability_patched__"


def _current_inst() -> "Installed | None":
    return _ACTIVE


@dataclass
class Installed:
    store: TraceStore
    phase_map: dict[str, str] = field(default_factory=dict)
    patched: list[str] = field(default_factory=list)
    _worker_counters: dict = field(default_factory=dict)
    _worker_lock: threading.Lock = field(default_factory=threading.Lock)

    def _phase(self, key: str) -> str:
        return self.phase_map.get(key, DEFAULT_PHASES[key])

    def _next_worker(self, trace_id: str) -> int:
        with self._worker_lock:
            n = self._worker_counters.get(trace_id, 0) + 1
            self._worker_counters[trace_id] = n
            return n

    def _record_decision(self, trace_id: str, ph: str, data: dict) -> None:
        try:
            self.store.write({"type": "decision", "trace_id": trace_id, "phase": ph,
                              "ts": now_millis(), "data": data})
        except Exception as e:  # noqa: BLE001 - degrade, never break business
            warnings.warn(f"observability: failed to record decision: {e}")

    def _wrap(self, key: str, target, patch_name: str) -> None:
        factories = {
            "Chat.respond": self._wrap_respond,
            "ClassificationService.classify": self._wrap_classify,
            "Router.route": self._wrap_route,
            "Strategy.process": self._wrap_strategy,
            "Orchestrator._plan": self._wrap_plan,
            "Orchestrator._worker": self._wrap_worker,
            "Orchestrator._aggregate": self._wrap_aggregate,
            "Orchestrator._direct_answer": self._wrap_direct,
        }
        try:
            original = getattr(target, patch_name)
            if getattr(original, _PATCH_MARKER, None) == key:
                return  # idempotent: already wrapped by a previous install
            wrapper = factories[key](original, key)
            setattr(target, patch_name, functools.wraps(original)(wrapper))
            wrapper.__setattr__(_PATCH_MARKER, key)
            self.patched.append(key)
        except Exception as e:  # noqa: BLE001 - degrade, never block business
            warnings.warn(f"observability: failed to patch {key}: {e}")

    # Wrapper factories. The wrapper's first positional arg is the business
    # instance (chat, cls, strat, orch...). Each wrapper resolves the active
    # install via `_current_inst()` at call time (not wrap time), so a
    # re-install/apply() with a new store writes to that store. Transparent
    # passthrough when no install is active.

    def _wrap_respond(self, original, key):
        def wrapper(chat, question):
            inst = _current_inst()
            if inst is None:
                return original(chat, question)
            with trace_span() as tid:
                ph = inst._phase(key)
                try:
                    inst.store.write({"type": "trace_start", "trace_id": tid, "phase": ph,
                                      "ts": now_millis(), "question": question,
                                      "domain": getattr(chat.domain, "name", None)})
                except Exception as e:  # noqa: BLE001 - degrade, never break business
                    warnings.warn(f"observability: failed to record trace_start: {e}")
                response = original(chat, question)
                calls = inst.store.trace_llm_calls(tid)
                total_tokens = sum(c.get("total_tokens") or 0 for c in calls)
                total_lat = sum(c.get("latency_ms") or 0 for c in calls)
                try:
                    inst.store.write({"type": "trace_end", "trace_id": tid, "phase": ph,
                                      "ts": now_millis(), "answer_len": len(response.text),
                                      "total_llm_calls": len(calls), "total_tokens": total_tokens,
                                      "total_latency_ms": round(total_lat, 1),
                                      "reject": response.kind == "reject"})
                except Exception as e:  # noqa: BLE001 - degrade, never break business
                    warnings.warn(f"observability: failed to record trace_end: {e}")
                try:
                    if calls:
                        print(format_trace_summary(tid, calls))
                except Exception:  # noqa: BLE001 - display must never break business
                    pass
                return response
        return wrapper

    def _wrap_classify(self, original, key):
        def wrapper(cls, question, *, model=None):
            inst = _current_inst()
            if inst is None:
                return original(cls, question, model=model)
            with phase(inst._phase(key)):
                result = original(cls, question, model=model)
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, inst._phase(key), {
                        "in_domain": result.in_domain, "intent": result.intent,
                        "complexity": result.complexity, "reason": result.reason})
                return result
        return wrapper

    def _wrap_route(self, original, key):
        def wrapper(rtr, question):
            inst = _current_inst()
            if inst is None:
                return original(rtr, question)
            with phase(inst._phase(key)):
                result = original(rtr, question)
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, inst._phase(key), {
                        "in_domain": result.in_domain, "strategy": result.strategy,
                        "intent": result.intent, "complexity": result.complexity,
                        "orchestrate": result.orchestrate, "reject_reason": result.reject_reason})
                return result
        return wrapper

    def _wrap_strategy(self, original, key):
        def wrapper(strat, client, question, history, *, model=None):
            inst = _current_inst()
            if inst is None:
                return original(strat, client, question, history, model=model)
            with phase(f"{inst._phase(key)}.{strat.strategy_id}"):
                return original(strat, client, question, history, model=model)
        return wrapper

    def _wrap_plan(self, original, key):
        def wrapper(orch, question, strategy, context, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, model)
            with phase(inst._phase(key)):
                tasks = original(orch, question, strategy, context, model)
                tid = current_trace_id()
                if tid:
                    data = {"degraded": True} if tasks is None else {
                        "tasks": [{"title": t, "instruction": i} for t, i in tasks]}
                    inst._record_decision(tid, inst._phase(key), data)
                return tasks
        return wrapper

    def _wrap_worker(self, original, key):
        def wrapper(orch, question, task, context, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, task, context, model)
            base = inst._phase(key)
            n = inst._next_worker(current_trace_id() or "")
            with phase(f"{base}.{n}"):
                tid = current_trace_id()
                if tid:
                    inst._record_decision(tid, f"{base}.{n}", {"task": task[0]})
                return original(orch, question, task, context, model)
        return wrapper

    def _wrap_aggregate(self, original, key):
        def wrapper(orch, question, strategy, context, tasks, outputs, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, tasks, outputs, model)
            with phase(inst._phase(key)):
                return original(orch, question, strategy, context, tasks, outputs, model)
        return wrapper

    def _wrap_direct(self, original, key):
        def wrapper(orch, question, strategy, context, model):
            inst = _current_inst()
            if inst is None:
                return original(orch, question, strategy, context, model)
            with phase(inst._phase(key)):
                return original(orch, question, strategy, context, model)
        return wrapper

    def apply(self) -> "Installed":
        global _ACTIVE
        targets = [
            ("Chat.respond", Chat, "respond"),
            ("ClassificationService.classify", ClassificationService, "classify"),
            ("Router.route", Router, "route"),
            ("Strategy.process", Strategy, "process"),
            ("Orchestrator._plan", Orchestrator, "_plan"),
            ("Orchestrator._worker", Orchestrator, "_worker"),
            ("Orchestrator._aggregate", Orchestrator, "_aggregate"),
            ("Orchestrator._direct_answer", Orchestrator, "_direct_answer"),
        ]
        for key, cls, method in targets:
            self._wrap(key, cls, method)
        _ACTIVE = self
        return self
