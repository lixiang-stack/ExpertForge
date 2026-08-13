"""Read model for the observability event stream.

`events` is the flat event stream written by the observability layer; this
module is a read-only "read model" that reorganizes raw events into three
render-ready views:

1. summary stats  - one TraceSummary per trace, plus model/global aggregates
2. timeline       - chronological Step list per trace
3. grouped stages - flat timeline re-grouped into ordered Stage blocks

All functions are pure/read-only and never raise.

This is the READ side of the "write-side minimal, read-side derives"
philosophy: tracing.py only tags each event with "where am I now"
(trace_id + innermost phase), and any hierarchy is rebuilt here. The
one-to-many "a phase contains many sub-events" is expressed by events that
share the same phase string in the stream. Example (view 3):

    flat stream:
      {"type":"llm_call","phase":"orchestration.planner","ts":10}
      {"type":"llm_call","phase":"orchestration.worker.1","ts":20}
      {"type":"llm_call","phase":"orchestration.worker.1","ts":30}
      {"type":"llm_call","phase":"orchestration.worker.2","ts":40}
    reconstructs into:
      Stage("orchestration.worker")
        workers: [WorkerGroup(1, steps=[ts20, ts30]), WorkerGroup(2, steps=[ts40])]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ============================ View 1: summary stats ============================
@dataclass
class TraceSummary:
    trace_id: str = ""
    question: str = ""
    domain: str | None = None
    in_tokens: int = 0
    out_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    total_latency_ms: float = 0.0
    reject: bool = False
    has_error: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelStat:
    model: str = ""
    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    total_latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def summarize_traces(events: list[dict]) -> list[TraceSummary]:
    """One TraceSummary per trace_id, first-seen order. Never raises."""
    order: list[str] = []
    rows: dict[str, TraceSummary] = {}
    for e in events:
        tid = e.get("trace_id")
        if not tid:
            continue
        if tid not in rows:
            rows[tid] = TraceSummary(trace_id=tid)
            order.append(tid)
        r = rows[tid]
        typ = e.get("type")
        if typ == "trace_start":
            if not r.question:
                r.question = e.get("question", "")
            if r.domain is None:
                r.domain = e.get("domain")
        elif typ == "trace_end":
            r.reject = bool(e.get("reject"))
            r.total_latency_ms = float(e.get("total_latency_ms") or r.total_latency_ms)
        elif typ == "llm_call":
            r.llm_calls += 1
            r.in_tokens += e.get("prompt_tokens") or 0
            r.out_tokens += e.get("completion_tokens") or 0
            r.total_tokens += e.get("total_tokens") or 0
            r.total_latency_ms += e.get("latency_ms") or 0
            if e.get("status") == "error":
                r.has_error = True
    return [rows[tid] for tid in order]


def total_stats(events: list[dict]) -> dict:
    """Header aggregates: traces, llm_calls, in/out/total tokens, latency, has_error."""
    rows = summarize_traces(events)
    return {
        "traces": len(rows),
        "llm_calls": sum(r.llm_calls for r in rows),
        "in_tokens": sum(r.in_tokens for r in rows),
        "out_tokens": sum(r.out_tokens for r in rows),
        "total_tokens": sum(r.total_tokens for r in rows),
        "total_latency_ms": round(sum(r.total_latency_ms for r in rows), 1),
        "has_error": any(r.has_error for r in rows),
    }


def model_stats(events: list[dict]) -> list[ModelStat]:
    """One ModelStat per model, sorted by total (in+out) tokens descending."""
    acc: dict[str, ModelStat] = {}
    for e in events:
        if e.get("type") != "llm_call":
            continue
        m = e.get("model") or "?"
        st = acc.setdefault(m, ModelStat(model=m))
        st.calls += 1
        st.in_tokens += e.get("prompt_tokens") or 0
        st.out_tokens += e.get("completion_tokens") or 0
        st.total_latency_ms += e.get("latency_ms") or 0
    return sorted(acc.values(), key=lambda s: s.in_tokens + s.out_tokens, reverse=True)
# ============================ /View 1: summary stats ============================


# ============================ View 2: timeline ============================
@dataclass
class Step:
    ts: int = 0
    kind: str = ""  # "decision" | "llm_call" | "result"
    phase: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _decision_type(phase: str) -> str:
    if phase == "classification":
        return "classification"
    if phase == "route":
        return "route"
    if phase == "orchestration.planner":
        return "plan"
    if phase.startswith("orchestration.worker"):
        return "worker"
    return "decision"


def build_timeline(events: list[dict]) -> dict[str, list[Step]]:
    """Per trace_id, chronological ordered steps. Unknown events and missing
    trace_ids are skipped. Never raises."""
    by_tid: dict[str, list[Step]] = {}
    for e in events:
        tid = e.get("trace_id")
        if not tid:
            continue
        typ = e.get("type")
        ts = e.get("ts") or 0
        ph = e.get("phase") or "?"
        if typ == "decision":
            step = Step(ts=ts, kind="decision", phase=ph,
                        detail={"type": _decision_type(ph), "data": e.get("data") or {}})
        elif typ == "llm_call":
            step = Step(ts=ts, kind="llm_call", phase=ph, detail={
                "model": e.get("model"), "in_tokens": e.get("prompt_tokens"),
                "out_tokens": e.get("completion_tokens"), "total_tokens": e.get("total_tokens"),
                "latency_ms": e.get("latency_ms"), "status": e.get("status"),
                "error": e.get("error")})
        elif typ == "trace_end":
            step = Step(ts=ts, kind="result", phase=ph, detail={
                "answer_len": e.get("answer_len"), "total_llm_calls": e.get("total_llm_calls"),
                "total_tokens": e.get("total_tokens"), "total_latency_ms": e.get("total_latency_ms"),
                "reject": e.get("reject")})
        else:
            continue
        if tid not in by_tid:
            by_tid[tid] = []
        by_tid[tid].append(step)
    for steps in by_tid.values():
        steps.sort(key=lambda s: s.ts)
    return {tid: steps for tid, steps in by_tid.items() if steps}
# ============================ /View 2: timeline ============================


# ============================ View 3: grouped stages ============================
@dataclass
class WorkerGroup:
    number: int = 0
    task_title: str = ""
    steps: list[Step] = field(default_factory=list)


@dataclass
class Stage:
    title: str = ""
    steps: list[Step] = field(default_factory=list)
    workers: list[WorkerGroup] = field(default_factory=list)


def _stage_title(step: Step) -> str:
    ph = step.phase or "?"
    if step.kind == "result":
        return "result"
    if ph == "classification":
        return "classification"
    if ph == "route":
        return "route"
    if ph.startswith("strategy."):
        return ph
    if ph == "orchestration.planner":
        return "orchestration.planner"
    if ph.startswith("orchestration.worker."):
        return "orchestration.worker"
    if ph == "orchestration.aggregate":
        return "orchestration.aggregate"
    return ph


def _worker_number(phase: str) -> int:
    try:
        return int(phase.rsplit(".", 1)[1])
    except (IndexError, ValueError):
        return 0


def _worker_task(step: Step) -> str:
    if step.kind == "decision":
        return str((step.detail.get("data") or {}).get("task") or "")
    return ""


def group_stages(timeline: dict[str, list[Step]]) -> dict[str, list[Stage]]:
    """Per trace, group flat steps into ordered Stage groups. Worker phases
    collapse into one stage with per-worker sub-blocks. Never raises."""
    result: dict[str, list[Stage]] = {}
    for tid, steps in timeline.items():
        by_title: dict[str, Stage] = {}
        order: list[str] = []
        for s in steps:
            title = _stage_title(s)
            if title not in by_title:
                by_title[title] = Stage(title=title)
                order.append(title)
            stage = by_title[title]
            if title == "orchestration.worker":
                n = _worker_number(s.phase)
                wg = next((w for w in stage.workers if w.number == n), None)
                if wg is None:
                    wg = WorkerGroup(number=n, task_title=_worker_task(s))
                    stage.workers.append(wg)
                if s.kind != "decision":
                    wg.steps.append(s)
            else:
                stage.steps.append(s)
        result[tid] = [by_title[t] for t in order]
    return result
# ============================ /View 3: grouped stages ============================