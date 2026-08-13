from __future__ import annotations

import contextvars
import json
import threading
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


class TraceStore:
    """Append-only JSONL store, one file per day. Writes never raise."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            warnings.warn(f"observability: cannot create data dir: {e}")
        self._lock = threading.Lock()
        self._day: str | None = None
        self._file = None
        self._in_memory: dict[str, list[dict]] = {}
        self._memory_order: list[str] = []
        self._MAX_MEMORY_TRACES = 100

    def _current_file(self):
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._day:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
            self._day = day
            self._file = (self.data_dir / f"trace-{day}.jsonl").open("a", encoding="utf-8")
        return self._file

    def write(self, event: dict) -> None:
        try:
            line = json.dumps(event, ensure_ascii=False) + "\n"
            with self._lock:
                f = self._current_file()
                if f is None:
                    raise ValueError("unavailable file handle")
                f.write(line)
                f.flush()
                tid = event.get("trace_id")
                if event.get("type") == "llm_call" and tid is not None:
                    if tid not in self._in_memory:
                        self._in_memory[tid] = []
                        self._memory_order.append(tid)
                        if len(self._memory_order) > self._MAX_MEMORY_TRACES:
                            oldest = self._memory_order.pop(0)
                            self._in_memory.pop(oldest, None)
                    self._in_memory[tid].append(event)
        except Exception as e:  # noqa: BLE001 - observability must never break business
            warnings.warn(f"observability: failed to write trace event: {e}")

    def trace_llm_calls(self, trace_id: str) -> list[dict]:
        return list(self._in_memory.get(trace_id, []))

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
            self._file = None
            self._day = None


def read_events(data_dir: str | Path, *, day: str | None = None) -> tuple[list[dict], int]:
    base = Path(data_dir)
    pattern = f"trace-{day}.jsonl" if day else "trace-*.jsonl"
    bad = 0
    events: list[dict] = []
    try:
        paths = sorted(base.glob(pattern))
    except OSError:
        return events, 1
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        bad += 1
                        continue
                    if isinstance(data, dict):
                        events.append(data)
                    else:
                        bad += 1
        except OSError:
            bad += 1
    return events, bad


@dataclass
class _Span:
    trace_id: str
    phases: list[str] = field(default_factory=list)


_span_var: contextvars.ContextVar[_Span | None] = contextvars.ContextVar(
    "observability_span", default=None
)


@contextmanager
def trace_span() -> Iterator[str]:
    trace_id = uuid.uuid4().hex[:12]
    token = _span_var.set(_Span(trace_id=trace_id))
    try:
        yield trace_id
    finally:
        _span_var.reset(token)


@contextmanager
def phase(name: str) -> Iterator[None]:
    span = _span_var.get()
    if span is None:
        yield
        return
    span.phases.append(name)
    try:
        yield
    finally:
        span.phases.pop()


def current_trace_id() -> str | None:
    span = _span_var.get()
    return span.trace_id if span else None


def current_phase() -> str | None:
    span = _span_var.get()
    if span is None or not span.phases:
        return None
    return span.phases[-1]


def now_millis() -> int:
    return int(datetime.now().timestamp() * 1000)


def format_trace_summary(trace_id: str, calls: list[dict]) -> str:
    in_tokens = sum(c.get("prompt_tokens") or 0 for c in calls)
    out_tokens = sum(c.get("completion_tokens") or 0 for c in calls)
    total_s = sum(c.get("latency_ms") or 0 for c in calls) / 1000
    return f"[trace {trace_id}] in={in_tokens} out={out_tokens} {total_s:.1f}s"
