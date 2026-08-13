from __future__ import annotations

from .client import TracedLLMClient
from .patch import Installed
from .tracing import TraceStore, format_trace_summary, read_events


def install(client, config, domain=None):
    obs = config.observability
    if obs is None or not obs.enabled:
        return client, None
    store = TraceStore(obs.data_dir)
    traced = TracedLLMClient(client, store)
    installed = Installed(store=store, phase_map=obs.phase_map).apply()
    return traced, installed


__all__ = [
    "TraceStore",
    "TracedLLMClient",
    "Installed",
    "format_trace_summary",
    "read_events",
    "install",
]
