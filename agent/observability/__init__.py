"""Facade / single wiring point for the observability subsystem.

Unlike a plain (empty) package marker, this `__init__` exists to hide the
assembly order of the cross-file internals (client.py / patch.py /
tracing.py): callers get one entry point, `install()`, and a fixed return
contract, so they never have to know how the pieces fit together.

`install()` does the whole three-step wiring:
  1. switch: if observability is disabled in config, return the client
     untouched (None plugin) -- business is never affected when off;
  2. build the TraceStore (persistent JSONL + in-memory hot cache);
  3. inject on both paths:
       - wrap the LLM client (TracedLLMClient) to record llm_call events,
       - monkey-patch the business pipeline (Installed(...).apply()).
  It returns (traced_client, installed) so the caller runs business on the
  wrapped client and keeps the plugin handle for later queries.

This file also re-exports the public symbols of the internals so users can
`from agent.observability import install, TraceStore, ...` without knowing
the internal layout.
"""

from __future__ import annotations

from .client import TracedLLMClient
from .patch import Installed
from .tracing import TraceStore, format_trace_summary, read_events


def install(client, config, domain=None):
    """Wire observability into the pipeline. Returns (traced, plugin); when
    disabled, returns (client, None) unchanged. Never raises."""
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
