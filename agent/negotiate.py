from __future__ import annotations

from .capabilities import ProviderCapabilities


def negotiate_structured_output(
    caps: ProviderCapabilities,
    *,
    json_mode: bool,
    json_schema: dict | None,
) -> str | None:
    """Pick the structured-output mechanism.

    json_object is the universal default for OpenAI-compat targets. json_schema
    is preferred when the provider declares it. Returns None only when the
    caller requested no structured output (plain-answer paths) — never as a
    capability fallback.
    """
    if json_schema is not None:
        return "json_schema" if caps.supports_json_schema else "json_object"
    if json_mode:
        return "json_object"
    return None