from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from .capabilities import ProviderCapabilities
from .loggers import get_logger
from .negotiate import negotiate_structured_output

logger = get_logger("llm")


class LLMError(Exception):
    """Raised when an LLM API call fails."""


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_tokens: int = 0


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float | None = None,
                 provider: str = "", capability_overrides: dict | None = None):
        kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            kwargs["timeout"] = timeout
        self.client = OpenAI(**kwargs)
        self.base_url = base_url
        self.model = model
        self.capabilities = ProviderCapabilities(
            provider=provider or "unknown", **capability_overrides or {}
        )

    def chat_completion(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        disable_thinking: bool = False,
        json_mode: bool = False,
        json_schema: dict | None = None,
    ) -> ChatResult:
        if not any(m.get("role") == "user" for m in messages):
            raise LLMError(
                "Every chat_completion call must include at least one user message "
                "(all supported providers require or expect a user turn)."
            )
        try:
            mode = negotiate_structured_output(
                self.capabilities, json_mode=json_mode, json_schema=json_schema
            )
            kwargs = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": json_schema,
                        "strict": False,
                    },
                }
            elif mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
            if disable_thinking and self.capabilities.supports_thinking_toggle:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = self.client.chat.completions.create(**kwargs)
            u = resp.usage
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None)
            return ChatResult(
                text=resp.choices[0].message.content or "",
                model=resp.model or (model or self.model),
                prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
                completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
                total_tokens=getattr(u, "total_tokens", 0) if u else 0,
                cache_tokens=cached if isinstance(cached, int) else 0,
            )
        except OpenAIError as e:
            logger.exception("llm error", model=self.model, endpoint=self.base_url)
            raise LLMError(f"LLM API call failed: {e}") from e
