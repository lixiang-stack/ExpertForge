from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI, OpenAIError


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
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float | None = None):
        kwargs: dict = {"api_key": api_key, "base_url": base_url}
        if timeout is not None:
            kwargs["timeout"] = timeout
        self.client = OpenAI(**kwargs)
        self.model = model

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
        try:
            kwargs = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }
            if json_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_result",
                        "schema": json_schema,
                        "strict": False,
                    },
                }
            elif json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if disable_thinking:
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
            raise LLMError(f"LLM API call failed: {e}") from e

    def chat_completion_stream(
        self, messages: list[dict], *, model: str | None = None, temperature: float = 0.7
    ) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                choices = chunk.choices
                if choices:
                    content = choices[0].delta.content
                    if content:
                        yield content
        except OpenAIError as e:
            raise LLMError(f"LLM API call failed: {e}") from e
