from __future__ import annotations

from typing import Iterator

from openai import OpenAI, OpenAIError


class LLMError(Exception):
    """Raised when an LLM API call fails."""


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
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
    ) -> str:
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
            content = resp.choices[0].message.content
            return content or ""
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
