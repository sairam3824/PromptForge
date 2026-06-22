from __future__ import annotations

import os

from ..cache import LLMCache
from .base import LLMBackend


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        cache: LLMCache | None = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__()
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model
        self._cache = cache or LLMCache()
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[dict], **kwargs) -> str:
        cached = self._cache.get(self._model, messages)
        if cached is not None:
            self.cache_hits += 1
            return cached

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            **kwargs,
        )
        result: str = response.choices[0].message.content or ""
        self._cache.set(self._model, messages, result)
        self.call_count += 1
        return result
