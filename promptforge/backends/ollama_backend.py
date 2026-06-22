from __future__ import annotations

import httpx

from ..cache import LLMCache
from .base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        cache: LLMCache | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._cache = cache or LLMCache()
        self._timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, messages: list[dict], **kwargs) -> str:
        cached = self._cache.get(self._model, messages)
        if cached is not None:
            self.cache_hits += 1
            return cached

        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={"model": self._model, "messages": messages, "stream": False},
            timeout=self._timeout,
        )
        response.raise_for_status()
        result: str = response.json()["message"]["content"]
        self._cache.set(self._model, messages, result)
        self.call_count += 1
        return result
