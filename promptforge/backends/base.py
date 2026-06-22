from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    def __init__(self) -> None:
        self.call_count: int = 0
        self.cache_hits: int = 0

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    def total_calls(self) -> int:
        return self.call_count + self.cache_hits
