from __future__ import annotations

import hashlib
import json
from pathlib import Path

import diskcache


class LLMCache:
    """Disk-backed cache keyed by (model, SHA-256 of serialised messages)."""

    def __init__(self, cache_dir: str | Path = ".promptforge_cache"):
        self._cache = diskcache.Cache(str(cache_dir), size_limit=2**30)

    def _key(self, model: str, messages: list) -> str:
        payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model: str, messages: list) -> str | None:
        return self._cache.get(self._key(model, messages))

    def set(self, model: str, messages: list, response: str) -> None:
        self._cache.set(self._key(model, messages), response)

    def __len__(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        self._cache.close()
