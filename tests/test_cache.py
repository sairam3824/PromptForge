import tempfile

import pytest

from promptforge.cache import LLMCache


@pytest.fixture
def tmp_cache(tmp_path):
    cache = LLMCache(tmp_path / "cache")
    yield cache
    cache.close()


def test_cache_miss_returns_none(tmp_cache):
    assert tmp_cache.get("gpt-4o-mini", [{"role": "user", "content": "hi"}]) is None


def test_cache_set_then_get(tmp_cache):
    msgs = [{"role": "user", "content": "hello"}]
    tmp_cache.set("gpt-4o-mini", msgs, "world")
    assert tmp_cache.get("gpt-4o-mini", msgs) == "world"


def test_cache_model_isolation(tmp_cache):
    msgs = [{"role": "user", "content": "hello"}]
    tmp_cache.set("gpt-4o-mini", msgs, "mini")
    tmp_cache.set("gpt-4o", msgs, "full")
    assert tmp_cache.get("gpt-4o-mini", msgs) == "mini"
    assert tmp_cache.get("gpt-4o", msgs) == "full"


def test_cache_content_isolation(tmp_cache):
    m1 = [{"role": "user", "content": "hello"}]
    m2 = [{"role": "user", "content": "world"}]
    tmp_cache.set("gpt-4o-mini", m1, "answer1")
    assert tmp_cache.get("gpt-4o-mini", m2) is None


def test_cache_len(tmp_cache):
    assert len(tmp_cache) == 0
    msgs = [{"role": "user", "content": "test"}]
    tmp_cache.set("m", msgs, "r")
    assert len(tmp_cache) == 1


def test_cache_overwrite(tmp_cache):
    msgs = [{"role": "user", "content": "q"}]
    tmp_cache.set("m", msgs, "first")
    tmp_cache.set("m", msgs, "second")
    assert tmp_cache.get("m", msgs) == "second"
