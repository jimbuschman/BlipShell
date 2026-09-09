"""LLMClient.generate() response cache: the key must include every kwarg that
changes the reply, and callers must be able to bypass it.

Found 2026-09-09 by the benchmark harness: a think=True diagnostic pass came
back byte-identical to think=False in 0.0s, because the key was
(model, system, prompt). The same hole would hand a schema-constrained
(`format`) call a cached free-text reply.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from blipshell.llm import client as client_mod
from blipshell.llm.client import LLMClient


@pytest.fixture(autouse=True)
def _clean_cache():
    client_mod._response_cache.clear()
    yield
    client_mod._response_cache.clear()


def _client() -> tuple[LLMClient, AsyncMock]:
    c = LLMClient(host="http://unused:1", max_retries=0, timeout=5.0)
    fake = AsyncMock(return_value={"message": {"content": "reply"}})
    c._client.chat = fake  # never touches the network
    return c, fake


async def test_identical_call_is_cached_once():
    c, fake = _client()
    assert await c.generate("p", "m", system="s") == "reply"
    assert await c.generate("p", "m", system="s") == "reply"
    assert fake.await_count == 1


async def test_think_is_part_of_the_key():
    c, fake = _client()
    await c.generate("p", "m", think=False)
    await c.generate("p", "m", think=True)
    assert fake.await_count == 2, "think=True was served the think=False reply"


async def test_format_is_part_of_the_key():
    c, fake = _client()
    await c.generate("p", "m")
    await c.generate("p", "m", format={"type": "object"})
    assert fake.await_count == 2, "a schema-constrained call was served a free-text reply"


async def test_options_are_part_of_the_key():
    c, fake = _client()
    await c.generate("p", "m", options={"num_ctx": 8192})
    await c.generate("p", "m", options={"num_ctx": 32768})
    assert fake.await_count == 2


async def test_use_cache_false_always_calls_and_does_not_store():
    c, fake = _client()
    await c.generate("p", "m", use_cache=False)
    await c.generate("p", "m", use_cache=False)
    assert fake.await_count == 2
    assert not client_mod._response_cache
    # and a cached entry does not short-circuit a use_cache=False call
    await c.generate("p", "m")
    await c.generate("p", "m", use_cache=False)
    assert fake.await_count == 4


class TestRouterForwardsUseCache:
    @pytest.fixture
    def router(self):
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
        from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig
        cfg = [EndpointConfig(name="local", url="http://localhost:11434", provider="ollama",
                              roles=["reasoning"], priority=1, max_concurrent=1)]
        r = LLMRouter(ModelsConfig(reasoning="qwen3:14b", embedding="e"),
                      EndpointManager(cfg, LLMConfig()), pii_enabled=False)
        r._gated_generate = AsyncMock(return_value="x")
        return r

    async def test_default_leaves_cache_on(self, router):
        await router.generate("reasoning", "p")
        assert "use_cache" not in router._gated_generate.await_args.args[-1]

    async def test_false_is_forwarded(self, router):
        await router.generate("reasoning", "p", use_cache=False)
        assert router._gated_generate.await_args.args[-1]["use_cache"] is False
