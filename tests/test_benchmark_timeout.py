"""Test that LLMClient.generate() actually times out.

Patches ollama.AsyncClient.chat to sleep forever, verifies the timeout
fires within the expected window. No Ollama server needed.
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from blipshell.llm.client import LLMClient


async def _hang_forever(**kwargs):
    """Simulate a model that never responds."""
    await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_generate_times_out():
    """LLMClient.generate() with a 2s timeout should raise within ~3s."""
    client = LLMClient(host="http://fake:11434", max_retries=0, timeout=2.0)

    with patch.object(client._client, "chat", new=AsyncMock(side_effect=_hang_forever)):
        t0 = time.monotonic()
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await client.generate(prompt="Hello", model="fake-model", use_cache=False)
        elapsed = time.monotonic() - t0

    # Should complete within timeout + small margin (not 600s or forever)
    assert elapsed < 5.0, f"Took {elapsed:.1f}s — timeout didn't fire"


@pytest.mark.asyncio
async def test_generate_times_out_with_think_false():
    """Same test but with think=False kwarg (the benchmark's actual call pattern)."""
    client = LLMClient(host="http://fake:11434", max_retries=0, timeout=2.0)

    with patch.object(client._client, "chat", new=AsyncMock(side_effect=_hang_forever)):
        t0 = time.monotonic()
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await client.generate(
                prompt="Hello", model="fake-model", system="Be brief.",
                use_cache=False, think=False,
            )
        elapsed = time.monotonic() - t0

    assert elapsed < 5.0, f"Took {elapsed:.1f}s — timeout didn't fire"


@pytest.mark.asyncio
async def test_multiple_timeouts_dont_compound():
    """Running 3 timed-out calls should take ~6s total (3 × 2s), not 360s+."""
    client = LLMClient(host="http://fake:11434", max_retries=0, timeout=2.0)

    with patch.object(client._client, "chat", new=AsyncMock(side_effect=_hang_forever)):
        t0 = time.monotonic()
        for _ in range(3):
            try:
                await client.generate(prompt="Hello", model="fake-model", use_cache=False)
            except (TimeoutError, asyncio.TimeoutError):
                pass
        elapsed = time.monotonic() - t0

    # 3 calls × 2s timeout = ~6s, with margin
    assert elapsed < 12.0, f"Took {elapsed:.1f}s — timeouts are compounding"


@pytest.mark.asyncio
async def test_outer_wait_for_cancels_generate():
    """asyncio.wait_for wrapping client.generate() should also work."""
    client = LLMClient(host="http://fake:11434", max_retries=0, timeout=10.0)

    with patch.object(client._client, "chat", new=AsyncMock(side_effect=_hang_forever)):
        t0 = time.monotonic()
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            # Outer timeout (2s) is shorter than client timeout (10s)
            await asyncio.wait_for(
                client.generate(prompt="Hello", model="fake-model", use_cache=False),
                timeout=2.0,
            )
        elapsed = time.monotonic() - t0

    assert elapsed < 5.0, f"Took {elapsed:.1f}s — outer wait_for didn't cancel"


# --- candidate router inherits the REAL config (2026-08-03) -----------------

class TestCandidateInheritsConfig:
    """`benchmark run qwen3:14b` timed out on the reasoning suite because the
    candidate router built EndpointManager with a default LLMConfig() — 120s —
    while config.yaml sets 300 ("5min for slower local models"). A local 14B
    doing plans/analysis exceeded 120s and scored 0 on ability it has.
    """

    def _config(self):
        from blipshell.core.config import ConfigManager
        return ConfigManager("config.yaml").load()

    def test_configured_timeout_reaches_the_client(self):
        from blipshell.benchmark.harness import build_candidate_router
        cfg = self._config()
        assert cfg.llm.timeout > 120.0, "config.yaml should raise this above the default"
        r = build_candidate_router("qwen3:14b", llm_config=cfg.llm)
        client = r._endpoint_manager._endpoints[0].client
        assert client.timeout == cfg.llm.timeout

    def test_omitting_llm_config_still_works(self):
        """Callers that pass nothing keep the library default rather than crash."""
        from blipshell.benchmark.harness import build_candidate_router
        from blipshell.models.config import LLMConfig
        r = build_candidate_router("m")
        assert r._endpoint_manager._endpoints[0].client.timeout == LLMConfig().timeout

    def test_num_ctx_is_sent_not_left_to_ollamas_default(self):
        """context_tokens=None sends no num_ctx at all, so Ollama picks its own
        window and silently truncates long coding/reasoning prompts — corrupting
        scores rather than just slowing them."""
        from blipshell.benchmark.harness import build_candidate_router
        r = build_candidate_router("m", context_tokens=32768)
        assert r._endpoint_manager._endpoints[0].context_tokens == 32768

    def test_context_window_matches_the_configured_local_endpoint(self):
        from blipshell.benchmark.runner import _candidate_context_tokens
        cfg = self._config()
        local = next(e for e in cfg.endpoints if e.name == "local")
        assert _candidate_context_tokens(cfg, local.url) == local.context_tokens

    def test_picks_the_smallest_window_when_endpoints_share_a_url(self):
        """localhost serves both `local` (32K) and `local-cloud` (128K). A 14B
        local model given a 128K num_ctx allocates an enormous KV cache on a GPU
        that has already failed CUDA init under memory pressure."""
        from blipshell.benchmark.runner import _candidate_context_tokens
        cfg = self._config()
        sharing = [e for e in cfg.endpoints
                   if (e.url or "").rstrip("/") == "http://localhost:11434"
                   and e.context_tokens]
        assert len(sharing) > 1, "fixture assumes localhost is shared"
        assert _candidate_context_tokens(cfg, "http://localhost:11434") == \
            min(e.context_tokens for e in sharing)

    def test_unknown_url_returns_none_rather_than_guessing(self):
        from blipshell.benchmark.runner import _candidate_context_tokens
        assert _candidate_context_tokens(self._config(), "http://nope.invalid") is None
