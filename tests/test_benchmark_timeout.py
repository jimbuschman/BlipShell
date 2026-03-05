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
