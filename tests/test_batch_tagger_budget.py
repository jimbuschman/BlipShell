"""Tests for BatchTagger time-budget early exit.

Uses real asyncio.sleep (not mocked) to simulate per-batch latency so
the time-budget logic is exercised against actual elapsed time. Per
team rule: never patch asyncio.sleep — mocked time tests don't prove
anything about real-world budget behavior.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from blipshell.memory.batch_tagger import BatchTagger


@dataclass
class _FakeMemoryConfig:
    batch_tag_batch_size: int = 10
    batch_tag_max_batches: int = 500


class _StubBatchTagger(BatchTagger):
    """BatchTagger subclass that replaces tag_batch with a sleeper.

    Each call sleeps `per_batch_seconds` (real time) and returns a fake
    success result. Lets us drive tag_all() at predictable per-batch
    latency without any LLM or DB.
    """

    def __init__(self, per_batch_seconds: float, *, total_available: int = 1000):
        # Bypass parent __init__ — we don't need sqlite/router/config wired
        self.sqlite = MagicMock()
        self.router = MagicMock()
        self.config = _FakeMemoryConfig()
        self.allow_new_tags = False
        self._per_batch_seconds = per_batch_seconds
        self._remaining = total_available
        self.batches_called = 0

    async def tag_batch(self) -> dict:
        await asyncio.sleep(self._per_batch_seconds)
        self.batches_called += 1
        if self._remaining <= 0:
            return {"memories_in_batch": 0, "memories_tagged": 0, "tags_assigned": 0, "error": None}
        consumed = min(self.config.batch_tag_batch_size, self._remaining)
        self._remaining -= consumed
        return {
            "memories_in_batch": consumed,
            "memories_tagged": consumed,
            "tags_assigned": consumed * 2,
            "error": None,
        }


@pytest.mark.asyncio
async def test_tag_all_respects_time_budget():
    """With a tight budget, tag_all stops early before the wall-clock cap."""
    tagger = _StubBatchTagger(per_batch_seconds=0.1, total_available=10_000)

    started = time.monotonic()
    result = await tagger.tag_all(time_budget_seconds=0.5)
    elapsed = time.monotonic() - started

    # Should have stopped early — the remaining_pool is huge, only the budget gates us.
    assert result["stopped_early"] is True
    assert "time budget" in result["stop_reason"].lower()
    # Stayed under the budget — no hard timeout from outside needed.
    assert elapsed < 1.0, f"tag_all overran budget: {elapsed:.2f}s"
    # Did some real work before exiting.
    assert result["batches"] >= 1
    assert result["memories_tagged"] >= 10


@pytest.mark.asyncio
async def test_tag_all_no_budget_runs_through():
    """Without time_budget_seconds, behavior is unchanged from prior code path."""
    tagger = _StubBatchTagger(per_batch_seconds=0.01, total_available=30)

    result = await tagger.tag_all(max_batches=10)

    assert result["stopped_early"] is False
    assert result["stop_reason"] is None
    # Drained the pool naturally.
    assert result["memories_tagged"] == 30
    # Stopped on empty batch, not on max_batches cap.
    assert result["batches"] <= 10


@pytest.mark.asyncio
async def test_tag_all_budget_with_quick_drain():
    """If pool drains before budget elapses, exit normally — not flagged early."""
    tagger = _StubBatchTagger(per_batch_seconds=0.01, total_available=20)

    result = await tagger.tag_all(time_budget_seconds=10.0)

    # Pool drained naturally — stopped_early should remain False.
    assert result["stopped_early"] is False
    assert result["memories_tagged"] == 20
