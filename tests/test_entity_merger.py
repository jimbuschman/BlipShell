"""Tests for retroactive entity merge time-budget behavior.

The merge pass scans the whole entity graph, which can't finish inside the
nightly per-job timeout. merge_pass() therefore takes a time_budget_seconds:
the scheduled nightly passes one so the job makes resumable partial progress
instead of timing out, while the standalone script passes None for a full pass.

These tests validate the budget gate deterministically (no Ollama): with no
budget the scan covers every entity; with an already-elapsed budget it stops
cleanly at the start, reporting partial stats rather than raising.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory.entity_merger import EntityMerger


def _make_merger(num_entities: int) -> EntityMerger:
    """EntityMerger over `num_entities` mergeable entities with no candidates.

    search_similar_entities returns [] so no merge/LLM work happens — the only
    variable under test is the time-budget loop gate.
    """
    sqlite = MagicMock()
    sqlite.get_mergeable_entities = AsyncMock(return_value=[
        {"id": i, "name": f"entity_{i}", "entity_type": "concept", "mentions": 1}
        for i in range(num_entities)
    ])
    vectors = MagicMock()
    vectors.search_similar_entities = MagicMock(return_value=[])
    router = MagicMock()
    return EntityMerger(sqlite, router, vectors)


class TestMergePassBudget:
    async def test_no_budget_scans_all(self):
        """time_budget_seconds=None scans the full graph, not stopped early."""
        merger = _make_merger(50)
        result = await merger.merge_pass(dry_run=True, time_budget_seconds=None)

        assert result["stopped_early"] is False
        assert result["entities_total"] == 50
        assert result["entities_scanned"] == 50
        assert result["would_merge"] == 0

    async def test_elapsed_budget_stops_early(self):
        """An already-elapsed budget stops the scan cleanly with partial stats."""
        merger = _make_merger(50)
        # Negative budget → deadline is already in the past → break on first check.
        result = await merger.merge_pass(dry_run=True, time_budget_seconds=-1)

        assert result["stopped_early"] is True
        assert result["entities_scanned"] < result["entities_total"]
        # No exception, and the candidate search was never reached.
        merger.vectors.search_similar_entities.assert_not_called()

    async def test_progress_callback_fires(self):
        """on_progress is invoked during the scan with (scanned, total, stats)."""
        merger = _make_merger(10)
        calls = []
        await merger.merge_pass(
            dry_run=True,
            on_progress=lambda scanned, total, stats: calls.append((scanned, total)),
        )
        assert calls, "on_progress should fire at least once"
        assert all(total == 10 for _, total in calls)
