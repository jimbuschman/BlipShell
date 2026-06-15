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


class TestVersionGuard:
    """Names differing only by a number/version are distinct, never duplicates."""

    @pytest.mark.parametrize("a,b", [
        ("projectecho_v1", "projectecho_v2"),       # versions
        ("corememorybackup1", "corememorybackup2"),  # instances
        ("llama 3.2b", "llama 3.2"),                 # variant suffix
        ("ws2812 leds", "ws2812b leds"),             # chip revision
        ("gpt-4", "gpt-4o"),                         # distinct models
    ])
    def test_version_distinguished_blocks(self, a, b):
        assert EntityMerger._version_distinguished(a, b) is True

    @pytest.mark.parametrize("a,b", [
        ("langchain-huggingface", "langchain_huggingface"),  # separator only
        ("deepseek-r1-7b", "deepseek-r1:7b"),                # same numbers
        ("small cell lung cancer", "small-cell lung cancer"),
        ("emotion topography map", "emotion topography maps"),  # plural
    ])
    def test_formatting_variants_allowed(self, a, b):
        assert EntityMerger._version_distinguished(a, b) is False

    async def test_scan_skips_version_pairs(self):
        """A high-similarity v1/v2 pair is counted as a version_skip, not merged."""
        sqlite = MagicMock()
        sqlite.get_mergeable_entities = AsyncMock(return_value=[
            {"id": 1, "name": "projectecho_v1", "entity_type": "project", "mentions": 5},
            {"id": 2, "name": "projectecho_v2", "entity_type": "project", "mentions": 3},
        ])
        vectors = MagicMock()
        # id=1's nearest neighbor is id=2 at 0.99 (would auto-merge without guard)
        vectors.search_similar_entities = MagicMock(return_value=[
            {"id": 2, "similarity": 0.99},
        ])
        merger = EntityMerger(MagicMock(), MagicMock(), vectors)
        merger.sqlite = sqlite
        result = await merger.merge_pass(dry_run=True)

        assert result["would_merge"] == 0
        assert result["auto_merges"] == 0
        assert result["version_skips"] == 1


class TestApplyPlan:
    async def test_applies_and_blocks_version_pairs(self):
        sqlite = MagicMock()
        sqlite.get_mergeable_entities = AsyncMock(return_value=[
            {"id": 1}, {"id": 2}, {"id": 10}, {"id": 11},
        ])
        sqlite.merge_entity = AsyncMock()
        sqlite.record_entity_alias = AsyncMock()
        sqlite.archive_entities = AsyncMock()
        vectors = MagicMock()
        vectors.delete_entity = MagicMock()
        merger = EntityMerger(sqlite, MagicMock(), vectors)

        plan = [
            # legitimate formatting merge → applied
            {"drop_id": 2, "drop_name": "langchain_chroma", "keep_id": 1,
             "keep_name": "langchain-chroma", "method": "retroactive_embedding"},
            # version pair → blocked by guard even though it's in the file
            {"drop_id": 11, "drop_name": "projectecho_v2", "keep_id": 10,
             "keep_name": "projectecho_v1", "method": "retroactive_embedding"},
        ]
        result = await merger.apply_plan(plan)

        assert result["merged"] == 1
        assert result["guard_skipped"] == 1
        sqlite.merge_entity.assert_awaited_once_with(2, 1)

    async def test_idempotent_skips_already_archived(self):
        """Losers no longer in the live set (already merged) are skipped."""
        sqlite = MagicMock()
        # id=2 is NOT in the live set → treated as already merged away
        sqlite.get_mergeable_entities = AsyncMock(return_value=[{"id": 1}])
        sqlite.merge_entity = AsyncMock()
        sqlite.record_entity_alias = AsyncMock()
        sqlite.archive_entities = AsyncMock()
        merger = EntityMerger(sqlite, MagicMock(), MagicMock())

        plan = [{"drop_id": 2, "drop_name": "foo", "keep_id": 1,
                 "keep_name": "foo", "method": "retroactive_embedding"}]
        result = await merger.apply_plan(plan)

        assert result["merged"] == 0
        assert result["skipped"] == 1
        sqlite.merge_entity.assert_not_awaited()


class TestNormalizeName:
    @pytest.mark.parametrize("a,b", [
        ("chat gpt", "chatgpt"),
        ("self-reflection", "self_reflection"),
        ("phi-3", "phi3"),
        ("esp32-s3", "esp32 s3"),
        ("llama 3.2", "llama3.2"),
    ])
    def test_variants_normalize_equal(self, a, b):
        assert EntityMerger._normalize_name(a) == EntityMerger._normalize_name(b)

    @pytest.mark.parametrize("a,b", [
        ("c#", "c++"),                       # symbols kept
        ("projectecho_v1", "projectecho_v2"),  # different numbers
        ("gemini", "gemma"),
    ])
    def test_distinct_normalize_differently(self, a, b):
        assert EntityMerger._normalize_name(a) != EntityMerger._normalize_name(b)


class TestLexicalMergePass:
    def _merger(self, entities):
        sqlite = MagicMock()
        sqlite.get_mergeable_entities = AsyncMock(return_value=entities)
        sqlite.merge_entity = AsyncMock()
        sqlite.record_entity_alias = AsyncMock()
        sqlite.archive_entities = AsyncMock()
        vectors = MagicMock()
        vectors.delete_entity = MagicMock()
        return EntityMerger(sqlite, MagicMock(), vectors)

    async def test_dry_run_groups_variants(self):
        merger = self._merger([
            {"id": 1, "name": "chatgpt", "entity_type": "technology", "mentions": 858},
            {"id": 2, "name": "chat gpt", "entity_type": "technology", "mentions": 13},
            {"id": 3, "name": "python", "entity_type": "technology", "mentions": 800},
        ])
        result = await merger.lexical_merge_pass(dry_run=True)
        assert result["would_merge"] == 1
        p = result["plan"][0]
        assert p["drop_id"] == 2 and p["keep_id"] == 1  # keeper = more mentions
        merger.sqlite.merge_entity.assert_not_awaited()

    async def test_apply_merges_phi_variant_despite_version_guard(self):
        """phi-3/phi3 trip the numeric guard but share a normalized name → merge."""
        merger = self._merger([
            {"id": 10, "name": "phi3", "entity_type": "technology", "mentions": 15},
            {"id": 11, "name": "phi-3", "entity_type": "technology", "mentions": 16},
        ])
        result = await merger.lexical_merge_pass(dry_run=False)
        assert result["merged"] == 1
        # keeper is phi-3 (16 > 15); phi3 merged into it
        merger.sqlite.merge_entity.assert_awaited_once_with(10, 11)

    async def test_does_not_merge_across_types(self):
        merger = self._merger([
            {"id": 1, "name": "memory", "entity_type": "concept", "mentions": 300},
            {"id": 2, "name": "memory", "entity_type": "technology", "mentions": 50},
        ])
        result = await merger.lexical_merge_pass(dry_run=True)
        assert result["would_merge"] == 0


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
