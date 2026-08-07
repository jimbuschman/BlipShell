"""Four entity-pipeline defects from the 2026-08-04 deep dive.

Two of them were live on every turn: a full entity-table read on the search
path, and an auto-merge threshold with no version guard — the guard existed
only on EntityMerger, which ships disabled, so the enabled path had none.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory.entity_extractor import EntityExtractor
from blipshell.memory.entity_merger import EntityMerger
from blipshell.memory.entity_names import (
    normalize_name, numeric_tokens, version_distinguished,
)
from blipshell.memory.sqlite_store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "entities.db"))
    await s.initialize()
    yield s
    await s.close()


# --- the shared name rules -------------------------------------------------


class TestVersionRules:
    @pytest.mark.parametrize("a,b", [
        ("projectecho_v1", "projectecho_v2"),
        ("corememorybackup1", "corememorybackup2"),
        ("llama 3.2b", "llama 3.2"),
        ("ws2812", "ws2812b"),
    ])
    def test_version_variants_are_distinguished(self, a, b):
        assert version_distinguished(a, b) is True

    @pytest.mark.parametrize("a,b", [
        ("langchain-x", "langchain_x"),
        ("deepseek-r1-7b", "deepseek-r1:7b"),
        ("chat gpt", "chatgpt"),
    ])
    def test_formatting_variants_are_not(self, a, b):
        assert version_distinguished(a, b) is False

    def test_merger_still_uses_the_same_rules(self):
        """Extracting these to a module must not change merger behaviour."""
        assert EntityMerger._version_distinguished("projectecho_v1", "projectecho_v2")
        assert not EntityMerger._version_distinguished("langchain-x", "langchain_x")
        assert EntityMerger._normalize_name("chat gpt") == "chatgpt"
        assert EntityMerger._numeric_tokens("llama 3.2b") == numeric_tokens("llama 3.2b")


# --- the guard on the LIVE path --------------------------------------------


def _extractor(store, vectors):
    return EntityExtractor(
        store, router=MagicMock(), vectors=vectors,
        entity_resolution_enabled=True,
        entity_auto_merge_threshold=0.85,
    )


class TestCreationTimeVersionGuard:
    """entity_resolution is ENABLED in config while entity_merge is disabled,
    so this is the path that actually runs against the graph — and it had no
    version guard at all."""

    async def test_version_variant_is_not_auto_merged(self, store):
        existing = await store.get_or_create_entity("projectecho_v1", "project")

        vectors = MagicMock()
        vectors.search_similar_entities.return_value = [
            {"id": existing, "name": "projectecho_v1",
             "similarity": 0.996, "entity_type": "project"},
        ]
        ex = _extractor(store, vectors)

        resolved = await ex._resolve_entity("projectecho_v2", "project")

        assert resolved != existing, (
            "v1 and v2 were collapsed into one entity at 0.996 similarity"
        )
        assert await store.get_entity_id_by_name("projectecho_v2", "project")

    async def test_a_blocked_candidate_does_not_abandon_the_rest(self, store):
        """Candidates arrive sorted by similarity, so a version variant often
        outranks the genuine match. Blocking one must skip it, not stop."""
        v1 = await store.get_or_create_entity("projectecho_v1", "project")
        spaced = await store.get_or_create_entity("projectecho v2", "project")

        vectors = MagicMock()
        vectors.search_similar_entities.return_value = [
            {"id": v1, "name": "projectecho_v1",
             "similarity": 0.996, "entity_type": "project"},
            {"id": spaced, "name": "projectecho v2",
             "similarity": 0.990, "entity_type": "project"},
        ]
        ex = _extractor(store, vectors)

        resolved = await ex._resolve_entity("projectecho_v2", "project")

        assert resolved == spaced, (
            "the real match was never reached — the v1 candidate ended the loop"
        )

    async def test_a_newly_created_entity_gets_an_embedding(self, store):
        """Skipping every candidate must still reach the creation path that
        upserts the vector; an unembedded entity is invisible to all later
        resolution."""
        v1 = await store.get_or_create_entity("projectecho_v1", "project")

        vectors = MagicMock()
        vectors.search_similar_entities.return_value = [
            {"id": v1, "name": "projectecho_v1",
             "similarity": 0.996, "entity_type": "project"},
        ]
        ex = _extractor(store, vectors)

        resolved = await ex._resolve_entity("projectecho_v2", "project")

        vectors.upsert_entity.assert_called_once_with(
            resolved, "projectecho_v2", "project",
        )

    async def test_formatting_variant_still_merges(self, store):
        """The guard must not block genuine duplicates."""
        existing = await store.get_or_create_entity("langchain-x", "technology")

        vectors = MagicMock()
        vectors.search_similar_entities.return_value = [
            {"id": existing, "name": "langchain-x",
             "similarity": 0.97, "entity_type": "technology"},
        ]
        ex = _extractor(store, vectors)

        resolved = await ex._resolve_entity("langchain_x", "technology")
        assert resolved == existing


# --- typed lookup ----------------------------------------------------------


class TestTypedEntityLookup:
    """The table's uniqueness is (name, entity_type), but the lookup matched
    on name alone and took fetchone() — so a technology-typed triple could
    bind to a concept-typed entity of the same name, nondeterministically."""

    async def test_same_name_different_type_are_separate(self, store):
        as_tech = await store.get_or_create_entity("mercury", "technology")
        as_place = await store.get_or_create_entity("mercury", "place")
        assert as_tech != as_place

        assert await store.get_entity_id_by_name("mercury", "technology") == as_tech
        assert await store.get_entity_id_by_name("mercury", "place") == as_place

    async def test_untyped_lookup_is_deterministic(self, store):
        """Callers without a type still get a stable answer, not a coin flip."""
        first = await store.get_or_create_entity("mercury", "technology")
        await store.get_or_create_entity("mercury", "place")

        seen = {await store.get_entity_id_by_name("mercury") for _ in range(5)}
        assert seen == {first}

    async def test_missing_type_returns_none(self, store):
        await store.get_or_create_entity("mercury", "technology")
        assert await store.get_entity_id_by_name("mercury", "organization") is None


# --- extraction failures ---------------------------------------------------


class TestFailedExtractionIsRetried:
    """Marking a failed extraction as done meant its triples were lost from
    the graph silently and permanently, with no error column to find it by."""

    async def test_a_failed_memory_stays_unextracted(self, store):
        from blipshell.models.memory import Memory

        sid = await store.create_session("s")
        mid = await store.create_memory(Memory(
            session_id=sid, role="user",
            content="user works at acme", summary="user works at acme",
        ))

        router = MagicMock()
        router.generate = AsyncMock(side_effect=RuntimeError("LLM died"))
        ex = EntityExtractor(store, router=router, vectors=None)

        stats = await ex.extract_batch()

        assert stats["errors"] == 1
        assert stats["retryable"] == 1
        assert mid in await store.get_unextracted_memory_ids(limit=10), (
            "a failed extraction was marked done and will never be retried"
        )

    async def test_a_summaryless_memory_is_marked_not_retried(self, store):
        """That one is a permanent skip, not a failure — mark it so it stops
        occupying a slot in every batch."""
        from blipshell.models.memory import Memory

        sid = await store.create_session("s")
        mid = await store.create_memory(Memory(
            session_id=sid, role="user", content="x", summary=None,
        ))
        ex = EntityExtractor(store, router=MagicMock(), vectors=None)

        await ex._extract_one(mid)

        assert mid not in await store.get_unextracted_memory_ids(limit=10)


# --- the search-path cache -------------------------------------------------


class TestEntityNameCache:
    """get_all_entity_names ran on EVERY search query — a full table read plus
    serialization, on the interactive path."""

    async def test_repeated_calls_hit_the_cache(self, store):
        await store.get_or_create_entity("acme", "organization")

        first = await store.get_all_entity_names()
        second = await store.get_all_entity_names()
        assert first is second, "the entity list was re-read from SQLite"

    async def test_creating_an_entity_invalidates_it(self, store):
        await store.get_or_create_entity("acme", "organization")
        await store.get_all_entity_names()

        await store.get_or_create_entity("globex", "organization")
        names = await store.get_all_entity_names()

        assert "globex" in names, "a new entity was invisible to search"

    async def test_archiving_invalidates_it(self, store):
        eid = await store.get_or_create_entity("acme", "organization")
        assert "acme" in await store.get_all_entity_names()

        await store.archive_entities([eid])

        assert "acme" not in await store.get_all_entity_names(), (
            "an archived entity kept expanding searches"
        )

    async def test_resolving_an_existing_entity_keeps_the_cache(self, store):
        """get_or_create_entity is mostly a read, and extraction calls it for
        every triple — invalidating on the read path makes the cache useless."""
        await store.get_or_create_entity("acme", "organization")
        cached = await store.get_all_entity_names()

        await store.get_or_create_entity("acme", "organization")

        assert await store.get_all_entity_names() is cached

    async def test_reviving_an_active_entity_keeps_the_cache(self, store):
        """create_entity_relationship revives both endpoints of every triple;
        almost none are actually archived."""
        eid = await store.get_or_create_entity("acme", "organization")
        cached = await store.get_all_entity_names()

        assert await store.revive_entities([eid]) == 0
        assert await store.get_all_entity_names() is cached

    async def test_revive_invalidates_it(self, store):
        eid = await store.get_or_create_entity("acme", "organization")
        await store.archive_entities([eid])
        await store.get_all_entity_names()

        await store.revive_entities([eid])

        assert "acme" in await store.get_all_entity_names()
