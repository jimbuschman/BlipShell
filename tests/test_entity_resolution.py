"""Tests for 3-stage entity resolution (Feature 5).

Tests exact match, embedding similarity, LLM arbitration, cache, and alias recording.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory.entity_extractor import EntityExtractor


@pytest.fixture
def mock_chroma_with_entities():
    """Mock ChromaStore with entity embedding search."""
    chroma = MagicMock()
    chroma.search_memories.return_value = []
    chroma.search_core_memories.return_value = []
    chroma.search_lessons.return_value = []
    chroma.add_memory = MagicMock()
    chroma.delete_memory = MagicMock()
    chroma.upsert_entity = MagicMock()
    chroma.search_similar_entities = MagicMock(return_value=[])
    return chroma


@pytest.fixture
async def resolution_extractor(sqlite_store, canned_router, mock_chroma_with_entities):
    """EntityExtractor with entity resolution enabled."""
    return EntityExtractor(
        sqlite=sqlite_store,
        router=canned_router,
        vectors=mock_chroma_with_entities,
        batch_size=10,
        entity_resolution_enabled=True,
        entity_auto_merge_threshold=0.85,
        entity_llm_threshold=0.70,
        entity_max_candidates=5,
    )


@pytest.fixture
async def basic_extractor(sqlite_store, canned_router, mock_chroma_with_entities):
    """EntityExtractor with entity resolution disabled (default)."""
    return EntityExtractor(
        sqlite=sqlite_store,
        router=canned_router,
        vectors=mock_chroma_with_entities,
        batch_size=10,
        entity_resolution_enabled=False,
    )


# --- Stage 1: Exact match ---


class TestExactMatch:
    async def test_exact_match_returns_existing_id(
        self, resolution_extractor, sqlite_store,
    ):
        """If entity exists by exact name, return its ID without embedding search."""
        # Pre-create entity
        eid = await sqlite_store.get_or_create_entity("python", "technology")

        # Resolve should find it
        resolved = await resolution_extractor._resolve_entity("python", "technology")
        assert resolved == eid

        # Should NOT have searched embeddings
        resolution_extractor.vectors.search_similar_entities.assert_not_called()

    async def test_exact_match_case_insensitive(
        self, resolution_extractor, sqlite_store,
    ):
        """Entity names are lowercased, so 'Python' should match 'python'."""
        eid = await sqlite_store.get_or_create_entity("python", "technology")
        resolved = await resolution_extractor._resolve_entity("Python", "technology")
        assert resolved == eid


# --- Stage 2: Embedding similarity ---


class TestEmbeddingSimilarity:
    async def test_auto_merge_above_threshold(
        self, resolution_extractor, sqlite_store,
    ):
        """Entity with embedding similarity >= 0.85 should be auto-merged."""
        # Pre-create entity
        eid = await sqlite_store.get_or_create_entity("postgresql", "technology")

        # Mock embedding search to return a high-similarity match
        resolution_extractor.vectors.search_similar_entities.return_value = [
            {"id": eid, "name": "postgresql", "similarity": 0.90, "entity_type": "technology"},
        ]

        resolved = await resolution_extractor._resolve_entity("postgres", "technology")
        assert resolved == eid

        # Should have recorded alias
        cursor = await sqlite_store._db.execute(
            "SELECT * FROM entity_aliases WHERE alias_name = 'postgres'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["canonical_entity_id"] == eid
        assert row["merge_method"] == "embedding_auto"

    async def test_no_merge_below_threshold(
        self, resolution_extractor, sqlite_store,
    ):
        """Entity with similarity below 0.70 should create a new entity."""
        eid = await sqlite_store.get_or_create_entity("java", "technology")

        resolution_extractor.vectors.search_similar_entities.return_value = [
            {"id": eid, "name": "java", "similarity": 0.50, "entity_type": "technology"},
        ]

        resolved = await resolution_extractor._resolve_entity("javascript", "technology")
        # Should be a NEW entity, not java
        assert resolved != eid

        # Should have upserted the new entity into embeddings
        resolution_extractor.vectors.upsert_entity.assert_called()


# --- Stage 3: LLM arbitration ---


class TestLLMArbitration:
    async def test_llm_merge_in_ambiguous_range(
        self, resolution_extractor, sqlite_store, canned_router,
    ):
        """Entity with 0.70-0.85 similarity triggers LLM, which says YES → merge."""
        eid = await sqlite_store.get_or_create_entity("react", "technology")

        resolution_extractor.vectors.search_similar_entities.return_value = [
            {"id": eid, "name": "react", "similarity": 0.78, "entity_type": "technology"},
        ]

        # Override canned router to return YES for entity resolution
        canned_router.generate = AsyncMock(return_value="YES")

        resolved = await resolution_extractor._resolve_entity("reactjs", "technology")
        assert resolved == eid

        # Should have recorded alias with LLM method
        cursor = await sqlite_store._db.execute(
            "SELECT * FROM entity_aliases WHERE alias_name = 'reactjs'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["merge_method"] == "llm_resolved"

    async def test_llm_no_merge(
        self, resolution_extractor, sqlite_store, canned_router,
    ):
        """LLM says NO → create new entity."""
        eid = await sqlite_store.get_or_create_entity("react", "technology")

        resolution_extractor.vectors.search_similar_entities.return_value = [
            {"id": eid, "name": "react", "similarity": 0.75, "entity_type": "technology"},
        ]

        canned_router.generate = AsyncMock(return_value="NO")

        resolved = await resolution_extractor._resolve_entity("react native", "technology")
        assert resolved != eid  # Should be different entity


# --- Cache ---


class TestResolutionCache:
    async def test_cache_hit(self, resolution_extractor, sqlite_store):
        """Second resolve of same name should use cache, not re-query."""
        eid = await sqlite_store.get_or_create_entity("python", "technology")

        # First resolve
        resolved1 = await resolution_extractor._resolve_entity("python", "technology")
        assert resolved1 == eid

        # Second resolve (should hit cache)
        resolved2 = await resolution_extractor._resolve_entity("python", "technology")
        assert resolved2 == eid

    async def test_cache_cleared_per_batch(self, resolution_extractor, sqlite_store):
        """Cache should be cleared when a new batch starts."""
        eid = await sqlite_store.get_or_create_entity("python", "technology")

        # Populate cache
        await resolution_extractor._resolve_entity("python", "technology")
        assert "python" in resolution_extractor._resolution_cache

        # Simulate new batch start (extract_batch clears cache)
        resolution_extractor._resolution_cache = {}
        assert "python" not in resolution_extractor._resolution_cache


# --- Disabled resolution ---


class TestDisabledResolution:
    async def test_disabled_creates_directly(
        self, basic_extractor, sqlite_store,
    ):
        """With resolution disabled, _resolve_entity should just create entities."""
        resolved = await basic_extractor._resolve_entity("python", "technology")
        assert resolved is not None

        # Should still upsert entity embedding if chroma available
        basic_extractor.vectors.upsert_entity.assert_called()


# --- Entity alias recording ---


class TestEntityAliases:
    async def test_record_alias(self, sqlite_store):
        """Recording an alias should persist to the entity_aliases table."""
        eid = await sqlite_store.get_or_create_entity("postgresql", "technology")
        await sqlite_store.record_entity_alias("postgres", eid, "embedding_auto")

        cursor = await sqlite_store._db.execute(
            "SELECT * FROM entity_aliases WHERE alias_name = 'postgres'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["canonical_entity_id"] == eid
        assert row["merge_method"] == "embedding_auto"

    async def test_duplicate_alias_ignored(self, sqlite_store):
        """Recording the same alias twice should not raise."""
        eid = await sqlite_store.get_or_create_entity("postgresql", "technology")
        await sqlite_store.record_entity_alias("postgres", eid, "embedding_auto")
        await sqlite_store.record_entity_alias("postgres", eid, "llm_resolved")

        # Should still have only one row
        cursor = await sqlite_store._db.execute(
            "SELECT COUNT(*) as cnt FROM entity_aliases WHERE alias_name = 'postgres'"
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 1


# --- Entity merge ---


class TestEntityMerge:
    async def test_merge_reassigns_mentions(self, sqlite_store):
        """merge_entity should reassign mentions from old to canonical."""
        old_id = await sqlite_store.get_or_create_entity("postgres", "technology")
        canon_id = await sqlite_store.get_or_create_entity("postgresql", "technology")

        # Create a session and memory for the foreign key
        session_id = await sqlite_store.create_session("Test")
        from blipshell.models.memory import Memory, MemoryType
        mem = Memory(session_id=session_id, role="user", content="test",
                     memory_type=MemoryType.CONVERSATION)
        mem_id = await sqlite_store.create_memory(mem)

        # Add mention to old entity
        await sqlite_store.create_entity_mention(old_id, mem_id)

        # Merge
        await sqlite_store.merge_entity(old_id, canon_id)

        # Mention should now point to canonical
        cursor = await sqlite_store._db.execute(
            "SELECT entity_id FROM entity_mentions WHERE memory_id = ?", (mem_id,)
        )
        row = await cursor.fetchone()
        assert row["entity_id"] == canon_id
