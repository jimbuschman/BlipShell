"""Tests for semantic memory search (memory/search.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from blipshell.memory.search import MemorySearch, SearchResult
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory, MemoryType


@pytest.fixture
def search_config():
    return MemoryConfig(
        similarity_threshold=0.5,
        importance_boost_weight=0.2,
        search_overfetch_multiplier=2,
        min_rank_threshold=3,
        recall_search_limit=10,
    )


@pytest.fixture
def memory_search(sqlite_store, mock_chroma, mock_router, search_config):
    return MemorySearch(
        sqlite=sqlite_store,
        chroma=mock_chroma,
        router=mock_router,
        config=search_config,
    )


class TestMemorySearch:
    async def test_noise_query_returns_empty(self, memory_search):
        result = await memory_search.search("hi")
        assert result == []

    async def test_empty_chroma_returns_empty(self, memory_search):
        memory_search.chroma.search_memories.return_value = []
        result = await memory_search.search("tell me about the project architecture")
        assert result == []

    async def test_low_similarity_filtered(self, memory_search, sqlite_store):
        # Create a memory in SQLite
        mem = Memory(session_id=1, role="user", content="test content",
                     summary="test summary", rank=4, importance=0.8)
        mid = await sqlite_store.create_memory(mem)

        memory_search.chroma.search_memories.return_value = [
            {"id": mid, "similarity": 0.3, "metadata": {}},  # below threshold
        ]
        result = await memory_search.search("tell me about the project architecture")
        assert len(result) == 0

    async def test_low_rank_filtered(self, memory_search, sqlite_store):
        mem = Memory(session_id=1, role="user", content="test content",
                     summary="test summary", rank=1, importance=0.8)
        mid = await sqlite_store.create_memory(mem)

        memory_search.chroma.search_memories.return_value = [
            {"id": mid, "similarity": 0.8, "metadata": {}},
        ]
        result = await memory_search.search("tell me about the project architecture")
        assert len(result) == 0  # rank 1 < min_rank 3

    async def test_successful_search_with_boost(self, memory_search, sqlite_store):
        mem = Memory(session_id=1, role="user", content="architecture discussion",
                     summary="We discussed microservice architecture", rank=5, importance=0.9)
        mid = await sqlite_store.create_memory(mem)

        memory_search.chroma.search_memories.return_value = [
            {"id": mid, "similarity": 0.85, "metadata": {}},
        ]
        result = await memory_search.search("tell me about the project architecture")
        assert len(result) == 1
        assert result[0].rank == 5
        # Boost: (5-1)/4 * 0.2 = 0.2 → boosted = 0.85 + 0.2 = 1.05
        assert result[0].boosted_score == pytest.approx(1.05)

    async def test_current_session_excluded(self, memory_search, sqlite_store):
        mem = Memory(session_id=42, role="user", content="test",
                     summary="test", rank=5, importance=0.9)
        mid = await sqlite_store.create_memory(mem)

        memory_search.chroma.search_memories.return_value = [
            {"id": mid, "similarity": 0.9, "metadata": {"session_id": "42"}},
        ]
        result = await memory_search.search(
            "tell me about the project architecture", current_session_id=42,
        )
        assert len(result) == 0

    async def test_results_sorted_by_boosted_score(self, memory_search, sqlite_store):
        mem1 = Memory(session_id=1, role="user", content="low rank",
                      summary="low", rank=3, importance=0.5)
        mid1 = await sqlite_store.create_memory(mem1)

        mem2 = Memory(session_id=1, role="user", content="high rank",
                      summary="high", rank=5, importance=0.9)
        mid2 = await sqlite_store.create_memory(mem2)

        memory_search.chroma.search_memories.return_value = [
            {"id": mid1, "similarity": 0.8, "metadata": {}},
            {"id": mid2, "similarity": 0.7, "metadata": {}},
        ]
        result = await memory_search.search("architecture discussion topic details")
        assert len(result) == 2
        # mid2 has higher boost despite lower similarity
        assert result[0].rank == 5

    async def test_search_lessons_delegates(self, memory_search):
        memory_search.chroma.search_lessons.return_value = [
            {"id": 1, "document": "lesson text", "similarity": 0.9, "metadata": {}},
        ]
        result = await memory_search.search_lessons("test query")
        assert len(result) == 1

    async def test_config_values_applied(self, search_config):
        ms = MemorySearch(
            sqlite=MagicMock(),
            chroma=MagicMock(),
            router=MagicMock(),
            config=search_config,
        )
        assert ms.similarity_threshold == 0.5
        assert ms.importance_boost_weight == 0.2
        assert ms.search_overfetch_multiplier == 2
        assert ms.min_rank == 3
