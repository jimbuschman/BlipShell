"""Shared test fixtures for BlipShell tests."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blipshell.memory.manager import MemoryManager, Pool, PoolItem
from blipshell.models.config import (
    BlipShellConfig,
    EndpointConfig,
    LLMConfig,
    MemoryConfig,
    MemoryPoolsConfig,
    ModelsConfig,
    PoolConfig,
)


@pytest.fixture
def memory_config():
    """Default memory config for tests."""
    return MemoryConfig(
        total_context_tokens=4096,
        system_prompt_reserve=256,
        overflow_batch_size=2,
        recall_search_limit=10,
        min_rank_threshold=3,
        similarity_threshold=0.5,
        importance_boost_weight=0.2,
        search_overfetch_multiplier=2,
    )


@pytest.fixture
def memory_manager(memory_config):
    """A MemoryManager with small test budgets."""
    return MemoryManager(memory_config)


@pytest.fixture
def blipshell_config():
    """Full BlipShell config for integration-style tests."""
    return BlipShellConfig(
        models=ModelsConfig(reasoning="test-model", summarization="test-model"),
        endpoints=[
            EndpointConfig(name="test", url="http://localhost:11434", roles=["reasoning"]),
        ],
    )


@pytest.fixture
def temp_db_path():
    """Temporary SQLite DB path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
async def sqlite_store(temp_db_path):
    """Initialized SQLiteStore with temp DB."""
    from blipshell.memory.sqlite_store import SQLiteStore
    store = SQLiteStore(temp_db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def mock_chroma():
    """Mock ChromaStore."""
    chroma = MagicMock()
    chroma.search_memories.return_value = []
    chroma.search_core_memories.return_value = []
    chroma.search_lessons.return_value = []
    chroma.add_memory = MagicMock()
    chroma.add_core_memory = MagicMock()
    chroma.add_lesson = MagicMock()
    chroma.delete_memory = MagicMock()
    chroma.delete_lesson = MagicMock()
    chroma.get_counts.return_value = {"memories": 0, "core_memories": 0, "lessons": 0}
    return chroma


@pytest.fixture
def mock_router():
    """Mock LLMRouter."""
    router = MagicMock()
    router.generate = AsyncMock(return_value="test summary")
    router.get_model.return_value = "test-model"
    router.get_client = AsyncMock(return_value=MagicMock())
    return router


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient."""
    client = MagicMock()
    client.chat = AsyncMock(return_value={
        "message": {"content": "test response", "tool_calls": None},
    })
    client.generate = AsyncMock(return_value="test result")
    client.check_health = AsyncMock(return_value=True)
    return client
