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
    """Mock VectorStore (kept as mock_chroma for test compatibility)."""
    vectors = MagicMock()
    vectors.search_memories.return_value = []
    vectors.search_core_memories.return_value = []
    vectors.search_lessons.return_value = []
    vectors.add_memory = MagicMock()
    vectors.add_core_memory = MagicMock()
    vectors.add_lesson = MagicMock()
    vectors.delete_memory = MagicMock()
    vectors.delete_lesson = MagicMock()
    vectors.get_counts.return_value = {"memories": 0, "core_memories": 0, "lessons": 0, "entities": 0}
    return vectors


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


# --- Canned response fixtures for integration tests ---

_CANNED_RESPONSES = {
    "summarization": "User discussed Python performance tuning and optimization strategies.",
    "ranking_importance": "3 0.5",
    "reasoning": "When discussing performance, focus on profiling before optimizing.",
    "entity_extraction": (
        "user | discussed | python | person | technology\n"
        "python | is_a | programming_language | technology | concept"
    ),
}


def _canned_generate(task_type, prompt="", system=None, think=None, **kwargs):
    """Side-effect function for canned_router.generate()."""
    if task_type == "summarization":
        return _CANNED_RESPONSES["summarization"]
    if task_type == "ranking_importance":
        return _CANNED_RESPONSES["ranking_importance"]
    if task_type == "session_review":
        return (
            "When discussing performance topics, profile before optimizing "
            "and show concrete benchmark numbers rather than vague claims."
        )
    if task_type == "reasoning":
        # Detect entity extraction by checking prompt/system content
        if system and "triple" in system.lower():
            return _CANNED_RESPONSES["entity_extraction"]
        # Detect tag discovery
        if system and "tag" in system.lower() and "pattern" in system.lower():
            return "docker: \\bdocker\\b|\\bcontainer\\b"
        # Detect contradiction check
        if system and "contradict" in system.lower():
            return "NO"
        # Default reasoning (lesson extraction, etc.)
        return _CANNED_RESPONSES["reasoning"]
    return "test response"


@pytest.fixture
def canned_router():
    """Mock LLMRouter with realistic canned responses per task type.

    Uses side_effect to return different responses based on the task_type
    argument, enabling integration tests without Ollama.
    """
    router = MagicMock()
    router.generate = AsyncMock(side_effect=_canned_generate)
    router.get_model.return_value = "test-model"
    router.get_fallback_model.return_value = None
    router.get_client = AsyncMock(return_value=MagicMock())
    router.get_model_and_client = AsyncMock(
        return_value=("test-model", MagicMock()),
    )
    return router


@pytest.fixture
async def memory_processor(sqlite_store, mock_chroma, canned_router, memory_config):
    """MemoryProcessor wired to real SQLite + mocked Chroma/LLM."""
    from blipshell.memory.processor import MemoryProcessor
    return MemoryProcessor(
        sqlite=sqlite_store,
        vectors=mock_chroma,
        router=canned_router,
        config=memory_config,
    )


@pytest.fixture
async def entity_extractor(sqlite_store, canned_router):
    """EntityExtractor wired to real SQLite + mocked LLM."""
    from blipshell.memory.entity_extractor import EntityExtractor
    return EntityExtractor(
        sqlite=sqlite_store,
        router=canned_router,
        batch_size=10,
    )
