"""Tests for 'loud absence' — making an empty memory search visible to the model.

When the recall search returns nothing, a silently empty Recall pool invites the
model to fill the gap with invented "remembered" specifics (the confabulation
failure mode). _search_relevant_memories now injects an explicit absence marker
into the Recall pool so the model can say "I don't have that" instead.

These exercise the real method against a real MemoryManager with the search
layer stubbed to control hit counts — the absence logic depends only on whether
the search returned hits, which is exactly what's stubbed.
"""

import types
from unittest.mock import AsyncMock

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.memory.manager import MemoryManager
from blipshell.models.config import MemoryConfig

ABSENCE_MARKER = "No past-conversation memories matched"


@pytest.fixture
def memory_manager():
    return MemoryManager(MemoryConfig(total_context_tokens=4096))


def _fake_self(memory_manager, mem_results):
    """Minimal stand-in exposing only what _search_relevant_memories touches."""
    search = types.SimpleNamespace(
        search=AsyncMock(return_value=mem_results),
        search_core_memories=AsyncMock(return_value=[]),
        search_lessons=AsyncMock(return_value=[]),
        last_search_stats={},
    )
    return types.SimpleNamespace(
        search=search,
        memory_manager=memory_manager,
        active_project=None,
        session_manager=types.SimpleNamespace(session_id=1),
        _log_event=AsyncMock(),
        _search_self_thoughts=AsyncMock(return_value=None),
    )


def _recall_texts(memory_manager):
    pool = memory_manager.get_pool("Recall")
    return [e.text for e in pool.get_top_entries(100_000)]


def _search_result(text):
    return types.SimpleNamespace(
        text=text, summary=text, timestamp=None, boosted_score=0.9,
    )


@pytest.mark.asyncio
async def test_empty_search_injects_absence_marker(memory_manager):
    fake = _fake_self(memory_manager, mem_results=[])

    await ChatMixin._search_relevant_memories(fake, "where do I work")

    texts = _recall_texts(memory_manager)
    assert any(ABSENCE_MARKER in t for t in texts), texts


@pytest.mark.asyncio
async def test_hits_do_not_inject_absence_marker(memory_manager):
    fake = _fake_self(memory_manager, mem_results=[_search_result("You work at Xanatek.")])

    await ChatMixin._search_relevant_memories(fake, "where do I work")

    texts = _recall_texts(memory_manager)
    assert not any(ABSENCE_MARKER in t for t in texts), texts
    assert any("Xanatek" in t for t in texts), texts


@pytest.mark.asyncio
async def test_short_query_does_not_inject_marker(memory_manager):
    """Queries below the noise-filter length get no marker (0 hits isn't a real
    'nothing matched' signal for a 2-char query)."""
    fake = _fake_self(memory_manager, mem_results=[])

    await ChatMixin._search_relevant_memories(fake, "hi")

    texts = _recall_texts(memory_manager)
    assert not any(ABSENCE_MARKER in t for t in texts), texts
