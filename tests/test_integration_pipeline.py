"""Full integration pipeline tests — round-trip through multiple components.

Tests the complete memory lifecycle: store → process → search → retrieve,
entity extraction → graph expansion, session lifecycle, and turn event logging.
All LLM calls are mocked with canned responses; SQLite is real (temp files).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from blipshell.memory.entity_extractor import EntityExtractor
from blipshell.memory.manager import MemoryManager, PoolItem
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.search import MemorySearch
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory, MemoryType
from blipshell.models.session import MessageRole
from blipshell.session.manager import SessionManager


class TestStoreAndSearch:
    """Test the full store → search round-trip."""

    async def test_store_then_fts_search(
        self, memory_processor, sqlite_store, mock_chroma, canned_router, memory_config
    ):
        """Process a message, then search for it via FTS and mock Chroma."""
        # Create a session
        session_id = await sqlite_store.create_session("Test")

        # Process a message through the full pipeline
        mem_id = await memory_processor.process_message(
            text="I think Python performance tuning with cProfile and line_profiler tools is what I want to explore next for my project",
            role="user",
            session_id=session_id,
        )
        assert mem_id is not None

        # Get the stored memory to know its summary
        memory = await sqlite_store.get_memory(mem_id)
        assert memory is not None

        # Set up mock Chroma to return this memory with high similarity
        mock_chroma.search_memories.return_value = [
            {"id": mem_id, "similarity": 0.85, "metadata": {"session_id": str(session_id)}},
        ]

        # Search for it
        search = MemorySearch(
            sqlite=sqlite_store,
            chroma=mock_chroma,
            router=canned_router,
            config=memory_config,
        )
        results = await search.search(
            query="Python performance profiling",
            current_session_id=session_id + 1,  # different session so it's not excluded
            n_results=10,
        )

        assert len(results) >= 1
        found = [r for r in results if r.memory_id == mem_id]
        assert len(found) == 1
        assert found[0].boosted_score > 0
        assert found[0].rank >= 1

    async def test_search_stats_populated(
        self, memory_processor, sqlite_store, mock_chroma, canned_router, memory_config
    ):
        """After a search, last_search_stats should be populated."""
        session_id = await sqlite_store.create_session("Test")
        mem_id = await memory_processor.process_message(
            text="I think Rust memory safety and ownership model is really interesting, want to learn more about how it prevents data races",
            role="user",
            session_id=session_id,
        )

        mock_chroma.search_memories.return_value = [
            {"id": mem_id, "similarity": 0.9, "metadata": {}},
        ]

        search = MemorySearch(
            sqlite=sqlite_store,
            chroma=mock_chroma,
            router=canned_router,
            config=memory_config,
        )
        await search.search("Rust ownership", n_results=10)

        stats = search.last_search_stats
        assert stats is not None
        assert "chroma_hits" in stats
        assert "fts_hits" in stats
        assert "entity_hits" in stats
        assert "final_returned" in stats


class TestEntityExpansion:
    """Test entity extraction → graph expansion → search enrichment."""

    async def test_entity_expansion_enriches_search(
        self, sqlite_store, mock_chroma, canned_router, memory_config
    ):
        """Memories found via entity graph should appear in search results."""
        session_id = await sqlite_store.create_session("Test")

        # Create two memories
        mem1 = Memory(
            session_id=session_id, role="user",
            content="Jim uses Python for data analysis",
            summary="Jim uses Python for data analysis",
            memory_type=MemoryType.CONVERSATION, rank=4, importance=0.6,
        )
        mem1_id = await sqlite_store.create_memory(mem1)

        mem2 = Memory(
            session_id=session_id, role="user",
            content="Jim also likes Rust for systems programming",
            summary="Jim also likes Rust for systems programming",
            memory_type=MemoryType.CONVERSATION, rank=4, importance=0.6,
        )
        mem2_id = await sqlite_store.create_memory(mem2)

        # Manually create entities and relationships linking both through "jim"
        jim_id = await sqlite_store.get_or_create_entity("jim", "person")
        python_id = await sqlite_store.get_or_create_entity("python", "technology")
        rust_id = await sqlite_store.get_or_create_entity("rust", "technology")

        await sqlite_store.create_entity_relationship(jim_id, "uses", python_id, mem1_id)
        await sqlite_store.create_entity_mention(jim_id, mem1_id)
        await sqlite_store.create_entity_mention(python_id, mem1_id)

        await sqlite_store.create_entity_relationship(jim_id, "likes", rust_id, mem2_id)
        await sqlite_store.create_entity_mention(jim_id, mem2_id)
        await sqlite_store.create_entity_mention(rust_id, mem2_id)

        # Search for "jim" — Chroma returns mem1, entity expansion should find mem2
        mock_chroma.search_memories.return_value = [
            {"id": mem1_id, "similarity": 0.8, "metadata": {}},
        ]

        search = MemorySearch(
            sqlite=sqlite_store,
            chroma=mock_chroma,
            router=canned_router,
            config=memory_config,
        )
        results = await search.search(
            query="what does jim use",
            current_session_id=session_id + 1,
            n_results=10,
        )

        result_ids = {r.memory_id for r in results}
        assert mem1_id in result_ids  # from Chroma
        # mem2 should come from entity expansion (connected through "jim")
        assert mem2_id in result_ids


class TestSessionLifecycle:
    """Test the full session lifecycle with real memory processing."""

    async def test_dump_creates_memories(
        self, sqlite_store, mock_chroma, canned_router, memory_config
    ):
        """dump_to_memory should process messages through the full pipeline."""
        processor = MemoryProcessor(
            sqlite=sqlite_store, chroma=mock_chroma,
            router=canned_router, config=memory_config,
        )
        mm = MemoryManager(memory_config)
        session_mgr = SessionManager(
            sqlite=sqlite_store,
            memory_manager=mm,
            processor=processor,
            router=canned_router,
            summary_chunk_size=20,
        )

        session_id = await session_mgr.start_session()
        assert session_id is not None

        # Add messages
        for i in range(5):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            session_mgr.add_message(role, f"I think we should discuss Python topic {i} in more detail because it affects how we design the system")

        # Dump to memory
        await session_mgr.dump_to_memory()

        # Verify memories were created in SQLite
        memories = await sqlite_store.get_memories_by_session(session_id)
        assert len(memories) >= 3  # at least some messages processed (noise filter may skip short ones)

        # Verify they have summaries and ranks
        for mem in memories:
            assert mem.summary is not None

    async def test_end_session_creates_summary(
        self, sqlite_store, mock_chroma, canned_router, memory_config
    ):
        """end_session should generate a session summary and extract lessons."""
        processor = MemoryProcessor(
            sqlite=sqlite_store, chroma=mock_chroma,
            router=canned_router, config=memory_config,
        )
        mm = MemoryManager(memory_config)
        session_mgr = SessionManager(
            sqlite=sqlite_store,
            memory_manager=mm,
            processor=processor,
            router=canned_router,
            summary_chunk_size=20,
        )

        session_id = await session_mgr.start_session()

        # Add enough messages for lesson extraction (5+)
        for i in range(6):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            session_mgr.add_message(
                role,
                f"I think we should discuss Python performance optimization and profiling tools because you mentioned it earlier in message {i}",
            )

        await session_mgr.end_session()

        # Verify session summary was created
        session = await sqlite_store.get_session(session_id)
        assert session is not None
        assert session.summary is not None
        assert len(session.summary) > 0

        # Verify a lesson was extracted
        lessons = await sqlite_store.get_all_lessons()
        assert len(lessons) >= 1


class TestTurnEvents:
    """Test that conversation flow events are logged."""

    async def test_turn_events_logged(self, sqlite_store):
        """Verify turn events can be logged and retrieved."""
        session_id = await sqlite_store.create_session("Test")

        # Log events for a turn
        await sqlite_store.log_turn_event(session_id, 1, "turn_start", {
            "query_length": 42,
            "route": "simple",
        })
        await sqlite_store.log_turn_event(session_id, 1, "search_complete", {
            "chroma_hits": 10,
            "fts_hits": 5,
            "entity_hits": 2,
            "final_returned": 8,
        })
        await sqlite_store.log_turn_event(session_id, 1, "context_built", {
            "query_profile": "balanced",
            "total_context_items": 15,
        })
        await sqlite_store.log_turn_event(session_id, 1, "llm_complete", {
            "endpoint": "local",
            "model": "test-model",
            "response_length": 256,
        })

        # Retrieve all events for the session
        events = await sqlite_store.get_turn_events(session_id)
        assert len(events) == 4

        # Verify event types
        types = [e["event_type"] for e in events]
        assert "turn_start" in types
        assert "search_complete" in types
        assert "context_built" in types
        assert "llm_complete" in types

        # Verify data is parsed correctly
        start_event = [e for e in events if e["event_type"] == "turn_start"][0]
        assert start_event["data"]["query_length"] == 42
        assert start_event["data"]["route"] == "simple"

        # Retrieve events for specific turn
        turn_events = await sqlite_store.get_turn_events_for_turn(session_id, 1)
        assert len(turn_events) == 4

        # Different turn should be empty
        empty = await sqlite_store.get_turn_events_for_turn(session_id, 99)
        assert len(empty) == 0
