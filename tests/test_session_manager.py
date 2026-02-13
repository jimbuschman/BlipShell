"""Tests for session management (session/manager.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from blipshell.memory.manager import MemoryManager
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import MemoryConfig
from blipshell.models.session import MessageRole
from blipshell.session.manager import SessionManager


@pytest.fixture
def session_manager(sqlite_store, memory_config, mock_router):
    mm = MemoryManager(memory_config)
    processor = MagicMock(spec=MemoryProcessor)
    processor.process_message = AsyncMock(return_value=1)
    processor.process_lesson = AsyncMock(return_value=1)
    return SessionManager(
        sqlite=sqlite_store,
        memory_manager=mm,
        processor=processor,
        router=mock_router,
        summary_chunk_size=5,
    )


class TestSessionManager:
    async def test_start_new_session(self, session_manager):
        sid = await session_manager.start_session(project="test-project")
        assert sid is not None
        assert sid > 0
        assert session_manager.session_id == sid
        assert session_manager.project == "test-project"

    async def test_add_message(self, session_manager):
        await session_manager.start_session()
        session_manager.add_message(MessageRole.USER, "hello world")
        assert session_manager.message_count == 1
        msgs = session_manager.get_messages()
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.USER
        assert msgs[0].content == "hello world"

    async def test_add_message_cleans_text(self, session_manager):
        await session_manager.start_session()
        session_manager.add_message(MessageRole.USER, "  hello  \r\n  world  ")
        msgs = session_manager.get_messages()
        assert msgs[0].content == "hello \n world"

    async def test_get_ollama_messages(self, session_manager):
        await session_manager.start_session()
        session_manager.add_message(MessageRole.USER, "test message")
        ollama_msgs = session_manager.get_ollama_messages()
        assert len(ollama_msgs) == 1
        assert ollama_msgs[0]["role"] == "user"
        assert ollama_msgs[0]["content"] == "test message"

    async def test_get_undumped_messages(self, session_manager):
        await session_manager.start_session()
        session_manager.add_message(MessageRole.USER, "first message is undumped")
        undumped = session_manager.get_undumped_messages()
        assert len(undumped) == 1

    async def test_dump_to_memory(self, session_manager):
        await session_manager.start_session()
        session_manager.add_message(MessageRole.USER, "this should be dumped to memory")
        session_manager.add_message(MessageRole.ASSISTANT, "I understand, here is my response")
        await session_manager.dump_to_memory()
        # After dump, undumped should be empty
        undumped = session_manager.get_undumped_messages()
        assert len(undumped) == 0

    async def test_dump_skips_when_saving(self, session_manager):
        await session_manager.start_session()
        session_manager._currently_saving = True
        session_manager.add_message(MessageRole.USER, "should not be dumped")
        await session_manager.dump_to_memory()
        # Still undumped because _currently_saving was True
        undumped = session_manager.get_undumped_messages()
        assert len(undumped) == 1
        session_manager._currently_saving = False

    async def test_resume_session(self, session_manager, sqlite_store):
        # Create a session first
        sid = await sqlite_store.create_session(title="Resumable")
        # Resume it
        resumed_id = await session_manager.start_session(resume_session_id=sid)
        assert resumed_id == sid

    async def test_message_count(self, session_manager):
        await session_manager.start_session()
        assert session_manager.message_count == 0
        session_manager.add_message(MessageRole.USER, "one")
        session_manager.add_message(MessageRole.ASSISTANT, "two")
        assert session_manager.message_count == 2

    async def test_clean_text_static(self):
        assert SessionManager._clean_text("") == ""
        assert SessionManager._clean_text("  hello  ") == "hello"
        assert SessionManager._clean_text("a\r\nb") == "a\nb"
        assert SessionManager._clean_text("a\tb") == "a b"
        assert SessionManager._clean_text("a  b") == "a b"
