"""Tests for session reflection — parser, SQL queries, conversation preparation."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory.processor import MemoryProcessor


# --- _parse_reflection tests ---


class TestParseReflection:
    def test_well_formed(self):
        text = (
            "EFFECTIVENESS: effective\n\n"
            "WHAT_WORKED:\n"
            "- Breaking the migration into per-table scripts avoided timeout issues\n"
            "- Using AST parsing instead of regex\n\n"
            "WHAT_DIDNT_WORK:\n"
            "- Initial regex approach was too brittle\n\n"
            "TECHNICAL_INSIGHTS:\n"
            "- ChromaDB PersistentClient has no close() method\n"
            "- nomic-embed-text supports 8192 tokens\n\n"
            "PROCESS_INSIGHTS:\n"
            "- Test schema changes on a copy before production"
        )
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "effective"
        assert "Breaking the migration" in result["what_worked"]
        assert "regex approach" in result["what_didnt_work"]
        assert "ChromaDB" in result["technical_insights"]
        assert "Test schema" in result["process_insights"]

    def test_partially_effective(self):
        text = (
            "EFFECTIVENESS: partially_effective\n\n"
            "WHAT_WORKED:\n- Incremental approach\n\n"
            "WHAT_DIDNT_WORK:\n- Nothing specific\n\n"
            "TECHNICAL_INSIGHTS:\n- SQLite WAL mode helps concurrency\n\n"
            "PROCESS_INSIGHTS:\n- Start with a plan"
        )
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "partially_effective"

    def test_ineffective(self):
        text = (
            "EFFECTIVENESS: ineffective\n"
            "WHAT_WORKED:\n- Nothing stood out\n"
            "WHAT_DIDNT_WORK:\n- Everything was a dead end\n"
            "TECHNICAL_INSIGHTS:\n- The API was undocumented\n"
            "PROCESS_INSIGHTS:\n- Read the docs first"
        )
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "ineffective"

    def test_missing_sections(self):
        text = (
            "EFFECTIVENESS: effective\n"
            "WHAT_WORKED:\n- Great session\n"
        )
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "effective"
        assert result["what_worked"] is not None
        assert result["what_didnt_work"] is None
        assert result["technical_insights"] is None
        assert result["process_insights"] is None

    def test_garbage_input(self):
        text = "This is just random text with no structure at all."
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "unclear"
        assert result["what_worked"] is None

    def test_colon_after_label(self):
        """Labels may or may not have colons."""
        text = (
            "EFFECTIVENESS: effective\n"
            "WHAT_WORKED:\n- approach A\n"
            "WHAT_DIDNT_WORK\n- approach B\n"
            "TECHNICAL_INSIGHTS:\n- insight C\n"
            "PROCESS_INSIGHTS\n- insight D"
        )
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "effective"
        assert "approach A" in result["what_worked"]
        assert "approach B" in result["what_didnt_work"]

    def test_effectiveness_with_explanation(self):
        """LLM might write 'effective - the session was productive'."""
        text = (
            "EFFECTIVENESS: effective - the session achieved its goals\n"
            "WHAT_WORKED:\n- item\n"
            "WHAT_DIDNT_WORK:\n- item\n"
            "TECHNICAL_INSIGHTS:\n- item\n"
            "PROCESS_INSIGHTS:\n- item"
        )
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "effective"

    def test_case_insensitive_labels(self):
        text = (
            "effectiveness: effective\n"
            "what_worked:\n- item\n"
            "what_didnt_work:\n- item\n"
            "technical_insights:\n- item\n"
            "process_insights:\n- item"
        )
        result = MemoryProcessor._parse_reflection(text)
        assert result["effectiveness"] == "effective"
        assert result["what_worked"] is not None


class TestBuildReflectionEmbedText:
    def test_full_reflection(self):
        parsed = {
            "what_worked": "- Used AST parsing",
            "what_didnt_work": "- Regex was brittle",
            "technical_insights": "- nomic-embed has 8K context",
            "process_insights": "- Plan before coding",
        }
        text = MemoryProcessor._build_reflection_embed_text(parsed)
        assert "AST parsing" in text
        assert "Regex" in text
        assert "nomic-embed" in text
        assert "Plan before" in text

    def test_partial_reflection(self):
        parsed = {
            "what_worked": "- Something worked",
            "what_didnt_work": None,
            "technical_insights": None,
            "process_insights": None,
        }
        text = MemoryProcessor._build_reflection_embed_text(parsed)
        assert "Something worked" in text
        assert "didn't work" not in text

    def test_empty_reflection(self):
        parsed = {
            "what_worked": None,
            "what_didnt_work": None,
            "technical_insights": None,
            "process_insights": None,
        }
        text = MemoryProcessor._build_reflection_embed_text(parsed)
        assert text == "Session reflection"


# --- prepare_conversation_for_reflection tests ---


class TestPrepareConversation:
    @pytest.fixture
    def mock_sqlite(self):
        return AsyncMock()

    @pytest.fixture
    def processor(self, mock_sqlite):
        mock_chroma = MagicMock()
        mock_router = MagicMock()
        return MemoryProcessor(mock_sqlite, mock_chroma, mock_router)

    @pytest.mark.asyncio
    async def test_short_session(self, processor, mock_sqlite):
        """Sessions <= 30 messages return full text."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        mock_sqlite.get_session_messages_for_lesson.return_value = messages

        text = await processor.prepare_conversation_for_reflection(1, "Summary")
        assert "Message 0" in text
        assert "Message 9" in text
        assert "[Session summary" not in text  # no truncation header

    @pytest.mark.asyncio
    async def test_empty_session(self, processor, mock_sqlite):
        """Empty sessions return just the summary."""
        mock_sqlite.get_session_messages_for_lesson.return_value = []
        mock_sqlite.get_memories_by_session.return_value = []
        text = await processor.prepare_conversation_for_reflection(1, "Summary text")
        assert text == "Summary text"

    @pytest.mark.asyncio
    async def test_long_session_truncation(self, processor, mock_sqlite):
        """Sessions > 30 messages get smart truncation."""
        messages = [
            {"role": "user", "content": f"Normal message {i}"}
            for i in range(40)
        ]
        # Add key-term messages in the middle
        messages[15] = {"role": "assistant", "content": "I found the error in line 42"}
        messages[20] = {"role": "user", "content": "That fixed the problem!"}
        messages[25] = {"role": "assistant", "content": "The solution was to use AST"}

        mock_sqlite.get_session_messages_for_lesson.return_value = messages

        text = await processor.prepare_conversation_for_reflection(1, "Test summary")

        # Should have summary
        assert "[Session summary: Test summary]" in text
        # Should have first messages
        assert "Normal message 0" in text
        assert "Normal message 4" in text
        # Should have last messages
        assert "Normal message 39" in text
        # Should have key messages (contain "error", "fixed", "solution")
        assert "found the error" in text
        assert "fixed the problem" in text
        assert "solution was to use" in text
        # Should NOT have all middle messages
        assert "Normal message 10" not in text


# --- SQLite query tests ---


class TestGetSessionsMissingReflections:
    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_finds_unreflected_sessions(self, db_path):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(db_path)
        await store.initialize()

        # Create a session with summary and messages
        sid = await store.create_session(title="Test Session")
        await store.update_session(sid, summary="A productive session")
        for i in range(6):
            await store.save_session_message(sid, "user" if i % 2 == 0 else "assistant", f"msg {i}")

        sessions = await store.get_sessions_missing_reflections(limit=10)
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid

        await store.close()

    @pytest.mark.asyncio
    async def test_excludes_reflected_sessions(self, db_path):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(db_path)
        await store.initialize()

        sid = await store.create_session(title="Test Session")
        await store.update_session(sid, summary="Done")
        for i in range(6):
            await store.save_session_message(sid, "user", f"msg {i}")

        # Add a reflection
        await store.create_session_reflection(
            session_id=sid,
            effectiveness="effective",
            reflection_text="Good session",
        )

        sessions = await store.get_sessions_missing_reflections(limit=10)
        assert len(sessions) == 0

        await store.close()

    @pytest.mark.asyncio
    async def test_excludes_sessions_without_summary(self, db_path):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(db_path)
        await store.initialize()

        sid = await store.create_session(title="No Summary")
        for i in range(6):
            await store.save_session_message(sid, "user", f"msg {i}")

        sessions = await store.get_sessions_missing_reflections(limit=10)
        assert len(sessions) == 0

        await store.close()

    @pytest.mark.asyncio
    async def test_excludes_sessions_with_single_message(self, db_path):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(db_path)
        await store.initialize()

        sid = await store.create_session(title="Short")
        await store.update_session(sid, summary="Brief chat")
        await store.save_session_message(sid, "user", "single msg")  # < 2 messages

        sessions = await store.get_sessions_missing_reflections(limit=10)
        assert len(sessions) == 0

        await store.close()


class TestCreateAndGetReflection:
    @pytest.mark.asyncio
    async def test_create_and_retrieve(self, tmp_path):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        await store.initialize()

        sid = await store.create_session(title="Test")
        rid = await store.create_session_reflection(
            session_id=sid,
            effectiveness="effective",
            reflection_text="Full reflection text here",
            technical_insights="- Insight 1\n- Insight 2",
            process_insights="- Process tip",
            what_worked="- Approach A",
            what_didnt_work="- Approach B failed",
        )
        assert rid > 0

        reflection = await store.get_session_reflection(sid)
        assert reflection is not None
        assert reflection["effectiveness"] == "effective"
        assert reflection["technical_insights"] == "- Insight 1\n- Insight 2"
        assert reflection["what_worked"] == "- Approach A"

        await store.close()

    @pytest.mark.asyncio
    async def test_unique_constraint(self, tmp_path):
        """Second reflection for same session should fail."""
        import sqlite3
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        await store.initialize()

        sid = await store.create_session(title="Test")
        await store.create_session_reflection(
            session_id=sid, effectiveness="effective",
            reflection_text="First",
        )
        with pytest.raises(Exception):  # IntegrityError
            await store.create_session_reflection(
                session_id=sid, effectiveness="ineffective",
                reflection_text="Second",
            )

        await store.close()

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tmp_path):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        await store.initialize()

        result = await store.get_session_reflection(99999)
        assert result is None

        await store.close()


class TestGetRecentReflections:
    @pytest.mark.asyncio
    async def test_returns_recent(self, tmp_path):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(str(tmp_path / "test.db"))
        await store.initialize()

        for i in range(3):
            sid = await store.create_session(title=f"Session {i}")
            await store.create_session_reflection(
                session_id=sid, effectiveness="effective",
                reflection_text=f"Reflection {i}",
            )

        recent = await store.get_recent_reflections(limit=2)
        assert len(recent) == 2

        await store.close()
