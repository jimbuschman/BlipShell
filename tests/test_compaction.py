"""Tests for structured compaction, partial compaction, file restoration, and session notes."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from blipshell.core.chat_loop import (
    _find_split_point,
    _messages_to_text,
    _restore_files_post_compaction,
    compact_messages,
    estimate_messages_tokens,
    structured_compact_messages,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _msg(role, content, tool_calls=None, tool_call_id=None):
    """Build a message dict."""
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    if tool_call_id:
        m["tool_call_id"] = tool_call_id
    return m


def _make_conversation(n_user_msgs=10, with_tools=False):
    """Build a synthetic conversation with system prefix + user/assistant pairs."""
    messages = [_msg("system", "You are a helpful assistant.")]
    for i in range(n_user_msgs):
        messages.append(_msg("user", f"User message {i}: " + "x" * 200))
        if with_tools:
            tc = [{"id": f"tc_{i}", "function": {"name": "read_file", "arguments": "{}"}}]
            messages.append(_msg("assistant", "", tool_calls=tc))
            messages.append(_msg("tool", f"File content for call {i}: " + "y" * 300, tool_call_id=f"tc_{i}"))
        else:
            messages.append(_msg("assistant", f"Assistant response {i}: " + "z" * 200))
    return messages


# ── TestSplitPoint ─────────────────────────────────────────────────────────


class TestSplitPoint:
    """Test _find_split_point logic."""

    def test_keeps_min_user_messages(self):
        msgs = _make_conversation(n_user_msgs=10)
        split = _find_split_point(msgs, min_recent_user_msgs=5, min_recent_tokens=0)
        # Count user messages in recent portion
        recent = msgs[split:]
        user_count = sum(1 for m in recent if m["role"] == "user")
        assert user_count >= 5

    def test_keeps_min_tokens(self):
        msgs = _make_conversation(n_user_msgs=10)
        split = _find_split_point(msgs, min_recent_user_msgs=0, min_recent_tokens=1000)
        recent = msgs[split:]
        total = sum(len(m.get("content", "")) // 4 for m in recent)
        assert total >= 250  # 1000 tokens ≈ at least some content

    def test_never_splits_mid_tool_pair(self):
        msgs = _make_conversation(n_user_msgs=10, with_tools=True)
        split = _find_split_point(msgs, min_recent_user_msgs=3, min_recent_tokens=0)
        # The message at the split point should NOT be a tool result
        if split < len(msgs):
            assert msgs[split]["role"] != "tool", \
                f"Split at index {split} is a tool result — should not split here"

    def test_all_recent_returns_start(self):
        """When everything is 'recent', split at the system prefix boundary."""
        msgs = _make_conversation(n_user_msgs=3)
        split = _find_split_point(msgs, min_recent_user_msgs=10, min_recent_tokens=999999)
        # Should return 0 or the first non-system index
        assert split <= 1

    def test_short_conversation_no_split(self):
        msgs = [_msg("system", "sys"), _msg("user", "hi"), _msg("assistant", "hello")]
        split = _find_split_point(msgs, min_recent_user_msgs=5, min_recent_tokens=10000)
        # Everything is recent
        assert split <= 1

    def test_system_prefix_preserved(self):
        msgs = _make_conversation(n_user_msgs=10)
        split = _find_split_point(msgs, min_recent_user_msgs=3, min_recent_tokens=0)
        # Split should be at or after the first non-system message
        assert split >= 1  # system message at index 0


# ── TestMessagesToText ─────────────────────────────────────────────────────


class TestMessagesToText:
    """Test _messages_to_text conversion."""

    def test_user_and_assistant(self):
        msgs = [
            _msg("user", "Hello"),
            _msg("assistant", "Hi there"),
        ]
        text = _messages_to_text(msgs)
        assert "User: Hello" in text
        assert "Assistant: Hi there" in text

    def test_skips_system(self):
        msgs = [_msg("system", "secret"), _msg("user", "Hello")]
        text = _messages_to_text(msgs)
        assert "secret" not in text
        assert "User: Hello" in text

    def test_truncates_long_tool_results(self):
        msgs = [_msg("tool", "x" * 1000, tool_call_id="tc1")]
        text = _messages_to_text(msgs)
        assert "[1000 chars total]" in text
        assert len(text) < 1000


# ── TestStructuredCompaction ──────────────────────────────────────────────


class TestStructuredCompaction:
    """Test structured_compact_messages with mocked LLM."""

    @pytest.mark.asyncio
    async def test_llm_summary_replaces_old_messages(self):
        msgs = _make_conversation(n_user_msgs=10)
        router = MagicMock()
        router.generate = AsyncMock(return_value="## 1. Primary Request\nUser asked about X.")

        config = MagicMock()
        config.partial_compaction = True
        config.min_recent_user_messages = 3
        config.min_recent_tokens = 0
        config.file_restoration = False
        config.summary_timeout = 30.0
        config.use_llm = True

        result = await structured_compact_messages(msgs, router, config)

        # Should have system prefix + summary + recent messages + continuation
        assert any("[Compacted conversation summary]" in m.get("content", "") for m in result)
        # Should be shorter than original
        assert len(result) < len(msgs)
        # Router should have been called
        router.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_recent_portion_preserved_verbatim(self):
        msgs = _make_conversation(n_user_msgs=10)
        original_last_user = [m for m in msgs if m["role"] == "user"][-1]

        router = MagicMock()
        router.generate = AsyncMock(return_value="Summary of old messages.")

        config = MagicMock()
        config.partial_compaction = True
        config.min_recent_user_messages = 3
        config.min_recent_tokens = 0
        config.file_restoration = False
        config.summary_timeout = 30.0

        result = await structured_compact_messages(msgs, router, config)

        # Last user message should still be in the result verbatim
        result_contents = [m.get("content", "") for m in result]
        assert original_last_user["content"] in result_contents

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_mechanical(self):
        msgs = _make_conversation(n_user_msgs=10, with_tools=True)

        router = MagicMock()
        router.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        config = MagicMock()
        config.partial_compaction = True
        config.min_recent_user_messages = 3
        config.min_recent_tokens = 0
        config.file_restoration = False
        config.summary_timeout = 5.0

        result = await structured_compact_messages(msgs, router, config)

        # Should fall back to mechanical — result should still be shorter
        assert len(result) <= len(msgs)
        # Should contain compacted markers from mechanical compaction
        compacted = [m for m in result if "[Compacted]" in m.get("content", "")]
        assert len(compacted) > 0

    @pytest.mark.asyncio
    async def test_continuation_prompt_appended(self):
        msgs = _make_conversation(n_user_msgs=10)
        router = MagicMock()
        router.generate = AsyncMock(return_value="Summary.")

        config = MagicMock()
        config.partial_compaction = True
        config.min_recent_user_messages = 3
        config.min_recent_tokens = 0
        config.file_restoration = False
        config.summary_timeout = 30.0

        result = await structured_compact_messages(msgs, router, config)

        # Should have continuation message
        last_msgs = [m.get("content", "") for m in result]
        assert any("compacted" in c.lower() and "resume" in c.lower() for c in last_msgs)

    @pytest.mark.asyncio
    async def test_empty_summary_falls_back(self):
        msgs = _make_conversation(n_user_msgs=10)
        router = MagicMock()
        router.generate = AsyncMock(return_value="")

        config = MagicMock()
        config.partial_compaction = True
        config.min_recent_user_messages = 3
        config.min_recent_tokens = 0
        config.file_restoration = False
        config.summary_timeout = 30.0

        result = await structured_compact_messages(msgs, router, config)
        # Should fall back to mechanical
        assert len(result) <= len(msgs)


# ── TestFileRestoration ──────────────────────────────────────────────────


class TestFileRestoration:
    """Test _restore_files_post_compaction."""

    def test_restores_cached_files(self):
        messages = [_msg("system", "sys"), _msg("user", "hi")]
        files_read = {"file1.py", "file2.py"}
        file_cache = {
            "file1.py": "def foo(): pass",
            "file2.py": "def bar(): pass",
        }
        config = MagicMock()
        config.file_restoration = True
        config.max_restore_files = 5
        config.max_restore_tokens_per_file = 5000
        config.max_restore_tokens_total = 25000

        result = _restore_files_post_compaction(messages, files_read, file_cache, config)
        # Should have original messages + restored files
        assert len(result) > len(messages)
        restored = [m for m in result if "restored after compaction" in m.get("content", "").lower()]
        assert len(restored) == 2

    def test_respects_max_files_cap(self):
        messages = [_msg("system", "sys")]
        files_read = {f"file{i}.py" for i in range(10)}
        file_cache = {f"file{i}.py": f"content {i}" for i in range(10)}
        config = MagicMock()
        config.file_restoration = True
        config.max_restore_files = 3
        config.max_restore_tokens_per_file = 5000
        config.max_restore_tokens_total = 25000

        result = _restore_files_post_compaction(messages, files_read, file_cache, config)
        restored = [m for m in result if "restored after compaction" in m.get("content", "").lower()]
        assert len(restored) <= 3

    def test_no_files_when_disabled(self):
        messages = [_msg("system", "sys")]
        config = MagicMock()
        config.file_restoration = False

        result = _restore_files_post_compaction(messages, {"f.py"}, {"f.py": "x"}, config)
        assert result == messages

    def test_respects_total_token_limit(self):
        messages = [_msg("system", "sys")]
        # Use varied text so token estimation is realistic
        files_read = {f"file{i}.py" for i in range(5)}
        file_cache = {f"file{i}.py": f"def func_{i}(): " + "return " * 200 for i in range(5)}
        config = MagicMock()
        config.file_restoration = True
        config.max_restore_files = 5
        config.max_restore_tokens_per_file = 5000
        config.max_restore_tokens_total = 200  # very tight limit

        result = _restore_files_post_compaction(messages, files_read, file_cache, config)
        restored = [m for m in result if "restored after compaction" in m.get("content", "").lower()]
        assert len(restored) < 5  # should be limited by total tokens

    def test_no_crash_when_no_files(self):
        messages = [_msg("system", "sys")]
        config = MagicMock()
        config.file_restoration = True

        result = _restore_files_post_compaction(messages, None, None, config)
        assert result == messages


# ── TestSessionNotesPersistence ──────────────────────────────────────────


class TestSessionNotesPersistence:
    """Test session notes persistence in sqlite_store."""

    @pytest.mark.asyncio
    async def test_save_and_load_notes(self):
        """Test basic save/load cycle via sqlite methods."""
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(":memory:")
        await store.initialize()
        session_id = await store.create_session(title="test")

        # Save notes
        notes = {"task": "Build feature X", "decision": "Use async pattern"}
        await store.save_session_notes(session_id, notes)

        # Load notes
        loaded = await store.get_session_notes(session_id)
        assert loaded == notes

    @pytest.mark.asyncio
    async def test_notes_merge_with_metadata(self):
        """Notes should not clobber other metadata."""
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(":memory:")
        await store.initialize()
        session_id = await store.create_session(title="test")

        # Set some other metadata first
        await store._db.execute(
            "UPDATE sessions SET metadata_json = ? WHERE id = ?",
            (json.dumps({"other": "data"}), session_id),
        )
        await store._db.commit()

        # Save notes — should merge, not replace
        await store.save_session_notes(session_id, {"task": "test"})

        cursor = await store._db.execute(
            "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,),
        )
        row = await cursor.fetchone()
        metadata = json.loads(row["metadata_json"])
        assert metadata["other"] == "data"
        assert metadata["notes"] == {"task": "test"}

    @pytest.mark.asyncio
    async def test_clear_notes(self):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(":memory:")
        await store.initialize()
        session_id = await store.create_session(title="test")

        await store.save_session_notes(session_id, {"task": "test"})
        await store.clear_session_notes(session_id)

        loaded = await store.get_session_notes(session_id)
        assert loaded == {}

    @pytest.mark.asyncio
    async def test_empty_session_returns_empty(self):
        from blipshell.memory.sqlite_store import SQLiteStore

        store = SQLiteStore(":memory:")
        await store.initialize()
        session_id = await store.create_session(title="test")

        loaded = await store.get_session_notes(session_id)
        assert loaded == {}


# ── TestNoteTools ────────────────────────────────────────────────────────


class TestSaveNoteTool:
    """Unit tests for SaveNoteTool."""

    @pytest.mark.asyncio
    async def test_basic_save(self):
        from blipshell.core.tools.note_tools import SaveNoteTool

        sqlite = AsyncMock()
        notes = {}
        config = MagicMock()
        config.max_notes = 50
        config.max_note_tokens = 2000
        config.max_total_tokens = 12000

        tool = SaveNoteTool(sqlite, session_id=1, notes_config=config, notes=notes)
        result = await tool.execute(name="task", content="Build feature X")

        assert "saved" in result
        assert notes["task"] == "Build feature X"
        sqlite.save_session_notes.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_existing(self):
        from blipshell.core.tools.note_tools import SaveNoteTool

        sqlite = AsyncMock()
        notes = {"task": "old value"}
        config = MagicMock()
        config.max_notes = 50
        config.max_note_tokens = 2000
        config.max_total_tokens = 12000

        tool = SaveNoteTool(sqlite, session_id=1, notes_config=config, notes=notes)
        result = await tool.execute(name="task", content="new value")

        assert "updated" in result
        assert notes["task"] == "new value"

    @pytest.mark.asyncio
    async def test_rejects_over_count_limit(self):
        from blipshell.core.tools.note_tools import SaveNoteTool

        sqlite = AsyncMock()
        notes = {f"note{i}": "x" for i in range(50)}
        config = MagicMock()
        config.max_notes = 50
        config.max_note_tokens = 2000
        config.max_total_tokens = 12000

        tool = SaveNoteTool(sqlite, session_id=1, notes_config=config, notes=notes)
        result = await tool.execute(name="new_note", content="test")

        assert "Error" in result
        assert "new_note" not in notes


class TestGetNotesTool:
    """Unit tests for GetNotesTool."""

    @pytest.mark.asyncio
    async def test_list_all(self):
        from blipshell.core.tools.note_tools import GetNotesTool

        notes = {"task": "Build X", "decision": "Use Y"}
        tool = GetNotesTool(AsyncMock(), session_id=1, notes=notes)
        result = await tool.execute()

        assert "task" in result
        assert "Build X" in result
        assert "decision" in result

    @pytest.mark.asyncio
    async def test_get_specific(self):
        from blipshell.core.tools.note_tools import GetNotesTool

        notes = {"task": "Build X"}
        tool = GetNotesTool(AsyncMock(), session_id=1, notes=notes)
        result = await tool.execute(name="task")

        assert "Build X" in result

    @pytest.mark.asyncio
    async def test_not_found(self):
        from blipshell.core.tools.note_tools import GetNotesTool

        notes = {"task": "Build X"}
        tool = GetNotesTool(AsyncMock(), session_id=1, notes=notes)
        result = await tool.execute(name="missing")

        assert "not found" in result


class TestDeleteNoteTool:
    """Unit tests for DeleteNoteTool."""

    @pytest.mark.asyncio
    async def test_delete(self):
        from blipshell.core.tools.note_tools import DeleteNoteTool

        sqlite = AsyncMock()
        notes = {"task": "Build X", "other": "Y"}
        tool = DeleteNoteTool(sqlite, session_id=1, notes=notes)
        result = await tool.execute(name="task")

        assert "deleted" in result
        assert "task" not in notes
        assert "other" in notes

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        from blipshell.core.tools.note_tools import DeleteNoteTool

        notes = {}
        tool = DeleteNoteTool(AsyncMock(), session_id=1, notes=notes)
        result = await tool.execute(name="missing")

        assert "not found" in result


# ── TestCompactionInSimpleChat ───────────────────────────────────────────


class TestCompactionConfig:
    """Test that CompactionConfig and NotesConfig are properly defined."""

    def test_compaction_config_defaults(self):
        from blipshell.models.config import CompactionConfig
        cfg = CompactionConfig()
        assert cfg.enabled is False
        assert cfg.use_llm is True
        assert cfg.compaction_threshold == 0.95
        assert cfg.partial_compaction is True
        assert cfg.min_recent_user_messages == 5
        assert cfg.min_recent_tokens == 10000
        assert cfg.file_restoration is True
        assert cfg.max_restore_files == 5

    def test_notes_config_defaults(self):
        from blipshell.models.config import NotesConfig
        cfg = NotesConfig()
        assert cfg.enabled is True
        assert cfg.max_notes == 50
        assert cfg.max_total_tokens == 4000
        assert cfg.max_note_tokens == 2000

    def test_configs_on_blipshell_config(self):
        from blipshell.models.config import BlipShellConfig
        cfg = BlipShellConfig()
        assert hasattr(cfg, "compaction")
        assert hasattr(cfg, "notes")
        assert cfg.compaction.enabled is False
        assert cfg.notes.enabled is True


# ── TestMechanicalCompactionPreserved ────────────────────────────────────


class TestMechanicalCompactionPreserved:
    """Verify existing mechanical compaction still works (regression test)."""

    def test_keeps_last_n_tool_results(self):
        msgs = _make_conversation(n_user_msgs=10, with_tools=True)
        compacted = compact_messages(msgs, keep_last_n=3)
        # Last 3 tool results should be intact (not compacted)
        tool_results = [m for m in compacted if m["role"] == "tool"]
        intact = [m for m in tool_results if "[Compacted]" not in m.get("content", "")]
        assert len(intact) >= 3

    def test_preserves_system_messages(self):
        msgs = _make_conversation(n_user_msgs=10)
        compacted = compact_messages(msgs)
        system_msgs = [m for m in compacted if m["role"] == "system"]
        assert len(system_msgs) >= 1
