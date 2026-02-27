"""Tests for message persistence and startup sweep (CLAUDE.md item 13).

Verifies:
- Messages persist to session_messages table with is_processed=0
- mark_message_processed() sets is_processed=1
- get_unprocessed_messages() returns only unprocessed user/assistant messages
- Startup sweep reprocesses failed messages
- Executor narrative builder extracts reasoning and compresses tool calls
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from blipshell.core.executor import build_executor_narrative


# ── SQLite persistence layer ─────────────────────────────────────────────────


class TestSaveSessionMessage:
    async def test_message_persisted(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        msg_id = await sqlite_store.save_session_message(
            session_id, "user", "hello world",
        )
        assert msg_id is not None
        assert msg_id > 0

    async def test_message_starts_unprocessed(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        await sqlite_store.save_session_message(session_id, "user", "hello")
        unprocessed = await sqlite_store.get_unprocessed_messages()
        assert len(unprocessed) == 1
        assert unprocessed[0]["content"] == "hello"
        assert unprocessed[0]["role"] == "user"

    async def test_multiple_messages_ordered_by_id(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        await sqlite_store.save_session_message(session_id, "user", "first")
        await sqlite_store.save_session_message(session_id, "assistant", "second")
        await sqlite_store.save_session_message(session_id, "user", "third")
        unprocessed = await sqlite_store.get_unprocessed_messages()
        assert len(unprocessed) == 3
        assert [m["content"] for m in unprocessed] == ["first", "second", "third"]


class TestMarkMessageProcessed:
    async def test_marking_removes_from_unprocessed(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        msg_id = await sqlite_store.save_session_message(
            session_id, "user", "process me",
        )
        await sqlite_store.mark_message_processed(msg_id)
        unprocessed = await sqlite_store.get_unprocessed_messages()
        assert len(unprocessed) == 0

    async def test_partial_marking(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        id1 = await sqlite_store.save_session_message(session_id, "user", "msg1")
        id2 = await sqlite_store.save_session_message(session_id, "user", "msg2")
        id3 = await sqlite_store.save_session_message(session_id, "user", "msg3")
        # Only mark first and third
        await sqlite_store.mark_message_processed(id1)
        await sqlite_store.mark_message_processed(id3)
        unprocessed = await sqlite_store.get_unprocessed_messages()
        assert len(unprocessed) == 1
        assert unprocessed[0]["content"] == "msg2"


class TestGetUnprocessedMessages:
    async def test_limit_respected(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        for i in range(10):
            await sqlite_store.save_session_message(session_id, "user", f"msg{i}")
        unprocessed = await sqlite_store.get_unprocessed_messages(limit=3)
        assert len(unprocessed) == 3
        # Should be first 3 (FIFO order)
        assert unprocessed[0]["content"] == "msg0"

    async def test_excludes_system_and_tool_roles(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        await sqlite_store.save_session_message(session_id, "user", "user msg")
        await sqlite_store.save_session_message(session_id, "system", "sys msg")
        await sqlite_store.save_session_message(session_id, "tool", "tool msg")
        await sqlite_store.save_session_message(session_id, "assistant", "asst msg")
        unprocessed = await sqlite_store.get_unprocessed_messages()
        roles = [m["role"] for m in unprocessed]
        assert "system" not in roles
        assert "tool" not in roles
        assert set(roles) == {"user", "assistant"}

    async def test_empty_when_all_processed(self, sqlite_store):
        session_id = await sqlite_store.create_session("Test")
        msg_id = await sqlite_store.save_session_message(
            session_id, "user", "done",
        )
        await sqlite_store.mark_message_processed(msg_id)
        unprocessed = await sqlite_store.get_unprocessed_messages()
        assert unprocessed == []

    async def test_empty_when_no_messages(self, sqlite_store):
        unprocessed = await sqlite_store.get_unprocessed_messages()
        assert unprocessed == []

    async def test_includes_session_id(self, sqlite_store):
        s1 = await sqlite_store.create_session("Session1")
        s2 = await sqlite_store.create_session("Session2")
        await sqlite_store.save_session_message(s1, "user", "from s1")
        await sqlite_store.save_session_message(s2, "user", "from s2")
        unprocessed = await sqlite_store.get_unprocessed_messages()
        session_ids = {m["session_id"] for m in unprocessed}
        assert s1 in session_ids
        assert s2 in session_ids


# ── Executor narrative builder ───────────────────────────────────────────────


class TestBuildExecutorNarrative:
    def test_empty_messages(self):
        assert build_executor_narrative([]) == ""

    def test_extracts_user_task(self):
        messages = [
            {"role": "user", "content": "Task: add a login button\n\nAPPROACH: ..."},
        ]
        narrative = build_executor_narrative(messages)
        assert "add a login button" in narrative
        assert "APPROACH" not in narrative

    def test_extracts_assistant_reasoning(self):
        messages = [
            {"role": "assistant", "content": "I'll read the file first to understand the structure."},
        ]
        narrative = build_executor_narrative(messages)
        assert "read the file first" in narrative

    def test_drops_system_messages(self):
        messages = [
            {"role": "system", "content": "You are a coding assistant with these rules..."},
            {"role": "assistant", "content": "I'll help you."},
        ]
        narrative = build_executor_narrative(messages)
        assert "coding assistant" not in narrative
        assert "help you" in narrative

    def test_drops_tool_results(self):
        messages = [
            {"role": "assistant", "content": "Reading the file now."},
            {"role": "tool", "content": "def main():\n    print('hello')\n    ...500 lines..."},
        ]
        narrative = build_executor_narrative(messages)
        assert "def main" not in narrative
        assert "Reading the file" in narrative

    def test_summarizes_read_file_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "src/app.py"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Read: src/app.py" in narrative

    def test_summarizes_write_file_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "write_file", "arguments": {"path": "new_file.py"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Created: new_file.py" in narrative

    def test_summarizes_edit_file_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "edit_file", "arguments": {"path": "config.py"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Edited: config.py" in narrative

    def test_summarizes_grep_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "grep_files", "arguments": {"pattern": "TODO"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Searched: TODO" in narrative

    def test_summarizes_run_command(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "run_command", "arguments": {"command": "pytest tests/"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Ran: pytest tests/" in narrative

    def test_deduplicates_reads(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "app.py"}}},
                    {"function": {"name": "read_file", "arguments": {"path": "app.py"}}},
                    {"function": {"name": "read_file", "arguments": {"path": "other.py"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        # Should show app.py once, not twice
        assert narrative.count("app.py") == 1
        assert "other.py" in narrative

    def test_strips_task_complete_prefix(self):
        messages = [
            {"role": "assistant", "content": "TASK_COMPLETE I finished the work."},
        ]
        narrative = build_executor_narrative(messages)
        assert "TASK_COMPLETE" not in narrative
        assert "finished the work" in narrative

    def test_skips_nudge_continuations(self):
        messages = [
            {"role": "user", "content": "Continue. Keep going."},
        ]
        narrative = build_executor_narrative(messages)
        # "Continue." messages are skipped
        assert "Keep going" not in narrative

    def test_full_conversation(self):
        """Integration: realistic executor conversation produces clean narrative."""
        messages = [
            {"role": "system", "content": "You are a coding assistant..."},
            {"role": "user", "content": "Task: fix the login bug\n\nAPPROACH: read then fix"},
            {"role": "assistant", "content": "I'll read the auth module first.",
             "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "auth.py"}}}]},
            {"role": "tool", "content": "class Auth:\n    def login(self):\n        pass\n...(200 lines)"},
            {"role": "assistant", "content": "Found the issue. Fixing the login method.",
             "tool_calls": [{"function": {"name": "edit_file", "arguments": {"path": "auth.py"}}}]},
            {"role": "tool", "content": "Successfully edited auth.py"},
            {"role": "assistant", "content": "The login bug is fixed. The issue was a missing null check."},
        ]
        narrative = build_executor_narrative(messages)
        # Should have reasoning, actions, no noise
        assert "read the auth module" in narrative
        assert "login bug is fixed" in narrative
        assert "Read: auth.py" in narrative
        assert "Edited: auth.py" in narrative
        # Should NOT have system prompt or tool results
        assert "coding assistant" not in narrative
        assert "class Auth" not in narrative
        assert "200 lines" not in narrative

    def test_caps_searches_at_five(self):
        tool_calls = [
            {"function": {"name": "grep_files", "arguments": {"pattern": f"pat{i}"}}}
            for i in range(10)
        ]
        messages = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
        narrative = build_executor_narrative(messages)
        # Only first 5 search patterns should appear
        assert "pat0" in narrative
        assert "pat4" in narrative
        assert "pat5" not in narrative

    def test_caps_commands_at_five(self):
        tool_calls = [
            {"function": {"name": "run_command", "arguments": {"command": f"cmd{i}"}}}
            for i in range(10)
        ]
        messages = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
        narrative = build_executor_narrative(messages)
        assert "cmd0" in narrative
        assert "cmd4" in narrative
        assert "cmd5" not in narrative

    def test_truncates_long_commands(self):
        long_cmd = "a" * 100
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "run_command", "arguments": {"command": long_cmd}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        # Commands truncated to 60 chars
        assert "a" * 60 in narrative
        assert "a" * 61 not in narrative

    def test_ollama_string_format_tool_calls(self):
        """Handle Ollama's stringified tool call format."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    "function=Function(name='read_file', arguments={'path': 'test.py'})",
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Read: test.py" in narrative
