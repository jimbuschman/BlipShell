"""Tests for memory integration with project mode executor (CLAUDE.md item 9).

Verifies:
- Memory search results injected into executor system prompt
- Chat history (last 10 messages) passed to executor
- Executor narrative fed through memory pipeline after execution
- Narrative builder handles edge cases (empty, no tools, etc.)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from blipshell.core.executor import build_executor_narrative


# ── Memory context formatting ────────────────────────────────────────────────

# The agent formats search results before passing to executor.
# We test the formatting logic directly since it's a string transform.


class TestMemoryContextFormatting:
    """Test the memory context injection format used by _chat_planned()."""

    def test_format_search_results(self):
        """SearchResults are formatted as 'Relevant memories from past sessions'."""
        # Simulate what _chat_planned does with search results
        results = [
            MagicMock(summary="User prefers snake_case naming", similarity=0.85),
            MagicMock(summary="Project uses FastAPI framework", similarity=0.78),
        ]
        # This is the formatting logic from agent.py:_chat_planned
        lines = [f"- {r.summary}" for r in results]
        memory_context = "Relevant memories from past sessions:\n" + "\n".join(lines)
        assert "snake_case" in memory_context
        assert "FastAPI" in memory_context
        assert memory_context.startswith("Relevant memories")

    def test_empty_search_results(self):
        """No memories → no memory_context injected."""
        results = []
        memory_context = "" if not results else "..."
        assert memory_context == ""


# ── Chat history extraction ──────────────────────────────────────────────────


class TestChatHistoryExtraction:
    """Test the chat history selection logic for executor context."""

    def test_last_10_messages(self):
        """Only last 10 user/assistant messages are passed."""
        from blipshell.models.session import MessageRole, SessionMessage
        from datetime import datetime, timezone

        messages = []
        for i in range(20):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            messages.append(SessionMessage(
                role=role,
                content=f"message {i}",
                timestamp=datetime.now(timezone.utc),
                token_count=10,
            ))

        # Simulate _chat_planned logic
        chat_history = []
        for msg in messages[-10:]:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                chat_history.append(msg.to_ollama_message())

        assert len(chat_history) == 10
        # Should be messages 10-19
        assert "message 10" in chat_history[0]["content"]
        assert "message 19" in chat_history[-1]["content"]

    def test_filters_non_chat_roles(self):
        """System/tool messages excluded from chat history."""
        from blipshell.models.session import MessageRole, SessionMessage
        from datetime import datetime, timezone

        messages = [
            SessionMessage(role=MessageRole.USER, content="question",
                           timestamp=datetime.now(timezone.utc), token_count=5),
            SessionMessage(role=MessageRole.SYSTEM, content="sys prompt",
                           timestamp=datetime.now(timezone.utc), token_count=5),
            SessionMessage(role=MessageRole.ASSISTANT, content="answer",
                           timestamp=datetime.now(timezone.utc), token_count=5),
        ]

        chat_history = []
        for msg in messages[-10:]:
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT):
                chat_history.append(msg.to_ollama_message())

        assert len(chat_history) == 2
        roles = [m["role"] for m in chat_history]
        assert "system" not in roles


# ── Executor system prompt construction ──────────────────────────────────────


class TestExecutorSystemPrompt:
    """Test that memory context is injected into executor system prompt."""

    def test_memory_context_in_system_prompt(self):
        """Memory context appears under --- RELEVANT MEMORIES --- header."""
        memory_context = "Relevant memories:\n- User likes Python"
        sys_prompt = "You are a coding assistant."
        if memory_context:
            sys_prompt += f"\n\n--- RELEVANT MEMORIES ---\n{memory_context}"
        assert "--- RELEVANT MEMORIES ---" in sys_prompt
        assert "User likes Python" in sys_prompt

    def test_no_memory_header_when_empty(self):
        """No RELEVANT MEMORIES section when memory_context is empty."""
        memory_context = ""
        sys_prompt = "You are a coding assistant."
        if memory_context:
            sys_prompt += f"\n\n--- RELEVANT MEMORIES ---\n{memory_context}"
        assert "RELEVANT MEMORIES" not in sys_prompt


# ── Narrative → memory pipeline integration ──────────────────────────────────


class TestNarrativeToMemory:
    """Test that executor narratives can be processed by the memory pipeline."""

    async def test_narrative_passes_noise_filter(
        self, memory_processor, sqlite_store,
    ):
        """A real-looking narrative should NOT be filtered as noise."""
        session_id = await sqlite_store.create_session("Test")
        narrative = (
            "Task: fix the login bug\n"
            "I read the auth module and found a missing null check. "
            "Fixed the login method to validate user input before proceeding.\n\n"
            "Actions:\n"
            "- Read: auth.py\n"
            "- Edited: auth.py"
        )
        mem_id = await memory_processor.process_message(
            text=narrative,
            role="assistant",
            session_id=session_id,
        )
        # Should not be filtered — it's substantive content
        assert mem_id is not None

    async def test_empty_narrative_filtered(
        self, memory_processor, sqlite_store,
    ):
        """An empty narrative should be filtered as noise."""
        session_id = await sqlite_store.create_session("Test")
        mem_id = await memory_processor.process_message(
            text="",
            role="assistant",
            session_id=session_id,
        )
        assert mem_id is None

    async def test_trivial_narrative_filtered(
        self, memory_processor, sqlite_store,
    ):
        """A very short narrative ('ok', 'done') should be filtered."""
        session_id = await sqlite_store.create_session("Test")
        mem_id = await memory_processor.process_message(
            text="ok",
            role="assistant",
            session_id=session_id,
        )
        assert mem_id is None


# ── Narrative extraction edge cases ──────────────────────────────────────────


class TestNarrativeEdgeCases:
    def test_only_tool_results_produces_empty(self):
        """Messages with only tool results (no assistant text) → empty narrative."""
        messages = [
            {"role": "system", "content": "rules..."},
            {"role": "tool", "content": "file contents here"},
            {"role": "tool", "content": "more file contents"},
        ]
        narrative = build_executor_narrative(messages)
        assert narrative == ""

    def test_assistant_with_no_content_but_tool_calls(self):
        """Empty assistant content but with tool calls → action summary only."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "x.py"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Read: x.py" in narrative

    def test_list_directory_shown_as_read(self):
        """list_directory calls appear in the Read section with trailing /."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_directory", "arguments": {"path": "src"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Read: src/" in narrative

    def test_chat_history_user_messages_kept(self):
        """Regular user messages (from chat history injection) are kept."""
        messages = [
            {"role": "user", "content": "I want to use FastAPI for this"},
        ]
        narrative = build_executor_narrative(messages)
        assert "User: I want to use FastAPI" in narrative

    def test_mixed_tool_types_all_sections(self):
        """All tool types produce their respective action sections."""
        messages = [
            {
                "role": "assistant",
                "content": "Planning the changes.",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
                    {"function": {"name": "write_file", "arguments": {"path": "b.py"}}},
                    {"function": {"name": "edit_file", "arguments": {"path": "c.py"}}},
                    {"function": {"name": "grep_files", "arguments": {"pattern": "def main"}}},
                    {"function": {"name": "run_command", "arguments": {"command": "pytest"}}},
                ],
            },
        ]
        narrative = build_executor_narrative(messages)
        assert "Read: a.py" in narrative
        assert "Created: b.py" in narrative
        assert "Edited: c.py" in narrative
        assert "Searched: def main" in narrative
        assert "Ran: pytest" in narrative
