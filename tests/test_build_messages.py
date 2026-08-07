"""_build_messages must actually be CALLED by a test.

test_time_awareness.py says in its docstring that it covers "the
conversation-history stamping in _build_messages", but every test in it calls
format_relative_time directly. The helper was exhaustively tested and its only
caller was never invoked once — so when b785754 extracted the continuity
assembly and took `now = datetime.now(timezone.utc)` with it, leaving the
reference behind, 1417 tests passed on a build where every chat turn with any
prior message died with "name 'now' is not defined".

These tests call the real method.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.models.session import MessageRole


class _Msg:
    def __init__(self, role, content, timestamp):
        self.role = role
        self.content = content
        self.timestamp = timestamp

    def to_ollama_message(self):
        return {"role": self.role.value, "content": self.content}


def _agent(messages):
    """Minimal object exposing exactly what _build_messages touches."""
    a = ChatMixin.__new__(ChatMixin)

    a.memory_manager = MagicMock()
    a.memory_manager.gather_memory.return_value = []
    a.memory_manager.get_hard_caps.return_value = {}
    a.endpoint_manager = MagicMock()
    a.endpoint_manager.get_context_tokens_for_role.return_value = 65536
    a.session_manager = MagicMock()
    a.session_manager.get_messages.return_value = messages
    a.router = MagicMock()
    a.config = MagicMock()
    a.model_settings = MagicMock()
    a.active_project = None
    a._project_context = None
    a._files_read = set()
    a._pending_follow_ups = []
    a._last_context_stats = {}
    a._last_rendered_pool_texts = set()
    a._build_capability_block = lambda: "CAPABILITIES"
    a._build_continuity_block = lambda **kw: ""
    a._render_time_anchor = lambda: "\n\nTIME"
    return a


NOW = datetime.now(timezone.utc)


class TestBuildMessages:
    def test_builds_with_conversation_history(self):
        """The exact shape that was broken: history present, so the stamping
        branch runs."""
        agent = _agent([
            _Msg(MessageRole.USER, "hello", NOW - timedelta(hours=3)),
            _Msg(MessageRole.ASSISTANT, "hi there", NOW - timedelta(hours=3)),
        ])

        messages = agent._build_messages("what's up?")

        assert messages[0]["role"] == "system"
        assert len(messages) == 3

    def test_history_messages_are_time_stamped(self):
        agent = _agent([
            _Msg(MessageRole.USER, "hello", NOW - timedelta(hours=3)),
        ])

        messages = agent._build_messages("q")

        assert messages[1]["content"].startswith("[3h ago]"), messages[1]["content"]

    def test_empty_history_still_builds(self):
        """The path that kept working, and therefore hid the bug — a first
        turn never enters the stamping branch."""
        messages = _agent([])._build_messages("first message")

        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_recent_messages_are_not_stamped(self):
        """MESSAGE_STAMP_MIN_AGE_SECONDS suppresses '[0m ago]' noise."""
        agent = _agent([
            _Msg(MessageRole.USER, "just now", NOW - timedelta(seconds=5)),
        ])

        assert agent._build_messages("q")[1]["content"] == "just now"

    def test_stamping_does_not_mutate_stored_history(self):
        stored = _Msg(MessageRole.USER, "hello", NOW - timedelta(hours=3))
        agent = _agent([stored])

        agent._build_messages("q")

        assert stored.content == "hello"

    def test_tool_messages_are_not_stamped(self):
        agent = _agent([
            _Msg(MessageRole.TOOL, "tool output", NOW - timedelta(hours=3)),
        ])

        assert agent._build_messages("q")[1]["content"] == "tool output"


class TestNoUndefinedNames:
    """A NameError on a hot path is invisible to a suite that never calls the
    function. Catch the whole class statically instead."""

    def test_package_has_no_undefined_names(self):
        pyflakes = pytest.importorskip(
            "pyflakes.api", reason="pip install pyflakes to run this check",
        )
        import io

        from pyflakes.reporter import Reporter

        # Reporter(warningStream, errorStream): flake findings go to the
        # WARNING stream. Reading the error stream instead makes this test
        # incapable of failing — it passed against the very NameError it
        # exists to catch.
        out = io.StringIO()
        pyflakes.checkRecursive(["blipshell"], Reporter(out, io.StringIO()))

        undefined = [
            line for line in out.getvalue().splitlines()
            if "undefined name" in line
            # Quoted forward-reference annotations resolve at runtime via a
            # local import; pyflakes cannot see those.
            and "PauseResult" not in line
        ]
        assert not undefined, "undefined names:\n" + "\n".join(undefined)
