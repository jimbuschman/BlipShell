"""Both chat paths carry the same cross-session continuity.

`_chat_simple` assembled scratchpad, session notes, follow-ups and the time
anchor inline; `execute_dynamic` got none of them. So `!plan` — the path you
reach for when a task is HARD — ran with less continuity than ordinary
conversation, which is backwards for a system whose stated moat is memory
(deep-dive 2026-08-04). One builder now feeds both.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.chat_loop import LoopResult
from blipshell.core.executor import TaskExecutor
from blipshell.core.tools.base import ToolRegistry
from blipshell.models.config import PlannerConfig


class _Host(ChatMixin):
    """Minimal host for the builder."""

    def __init__(self, scratchpad="", notes=None, follow_ups=""):
        self._scratch = scratchpad
        self._session_notes = notes or {}
        self._pending_follow_ups = follow_ups

    def _read_scratchpad(self) -> str:
        return self._scratch


class TestContinuityBlock:
    def test_includes_scratchpad(self):
        block = _Host(scratchpad="decided: archive never delete")._build_continuity_block()
        assert "SCRATCHPAD" in block
        assert "archive never delete" in block

    def test_includes_session_notes_with_names(self):
        block = _Host(notes={"plan": "step one then two"})._build_continuity_block()
        assert "SESSION NOTES" in block
        assert "[plan]" in block
        assert "step one then two" in block

    def test_includes_follow_ups_when_asked(self):
        host = _Host(follow_ups="Pending: verify the migration ran")
        assert "verify the migration ran" in host._build_continuity_block()

    def test_follow_ups_can_be_excluded(self):
        """The chat path places them after its memory pools, so it opts out."""
        host = _Host(follow_ups="Pending: something")
        block = host._build_continuity_block(include_followups=False)
        assert "something" not in block

    def test_time_anchor_present_by_default(self):
        assert "Current date/time:" in _Host()._build_continuity_block()

    def test_time_anchor_can_be_excluded(self):
        block = _Host()._build_continuity_block(include_time=False)
        assert "Current date/time:" not in block

    def test_empty_state_yields_only_the_time_anchor(self):
        block = _Host()._build_continuity_block()
        assert block.strip().startswith("Current date/time:")

    def test_fully_empty_when_nothing_and_no_time(self):
        assert _Host()._build_continuity_block(include_time=False) == ""

    def test_missing_attributes_do_not_explode(self):
        """Built before the agent finished initializing."""
        class _Bare(ChatMixin):
            def _read_scratchpad(self):
                return ""

        assert "Current date/time:" in _Bare()._build_continuity_block()


class TestExecutorReceivesIt:
    async def test_continuity_reaches_the_executor_system_prompt(self):
        captured = {}

        async def _runner(messages, config, on_token=None, on_tool_executed=None):
            captured["system"] = messages[0]["content"]
            return LoopResult(response="done", messages=list(messages)), "e", "m", False

        ex = TaskExecutor(
            router=MagicMock(), sqlite=MagicMock(),
            tool_registry=ToolRegistry(), config=PlannerConfig(),
        )
        ex.chat_loop_runner = _runner
        ex.router._endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=None)

        await ex.execute_dynamic(
            "refactor the thing",
            continuity_context="\n\n--- SCRATCHPAD ---\ndecided: archive never delete",
        )

        assert "SCRATCHPAD" in captured["system"]
        assert "archive never delete" in captured["system"]

    async def test_executor_works_without_continuity(self):
        """Callers that don't pass it (benchmark harness) must still run."""
        async def _runner(messages, config, on_token=None, on_tool_executed=None):
            return LoopResult(response="ok", messages=list(messages)), "e", "m", False

        ex = TaskExecutor(
            router=MagicMock(), sqlite=MagicMock(),
            tool_registry=ToolRegistry(), config=PlannerConfig(),
        )
        ex.chat_loop_runner = _runner
        ex.router._endpoint_manager.get_endpoint_for_role = AsyncMock(return_value=None)

        assert await ex.execute_dynamic("do a thing") == "ok"


class TestBothPathsAgree:
    def test_same_builder_feeds_both(self):
        """The point of the extraction: one source of truth. If chat and the
        executor built this separately they would drift, which is how the gap
        appeared in the first place."""
        host = _Host(
            scratchpad="a decision",
            notes={"n": "a note"},
            follow_ups="Pending: a follow-up",
        )
        executor_block = host._build_continuity_block()
        chat_block = host._build_continuity_block(
            include_followups=False, include_time=False,
        )

        # The executor takes the whole thing; chat takes the same pieces minus
        # the two it positions itself.
        for fragment in ("a decision", "a note"):
            assert fragment in executor_block
            assert fragment in chat_block
        assert "a follow-up" in executor_block
        assert "Current date/time:" in executor_block
