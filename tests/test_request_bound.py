"""The executor's first request obeys the same hard bound as chat (review
finding 6, executor integration, 2026-09-10)."""

from __future__ import annotations

import pytest

from blipshell.core.request_bound import bound_initial_request, response_reserve_for
from blipshell.llm.exceptions import ContextOverflowError

TASK = {"role": "user", "content": "do the task"}
TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "x" * 200,
                                           "parameters": {"type": "object", "properties": {}}}} for i in range(10)]


def _est(bounded):
    return bounded.estimated_tokens + bounded.response_reserve


def test_fitting_request_is_untouched():
    b = bound_initial_request(base_prompt="You are BlipShell.", capability_context="cap", memory_context="- a memory",
                              continuity_context="\n\nnotes", chat_history=[{"role": "user", "content": "hi"}],
                              task_message=TASK, tools=TOOLS, context_limit=20000)
    assert b.omissions == [] and b.tools is TOOLS and len(b.messages) == 3
    assert "--- RELEVANT MEMORIES ---" in b.messages[0]["content"] and "notes" in b.messages[0]["content"]
    assert _est(b) <= 20000


def test_memory_is_trimmed_first_then_history_then_continuity_then_tools():
    big_memory = "memory line\n" * 2000          # ~6k tokens
    history = [{"role": "user", "content": "old turn " * 200} for _ in range(6)]  # ~2.4k tokens
    b = bound_initial_request(base_prompt="base", capability_context="", memory_context=big_memory,
                              continuity_context="\n\n[scratchpad] " + "n " * 300, chat_history=history,
                              task_message=TASK, tools=TOOLS, context_limit=4000)
    assert _est(b) <= 4000, b.omissions
    joined = " | ".join(b.omissions)
    assert "memory block" in joined
    # the mandatory parts survived
    assert b.messages[0]["content"].startswith("base") and b.messages[-1] is TASK
    # trimming stopped as early as it could: if tools survived, continuity was tried first
    order = [o.split(" ")[0] for o in b.omissions]
    assert order == sorted(order, key=["memory", "chat", "scratchpad/notes", "tool"].index)


def test_tools_are_the_last_cut_and_are_said_so():
    b = bound_initial_request(base_prompt="base " * 300, capability_context="", memory_context="",
                              continuity_context="", chat_history=None, task_message=TASK, tools=TOOLS,
                              context_limit=1000)
    assert b.tools is None and any(o.startswith("tool schemas omitted") for o in b.omissions)
    assert _est(b) <= 1000


def test_mandatory_content_that_cannot_fit_raises():
    with pytest.raises(ContextOverflowError) as e:
        bound_initial_request(base_prompt="word " * 3000, capability_context="", memory_context="x",
                              continuity_context="", chat_history=None, task_message=TASK, tools=None,
                              context_limit=2000)
    assert "does not fit" in str(e.value)


def test_reserve_formula_matches_chat():
    assert response_reserve_for(2000) == 256 and response_reserve_for(65536) == 2048 and response_reserve_for(8000) == 1000
