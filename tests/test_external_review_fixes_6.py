"""Review finding 6 (2026-09-10): the whole-request budget is a HARD bound.

Trimmable fixed blocks are cut with traceable omissions; when the mandatory
content cannot fit, the builder refuses with an explicit error instead of
sending an over-limit request.
"""

from __future__ import annotations

import pytest

from blipshell.core.agent_chat import MIN_MEMORY_TOKENS
from blipshell.llm.exceptions import ContextOverflowError


async def _agent(tmp_path, name="a.db"):
    from blipshell.benchmark.continuity import bootstrap_headless_agent
    return await bootstrap_headless_agent(tmp_path / name)


async def _close(agent):
    try:
        await agent.session_manager.flush_pending_persists()
    except Exception:
        pass
    await agent.force_cleanup()


def _within(s):
    return s["request_tokens_estimate"] + s["response_reserve"] <= s["context_limit"]


async def test_mandatory_content_that_cannot_fit_is_refused_not_sent(tmp_path):
    agent, client = await _agent(tmp_path)
    try:
        await agent.start_session()
        agent.endpoint_manager._endpoints[0].context_tokens = 2000
        agent.config.agent.system_prompt = "word " * 4000
        client.sent.clear()
        reply = await agent.chat("hello")
        assert reply.startswith("Error:") and "does not fit" in reply
        assert client.sent == [], "nothing was sent to the model"
        with pytest.raises(ContextOverflowError):
            agent._build_messages("hello")
    finally:
        await _close(agent)


async def test_oversized_project_context_is_trimmed_with_a_traceable_omission(tmp_path):
    agent, client = await _agent(tmp_path)
    try:
        await agent.start_session()
        agent.endpoint_manager._endpoints[0].context_tokens = 6000
        agent.active_project = {"name": "big", "root_path": str(tmp_path)}
        agent._project_context = "=== README ===\n" + ("repo line of text\n" * 3000)
        client.sent.clear()
        reply = await agent.chat("hello")
        assert not reply.startswith("Error:")
        s = agent._last_context_stats
        assert any(o.startswith("project context truncated") for o in s["omitted_fixed"]), s["omitted_fixed"]
        assert _within(s), s
        assert s["available_tokens"] >= MIN_MEMORY_TOKENS
        sysmsg = client.sent[0][0]["content"]
        assert "[project context truncated:" in sysmsg
    finally:
        await _close(agent)


async def test_tiny_window_drops_tool_schemas_before_refusing(tmp_path):
    agent, client = await _agent(tmp_path)
    try:
        await agent.start_session()
        agent.endpoint_manager._endpoints[0].context_tokens = 2000
        client.sent.clear()
        reply = await agent.chat("hello")
        s = agent._last_context_stats
        if reply.startswith("Error:"):
            pytest.fail(f"default prompt should fit a 2000 window once tools are dropped: {reply}")
        assert _within(s), s
        assert s["available_tokens"] >= MIN_MEMORY_TOKENS
        # either everything fit, or the trim is recorded and tools were the last cut
        if s["omitted_fixed"]:
            assert s["tools_omitted"] is True
    finally:
        await _close(agent)


async def test_user_message_larger_than_the_window_is_refused(tmp_path):
    agent, client = await _agent(tmp_path)
    try:
        await agent.start_session()
        agent.endpoint_manager._endpoints[0].context_tokens = 4000
        client.sent.clear()
        reply = await agent.chat("x " * 6000)
        assert reply.startswith("Error:") and "context window" in reply
        assert client.sent == []
    finally:
        await _close(agent)


async def test_normal_request_records_no_omissions(tmp_path):
    agent, client = await _agent(tmp_path)
    try:
        await agent.start_session()
        await agent.chat("hello")
        s = agent._last_context_stats
        assert s["omitted_fixed"] == [] and s["tools_omitted"] is False and _within(s)
    finally:
        await _close(agent)
