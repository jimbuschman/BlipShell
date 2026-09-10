"""One representation of the conversation; one copy of each memory; a budget
for the whole request (V3_PLAN Stage B1).

Before: SessionManager.add_message fed every turn into the ActiveSession pool
(rendered inside the system message) AND _build_messages appended the last
20 turns as role messages - the current user turn appeared twice (review F4).
Recent-session content reached the request twice too: once via Recall with
a time label, once via RecentHistory without (continuity baseline: 16
duplicated renders over 13 cases). And the pool budget was computed against
`context_limit - user - 1000`, ignoring the system prefix, the tool schemas
and any room for the answer.

Now: the conversation is role messages only, windowed by the ActiveSession
token share (oldest excluded turns are summarised into RecentHistory once);
a memory that Recall found is skipped when RecentHistory offers the same
memory_id; and the pools are budgeted against what is left after the prefix,
tools and a response reserve.

Uses the headless real agent from the continuity harness for the end-to-end
checks and the real MemoryManager/SessionManager for the unit ones.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from blipshell.benchmark import continuity
from blipshell.memory.manager import MemoryManager, Pool, PoolItem
from blipshell.models.config import MemoryConfig
from blipshell.models.session import MessageRole


# ---------------------------------------------------------------------------
# Pool / manager units
# ---------------------------------------------------------------------------

class TestRecallDedup:

    def test_get_top_entries_can_exclude_memory_ids(self):
        pool = Pool("RecentHistory", 1000)
        pool.add(PoolItem(text="a", priority_score=3, memory_id=7))
        pool.add(PoolItem(text="b", priority_score=2, memory_id=8))
        pool.add(PoolItem(text="c", priority_score=1))  # no id: never excluded
        got = pool.get_top_entries(1000, exclude_keys={("memory", 7)})
        assert [i.text for i in got] == ["b", "c"]

    def test_gather_skips_history_copies_of_recalled_memories(self, memory_config):
        mm = MemoryManager(memory_config, context_tokens=8000)
        mm.add_memory("Recall", PoolItem(text="[2d ago] user: the pi is at .77", priority_score=1.0, memory_id=42, source="memory"))
        mm.add_memory("RecentHistory", PoolItem(text="[2d ago] the pi is at .77", priority_score=2.5, memory_id=42, source="history"))
        mm.add_memory("RecentHistory", PoolItem(text="[3d ago] unrelated", priority_score=2.0, memory_id=43, source="history"))
        items = mm.gather_memory(token_budget=4000)
        by_pool = {}
        for i in items:
            by_pool.setdefault(i.pool_name, []).append(i.memory_id)
        assert by_pool["Recall"] == [42]
        assert by_pool["RecentHistory"] == [43], "the RecentHistory copy of memory 42 must be skipped, not the unrelated one"

    def test_history_budget_is_not_consumed_by_the_skipped_copy(self, memory_config):
        mm = MemoryManager(memory_config, context_tokens=8000)
        mm.add_memory("Recall", PoolItem(text="x " * 40, priority_score=1.0, memory_id=1))
        # RecentHistory: the duplicate is highest priority; a small cap would
        # have been eaten by it before the skip existed
        mm.add_memory("RecentHistory", PoolItem(text="x " * 40, priority_score=9.0, memory_id=1))
        mm.add_memory("RecentHistory", PoolItem(text="keep me", priority_score=1.0, memory_id=2))
        items = mm.gather_memory(token_budget=4000, pool_budgets={"RecentHistory": 10, "Recall": 400,
                                                                  "Core": 100, "Lessons": 100, "ActiveSession": 100})
        hist = [i for i in items if i.pool_name == "RecentHistory"]
        assert [i.memory_id for i in hist] == [2]


class TestOverflowSummary:

    async def test_schedule_overflow_summary_lands_in_recent_history(self, memory_config):
        mm = MemoryManager(memory_config, context_tokens=8000)
        seen = []

        async def summarize(text):
            seen.append(text)
            return "SUMMARY: " + text[:20]

        mm.set_summarize_callback(summarize)
        task = mm.schedule_overflow_summary("user: old turn one assistant: old reply")
        assert task is not None
        await task
        assert seen == ["user: old turn one assistant: old reply"]
        texts = [i.text for i in mm.get_pool("RecentHistory")._items]
        assert any(t.startswith("SUMMARY:") for t in texts)

    def test_schedule_without_callback_or_text_is_a_noop(self, memory_config):
        mm = MemoryManager(memory_config, context_tokens=8000)
        assert mm.schedule_overflow_summary("anything") is None
        mm.set_summarize_callback(AsyncMock(return_value="s"))
        assert mm.schedule_overflow_summary("   ") is None


# ---------------------------------------------------------------------------
# End to end on the headless real agent
# ---------------------------------------------------------------------------

async def _agent(tmp_path):
    return await continuity.bootstrap_headless_agent(tmp_path / "ctx.db")


async def _close(agent):
    await agent.session_manager.flush_pending_persists()
    await agent.sqlite.close()
    agent.vectors.close()


def _system_text(request: list[dict]) -> str:
    return "\n".join(m["content"] for m in request if m["role"] == "system")


class TestOneConversation:

    async def test_the_user_turn_appears_once_as_a_role_message(self, tmp_path):
        agent, client = await _agent(tmp_path)
        try:
            await agent.start_session()
            await agent.chat("My cat is named Luna and she is grey.")
            client.sent.clear()
            await agent.chat("What colour is Luna?")
            req = client.sent[0]
            user_msgs = [m for m in req if m["role"] == "user" and "What colour is Luna" in m["content"]]
            assert len(user_msgs) == 1
            assert "What colour is Luna" not in _system_text(req), "the current turn was rendered inside the system message"
            # the previous turn is history, once, as role messages
            assert sum("named Luna" in m["content"] for m in req if m["role"] == "user") == 1
            assert "named Luna" not in _system_text(req)
            assert agent.memory_manager.get_pool("ActiveSession").item_count == 0
        finally:
            await _close(agent)

    async def test_history_window_is_token_budgeted_and_overflow_is_summarised_once(self, tmp_path):
        agent, client = await _agent(tmp_path)
        try:
            await agent.start_session()
            scheduled = []
            orig = agent.memory_manager.schedule_overflow_summary

            def spy(text):
                scheduled.append(text)
                return orig(text)

            agent.memory_manager.schedule_overflow_summary = spy
            # shrink the history share so a handful of turns overflows it
            from blipshell.memory import query_profiles
            orig_budgets = query_profiles.compute_pool_budgets

            def tiny(profile, total, caps):
                b = orig_budgets(profile, total, caps)
                b["ActiveSession"] = 60  # tokens
                return b

            import blipshell.core.agent_chat as ac
            monkey = pytest.MonkeyPatch()
            monkey.setattr(ac, "compute_pool_budgets", tiny)
            try:
                for i in range(6):
                    await agent.chat(f"turn number {i} about topic {i} with some more words to spend tokens")
                req = client.sent[-1]
                history = [m for m in req if m["role"] in ("user", "assistant")]
                assert len(history) < 12, "every turn was sent regardless of the budget"
                assert history[-1]["role"] == "user" and "turn number 5" in history[-1]["content"]
                assert scheduled, "excluded turns were never summarised"
                joined = "\n".join(scheduled)
                assert "turn number 0" in joined
                # each excluded turn is summarised exactly once across turns
                assert joined.count("turn number 0 about") == 1
                stats = agent._last_context_stats
                assert stats["history_dropped"] > 0 and stats["history_messages"] == len(history)
            finally:
                monkey.undo()
        finally:
            await _close(agent)


class TestWholeRequestBudget:

    async def test_stats_account_for_prefix_tools_and_reserve(self, tmp_path):
        agent, client = await _agent(tmp_path)
        try:
            await agent.start_session()
            await agent.chat("hello")
            s = agent._last_context_stats
            for key in ("context_limit", "prefix_tokens", "tools_tokens", "response_reserve",
                        "history_tokens", "available_tokens", "request_tokens_estimate"):
                assert key in s, key
            assert s["prefix_tokens"] > 0
            assert s["response_reserve"] > 0
            assert s["request_tokens_estimate"] + s["response_reserve"] <= s["context_limit"]
            # available for pools is what is left after the fixed parts
            assert s["available_tokens"] < s["context_limit"] - s["prefix_tokens"] - s["tools_tokens"]
        finally:
            await _close(agent)

    async def test_tiny_context_still_leaves_a_floor_for_memory(self, tmp_path):
        agent, client = await _agent(tmp_path)
        try:
            agent.endpoint_manager._endpoints[0].context_tokens = 2000
            await agent.start_session()
            await agent.chat("hello")
            s = agent._last_context_stats
            assert s["available_tokens"] >= 256
        finally:
            await _close(agent)


class TestContinuityGauge:

    async def test_recent_session_content_is_rendered_once(self):
        cases = {c.name: c for c in continuity._load_dataset("benchmark_continuity").CASES}
        r = await continuity.run_case(cases["fact_split_across_sessions"])
        assert r.passed
        assert r.duplicated_renders == 0, "recall and recent-history rendered the same memory"
