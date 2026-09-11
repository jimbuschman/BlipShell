"""Conversation continuity across sessions (2026-09-10 investigation).

Grounded in the corpus: sessions 1919/1920 (2026-08-11) - "a fresh
instantiation that read the notes... the information is there, the
continuity isn't" - ended with message_count 0 (no summary, and a close-only
handoff would have been lost); session 1926 (2026-09-02) - "no texture
carrying over... a clean cold start every time" - booted with the previous
substantive session's TOP-IMPORTANCE lines and a third-person summary, not
the exchange it stopped on; its Recall for "do you remember" returned old
meta-questions about forgetting.

Three deterministic changes: the working-state note is refreshed
mid-session; the previous session's last exchanges are carried verbatim as a
stop block; a continuity question gets a budget profile that favours where
we stopped over similar old memories.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blipshell.core import handoff
from blipshell.memory.query_profiles import PROFILES, classify_query, compute_pool_budgets


class TestStopBlock:
    def test_last_exchanges_verbatim_in_order(self):
        mems = [SimpleNamespace(role="user", content="a"), SimpleNamespace(role="assistant", content="b"),
                SimpleNamespace(role="user", content="c"), SimpleNamespace(role="assistant", content="d, half-formed"),
                SimpleNamespace(role="user", content="e?"), SimpleNamespace(role="assistant", content="f - I had not finished")]
        block = handoff.stop_block(mems, saved_when="2026-09-02", max_pairs=2)
        assert block.startswith("Where the last session stopped (2026-09-02), verbatim:")
        assert block.splitlines()[1:] == ["user: c", "assistant: d, half-formed", "user: e?", "assistant: f - I had not finished"]

    def test_too_short_is_none(self):
        assert handoff.stop_block([SimpleNamespace(role="user", content="hi")]) is None
        assert handoff.stop_block([]) is None


class TestContinuityProfile:
    def test_continuity_questions_classify_first(self):
        for q in ("do you remember the last thing we talked about?", "Where did we leave off?", "what were we in the middle of",
                  "so, picking back up where we stopped", "last time you said the note was written at close"):
            assert classify_query(q) == "continuity", q
        assert classify_query("Do you remember my cat's name?") == "recall"
        assert classify_query("ok") == "session"

    def test_profile_favours_where_we_stopped(self):
        p = PROFILES["continuity"]
        assert p["RecentHistory"] > p["Recall"] and p["Core"] > PROFILES["balanced"]["Core"]
        b = compute_pool_budgets("continuity", 10000, {"Core": None, "Lessons": None, "ActiveSession": None,
                                                        "RecentHistory": None, "Recall": None})
        assert b["RecentHistory"] > b["Recall"]


class TestBootCarriesTheLiveThread:
    async def test_stop_block_and_end_of_session_lines_reach_recent_history(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        from blipshell.models.memory import Memory
        agent, client = await bootstrap_headless_agent(tmp_path / "boot.db")
        try:
            then = datetime.now(timezone.utc) - timedelta(days=2)
            sid = await agent.sqlite.create_session(title="prev", created_at=then)
            # 24 high-importance filler lines, then the low-importance final exchange
            for i in range(24):
                await agent.sqlite.create_memory(Memory(session_id=sid, role="user" if i % 2 == 0 else "assistant",
                                                        content=f"filler point {i} about the column renderer",
                                                        summary=f"filler {i}", importance=0.8, rank=3,
                                                        timestamp=then + timedelta(minutes=i)))
            await agent.sqlite.create_memory(Memory(session_id=sid, role="user", content="so the note only gets written at close?",
                                                    summary="q", importance=0.35, rank=3, timestamp=then + timedelta(minutes=30)))
            await agent.sqlite.create_memory(Memory(session_id=sid, role="assistant",
                                                    content="what if the state note were written every few turns instead - HALF_FORMED_739",
                                                    summary="a", importance=0.35, rank=3, timestamp=then + timedelta(minutes=31)))
            await agent.start_session()
            texts = [i.text for i in agent.memory_manager.get_pool("RecentHistory")._items]
            joined = "\n".join(texts)
            assert "Where the last session stopped" in joined and "HALF_FORMED_739" in joined
            assert "so the note only gets written at close?" in joined
            client.sent.clear()
            await agent.chat("do you remember the last thing we talked about?")
            sysmsg = client.sent[0][0]["content"]
            assert "HALF_FORMED_739" in sysmsg
            assert agent._last_context_stats["query_profile"] == "continuity"
        finally:
            await agent.force_cleanup()


class TestLiveHandoffRefresh:
    async def test_note_is_refreshed_every_n_turns_in_the_background(self, tmp_path):
        import asyncio
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        agent, client = await bootstrap_headless_agent(tmp_path / "live.db")
        try:
            agent.config.handoff.refresh_every_turns = 2
            await agent.start_session()
            await agent.chat("first turn about the raycaster")
            assert await agent.sqlite.get_metadata(handoff.HANDOFF_KEY) is None, "turn 1: no refresh yet"
            await agent.chat("second turn, still on the raycaster")
            task = agent._last_handoff_refresh_task
            assert task is not None
            await task
            note = await agent.sqlite.get_metadata(handoff.HANDOFF_KEY)
            meta = json.loads(await agent.sqlite.get_metadata(handoff.HANDOFF_META_KEY))
            assert note and meta["midsession"] is True and meta["turn"] == 2
            assert meta["session_id"] == agent.session_manager.session_id
        finally:
            await agent.force_cleanup()

    async def test_disabled_by_zero(self, tmp_path):
        from blipshell.benchmark.continuity import bootstrap_headless_agent
        agent, client = await bootstrap_headless_agent(tmp_path / "off.db")
        try:
            agent.config.handoff.refresh_every_turns = 0
            await agent.start_session()
            for _ in range(4):
                await agent.chat("turn")
            assert agent._last_handoff_refresh_task is None
        finally:
            await agent.force_cleanup()
