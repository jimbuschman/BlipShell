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
        assert block.startswith("Where the last session stopped (2026-09-02), in order, word for word")
        assert block.splitlines()[1:] == ["user: c", "assistant: d, half-formed", "user: e?", "assistant: f - I had not finished"]

    def test_too_short_is_none(self):
        assert handoff.stop_block([SimpleNamespace(role="user", content="hi")]) is None
        assert handoff.stop_block([]) is None


class TestTheEndOfALongTurnSurvives:
    """Review 2026-09-11, finding 1 (reproduction inverted): both handoff
    inputs kept only each message's first 400 characters, so a turn that
    explains for a while and THEN says what is unfinished lost the
    unfinished part - in the block the next boot reads and in the transcript
    the note is written from. The end of the turn is the state."""

    LONG_ASSISTANT = ("Background explanation about the renderer. " * 20 +
                      "UNFINISHED_THREAD_739: next investigate clipping, not shading.")
    LONG_USER = ("Some context about the export job that I keep repeating. " * 20 +
                 "CORRECTION_411: no, we rejected hourly exports.")

    def _rows(self):
        return [SimpleNamespace(role="user", content=self.LONG_USER, id=11),
                SimpleNamespace(role="assistant", content=self.LONG_ASSISTANT, id=12)]

    def test_stop_block_keeps_the_end_of_both_roles(self):
        block = handoff.stop_block(self._rows(), max_pairs=2)
        assert "UNFINISHED_THREAD_739: next investigate clipping, not shading." in block
        assert "CORRECTION_411: no, we rejected hourly exports." in block

    def test_transcript_tail_keeps_the_end_of_both_roles(self):
        tail = handoff.transcript_tail(self._rows(), max_chars=1600)
        assert "UNFINISHED_THREAD_739: next investigate clipping, not shading." in tail
        assert "CORRECTION_411: no, we rejected hourly exports." in tail

    def test_the_opening_is_kept_too_so_the_end_has_an_antecedent(self):
        block = handoff.stop_block(self._rows(), max_pairs=2)
        assert "Background explanation about the renderer." in block
        assert "Some context about the export job" in block

    def test_an_excerpt_is_labelled_with_its_memory_id_and_not_called_verbatim(self):
        block = handoff.stop_block(self._rows(), max_pairs=2)
        head = block.splitlines()[0]
        assert "excerpted" in head and "word for word" not in head
        assert "assistant [excerpt of memory 12]:" in block
        assert "chars elided" in block

    def test_short_turns_are_still_word_for_word(self):
        rows = [SimpleNamespace(role="user", content="short one", id=1),
                SimpleNamespace(role="assistant", content="short two", id=2)]
        block = handoff.stop_block(rows, max_pairs=2)
        assert "word for word" in block.splitlines()[0]
        assert "excerpt" not in block

    def test_packing_stays_inside_the_budget_and_drops_the_oldest_first(self):
        rows = [SimpleNamespace(role="user", content=f"turn {i} " + "x" * 900, id=i)
                for i in range(6)]
        block = handoff.stop_block(rows, max_pairs=3, total_chars=600)
        assert len(block) <= 600 + len(block.splitlines()[0])
        assert "turn 5" in block, "the most recent turn is never the one dropped"
        assert "turn 0" not in block and "oldest turns omitted" in block.splitlines()[0]

    def test_a_short_turn_hands_its_surplus_to_a_long_one(self):
        rows = [SimpleNamespace(role="user", content="ok", id=1),
                SimpleNamespace(role="assistant", content="y" * 4000, id=2)]
        generous = handoff.stop_block(rows, max_pairs=1, total_chars=2000)
        assert len(generous.splitlines()[2]) > 1500, "the long turn gets the unused share"


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


class TestTheExcerptPointerIsRedeemable:
    """The stop block names the memory an excerpt came from; that pointer is
    only honest if something can fetch the whole turn. `search_memories`
    takes `memory_ids` (review 2026-09-11, finding 1: "permit retrieval of
    the full exchange when needed")."""

    async def _tool(self, sqlite_store):
        from types import SimpleNamespace as NS
        from blipshell.core.tools.memory_tools import SearchMemoriesTool
        return SearchMemoriesTool(NS(sqlite=sqlite_store))

    async def test_a_named_memory_comes_back_whole(self, sqlite_store):
        sid = await sqlite_store.create_session("prev")
        long_turn = ("Background on the renderer. " * 120) + "UNFINISHED: next investigate clipping."
        mid = await sqlite_store.save_raw_memory(sid, "assistant", long_turn)
        block = handoff.stop_block(
            [SimpleNamespace(role="user", content="where next?", id=mid - 1),
             SimpleNamespace(role="assistant", content=long_turn, id=mid)], max_pairs=2)
        assert f"[excerpt of memory {mid}]" in block

        out = await (await self._tool(sqlite_store)).execute(query="", memory_ids=str(mid))
        assert long_turn in out and "chars elided" not in out

    async def test_several_ids_and_a_missing_one(self, sqlite_store):
        sid = await sqlite_store.create_session("prev")
        a = await sqlite_store.save_raw_memory(sid, "user", "first turn")
        b = await sqlite_store.save_raw_memory(sid, "assistant", "second turn")
        out = await (await self._tool(sqlite_store)).execute(query="", memory_ids=f"{a}, {b}, 999999")
        assert "first turn" in out and "second turn" in out and "[memory 999999: not found]" in out

    async def test_a_non_numeric_id_is_a_tool_failure(self, sqlite_store):
        from blipshell.core.tools.base import ToolFailure
        out = await (await self._tool(sqlite_store)).execute(query="", memory_ids="the last one")
        assert isinstance(out, ToolFailure)
