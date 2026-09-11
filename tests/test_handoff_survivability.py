"""What survives an abnormal session end, verified against persisted data
after a RESTART (a second agent on the same database).

Corpus facts these mirror (2026-09-02 snapshot): sessions 1919 and 1920
ended with message_count 0 and last_active == created_at - end_session never
ran - and the orphan sweep skipped them because it keyed on message_count.
"""

from __future__ import annotations

import json

import pytest

from blipshell.core.handoff import HANDOFF_KEY, HANDOFF_META_KEY


async def _boot(tmp_path, name="s.db", reply="ok"):
    from blipshell.benchmark.continuity import bootstrap_headless_agent
    return await bootstrap_headless_agent(tmp_path / name, reply=reply)


def _recent_history_text(agent) -> str:
    return "\n".join(i.text for i in agent.memory_manager.get_pool("RecentHistory")._items)


def _core_text(agent) -> str:
    return "\n".join(i.text for i in agent.memory_manager.get_pool("Core")._items)


class TestEndBeforeFirstRefresh:
    async def test_no_note_but_the_stop_block_and_an_orphan_summary_survive(self, tmp_path):
        agent, client = await _boot(tmp_path)
        agent.config.handoff.refresh_every_turns = 6
        await agent.start_session()
        sid = agent.session_manager.session_id
        for i in range(3):  # ends before turn 6, no end_session: a kill
            await agent.chat(f"turn {i}: about the raycaster column renderer")
        await agent.session_manager.flush_pending_persists()
        await agent.force_cleanup()

        agent2, client2 = await _boot(tmp_path)
        try:
            assert await agent2.sqlite.get_metadata(HANDOFF_KEY) is None, "nothing was written before the first refresh"
            row = await agent2.sqlite.get_session(sid)
            assert row.message_count == 0 and not row.summary, "the orphan as the corpus shows it"
            await agent2.start_session()
            rh = _recent_history_text(agent2)
            assert "Where the last session stopped" in rh and "turn 2: about the raycaster column renderer" in rh
            row = await agent2.sqlite.get_session(sid)
            assert row.summary and row.message_count == 6, "the orphan sweep now judges by persisted memories"
        finally:
            await agent2.force_cleanup()


class TestRefreshSurvivesAKill:
    async def test_note_written_mid_session_is_loaded_after_restart(self, tmp_path):
        agent, client = await _boot(tmp_path)
        agent.config.handoff.refresh_every_turns = 2
        await agent.start_session()
        sid = agent.session_manager.session_id
        await agent.chat("first turn about the raycaster")
        await agent.chat("second turn, still on the raycaster")
        await agent._last_handoff_refresh_task
        await agent.session_manager.flush_pending_persists()
        await agent.force_cleanup()  # no end_session

        agent2, client2 = await _boot(tmp_path)
        try:
            meta = json.loads(await agent2.sqlite.get_metadata(HANDOFF_META_KEY))
            assert meta["session_id"] == sid and meta["midsession"] is True and meta["turn"] == 2
            await agent2.start_session()
            core = _core_text(agent2)
            assert "Your note to yourself from the end of your previous session" in core
            assert "left no note" not in core, "the note IS from the most recent session"
        finally:
            await agent2.force_cleanup()


class TestFailedWrite:
    async def test_atomic_write_leaves_both_keys_or_neither(self, sqlite_store, monkeypatch):
        s = sqlite_store
        await s.set_metadata_many({HANDOFF_KEY: "old note", HANDOFF_META_KEY: json.dumps({"session_id": 1})})
        real_execute = s._db.execute
        calls = {"n": 0}

        async def flaky(sql, *a, **k):
            if "INSERT OR REPLACE INTO app_metadata" in sql:
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("disk full")
            return await real_execute(sql, *a, **k)
        monkeypatch.setattr(s._db, "execute", flaky)
        with pytest.raises(RuntimeError):
            await s.set_metadata_many({HANDOFF_KEY: "new note", HANDOFF_META_KEY: json.dumps({"session_id": 2})})
        monkeypatch.undo()
        assert await s.get_metadata(HANDOFF_KEY) == "old note"
        assert json.loads(await s.get_metadata(HANDOFF_META_KEY))["session_id"] == 1

    async def test_older_note_is_framed_as_such_when_the_last_session_left_none(self, tmp_path):
        # session A writes a note (refresh at turn 2), then session B runs 3 turns and is killed
        agent, _ = await _boot(tmp_path)
        agent.config.handoff.refresh_every_turns = 2
        await agent.start_session()
        sid_a = agent.session_manager.session_id
        await agent.chat("A one about the raycaster")
        await agent.chat("A two about the raycaster")
        await agent._last_handoff_refresh_task
        await agent.session_manager.flush_pending_persists()
        await agent.force_cleanup()

        agent_b, _ = await _boot(tmp_path)
        agent_b.config.handoff.refresh_every_turns = 6
        await agent_b.start_session()
        sid_b = agent_b.session_manager.session_id
        await agent_b.chat("B one about the digest export")
        await agent_b.chat("B two about the digest export")
        await agent_b.session_manager.flush_pending_persists()
        await agent_b.force_cleanup()  # killed before any refresh

        agent_c, _ = await _boot(tmp_path)
        try:
            await agent_c.start_session()
            meta = json.loads(await agent_c.sqlite.get_metadata(HANDOFF_META_KEY))
            assert meta["session_id"] == sid_a
            core = _core_text(agent_c)
            assert "Your note to yourself" in core and "left no note" in core, core
            rh = _recent_history_text(agent_c)
            assert "B two about the digest export" in rh, "the stop block still carries where B stopped"
            row = await agent_c.sqlite.get_session(sid_b)
            assert row.last_active > row.timestamp, "last_active is touched per turn, so B sorts as most recent"
        finally:
            await agent_c.force_cleanup()

    async def test_failed_generation_keeps_the_previous_note(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock
        agent, _ = await _boot(tmp_path)
        try:
            await agent.sqlite.set_metadata_many({HANDOFF_KEY: "earlier note", HANDOFF_META_KEY: json.dumps({"session_id": 1, "saved_at": "2026-09-01T00:00:00+00:00"})})
            await agent.start_session()
            for i in range(4):
                agent.session_manager.add_message(__import__("blipshell.models.session", fromlist=["MessageRole"]).MessageRole.USER, f"m{i}")
            monkeypatch.setattr(agent.router, "generate", AsyncMock(side_effect=RuntimeError("model down")))
            await agent._write_session_handoff(midsession=True)
            assert agent._last_handoff_written is False
            assert await agent.sqlite.get_metadata(HANDOFF_KEY) == "earlier note"
        finally:
            await agent.force_cleanup()
