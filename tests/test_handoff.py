"""Session handoff: the working-state note across the process boundary.

The continuity feature BlipShell asked for (2026-09-02): a note-to-self at
session close, loaded at next boot ahead of the digests. These tests pin the
pure logic and drive the real write/load methods with fakes. The `enabled`
toggle is the pre-registered A/B switch for the continuity probe.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from blipshell.core.agent import Agent
from blipshell.core.agent_session import SessionMixin
from blipshell.core.handoff import (
    HANDOFF_KEY,
    HANDOFF_META_KEY,
    clean_note,
    frame_for_boot,
    handoff_prompt,
    is_stale,
    should_generate,
    transcript_tail,
)
from blipshell.models.config import BlipShellConfig


# --- pure half ----------------------------------------------------------------

def _msg(role, content):
    return SimpleNamespace(role=role, content=content)


def test_transcript_tail_takes_the_end_and_caps():
    messages = [_msg("user", f"message {i}") for i in range(50)]
    tail = transcript_tail(messages, max_messages=5)
    assert "message 49" in tail and "message 44" not in tail
    long = [_msg("user", "x" * 1000)] * 20
    assert len(transcript_tail(long, max_chars=3000)) <= 3000


def test_should_generate_skips_trivial_sessions():
    assert should_generate(1) is False
    assert should_generate(4) is True


def test_clean_note_handles_decline_and_caps():
    assert clean_note("NOTHING") is None
    assert clean_note("Nothing pressing here.") is None
    assert clean_note("") is None
    assert clean_note(None) is None
    assert clean_note("  I was mid-refactor on the tile editor. ") == \
        "I was mid-refactor on the tile editor."
    assert len(clean_note("y" * 5000)) == 1400


def test_is_stale():
    now = datetime(2026, 9, 3, tzinfo=timezone.utc)
    fresh = (now - timedelta(days=2)).isoformat()
    old = (now - timedelta(days=20)).isoformat()
    assert is_stale(fresh, now) is False
    assert is_stale(old, now) is True
    assert is_stale(None, now) is True
    assert is_stale("not-a-date", now) is True


def test_frame_reads_as_its_own_note():
    framed = frame_for_boot("Mid-flight on the raycaster.", "2026-09-02T20:00:00+00:00")
    assert "note to yourself" in framed
    assert "(2026-09-02)" in framed
    assert "Mid-flight on the raycaster." in framed
    assert "prompt" not in framed.lower()  # no mechanism-speak in-stream


def test_prompt_is_state_not_recap():
    p = handoff_prompt("user: hello")
    assert "note to your next self" in p.lower() or "note" in p.lower()


# --- write side (Agent._write_session_handoff) ---------------------------------

def _writer_agent(sqlite, reply="I was mid-thread on X.", n_messages=6,
                  enabled=True):
    fake = SimpleNamespace()
    fake.config = BlipShellConfig()
    fake.config.handoff.enabled = enabled
    fake.sqlite = sqlite
    fake.router = MagicMock()
    fake.router.generate = AsyncMock(return_value=reply)
    fake.session_manager = MagicMock()
    fake.session_manager.session_id = 7
    fake.session_manager.get_messages.return_value = [
        _msg("user", f"m{i}") for i in range(n_messages)]
    return fake


async def test_write_persists_note_and_meta(sqlite_store):
    fake = _writer_agent(sqlite_store)
    await Agent._write_session_handoff(fake)
    assert await sqlite_store.get_metadata(HANDOFF_KEY) == "I was mid-thread on X."
    meta = json.loads(await sqlite_store.get_metadata(HANDOFF_META_KEY))
    assert meta["session_id"] == 7 and meta["saved_at"]


async def test_write_skips_trivial_and_declined(sqlite_store):
    await Agent._write_session_handoff(_writer_agent(sqlite_store, n_messages=2))
    assert await sqlite_store.get_metadata(HANDOFF_KEY) is None
    await Agent._write_session_handoff(_writer_agent(sqlite_store, reply="NOTHING"))
    assert await sqlite_store.get_metadata(HANDOFF_KEY) is None


async def test_write_respects_toggle(sqlite_store):
    fake = _writer_agent(sqlite_store, enabled=False)
    await Agent._write_session_handoff(fake)
    fake.router.generate.assert_not_awaited()


# --- load side (SessionMixin._load_handoff) -------------------------------------

def _loader_agent(sqlite, enabled=True, session_id=99):
    fake = SimpleNamespace()
    fake.config = BlipShellConfig()
    fake.config.handoff.enabled = enabled
    fake.sqlite = sqlite
    fake.memory_manager = MagicMock()
    fake.session_manager = MagicMock()
    fake.session_manager.session_id = session_id
    return fake


async def _seed_note(sqlite, session_id=7, days_ago=1):
    saved = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    await sqlite.set_metadata(HANDOFF_KEY, "Mid-flight on the tile editor.")
    await sqlite.set_metadata(HANDOFF_META_KEY, json.dumps(
        {"saved_at": saved, "session_id": session_id}))


async def test_load_adds_framed_note_between_core_and_user_model(sqlite_store):
    await _seed_note(sqlite_store)
    fake = _loader_agent(sqlite_store)
    await SessionMixin._load_handoff(fake)
    (pool, item), _ = fake.memory_manager.add_memory.call_args
    assert pool == "Core"
    assert "Mid-flight on the tile editor." in item.text
    assert "note to yourself" in item.text
    assert 0.9 < item.priority_score < 1.0  # above user model, below core facts


async def test_load_skips_stale_own_session_absent_and_disabled(sqlite_store):
    fake = _loader_agent(sqlite_store)
    await SessionMixin._load_handoff(fake)            # no note at all
    fake.memory_manager.add_memory.assert_not_called()

    await _seed_note(sqlite_store, days_ago=30)       # stale
    await SessionMixin._load_handoff(fake)
    fake.memory_manager.add_memory.assert_not_called()

    await _seed_note(sqlite_store, session_id=99)     # its own closing note
    await SessionMixin._load_handoff(fake)
    fake.memory_manager.add_memory.assert_not_called()

    await _seed_note(sqlite_store, session_id=7)      # fresh + other session,
    disabled = _loader_agent(sqlite_store, enabled=False)   # but toggled off
    await SessionMixin._load_handoff(disabled)
    disabled.memory_manager.add_memory.assert_not_called()
