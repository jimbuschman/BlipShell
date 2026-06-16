"""friction_analysis per-session timeout.

The job had a between-sessions time budget but no per-call bound, so one slow
REASONING call blocked the loop past _JOB_TIMEOUT (the budget check only runs
between sessions). These tests pin the fix: a session whose analysis exceeds
the per-session timeout is abandoned and the loop CONTINUES — using a real
asyncio.wait_for against a real over-long await (shrunk timeout), not a faked
one. They validate loop-progress-on-timeout, which holds whether or not the
underlying call is truly cancellable.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from blipshell.core.nightly import NightlyRunner


def _runner(processor, sqlite):
    return NightlyRunner(
        config=Mock(), sqlite=sqlite, vectors=Mock(),
        router=Mock(), processor=processor,
    )


def _sessions(*ids):
    return [{"id": i, "summary": f"summary {i}", "project": None} for i in ids]


async def test_slow_session_times_out_and_loop_continues(monkeypatch):
    monkeypatch.setattr("blipshell.core.nightly._FRICTION_SESSION_TIMEOUT", 0.05)

    sqlite = Mock()
    sqlite.get_sessions_missing_friction_analysis = AsyncMock(
        return_value=_sessions(1, 2, 3),
    )
    sqlite.add_friction_entry = AsyncMock()

    processor = Mock()
    processor.prepare_conversation_for_reflection = AsyncMock(
        return_value=(["conversation text"], None),
    )

    async def analyze(session_id, session_summary, conversation_text, project):
        if session_id == 2:
            await asyncio.sleep(5)   # hung call — must NOT block the loop
            return []
        return [{"source": "test", "category": "workflow", "description": "d"}]

    processor.analyze_session_friction = analyze

    result = await asyncio.wait_for(
        _runner(processor, sqlite)._job_friction_analysis(on_status=lambda *_: None),
        timeout=3.0,   # the whole job must finish fast despite the 5s hung call
    )

    assert result["timed_out"] == 1          # session 2 abandoned
    assert result["processed"] == 2          # sessions 1 and 3 still done
    assert result["friction_items"] == 2
    assert result["total"] == 3


async def test_no_chunks_session_is_left_unmarked():
    sqlite = Mock()
    sqlite.get_sessions_missing_friction_analysis = AsyncMock(
        return_value=_sessions(1),
    )
    sqlite.add_friction_entry = AsyncMock()

    processor = Mock()
    # No chunks -> nothing to analyze -> session must not be marked/processed.
    processor.prepare_conversation_for_reflection = AsyncMock(return_value=([], None))
    processor.analyze_session_friction = AsyncMock()

    result = await _runner(processor, sqlite)._job_friction_analysis(
        on_status=lambda *_: None,
    )

    assert result["processed"] == 0
    assert result["timed_out"] == 0
    processor.analyze_session_friction.assert_not_called()
    sqlite.add_friction_entry.assert_not_called()
