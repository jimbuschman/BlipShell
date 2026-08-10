"""On-return reflection: the quiet gap happened while the process was off.

The idle loop only sees gaps while the app RUNS; on open-chat-close usage
that produced ~1 thought/month, making the self-gravity step-2 gate ("10
new thoughts") a year away. Startup after a 3h+ absence now counts as the
end of a quiet gap. The character rule under test: ONE thought per gap —
restarting three times across one absence is one return, not three.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.core.agent import Agent
from blipshell.core.self_reflection import should_reflect_on_return

NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
IDLE = 10800.0  # the 3h default


class TestDecision:
    def test_long_gap_returns_a_stamp(self):
        last = NOW - timedelta(hours=5)
        stamp = should_reflect_on_return(last, None, IDLE, now=NOW)
        assert stamp == last.isoformat()

    def test_short_gap_is_not_a_return(self):
        assert should_reflect_on_return(
            NOW - timedelta(minutes=30), None, IDLE, now=NOW) is None

    def test_same_gap_reflects_once(self):
        """Three restarts across one absence are ONE return."""
        last = NOW - timedelta(hours=5)
        stamp = should_reflect_on_return(last, None, IDLE, now=NOW)
        again = should_reflect_on_return(last, stamp, IDLE, now=NOW)
        assert again is None

    def test_a_new_gap_reflects_again(self):
        old_gap = NOW - timedelta(days=3)
        old_stamp = should_reflect_on_return(old_gap, None, IDLE, now=NOW)
        new_gap = NOW - timedelta(hours=4)
        assert should_reflect_on_return(new_gap, old_stamp, IDLE, now=NOW) \
            == new_gap.isoformat()

    def test_fresh_install_never_reflects(self):
        assert should_reflect_on_return(None, None, IDLE, now=NOW) is None

    def test_iso_string_input(self):
        last = (NOW - timedelta(hours=6)).isoformat()
        assert should_reflect_on_return(last, None, IDLE, now=NOW) is not None

    def test_naive_timestamp_treated_as_utc(self):
        last = (NOW - timedelta(hours=6)).replace(tzinfo=None)
        assert should_reflect_on_return(last, None, IDLE, now=NOW) is not None

    def test_garbage_timestamp_is_safe(self):
        assert should_reflect_on_return("not a date", None, IDLE, now=NOW) is None


def _agent(last_active, marker=None, current_session_id=None):
    a = Agent.__new__(Agent)
    a.config = MagicMock()
    a.config.reflection.idle_seconds = IDLE
    a.sqlite = MagicMock()
    last = MagicMock(last_active=last_active) if last_active else None
    a.sqlite.get_latest_session = AsyncMock(return_value=last)
    a.sqlite.get_metadata = AsyncMock(return_value=marker)
    a.sqlite.set_metadata = AsyncMock()
    a.session_manager = (
        MagicMock(session_id=current_session_id) if current_session_id else None
    )
    a._generate_lingering_thought = AsyncMock()
    return a


class TestAgentWiring:
    async def test_long_gap_generates_and_stamps(self):
        last = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        a = _agent(last)

        await a._reflect_on_return()

        a._generate_lingering_thought.assert_awaited_once()
        a.sqlite.set_metadata.assert_awaited_once()
        # Stamped BEFORE generating: a failed generation must not retry on
        # every restart of the same gap.
        assert a.sqlite.set_metadata.await_args.args[0] == \
            Agent._RETURN_REFLECTION_MARKER

    async def test_short_gap_generates_nothing(self):
        last = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        a = _agent(last)

        await a._reflect_on_return()

        a._generate_lingering_thought.assert_not_awaited()
        a.sqlite.set_metadata.assert_not_awaited()

    async def test_current_session_is_excluded_from_the_gap_question(self):
        """This task races start_session — a just-created session's
        last_active is 'now', which would read as gap=0 forever."""
        last = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        a = _agent(last, current_session_id=42)

        await a._reflect_on_return()

        assert a.sqlite.get_latest_session.await_args.kwargs == {"exclude_id": 42}
        a._generate_lingering_thought.assert_awaited_once()

    async def test_failure_is_contained(self):
        a = _agent((datetime.now(timezone.utc) - timedelta(hours=5)).isoformat())
        a.sqlite.get_latest_session = AsyncMock(side_effect=RuntimeError("db"))

        await a._reflect_on_return()   # must not raise

        a._generate_lingering_thought.assert_not_awaited()
