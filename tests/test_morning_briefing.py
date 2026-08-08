"""Morning briefing (V2_PLAN 5.2): first turn of the day carries the
overnight digest — including failures. The trust property is that a bad
night CANNOT read as a good one, and a shown briefing is never repeated
the same day.
"""

import json
from datetime import date

import pytest

from blipshell.core.morning_briefing import LAST_SHOWN_KEY, build_briefing


async def _store_report(sqlite, report):
    await sqlite.set_metadata("nightly_report", json.dumps(report))


class TestBriefing:
    async def test_clean_night_reads_clean(self, sqlite_store):
        await _store_report(sqlite_store, {
            "timestamp": 1754800000, "errors": [], "warnings": [],
            "summary": {"consolidate": {"merged": 7}},
        })

        b = await build_briefing(sqlite_store)

        assert b is not None
        assert "ran clean" in b
        assert "7 near-duplicate memories merged" in b

    async def test_failures_are_front_and_center(self, sqlite_store):
        """The reason this feature exists: nightly failures reach the user
        without being asked for."""
        await _store_report(sqlite_store, {
            "timestamp": 1754800000,
            "errors": ["backup: disk full", "consolidate: timed out after 300s"],
            "warnings": ["3 job(s) skipped (Ollama down): a, b, c"],
            "summary": {},
        })

        b = await build_briefing(sqlite_store)

        assert "problems" in b
        assert "backup: disk full" in b
        assert "Ollama down" in b
        assert "ran clean" not in b

    async def test_shown_once_per_day(self, sqlite_store):
        await _store_report(sqlite_store, {
            "timestamp": 1754800000, "errors": [], "warnings": [], "summary": {},
        })

        first = await build_briefing(sqlite_store)
        second = await build_briefing(sqlite_store)

        assert first is not None
        assert second is None, "the same day's briefing repeated"
        assert await sqlite_store.get_metadata(LAST_SHOWN_KEY) == date.today().isoformat()

    async def test_stale_stamp_from_yesterday_shows_again(self, sqlite_store):
        await _store_report(sqlite_store, {
            "timestamp": 1754800000, "errors": [], "warnings": [], "summary": {},
        })
        await sqlite_store.set_metadata(LAST_SHOWN_KEY, "2020-01-01")

        assert await build_briefing(sqlite_store) is not None

    async def test_no_report_says_so_instead_of_silence(self, sqlite_store):
        """A machine where nightly never ran should ask for /nightly, not
        pretend all is well."""
        b = await build_briefing(sqlite_store)
        assert b is not None
        assert "/nightly" in b

    async def test_user_model_revision_is_mentioned(self, sqlite_store):
        await _store_report(sqlite_store, {
            "timestamp": 1754800000, "errors": [], "warnings": [],
            "summary": {"update_user_model": {
                "status": "ok", "revised": True, "lines": 9, "evidence": 4,
            }},
        })

        b = await build_briefing(sqlite_store)

        assert "working model of the user was revised" in b

    async def test_overflow_of_problems_is_counted_not_dropped(self, sqlite_store):
        await _store_report(sqlite_store, {
            "timestamp": 1754800000,
            "errors": [f"job{i}: boom" for i in range(6)],
            "warnings": [f"warn{i}" for i in range(5)],
            "summary": {},
        })

        b = await build_briefing(sqlite_store)

        assert "...and 4 more" in b
