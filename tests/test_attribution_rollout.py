"""Phase-1 rollout toggles (V3 D2a): collection is logging only and on by
default; the background judge is opt-in. Neither path touches a lesson."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.models.config import AttributionConfig, BlipShellConfig
from blipshell.models.memory import Lesson


def _host(sqlite_store, *, enabled=True, judge_enabled=False, lessons_sent=()):
    host = SimpleNamespace(
        config=SimpleNamespace(attribution=AttributionConfig(enabled=enabled, judge_enabled=judge_enabled)),
        sqlite=sqlite_store,
        router=SimpleNamespace(generate=AsyncMock(return_value='{"attribution": "unrelated", "lesson_id": null, "confidence": 0.9}')),
        session_manager=SimpleNamespace(session_id=None),
        _turn_number=3,
        _last_lessons_sent=list(lessons_sent),
        _background_tasks=set(),
    )
    host._attribution_setting = ChatMixin._attribution_setting.__get__(host)
    host._record_lesson_uses = ChatMixin._record_lesson_uses.__get__(host)
    host._record_correction_for_attribution = ChatMixin._record_correction_for_attribution.__get__(host)
    return host


async def _rows(sqlite_store, table):
    cur = await sqlite_store._db.execute(f"SELECT * FROM {table}")
    return [dict(r) for r in await cur.fetchall()]


def test_defaults_are_collection_only():
    cfg = BlipShellConfig()
    assert cfg.attribution.enabled is True and cfg.attribution.judge_enabled is False


async def test_collection_writes_rows_without_a_model_call(sqlite_store):
    lid = await sqlite_store.create_lesson(Lesson(content="Answer in the user's units.", importance=0.5))
    host = _host(sqlite_store, lessons_sent=[(lid, "always_on")])
    await host._record_lesson_uses()
    await host._record_correction_for_attribution("Actually I meant metres.", "That is 12 feet.")
    assert len(await _rows(sqlite_store, "lesson_uses")) == 1
    corr = await _rows(sqlite_store, "corrections")
    assert len(corr) == 1 and corr[0]["attribution"] == "unattributed"
    assert host._background_tasks == set(), "no judge task in the collection-only rollout"
    host.router.generate.assert_not_awaited()
    lesson = await sqlite_store.get_lesson(lid)
    assert lesson.importance == 0.5


async def test_judge_runs_only_when_enabled(sqlite_store):
    lid = await sqlite_store.create_lesson(Lesson(content="Answer in the user's units.", importance=0.5))
    host = _host(sqlite_store, judge_enabled=True, lessons_sent=[(lid, "always_on")])
    await host._record_correction_for_attribution("Actually I meant metres.", "That is 12 feet.")
    assert host._background_tasks, "judge task scheduled"
    await asyncio.gather(*host._background_tasks)
    host.router.generate.assert_awaited()
    corr = await _rows(sqlite_store, "corrections")
    assert corr[0]["attribution"] == "unrelated"
    assert (await sqlite_store.get_lesson(lid)).importance == 0.5, "the verdict changes no lesson"


async def test_disabled_writes_nothing(sqlite_store):
    lid = await sqlite_store.create_lesson(Lesson(content="x", importance=0.5))
    host = _host(sqlite_store, enabled=False, judge_enabled=True, lessons_sent=[(lid, "always_on")])
    await host._record_lesson_uses()
    await host._record_correction_for_attribution("Actually I meant metres.", "That is 12 feet.")
    assert await _rows(sqlite_store, "lesson_uses") == [] and await _rows(sqlite_store, "corrections") == []
    assert host._background_tasks == set()
