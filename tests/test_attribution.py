"""Correction attribution, phase 1: RECORD ONLY (V3 D2a, approved 2026-09-09).

What is asserted, in order of importance:
1. Nothing about any lesson changes - importance, status, selection - because
   of anything here. Phase 2 is gated on the evaluation and explicit approval.
2. lesson_uses records which lessons were in each turn's request.
3. An accepted correction becomes a row carrying the lessons that were
   present on the corrected turn, and the local judge's verdict is stored
   beside it, strictly parsed, unattributed on anything ambiguous or below
   the confidence floor.
4. The readout renders rows, per-lesson counts, and the agreement gate.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory import attribution as att
from blipshell.models.memory import Lesson


# ---------------------------------------------------------------- parse

class TestParseVerdict:
    def test_valid(self):
        v = att.parse_verdict(json.dumps({"lesson_id": 7, "attribution": "lesson_ignored",
                                          "confidence": 0.9, "reason": "it said verify first"}), [7, 8])
        assert (v.attribution, v.lesson_id, v.confidence) == ("lesson_ignored", 7, 0.9)

    def test_unrelated_drops_lesson_id(self):
        v = att.parse_verdict('{"lesson_id": 7, "attribution": "unrelated", "confidence": 0.95}', [7])
        assert v.attribution == "unrelated" and v.lesson_id is None

    @pytest.mark.parametrize("raw", [
        "",
        "lesson_wrong",
        '{"attribution": "lesson_wrong", "confidence": 0.9}',                    # no lesson id
        '{"lesson_id": 99, "attribution": "lesson_wrong", "confidence": 0.9}',   # lesson not present
        '{"lesson_id": 7, "attribution": "demote", "confidence": 0.9}',          # unknown class
        '{"lesson_id": 7, "attribution": "lesson_wrong", "confidence": 1.5}',    # out of range
        '{"lesson_id": 7, "attribution": "lesson_wrong", "confidence": true}',
        '{"lesson_id": true, "attribution": "lesson_wrong", "confidence": 0.9}',
        '{"lesson_id": 7, "attribution": "unattributed", "confidence": 0.9}',    # judge may not claim this
        '[{"lesson_id": 7}]',
    ])
    def test_ambiguous_is_unattributed(self, raw):
        v = att.parse_verdict(raw, [7])
        assert v.attribution == "unattributed" and v.lesson_id is None

    def test_low_confidence_is_unattributed_but_keeps_the_raw(self):
        raw = '{"lesson_id": 7, "attribution": "lesson_wrong", "confidence": 0.55, "reason": "maybe"}'
        v = att.parse_verdict(raw, [7])
        assert v.attribution == "unattributed" and v.lesson_id is None
        assert v.confidence == 0.55 and v.raw == raw, "the evaluation needs the judge's own answer"

    def test_fenced_json_tolerated(self):
        v = att.parse_verdict('```json\n{"lesson_id": 7, "attribution": "lesson_wrong", "confidence": 0.8}\n```', [7])
        assert v.attribution == "lesson_wrong"


# ---------------------------------------------------------------- records + judge (record only)

async def _lessons_snapshot(sqlite_store):
    cur = await sqlite_store._db.execute("SELECT id, content, importance, rank, source_type, verification_state, added_by FROM lessons ORDER BY id")
    return [tuple(r) for r in await cur.fetchall()]


class TestRecordOnly:
    async def test_attribute_correction_stores_verdict_and_touches_no_lesson(self, sqlite_store):
        l1 = await sqlite_store.create_lesson(Lesson(content="Verify before claiming success.", importance=0.7, source_type="reflection"))
        l2 = await sqlite_store.create_lesson(Lesson(content="Prefer short answers.", importance=0.6, source_type="user_statement"))
        before = await _lessons_snapshot(sqlite_store)
        cid = await att.record_correction(sqlite_store, session_id=None, turn_index=3,
                                          text="No, that's wrong - you said the tests passed and they did not.",
                                          prev_assistant="All tests pass.", lessons_present=[l1, l2])
        router = MagicMock()
        router.generate = AsyncMock(return_value=json.dumps(
            {"lesson_id": l1, "attribution": "lesson_ignored", "confidence": 0.92, "reason": "it did not verify"}))

        v = await att.attribute_correction(sqlite_store, router, cid)

        assert v.attribution == "lesson_ignored" and v.lesson_id == l1
        row = await att.get_correction(sqlite_store, cid)
        assert row["attribution"] == "lesson_ignored" and row["lesson_id"] == l1 and row["judged_by"] == "local_judge"
        assert json.loads(row["lessons_present"]) == [l1, l2]
        assert await _lessons_snapshot(sqlite_store) == before, "phase 1 changes no lesson"
        # the judge saw exactly the present lessons, with ids
        prompt = router.generate.await_args.args[1]
        assert f"[{l1}] Verify before" in prompt and f"[{l2}] Prefer short" in prompt

    async def test_judge_failure_is_recorded_unattributed(self, sqlite_store):
        l1 = await sqlite_store.create_lesson(Lesson(content="x", importance=0.5))
        cid = await att.record_correction(sqlite_store, session_id=None, turn_index=1, text="that's wrong",
                                          prev_assistant="", lessons_present=[l1])
        router = MagicMock()
        router.generate = AsyncMock(side_effect=RuntimeError("model down"))
        v = await att.attribute_correction(sqlite_store, router, cid)
        assert v.attribution == "unattributed"
        assert (await att.get_correction(sqlite_store, cid))["attribution"] == "unattributed"

    async def test_no_lessons_present_is_unrelated_without_a_call(self, sqlite_store):
        cid = await att.record_correction(sqlite_store, session_id=None, turn_index=1, text="that's wrong",
                                          prev_assistant="", lessons_present=[])
        router = MagicMock()
        router.generate = AsyncMock()
        v = await att.attribute_correction(sqlite_store, router, cid)
        assert v.attribution == "unrelated"
        router.generate.assert_not_awaited()

    async def test_record_lesson_uses(self, sqlite_store):
        n = await att.record_lesson_uses(sqlite_store, session_id=5, turn_index=2, uses=[(1, "pool"), (2, "recall"), (0, "pool")])
        assert n == 2
        cur = await sqlite_store._db.execute("SELECT lesson_id, selected_by FROM lesson_uses ORDER BY lesson_id")
        assert [tuple(r) for r in await cur.fetchall()] == [(1, "pool"), (2, "recall")]


# ---------------------------------------------------------------- end to end on the headless agent

async def _close(agent):
    await agent.session_manager.flush_pending_persists()
    await agent.sqlite.close()
    agent.vectors.close()


class TestAgentWiring:
    async def test_lesson_uses_are_recorded_per_turn(self, tmp_path):
        from blipshell.benchmark import continuity
        agent, client = await continuity.bootstrap_headless_agent(tmp_path / "uses.db")
        try:
            l1 = await agent.sqlite.create_lesson(Lesson(content="Profile before optimising.", importance=0.9, source_type="reflection"))
            l2 = await agent.sqlite.create_lesson(Lesson(content="Answer in the user's units.", importance=0.8, source_type="user_statement"))
            await agent.start_session()
            await agent.chat("hello there, what should I profile first?")
            cur = await agent.sqlite._db.execute("SELECT lesson_id, selected_by, turn_index FROM lesson_uses ORDER BY lesson_id")
            rows = [tuple(r) for r in await cur.fetchall()]
            assert {r[0] for r in rows} == {l1, l2}
            assert all(r[1] == "pool" for r in rows)  # always-on Lessons pool, this turn
            assert all(r[2] == 1 for r in rows)
        finally:
            await _close(agent)

    async def test_correction_creates_a_row_with_the_previous_turns_lessons_and_a_verdict(self, tmp_path):
        from blipshell.benchmark import continuity
        agent, client = await continuity.bootstrap_headless_agent(tmp_path / "corr.db")
        try:
            l1 = await agent.sqlite.create_lesson(Lesson(content="Verify test results before claiming success.", importance=0.9, source_type="reflection"))
            before = await _lessons_snapshot(agent.sqlite)

            async def gen(task_type, prompt="", system=None, **kw):
                s = (system or "")
                if s.startswith("You screen one chat message"):       # correction judge
                    return "YES"
                if s.startswith("You attribute ONE user correction"):  # attribution judge
                    return json.dumps({"lesson_id": l1, "attribution": "lesson_ignored", "confidence": 0.9, "reason": "r"})
                return continuity.canned_generate(task_type, prompt=prompt, system=system, **kw)
            agent.router.generate = AsyncMock(side_effect=gen)

            await agent.start_session()
            await agent.chat("did the tests pass?")                       # turn 1: lessons present
            await agent.chat("No, that's wrong - the tests did not pass.")  # turn 2: a correction
            task = getattr(agent, "_last_attribution_task", None)
            assert task is not None
            await task

            cur = await agent.sqlite._db.execute("SELECT * FROM corrections")
            rows = [dict(r) for r in await cur.fetchall()]
            assert len(rows) == 1
            row = rows[0]
            assert "did not pass" in row["text"]
            assert json.loads(row["lessons_present"]) == [l1], "the lessons present on the CORRECTED turn"
            assert row["attribution"] == "lesson_ignored" and row["lesson_id"] == l1
            assert row["human_attribution"] is None
            # phase 1 is record only: every lesson that existed before is byte-identical.
            # (The existing detector still mints its anti-pattern lesson as a NEW row -
            # unchanged behaviour, asserted below - so compare the pre-existing rows.)
            after_snapshot = await _lessons_snapshot(agent.sqlite)
            assert after_snapshot[:len(before)] == before, "phase 1 changed an existing lesson"
            after = await agent.sqlite.get_all_lessons()
            assert any(l.added_by == "correction_detector" for l in after)
        finally:
            await _close(agent)


# ---------------------------------------------------------------- readout

class TestReadout:
    async def test_render_and_agreement(self, sqlite_store, tmp_path):
        import sqlite3
        from scripts import attribution_readout as ro
        l1 = await sqlite_store.create_lesson(Lesson(content="A", importance=0.5))
        for i in range(3):
            cid = await att.record_correction(sqlite_store, session_id=None, turn_index=i, text=f"correction {i}",
                                              prev_assistant="", lessons_present=[l1])
            await sqlite_store._db.execute(
                "UPDATE corrections SET attribution='lesson_wrong', lesson_id=?, confidence=0.9 WHERE id=?", (l1, cid))
        await sqlite_store._db.commit()
        db_path = sqlite_store.db_path
        # human labels: two agree, one says the lesson was merely ignored
        ro.label(str(db_path), 1, "lesson_wrong", l1)
        ro.label(str(db_path), 2, "lesson_wrong", l1)
        ro.label(str(db_path), 3, "lesson_ignored", l1)
        con = sqlite3.connect(str(db_path)); con.row_factory = sqlite3.Row
        rows = ro.load_rows(con); con.close()
        a = ro.agreement(rows)
        assert a["labelled"] == 3 and a["positives_lesson_wrong"] == 2 and a["meaningful"] is False
        assert a["per_class"]["lesson_wrong"]["agreement"] == 1.0
        assert a["per_class"]["lesson_ignored"]["agreement"] == 0.0
        assert a["lesson_wrong_false_positive_rate"] == pytest.approx(1 / 3, abs=1e-3)
        text = ro.render(rows)
        assert "corrections: 3" in text and "phase 2 gate" in text and "need >= 10" in text
