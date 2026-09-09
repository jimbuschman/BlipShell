"""The D2a evaluation boundary (approved 2026-09-09): a frozen, hand-labelled
set built against the current selection behaviour; versioned judge runs;
no tuning against the held labels; texts never leave data/."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory import attribution as att
from blipshell.memory import attribution_eval as ev
from blipshell.models.memory import Lesson


async def _seed(sqlite_store):
    """Two ordinary lessons, one historical detector lesson, one phase-1 corrections row."""
    now = datetime.now(timezone.utc)
    l1 = await sqlite_store.create_lesson(Lesson(content="Verify test results before claiming success.",
                                                 importance=0.9, timestamp=now - timedelta(days=10)))
    l2 = await sqlite_store.create_lesson(Lesson(content="Answer in the user's units.",
                                                 importance=0.5, timestamp=now - timedelta(days=9)))
    hist = await sqlite_store.create_lesson(Lesson(
        content=('ANTI-PATTERN: User corrected the assistant. Signal: "that\'s wrong". '
                 'Previous response (excerpt): "All tests pass...". '
                 'User said: "No, that\'s wrong - the tests did not pass."'),
        importance=0.5, timestamp=now - timedelta(days=5), added_by="correction_detector",
        source_type="user_statement"))
    cid = await att.record_correction(sqlite_store, session_id=None, turn_index=4,
                                      text="Actually I meant metres, not feet.",
                                      prev_assistant="That is 12 feet.", lessons_present=[l2])
    return l1, l2, hist, cid


class TestBuild:
    async def test_items_from_both_sources(self, sqlite_store, tmp_path):
        l1, l2, hist, cid = await _seed(sqlite_store)
        s = await ev.build_set(sqlite_store, generation="pre-D1", selection_behavior="pre-D1 pool", db_name="t.db")
        by = {i.item_id: i for i in s.items}
        assert f"c{cid}" in by and f"h{hist}" in by
        c = by[f"c{cid}"]
        assert c.source == "corrections_row" and c.lessons_present == [l2] and c.lessons_present_source == "recorded"
        assert c.lessons == [[l2, "Answer in the user's units."]]
        h = by[f"h{hist}"]
        assert h.text == "No, that's wrong - the tests did not pass."
        assert h.prev_assistant == "All tests pass"
        assert h.lessons_present_source == "reconstructed_top30"
        # the pool at the time: both ordinary lessons existed, the detector lesson itself is excluded
        assert set(h.lessons_present) == {l1, l2}
        assert h.lessons[0][0] == l1, "top by importance first"
        assert not s.frozen

    def test_parse_anti_pattern(self):
        text, prev = ev.parse_anti_pattern('ANTI-PATTERN: User corrected the assistant. Signal: "x". '
                                           'Previous response (excerpt): "Sure thing...". User said: "no I said tabs"')
        assert text == "no I said tabs" and prev == "Sure thing"
        assert ev.parse_anti_pattern("not a detector lesson") == (None, "")

    async def test_save_and_load_round_trip(self, sqlite_store, tmp_path):
        await _seed(sqlite_store)
        s = await ev.build_set(sqlite_store, generation="pre-D1", selection_behavior="x")
        p = ev.save_set(s, base=tmp_path)
        assert p.name == "pre-D1.json"
        again = ev.load_set("pre-D1", base=tmp_path)
        assert [i.item_id for i in again.items] == [i.item_id for i in s.items]
        assert again.selection_behavior == "x"


class TestLabelAndFreeze:
    async def test_label_freeze_and_read_only(self, sqlite_store, tmp_path):
        l1, l2, hist, cid = await _seed(sqlite_store)
        s = await ev.build_set(sqlite_store, generation="pre-D1", selection_behavior="x")
        with pytest.raises(ValueError):
            ev.freeze(s)  # nothing labelled yet
        with pytest.raises(ValueError):
            ev.label_item(s, f"c{cid}", "lesson_wrong", l1)  # l1 was not present for that item
        ev.label_item(s, f"c{cid}", "unrelated", None)
        ev.label_item(s, f"h{hist}", "lesson_ignored", l1)
        c = ev.freeze(s)
        assert s.frozen and c["labelled"] == 2 and c["positives_lesson_wrong"] == 0 and c["positives_ok"] is False
        with pytest.raises(PermissionError):
            ev.label_item(s, f"c{cid}", "lesson_wrong", l2)
        assert ev.freeze(s) == c  # idempotent

    def test_bad_attribution_rejected(self):
        s = ev.EvalSet(generation="g", selection_behavior="x", created_at="", db_name="")
        s.items.append(ev.EvalItem(item_id="a", source="corrections_row", source_id=1, text="t", prev_assistant="",
                                   lessons_present=[1], lessons_present_source="recorded", lessons=[[1, "l"]]))
        with pytest.raises(ValueError):
            ev.label_item(s, "a", "unattributed", 1)
        with pytest.raises(KeyError):
            ev.label_item(s, "zzz", "unrelated", None)


def _frozen_set(n_wrong: int = 12, n_other: int = 8) -> ev.EvalSet:
    s = ev.EvalSet(generation="pre-D1", selection_behavior="x", created_at="", db_name="")
    for k in range(n_wrong):
        s.items.append(ev.EvalItem(item_id=f"w{k}", source="corrections_row", source_id=k, text=f"wrong {k}",
                                   prev_assistant="", lessons_present=[1, 2], lessons_present_source="recorded",
                                   lessons=[[1, "lesson one"], [2, "lesson two"]],
                                   human_attribution="lesson_wrong", human_lesson_id=1))
    for k in range(n_other):
        s.items.append(ev.EvalItem(item_id=f"o{k}", source="corrections_row", source_id=100 + k, text=f"other {k}",
                                   prev_assistant="", lessons_present=[1, 2], lessons_present_source="recorded",
                                   lessons=[[1, "lesson one"], [2, "lesson two"]],
                                   human_attribution="unrelated", human_lesson_id=None))
    s.frozen_at = "2026-09-09T00:00:00+00:00"
    return s


def _router(reply_for):
    r = MagicMock()

    async def gen(task_type, prompt="", system=None, **kw):
        return reply_for(prompt)
    r.generate = AsyncMock(side_effect=gen)
    return r


class TestRun:
    async def test_refuses_unfrozen(self):
        s = _frozen_set()
        s.frozen_at = None
        with pytest.raises(PermissionError):
            await ev.run_judge(s, _router(lambda p: ""), repeats=1)

    async def test_perfect_judge_passes_the_gate(self):
        s = _frozen_set()

        def perfect(prompt):
            if "wrong " in prompt.split("User's correction:")[1]:
                return json.dumps({"lesson_id": 1, "attribution": "lesson_wrong", "confidence": 0.95})
            return json.dumps({"lesson_id": None, "attribution": "unrelated", "confidence": 0.9})
        res = await ev.run_judge(s, _router(perfect), repeats=3)
        assert res.mean_agreement["lesson_wrong"] == 1.0 and res.mean_agreement["unrelated"] == 1.0
        assert res.mean_fp_rate == 0.0 and res.spread_agreement["lesson_wrong"] == 0.0
        assert res.gate["passes"] is True
        assert len(res.verdicts) == 3 * len(s.items)
        assert res.judge_hash == ev.judge_hash()

    async def test_blaming_ignored_lessons_fails_the_fp_gate(self):
        """The failure that matters most: a correct lesson blamed for being ignored."""
        s = _frozen_set()
        always_wrong = lambda p: json.dumps({"lesson_id": 1, "attribution": "lesson_wrong", "confidence": 0.9})
        res = await ev.run_judge(s, _router(always_wrong), repeats=2)
        assert res.mean_agreement["lesson_wrong"] == 1.0
        assert res.mean_agreement["unrelated"] == 0.0
        assert res.mean_fp_rate == pytest.approx(8 / 20, abs=1e-3)
        assert res.gate["passes"] is False
        assert any("false-positive" in r for r in res.gate["reasons"])

    async def test_too_few_positives_is_not_meaningful(self):
        s = _frozen_set(n_wrong=3, n_other=5)
        perfect = lambda p: json.dumps({"lesson_id": 1, "attribution": "lesson_wrong", "confidence": 0.95}) \
            if "wrong " in p.split("User's correction:")[1] else \
            json.dumps({"lesson_id": None, "attribution": "unrelated", "confidence": 0.9})
        res = await ev.run_judge(s, _router(perfect), repeats=1)
        assert res.gate["passes"] is False
        assert any("positives" in r for r in res.gate["reasons"])

    async def test_write_run_keeps_texts_local_and_versions_the_judge(self, tmp_path, monkeypatch):
        s = _frozen_set()
        res = await ev.run_judge(s, _router(lambda p: ""), repeats=1)
        base = tmp_path / "local"
        summ = tmp_path / "results"
        full, summary = ev.write_run(res, base=base, summary_dir=summ, git_sha="abc123")
        full_rec = json.loads(full.read_text())
        assert full_rec["judge_version"] == "baseline" and full_rec["verdicts"]
        summ_rec = json.loads(summary.read_text())
        assert "verdicts" not in summ_rec and summ_rec["kind"] == "attribution_eval"
        assert "wrong 0" not in summary.read_text(), "no correction text in the committed summary"
        # a later run with a different judge is a NEW version, not the baseline
        monkeypatch.setattr(att, "JUDGE_SYSTEM", att.JUDGE_SYSTEM + " (revised)")
        res2 = await ev.run_judge(s, _router(lambda p: ""), repeats=1)
        res2.run_ts = res.run_ts + "b"
        full2, _ = ev.write_run(res2, base=base, summary_dir=summ, git_sha="abc124")
        assert json.loads(full2.read_text())["judge_version"].startswith("changed-from-")
        assert ev.render(res).startswith("generation=pre-D1")
