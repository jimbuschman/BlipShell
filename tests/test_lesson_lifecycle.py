"""Lesson lifecycle: nightly family folding + evidence revoting.

The 2026-09-02 audit's prevention layer. Folding keeps paraphrase duplicates
from re-accumulating (the one-shot cleanup removed 321 by hand); revoting
gives lessons the evidence lifecycle they never had — importance was scored
once at extraction and never revisited, which is how false and stale guidance
kept its seat in the top-30 pool for months.
"""

from unittest.mock import AsyncMock, MagicMock

from blipshell.core.nightly import _OLLAMA_JOBS, JOB_ORDER, NightlyRunner
from blipshell.memory.lesson_revote import (
    CONFIRMS,
    CONTRADICTS,
    NEUTRAL,
    adjusted_importance,
    parse_verdict,
    revote_prompt,
)
from blipshell.models.config import BlipShellConfig, MemoryConfig
from blipshell.models.memory import Lesson


# --- pure half ----------------------------------------------------------------

def test_parse_verdict_strict():
    assert parse_verdict("CONFIRMS") == CONFIRMS
    assert parse_verdict("contradicts.") == CONTRADICTS
    assert parse_verdict("Neutral — unrelated topic") == NEUTRAL
    assert parse_verdict("The lesson seems fine") is None
    assert parse_verdict("") is None
    assert parse_verdict(None) is None


def test_adjusted_importance_asymmetric_and_clamped():
    assert adjusted_importance(0.5, CONFIRMS, 0.05, 0.15) == 0.55
    assert adjusted_importance(0.5, CONTRADICTS, 0.05, 0.15) == 0.35
    # Down moves harder than up by default: stale standing instruction is
    # costlier than a slow climb.
    assert adjusted_importance(0.98, CONFIRMS, 0.05, 0.15) == 1.0   # ceiling
    assert adjusted_importance(0.15, CONTRADICTS, 0.05, 0.15) == 0.1  # floor
    assert adjusted_importance(0.5, NEUTRAL, 0.05, 0.15) == 0.5


def test_prompt_carries_both_sides():
    p = revote_prompt("User prefers X.", "Session showed Y.")
    assert "User prefers X." in p and "Session showed Y." in p


# --- revote job ---------------------------------------------------------------

def _runner(sqlite, memory_cfg, verdicts):
    config = BlipShellConfig(memory=memory_cfg)
    router = MagicMock()
    router.generate = AsyncMock(side_effect=list(verdicts))
    vectors = MagicMock()
    runner = NightlyRunner(config, sqlite, vectors=vectors, router=router,
                           processor=None)
    return runner, router, vectors


def test_revote_registered():
    assert "revote_lessons" in JOB_ORDER
    assert "revote_lessons" in _OLLAMA_JOBS  # local judge calls


async def test_revote_disabled_by_default(sqlite_store):
    runner, router, _ = _runner(sqlite_store, MemoryConfig(), [])
    result = await runner._job_revote_lessons(lambda m: None)
    assert result["skipped"] == "lesson_revote_enabled=false"
    router.generate.assert_not_awaited()


async def _seed(sqlite_store):
    lid = await sqlite_store.create_lesson(Lesson(
        content="User prefers lengthy comprehensive answers.",
        rank=4, importance=0.7))
    sid = await sqlite_store.create_session(title="s")
    await sqlite_store.create_session_reflection(
        session_id=sid, effectiveness="effective",
        reflection_text="User asked for shorter replies; trimming worked.")
    return lid


async def test_revote_dry_run_reports_but_does_not_write(sqlite_store):
    lid = await _seed(sqlite_store)
    cfg = MemoryConfig(lesson_revote_enabled=True)  # dry_run defaults True
    runner, _, vectors = _runner(sqlite_store, cfg, ["CONTRADICTS"])
    vectors.search_lessons.return_value = [{"id": lid, "similarity": 0.8}]

    result = await runner._job_revote_lessons(lambda m: None)
    assert result["contradicts"] == 1 and result["dry_run"] is True
    assert result["votes"][0]["new"] == 0.55
    lessons = await sqlite_store.get_all_lessons()
    assert lessons[0].importance == 0.7  # untouched


async def test_revote_apply_writes_and_watermark_advances(sqlite_store):
    lid = await _seed(sqlite_store)
    cfg = MemoryConfig(lesson_revote_enabled=True, lesson_revote_dry_run=False)
    runner, _, vectors = _runner(sqlite_store, cfg, ["CONTRADICTS"])
    vectors.search_lessons.return_value = [{"id": lid, "similarity": 0.8}]

    result = await runner._job_revote_lessons(lambda m: None)
    assert result["dry_run"] is False
    lessons = await sqlite_store.get_all_lessons()
    assert lessons[0].importance == 0.55
    # Watermark advanced: the same evidence is not re-judged next run.
    result2 = await runner._job_revote_lessons(lambda m: None)
    assert result2 == {"reflections": 0, "pairs": 0}


async def test_revote_judge_failure_votes_nothing(sqlite_store):
    lid = await _seed(sqlite_store)
    cfg = MemoryConfig(lesson_revote_enabled=True, lesson_revote_dry_run=False)
    runner, _, vectors = _runner(sqlite_store, cfg, ["I think maybe..."])
    vectors.search_lessons.return_value = [{"id": lid, "similarity": 0.8}]

    result = await runner._job_revote_lessons(lambda m: None)
    assert result["no_verdict"] == 1
    lessons = await sqlite_store.get_all_lessons()
    assert lessons[0].importance == 0.7


# --- family folding in clean_junk_lessons ---------------------------------------

async def test_fold_removes_paraphrases_keeps_distinct(sqlite_store, tmp_path):
    # Two blatant paraphrases + one distinct lesson.
    a = await sqlite_store.create_lesson(Lesson(
        content="User prefers direct practical code solutions over lengthy "
                "theoretical explanations when debugging.", importance=0.7))
    b = await sqlite_store.create_lesson(Lesson(
        content="User prefers direct practical code solutions rather than "
                "lengthy theoretical explanations while debugging errors.",
        importance=0.8))
    c = await sqlite_store.create_lesson(Lesson(
        content="Validate feelings before advice in emotionally heavy "
                "conversations about relationships.", importance=0.7))

    config = BlipShellConfig(memory=MemoryConfig())
    config.database.path = str(tmp_path / "blipshell.db")  # receipts land here
    vectors = MagicMock()
    runner = NightlyRunner(config, sqlite_store, vectors=vectors,
                           router=MagicMock(), processor=None)
    result = await runner._job_clean_junk_lessons(lambda m: None)

    assert result["folded"] == 1
    remaining = {l.id for l in await sqlite_store.get_all_lessons()}
    assert remaining == {b, c}  # b kept (higher importance), c untouched
    receipts = list(tmp_path.glob("lessons_folded_*.json"))
    assert len(receipts) == 1 and str(a) in receipts[0].read_text(encoding="utf-8")


async def test_fold_disabled_by_zero_threshold(sqlite_store, tmp_path):
    await sqlite_store.create_lesson(Lesson(content="Same lesson text here about coding.", importance=0.7))
    await sqlite_store.create_lesson(Lesson(content="Same lesson text here about coding again.", importance=0.7))
    config = BlipShellConfig(memory=MemoryConfig(lesson_family_fold_threshold=0.0))
    config.database.path = str(tmp_path / "blipshell.db")
    runner = NightlyRunner(config, sqlite_store, vectors=MagicMock(),
                           router=MagicMock(), processor=None)
    result = await runner._job_clean_junk_lessons(lambda m: None)
    assert result["folded"] == 0
    assert len(await sqlite_store.get_all_lessons()) == 2


# --- backfill exclusions (2026-09-03 timeout follow-up) -------------------------

async def test_folded_sessions_are_excluded_from_backfill(sqlite_store):
    """Deleting a duplicate lesson must not make its session look
    never-extracted — that re-creates the duplicate via backfill_lessons."""
    from blipshell.models.memory import Memory

    sid = await sqlite_store.create_session(title="s")
    for i in range(6):
        await sqlite_store.create_memory(Memory(
            session_id=sid, role="user", content=f"substantive message {i} here",
            importance=0.5))
    # No lessons -> session is missing one.
    missing = await sqlite_store.get_sessions_missing_lessons()
    assert any(s["id"] == sid for s in missing)
    # Marked as deliberately consolidated -> no longer "missing".
    await sqlite_store.add_lesson_backfill_exclusions([sid])
    missing = await sqlite_store.get_sessions_missing_lessons()
    assert not any(s["id"] == sid for s in missing)
    # Idempotent, accumulative, None-tolerant.
    n = await sqlite_store.add_lesson_backfill_exclusions([sid, None, 999])
    assert n == 2
    assert await sqlite_store.get_lesson_backfill_exclusions() == {sid, 999}


async def test_nightly_fold_records_exclusions(sqlite_store, tmp_path):
    from blipshell.models.memory import Lesson

    sid = await sqlite_store.create_session(title="dup source")
    await sqlite_store.create_lesson(Lesson(
        content="User prefers direct practical code solutions over lengthy "
                "theoretical explanations when debugging.", importance=0.7,
        source_session_id=sid))
    await sqlite_store.create_lesson(Lesson(
        content="User prefers direct practical code solutions rather than "
                "lengthy theoretical explanations while debugging errors.",
        importance=0.8))

    config = BlipShellConfig(memory=MemoryConfig())
    config.database.path = str(tmp_path / "blipshell.db")
    runner = NightlyRunner(config, sqlite_store, vectors=MagicMock(),
                           router=MagicMock(), processor=None)
    result = await runner._job_clean_junk_lessons(lambda m: None)
    assert result["folded"] == 1
    # The deleted duplicate's session is now excluded from re-extraction.
    assert sid in await sqlite_store.get_lesson_backfill_exclusions()
