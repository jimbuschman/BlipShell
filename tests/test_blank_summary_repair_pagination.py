"""The blank-summary repair walks the WHOLE backlog, and says what it left.

It used to be one `ORDER BY id LIMIT 100`. A row it cannot repair stays
blank, so it stays in the result set, so the SAME hundred rows came back on
the next call and on the next run. A hundred unrecoverable rows at the head
of the backlog hid every repairable row behind them - permanently - and the
command reported a clean run.

Every test here uses a temp DB and a scripted router. No model is called.
"""

import pytest

from blipshell.memory.processor import (
    count_blank_summaries,
    find_blank_summaries,
    repair_blank_summaries,
)
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.memory import Memory


class ScriptedRouter:
    """Summarizes by rule; records every call; can fail on demand."""

    def __init__(self, *, fail_on=(), reply=None):
        self.calls: list[str] = []
        self._fail_on = set(fail_on)
        self._reply = reply

    async def generate(self, task_type, prompt, system=None, **kwargs):
        self.calls.append(prompt)
        for marker in self._fail_on:
            if marker in prompt:
                raise RuntimeError(f"scripted failure for {marker}")
        if self._reply is not None:
            return self._reply
        return "a generated summary"


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(str(tmp_path / "t.db"))
    await s.initialize()
    yield s
    await s.close()


async def _seed(store, rows):
    """rows: list of (content, summary). Returns ids in insertion order."""
    session_id = await store.create_session(title="t")
    ids = []
    for content, summary in rows:
        ids.append(await store.create_memory(Memory(
            session_id=session_id, role="user", content=content, summary=summary,
        )))
    return ids


# --- pagination covers the whole backlog ------------------------------------------


async def test_more_than_one_page_of_repairable_rows_is_fully_repaired(store):
    """125 repairable rows, page size 100: all 125, not the first 100."""
    ids = await _seed(store, [(f"content number {i:04d}", "") for i in range(125)])
    router = ScriptedRouter()

    stats = await repair_blank_summaries(store, router, dry_run=False)

    assert stats["backlog"] == 125
    assert stats["scanned"] == 125
    assert stats["resummarized"] == 125
    assert stats["remaining"] == 0
    assert stats["incomplete"] is False
    assert len(router.calls) == 125
    assert await count_blank_summaries(store) == 0
    for mem_id in ids:
        mem = await store.get_memory(mem_id)
        assert mem.summary == "a generated summary"


async def test_the_cursor_advances_past_an_unrepaired_row(store):
    """find_blank_summaries is an ID cursor, not an offset."""
    ids = await _seed(store, [(f"c{i}", "") for i in range(5)])
    first = await find_blank_summaries(store, limit=2)
    assert [r["id"] for r in first] == ids[:2]
    second = await find_blank_summaries(store, limit=2, after_id=first[-1]["id"])
    assert [r["id"] for r in second] == ids[2:4]


# --- unrecoverable rows do not block the rows behind them -------------------------


async def test_a_hundred_unrecoverable_rows_do_not_hide_the_repairable_ones(store):
    """THE regression: 100 contentless rows ahead of 10 repairable ones."""
    dead = await _seed(store, [("", "") for _ in range(100)])
    good = await _seed(store, [(f"real content {i}", "") for i in range(10)])
    router = ScriptedRouter()

    stats = await repair_blank_summaries(store, router, dry_run=False)

    assert stats["scanned"] == 110
    assert stats["unrecoverable"] == 100
    assert stats["resummarized"] == 10
    assert len(router.calls) == 10
    for mem_id in good:
        assert (await store.get_memory(mem_id)).summary == "a generated summary"
    for mem_id in dead:
        assert (await store.get_memory(mem_id)).summary == ""


async def test_an_unrecoverable_row_is_not_retried_within_one_run(store):
    """It stays blank, so without a cursor it comes back for ever."""
    await _seed(store, [("", "") for _ in range(30)])
    await _seed(store, [("real", "")])
    router = ScriptedRouter()

    stats = await repair_blank_summaries(store, router, dry_run=False,
                                         page_size=10)

    assert stats["scanned"] == 31, "a row was examined twice"
    assert stats["unrecoverable"] == 30
    assert len(router.calls) == 1


async def test_whitespace_only_content_is_unrecoverable_not_a_crash(store):
    await _seed(store, [("   ", ""), ("\t\n", ""), ("real", "")])
    router = ScriptedRouter()

    stats = await repair_blank_summaries(store, router, dry_run=False)

    assert stats["unrecoverable"] == 2
    assert stats["resummarized"] == 1


# --- a model failure does not block the rows behind it ----------------------------


async def test_a_failing_row_is_reported_and_the_run_continues(store):
    await _seed(store, [("alpha", ""), ("POISON", ""), ("omega", "")])
    router = ScriptedRouter(fail_on=["POISON"])

    stats = await repair_blank_summaries(store, router, dry_run=False)

    assert stats["scanned"] == 3
    assert stats["failed"] == 1
    assert stats["resummarized"] == 2
    assert stats["remaining"] == 1
    assert stats["incomplete"] is True


async def test_a_total_outage_leaves_every_row_for_a_retry(store):
    await _seed(store, [(f"c{i}", "") for i in range(120)])
    router = ScriptedRouter(fail_on=["c"])

    stats = await repair_blank_summaries(store, router, dry_run=False)

    assert stats["scanned"] == 120
    assert stats["failed"] == 120
    assert stats["resummarized"] == 0
    assert stats["remaining"] == 120
    assert stats["incomplete"] is True
    assert await count_blank_summaries(store) == 120


# --- the cap, and disclosing incomplete work --------------------------------------


async def test_max_rows_caps_the_run_and_the_run_says_it_is_incomplete(store):
    await _seed(store, [(f"content {i}", "") for i in range(50)])
    router = ScriptedRouter()
    said: list[str] = []

    stats = await repair_blank_summaries(store, router, dry_run=False,
                                         max_rows=20, on_status=said.append)

    assert stats["scanned"] == 20
    assert stats["resummarized"] == 20
    assert stats["not_scanned"] == 30
    assert stats["remaining"] == 30
    assert stats["incomplete"] is True
    assert any("not examined" in m for m in said)
    assert len(router.calls) == 20


async def test_a_capped_run_resumes_where_it_stopped(store):
    await _seed(store, [(f"content {i}", "") for i in range(30)])
    router = ScriptedRouter()

    first = await repair_blank_summaries(store, router, dry_run=False, max_rows=10)
    second = await repair_blank_summaries(store, router, dry_run=False, max_rows=10)
    third = await repair_blank_summaries(store, router, dry_run=False, max_rows=10)

    assert [s["resummarized"] for s in (first, second, third)] == [10, 10, 10]
    assert third["remaining"] == 0
    assert third["incomplete"] is False
    assert len(router.calls) == 30


async def test_a_cap_smaller_than_a_page_is_honoured(store):
    await _seed(store, [(f"content {i}", "") for i in range(10)])
    router = ScriptedRouter()

    stats = await repair_blank_summaries(store, router, dry_run=False,
                                         max_rows=3, page_size=100)

    assert stats["scanned"] == 3
    assert len(router.calls) == 3


async def test_a_clean_complete_run_is_not_marked_incomplete(store):
    await _seed(store, [("alpha", ""), ("beta", "")])
    stats = await repair_blank_summaries(store, ScriptedRouter(), dry_run=False)
    assert stats["incomplete"] is False
    assert stats["remaining"] == 0


async def test_an_empty_backlog_reports_zeroes(store):
    stats = await repair_blank_summaries(store, ScriptedRouter(), dry_run=False)
    assert stats["backlog"] == 0
    assert stats["scanned"] == 0
    assert stats["incomplete"] is False


# --- reporting ---------------------------------------------------------------------


async def test_every_scanned_row_lands_in_exactly_one_outcome(store):
    await _seed(store, [("", "")] * 3)          # unrecoverable
    await _seed(store, [("POISON a", ""), ("POISON b", "")])  # failed
    await _seed(store, [("good one", ""), ("good two", "")])  # resummarized
    router = ScriptedRouter(fail_on=["POISON"])

    stats = await repair_blank_summaries(store, router, dry_run=False)

    outcomes = (stats["resummarized"] + stats["content_fallback"]
                + stats["unrecoverable"] + stats["failed"])
    assert outcomes == stats["scanned"] == 7
    assert stats["remaining"] == 5  # 3 unrecoverable + 2 failed


async def test_a_skip_verdict_falls_back_to_content_and_never_archives(store):
    ids = await _seed(store, [("the raw content", "")])
    router = ScriptedRouter(reply="SKIP")

    stats = await repair_blank_summaries(store, router, dry_run=False)

    assert stats["skip_verdict"] == 1
    assert stats["content_fallback"] == 1
    mem = await store.get_memory(ids[0])
    assert mem.summary == "the raw content"
    assert mem.is_archived is False


async def test_dry_run_scans_the_whole_backlog_without_calling_the_model(store):
    await _seed(store, [(f"content {i}", "") for i in range(150)])
    await _seed(store, [("", "") for _ in range(5)])
    router = ScriptedRouter()
    said: list[str] = []

    stats = await repair_blank_summaries(store, router, dry_run=True,
                                         on_status=said.append)

    assert router.calls == []
    assert stats["backlog"] == 155
    assert stats["scanned"] == 155
    assert stats["unrecoverable"] == 5
    assert stats["remaining"] == 155  # nothing was changed
    assert sum("would repair" in m for m in said) == 150


async def test_dry_run_honours_the_cap(store):
    await _seed(store, [(f"content {i}", "") for i in range(40)])
    stats = await repair_blank_summaries(store, ScriptedRouter(), dry_run=True,
                                         max_rows=5)
    assert stats["scanned"] == 5
    assert stats["not_scanned"] == 35


# --- the defect, pinned ------------------------------------------------------------


async def test_the_old_unpaginated_window_saw_only_the_dead_rows(store):
    """Pin the premise: `ORDER BY id LIMIT 100` with no cursor.

    A bounded reproduction of the whole pre-fix run lives alongside this: with
    find_blank_summaries ignoring its cursor, a 500-row budget over this same
    fixture re-examines the same 100 dead rows and repairs 0 of the 10
    repairable ones (the fixed code repairs 10 of 10 and scans 110).
    """
    await _seed(store, [("", "") for _ in range(100)])
    good = await _seed(store, [(f"real content {i}", "") for i in range(10)])

    from blipshell.memory.processor import BLANK_SUMMARY_SQL
    cursor = await store._db.execute(
        f"SELECT id, content FROM memories WHERE {BLANK_SUMMARY_SQL} "
        f"ORDER BY id LIMIT 100"
    )
    window = await cursor.fetchall()

    assert len(window) == 100
    assert all(not r["content"].strip() for r in window), "the old window was all dead rows"
    assert not (set(good) & {r["id"] for r in window}), "no repairable row was reachable"

    # ...and the cursor walk does reach them.
    reachable = {r["id"] for r in await find_blank_summaries(
        store, limit=100, after_id=window[-1]["id"])}
    assert set(good) <= reachable
