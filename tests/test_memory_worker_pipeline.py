"""MemoryWorker INTEGRATION: the real pipeline behind the real thread.

test_memory_worker.py covers the worker's threading, dispatch and shutdown
with the processor FAKED. This file is the other half: the same real worker
driving the REAL MemoryProcessor (canned router, temp DB), so what's pinned
here is what a message actually becomes — a processed row, a skipped noise
row, a reprocessable row mid-crash — not just that a method was called.

(History note: this file originally OVERWROTE test_memory_worker.py because
V2_PLAN and CLAUDE.md both still said MemoryWorker had zero tests — it had
17, committed the day before the docs were last trusted. Recovered from
git; the two files are deliberate layers now. Verify before you build.)

The main thread verifies effects through its own read connection: the
worker owns its SQLite connection inside its own loop, and reaching into it
cross-thread would test a thing production never does.
"""

import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory.worker import MemoryWorker, WorkItem, WorkType
from blipshell.models.config import BlipShellConfig, DatabaseConfig
from tests.conftest import _canned_generate


def _wait_until(cond, timeout=15.0, interval=0.05):
    """Poll a condition with a deadline. Real waiting, not mocked."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


# The pipeline's noise filter skips messages under 80 chars that lack a
# signal word (deliberate — see memory/noise.py). Test messages must clear
# that bar or the worker "succeeds" by design while writing nothing: the
# first version of these tests used a 52-char message and spent a debugging
# round learning this.
LONG_MSG = ("I profiled the Python service today and found the hot loop in "
            "the parser was taking most of the runtime, so I rewrote it with "
            "a compiled regex and memoization")


def _canned_router():
    router = MagicMock()
    router.generate = AsyncMock(side_effect=_canned_generate)
    router.get_model.return_value = "test-model"
    router.get_fallback_model.return_value = None
    return router


def _vectors():
    v = MagicMock()
    v.search_memories.return_value = []
    v.search_core_memories.return_value = []
    v.search_lessons.return_value = []
    v.search_similar_entities.return_value = []
    v.get_counts.return_value = {}
    return v


class _Reader:
    """Main-thread read-only view of the worker's database."""

    def __init__(self, path):
        self.path = str(path)

    def one(self, sql, *params):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    def val(self, sql, *params):
        row = self.one(sql, *params)
        return row[0] if row else None


def _seed_session(db_path) -> int:
    """A real session row — memories.session_id is a FOREIGN KEY, so the
    production invariant (messages always belong to a session) is one the
    tests must honor too."""
    import asyncio

    from blipshell.memory.sqlite_store import SQLiteStore

    async def seed():
        s = SQLiteStore(str(db_path))
        await s.initialize()
        sid = await s.create_session("worker-test")
        await s.close()
        return sid

    return asyncio.run(seed())


@pytest.fixture
def worker_env(tmp_path):
    """A stopped-by-teardown worker + reader over its DB.

    poll_interval is compressed so the loop ticks fast; the idle interval is
    parked high by default so idle extraction can't fire unless a test asks
    for it.
    """
    made = []

    def factory(*, router=None, idle_extract_interval=3600.0, poll_interval=0.05):
        sid = _seed_session(tmp_path / "w.db")
        config = BlipShellConfig(database=DatabaseConfig(path=str(tmp_path / "w.db")))
        w = MemoryWorker(
            config, _vectors(),
            router_factory=lambda: (router or _canned_router()),
            start_timeout=15.0,
            idle_extract_interval=idle_extract_interval,
            poll_interval=poll_interval,
        )
        made.append(w)
        return w, _Reader(tmp_path / "w.db"), sid

    yield factory

    for w in made:
        if w.is_alive:
            w.shutdown(timeout=15.0)


class TestFailedInit:
    def test_failed_init_does_not_block_the_main_thread(self, tmp_path):
        """start() must return (logging an error) even when the worker can't
        come up — the main loop is supposed to survive a broken worker.

        A missing directory is NOT a failing case (initialize() mkdirs
        parents — the first version of this test learned that). A parent
        that is a FILE is unfixable."""
        (tmp_path / "blocker").write_text("not a directory")
        config = BlipShellConfig(
            database=DatabaseConfig(path=str(tmp_path / "blocker" / "w.db")),
        )
        w = MemoryWorker(config, _vectors(),
                         router_factory=_canned_router, start_timeout=3.0)
        t0 = time.monotonic()
        w.start()          # must not raise
        assert time.monotonic() - t0 < 10.0
        assert not w._started.is_set()


class TestMessageProcessing:
    def test_enqueued_message_lands_processed(self, worker_env):
        """End-to-end through the real pipeline: persisted, summarized by the
        canned router, marked processed."""
        w, db, sid = worker_env()
        w.start()

        w.enqueue(WorkItem(
            work_type=WorkType.PROCESS_MESSAGE,
            text=LONG_MSG,
            role="user", session_id=sid,
        ))

        assert _wait_until(lambda: db.val(
            "SELECT COUNT(*) FROM memories WHERE is_processed = 1") == 1), (
            "the enqueued message never became a processed memory"
        )
        row = db.one("SELECT summary, is_processed FROM memories")
        assert row["summary"], "processed memory has no summary"

    def test_queue_drains_before_shutdown(self, worker_env):
        """shutdown() appends SHUTDOWN after pending work, so everything
        already enqueued must complete — no lost messages."""
        w, db, sid = worker_env()
        w.start()
        for i in range(4):
            w.enqueue(WorkItem(
                work_type=WorkType.PROCESS_MESSAGE,
                text=f"{LONG_MSG} (variant {i})",
                role="user", session_id=sid,
            ))

        w.shutdown(timeout=30.0)

        assert not w.is_alive
        assert db.val("SELECT COUNT(*) FROM memories WHERE is_processed = 1") == 4, (
            "shutdown dropped enqueued work"
        )


class TestNoiseFilter:
    def test_noise_message_writes_nothing_and_calls_no_llm(self, worker_env):
        """A short signal-less message is skipped BY DESIGN: no row, no LLM
        call, and the worker reports success. Pinned here because it looks
        exactly like a lost message from the outside."""
        router = _canned_router()
        w, db, sid = worker_env(router=router)
        w.start()

        w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE,
                           text="ok sounds good", role="user", session_id=sid))
        w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE,
                           text=LONG_MSG, role="user", session_id=sid))

        # The substantive message lands; the noise one never becomes a row.
        assert _wait_until(lambda: db.val(
            "SELECT COUNT(*) FROM memories WHERE is_processed = 1") == 1)
        assert db.val("SELECT COUNT(*) FROM memories") == 1
        assert w.is_alive


class TestCrashSafety:
    def test_live_session_row_stays_reprocessable_until_pipeline_completes(self, worker_env, tmp_path):
        """The crash-safety contract, on the path that actually carries it:
        live sessions persist the raw message on the MAIN thread
        (save_raw_memory, is_processed=0) and enqueue the id; the worker
        updates that row and flips the flag only at the END. Blocking the
        summarize step and reading mid-flight pins the ordering — a crash
        while the LLM is thinking must leave a row the startup sweep finds.

        (First version asserted persist-before-LLM on the memory_id=None
        path. Wrong: THAT path — imports, crash recovery — creates the row
        after summarize; the immediate-persist guarantee belongs to
        save_raw_memory. Reading the code beats assuming the doc.)
        """
        import asyncio
        import threading

        from blipshell.memory.sqlite_store import SQLiteStore

        release = threading.Event()

        async def blocking(task_type, prompt="", system=None, think=None, **kw):
            if task_type == "summarization":
                while not release.is_set():
                    await asyncio.sleep(0.02)
            return _canned_generate(task_type, prompt, system, think, **kw)

        router = MagicMock()
        router.generate = AsyncMock(side_effect=blocking)
        router.get_model.return_value = "test-model"
        w, db, sid = worker_env(router=router)

        async def persist_raw():
            s = SQLiteStore(str(tmp_path / "w.db"))
            await s.initialize()
            mid = await s.save_raw_memory(sid, "user", LONG_MSG)
            await s.close()
            return mid

        mid = asyncio.run(persist_raw())
        row = db.one("SELECT is_processed FROM memories WHERE id = ?", mid)
        assert row["is_processed"] == 0, "raw persist must start unprocessed"

        w.start()
        try:
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE,
                               text=LONG_MSG, role="user", session_id=sid,
                               memory_id=mid))

            # Mid-flight (summarize blocked): still reprocessable.
            assert _wait_until(lambda: router.generate.await_count >= 1), (
                "the worker never started the pipeline"
            )
            assert db.val(
                "SELECT is_processed FROM memories WHERE id = ?", mid) == 0, (
                "marked processed while the pipeline was still running — a "
                "crash here would strand the message"
            )
        finally:
            release.set()   # unblock even on failure, or teardown hangs 15s

        assert _wait_until(lambda: db.val(
            "SELECT is_processed FROM memories WHERE id = ?", mid) == 1), (
            "the flag never flipped after the pipeline completed"
        )

    def test_worker_survives_a_failing_item(self, worker_env):
        """One poisoned item must not kill the loop for the next one. The
        poison is a foreign-key violation (a session id that doesn't exist)
        — it reliably escapes the pipeline's own error handling."""
        w, db, sid = worker_env()
        w.start()

        w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE,
                           text=LONG_MSG, role="user", session_id=999999))
        w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE,
                           text=LONG_MSG + " second",
                           role="user", session_id=sid))

        assert _wait_until(lambda: db.val(
            "SELECT COUNT(*) FROM memories WHERE is_processed = 1") == 1), (
            "the loop died on the poisoned item and never processed the next"
        )
        assert w.is_alive


class TestIdleEntityExtraction:
    def _seed_unextracted(self, db_path):
        """A processed memory awaiting entity extraction, written before the
        worker starts (its own store initializes the schema on the same file)."""
        from blipshell.memory.sqlite_store import SQLiteStore
        import asyncio

        async def seed():
            s = SQLiteStore(str(db_path))
            await s.initialize()
            from blipshell.models.memory import Memory
            sid = await s.create_session("s")
            await s.create_memory(Memory(
                session_id=sid, role="user",
                content="user discussed python", summary="User uses Python",
            ))
            await s.close()

        asyncio.run(seed())

    def test_idle_extraction_fires_and_extracts(self, worker_env, tmp_path):
        """The branch the old wall-clock test never reached: queue empty past
        the interval → extraction runs → the memory is marked extracted and
        entities exist. Asserting on EFFECTS, not on time having passed."""
        self._seed_unextracted(tmp_path / "w.db")
        w, db, sid = worker_env(idle_extract_interval=0.2, poll_interval=0.05)
        w.start()

        assert _wait_until(lambda: db.val(
            "SELECT COUNT(*) FROM memories WHERE entities_extracted_at IS NOT NULL") == 1,
            timeout=20.0), "idle extraction never fired"
        assert db.val("SELECT COUNT(*) FROM entities") >= 2, (
            "extraction 'ran' but produced no entities — vacuous"
        )

    def test_idle_extraction_skipped_when_shutting_down(self, worker_env, tmp_path):
        """_shutting_down gates idle work: extraction is slow and uses the
        shared VectorStore, which closes right after shutdown."""
        self._seed_unextracted(tmp_path / "w.db")
        w, db, sid = worker_env(idle_extract_interval=0.2, poll_interval=0.05)
        w.start()
        # Flag set, but the loop still runs (no SHUTDOWN item yet): idle
        # ticks keep happening and must all decline to extract.
        w._shutting_down.set()
        time.sleep(1.0)

        assert db.val(
            "SELECT COUNT(*) FROM memories WHERE entities_extracted_at IS NOT NULL") == 0, (
            "idle extraction ran during shutdown"
        )
        w._shutting_down.clear()   # let teardown shut down normally
