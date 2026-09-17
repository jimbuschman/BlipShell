"""MemoryWorker — the background thread that writes the memory store.

This module had ZERO tests despite being the riskiest concurrency in the
codebase: it is the only place running a second asyncio event loop in a second
OS thread, with its own SQLiteStore and a SHARED VectorStore, writing the same
SQLite file as the main loop (deep-dive 2026-08-04).

These drive the REAL worker — real thread, real event loop, real queue, real
SQLiteStore against a temp DB. Only MemoryProcessor and EntityExtractor are
faked, because those are the LLM-dependent parts; everything about the
threading, dispatch, isolation and shutdown ordering is the production code.
"""

import asyncio
import sqlite3
import threading
import time

import pytest

from blipshell.llm.ollama_gate import BACKGROUND, get_gate
from blipshell.memory import worker as worker_mod
from blipshell.memory.worker import MemoryWorker, WorkItem, WorkType
from blipshell.models.config import BlipShellConfig


# --- fakes for the LLM-dependent collaborators -----------------------------


class RecordingProcessor:
    """Stands in for MemoryProcessor; records calls, optionally raises."""

    instances: list["RecordingProcessor"] = []

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple] = []
        self.fail_on: set[str] = set()
        self._event = threading.Event()
        RecordingProcessor.instances.append(self)

    def _record(self, name, payload):
        self.calls.append((name, payload))
        self._event.set()
        if name in self.fail_on:
            # NOT RuntimeError: _process_loop has a dedicated RuntimeError
            # clause, so using it would mask whether the generic handlers work.
            raise ValueError(f"{name} boom")

    async def process_message(self, text=None, role=None, session_id=None,
                              metadata=None, memory_id=None):
        self._record("process_message", {
            "text": text, "role": role, "session_id": session_id,
            "memory_id": memory_id,
        })

    async def process_lesson(self, text, session_id, project=None):
        self._record("process_lesson", {
            "text": text, "session_id": session_id, "project": project,
        })

    async def process_core_memory(self, text, session_id=None):
        self._record("process_core_memory", {"text": text, "session_id": session_id})

    def names(self):
        return [c[0] for c in self.calls]


class RecordingExtractor:
    """Stands in for EntityExtractor."""

    runs = 0

    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.get("batch_size")

    async def extract_batch(self):
        RecordingExtractor.runs += 1
        return {"triples": 0, "extracted": 0, "errors": 0}


@pytest.fixture(autouse=True)
def _reset_fakes():
    RecordingProcessor.instances.clear()
    RecordingExtractor.runs = 0
    yield
    RecordingProcessor.instances.clear()


@pytest.fixture
def config(tmp_path):
    cfg = BlipShellConfig()
    cfg.database.path = str(tmp_path / "worker.db")
    cfg.endpoints = []          # no clients to build, no network
    return cfg


@pytest.fixture
async def seeded_config(config):
    """config, plus one memory awaiting entity extraction.

    _idle_extract_entities returns early when get_unextracted_memory_ids is
    empty, so against a fresh DB the idle branch never reaches the extractor
    and any assertion about it is vacuous.
    """
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.models.memory import Memory

    store = SQLiteStore(config.database.path)
    await store.initialize()
    sid = await store.create_session("seed")   # memories.session_id is an FK
    await store.create_memory(Memory(
        session_id=sid, role="user",
        content="the entity graph merge thresholds and the version guard",
        summary="discussed entity merge thresholds",
    ))
    await store.close()
    return config


@pytest.fixture
def patched(monkeypatch):
    """Swap in the fakes at their import sites (the worker imports lazily)."""
    monkeypatch.setattr(
        "blipshell.memory.processor.MemoryProcessor", RecordingProcessor,
    )
    monkeypatch.setattr(
        "blipshell.memory.entity_extractor.EntityExtractor", RecordingExtractor,
    )
    monkeypatch.setattr(worker_mod, "_START_TIMEOUT", 5.0)


@pytest.fixture
def started_worker(config, patched):
    """A running worker, always shut down even if the test fails."""
    w = MemoryWorker(config, vectors=object())
    w.start()
    yield w
    w.shutdown(timeout=5.0)


def _wait(predicate, timeout=6.0, interval=0.02):
    """Poll until predicate() is truthy. The worker's queue poll blocks for 1s,
    so anything crossing the thread boundary needs a real wait, not a sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _processor():
    assert RecordingProcessor.instances, "worker never built its processor"
    return RecordingProcessor.instances[0]


# --- lifecycle -------------------------------------------------------------


class TestLifecycle:
    def test_start_brings_up_a_named_daemon_thread(self, started_worker):
        assert started_worker.is_alive
        assert started_worker._thread.name == "memory-worker"
        assert started_worker._thread.daemon, (
            "a non-daemon worker would hang process exit"
        )

    def test_worker_builds_its_own_resources(self, started_worker):
        """Its own SQLiteStore/router/processor is the whole point — sharing
        the main loop's would put background work on the interactive path."""
        assert _wait(lambda: len(RecordingProcessor.instances) == 1)

    def test_shutdown_stops_the_thread(self, config, patched):
        w = MemoryWorker(config, vectors=object())
        w.start()
        assert w.is_alive
        w.shutdown(timeout=5.0)
        assert not w.is_alive

    def test_shutdown_is_safe_before_start(self, config, patched):
        """Teardown paths call shutdown unconditionally."""
        MemoryWorker(config, vectors=object()).shutdown(timeout=1.0)

    def test_double_shutdown_is_safe(self, config, patched):
        w = MemoryWorker(config, vectors=object())
        w.start()
        w.shutdown(timeout=5.0)
        w.shutdown(timeout=1.0)
        assert not w.is_alive

    def test_slow_start_does_not_block_forever(self, config, monkeypatch):
        """start() deliberately continues when the handshake times out, so a
        broken worker can't stop the app from coming up."""
        monkeypatch.setattr(worker_mod, "_START_TIMEOUT", 0.2)

        async def _never_ready(self, loop):
            await __import__("asyncio").sleep(30)

        monkeypatch.setattr(MemoryWorker, "_run", _never_ready)
        w = MemoryWorker(config, vectors=object())
        t0 = time.monotonic()
        w.start()
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"start() blocked {elapsed:.1f}s past its timeout"
        assert not w._started.is_set()


# --- dispatch --------------------------------------------------------------


class TestDispatch:
    def test_process_message_reaches_the_processor(self, started_worker):
        started_worker.enqueue(WorkItem(
            work_type=WorkType.PROCESS_MESSAGE,
            text="the entity merge threshold discussion",
            role="user", session_id=7, memory_id=42,
        ))
        assert _wait(lambda: RecordingProcessor.instances
                     and _processor().names() == ["process_message"])
        payload = _processor().calls[0][1]
        assert payload["text"] == "the entity merge threshold discussion"
        assert payload["role"] == "user"
        assert payload["session_id"] == 7
        assert payload["memory_id"] == 42

    def test_process_lesson_carries_project(self, started_worker):
        started_worker.enqueue(WorkItem(
            work_type=WorkType.PROCESS_LESSON, text="a lesson",
            session_id=3, project="blipshell",
        ))
        assert _wait(lambda: RecordingProcessor.instances
                     and _processor().names() == ["process_lesson"])
        assert _processor().calls[0][1]["project"] == "blipshell"

    def test_process_core_memory_dispatches(self, started_worker):
        started_worker.enqueue(WorkItem(
            work_type=WorkType.PROCESS_CORE_MEMORY, text="a core fact", session_id=1,
        ))
        assert _wait(lambda: RecordingProcessor.instances
                     and _processor().names() == ["process_core_memory"])

    def test_extract_entities_runs_the_extractor(self, started_worker):
        started_worker.enqueue(WorkItem(
            work_type=WorkType.EXTRACT_ENTITIES, text="startup",
        ))
        assert _wait(lambda: RecordingExtractor.runs >= 1)

    def test_items_queued_before_start_are_processed(self, config, patched):
        """Startup enqueues the unprocessed sweep before the thread is up."""
        w = MemoryWorker(config, vectors=object())
        w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, text="early", session_id=1))
        w.start()
        try:
            assert _wait(lambda: RecordingProcessor.instances
                         and _processor().names() == ["process_message"])
        finally:
            w.shutdown(timeout=5.0)

    def test_queue_depth_reports_pending_work(self, config, patched):
        w = MemoryWorker(config, vectors=object())
        for i in range(3):
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, text=f"m{i}"))
        assert w.queue_depth == 3


# --- failure isolation -----------------------------------------------------


class TestFailureIsolation:
    def test_one_failing_item_does_not_kill_the_loop(self, started_worker):
        """A single bad memory must not silently end background processing for
        the rest of the process's life."""
        assert _wait(lambda: bool(RecordingProcessor.instances))
        proc = _processor()
        proc.fail_on = {"process_message"}

        started_worker.enqueue(WorkItem(
            work_type=WorkType.PROCESS_MESSAGE, text="explodes", session_id=1))
        assert _wait(lambda: "process_message" in proc.names())

        proc.fail_on = set()
        started_worker.enqueue(WorkItem(
            work_type=WorkType.PROCESS_LESSON, text="still works", session_id=1))
        assert _wait(lambda: "process_lesson" in proc.names()), (
            "loop died after one failed item"
        )
        assert started_worker.is_alive

    def test_worker_survives_a_failing_extraction(self, started_worker, monkeypatch):
        class Boom:
            def __init__(self, *a, **k):
                pass

            async def extract_batch(self):
                raise RuntimeError("extraction exploded")

        monkeypatch.setattr("blipshell.memory.entity_extractor.EntityExtractor", Boom)
        started_worker.enqueue(WorkItem(work_type=WorkType.EXTRACT_ENTITIES, text="x"))
        time.sleep(0.5)

        monkeypatch.setattr(
            "blipshell.memory.entity_extractor.EntityExtractor", RecordingExtractor)
        started_worker.enqueue(WorkItem(
            work_type=WorkType.PROCESS_MESSAGE, text="after", session_id=1))
        assert _wait(lambda: RecordingProcessor.instances
                     and "process_message" in _processor().names())


# --- the shutdown race -----------------------------------------------------


class TestShutdownRace:
    def test_shutdown_flag_is_set_before_the_signal_is_queued(self, config, patched):
        """The documented ordering: _shutting_down must already be set when the
        SHUTDOWN item lands, or idle entity extraction can start against a
        VectorStore the main thread is about to close."""
        w = MemoryWorker(config, vectors=object())
        w.start()
        try:
            observed = {}
            real_put = w._queue.put

            def spy(item, *a, **k):
                observed.setdefault("flag_when_queued", w._shutting_down.is_set())
                return real_put(item, *a, **k)

            w._queue.put = spy
            w.shutdown(timeout=5.0)
            assert observed.get("flag_when_queued") is True
        finally:
            if w.is_alive:
                w.shutdown(timeout=2.0)

    def test_idle_extraction_runs_when_the_queue_is_quiet(self, seeded_config,
                                                          patched, monkeypatch):
        """Control for the test below: with the interval shrunk, idle
        extraction genuinely fires. Without this, the skip test would pass
        simply because 60s never elapsed."""
        monkeypatch.setattr(worker_mod, "_IDLE_EXTRACT_INTERVAL", 0.1)
        w = MemoryWorker(seeded_config, vectors=object())
        w.start()
        try:
            assert _wait(lambda: RecordingExtractor.runs >= 1), (
                "idle extraction never ran even with the interval shrunk"
            )
        finally:
            w.shutdown(timeout=5.0)

    def test_idle_extraction_is_skipped_while_shutting_down(self, seeded_config,
                                                            patched, monkeypatch):
        """The guard that keeps slow extraction from racing VectorStore.close.
        The interval is shrunk so the idle branch is genuinely due — the
        _shutting_down flag is then the only thing holding it back."""
        monkeypatch.setattr(worker_mod, "_IDLE_EXTRACT_INTERVAL", 0.1)
        w = MemoryWorker(seeded_config, vectors=object())
        w._shutting_down.set()          # set before start: never a quiet window
        w.start()
        try:
            assert _wait(lambda: bool(RecordingProcessor.instances))
            time.sleep(1.5)             # several idle polls go by
            assert RecordingExtractor.runs == 0, (
                "idle extraction ran while shutting down — it can race "
                "VectorStore.close() on the main thread"
            )
        finally:
            w.shutdown(timeout=5.0)


class TestIdleExtractionDefersToChat:
    """Scheduling under load (2026-09-16 acceptance, "Remaining live findings"):
    the worker used to judge the system idle from ITS OWN queue alone, so idle
    entity extraction started while a chat turn was generating and the turn's
    embeddings timed out behind it. An active interactive turn on the gate is
    now the other half of "idle"."""

    def test_idle_extraction_waits_for_the_turn_and_resumes_after(
            self, seeded_config, patched, monkeypatch):
        monkeypatch.setattr(worker_mod, "_IDLE_EXTRACT_INTERVAL", 0.1)
        w = MemoryWorker(seeded_config, vectors=object())
        turn = get_gate().interactive_turn()
        turn.__enter__()
        released = False
        try:
            w.start()
            assert _wait(lambda: bool(RecordingProcessor.instances))
            time.sleep(1.5)             # several idle polls, all due
            assert RecordingExtractor.runs == 0, (
                "idle extraction started while a chat turn was active"
            )
            turn.__exit__(None, None, None)
            released = True
            # The deferred work is not lost: it runs once the turn ends.
            assert _wait(lambda: RecordingExtractor.runs >= 1), (
                "idle extraction never resumed after the chat turn ended"
            )
        finally:
            if not released:
                turn.__exit__(None, None, None)
            w.shutdown(timeout=5.0)


# --- bounded shutdown ---------------------------------------------------------


class SlowProcessor(RecordingProcessor):
    """process_message takes `delay` seconds (cancellable sleep) before recording."""

    delay = 0.3

    async def process_message(self, **kw):
        await asyncio.sleep(type(self).delay)
        self._record("process_message", kw)


class GateWaitingProcessor(RecordingProcessor):
    """process_message wants the model at BACKGROUND priority - and parks
    behind an open chat turn, exactly where a real summarisation would."""

    async def process_message(self, **kw):
        async with get_gate().async_gate(BACKGROUND):
            self._record("process_message", kw)


class HangingExtractor(RecordingExtractor):
    async def extract_batch(self):
        HangingExtractor.runs += 1
        await asyncio.sleep(1000)
        return {"triples": 0, "extracted": 0, "errors": 0}


def _make_session(db_path) -> int:
    """memories.session_id is a FOREIGN KEY: a deferred raw persist needs one."""
    from blipshell.memory.sqlite_store import SQLiteStore

    async def seed():
        s = SQLiteStore(str(db_path))
        await s.initialize()
        sid = await s.create_session("bounded-shutdown")
        await s.close()
        return sid

    return asyncio.run(seed())


class TestBoundedShutdown:
    """Session close used to DRAIN the queue (15 s per item, no upper bound)
    and had no way to stop an in-flight item; the 2026-09-16 acceptance run
    overran its 30 s wait with one extraction batch in flight, and a worker
    stuck in a model call kept the interpreter alive (its aiosqlite thread is
    not a daemon). Now shutdown() is bounded and nothing is lost: queued
    items are deferred to the startup sweep, the in-flight item is cancelled
    at the deadline and stays retryable."""

    def test_queued_items_are_deferred_not_drained(self, config, patched, monkeypatch):
        SlowProcessor.delay = 1000.0
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", SlowProcessor)
        w = MemoryWorker(config, vectors=object())
        w.start()
        try:
            for i in range(5):
                w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, text=f"msg {i}",
                                   session_id=1, memory_id=100 + i))
            assert _wait(lambda: w.queue_depth <= 4)      # first item in flight
            t0 = time.monotonic()
            report = w.shutdown(timeout=0.5)
            elapsed = time.monotonic() - t0
            assert w.last_shutdown is report
        finally:
            w.shutdown(timeout=2.0)
        assert not w.is_alive
        assert report.exited
        assert elapsed < 10.0, f"shutdown took {elapsed:.1f}s: it drained instead of deferring"
        assert report.deferred == {"process_message": 4}, report
        assert report.interrupted == "process_message"
        assert "4 process_message" in report.describe()

    def test_deferred_message_without_a_row_is_persisted_raw(self, config, patched, monkeypatch):
        """The one queued item that is NOT durable by construction: a message
        whose raw persist never landed. Dropping it would lose the message;
        it must reach the table as is_processed=0 so the sweep finds it."""
        sid = _make_session(config.database.path)
        SlowProcessor.delay = 1000.0
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", SlowProcessor)
        w = MemoryWorker(config, vectors=object())
        w.start()
        try:
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, text="in flight",
                               session_id=sid, memory_id=None))
            assert _wait(lambda: w.queue_depth == 0)
            for i in range(3):
                w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, role="user",
                                   text=f"deferred message {i} about the gate design",
                                   session_id=sid, memory_id=None))
            # Noise is dropped by design (never a row), at shutdown too.
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, role="user",
                               text="ok sounds good", session_id=sid, memory_id=None))
            report = w.shutdown(timeout=0.5)
        finally:
            w.shutdown(timeout=2.0)
        assert report.deferred == {"process_message": 4}
        conn = sqlite3.connect(config.database.path)
        try:
            rows = conn.execute(
                "SELECT content, is_processed FROM memories WHERE content LIKE 'deferred message%' "
                "ORDER BY id").fetchall()
            noise_rows = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content = 'ok sounds good'").fetchone()[0]
        finally:
            conn.close()
        assert [r[1] for r in rows] == [0, 0, 0], rows
        assert [r[0] for r in rows] == [f"deferred message {i} about the gate design" for i in range(3)]
        assert noise_rows == 0, "a noise message was given a row at shutdown"

    def test_in_flight_item_gets_the_grace_period(self, config, patched, monkeypatch):
        SlowProcessor.delay = 0.3
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", SlowProcessor)
        w = MemoryWorker(config, vectors=object())
        w.start()
        try:
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, text="quick",
                               session_id=1, memory_id=7))
            assert _wait(lambda: w.queue_depth == 0)
            report = w.shutdown(timeout=5.0)
        finally:
            w.shutdown(timeout=2.0)
        assert report.exited and report.interrupted is None and report.deferred == {}
        assert [c[0] for c in RecordingProcessor.instances[0].calls] == ["process_message"]
        assert report.describe().startswith("queue empty, nothing in flight")

    def test_shutdown_cancels_a_worker_parked_behind_a_chat_turn(self, config, patched, monkeypatch):
        """Ctrl+C mid-turn: the worker is parked at the gate (interactive_turn
        open). Cancelling it must withdraw the waiter and exit the thread."""
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", GateWaitingProcessor)
        gate = get_gate()
        w = MemoryWorker(config, vectors=object())
        turn = gate.interactive_turn()
        turn.__enter__()
        try:
            w.start()
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, text="parked",
                               session_id=1, memory_id=7))
            assert _wait(lambda: gate.waiter_count == 1), "worker never reached the gate"
            report = w.shutdown(timeout=0.5)
            assert report.exited and report.interrupted == "process_message"
            assert gate.waiter_count == 0, "cancelled worker left a waiter parked at the gate"
        finally:
            turn.__exit__(None, None, None)
            w.shutdown(timeout=2.0)
        assert not gate.is_active

    def test_idle_extraction_in_flight_is_cancelled_at_the_deadline(self, seeded_config, patched, monkeypatch):
        monkeypatch.setattr(worker_mod, "_IDLE_EXTRACT_INTERVAL", 0.1)
        monkeypatch.setattr("blipshell.memory.entity_extractor.EntityExtractor", HangingExtractor)
        HangingExtractor.runs = 0
        w = MemoryWorker(seeded_config, vectors=object())
        w.start()
        try:
            assert _wait(lambda: HangingExtractor.runs >= 1)
            report = w.shutdown(timeout=0.5)
        finally:
            w.shutdown(timeout=2.0)
        assert report.exited
        assert report.interrupted == "idle_extract_entities"

    def test_report_describes_a_thread_that_would_not_exit(self):
        report = worker_mod.ShutdownReport(exited=False, deferred={"process_message": 2},
                                           interrupted="extract_entities", waited_s=35.0)
        text = report.describe()
        assert "2 process_message" in text and "interrupted extract_entities" in text
        assert "thread still alive" in text


# --- review findings on the bounded shutdown ---------------------------------


class RecordThenHangProcessor(RecordingProcessor):
    """Records the call it was given, then hangs (cancellable) - so a test can
    see WHICH memory_id the pipeline received before shutdown cancels it."""

    async def process_message(self, **kw):
        self._record("process_message", kw)
        await asyncio.sleep(1000)


class BlockingVectors:
    """A vector store whose write blocks on a pool thread until released -
    the shape of an embedding HTTP call or a gated vector write."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def add_memory(self, *_a, **_k):
        self.entered.set()
        self.release.wait(30.0)
        self.finished.set()


class ToThreadVectorProcessor(RecordingProcessor):
    """process_message does what the real pipeline does at its embed step:
    hands the vector write to a pool thread with asyncio.to_thread."""

    def __init__(self, sqlite, vectors, router, *a, **kw):
        super().__init__()
        self.vectors = vectors

    async def process_message(self, **kw):
        await asyncio.to_thread(self.vectors.add_memory, kw.get("memory_id"), kw.get("text"), {})
        self._record("process_message", kw)


class TestShutdownReviewFindings:
    def test_in_flight_rowless_message_is_durable_before_it_can_be_cancelled(
            self, config, patched, monkeypatch):
        """Finding 1. A raw-persist failure leaves a PROCESS_MESSAGE with
        memory_id=None; if shutdown cancelled it mid-pipeline before the
        processor created a row, the message was gone. The worker now gives
        it a row BEFORE the cancellable task starts and processes it as an
        update to that row."""
        sid = _make_session(config.database.path)
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", RecordThenHangProcessor)
        w = MemoryWorker(config, vectors=object())
        w.start()
        try:
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, role="user",
                               text="We decided the entity-graph version guard blocks merges of projectecho_v1 and _v2; this row-less message must survive a shutdown that cancels the pipeline",
                               session_id=sid, memory_id=None))
            assert _wait(lambda: RecordingProcessor.instances
                         and RecordingProcessor.instances[0].calls)
            received = RecordingProcessor.instances[0].calls[0][1]
            report = w.shutdown(timeout=0.3)
        finally:
            w.shutdown(timeout=2.0)
        assert report.exited and report.interrupted == "process_message"

        conn = sqlite3.connect(config.database.path)
        try:
            row = conn.execute(
                "SELECT id, is_processed FROM memories WHERE content = ?",
                ("We decided the entity-graph version guard blocks merges of projectecho_v1 and _v2; this row-less message must survive a shutdown that cancels the pipeline",)).fetchone()
        finally:
            conn.close()
        assert row is not None, "the cancelled in-flight message left no row"
        assert row[1] == 0, "row must stay is_processed=0 for the startup sweep"
        assert received["memory_id"] == row[0], (
            "the pipeline was not handed the row it must update"
        )

    def test_shutdown_reports_live_executor_work_and_keeps_the_thread_alive(
            self, config, patched, monkeypatch):
        """Finding 2. Cancelling the awaiting task does not stop a blocking
        call already running on a pool thread. The worker must not exit (and
        shutdown() must not report the stores safe to close) while that call
        is alive; once it returns, the thread exits on its own."""
        monkeypatch.setattr(worker_mod, "_CANCEL_GRACE", 0.5)
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", ToThreadVectorProcessor)
        vectors = BlockingVectors()
        w = MemoryWorker(config, vectors=vectors)
        w.start()
        try:
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, text="embed me",
                               session_id=1, memory_id=7))
            assert vectors.entered.wait(5.0), "the blocking vector write never started"

            report = w.shutdown(timeout=0.3)

            assert report.exited is False, "reported exited while a pool thread was mid-call"
            assert report.executor_busy is True
            assert "pool thread" in report.describe() and "stores left open" in report.describe()
            assert w.is_alive
            assert not vectors.finished.is_set()
            assert report.interrupted == "process_message"

            vectors.release.set()
            assert _wait(lambda: not w.is_alive), "thread did not exit once the call returned"
            assert vectors.finished.is_set()
            second = w.shutdown(timeout=1.0)
            assert second.exited and second.executor_busy is False
        finally:
            vectors.release.set()
            w.shutdown(timeout=2.0)

    async def test_force_cleanup_does_not_close_the_vector_store_over_a_live_worker(self):
        """The consumer side of finding 2: Agent.force_cleanup used to close
        the VectorStore unconditionally after shutdown(); with executor work
        alive that tears the connection out of a running call."""
        from unittest.mock import MagicMock

        from blipshell.core.agent import Agent

        def bare_agent(worker_alive: bool):
            a = Agent.__new__(Agent)
            a._background_tasks = set()
            a._nightly_scheduler_task = None
            a._memory_flush_task = None
            a._health_check_task = None
            a._friction_probe_task = None
            a.sqlite = None
            a.vectors = MagicMock()
            worker = MagicMock()
            worker.is_alive = worker_alive
            a._memory_worker = worker
            return a

        alive = bare_agent(worker_alive=True)
        await alive.force_cleanup()
        alive._memory_worker.shutdown.assert_called_once()
        alive.vectors.close.assert_not_called()

        dead = bare_agent(worker_alive=False)
        await dead.force_cleanup()
        dead.vectors.close.assert_called_once()


# --- review of 1ef97d6: durability is a hard boundary ------------------------


SUBSTANTIVE = ("We decided the entity-graph version guard blocks merges of projectecho_v1 "
               "and _v2; this row-less message must never be processed without a row")


class TestDurabilityBoundary:
    def test_message_is_never_processed_while_its_raw_persist_fails(
            self, config, patched, monkeypatch):
        """Finding 1. When the worker-side raw persist itself fails, the item
        must not enter the cancellable pipeline with memory_id=None (the
        original loss case). It is retried; once a row exists it is processed
        as an update to that row."""
        from blipshell.memory.sqlite_store import SQLiteStore
        sid = _make_session(config.database.path)
        state = {"fail": True, "attempts": 0}
        original = SQLiteStore.save_raw_memory

        async def flaky(self, *a, **kw):
            state["attempts"] += 1
            if state["fail"]:
                raise RuntimeError("disk full")
            return await original(self, *a, **kw)

        monkeypatch.setattr(SQLiteStore, "save_raw_memory", flaky)
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", RecordThenHangProcessor)
        w = MemoryWorker(config, vectors=object(), poll_interval=0.05)
        w.start()
        try:
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, role="user",
                               text=SUBSTANTIVE, session_id=sid, memory_id=None))
            assert _wait(lambda: state["attempts"] >= 3), "persist was not retried"
            assert _wait(lambda: bool(RecordingProcessor.instances))
            assert not RecordingProcessor.instances[0].calls, (
                "the pipeline ran on a message that has no row"
            )
            state["fail"] = False
            assert _wait(lambda: RecordingProcessor.instances[0].calls), (
                "the message was not processed once a row could be written"
            )
            received = RecordingProcessor.instances[0].calls[0][1]
        finally:
            w.shutdown(timeout=1.0)
        conn = sqlite3.connect(config.database.path)
        try:
            row = conn.execute("SELECT id FROM memories WHERE content = ?", (SUBSTANTIVE,)).fetchone()
        finally:
            conn.close()
        assert row is not None and received["memory_id"] == row[0]

    def test_shutdown_during_the_durability_write_starts_no_processing(
            self, config, patched, monkeypatch):
        """Finding 2. shutdown() can issue its cancel while `_make_durable` is
        still awaiting the row write - there is no task in flight, so the
        cancel is a no-op. When the write completes, the worker must not
        start the pipeline; the (now durable) item is deferred instead."""
        from blipshell.memory.sqlite_store import SQLiteStore
        sid = _make_session(config.database.path)
        hold = threading.Event()
        original = SQLiteStore.save_raw_memory

        async def slow(self, *a, **kw):
            while not hold.is_set():
                await asyncio.sleep(0.01)
            return await original(self, *a, **kw)

        monkeypatch.setattr(SQLiteStore, "save_raw_memory", slow)
        monkeypatch.setattr(worker_mod, "_CANCEL_GRACE", 0.3)
        monkeypatch.setattr("blipshell.memory.processor.MemoryProcessor", RecordThenHangProcessor)
        w = MemoryWorker(config, vectors=object(), poll_interval=0.05)
        w.start()
        outcome = {}
        try:
            w.enqueue(WorkItem(work_type=WorkType.PROCESS_MESSAGE, role="user",
                               text=SUBSTANTIVE, session_id=sid, memory_id=None))
            assert _wait(lambda: w.queue_depth == 0)      # taken; durability write in progress
            time.sleep(0.2)

            shutter = threading.Thread(
                target=lambda: outcome.update(first=w.shutdown(timeout=0.2)))
            shutter.start()
            shutter.join(5.0)                              # timeout + grace both elapse
            first = outcome["first"]
            # Honest while the write is still in progress: not exited.
            assert first.exited is False and w.is_alive
            assert not RecordingProcessor.instances[0].calls

            hold.set()                                     # the row write completes
            assert _wait(lambda: not w.is_alive), "worker did not exit after the write"
            assert not RecordingProcessor.instances[0].calls, (
                "processing started after shutdown had begun"
            )
            second = w.shutdown(timeout=1.0)
            assert second.exited and second.deferred == {"process_message": 1}, second
        finally:
            hold.set()
            w.shutdown(timeout=2.0)
        conn = sqlite3.connect(config.database.path)
        try:
            row = conn.execute("SELECT is_processed FROM memories WHERE content = ?",
                               (SUBSTANTIVE,)).fetchone()
        finally:
            conn.close()
        assert row is not None and row[0] == 0, "the deferred message has no raw row"
