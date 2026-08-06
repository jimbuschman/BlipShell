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

import threading
import time

import pytest

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
