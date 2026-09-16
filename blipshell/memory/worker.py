"""Dedicated background thread for memory processing.

Runs its own asyncio event loop with its own SQLiteStore and LLMRouter
so background memory work (summarization, ranking, dedup) never competes
with the main chat event loop for I/O time.

Communication: main thread enqueues WorkItems via thread-safe queue.Queue.
VectorStore is shared (synchronous, thread-safe with internal lock).
SQLite is safe for concurrent writes (WAL mode).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional
from blipshell.llm.ollama_gate import background_model_work, get_gate

if TYPE_CHECKING:
    from blipshell.memory.vector_store import VectorStore
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

# How long start() waits for the worker to finish building its own SQLite
# store, router and processor before giving up and continuing anyway.
# Module-level so tests can shrink it (same convention as nightly's timeouts).
_START_TIMEOUT = 10.0

# Idle entity extraction: how long the queue must stay empty before the worker
# chips away at unextracted memories, and how many to take. Module-level for
# the same reason — a test can't wait 60s to reach the idle branch.
_IDLE_EXTRACT_INTERVAL = 60.0
_IDLE_EXTRACT_BATCH = 10

# After shutdown()'s grace period, how long to wait for the cancelled
# in-flight item to unwind before reporting the thread still alive.
_CANCEL_GRACE = 5.0


@dataclass
class ShutdownReport:
    """What shutdown() did.

    `deferred` counts the queued items handed to the startup sweep instead of
    being run, by work type; `interrupted` names the in-flight item cancelled
    at the deadline (its work is left retryable); `exited` says whether the
    thread is actually gone.
    """
    exited: bool
    deferred: dict = field(default_factory=dict)
    interrupted: Optional[str] = None
    waited_s: float = 0.0
    # The cancelled item had handed a blocking call (embedding, vector write)
    # to a pool thread and that call has not returned. Cancellation cannot
    # reach it; the worker thread stays alive until it does, and the stores
    # it uses are NOT safe to close until then.
    executor_busy: bool = False

    def describe(self) -> str:
        parts = []
        if self.deferred:
            parts.append("deferred to the startup sweep: " + ", ".join(
                f"{n} {kind}" for kind, n in sorted(self.deferred.items())))
        if self.interrupted:
            parts.append(f"interrupted {self.interrupted} (retryable)")
        if not parts:
            parts.append("queue empty, nothing in flight")
        if self.executor_busy:
            parts.append("a blocking call on a pool thread has not returned; stores left open")
        elif not self.exited:
            parts.append("thread still alive")
        return "; ".join(parts) + f" ({self.waited_s:.1f}s)"


class WorkType(Enum):
    PROCESS_MESSAGE = "process_message"
    PROCESS_LESSON = "process_lesson"
    PROCESS_CORE_MEMORY = "process_core_memory"
    EXTRACT_ENTITIES = "extract_entities"
    SHUTDOWN = "shutdown"


@dataclass
class WorkItem:
    work_type: WorkType
    text: str
    role: str = "user"
    session_id: int = 0
    metadata: str = "{}"
    project: Optional[str] = None  # for process_lesson
    memory_id: Optional[int] = None  # existing memories row ID (live sessions)


class MemoryWorker:
    """Background memory processor running in a dedicated thread.

    Owns its own event loop, SQLiteStore, and LLMRouter so it never
    competes with the main chat event loop for I/O time.
    """

    def __init__(self, config: BlipShellConfig, vectors: VectorStore, *,
                 router_factory=None, local_policy=None,
                 start_timeout: Optional[float] = None,
                 idle_extract_interval: Optional[float] = None,
                 poll_interval: float = 1.0):
        """Timings are constructor-injectable, NOT patch-the-module-constant:
        the worker's whole behavior is timing-driven (queue poll tick → idle
        branch → extraction), and a test that can't compress those waits
        either takes minutes or silently never reaches the branch it claims
        to cover — two such vacuous tests were caught by mutation testing on
        2026-08-06. `router_factory` lets tests supply a canned router; the
        worker otherwise builds its own EndpointManager + LLMRouter so its
        HTTP clients never touch the main loop's.
        """
        self._config = config
        self._vectors = vectors
        self._router_factory = router_factory
        self._local_policy = local_policy
        self._start_timeout = start_timeout if start_timeout is not None else _START_TIMEOUT
        self._idle_extract_interval = (
            idle_extract_interval if idle_extract_interval is not None
            else _IDLE_EXTRACT_INTERVAL
        )
        self._poll_interval = poll_interval
        self._queue: queue.Queue[WorkItem] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._shutting_down = threading.Event()  # signal to skip idle work
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_task: Optional[asyncio.Task] = None  # the in-flight item
        self._deferred: dict[str, int] = {}
        self._interrupted: Optional[str] = None
        self._executor_busy = threading.Event()  # set while draining pool threads at exit
        self.last_shutdown: Optional[ShutdownReport] = None

    def start(self):
        """Start the worker thread. Call from the main thread."""
        self._thread = threading.Thread(
            target=self._thread_main,
            name="memory-worker",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=self._start_timeout)
        if self._started.is_set():
            logger.info("Memory worker started (dedicated thread + event loop)")
        else:
            # Deliberately continues: the main loop should still come up even
            # if background memory processing is broken. Enqueued work then
            # queues until the worker recovers or the process exits.
            logger.error(
                "Memory worker failed to start within %.0fs — continuing without it",
                self._start_timeout,
            )

    def _thread_main(self):
        """Entry point for the worker thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._run(loop))
        except Exception as e:
            logger.error("Memory worker thread crashed: %s", e)
        finally:
            loop.close()

    @background_model_work
    async def _run(self, loop: asyncio.AbstractEventLoop):
        """Initialize resources, signal ready, then process loop."""
        from blipshell.llm.routing import build_routing
        from blipshell.memory.processor import MemoryProcessor
        from blipshell.memory.sqlite_store import SQLiteStore

        # Own SQLiteStore — same DB file, separate aiosqlite connection
        sqlite = SQLiteStore(self._config.database.path)
        await sqlite.initialize()

        # Own EndpointManager + Router — separate HTTP clients. Tests inject
        # a canned router instead; the pipeline is then fully deterministic.
        if self._router_factory is not None:
            router = self._router_factory()
        else:
            # Local mode too, not just the PII flags: the worker summarizes
            # and ranks the same messages chat does.
            endpoint_mgr, router = build_routing(self._config)
            if self._local_policy is not None:
                endpoint_mgr.local_policy = self._local_policy

        # Own MemoryProcessor — uses worker's sqlite + router, shared chroma
        processor = MemoryProcessor(
            sqlite, self._vectors, router,
            config=self._config.memory,
            max_tags=self._config.tagging.max_tags,
        )

        self._started.set()

        try:
            await self._process_loop(loop, processor, sqlite, router)
        finally:
            # A cancelled item may have a blocking call (embedding HTTP, vector
            # write under the gate) still running on a pool thread: cancelling
            # the awaiting task does not stop it. Wait for it here so the
            # thread's liveness is the truth about whether the shared
            # VectorStore is still in use. loop.close() alone would abandon
            # the pool thread mid-call.
            self._executor_busy.set()
            try:
                await loop.shutdown_default_executor()
            finally:
                self._executor_busy.clear()
            await sqlite.close()

    async def _process_loop(self, loop, processor, sqlite, router):
        """Main processing loop. Polls the thread-safe queue.

        Shutdown is BOUNDED (2026-09-16): once `_shutting_down` is set the
        loop takes no further items - whatever is still queued is handed to
        the startup sweep by `_defer_queued`, and the one in-flight item is
        given shutdown()'s grace period before it is cancelled. Draining the
        whole queue at session close used to cost several model calls per
        queued message with no upper bound.
        """
        last_idle_extract = time.monotonic()

        while True:
            if self._shutting_down.is_set():
                await self._defer_queued(sqlite)
                break
            try:
                item = await loop.run_in_executor(
                    None, self._queue_get,
                )
                if item is None:
                    # Queue empty — chip away at unextracted entities during idle,
                    # but ONLY if we're not shutting down and no chat turn is
                    # open. Entity extraction is slow and uses the shared
                    # VectorStore which gets closed shortly after shutdown.
                    if (not self._shutting_down.is_set()
                            and not get_gate().interactive_active
                            and time.monotonic() - last_idle_extract > self._idle_extract_interval):
                        await self._run_cancellable(
                            "idle_extract_entities",
                            self._idle_extract_entities(sqlite, router, _IDLE_EXTRACT_BATCH),
                        )
                        last_idle_extract = time.monotonic()
                    continue

                if item.work_type == WorkType.SHUTDOWN:
                    logger.info("Memory worker received shutdown signal")
                    await self._defer_queued(sqlite)
                    break

                await self._make_durable(item, sqlite)
                await self._run_cancellable(
                    item.work_type.value,
                    self._process_item(item, processor, sqlite, router),
                )
                last_idle_extract = time.monotonic()  # reset after real work

            except RuntimeError as e:
                if "shutdown" in str(e).lower():
                    logger.debug("Memory worker stopping (executor shut down)")
                    break
                logger.error("Memory worker loop error: %s", e)
            except Exception as e:
                logger.error("Memory worker loop error: %s", e)

    def _queue_get(self) -> Optional[WorkItem]:
        """Blocking get with a short timeout so the loop can check for
        shutdown and reach the idle branch."""
        try:
            return self._queue.get(timeout=self._poll_interval)
        except queue.Empty:
            return None

    async def _run_cancellable(self, label: str, coro) -> None:
        """Run one unit of work as a task shutdown() can cancel at its deadline.

        A cancelled unit is recorded as `interrupted`, never re-raised: every
        unit is retryable by construction (a message row stays is_processed=0,
        an extraction stays unmarked), so cancellation loses time, not work.
        """
        self._current_task = asyncio.ensure_future(coro)
        try:
            await self._current_task
        except asyncio.CancelledError:
            if not self._shutting_down.is_set():
                raise
            self._interrupted = label
            logger.info("Memory worker: %s interrupted at shutdown; its work stays retryable", label)
        finally:
            self._current_task = None

    @staticmethod
    def _needs_row(item: WorkItem) -> bool:
        """The one kind of work that is NOT durable by construction: a message
        whose raw persist never landed, so no row names it yet.

        Noise is excluded with the processor's own deterministic filter: a
        short signal-less message never becomes a row BY DESIGN (pinned in
        test_memory_worker_pipeline), so dropping it at shutdown loses nothing.
        """
        from blipshell.memory.noise import should_skip_memory
        return (item.work_type == WorkType.PROCESS_MESSAGE
                and item.memory_id is None and bool(item.text.strip())
                and not should_skip_memory(item.text))

    async def _persist_raw(self, item: WorkItem, sqlite) -> Optional[int]:
        """Give the message its is_processed=0 row; None if even that failed."""
        try:
            return await sqlite.save_raw_memory(
                item.session_id, item.role, item.text, metadata=item.metadata,
            )
        except Exception as e:
            logger.error(
                "Memory worker: could not persist a message raw: %s "
                "(session_id=%s text=%r)", e, item.session_id, item.text[:60],
            )
            return None

    async def _make_durable(self, item: WorkItem, sqlite) -> None:
        """Persist a row-less message BEFORE it enters the cancellable task.

        Runs outside `_run_cancellable`, so shutdown's cancel cannot land in
        the middle of it. The processor then updates that row (the same path
        a live-session message takes), and a cancellation mid-pipeline leaves
        a raw row with is_processed=0 for the startup sweep instead of nothing.
        """
        if not self._needs_row(item):
            return
        mem_id = await self._persist_raw(item, sqlite)
        if mem_id is not None:
            item.memory_id = mem_id

    async def _defer_queued(self, sqlite) -> None:
        """Hand the remaining queue to the startup sweep instead of running it.

        Every queued item is already durable - a PROCESS_MESSAGE names a raw
        row with is_processed=0, and unextracted memories are re-found - with
        one exception: a message whose raw persist never landed (memory_id
        None) would vanish with the queue, so it is persisted raw here.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item.work_type == WorkType.SHUTDOWN:
                continue
            if self._needs_row(item):
                await self._persist_raw(item, sqlite)
            kind = item.work_type.value
            self._deferred[kind] = self._deferred.get(kind, 0) + 1
        if self._deferred:
            logger.info(
                "Memory worker: %s deferred to the startup sweep",
                ", ".join(f"{n} {k}" for k, n in sorted(self._deferred.items())),
            )

    async def _process_item(self, item: WorkItem, processor, sqlite, router):
        """Process a single work item."""
        t0 = time.monotonic()

        try:
            if item.work_type == WorkType.PROCESS_MESSAGE:
                await processor.process_message(
                    text=item.text,
                    role=item.role,
                    session_id=item.session_id,
                    metadata=item.metadata,
                    memory_id=item.memory_id,
                )

            elif item.work_type == WorkType.PROCESS_LESSON:
                await processor.process_lesson(
                    item.text, item.session_id, project=item.project,
                )

            elif item.work_type == WorkType.PROCESS_CORE_MEMORY:
                await processor.process_core_memory(
                    item.text, session_id=item.session_id,
                )

            elif item.work_type == WorkType.EXTRACT_ENTITIES:
                if not self._shutting_down.is_set():
                    await self._run_entity_extraction(
                        sqlite, router,
                        batch_size=self._config.memory.entity_extraction_batch_size,
                    )

            elapsed = time.monotonic() - t0
            logger.info(
                "Worker: %s in %.1fs (queue: %d remaining)",
                item.work_type.value, elapsed, self._queue.qsize(),
            )

        except Exception as e:
            elapsed = time.monotonic() - t0
            preview = (item.text or "")[:60].replace("\n", " ")
            logger.error(
                "Worker: %s failed after %.1fs: %s "
                "(session_id=%s memory_id=%s role=%s text=%r)",
                item.work_type.value, elapsed, e,
                item.session_id, item.memory_id, item.role, preview,
            )

    # --- Entity extraction helpers ---

    async def _run_entity_extraction(self, sqlite, router, batch_size: int = 50):
        """Run entity extraction batch using the worker's own resources."""
        from blipshell.memory.entity_extractor import EntityExtractor

        er_cfg = self._config.memory.entity_resolution
        extractor = EntityExtractor(
            sqlite, router,
            vectors=self._vectors,
            batch_size=batch_size,
            entity_resolution_enabled=er_cfg.enabled,
            entity_auto_merge_threshold=er_cfg.embedding_auto_merge_threshold,
            entity_llm_threshold=er_cfg.llm_arbitration_threshold,
            entity_max_candidates=er_cfg.max_candidates,
        )
        stats = await extractor.extract_batch()
        if stats.get("triples", 0) > 0:
            logger.info(
                "Entity extraction: %d triples from %d memories",
                stats["triples"], stats["extracted"],
            )
        return stats

    async def _idle_extract_entities(self, sqlite, router, batch_size: int = 10):
        """Extract entities from a small batch during idle periods."""
        try:
            # Quick check: are there any unextracted memories?
            unextracted = await sqlite.get_unextracted_memory_ids(limit=1)
            if not unextracted:
                return
            stats = await self._run_entity_extraction(sqlite, router, batch_size)
            if stats.get("extracted", 0) > 0:
                logger.info("Idle entity extraction: processed %d memories", stats["extracted"])
        except Exception as e:
            logger.debug("Idle entity extraction error: %s", e)

    # --- Public API (called from main thread) ---

    def enqueue(self, item: WorkItem):
        """Enqueue a work item. Thread-safe, non-blocking."""
        self._queue.put_nowait(item)

    def shutdown(self, timeout: float = 30.0) -> ShutdownReport:
        """Stop the worker within about `timeout` seconds; nothing is lost.

        Sets _shutting_down first so idle entity extraction stops immediately
        (it checks this flag every loop iteration), then sends the SHUTDOWN
        item to wake the queue poll. The loop takes no further items: the
        queue is deferred to the startup sweep (see `_defer_queued`), and the
        in-flight item gets `timeout` seconds to finish before it is
        cancelled and left retryable. The returned report says what happened;
        `describe()` is the status line for the user.
        """
        t0 = time.monotonic()
        self._shutting_down.set()
        self._queue.put(WorkItem(work_type=WorkType.SHUTDOWN, text=""))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.info(
                    "Memory worker: in-flight item still running after %.0fs; cancelling it",
                    timeout,
                )
                self._request_cancel()
                self._thread.join(timeout=_CANCEL_GRACE)
                if self._thread.is_alive():
                    if self._executor_busy.is_set():
                        logger.warning(
                            "Memory worker: a blocking call on a pool thread has not "
                            "returned after %.0fs; the thread stays alive until it does "
                            "and the vector store must not be closed yet",
                            timeout + _CANCEL_GRACE,
                        )
                    else:
                        logger.warning(
                            "Memory worker did not exit within %.0fs", timeout + _CANCEL_GRACE,
                        )
        report = ShutdownReport(
            exited=not self.is_alive,
            deferred=dict(self._deferred),
            interrupted=self._interrupted,
            waited_s=round(time.monotonic() - t0, 1),
            executor_busy=self.is_alive and self._executor_busy.is_set(),
        )
        self.last_shutdown = report
        if report.exited:
            logger.info("Memory worker stopped: %s", report.describe())
        return report

    def _request_cancel(self) -> None:
        """Cancel the in-flight item from another thread, via the worker's loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def _cancel():
            task = self._current_task
            if task is not None and not task.done():
                task.cancel()

        try:
            loop.call_soon_threadsafe(_cancel)
        except RuntimeError:
            pass  # the loop closed between the check and the call

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
