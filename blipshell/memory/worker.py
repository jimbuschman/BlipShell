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
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

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
                 router_factory=None,
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
        try:
            loop.run_until_complete(self._run(loop))
        except Exception as e:
            logger.error("Memory worker thread crashed: %s", e)
        finally:
            loop.close()

    async def _run(self, loop: asyncio.AbstractEventLoop):
        """Initialize resources, signal ready, then process loop."""
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
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
            endpoint_mgr = EndpointManager(
                self._config.endpoints, self._config.llm,
            )
            router = LLMRouter(self._config.models, endpoint_mgr, pii_enabled=self._config.pii.enabled)

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
            await sqlite.close()

    async def _process_loop(self, loop, processor, sqlite, router):
        """Main processing loop. Polls the thread-safe queue."""
        last_idle_extract = time.monotonic()

        while True:
            try:
                item = await loop.run_in_executor(
                    None, self._queue_get,
                )
                if item is None:
                    # Queue empty — chip away at unextracted entities during idle,
                    # but ONLY if we're not shutting down. Entity extraction is
                    # slow and uses the shared VectorStore which gets closed
                    # shortly after shutdown.
                    if (not self._shutting_down.is_set()
                            and time.monotonic() - last_idle_extract > self._idle_extract_interval):
                        await self._idle_extract_entities(sqlite, router, _IDLE_EXTRACT_BATCH)
                        last_idle_extract = time.monotonic()
                    continue

                if item.work_type == WorkType.SHUTDOWN:
                    logger.info("Memory worker received shutdown signal")
                    break

                await self._process_item(item, processor, sqlite, router)
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

    def shutdown(self, timeout: float = 30.0):
        """Signal shutdown and wait for the worker thread to finish.

        Sets _shutting_down first so idle entity extraction stops immediately
        (it checks this flag every loop iteration). Then sends the SHUTDOWN
        work item so the loop breaks after its current task finishes.
        """
        self._shutting_down.set()
        self._queue.put(WorkItem(work_type=WorkType.SHUTDOWN, text=""))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "Memory worker did not exit within %.0fs", timeout,
                )

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
