"""Dedicated background thread for memory processing.

Runs its own asyncio event loop with its own SQLiteStore and LLMRouter
so background memory work (summarization, ranking, dedup) never competes
with the main chat event loop for I/O time.

Communication: main thread enqueues WorkItems via thread-safe queue.Queue.
ChromaDB is shared (synchronous, thread-safe).
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

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)


class WorkType(Enum):
    PROCESS_MESSAGE = "process_message"
    PROCESS_LESSON = "process_lesson"
    PROCESS_CORE_MEMORY = "process_core_memory"
    SHUTDOWN = "shutdown"


@dataclass
class WorkItem:
    work_type: WorkType
    text: str
    role: str = "user"
    session_id: int = 0
    metadata: str = "{}"
    project: Optional[str] = None  # for process_lesson
    message_db_id: Optional[int] = None  # session_messages row ID


class MemoryWorker:
    """Background memory processor running in a dedicated thread.

    Owns its own event loop, SQLiteStore, and LLMRouter so it never
    competes with the main chat event loop for I/O time.
    """

    def __init__(self, config: BlipShellConfig, chroma: ChromaStore):
        self._config = config
        self._chroma = chroma
        self._queue: queue.Queue[WorkItem] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        """Start the worker thread. Call from the main thread."""
        self._thread = threading.Thread(
            target=self._thread_main,
            name="memory-worker",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=10)
        if self._started.is_set():
            logger.info("Memory worker started (dedicated thread + event loop)")
        else:
            logger.error("Memory worker failed to start within 10s")

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

        # Own EndpointManager + Router — separate HTTP clients
        endpoint_mgr = EndpointManager(
            self._config.endpoints, self._config.llm,
        )
        router = LLMRouter(self._config.models, endpoint_mgr)

        # Own MemoryProcessor — uses worker's sqlite + router, shared chroma
        processor = MemoryProcessor(
            sqlite, self._chroma, router,
            config=self._config.memory,
            max_tags=self._config.tagging.max_tags,
        )

        self._started.set()

        try:
            await self._process_loop(loop, processor, sqlite)
        finally:
            await sqlite.close()

    async def _process_loop(self, loop, processor, sqlite):
        """Main processing loop. Polls the thread-safe queue."""
        while True:
            try:
                item = await loop.run_in_executor(
                    None, self._queue_get,
                )
                if item is None:
                    continue  # timeout, loop again

                if item.work_type == WorkType.SHUTDOWN:
                    logger.info("Memory worker received shutdown signal")
                    break

                await self._process_item(item, processor, sqlite)

            except Exception as e:
                logger.error("Memory worker loop error: %s", e)

    def _queue_get(self) -> Optional[WorkItem]:
        """Blocking get with 1s timeout so the loop can check for shutdown."""
        try:
            return self._queue.get(timeout=1.0)
        except queue.Empty:
            return None

    async def _process_item(self, item: WorkItem, processor, sqlite):
        """Process a single work item."""
        t0 = time.monotonic()

        try:
            if item.work_type == WorkType.PROCESS_MESSAGE:
                result = await processor.process_message(
                    text=item.text,
                    role=item.role,
                    session_id=item.session_id,
                    metadata=item.metadata,
                )
                if item.message_db_id and result is not None:
                    try:
                        await sqlite.mark_message_processed(item.message_db_id)
                    except Exception:
                        pass

            elif item.work_type == WorkType.PROCESS_LESSON:
                await processor.process_lesson(
                    item.text, item.session_id, project=item.project,
                )

            elif item.work_type == WorkType.PROCESS_CORE_MEMORY:
                await processor.process_core_memory(
                    item.text, session_id=item.session_id,
                )

            elapsed = time.monotonic() - t0
            logger.info(
                "Worker: %s in %.1fs (queue: %d remaining)",
                item.work_type.value, elapsed, self._queue.qsize(),
            )

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "Worker: %s failed after %.1fs: %s",
                item.work_type.value, elapsed, e,
            )

    # --- Public API (called from main thread) ---

    def enqueue(self, item: WorkItem):
        """Enqueue a work item. Thread-safe, non-blocking."""
        self._queue.put_nowait(item)

    def shutdown(self, timeout: float = 30.0):
        """Signal shutdown and wait for the worker thread to finish."""
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
