"""Alive Worker — background thread for the inner monologue loop.

Follows the MemoryWorker pattern: own thread, own event loop, own DB connections.
Runs the monologue cycle periodically when no session is active.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)


class AliveWorker:
    """Background thread for inner monologue processing.

    Mirrors MemoryWorker pattern: own thread, own event loop, own DB connections.
    The monologue only runs when no session is active (paused via pause/resume).
    """

    def __init__(self, config: BlipShellConfig, chroma: ChromaStore):
        self._config = config
        self._chroma = chroma
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # starts paused — resumes when session ends

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Start the monologue thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="alive-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("AliveWorker started")

    def pause(self):
        """Pause the monologue loop (called when session starts)."""
        self._paused.set()
        logger.debug("AliveWorker paused (session active)")

    def resume(self):
        """Resume the monologue loop (called when session ends)."""
        self._paused.clear()
        logger.debug("AliveWorker resumed (no session)")

    def shutdown(self, timeout: float = 30.0):
        """Signal stop and join the thread."""
        self._stop_event.set()
        self._paused.clear()  # unblock if waiting on pause
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("AliveWorker thread did not exit within %.1fs", timeout)
            self._thread = None
        logger.info("AliveWorker stopped")

    def _thread_main(self):
        """Entry point: create event loop, initialize resources, run monologue loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._monologue_loop())
        except Exception as e:
            logger.error("AliveWorker crashed: %s", e)
        finally:
            loop.close()

    async def _monologue_loop(self):
        """Main loop: sleep → check pause → run cycle → maybe synthesize."""
        from blipshell.alive.monologue import InnerMonologue
        from blipshell.alive.thought_engine import ThoughtEngine
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter
        from blipshell.memory.sqlite_store import SQLiteStore

        # Own DB connection (WAL mode safe for concurrent reads/writes)
        sqlite = SQLiteStore(self._config.database.path)
        await sqlite.initialize()

        endpoint_manager = EndpointManager(self._config.endpoints, self._config.llm)
        router = LLMRouter(
            self._config.models, endpoint_manager,
            pii_enabled=self._config.pii.enabled,
        )

        alive_config = self._config.alive
        thought_engine = ThoughtEngine(sqlite, self._chroma, router, alive_config)
        monologue = InnerMonologue(sqlite, self._chroma, router, alive_config, thought_engine)

        interval_s = alive_config.inner_monologue.interval_minutes * 60
        logger.info("AliveWorker loop started (interval: %dm)", alive_config.inner_monologue.interval_minutes)

        while not self._stop_event.is_set():
            # Sleep in small increments so we can respond to stop quickly
            slept = 0.0
            while slept < interval_s and not self._stop_event.is_set():
                time.sleep(min(5.0, interval_s - slept))
                slept += 5.0

            if self._stop_event.is_set():
                break

            # Wait if paused (session is active — don't compete for GPU)
            while self._paused.is_set() and not self._stop_event.is_set():
                time.sleep(2.0)

            if self._stop_event.is_set():
                break

            # Run one monologue cycle
            try:
                result = await monologue.run_cycle()
                logger.info(
                    "Monologue cycle %d complete: %d thoughts, %.1fs",
                    result.cycle_number, result.thoughts_generated, result.elapsed_s,
                )
            except Exception as e:
                logger.error("Monologue cycle failed: %s", e)

        # Cleanup
        if sqlite._db:
            await sqlite._db.close()
        logger.info("AliveWorker loop exited")
