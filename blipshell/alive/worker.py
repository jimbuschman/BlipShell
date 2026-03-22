"""Alive Worker — background thread for the inner monologue loop.

Follows the MemoryWorker pattern: own thread, own event loop, own DB connections.
Runs the monologue cycle periodically during idle time within a session.
All LLM calls go to cloud (no local GPU contention).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)


class AliveWorker:
    """Background thread for inner monologue processing.

    Mirrors MemoryWorker pattern: own thread, own event loop, own DB connections.
    Runs continuously during the session — cloud LLM means no GPU contention.
    """

    def __init__(self, config: BlipShellConfig, chroma: ChromaStore):
        self._config = config
        self._chroma = chroma
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_cycle_complete: Optional[Callable] = None
        self._last_result = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_result(self):
        """Last monologue cycle result (for CLI notification)."""
        result = self._last_result
        self._last_result = None  # consume on read
        return result

    def set_on_cycle_complete(self, callback: Optional[Callable]):
        """Set callback for when a monologue cycle completes.

        Callback receives the MonologueCycleResult. Called from the worker
        thread — caller must handle thread safety.
        """
        self._on_cycle_complete = callback

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

    def shutdown(self, timeout: float = 30.0):
        """Signal stop and join the thread."""
        self._stop_event.set()
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
        """Main loop: sleep → run cycle → notify."""
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

            # Run one monologue cycle
            try:
                result = await monologue.run_cycle()
                logger.info(
                    "Monologue cycle %d complete: %d thoughts, %d refined, %d initiative (%.1fs)",
                    result.cycle_number, result.thoughts_generated,
                    result.thoughts_refined, result.initiative_items_added,
                    result.elapsed_s,
                )
                # Store for CLI polling and fire callback
                self._last_result = result
                if self._on_cycle_complete:
                    try:
                        self._on_cycle_complete(result)
                    except Exception as e:
                        logger.debug("Cycle complete callback failed: %s", e)
            except Exception as e:
                logger.error("Monologue cycle failed: %s", e)

        # Cleanup
        if sqlite._db:
            await sqlite._db.close()
        logger.info("AliveWorker loop exited")
