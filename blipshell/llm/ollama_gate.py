"""Priority gate for local Ollama access.

Serializes all local Ollama HTTP calls across threads with priority ordering.
Interactive calls (user-facing chat) preempt queued background work
(summarization, ranking, entity extraction).

Cloud endpoints bypass the gate entirely.

Architecture:
- Single threading.Lock protects internal state
- Waiters block on individual threading.Events (no busy-wait)
- heapq orders waiters by (priority, sequence) — lowest priority number wins
- async_acquire() offloads blocking wait to thread pool so event loops stay responsive
- infer_priority() auto-detects main thread vs memory-worker thread
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import threading
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)

# Priority levels (lower number = higher priority)
INTERACTIVE = 0   # User-facing chat, search queries
EMBEDDING = 1     # Reserved for future explicit embedding priority
BACKGROUND = 2    # Summarization, ranking, entity extraction

_WORKER_THREAD_NAME = "memory-worker"


class OllamaGate:
    """Thread-safe priority semaphore for local Ollama access.

    Only one Ollama call can proceed at a time. When multiple callers
    are waiting, the highest priority (lowest number) goes first.
    """

    INTERACTIVE = INTERACTIVE
    EMBEDDING = EMBEDDING
    BACKGROUND = BACKGROUND

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._waiters: list[tuple[int, int, threading.Event]] = []  # (priority, seq, event)
        self._seq = 0  # tie-breaker for same-priority (FIFO)
        # Stats
        self._total_acquisitions = 0
        self._total_waits = 0

    def acquire(self, priority: int = BACKGROUND) -> bool:
        """Block until the gate is available.

        Waits indefinitely — Ollama calls can legitimately take minutes
        (model loading, long generations, model swaps). The gate's job is
        to serialize, not to time out and allow concurrent access.
        """
        with self._lock:
            self._total_acquisitions += 1
            if not self._active:
                self._active = True
                logger.debug("OllamaGate: acquired immediately (P%d)", priority)
                return True

            # Register as a waiter
            event = threading.Event()
            self._seq += 1
            heapq.heappush(self._waiters, (priority, self._seq, event))
            self._total_waits += 1

        # Wait outside the lock — blocks this thread, not the lock
        logger.debug("OllamaGate: waiting (P%d, %d ahead)", priority, len(self._waiters))
        event.wait()  # No timeout — wait as long as needed

        logger.debug("OllamaGate: acquired after wait (P%d)", priority)
        return True

    def release(self):
        """Release the gate, wake the highest-priority waiter."""
        with self._lock:
            if self._waiters:
                _, _, event = heapq.heappop(self._waiters)
                # Transfer ownership — don't set _active=False
                event.set()
                logger.debug("OllamaGate: released → woke P%d waiter (%d remaining)",
                             self._waiters[0][0] if self._waiters else -1,
                             len(self._waiters))
            else:
                self._active = False
                logger.debug("OllamaGate: released (no waiters)")

    def infer_priority(self) -> int:
        """Infer priority from current thread name."""
        if threading.current_thread().name == _WORKER_THREAD_NAME:
            return BACKGROUND
        return INTERACTIVE

    @contextmanager
    def gate(self, priority: int = BACKGROUND):
        """Sync context manager for gating Ollama calls."""
        self.acquire(priority)
        try:
            yield
        finally:
            self.release()

    async def async_acquire(self, priority: int = BACKGROUND) -> bool:
        """Async wrapper — offloads blocking wait to thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.acquire, priority)

    @asynccontextmanager
    async def async_gate(self, priority: int = BACKGROUND):
        """Async context manager for gating Ollama calls."""
        await self.async_acquire(priority)
        try:
            yield
        finally:
            self.release()

    # --- Status / observability ---

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def waiter_count(self) -> int:
        return len(self._waiters)

    def get_stats(self) -> dict:
        """Get gate statistics for /context or /health display."""
        return {
            "active": self._active,
            "waiters": len(self._waiters),
            "total_acquisitions": self._total_acquisitions,
            "total_waits": self._total_waits,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_gate: OllamaGate | None = None
_gate_lock = threading.Lock()


def get_gate() -> OllamaGate:
    """Get or create the module-level OllamaGate singleton.

    Thread-safe lazy initialization.
    """
    global _gate
    if _gate is None:
        with _gate_lock:
            if _gate is None:
                _gate = OllamaGate()
    return _gate
