"""Priority gate for local Ollama access.

Serializes all local Ollama HTTP calls across threads with priority ordering.
Interactive calls (user-facing chat) preempt queued background work
(summarization, ranking, entity extraction).

Cloud endpoints bypass the gate entirely.

Architecture:
- Single threading.Lock protects internal state
- Sync waiters block on threading.Events; async waiters await asyncio Futures
  woken via call_soon_threadsafe (no thread-pool parking, so cancellation works)
- heapq orders waiters by (priority, sequence) — lowest priority number wins
- Each waiter carries a state (waiting/owned/cancelled) so timeout and task
  cancellation withdraw cleanly: release() skips cancelled waiters, and a
  waiter cancelled after ownership was already transferred passes the gate on
  instead of wedging it
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

# Waiter lifecycle (transitions happen under the gate lock)
_WAITING = "waiting"
_OWNED = "owned"        # release() transferred ownership to this waiter
_CANCELLED = "cancelled"  # timed out or task cancelled; release() skips it


class GateTimeout(TimeoutError):
    """acquire()/async_acquire() gave up after `timeout` seconds.

    The caller does NOT hold the gate when this is raised.
    """


class _Waiter:
    """One queued acquirer. `wake` is how release() signals it:
    threading.Event.set for sync waiters, a call_soon_threadsafe wrapper
    for async waiters.
    """

    __slots__ = ("priority", "seq", "state", "wake")

    def __init__(self, priority: int, seq: int, wake):
        self.priority = priority
        self.seq = seq
        self.state = _WAITING
        self.wake = wake

    def __lt__(self, other: "_Waiter") -> bool:  # heapq ordering
        return (self.priority, self.seq) < (other.priority, other.seq)


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
        self._waiters: list[_Waiter] = []
        self._seq = 0  # tie-breaker for same-priority (FIFO)
        # Stats
        self._total_acquisitions = 0
        self._total_waits = 0
        self._total_cancels = 0

    def acquire(self, priority: int = BACKGROUND,
                timeout: float | None = None) -> bool:
        """Block until the gate is available.

        With timeout=None (default) waits indefinitely — Ollama calls can
        legitimately take minutes (model loading, long generations, model
        swaps). Pass a timeout to give up instead: raises GateTimeout and
        the caller does not hold the gate.
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
            waiter = _Waiter(priority, self._seq, event.set)
            heapq.heappush(self._waiters, waiter)
            self._total_waits += 1

        # Wait outside the lock — blocks this thread, not the lock
        logger.debug("OllamaGate: waiting (P%d, %d ahead)", priority, len(self._waiters))
        if event.wait(timeout):
            logger.debug("OllamaGate: acquired after wait (P%d)", priority)
            return True

        # Timed out — but release() may have transferred ownership in the gap
        # between event.wait() giving up and us taking the lock.
        with self._lock:
            if waiter.state is _OWNED:
                return True
            waiter.state = _CANCELLED
            self._total_cancels += 1
        raise GateTimeout(
            f"OllamaGate: gave up waiting after {timeout}s (P{priority})"
        )

    def release(self):
        """Release the gate, wake the highest-priority live waiter."""
        with self._lock:
            self._release_locked()

    def _release_locked(self):
        """Transfer ownership to the next live waiter, or open the gate.

        Must be called with self._lock held. Skips cancelled waiters; a
        waiter whose wake fails (e.g. its event loop already closed) is
        treated as cancelled so ownership is never transferred into a void.
        """
        while self._waiters:
            waiter = heapq.heappop(self._waiters)
            if waiter.state is _CANCELLED:
                continue
            waiter.state = _OWNED
            try:
                waiter.wake()
            except Exception as e:
                logger.warning("OllamaGate: failed to wake waiter, skipping: %s", e)
                waiter.state = _CANCELLED
                self._total_cancels += 1
                continue
            # Ownership transferred — _active stays True
            logger.debug("OllamaGate: released → woke P%d waiter (%d remaining)",
                         waiter.priority, len(self._waiters))
            return
        self._active = False
        logger.debug("OllamaGate: released (no waiters)")

    def infer_priority(self) -> int:
        """Infer priority from current thread name."""
        if threading.current_thread().name == _WORKER_THREAD_NAME:
            return BACKGROUND
        return INTERACTIVE

    @contextmanager
    def gate(self, priority: int = BACKGROUND, timeout: float | None = None):
        """Sync context manager for gating Ollama calls."""
        self.acquire(priority, timeout)
        try:
            yield
        finally:
            self.release()

    async def async_acquire(self, priority: int = BACKGROUND,
                            timeout: float | None = None) -> bool:
        """Async acquire — cancellation-safe, optional timeout.

        The waiting task can be cancelled (Esc, asyncio.wait_for upstream,
        shutdown) without wedging the gate: the waiter withdraws from the
        queue, and if ownership had already been transferred to it in the
        race window, it passes the gate straight to the next waiter.
        Raises GateTimeout if `timeout` elapses first.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            self._total_acquisitions += 1
            if not self._active:
                self._active = True
                logger.debug("OllamaGate: acquired immediately (P%d)", priority)
                return True

            fut: asyncio.Future = loop.create_future()

            def _set_result():
                if not fut.done():
                    fut.set_result(True)

            def wake():
                # release() may run on any thread; hand the wake to our loop.
                loop.call_soon_threadsafe(_set_result)

            self._seq += 1
            waiter = _Waiter(priority, self._seq, wake)
            heapq.heappush(self._waiters, waiter)
            self._total_waits += 1

        logger.debug("OllamaGate: waiting (P%d, %d ahead)", priority, len(self._waiters))
        try:
            if timeout is None:
                await fut
            else:
                await asyncio.wait_for(fut, timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError) as e:
            timed_out = isinstance(e, asyncio.TimeoutError)
            with self._lock:
                if waiter.state is _OWNED:
                    # Ownership arrived before we could withdraw — pass it
                    # to the next waiter so the gate never wedges.
                    self._release_locked()
                else:
                    waiter.state = _CANCELLED
                    self._total_cancels += 1
            if timed_out:
                raise GateTimeout(
                    f"OllamaGate: gave up waiting after {timeout}s (P{priority})"
                ) from None
            raise
        logger.debug("OllamaGate: acquired after wait (P%d)", priority)
        return True

    @asynccontextmanager
    async def async_gate(self, priority: int = BACKGROUND,
                         timeout: float | None = None):
        """Async context manager for gating Ollama calls."""
        await self.async_acquire(priority, timeout)
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
        with self._lock:
            return sum(1 for w in self._waiters if w.state is not _CANCELLED)

    def get_stats(self) -> dict:
        """Get gate statistics for /context or /health display."""
        with self._lock:
            return {
                "active": self._active,
                "waiters": sum(1 for w in self._waiters if w.state is not _CANCELLED),
                "total_acquisitions": self._total_acquisitions,
                "total_waits": self._total_waits,
                "total_cancels": self._total_cancels,
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
