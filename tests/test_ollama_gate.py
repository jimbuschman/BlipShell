"""Tests for OllamaGate — real threads and real asyncio, no mocked waits.

The critical regressions guarded here: cancelling an async waiter (Esc, an
upstream asyncio.wait_for, shutdown) must never wedge the gate — including
the race where release() transfers ownership to a waiter in the same beat
its task is cancelled.
"""

import asyncio
import threading
import time

import pytest

from blipshell.llm.ollama_gate import (
    BACKGROUND,
    INTERACTIVE,
    GateTimeout,
    OllamaGate,
    get_gate,
)


def _poll(condition, timeout: float = 2.0) -> bool:
    """Spin (with tiny sleeps) until condition() or deadline. Real waiting."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


async def _apoll(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(0.005)
    return condition()


class TestBasics:
    def test_acquire_release(self):
        gate = OllamaGate()
        assert gate.acquire(INTERACTIVE)
        assert gate.is_active
        gate.release()
        assert not gate.is_active

    def test_serializes_across_threads(self):
        gate = OllamaGate()
        gate.acquire(INTERACTIVE)
        acquired = threading.Event()

        def contender():
            gate.acquire(BACKGROUND)
            acquired.set()
            gate.release()

        t = threading.Thread(target=contender)
        t.start()
        assert _poll(lambda: gate.waiter_count == 1)
        assert not acquired.is_set()

        gate.release()
        t.join(2.0)
        assert acquired.is_set()
        assert not gate.is_active

    def test_priority_ordering(self):
        """With both queued, the INTERACTIVE waiter is woken before BACKGROUND
        even though BACKGROUND queued first."""
        gate = OllamaGate()
        gate.acquire(INTERACTIVE)
        order = []

        def contender(priority, label):
            gate.acquire(priority)
            order.append(label)
            gate.release()

        t_bg = threading.Thread(target=contender, args=(BACKGROUND, "bg"))
        t_bg.start()
        assert _poll(lambda: gate.waiter_count == 1)
        t_int = threading.Thread(target=contender, args=(INTERACTIVE, "interactive"))
        t_int.start()
        assert _poll(lambda: gate.waiter_count == 2)

        gate.release()
        t_int.join(2.0)
        t_bg.join(2.0)
        assert order == ["interactive", "bg"]
        assert not gate.is_active

    def test_sync_context_manager_releases_on_exception(self):
        gate = OllamaGate()
        with pytest.raises(ValueError):
            with gate.gate(INTERACTIVE):
                assert gate.is_active
                raise ValueError("boom")
        assert not gate.is_active

    def test_singleton(self):
        assert get_gate() is get_gate()


class TestTimeout:
    def test_sync_timeout_raises_and_withdraws(self):
        gate = OllamaGate()
        gate.acquire(INTERACTIVE)
        with pytest.raises(GateTimeout):
            gate.acquire(BACKGROUND, timeout=0.05)
        # The timed-out waiter must be skipped: release opens the gate.
        gate.release()
        assert not gate.is_active
        assert gate.acquire(INTERACTIVE, timeout=0.5)
        gate.release()

    async def test_async_timeout_raises_and_withdraws(self):
        gate = OllamaGate()
        await gate.async_acquire(INTERACTIVE)
        with pytest.raises(GateTimeout):
            await gate.async_acquire(BACKGROUND, timeout=0.05)
        gate.release()
        assert not gate.is_active
        assert await gate.async_acquire(INTERACTIVE, timeout=0.5)
        gate.release()

    def test_gate_timeout_is_timeout_error(self):
        # Callers can catch plain TimeoutError.
        assert issubclass(GateTimeout, TimeoutError)


class TestAsyncCancellation:
    async def test_cancel_while_waiting_does_not_wedge(self):
        """THE regression: cancelling a queued async waiter must leave the
        gate fully usable. The old executor-thread implementation wedged
        permanently when release() transferred ownership to the abandoned
        waiter."""
        gate = OllamaGate()
        await gate.async_acquire(INTERACTIVE)

        task = asyncio.create_task(gate.async_acquire(INTERACTIVE))
        assert await _apoll(lambda: gate.waiter_count == 1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gate.waiter_count == 0

        gate.release()
        assert not gate.is_active  # cancelled waiter skipped, gate opened

        # Gate is immediately usable — this hung forever pre-fix.
        assert await asyncio.wait_for(gate.async_acquire(INTERACTIVE), 1.0)
        gate.release()

    async def test_cancel_after_ownership_transfer_passes_gate_on(self):
        """The race window: release() marks the waiter OWNED in the same beat
        its task is cancelled. The waiter must hand the gate to the next in
        line (here: open it) instead of keeping it forever."""
        gate = OllamaGate()
        await gate.async_acquire(INTERACTIVE)

        task = asyncio.create_task(gate.async_acquire(INTERACTIVE))
        assert await _apoll(lambda: gate.waiter_count == 1)

        # Transfer ownership, then cancel before the loop delivers the wake.
        gate.release()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not gate.is_active
        assert await asyncio.wait_for(gate.async_acquire(INTERACTIVE), 1.0)
        gate.release()

    async def test_cancel_one_of_two_waiters_wakes_the_other(self):
        gate = OllamaGate()
        await gate.async_acquire(INTERACTIVE)

        doomed = asyncio.create_task(gate.async_acquire(BACKGROUND))
        survivor = asyncio.create_task(gate.async_acquire(BACKGROUND))
        assert await _apoll(lambda: gate.waiter_count == 2)

        doomed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await doomed

        gate.release()
        assert await asyncio.wait_for(survivor, 1.0)
        gate.release()
        assert not gate.is_active

    async def test_async_context_manager_releases_on_exception(self):
        gate = OllamaGate()
        with pytest.raises(ValueError):
            async with gate.async_gate(INTERACTIVE):
                assert gate.is_active
                raise ValueError("boom")
        assert not gate.is_active


class TestCrossDomain:
    async def test_async_release_wakes_sync_thread_waiter(self):
        """Sync waiter parked on a threading.Event, released from the loop."""
        gate = OllamaGate()
        await gate.async_acquire(INTERACTIVE)
        acquired = threading.Event()

        def contender():
            gate.acquire(BACKGROUND)
            acquired.set()
            gate.release()

        t = threading.Thread(target=contender)
        t.start()
        assert await _apoll(lambda: gate.waiter_count == 1)

        gate.release()
        t.join(2.0)
        assert acquired.is_set()
        assert not gate.is_active

    async def test_thread_release_wakes_async_waiter(self):
        """Async waiter woken by a release() on a foreign thread — the
        call_soon_threadsafe path (this is how the memory worker hands the
        gate back to interactive chat)."""
        gate = OllamaGate()
        gate.acquire(BACKGROUND)

        task = asyncio.create_task(gate.async_acquire(INTERACTIVE))
        assert await _apoll(lambda: gate.waiter_count == 1)

        t = threading.Thread(target=gate.release)
        t.start()
        assert await asyncio.wait_for(task, 2.0)
        t.join(2.0)
        gate.release()
        assert not gate.is_active


class TestStats:
    async def test_stats_track_cancels(self):
        gate = OllamaGate()
        await gate.async_acquire(INTERACTIVE)
        with pytest.raises(GateTimeout):
            await gate.async_acquire(BACKGROUND, timeout=0.05)
        stats = gate.get_stats()
        assert stats["active"] is True
        assert stats["waiters"] == 0
        assert stats["total_acquisitions"] == 2
        assert stats["total_waits"] == 1
        assert stats["total_cancels"] == 1
        gate.release()
