"""Priority async job queue (port of LLMJobQueue.cs).

Sequential processing with priority buckets.
Lower priority number = higher priority (processed first).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass(order=True)
class LLMJob:
    """A queued LLM job with priority ordering."""
    priority: int
    job_fn: Callable[[], Coroutine] = field(compare=False)
    future: asyncio.Future = field(compare=False)


class LLMJobQueue:
    """Priority-based async job queue for LLM operations.

    Port of LLMJobQueue.cs:
    - Priority buckets (lower number = higher priority)
    - Sequential processing (one job at a time to avoid overwhelming Ollama)
    - Future-based result waiting
    """

    def __init__(self):
        self._queue: asyncio.PriorityQueue[LLMJob] = asyncio.PriorityQueue()
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self):
        """Start the background queue processor."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stop the queue processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def enqueue_and_wait(
        self,
        job_fn: Callable[[], Coroutine],
        priority: int = 10,
    ) -> Any:
        """Enqueue a job and wait for its result.

        Args:
            job_fn: Async callable that returns the result
            priority: Lower = higher priority (processed first)

        Returns:
            The result from job_fn
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        job = LLMJob(priority=priority, job_fn=job_fn, future=future)
        await self._queue.put(job)

        return await future

    def enqueue_fire_and_forget(
        self,
        job_fn: Callable[[], Coroutine],
        priority: int = 50,
    ):
        """Enqueue a job without waiting for the result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        job = LLMJob(priority=priority, job_fn=job_fn, future=future)
        self._queue.put_nowait(job)

    async def _process_queue(self):
        """Process jobs sequentially by priority."""
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                result = await job.job_fn()
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as e:
                logger.error("Job queue error: %s", e)
                if not job.future.done():
                    job.future.set_exception(e)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()
