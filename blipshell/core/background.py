"""Background task manager for async/long-running tasks.

Users (or the LLM) can kick off tasks that run asynchronously.
Tasks are tracked in SQLite and executed via the LLM router.
Supports targeting a specific endpoint (e.g. offloading to a remote PC).
"""

import asyncio
import logging
from typing import Callable, Optional

from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import WorkerConfig
from blipshell.models.task import BackgroundTask, BackgroundTaskStatus

# Avoid circular import — processor is optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from blipshell.memory.processor import MemoryProcessor

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manages submission, execution, and tracking of background tasks."""

    def __init__(
        self,
        router: LLMRouter,
        sqlite: SQLiteStore,
        worker_config: WorkerConfig,
        processor: Optional["MemoryProcessor"] = None,
    ):
        self.router = router
        self.sqlite = sqlite
        self.worker_config = worker_config
        self.processor = processor
        self._running_tasks: dict[int, asyncio.Task] = {}
        self._completed_ids: list[int] = []  # task IDs that finished since last check
        self._on_complete: Optional[Callable[[int, str, str], None]] = None

    def set_on_complete(self, callback: Optional[Callable[[int, str, str], None]]):
        """Set a callback for when a task completes: callback(task_id, title, status)."""
        self._on_complete = callback

    def pop_completed(self) -> list[int]:
        """Return and clear the list of task IDs that completed since last check."""
        ids = self._completed_ids[:]
        self._completed_ids.clear()
        return ids

    async def submit_task(
        self,
        title: str,
        task_type: str = "custom",
        prompt: str = "",
        session_id: Optional[int] = None,
        plan_id: Optional[int] = None,
        priority: int = 0,
        target_endpoint: Optional[str] = None,
    ) -> int:
        """Submit a background task. Returns the task ID.

        If target_endpoint is set, the task runs locally but routes the
        LLM call to that specific endpoint. If worker.enabled is True and
        no target is specified, auto-routes based on task_type config.
        """
        # Auto-route to remote if configured
        if not target_endpoint and self.worker_config.enabled:
            if task_type in self.worker_config.task_types_for_remote:
                target_endpoint = self.worker_config.default_remote_endpoint or None

        task = BackgroundTask(
            session_id=session_id,
            plan_id=plan_id,
            title=title,
            task_type=task_type,
            prompt=prompt,
            priority=priority,
            target_endpoint=target_endpoint,
        )
        task_id = await self.sqlite.create_background_task(task)

        # Always execute locally (using target endpoint's client if specified)
        asyncio_task = asyncio.create_task(
            self._run_task(task_id, prompt, target_endpoint),
        )
        self._running_tasks[task_id] = asyncio_task

        ep_label = f" on '{target_endpoint}'" if target_endpoint else ""
        logger.info("Background task #%d started%s: %s", task_id, ep_label, title)

        return task_id

    async def _run_task(
        self,
        task_id: int,
        prompt: str,
        target_endpoint: Optional[str] = None,
    ):
        """Execute a background task via LLM.

        A target constrains the router; it never bypasses its privacy policy.
        """
        token = None

        async def update(_task_id, **fields):
            if token is None:
                return False
            return await self.sqlite.update_claimed_background_task(task_id, token, **fields)

        try:
            token = await self.sqlite.claim_background_task(task_id)
            if not token:
                return  # Another local/remote runner owns this work.
            if not await update(
                task_id,
                status=BackgroundTaskStatus.RUNNING,
                progress_message="Starting...",
            ):
                return

            # Determine task type for routing
            task = await self.sqlite.get_background_task(task_id)
            task_type_map = {
                "research": TaskType.REASONING,
                "analysis": TaskType.REASONING,
                "summarization": TaskType.SUMMARIZATION,
                "code_review": TaskType.CODING,
            }
            llm_task_type = task_type_map.get(task.task_type, TaskType.REASONING)

            if not await update(
                task_id, progress_pct=0.5, progress_message="Processing...",
            ):
                return
            result = await self.router.generate(
                llm_task_type, prompt, target_endpoint=target_endpoint,
            )

            accepted = await update(
                task_id,
                status=BackgroundTaskStatus.COMPLETED,
                progress_pct=1.0,
                progress_message="Done",
                result=result,
            )
            if not accepted:
                return
            self._completed_ids.append(task_id)
            logger.info("Background task #%d completed", task_id)

            # Feed result through memory pipeline so it's searchable
            if self.processor and result:
                try:
                    await self.processor.process_message(
                        text=result,
                        role="assistant",
                        session_id=task.session_id or 0,
                    )
                except Exception as mem_err:
                    logger.error("Background task #%d memory save failed: %s", task_id, mem_err)

        except asyncio.CancelledError:
            await update(
                task_id,
                status=BackgroundTaskStatus.CANCELLED,
                progress_message="Cancelled",
            )
            logger.info("Background task #%d cancelled", task_id)

        except Exception as e:
            logger.error("Background task #%d failed: %s", task_id, e)
            await update(
                task_id,
                status=BackgroundTaskStatus.FAILED,
                error_message=str(e),
                progress_message="Failed",
            )
            self._completed_ids.append(task_id)

        finally:
            self._running_tasks.pop(task_id, None)

    async def get_status(self, task_id: int) -> Optional[BackgroundTask]:
        """Get the status of a background task."""
        return await self.sqlite.get_background_task(task_id)

    async def get_all_active(
        self, session_id: Optional[int] = None,
    ) -> list[BackgroundTask]:
        """Get all non-completed tasks."""
        all_tasks = await self.sqlite.list_background_tasks(session_id=session_id)
        return [
            t for t in all_tasks
            if t.status not in (
                BackgroundTaskStatus.COMPLETED,
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.CANCELLED,
            )
        ]

    async def cancel_task(self, task_id: int) -> bool:
        """Cancel a running or pending task."""
        task = await self.sqlite.get_background_task(task_id)
        if not task:
            return False

        if task.status in (
            BackgroundTaskStatus.COMPLETED,
            BackgroundTaskStatus.FAILED,
            BackgroundTaskStatus.CANCELLED,
        ):
            return False

        # Cancel the asyncio task if running locally
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
        else:
            # Just update DB (remote worker will see status change)
            await self.sqlite.update_background_task(
                task_id,
                status=BackgroundTaskStatus.CANCELLED,
                progress_message="Cancelled",
            )

        return True

    async def list_all(
        self, session_id: Optional[int] = None, limit: int = 50,
    ) -> list[BackgroundTask]:
        """List all background tasks for a session."""
        return await self.sqlite.list_background_tasks(
            session_id=session_id, limit=limit,
        )
