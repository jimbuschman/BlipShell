"""Background task manager for async/long-running tasks.

Users (or the LLM) can kick off tasks that run asynchronously.
Tasks are tracked in SQLite and executed via the LLM router.
When a target_endpoint is set, the task is left for a remote worker to pick up.
"""

import asyncio
import logging
from typing import Optional

from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import WorkerConfig
from blipshell.models.task import BackgroundTask, BackgroundTaskStatus

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manages submission, execution, and tracking of background tasks."""

    def __init__(
        self,
        router: LLMRouter,
        sqlite: SQLiteStore,
        worker_config: WorkerConfig,
    ):
        self.router = router
        self.sqlite = sqlite
        self.worker_config = worker_config
        self._running_tasks: dict[int, asyncio.Task] = {}

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

        If target_endpoint is set or the task_type is configured for remote,
        the task is left in 'pending' for a remote worker. Otherwise, it
        runs locally via asyncio.
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

        if target_endpoint:
            # Leave for remote worker to pick up
            logger.info(
                "Background task #%d queued for remote worker '%s': %s",
                task_id, target_endpoint, title,
            )
        else:
            # Execute locally
            asyncio_task = asyncio.create_task(self._run_task(task_id, prompt))
            self._running_tasks[task_id] = asyncio_task
            logger.info("Background task #%d started locally: %s", task_id, title)

        return task_id

    async def _run_task(self, task_id: int, prompt: str):
        """Execute a background task locally via LLM."""
        try:
            await self.sqlite.update_background_task(
                task_id,
                status=BackgroundTaskStatus.RUNNING,
                progress_message="Starting...",
            )

            # Determine task type for routing
            task = await self.sqlite.get_background_task(task_id)
            task_type_map = {
                "research": TaskType.REASONING,
                "analysis": TaskType.REASONING,
                "summarization": TaskType.SUMMARIZATION,
                "code_review": TaskType.CODING,
            }
            llm_task_type = task_type_map.get(task.task_type, TaskType.REASONING)

            # Run LLM generation
            await self.sqlite.update_background_task(
                task_id, progress_pct=0.5, progress_message="Processing...",
            )

            result = await self.router.generate(llm_task_type, prompt)

            await self.sqlite.update_background_task(
                task_id,
                status=BackgroundTaskStatus.COMPLETED,
                progress_pct=1.0,
                progress_message="Done",
                result=result,
            )
            logger.info("Background task #%d completed", task_id)

        except asyncio.CancelledError:
            await self.sqlite.update_background_task(
                task_id,
                status=BackgroundTaskStatus.CANCELLED,
                progress_message="Cancelled",
            )
            logger.info("Background task #%d cancelled", task_id)

        except Exception as e:
            logger.error("Background task #%d failed: %s", task_id, e)
            await self.sqlite.update_background_task(
                task_id,
                status=BackgroundTaskStatus.FAILED,
                error_message=str(e),
                progress_message="Failed",
            )

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
