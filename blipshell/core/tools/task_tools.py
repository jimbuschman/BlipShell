"""LLM-facing tools for background tasks and workflows."""

import json
from typing import Optional

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


class StartBackgroundTaskTool(Tool):
    """Allows the LLM to kick off background research/analysis tasks."""

    def __init__(self, background_manager, session_id: Optional[int] = None):
        self.background_manager = background_manager
        self.session_id = session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="start_background_task",
            description=(
                "Start a long-running background task like research or analysis. "
                "The task runs asynchronously and results can be checked later. "
                "Use this for tasks that may take a while, so the user doesn't have to wait."
            ),
            parameters=[
                ToolParameter(
                    name="title",
                    type=ToolParameterType.STRING,
                    description="Short title describing the task",
                ),
                ToolParameter(
                    name="prompt",
                    type=ToolParameterType.STRING,
                    description="Detailed prompt/instructions for the background task",
                ),
                ToolParameter(
                    name="task_type",
                    type=ToolParameterType.STRING,
                    description="Type of task: research, analysis, summarization, code_review, custom",
                    required=False,
                ),
            ],
        )

    async def execute(
        self, title: str, prompt: str, task_type: str = "custom", **kwargs,
    ) -> str:
        task_id = await self.background_manager.submit_task(
            title=title,
            task_type=task_type,
            prompt=prompt,
            session_id=self.session_id,
        )
        return f"Background task started (ID: {task_id}): {title}"


class CheckBackgroundTaskTool(Tool):
    """Allows the LLM to check the status of a background task."""

    def __init__(self, background_manager):
        self.background_manager = background_manager

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="check_background_task",
            description="Check the status and result of a background task by its ID.",
            parameters=[
                ToolParameter(
                    name="task_id",
                    type=ToolParameterType.INTEGER,
                    description="The ID of the background task to check",
                ),
            ],
        )

    async def execute(self, task_id: int, **kwargs) -> str:
        task = await self.background_manager.get_status(task_id)
        if not task:
            return f"Background task #{task_id} not found."

        parts = [
            f"Task #{task.id}: {task.title}",
            f"Status: {task.status}",
            f"Progress: {task.progress_pct:.0%} — {task.progress_message}",
        ]

        if task.result:
            parts.append(f"Result:\n{task.result[:2000]}")
        if task.error_message:
            parts.append(f"Error: {task.error_message}")

        return "\n".join(parts)


class ListBackgroundTasksTool(Tool):
    """Allows the LLM to list active background tasks."""

    def __init__(self, background_manager, session_id: Optional[int] = None):
        self.background_manager = background_manager
        self.session_id = session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_background_tasks",
            description="List all background tasks for the current session, showing their status and progress.",
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        tasks = await self.background_manager.list_all(session_id=self.session_id)
        if not tasks:
            return "No background tasks found."

        lines = []
        for t in tasks[:20]:
            status_icon = {
                "pending": "...",
                "claimed": "...",
                "running": ">>>",
                "completed": "[OK]",
                "failed": "[!!]",
                "cancelled": "[--]",
            }.get(t.status, "???")
            lines.append(
                f"{status_icon} #{t.id}: {t.title} ({t.status}, {t.progress_pct:.0%})"
            )
        return "\n".join(lines)


class RunWorkflowTool(Tool):
    """Allows the LLM to trigger named workflows with parameters."""

    def __init__(self, workflow_executor, session_id: Optional[int] = None):
        self.workflow_executor = workflow_executor
        self.session_id = session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_workflow",
            description=(
                "Run a named workflow template with parameter substitution. "
                "Workflows are predefined multi-step task sequences."
            ),
            parameters=[
                ToolParameter(
                    name="workflow_name",
                    type=ToolParameterType.STRING,
                    description="Name of the workflow to run (e.g., 'research', 'code-review')",
                ),
                ToolParameter(
                    name="params",
                    type=ToolParameterType.STRING,
                    description="JSON string of parameters for the workflow, e.g. '{\"topic\": \"Python async\"}'",
                    required=False,
                ),
            ],
        )

    async def execute(
        self, workflow_name: str, params: str = "{}", **kwargs,
    ) -> str:
        try:
            param_dict = json.loads(params)
        except json.JSONDecodeError:
            return f"Invalid params JSON: {params}"

        try:
            result = await self.workflow_executor.run_workflow(
                workflow_name, param_dict, session_id=self.session_id,
            )
            return result
        except KeyError:
            return f"Workflow '{workflow_name}' not found."
        except Exception as e:
            return f"Workflow '{workflow_name}' failed: {e}"
