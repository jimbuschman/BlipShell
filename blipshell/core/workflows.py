"""Workflow system: named, reusable YAML task templates.

Workflows define a fixed sequence of steps with {placeholder} parameter
substitution. The LLM fills in dynamic parts; the structure is fixed.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from blipshell.core.executor import TaskExecutor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.task import PlanStatus, StepStatus, TaskPlan, TaskStep

logger = logging.getLogger(__name__)


class WorkflowStep:
    """A single step in a workflow template."""

    def __init__(
        self,
        description: str,
        tool_hint: Optional[str] = None,
        prompt: Optional[str] = None,
        condition: Optional[str] = None,
    ):
        self.description = description
        self.tool_hint = tool_hint
        self.prompt = prompt
        self.condition = condition

    def substitute(self, params: dict[str, Any]) -> "WorkflowStep":
        """Return a new step with {placeholder} values replaced."""
        def _sub(text: str | None) -> str | None:
            if text is None:
                return None
            result = text
            for key, value in params.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result

        return WorkflowStep(
            description=_sub(self.description),
            tool_hint=_sub(self.tool_hint),
            prompt=_sub(self.prompt),
            condition=_sub(self.condition),
        )


class WorkflowDefinition:
    """A named workflow loaded from a YAML file."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[dict[str, str]],
        steps: list[WorkflowStep],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters  # [{name, description, default?}]
        self.steps = steps

    @classmethod
    def from_yaml(cls, path: Path) -> "WorkflowDefinition":
        """Load a workflow definition from a YAML file."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        steps = []
        for step_data in data.get("steps", []):
            steps.append(WorkflowStep(
                description=step_data["description"],
                tool_hint=step_data.get("tool_hint"),
                prompt=step_data.get("prompt"),
                condition=step_data.get("condition"),
            ))

        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            parameters=data.get("parameters", []),
            steps=steps,
        )

    def to_plan(
        self,
        params: dict[str, Any],
        session_id: Optional[int] = None,
    ) -> TaskPlan:
        """Convert this workflow + params into a TaskPlan."""
        # Apply defaults for missing params
        filled_params = {}
        for p in self.parameters:
            name = p["name"]
            filled_params[name] = params.get(name, p.get("default", ""))
        # Also include any extra params
        for k, v in params.items():
            if k not in filled_params:
                filled_params[k] = v

        # Build substituted steps
        task_steps = []
        for i, ws in enumerate(self.steps, 1):
            substituted = ws.substitute(filled_params)

            # Evaluate condition if present
            if substituted.condition:
                # Simple truthy check — condition is a param name
                cond_val = filled_params.get(substituted.condition, "")
                if not cond_val or cond_val.lower() in ("false", "no", "0", ""):
                    continue

            task_steps.append(TaskStep(
                step_number=i,
                description=substituted.prompt or substituted.description,
                tool_hint=substituted.tool_hint,
            ))

        # Re-number after condition filtering
        for i, step in enumerate(task_steps):
            step.step_number = i + 1

        user_request = f"Workflow '{self.name}' with params: {filled_params}"
        return TaskPlan(
            session_id=session_id,
            user_request=user_request,
            status=PlanStatus.APPROVED,
            steps=task_steps,
        )


class WorkflowRegistry:
    """Discovers and manages workflow YAML files from a directory."""

    def __init__(self, workflows_dir: str | Path = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self._workflows: dict[str, WorkflowDefinition] = {}
        self._load_all()

    def _load_all(self):
        """Load all workflow YAML files from the directory."""
        if not self.workflows_dir.exists():
            logger.debug("Workflows directory does not exist: %s", self.workflows_dir)
            return

        for path in self.workflows_dir.glob("*.yaml"):
            try:
                wf = WorkflowDefinition.from_yaml(path)
                self._workflows[wf.name] = wf
                logger.info("Loaded workflow: %s (%d steps)", wf.name, len(wf.steps))
            except Exception as e:
                logger.error("Failed to load workflow %s: %s", path, e)

    def get(self, name: str) -> Optional[WorkflowDefinition]:
        """Get a workflow by name."""
        return self._workflows.get(name)

    def list_all(self) -> list[WorkflowDefinition]:
        """List all available workflows."""
        return list(self._workflows.values())

    def reload(self):
        """Reload all workflows from disk."""
        self._workflows.clear()
        self._load_all()


class WorkflowExecutor:
    """Converts workflows into TaskPlans and executes them via TaskExecutor."""

    def __init__(
        self,
        registry: WorkflowRegistry,
        task_executor: TaskExecutor,
        sqlite: SQLiteStore,
    ):
        self.registry = registry
        self.executor = task_executor
        self.sqlite = sqlite

    async def run_workflow(
        self,
        workflow_name: str,
        params: dict[str, Any],
        session_id: Optional[int] = None,
        on_token: Optional[callable] = None,
    ) -> str:
        """Run a named workflow with parameter substitution.

        Returns the final summary from the executor.
        """
        workflow = self.registry.get(workflow_name)
        if not workflow:
            raise KeyError(f"Workflow '{workflow_name}' not found")

        # Convert to TaskPlan
        plan = workflow.to_plan(params, session_id=session_id)

        # Persist plan and steps
        plan_id = await self.sqlite.create_plan(plan)
        plan.id = plan_id
        for step in plan.steps:
            step.plan_id = plan_id
            step_id = await self.sqlite.create_step(step)
            step.id = step_id

        logger.info(
            "Running workflow '%s' as plan #%d (%d steps)",
            workflow_name, plan_id, len(plan.steps),
        )

        # Execute via TaskExecutor
        return await self.executor.execute_plan(plan, on_token=on_token)
