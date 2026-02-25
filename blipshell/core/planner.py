"""Task planning: plan generation via LLM.

ComplexityClassifier was removed — heuristic planning replaced by model judgment.
The !plan CLI prefix now directly sets force_plan=True in agent.chat().
See git history for the original ComplexityClassifier implementation.

TaskPlanner sends the request to the LLM to generate a numbered step list.
"""

import logging
import re
from typing import Optional

from blipshell.llm.prompts import generate_plan, UTILITY_SYSTEM_PROMPT
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import PlannerConfig
from blipshell.models.task import PlanStatus, TaskPlan, TaskStep

logger = logging.getLogger(__name__)

# Regex to parse numbered steps from LLM output
_STEP_PATTERN = re.compile(
    r"^\s*(\d+)\.\s*(.+?)(?:\s*\((\w+(?:_\w+)*)\))?\s*$",
    re.MULTILINE,
)


class TaskPlanner:
    """Generates multi-step plans from user requests via LLM."""

    def __init__(
        self,
        router: LLMRouter,
        sqlite: SQLiteStore,
        config: PlannerConfig,
    ):
        self.router = router
        self.sqlite = sqlite
        self.config = config
        self.active_project: dict | None = None  # set by Agent when project is active

    async def create_plan(
        self,
        user_request: str,
        session_id: Optional[int] = None,
        conversation_context: str = "",
    ) -> TaskPlan:
        """Generate a plan for the user request and persist it."""
        # Generate plan via LLM — use coding model when project is active
        task_type = TaskType.CODING if self.active_project else TaskType.TOOL_CALLING
        prompt = generate_plan(user_request, conversation_context=conversation_context)
        raw_response = await self.router.generate(
            task_type, prompt, system=UTILITY_SYSTEM_PROMPT,
        )

        # Parse steps
        steps = self._parse_steps(raw_response)
        if not steps:
            # Fallback: treat the whole request as a single step
            steps = [TaskStep(
                step_number=1,
                description=user_request,
            )]

        # Enforce max steps
        steps = steps[:self.config.max_steps]

        # Create plan in DB
        plan = TaskPlan(
            session_id=session_id,
            user_request=user_request,
            status=PlanStatus.APPROVED if self.config.auto_approve else PlanStatus.PLANNING,
        )
        plan_id = await self.sqlite.create_plan(plan)
        plan.id = plan_id

        # Create steps in DB
        for step in steps:
            step.plan_id = plan_id
            step_id = await self.sqlite.create_step(step)
            step.id = step_id

        plan.steps = steps
        logger.info(
            "Created plan #%d with %d steps for: %s",
            plan_id, len(steps), user_request[:80],
        )
        return plan

    def _parse_steps(self, raw_response: str) -> list[TaskStep]:
        """Parse numbered steps from LLM response."""
        steps = []
        for match in _STEP_PATTERN.finditer(raw_response):
            step_num = int(match.group(1))
            description = match.group(2).strip()
            tool_hint = match.group(3)  # may be None

            steps.append(TaskStep(
                step_number=step_num,
                description=description,
                tool_hint=tool_hint,
            ))

        # Re-number sequentially in case LLM numbering is off
        for i, step in enumerate(steps):
            step.step_number = i + 1

        return steps
