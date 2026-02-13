"""Task planning: complexity classification and plan generation.

ComplexityClassifier uses heuristics (no LLM call) to decide if a message
needs multi-step planning. TaskPlanner sends the request to the LLM to
generate a numbered step list.
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

# Patterns that suggest a complex multi-step request
_ACTION_VERBS = re.compile(
    r"\b(research|analyze|summarize|compare|create|build|find|search|fetch|"
    r"review|investigate|compile|generate|write|implement|set up|configure|"
    r"install|deploy|migrate|refactor|optimize|debug|diagnose)\b",
    re.IGNORECASE,
)

# Connectors that chain multiple actions
_CHAINING_WORDS = re.compile(
    r"\b(then|and then|after that|next|also|additionally|finally|"
    r"once .+ done|followed by|save .+ as|and save|and summarize)\b",
    re.IGNORECASE,
)

# Simple patterns that should skip planning
_SIMPLE_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|bye|"
    r"what is|what's|who is|who's|how are|good morning|good night)\b",
    re.IGNORECASE,
)


class ComplexityClassifier:
    """Decides if a message needs structured planning or can go through normal chat.

    Uses heuristic checks — no LLM call. Instant decision.
    """

    def __init__(self, config: PlannerConfig):
        self.config = config

    def needs_planning(self, message: str) -> bool:
        """Return True if the message should be planned, False for normal chat."""
        if not self.config.enabled:
            return False

        stripped = message.strip()

        # Skip very short messages
        if len(stripped) < 10:
            return False

        # Skip simple greetings/responses
        if _SIMPLE_PATTERNS.match(stripped):
            return False

        # Skip questions that are just asking for info (single-step)
        if stripped.endswith("?") and len(stripped.split()) < 15:
            return False

        word_count = len(stripped.split())

        # Check for chaining words (strong signal)
        if _CHAINING_WORDS.search(stripped):
            return True

        # Check for multiple action verbs
        action_matches = _ACTION_VERBS.findall(stripped)
        if len(action_matches) >= 2:
            return True

        # Long messages with at least one action verb
        if word_count >= self.config.complexity_threshold_words and action_matches:
            return True

        return False


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

    async def create_plan(
        self,
        user_request: str,
        session_id: Optional[int] = None,
    ) -> TaskPlan:
        """Generate a plan for the user request and persist it."""
        # Generate plan via LLM
        prompt = generate_plan(user_request)
        raw_response = await self.router.generate(
            TaskType.REASONING, prompt, system=UTILITY_SYSTEM_PROMPT,
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
