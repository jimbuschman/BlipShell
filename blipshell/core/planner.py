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

# Imperative action verbs — only match commands, not casual mentions
# These are verbs that imply the user is ASKING the agent to DO something
_ACTION_VERBS = re.compile(
    r"\b(research|analyze|summarize|compare|create|build|fetch|"
    r"investigate|compile|generate|implement|set up|configure|"
    r"install|deploy|migrate|refactor|optimize|debug|diagnose)\b",
    re.IGNORECASE,
)

# Connectors that explicitly chain multiple actions together
_CHAINING_WORDS = re.compile(
    r"\b(and then|after that|once .+ done|followed by|"
    r"save .+ as|and save|and summarize|then summarize|then save)\b",
    re.IGNORECASE,
)

# Patterns that should never trigger planning
_SKIP_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no|bye|"
    r"what is|what's|who is|who's|how are|good morning|good night|"
    r"i'm |i am |i was |i think|i feel|i want|i like|i need|"
    r"can you|could you|would you|do you|did you|have you|"
    r"tell me|show me|explain|help me|what do you)\b",
    re.IGNORECASE,
)


class ComplexityClassifier:
    """Decides if a message needs structured planning or can go through normal chat.

    Uses heuristic checks — no LLM call. Instant decision.
    Very conservative — only triggers on explicit multi-step commands.
    Users can always force planning with !plan prefix.
    """

    def __init__(self, config: PlannerConfig):
        self.config = config

    def needs_planning(self, message: str) -> bool:
        """Return True if the message should be planned, False for normal chat."""
        if not self.config.enabled:
            return False

        stripped = message.strip()

        # Skip short messages
        if len(stripped) < 20:
            return False

        # Skip conversational patterns (questions, statements about self, etc.)
        if _SKIP_PATTERNS.match(stripped):
            return False

        # Skip anything that ends with a question mark
        if stripped.endswith("?"):
            return False

        word_count = len(stripped.split())

        # MUST have explicit chaining words — this is the primary trigger
        # e.g. "research X and then summarize it and save to memory"
        has_chaining = bool(_CHAINING_WORDS.search(stripped))

        # Also need at least one action verb alongside the chaining
        action_matches = _ACTION_VERBS.findall(stripped)

        if has_chaining and len(action_matches) >= 1:
            return True

        # Multiple distinct action verbs in a long message (3+)
        if len(action_matches) >= 3 and word_count >= self.config.complexity_threshold_words:
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
