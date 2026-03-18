"""Tools for LLM-to-user interaction during execution."""

import logging
from typing import Callable, Awaitable, Optional

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)

# Type for the callback: async fn(question) -> answer
AskUserCallback = Callable[[str], Awaitable[str]]


class TaskCompleteTool(Tool):
    """Signal that the current task is complete.

    Instead of asking the LLM to output a magic string like "TASK_COMPLETE",
    this tool leverages the model's tool-calling training. The model calls this
    tool exactly like any other tool, which is far more reliable than expecting
    a specific text token in freeform output.

    Based on patterns from Cline (attempt_completion) and OpenHands (AgentFinishAction).
    """

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="task_complete",
            description=(
                "Signal that the current task is finished. Call this tool when you have "
                "completed all the work requested. Include a summary of what you did, "
                "files you changed, and anything the user should know or test. "
                "Do NOT call this until you have actually made the changes — "
                "planning or describing what you would do is not completion."
            ),
            parameters=[
                ToolParameter(
                    name="summary",
                    type=ToolParameterType.STRING,
                    description="Summary of what you did (2-4 sentences)",
                ),
                ToolParameter(
                    name="files_modified",
                    type=ToolParameterType.STRING,
                    description="Comma-separated list of files created or modified",
                    required=False,
                ),
                ToolParameter(
                    name="decisions_made",
                    type=ToolParameterType.STRING,
                    description="Key design decisions or trade-offs worth noting",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        summary: str = "",
        files_modified: str = "",
        decisions_made: str = "",
        **kwargs,
    ) -> str:
        # Build a structured completion message
        parts = []
        if summary:
            parts.append(summary)
        if files_modified:
            parts.append(f"Files: {files_modified}")
        if decisions_made:
            parts.append(f"Decisions: {decisions_made}")
        return "\n".join(parts) if parts else "Task complete."


class ConfirmPlanTool(Tool):
    """Guardrails tool: confirm a plan with the user before execution.

    The LLM lists its planned steps and postconditions. The user reviews
    and approves, modifies, or rejects before any changes are made.

    When a GuardrailsEngine is attached, the approved checklist is stored
    for later use by the completion audit.
    """
    read_only = True

    def __init__(
        self,
        callback: Optional[AskUserCallback] = None,
        guardrails_engine=None,
    ):
        self.callback = callback
        self.guardrails_engine = guardrails_engine

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="confirm_plan",
            description=(
                "Present your plan to the user for approval BEFORE making changes. "
                "Call this at the start of complex tasks (3+ files, ambiguous requirements, "
                "destructive changes). List your planned steps and what should be true when done.\n\n"
                "The user will approve, modify, or reject the plan. Do NOT proceed with "
                "changes until the plan is confirmed.\n\n"
                "Skip this for simple, obvious tasks (single file edit, clear instruction)."
            ),
            parameters=[
                ToolParameter(
                    name="plan",
                    type=ToolParameterType.STRING,
                    description="Your planned steps, one per line (e.g. '1. Read X\\n2. Edit Y\\n3. Create Z')",
                ),
                ToolParameter(
                    name="postconditions",
                    type=ToolParameterType.STRING,
                    description="What should be true when the task is done (e.g. 'Tests pass, new endpoint returns 200')",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        plan: str = "",
        postconditions: str = "",
        **kwargs,
    ) -> str:
        if not plan:
            return "Error: 'plan' argument is required."

        # Format for display
        formatted = f"Proposed plan:\n{plan}"
        if postconditions:
            formatted += f"\n\nDone when:\n{postconditions}"
        formatted += "\n\nApprove this plan? (yes / modify / reject)"

        # Get user confirmation
        if self.callback:
            try:
                answer = await self.callback(formatted)
            except Exception as e:
                logger.error("confirm_plan callback failed: %s", e)
                answer = "Plan review failed — proceed with caution and keep changes minimal."
        else:
            answer = "approved"

        # Parse steps for checklist storage
        steps = [
            line.strip().lstrip("0123456789.-) ")
            for line in plan.strip().splitlines()
            if line.strip()
        ]
        postcond_list = [
            line.strip().lstrip("0123456789.-) ")
            for line in postconditions.strip().splitlines()
            if line.strip()
        ] if postconditions else []

        answer_lower = answer.strip().lower()

        if answer_lower in ("yes", "y", "approved", "approve", "ok", "lgtm", "go", ""):
            # Store checklist for completion audit
            if self.guardrails_engine:
                self.guardrails_engine.record_checklist(steps, postcond_list)
            return "Plan approved by user. Proceed with execution."
        elif answer_lower.startswith(("reject", "no")):
            return f"Plan rejected by user: {answer}. Ask the user what they want instead."
        else:
            # User modified the plan
            if self.guardrails_engine:
                self.guardrails_engine.record_checklist(steps, postcond_list)
            return f"User feedback on plan: {answer}. Adjust your approach accordingly."


class AskUserTool(Tool):
    read_only = True
    """Allows the LLM to ask the user a question mid-execution.

    In interactive mode (CLI), prompts the user and returns their answer.
    In non-interactive mode (benchmarks), returns a canned response.

    Supports structured options: pass comma-separated options and the user
    sees a numbered selection list. Free-text input is always allowed unless
    allow_free_text is set to false.
    """

    def __init__(self, callback: Optional[AskUserCallback] = None):
        self.callback = callback

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ask_user",
            description=(
                "Ask the user a question when you need clarification. Use this when:\n"
                "- The task description is ambiguous or could be interpreted multiple ways\n"
                "- You need to choose between multiple valid approaches\n"
                "- Something has failed twice and you're unsure how to proceed\n"
                "- The task would delete or significantly modify existing code\n\n"
                "For multiple-choice questions, provide options as a comma-separated list.\n"
                "Do not ask trivial questions — only ask when user input would change your approach."
            ),
            parameters=[
                ToolParameter(
                    name="question",
                    type=ToolParameterType.STRING,
                    description="The question to ask the user",
                ),
                ToolParameter(
                    name="options",
                    type=ToolParameterType.STRING,
                    description="Comma-separated list of options (e.g. 'Option A, Option B, Option C'). User can pick by number or type a custom answer.",
                    required=False,
                ),
                ToolParameter(
                    name="allow_free_text",
                    type=ToolParameterType.BOOLEAN,
                    description="Allow user to type a custom answer beyond the options (default: true)",
                    required=False,
                ),
            ],
        )

    async def execute(self, question: str = "", options: str = "",
                      allow_free_text: bool = True, **kwargs) -> str:
        if not question:
            return "Error: 'question' argument is required."

        # Format the question with options if provided
        formatted = self._format_question(question, options, allow_free_text)

        if self.callback:
            try:
                answer = await self.callback(formatted)
                # If user answered with a number and we have options, resolve it
                if options:
                    answer = self._resolve_option(answer, options)
                return f"User's answer: {answer}"
            except Exception as e:
                logger.error("ask_user callback failed: %s", e)
                return "User did not respond. Make your best judgment and proceed."
        # Non-interactive fallback (benchmarks)
        return "No user available. Make your best judgment and proceed."

    @staticmethod
    def _format_question(question: str, options: str, allow_free_text: bool) -> str:
        """Format question with numbered options."""
        if not options:
            return question

        option_list = [o.strip() for o in options.split(",") if o.strip()]
        if not option_list:
            return question

        lines = [question, ""]
        for i, opt in enumerate(option_list, 1):
            lines.append(f"  {i}. {opt}")
        if allow_free_text:
            lines.append(f"\nEnter a number (1-{len(option_list)}) or type your own answer:")
        else:
            lines.append(f"\nEnter a number (1-{len(option_list)}):")
        return "\n".join(lines)

    @staticmethod
    def _resolve_option(answer: str, options: str) -> str:
        """If the answer is a number, resolve it to the corresponding option."""
        answer = answer.strip()
        option_list = [o.strip() for o in options.split(",") if o.strip()]
        try:
            idx = int(answer)
            if 1 <= idx <= len(option_list):
                return option_list[idx - 1]
        except ValueError:
            pass
        return answer
