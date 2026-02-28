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


class AskUserTool(Tool):
    read_only = True
    """Allows the LLM to ask the user a question mid-execution.

    In interactive mode (CLI), prompts the user and returns their answer.
    In non-interactive mode (benchmarks), returns a canned response.
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
                "- The task would delete or significantly modify existing code\n"
                "Do not ask trivial questions — only ask when user input would change your approach."
            ),
            parameters=[
                ToolParameter(
                    name="question",
                    type=ToolParameterType.STRING,
                    description="The question to ask the user",
                ),
            ],
        )

    async def execute(self, question: str = "", **kwargs) -> str:
        if not question:
            return "Error: 'question' argument is required."
        if self.callback:
            try:
                answer = await self.callback(question)
                return f"User's answer: {answer}"
            except Exception as e:
                logger.error("ask_user callback failed: %s", e)
                return "User did not respond. Make your best judgment and proceed."
        # Non-interactive fallback (benchmarks)
        return "No user available. Make your best judgment and proceed."
