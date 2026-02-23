"""Tools for LLM-to-user interaction during execution."""

import logging
from typing import Callable, Awaitable, Optional

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)

# Type for the callback: async fn(question) -> answer
AskUserCallback = Callable[[str], Awaitable[str]]


class AskUserTool(Tool):
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
                "Ask the user a question when you need clarification, want to "
                "confirm an approach before proceeding, or need to present options. "
                "Use this for important decisions — don't ask trivial questions."
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
