"""Plan mode tools — LLM self-restricts to read-only exploration."""

from blipshell.core.tools.base import Tool, ToolRegistry
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


class EnterPlanModeTool(Tool):
    """Switch to read-only planning mode before making complex changes."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="enter_plan_mode",
            description=(
                "Enter read-only planning mode. Use this BEFORE making complex "
                "multi-file changes to explore the codebase and design your approach.\n\n"
                "In plan mode, only read-only tools are available (read_file, grep_files, "
                "glob_files, list_directory, web_search, search_memories, git_status, "
                "git_diff, ask_user). Write tools are disabled until you call "
                "exit_plan_mode with your plan summary.\n\n"
                "When to use:\n"
                "- Complex multi-file changes where you need to understand the code first\n"
                "- Unfamiliar code where exploring first prevents wasted edits\n"
                "- Tasks where you want to present a plan before making changes\n\n"
                "When NOT to use:\n"
                "- Simple single-file edits where you already know what to change\n"
                "- Tasks where you've already read the relevant files"
            ),
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        if self.registry.in_plan_mode:
            return "Already in plan mode. Use exit_plan_mode when your plan is ready."
        self.registry._plan_mode = True
        return (
            "Plan mode activated. Only read-only tools are now available.\n"
            "Explore the codebase, then call exit_plan_mode(summary=...) with your plan."
        )


class ExitPlanModeTool(Tool):
    """Exit planning mode and present a plan summary."""
    read_only = True  # Must be available during plan mode

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="exit_plan_mode",
            description=(
                "Exit plan mode and present your plan. Write tools are re-enabled.\n\n"
                "Call this when you have explored enough to form a clear plan. Include:\n"
                "- What files to modify and why\n"
                "- What new files to create (if any)\n"
                "- Key design decisions\n"
                "- Order of operations"
            ),
            parameters=[
                ToolParameter(
                    name="summary",
                    type=ToolParameterType.STRING,
                    description="Summary of your planned approach",
                ),
            ],
        )

    async def execute(self, summary: str = "", **kwargs) -> str:
        if not self.registry.in_plan_mode:
            return "Not in plan mode. No need to exit."
        self.registry._plan_mode = False
        return (
            f"Plan mode exited. Write tools re-enabled. Proceed with implementation.\n\n"
            f"Plan: {summary}"
        )
