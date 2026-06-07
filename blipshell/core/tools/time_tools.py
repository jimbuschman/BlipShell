"""Tool for querying the current date and time.

The system prompt carries an absolute time anchor and the conversation history
is stamped with relative labels, so the model has passive time awareness. This
tool exists for the cases where the model wants to be precise on demand — e.g.
"you said you'd try this for a week; has it been a week yet?" — without relying
on whatever stamp happened to be in context.
"""

from datetime import datetime, timezone

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition


class GetCurrentTimeTool(Tool):
    """Return the current date and time. Read-only (safe in plan mode)."""

    read_only = True

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_current_time",
            description=(
                "Get the current date and time. Use this when you need to reason "
                "precisely about timing — how long since something happened, whether "
                "a deadline or duration has elapsed, or what day it is right now. "
                "Returns both local time and UTC."
            ),
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone()
        return (
            f"Local: {now_local.strftime('%Y-%m-%d %H:%M:%S %A')} "
            f"({now_local.tzname()})\n"
            f"UTC:   {now_utc.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(ISO: {now_utc.isoformat()})"
        )
