"""Follow-up queue tools — proactive memory across sessions.

Lets the LLM queue items to revisit later, check what's pending,
and mark items as resolved. Injected at session startup so the LLM
can proactively raise open items.
"""

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


class AddFollowUpTool(Tool):
    """Queue an item for follow-up in a future session."""
    read_only = False

    def __init__(self, sqlite, session_id: int | None = None, project: str | None = None):
        self._sqlite = sqlite
        self._session_id = session_id
        self._project = project

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="add_followup",
            description=(
                "Queue something to follow up on in a future session. Use this when:\n"
                "- The user mentions something they want to do later\n"
                "- A topic comes up that deserves investigation but isn't the current focus\n"
                "- You notice an unresolved issue worth tracking\n"
                "- The user says 'remind me', 'I should check', 'let's revisit', etc.\n\n"
                "Items appear at the start of the next session so nothing gets lost."
            ),
            parameters=[
                ToolParameter(
                    name="content",
                    type=ToolParameterType.STRING,
                    description="What to follow up on. Be specific enough to be useful later.",
                ),
                ToolParameter(
                    name="due_hint",
                    type=ToolParameterType.STRING,
                    description="Optional timing hint: 'tomorrow', 'next week', 'after deploy', etc.",
                    required=False,
                ),
            ],
        )

    async def execute(self, content: str, due_hint: str = "", **kwargs) -> str:
        fid = await self._sqlite.add_follow_up(
            content=content,
            session_id=self._session_id,
            project=self._project,
            due_hint=due_hint or None,
        )
        result = f"Follow-up #{fid} queued: {content}"
        if due_hint:
            result += f" (due: {due_hint})"
        return result


class ListFollowUpsTool(Tool):
    """Check what follow-ups are pending."""
    read_only = True

    def __init__(self, sqlite, project: str | None = None):
        self._sqlite = sqlite
        self._project = project

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_followups",
            description=(
                "List pending follow-up items. Use this to check what's queued up "
                "or to remind yourself what the user wanted to revisit."
            ),
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        items = await self._sqlite.get_pending_follow_ups(
            project=self._project, limit=20,
        )
        if not items:
            return "No pending follow-ups."

        lines = [f"Pending follow-ups ({len(items)}):"]
        for item in items:
            line = f"  #{item['id']}: {item['content']}"
            if item.get("due_hint"):
                line += f" [due: {item['due_hint']}]"
            if item.get("project"):
                line += f" (project: {item['project']})"
            age = item.get("created_at", "")
            if age:
                line += f" — added {age}"
            lines.append(line)
        return "\n".join(lines)


class ResolveFollowUpTool(Tool):
    """Mark a follow-up as done or dismiss it."""
    read_only = False

    def __init__(self, sqlite, session_id: int | None = None):
        self._sqlite = sqlite
        self._session_id = session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="resolve_followup",
            description=(
                "Mark a follow-up as resolved (completed) or dismissed (no longer relevant). "
                "Use the ID from list_followups."
            ),
            parameters=[
                ToolParameter(
                    name="id",
                    type=ToolParameterType.INTEGER,
                    description="Follow-up ID to resolve or dismiss.",
                ),
                ToolParameter(
                    name="action",
                    type=ToolParameterType.STRING,
                    description="'resolve' (done) or 'dismiss' (no longer relevant). Default: resolve.",
                    required=False,
                ),
            ],
        )

    async def execute(self, id: int, action: str = "resolve", **kwargs) -> str:
        if action == "dismiss":
            ok = await self._sqlite.dismiss_follow_up(id)
            return f"Follow-up #{id} dismissed." if ok else f"Follow-up #{id} not found or already resolved."
        else:
            ok = await self._sqlite.resolve_follow_up(id, self._session_id)
            return f"Follow-up #{id} resolved." if ok else f"Follow-up #{id} not found or already resolved."
