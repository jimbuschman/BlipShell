"""Session note tools — persistent state that survives context compaction.

Notes are key-value pairs stored in sessions.metadata_json. Both the LLM
(via tools) and user (via /notes CLI command) can manage them. Notes are
injected into the system prompt before each LLM call, so they're never
lost to compaction.
"""

import logging

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


class SaveNoteTool(Tool):
    """Save or update a named session note."""
    read_only = False

    def __init__(self, sqlite, session_id: int, notes_config, notes: dict[str, str]):
        self._sqlite = sqlite
        self._session_id = session_id
        self._config = notes_config
        self._notes = notes  # shared mutable dict — also used by GetNotesTool

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_note",
            description=(
                "Save a named session note that persists across context compaction.\n\n"
                "Use this to preserve important context that you'll need later:\n"
                "- The user's original request and key requirements\n"
                "- Important decisions or preferences stated by the user\n"
                "- Your current plan or approach\n"
                "- State you need to remember if the conversation gets compacted\n\n"
                "Notes are injected into every LLM call, so they're always visible.\n"
                "Update existing notes by using the same name."
            ),
            parameters=[
                ToolParameter(
                    name="name",
                    type=ToolParameterType.STRING,
                    description="Short name/key for this note (e.g. 'task', 'plan', 'decision')",
                ),
                ToolParameter(
                    name="content",
                    type=ToolParameterType.STRING,
                    description="The note content to save",
                ),
            ],
        )

    async def execute(self, name: str, content: str, **kwargs) -> str:
        from blipshell.memory.manager import estimate_tokens

        name = name.strip()
        content = content.strip()

        if not name:
            return "Error: note name cannot be empty."
        if not content:
            return "Error: note content cannot be empty."

        # Per-note token limit
        content_tokens = estimate_tokens(content)
        if content_tokens > self._config.max_note_tokens:
            return (
                f"Error: note '{name}' is {content_tokens} tokens, "
                f"max is {self._config.max_note_tokens}."
            )

        # Check note count (only for new notes, not updates)
        is_update = name in self._notes
        if not is_update and len(self._notes) >= self._config.max_notes:
            return (
                f"Error: max notes ({self._config.max_notes}) reached. "
                "Delete or update an existing note."
            )

        # Check total token budget
        total_tokens = sum(estimate_tokens(v) for k, v in self._notes.items() if k != name)
        total_tokens += content_tokens
        if total_tokens > self._config.max_total_tokens:
            return (
                f"Error: total notes would be {total_tokens} tokens, "
                f"max is {self._config.max_total_tokens}. Shorten this note or remove others."
            )

        # Save to in-memory cache
        self._notes[name] = content

        # Persist to database
        try:
            await self._sqlite.save_session_notes(self._session_id, self._notes)
        except Exception as e:
            logger.warning("Failed to persist note '%s': %s", name, e)
            return f"Note '{name}' saved in memory but failed to persist: {e}"

        action = "updated" if is_update else "saved"
        return f"Note '{name}' {action} ({content_tokens} tokens)."


class GetNotesTool(Tool):
    """Retrieve session notes."""
    read_only = True

    def __init__(self, sqlite, session_id: int, notes: dict[str, str]):
        self._sqlite = sqlite
        self._session_id = session_id
        self._notes = notes  # shared mutable dict with SaveNoteTool

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_notes",
            description=(
                "Retrieve session notes. Call with no arguments to list all notes, "
                "or pass a name to get a specific note."
            ),
            parameters=[
                ToolParameter(
                    name="name",
                    type=ToolParameterType.STRING,
                    description="Name of a specific note to retrieve (optional — omit for all)",
                    required=False,
                ),
            ],
        )

    async def execute(self, name: str = "", **kwargs) -> str:
        if not self._notes:
            return "No session notes."

        if name:
            name = name.strip()
            content = self._notes.get(name)
            if content is None:
                available = ", ".join(sorted(self._notes.keys()))
                return f"Note '{name}' not found. Available: {available}"
            return f"[{name}]\n{content}"

        # Return all notes
        parts = []
        for n, c in self._notes.items():
            parts.append(f"[{n}]\n{c}")
        return "\n\n".join(parts)


class DeleteNoteTool(Tool):
    """Delete a session note."""
    read_only = False

    def __init__(self, sqlite, session_id: int, notes: dict[str, str]):
        self._sqlite = sqlite
        self._session_id = session_id
        self._notes = notes

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="delete_note",
            description="Delete a session note by name.",
            parameters=[
                ToolParameter(
                    name="name",
                    type=ToolParameterType.STRING,
                    description="Name of the note to delete",
                ),
            ],
        )

    async def execute(self, name: str, **kwargs) -> str:
        name = name.strip()
        if name not in self._notes:
            available = ", ".join(sorted(self._notes.keys())) if self._notes else "none"
            return f"Note '{name}' not found. Available: {available}"

        del self._notes[name]

        try:
            await self._sqlite.save_session_notes(self._session_id, self._notes)
        except Exception as e:
            logger.warning("Failed to persist note deletion '%s': %s", name, e)
            return f"Note '{name}' deleted from memory but failed to persist: {e}"

        return f"Note '{name}' deleted."
