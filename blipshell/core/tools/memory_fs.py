"""Memory-as-filesystem tools — LLM-facing.

Exposes BlipShell's lessons, core memories, project digests, session summaries,
friction notes, and session notes as a navigable /memories/... tree.

Write surface (deliberately small):
  - /memories/core/    create/edit/delete, each gated by the approval callback
  - /memories/notes/   create/edit/delete, free (maps to session_notes store)
Everything else (lessons, digests, sessions, friction) is read-only.

Layered on top of:
  - MemoryFSBackend (persistent tiers, see blipshell/memory/fs_backend.py)
  - NotesBackend    (session_notes view, see blipshell/memory/fs_notes.py)
  - fs_paths.parse  (path parser + tier permission rules)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from blipshell.core.tools.base import Tool
from blipshell.memory.fs_backend import FSEntry, FSError, MemoryFSBackend
from blipshell.memory.fs_notes import NotesBackend
from blipshell.memory.fs_paths import (
    PathError,
    Tier,
    parse,
    requires_approval,
)
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


_TIER_OVERVIEW = """\
Memory layout:
  /memories/                       — list all tiers
  /memories/lessons/<project>/     — list lessons (READ-ONLY, pipeline-derived)
  /memories/lessons/<project>/<id>-<slug>.md   — read a specific lesson
  /memories/core/                  — list core memories
  /memories/core/<id>-<slug>.md    — read a core memory (write REQUIRES APPROVAL)
  /memories/digests/<project>.md   — read project digest (read-only)
  /memories/sessions/              — list recent sessions (read-only)
  /memories/sessions/<id>-<slug>.md — read a session summary
  /memories/friction/<id>-<slug>.md — read observed user-friction notes
  /memories/notes/<name>.md        — current-session working notes (read/write)
"""


def _format_listing(entries: list[FSEntry], header: str) -> str:
    if not entries:
        return f"{header}\n  (empty)"
    lines = [header]
    width = max(len(e.path) for e in entries) + 2
    for e in entries:
        marker = "[D]" if e.is_directory else "   "
        lines.append(f"  {marker} {e.path:<{width}}{e.summary}")
    return "\n".join(lines)


class _MemoryFSToolBase(Tool):
    """Shared init for the memory_fs tools.

    backend  — persistent tiers (lessons/core/digests/sessions/friction)
    notes    — NotesBackend over session_notes (the /memories/notes/ tier)
    approval_callback — invoked for /memories/core/ writes (path-based gate)
    """

    def __init__(
        self,
        backend: MemoryFSBackend,
        notes: NotesBackend,
        approval_callback: Optional[ApprovalCallback] = None,
    ):
        self.backend = backend
        self.notes = notes
        self._approval_callback = approval_callback

    async def _check_approval(self, tool_name: str, parsed_path, args: dict) -> Optional[str]:
        """Return None if approved/not-required, or an error string if denied."""
        if not parsed_path.tier:
            return None
        operation = args.get("_operation", "edit")
        if not requires_approval(parsed_path.tier, operation):
            return None
        if self._approval_callback is None:
            return (
                f"Write to {parsed_path.raw} requires user approval, but no "
                "approval callback is registered (headless mode)."
            )
        approved = await self._approval_callback(tool_name, args)
        if not approved:
            return f"User denied {operation} on {parsed_path.raw}"
        return None


class MemoryViewTool(_MemoryFSToolBase):
    """View a memory file (returns content) or directory (returns listing)."""

    read_only = True

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_view",
            description=(
                "Browse the agent's memory filesystem. Returns file content for "
                "leaf paths and a directory listing for tier/project paths.\n\n"
                + _TIER_OVERVIEW
                + "\nUse memory_view to deliberately browse a tier (e.g., 'show me "
                "all lessons for this project'). Use memory search for semantic "
                "lookup ('what do I know about X?'). They complement each other."
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type=ToolParameterType.STRING,
                    description=(
                        "/memories/... path. Use /memories alone to list tiers."
                    ),
                ),
            ],
        )

    async def execute(self, path: str = "", **kwargs) -> str:
        try:
            parsed = parse(path)
        except PathError as e:
            return f"Error: invalid path: {e}"

        try:
            if parsed.is_root:
                entries = await self.backend.list_root()
                return _format_listing(entries, "/memories/")

            if parsed.tier == Tier.NOTES:
                if parsed.is_directory:
                    return _format_listing(self.notes.list(), "/memories/notes/")
                return self.notes.read(parsed.slug)

            if parsed.is_directory:
                entries = await self.backend.list_directory(parsed)
                return _format_listing(entries, f"{parsed.raw.rstrip('/')}/")

            return await self.backend.read(parsed)
        except FSError as e:
            return str(e)


class MemoryCreateTool(_MemoryFSToolBase):
    """Create a core memory (approval) or a session note (free)."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_create",
            description=(
                "Create a new memory file.\n\n"
                "Writable tiers:\n"
                "  /memories/core/             — new core memory (REQUIRES APPROVAL)\n"
                "  /memories/notes/<name>.md   — current-session working note (free)\n\n"
                "Notes are for transient working state during a task (your plan, "
                "constraints, progress) — they persist across context compaction "
                "and are the same store as the save_note tool. Core memories are "
                "for durable facts about the user.\n\n"
                "Lessons, digests, sessions, and friction are read-only and cannot "
                "be created here. For core, you may pass /memories/core/ and a "
                "filename will be generated."
            ),
            parameters=[
                ToolParameter(
                    name="path",
                    type=ToolParameterType.STRING,
                    description="Target /memories/core/ or /memories/notes/<name>.md path",
                ),
                ToolParameter(
                    name="content",
                    type=ToolParameterType.STRING,
                    description="The file content (must be non-empty)",
                ),
            ],
        )

    async def execute(self, path: str = "", content: str = "", **kwargs) -> str:
        try:
            parsed = parse(path)
        except PathError as e:
            return f"Error: invalid path: {e}"

        # Notes — free write, no DB-vector concerns.
        if parsed.tier == Tier.NOTES:
            if parsed.is_directory or parsed.slug is None:
                return "Note creates require a name: /memories/notes/<name>.md"
            try:
                await self.notes.write(parsed.slug, content)
                return f"Created /memories/notes/{parsed.slug}.md"
            except FSError as e:
                return str(e)

        # Core — approval-gated.
        if parsed.tier == Tier.CORE:
            denial = await self._check_approval(
                "memory_create", parsed,
                {"path": path, "content": content[:200], "_operation": "create"},
            )
            if denial:
                return denial
            try:
                canonical = await self.backend.create(parsed, content)
                return f"Created {canonical.raw}"
            except FSError as e:
                return str(e)

        # Everything else is read-only.
        try:
            await self.backend.create(parsed, content)  # raises FSError with guidance
        except FSError as e:
            return str(e)
        return f"Cannot create at {path}"


class MemoryStrReplaceTool(_MemoryFSToolBase):
    """Replace exact text within a writable memory file (single-match required)."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_str_replace",
            description=(
                "Replace exact text in a memory file. old_text must appear "
                "EXACTLY ONCE — include surrounding context to disambiguate.\n\n"
                "Writable tiers:\n"
                "  /memories/core/<file>.md   (REQUIRES APPROVAL)\n"
                "  /memories/notes/<name>.md  (free)\n\n"
                "Lessons, digests, sessions, and friction are read-only."
            ),
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="/memories/... path to the file"),
                ToolParameter(name="old_text", type=ToolParameterType.STRING,
                              description="Exact text to replace (must match exactly once)"),
                ToolParameter(name="new_text", type=ToolParameterType.STRING,
                              description="Replacement text"),
            ],
        )

    async def execute(
        self, path: str = "", old_text: str = "", new_text: str = "", **kwargs
    ) -> str:
        try:
            parsed = parse(path)
        except PathError as e:
            return f"Error: invalid path: {e}"

        if parsed.is_root or parsed.is_directory:
            return f"Cannot edit a directory: {path}"

        if parsed.tier == Tier.NOTES:
            try:
                count = await self.notes.replace_text(parsed.slug, old_text, new_text)
                return f"Replaced {count} occurrence in {path}"
            except FSError as e:
                return str(e)

        if parsed.tier == Tier.CORE:
            denial = await self._check_approval(
                "memory_str_replace", parsed,
                {"path": path, "old_text": old_text[:200],
                 "new_text": new_text[:200], "_operation": "edit"},
            )
            if denial:
                return denial
            try:
                count = await self.backend.replace_text(parsed, old_text, new_text)
                return f"Replaced {count} occurrence in {path}"
            except FSError as e:
                return str(e)

        return (
            f"Cannot edit /memories/{parsed.tier.value}/ — read-only tier. "
            "Only core memories and notes are editable."
        )


class MemoryDeleteTool(_MemoryFSToolBase):
    """Delete a writable memory file (core deactivated; notes removed)."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="memory_delete",
            description=(
                "Delete a memory file. Core memories are soft-deleted "
                "(deactivated). Notes are removed from the session note store.\n\n"
                "Writable tiers:\n"
                "  /memories/core/<file>.md   (REQUIRES APPROVAL)\n"
                "  /memories/notes/<name>.md  (free)\n\n"
                "Lessons, digests, sessions, and friction cannot be deleted."
            ),
            parameters=[
                ToolParameter(name="path", type=ToolParameterType.STRING,
                              description="/memories/... path of the file to delete"),
            ],
        )

    async def execute(self, path: str = "", **kwargs) -> str:
        try:
            parsed = parse(path)
        except PathError as e:
            return f"Error: invalid path: {e}"

        if parsed.is_root or parsed.is_directory:
            return f"Cannot delete a directory: {path}"

        if parsed.tier == Tier.NOTES:
            try:
                await self.notes.delete(parsed.slug)
                return f"Deleted {path}"
            except FSError as e:
                return str(e)

        if parsed.tier == Tier.CORE:
            denial = await self._check_approval(
                "memory_delete", parsed, {"path": path, "_operation": "delete"},
            )
            if denial:
                return denial
            try:
                await self.backend.delete(parsed)
                return f"Deleted {path}"
            except FSError as e:
                return str(e)

        return (
            f"Cannot delete /memories/{parsed.tier.value}/ — read-only tier. "
            "Only core memories and notes are deletable."
        )
