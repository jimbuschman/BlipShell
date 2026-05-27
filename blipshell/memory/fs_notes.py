"""Notes tier backend — a filesystem view over the existing session_notes store.

The /memories/notes/<name>.md tier maps onto BlipShell's session notes
(sessions.metadata_json), the same key/value store used by the save_note /
get_notes / delete_note tools. This gives the LLM two interchangeable ways to
reach the same data: the dedicated note tools and the unified memory filesystem.

The backend shares the agent's in-memory notes dict (agent._session_notes) and
persists writes through sqlite.save_session_notes — identical to how the note
tools behave — so the two interfaces never drift apart.
"""

from __future__ import annotations

from typing import Callable, Optional

from blipshell.memory.fs_backend import FSEntry, FSError


class NotesBackend:
    """Path-aware adapter over the shared session_notes dict.

    Args:
        sqlite: SQLiteStore (for save_session_notes persistence)
        notes: the shared {name: content} dict (agent._session_notes)
        session_id_provider: returns the current session id (notes are
            current-session-scoped; the path tier carries no session id)
        max_notes: cap on note count, mirroring NotesConfig.max_notes
    """

    def __init__(
        self,
        sqlite,
        notes: dict[str, str],
        session_id_provider: Callable[[], Optional[int]],
        max_notes: int = 50,
    ):
        self.sqlite = sqlite
        self._notes = notes
        self._session_id_provider = session_id_provider
        self.max_notes = max_notes

    def _session_id(self) -> Optional[int]:
        try:
            return self._session_id_provider()
        except Exception:  # pragma: no cover — defensive
            return None

    async def _persist(self) -> None:
        session_id = self._session_id()
        if session_id is None:
            raise FSError("No active session — cannot persist notes")
        await self.sqlite.save_session_notes(session_id, self._notes)

    # ---- queries ----

    def list(self) -> list[FSEntry]:
        entries = []
        for name, content in sorted(self._notes.items()):
            preview = content.splitlines()[0][:60] if content else "(empty)"
            entries.append(FSEntry(
                path=f"/memories/notes/{name}.md",
                is_directory=False,
                summary=f"{len(content)} chars — {preview}",
            ))
        return entries

    def read(self, name: str) -> str:
        if name not in self._notes:
            available = ", ".join(sorted(self._notes)) or "none"
            raise FSError(
                f"No note '{name}'. Available: {available}"
            )
        return self._notes[name]

    # ---- writes (persisted to session_notes) ----

    async def write(self, name: str, content: str) -> None:
        if not content or not content.strip():
            raise FSError("Cannot create empty note — content is required")
        is_new = name not in self._notes
        if is_new and len(self._notes) >= self.max_notes:
            raise FSError(
                f"Max notes ({self.max_notes}) reached. Delete or update "
                "an existing note first."
            )
        self._notes[name] = content.strip()
        await self._persist()

    async def replace_text(self, name: str, old: str, new: str) -> int:
        content = self.read(name)
        count = content.count(old)
        if count == 0:
            raise FSError(f"Text not found in /memories/notes/{name}.md")
        if count > 1:
            raise FSError(
                f"Text matches {count} times in /memories/notes/{name}.md. "
                "Include more surrounding context to make it unique."
            )
        self._notes[name] = content.replace(old, new, 1)
        await self._persist()
        return 1

    async def delete(self, name: str) -> None:
        if name not in self._notes:
            available = ", ".join(sorted(self._notes)) or "none"
            raise FSError(f"No note '{name}'. Available: {available}")
        del self._notes[name]
        await self._persist()
