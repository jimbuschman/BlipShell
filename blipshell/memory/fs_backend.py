"""Backend for the memory filesystem — translates MemoryPath ops to SQLiteStore.

Wraps lesson/core/digest/session/friction CRUD with path-aware semantics.
Returns plain strings for file reads and FSEntry lists for directory views.

Scratch tier is handled separately (see fs_scratch.py).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from blipshell.memory.fs_paths import (
    MemoryPath,
    Tier,
    build_filename,
)

logger = logging.getLogger(__name__)


class FSError(Exception):
    """Operation failed on the memory filesystem. Message goes to the LLM."""


@dataclass
class FSEntry:
    """One row in a directory listing.

    path:        canonical /memories/... path (or tier name for root listing)
    is_directory: True for tier roots and project subdirs
    item_id:     backing row id (None for directories and digests)
    summary:     short preview / metadata line for the LLM
    """

    path: str
    is_directory: bool
    item_id: Optional[int] = None
    summary: str = ""


class MemoryFSBackend:
    """Path-aware wrapper around SQLiteStore for the persistent memory tiers.

    `vectors` is the VectorStore — required to keep the embedding index in sync
    when core memories are created/edited/deleted. Without it, writes would
    drift the SQLite and vector stores apart (the exact bug this tier guards
    against). May be None in read-only/test contexts.
    """

    def __init__(self, sqlite_store, vectors=None):
        self.sqlite = sqlite_store
        self.vectors = vectors

    # -------------------------------------------------------------------
    # Listings
    # -------------------------------------------------------------------

    async def list_root(self) -> list[FSEntry]:
        """List the top-level tiers under /memories/."""
        return [
            FSEntry(path="/memories/lessons/",  is_directory=True,
                    summary="project-scoped lessons (read-only, pipeline-derived)"),
            FSEntry(path="/memories/core/",     is_directory=True,
                    summary="persistent core memories (write requires approval)"),
            FSEntry(path="/memories/digests/",  is_directory=True,
                    summary="auto-maintained project digests (read-only)"),
            FSEntry(path="/memories/sessions/", is_directory=True,
                    summary="historical session summaries (read-only)"),
            FSEntry(path="/memories/friction/", is_directory=True,
                    summary="observed user-friction notes (read-only)"),
            FSEntry(path="/memories/notes/",    is_directory=True,
                    summary="current-session working notes (read/write)"),
        ]

    async def list_directory(self, path: MemoryPath) -> list[FSEntry]:
        """List contents of a tier or project directory.

        path.is_directory must be True. Root paths handled via list_root().
        Scratch tier is handled elsewhere (see fs_scratch.py).
        """
        if not path.is_directory:
            raise FSError(f"{path.raw} is not a directory")
        if path.tier == Tier.LESSONS:
            return await self._list_lessons(path)
        if path.tier == Tier.CORE:
            return await self._list_core()
        if path.tier == Tier.DIGESTS:
            return await self._list_digests()
        if path.tier == Tier.SESSIONS:
            return await self._list_sessions()
        if path.tier == Tier.FRICTION:
            return await self._list_friction()
        if path.tier == Tier.SCRATCH:
            raise FSError("Scratch listings are handled by the scratch backend")
        raise FSError(f"Cannot list {path.tier.value}")  # pragma: no cover

    async def _list_lessons(self, path: MemoryPath) -> list[FSEntry]:
        if path.project is None:
            # /memories/lessons/ — list known projects with lesson counts.
            all_lessons = await self.sqlite.get_all_lessons()
            projects: dict[str, int] = {}
            for lesson in all_lessons:
                key = lesson.project or "_global"
                projects[key] = projects.get(key, 0) + 1
            return [
                FSEntry(
                    path=f"/memories/lessons/{name}/",
                    is_directory=True,
                    summary=f"{count} lesson{'s' if count != 1 else ''}",
                )
                for name, count in sorted(projects.items())
            ]
        # /memories/lessons/<project>/ — list lessons in that project.
        if path.project == "_global":
            all_lessons = await self.sqlite.get_all_lessons()
            lessons = [l for l in all_lessons if not l.project]
        else:
            lessons = await self.sqlite.get_lessons_by_project(path.project)
        return [_lesson_to_entry(l, path.project) for l in sorted(
            lessons, key=lambda l: (-l.importance, -(l.id or 0))
        )]

    async def _list_core(self) -> list[FSEntry]:
        cores = await self.sqlite.get_active_core_memories()
        return [_core_to_entry(c) for c in sorted(
            cores, key=lambda c: (-c.importance, -(c.id or 0))
        )]

    async def _list_digests(self) -> list[FSEntry]:
        # Walk projects, include only those whose metadata_json has a digest.
        entries: list[FSEntry] = []
        # No bulk "list all projects" exists — use list_sessions to discover names.
        # The projects table is small; iterate via the sessions table to surface
        # project names actually in use.
        sessions = await self.sqlite.list_sessions(limit=500)
        seen: set[str] = set()
        for s in sessions:
            if not s.project or s.project in seen:
                continue
            seen.add(s.project)
            project_data = await self.sqlite.get_project(s.project)
            if not project_data:
                continue
            meta = json.loads(project_data.get("metadata_json") or "{}")
            digest = meta.get("digest")
            if not digest:
                continue
            preview = digest.strip().splitlines()[0][:80] if digest else ""
            entries.append(FSEntry(
                path=f"/memories/digests/{s.project}.md",
                is_directory=False,
                summary=preview or "(empty digest)",
            ))
        entries.sort(key=lambda e: e.path)
        return entries

    async def _list_sessions(self) -> list[FSEntry]:
        sessions = await self.sqlite.list_sessions(limit=50)
        entries: list[FSEntry] = []
        for s in sessions:
            if s.id is None:
                continue
            title = s.title or "untitled"
            filename = build_filename(s.id, title)
            preview = (s.summary or "")[:80]
            entries.append(FSEntry(
                path=f"/memories/sessions/{filename}",
                is_directory=False,
                item_id=s.id,
                summary=preview or "(no summary)",
            ))
        return entries

    async def _list_friction(self) -> list[FSEntry]:
        rows = await self.sqlite.get_friction_entries(limit=50)
        entries: list[FSEntry] = []
        for row in rows:
            row_id = row.get("id")
            if row_id is None:
                continue
            category = row.get("category", "unknown").lower()
            filename = build_filename(row_id, category)
            preview = (row.get("description") or "")[:80]
            entries.append(FSEntry(
                path=f"/memories/friction/{filename}",
                is_directory=False,
                item_id=row_id,
                summary=preview or "(no description)",
            ))
        return entries

    # -------------------------------------------------------------------
    # Reads
    # -------------------------------------------------------------------

    async def read(self, path: MemoryPath) -> str:
        """Read the content of a file path."""
        if path.is_root or path.is_directory:
            raise FSError(f"{path.raw} is a directory, not a file")
        if path.tier == Tier.LESSONS:
            lesson = await self._resolve_lesson(path)
            return lesson.content
        if path.tier == Tier.CORE:
            core = await self._resolve_core(path)
            return core.content
        if path.tier == Tier.DIGESTS:
            return await self._read_digest(path)
        if path.tier == Tier.SESSIONS:
            session = await self._resolve_session(path)
            return _format_session_for_read(session)
        if path.tier == Tier.FRICTION:
            row = await self._resolve_friction(path)
            return _format_friction_for_read(row)
        raise FSError(f"Cannot read {path.tier.value}")  # pragma: no cover

    async def _read_digest(self, path: MemoryPath) -> str:
        project_data = await self.sqlite.get_project(path.project)
        if not project_data:
            raise FSError(f"No project named {path.project!r}")
        meta = json.loads(project_data.get("metadata_json") or "{}")
        digest = meta.get("digest")
        if not digest:
            raise FSError(
                f"No digest exists yet for {path.project}. "
                "Run /project digest rebuild to generate one."
            )
        return digest

    # -------------------------------------------------------------------
    # Writes (create, replace, delete)
    # -------------------------------------------------------------------

    async def create(self, path: MemoryPath, content: str) -> MemoryPath:
        """Create a new entry. Returns the canonical path with assigned ID.

        Only core memories are creatable here (lessons are read-only;
        notes go through the NotesBackend). The LLM may pass a tier directory
        (filename generated from new ID + slug) or a full file path.
        """
        if not content or not content.strip():
            raise FSError("Cannot create empty entry — content is required")
        if path.tier == Tier.CORE:
            return await self._create_core(path, content)
        if path.tier == Tier.LESSONS:
            raise FSError(
                "Lessons are read-only — they are extracted and scored "
                "automatically. To record durable guidance, use a core memory "
                "(/memories/core/) or a session note (/memories/notes/)."
            )
        raise FSError(
            f"Cannot create entries in /memories/{path.tier.value}/ — "
            "this tier is read-only or handled by another backend."
        )

    async def _create_core(self, path: MemoryPath, content: str) -> MemoryPath:
        from blipshell.models.memory import CoreMemory  # local import

        core = CoreMemory(content=content.strip())
        new_id = await self.sqlite.create_core_memory(core)
        # Keep the embedding index in sync — without this, the new core memory
        # is invisible to semantic search until the nightly vector backfill.
        if self.vectors is not None:
            try:
                self.vectors.add_core_memory(new_id, content.strip())
            except Exception as e:
                logger.warning("Core memory %s embed failed: %s", new_id, e)
        filename = build_filename(new_id, path.slug or content[:60])
        return MemoryPath(
            raw=f"/memories/core/{filename}",
            tier=Tier.CORE,
            filename=filename,
            file_id=new_id,
            slug=filename[len(str(new_id)) + 1:-3] if "-" in filename else None,
        )

    async def replace_text(self, path: MemoryPath, old: str, new: str) -> int:
        """Replace `old` with `new` in the file at path. Returns occurrence count.

        Raises FSError if old text not found (or appears multiple times — to
        prevent ambiguous edits, the call must be made specific enough to match
        exactly once).
        """
        if path.tier == Tier.CORE:
            core = await self._resolve_core(path)
            count = core.content.count(old)
            if count == 0:
                raise FSError(f"Text not found in {path.raw}")
            if count > 1:
                raise FSError(
                    f"Text matches {count} times in {path.raw}. "
                    "Include more surrounding context to make it unique."
                )
            new_content = core.content.replace(old, new, 1)
            await self.sqlite.update_core_memory(core.id, content=new_content)
            # Re-embed so the vector index reflects the edited content.
            if self.vectors is not None:
                try:
                    self.vectors.add_core_memory(core.id, new_content)
                except Exception as e:
                    logger.warning("Core memory %s re-embed failed: %s", core.id, e)
            return 1
        raise FSError(
            f"Cannot edit /memories/{path.tier.value}/ — read-only tier "
            "(lessons, digests, sessions, friction are not editable here)."
        )

    async def delete(self, path: MemoryPath) -> None:
        """Delete an entry. Core memories are deactivated (soft delete)."""
        if path.tier == Tier.CORE:
            core = await self._resolve_core(path)
            await self.sqlite.deactivate_core_memory(core.id)
            # Remove from the vector index so a deactivated memory stops
            # surfacing in semantic search (no orphaned embedding).
            if self.vectors is not None:
                try:
                    self.vectors.delete_core_memory(core.id)
                except Exception as e:
                    logger.warning("Core memory %s vector delete failed: %s",
                                   core.id, e)
            return
        raise FSError(
            f"Cannot delete /memories/{path.tier.value}/ — read-only tier "
            "(lessons, digests, sessions, friction cannot be deleted here)."
        )

    # -------------------------------------------------------------------
    # Resolvers — translate path → DB row, supporting id or slug addressing.
    # -------------------------------------------------------------------

    async def _resolve_lesson(self, path: MemoryPath):
        if path.project is None or path.filename is None:
            raise FSError(f"Lesson path missing project or filename: {path.raw}")
        if path.file_id is not None:
            lesson = await self.sqlite.get_lesson(path.file_id)
            if not lesson:
                raise FSError(f"No lesson with id {path.file_id}")
            return lesson
        # Slug-only lookup — scan project lessons.
        if path.project == "_global":
            all_lessons = await self.sqlite.get_all_lessons()
            candidates = [l for l in all_lessons if not l.project]
        else:
            candidates = await self.sqlite.get_lessons_by_project(path.project)
        from blipshell.memory.fs_paths import slugify
        matches = [l for l in candidates if slugify(l.content) == path.slug]
        if not matches:
            raise FSError(
                f"No lesson matching slug {path.slug!r} in project "
                f"{path.project!r}. Use a path with the id prefix (e.g., "
                f"/memories/lessons/{path.project}/<id>-<slug>.md)."
            )
        if len(matches) > 1:
            raise FSError(
                f"Multiple lessons matched slug {path.slug!r} in {path.project}. "
                "Address by id explicitly."
            )
        return matches[0]

    async def _resolve_core(self, path: MemoryPath):
        if path.filename is None:
            raise FSError(f"Core path missing filename: {path.raw}")
        if path.file_id is not None:
            core = await self.sqlite.get_core_memory(path.file_id)
            if not core or not getattr(core, "id", None):
                raise FSError(f"No core memory with id {path.file_id}")
            return core
        from blipshell.memory.fs_paths import slugify
        cores = await self.sqlite.get_active_core_memories()
        matches = [c for c in cores if slugify(c.content) == path.slug]
        if not matches:
            raise FSError(
                f"No core memory matching slug {path.slug!r}. "
                "Use a path with the id prefix."
            )
        if len(matches) > 1:
            raise FSError(
                f"Multiple core memories matched slug {path.slug!r}. "
                "Address by id explicitly."
            )
        return matches[0]

    async def _resolve_session(self, path: MemoryPath):
        if path.file_id is None:
            raise FSError(
                "Sessions must be addressed by id: "
                "/memories/sessions/<id>-<slug>.md"
            )
        session = await self.sqlite.get_session(path.file_id)
        if not session:
            raise FSError(f"No session with id {path.file_id}")
        return session

    async def _resolve_friction(self, path: MemoryPath):
        if path.file_id is None:
            raise FSError(
                "Friction entries must be addressed by id: "
                "/memories/friction/<id>-<slug>.md"
            )
        # No singular get_friction_entry — fetch a batch and filter.
        rows = await self.sqlite.get_friction_entries(limit=500)
        for row in rows:
            if row.get("id") == path.file_id:
                return row
        raise FSError(f"No friction entry with id {path.file_id}")


# -------------------------------------------------------------------
# Formatters and helpers
# -------------------------------------------------------------------


def _lesson_to_entry(lesson, project: Optional[str]) -> FSEntry:
    proj = project or "_global"
    filename = build_filename(lesson.id or 0, lesson.content[:60])
    preview = lesson.content.replace("\n", " ")[:80]
    summary = (
        f"rank {lesson.rank}, importance {lesson.importance:.2f} — {preview}"
    )
    return FSEntry(
        path=f"/memories/lessons/{proj}/{filename}",
        is_directory=False,
        item_id=lesson.id,
        summary=summary,
    )


def _core_to_entry(core) -> FSEntry:
    filename = build_filename(core.id or 0, core.content[:60])
    preview = core.content.replace("\n", " ")[:80]
    summary = (
        f"category {core.category}, importance {core.importance:.2f} — {preview}"
    )
    return FSEntry(
        path=f"/memories/core/{filename}",
        is_directory=False,
        item_id=core.id,
        summary=summary,
    )


def _format_session_for_read(session) -> str:
    parts = []
    if session.title:
        parts.append(f"# {session.title}")
    parts.append(f"Session #{session.id}  ({session.timestamp:%Y-%m-%d %H:%M})")
    if session.project:
        parts.append(f"Project: {session.project}")
    parts.append(f"Messages: {session.message_count}")
    parts.append("")
    parts.append(session.summary or "(no summary)")
    return "\n".join(parts)


def _format_friction_for_read(row: dict) -> str:
    parts = [
        f"# Friction #{row.get('id')}",
        f"Category: {row.get('category', 'unknown')}",
        f"Source:   {row.get('source', 'unknown')}",
        f"Created:  {row.get('created_at', '?')}",
        "",
        row.get("description", "(no description)"),
    ]
    return "\n".join(parts)
