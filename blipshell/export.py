"""Data export functions for sessions, memories, core memories, and lessons."""

from datetime import datetime, timezone

from blipshell.memory.sqlite_store import SQLiteStore


async def export_sessions_json(
    sqlite: SQLiteStore,
    session_ids: list[int] | None = None,
) -> dict:
    """Export sessions with their memories as JSON."""
    if session_ids:
        sessions = []
        for sid in session_ids:
            s = await sqlite.get_session(sid)
            if s:
                sessions.append(s)
    else:
        sessions = await sqlite.list_sessions(limit=1000)

    result = []
    for s in sessions:
        memories = await sqlite.get_memories_by_session(s.id)
        result.append({
            "id": s.id,
            "title": s.title,
            "summary": s.summary,
            "project": s.project,
            "created_at": str(s.timestamp),
            "last_active": str(s.last_active),
            "message_count": s.message_count,
            "memories": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "summary": m.summary,
                    "rank": m.rank,
                    "importance": m.importance,
                    "timestamp": str(m.timestamp),
                    "tags": await sqlite.get_memory_tags(m.id) if m.id else [],
                }
                for m in memories
            ],
        })

    return {
        "type": "blipshell_sessions_export",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(result),
        "sessions": result,
    }


async def export_memories_json(
    sqlite: SQLiteStore,
    include_archived: bool = False,
) -> dict:
    """Export all memories as JSON."""
    memories, total = await sqlite.get_memories_paginated(
        page=1, limit=10000, include_archived=include_archived,
    )

    result = []
    for m in memories:
        tags = await sqlite.get_memory_tags(m.id) if m.id else []
        result.append({
            "id": m.id,
            "session_id": m.session_id,
            "role": m.role,
            "content": m.content,
            "summary": m.summary,
            "rank": m.rank,
            "importance": m.importance,
            "memory_type": m.memory_type.value,
            "is_archived": m.is_archived,
            "timestamp": str(m.timestamp),
            "tags": tags,
        })

    return {
        "type": "blipshell_memories_export",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(result),
        "include_archived": include_archived,
        "memories": result,
    }


async def export_core_memories_json(sqlite: SQLiteStore) -> dict:
    """Export core memories as JSON."""
    core_memories = await sqlite.get_active_core_memories()
    return {
        "type": "blipshell_core_memories_export",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(core_memories),
        "core_memories": [m.model_dump() for m in core_memories],
    }


async def export_lessons_json(sqlite: SQLiteStore) -> dict:
    """Export lessons as JSON."""
    lessons = await sqlite.get_all_lessons()
    return {
        "type": "blipshell_lessons_export",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(lessons),
        "lessons": [l.model_dump() for l in lessons],
    }


async def export_all_json(sqlite: SQLiteStore) -> dict:
    """Export everything as a single JSON document."""
    sessions = await export_sessions_json(sqlite)
    memories = await export_memories_json(sqlite, include_archived=True)
    core_memories = await export_core_memories_json(sqlite)
    lessons = await export_lessons_json(sqlite)

    return {
        "type": "blipshell_full_export",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sessions": sessions["sessions"],
        "memories": memories["memories"],
        "core_memories": core_memories["core_memories"],
        "lessons": lessons["lessons"],
    }


async def export_all_markdown(sqlite: SQLiteStore) -> str:
    """Export everything in human-readable markdown format."""
    lines = ["# BlipShell Export", ""]
    lines.append(f"Exported: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # Core memories
    core_memories = await sqlite.get_active_core_memories()
    lines.append("## Core Memories")
    lines.append("")
    for cm in core_memories:
        lines.append(f"- **{cm.category}** (importance: {cm.importance:.2f}): {cm.content}")
    lines.append("")

    # Lessons
    lessons = await sqlite.get_all_lessons()
    lines.append("## Lessons")
    lines.append("")
    for l in lessons:
        lines.append(f"- (rank: {l.rank}, importance: {l.importance:.2f}): {l.content[:200]}")
    lines.append("")

    # Sessions
    sessions = await sqlite.list_sessions(limit=100)
    lines.append("## Sessions")
    lines.append("")
    for s in sessions:
        lines.append(f"### Session #{s.id}: {s.title or 'Untitled'}")
        lines.append(f"Project: {s.project or 'None'} | Messages: {s.message_count}")
        if s.summary:
            lines.append(f"\n{s.summary}\n")
        lines.append("")

    return "\n".join(lines)
