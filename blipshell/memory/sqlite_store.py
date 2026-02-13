"""SQLite storage for structured data (port of MemoryDB.cs schema)."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from blipshell.models.memory import CoreMemory, Lesson, Memory, MemoryType
from blipshell.models.session import Session, SessionMessage

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    summary TEXT,
    project TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_archived BOOLEAN DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    rank INTEGER DEFAULT 0,
    importance REAL DEFAULT 0.0,
    memory_type TEXT DEFAULT 'conversation',
    is_archived BOOLEAN DEFAULT 0,
    metadata_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS core_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    importance REAL DEFAULT 0.5,
    source_session_id INTEGER,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (source_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    summary TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    rank INTEGER DEFAULT 3,
    importance REAL DEFAULT 0.5,
    source_session_id INTEGER,
    added_by TEXT DEFAULT 'system',
    FOREIGN KEY (source_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT DEFAULT 'topic',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, category)
);

CREATE TABLE IF NOT EXISTS memory_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    UNIQUE(memory_id, tag_id)
);

CREATE TABLE IF NOT EXISTS core_memory_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    core_memory_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (core_memory_id) REFERENCES core_memories(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    UNIQUE(core_memory_id, tag_id)
);

CREATE TABLE IF NOT EXISTS lesson_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE,
    UNIQUE(lesson_id, tag_id)
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_rank ON memories(rank);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_tags_memory ON memory_tags(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
"""


class SQLiteStore:
    """Async SQLite storage for structured data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Open connection and create schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # --- Sessions ---

    async def create_session(self, title: str = "New Session", project: Optional[str] = None) -> int:
        """Create a new session and return its ID."""
        cursor = await self._db.execute(
            "INSERT INTO sessions (title, project, created_at, last_active) VALUES (?, ?, ?, ?)",
            (title, project, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_session(self, session_id: int, **kwargs):
        """Update session fields."""
        allowed = {"title", "summary", "project", "last_active", "is_archived", "message_count"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [session_id]
        await self._db.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
        await self._db.commit()

    async def get_session(self, session_id: int) -> Optional[Session]:
        """Get a session by ID."""
        cursor = await self._db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return Session(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
            project=row["project"],
            timestamp=row["created_at"],
            last_active=row["last_active"],
            is_archived=bool(row["is_archived"]),
            message_count=row["message_count"],
            metadata_json=row["metadata_json"],
        )

    async def get_latest_session(self) -> Optional[Session]:
        """Get the most recent session."""
        cursor = await self._db.execute(
            "SELECT * FROM sessions ORDER BY last_active DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return Session(
            id=row["id"],
            title=row["title"],
            summary=row["summary"],
            project=row["project"],
            timestamp=row["created_at"],
            last_active=row["last_active"],
            is_archived=bool(row["is_archived"]),
            message_count=row["message_count"],
        )

    async def list_sessions(self, limit: int = 50, project: Optional[str] = None) -> list[Session]:
        """List sessions, optionally filtered by project."""
        if project:
            cursor = await self._db.execute(
                "SELECT * FROM sessions WHERE project = ? ORDER BY last_active DESC LIMIT ?",
                (project, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM sessions ORDER BY last_active DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [
            Session(
                id=r["id"],
                title=r["title"],
                summary=r["summary"],
                project=r["project"],
                timestamp=r["created_at"],
                last_active=r["last_active"],
                is_archived=bool(r["is_archived"]),
                message_count=r["message_count"],
            )
            for r in rows
        ]

    # --- Memories ---

    async def create_memory(self, memory: Memory) -> int:
        """Insert a memory and return its ID."""
        cursor = await self._db.execute(
            """INSERT INTO memories (session_id, role, content, summary, timestamp, rank,
               importance, memory_type, is_archived, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.session_id,
                memory.role,
                memory.content,
                memory.summary,
                memory.timestamp.isoformat(),
                memory.rank,
                memory.importance,
                memory.memory_type.value,
                memory.is_archived,
                memory.metadata_json,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_memory(self, memory_id: int, **kwargs):
        """Update memory fields."""
        allowed = {"summary", "rank", "importance", "is_archived", "metadata_json"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [memory_id]
        await self._db.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        await self._db.commit()

    async def get_memories_by_session(self, session_id: int) -> list[Memory]:
        """Get all memories for a session."""
        cursor = await self._db.execute(
            "SELECT * FROM memories WHERE session_id = ? ORDER BY timestamp", (session_id,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Get a single memory by ID."""
        cursor = await self._db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_memory(row)

    def _row_to_memory(self, row) -> Memory:
        return Memory(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            summary=row["summary"],
            timestamp=row["timestamp"],
            rank=row["rank"] or 0,
            importance=row["importance"] or 0.0,
            memory_type=MemoryType(row["memory_type"]),
            is_archived=bool(row["is_archived"]),
            metadata_json=row["metadata_json"],
        )

    # --- Core Memories ---

    async def create_core_memory(self, core_memory: CoreMemory) -> int:
        """Insert a core memory and return its ID."""
        cursor = await self._db.execute(
            """INSERT INTO core_memories (content, category, timestamp, importance, source_session_id)
               VALUES (?, ?, ?, ?, ?)""",
            (
                core_memory.content,
                core_memory.category,
                core_memory.timestamp.isoformat(),
                core_memory.importance,
                core_memory.source_session_id,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_active_core_memories(self) -> list[CoreMemory]:
        """Get all active core memories."""
        cursor = await self._db.execute(
            "SELECT * FROM core_memories WHERE is_active = 1 ORDER BY importance DESC"
        )
        rows = await cursor.fetchall()
        return [
            CoreMemory(
                id=r["id"],
                content=r["content"],
                category=r["category"],
                timestamp=r["timestamp"],
                importance=r["importance"],
                source_session_id=r["source_session_id"],
            )
            for r in rows
        ]

    async def deactivate_core_memory(self, core_memory_id: int):
        """Deactivate a core memory."""
        await self._db.execute(
            "UPDATE core_memories SET is_active = 0 WHERE id = ?", (core_memory_id,)
        )
        await self._db.commit()

    # --- Lessons ---

    async def create_lesson(self, lesson: Lesson) -> int:
        """Insert a lesson and return its ID."""
        cursor = await self._db.execute(
            """INSERT INTO lessons (content, summary, timestamp, rank, importance,
               source_session_id, added_by)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                lesson.content,
                lesson.summary,
                lesson.timestamp.isoformat(),
                lesson.rank,
                lesson.importance,
                lesson.source_session_id,
                "system",
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_all_lessons(self) -> list[Lesson]:
        """Get all lessons."""
        cursor = await self._db.execute("SELECT * FROM lessons ORDER BY timestamp DESC")
        rows = await cursor.fetchall()
        return [
            Lesson(
                id=r["id"],
                content=r["content"],
                summary=r["summary"],
                timestamp=r["timestamp"],
                rank=r["rank"],
                importance=r["importance"],
                source_session_id=r["source_session_id"],
            )
            for r in rows
        ]

    # --- Tags ---

    async def create_or_get_tag(self, name: str, category: str = "topic") -> int:
        """Get existing tag ID or create a new one."""
        cursor = await self._db.execute(
            "SELECT id FROM tags WHERE name = ? AND category = ?", (name, category)
        )
        row = await cursor.fetchone()
        if row:
            return row["id"]
        cursor = await self._db.execute(
            "INSERT INTO tags (name, category) VALUES (?, ?)", (name, category)
        )
        await self._db.commit()
        return cursor.lastrowid

    async def tag_memory(self, memory_id: int, tag_names: list[str]):
        """Associate tags with a memory."""
        for tag_name in tag_names:
            tag_id = await self.create_or_get_tag(tag_name)
            await self._db.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
                (memory_id, tag_id),
            )
        await self._db.commit()

    async def tag_core_memory(self, core_memory_id: int, tag_names: list[str]):
        """Associate tags with a core memory."""
        for tag_name in tag_names:
            tag_id = await self.create_or_get_tag(tag_name)
            await self._db.execute(
                "INSERT OR IGNORE INTO core_memory_tags (core_memory_id, tag_id) VALUES (?, ?)",
                (core_memory_id, tag_id),
            )
        await self._db.commit()

    async def tag_lesson(self, lesson_id: int, tag_names: list[str]):
        """Associate tags with a lesson."""
        for tag_name in tag_names:
            tag_id = await self.create_or_get_tag(tag_name)
            await self._db.execute(
                "INSERT OR IGNORE INTO lesson_tags (lesson_id, tag_id) VALUES (?, ?)",
                (lesson_id, tag_id),
            )
        await self._db.commit()

    async def get_memory_tags(self, memory_id: int) -> list[str]:
        """Get tag names for a memory."""
        cursor = await self._db.execute(
            """SELECT t.name FROM tags t
               INNER JOIN memory_tags mt ON mt.tag_id = t.id
               WHERE mt.memory_id = ?""",
            (memory_id,),
        )
        rows = await cursor.fetchall()
        return [r["name"] for r in rows]

    async def get_tag_count_for_memory(self, memory_id: int) -> int:
        """Get number of tags for a memory (used in importance calculation)."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) as cnt FROM memory_tags WHERE memory_id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        return row["cnt"]

    # --- Projects ---

    async def create_project(self, name: str, description: str = "") -> int:
        """Create a named project."""
        cursor = await self._db.execute(
            "INSERT OR IGNORE INTO projects (name, description) VALUES (?, ?)",
            (name, description),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def list_projects(self) -> list[dict]:
        """List all projects."""
        cursor = await self._db.execute(
            "SELECT * FROM projects ORDER BY last_active DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
