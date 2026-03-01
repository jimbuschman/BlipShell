"""SQLite storage for structured data (port of MemoryDB.cs schema)."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from blipshell.models.memory import CoreMemory, Lesson, Memory, MemoryType
from blipshell.models.session import Session, SessionMessage
from blipshell.models.task import (
    BackgroundTask,
    BackgroundTaskStatus,
    PlanStatus,
    StepStatus,
    TaskPlan,
    TaskStep,
)

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
    root_path TEXT,
    git_url TEXT,
    language TEXT,
    settings_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS task_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    user_request TEXT NOT NULL,
    status TEXT DEFAULT 'planning',
    result_summary TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS task_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL,
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    tool_hint TEXT,
    output_result TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES task_plans(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS background_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    plan_id INTEGER,
    title TEXT NOT NULL,
    task_type TEXT DEFAULT 'custom',
    prompt TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    progress_pct REAL DEFAULT 0.0,
    progress_message TEXT DEFAULT '',
    result TEXT,
    error_message TEXT,
    target_endpoint TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (plan_id) REFERENCES task_plans(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_rank ON memories(rank);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memory_tags_memory ON memory_tags(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_task_plans_session ON task_plans(session_id);
CREATE INDEX IF NOT EXISTS idx_task_steps_plan ON task_steps(plan_id);
CREATE INDEX IF NOT EXISTS idx_background_tasks_session ON background_tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_background_tasks_status ON background_tasks(status);
CREATE INDEX IF NOT EXISTS idx_core_memories_active ON core_memories(is_active);
CREATE INDEX IF NOT EXISTS idx_tags_name_category ON tags(name, category);

CREATE TABLE IF NOT EXISTS discovered_tag_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL,
    pattern TEXT NOT NULL,
    discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    UNIQUE(tag_name, pattern)
);

CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT DEFAULT 'concept',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, entity_type)
);

CREATE TABLE IF NOT EXISTS entity_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    predicate TEXT NOT NULL,
    object_id INTEGER NOT NULL,
    source_memory_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject_id) REFERENCES entities(id),
    FOREIGN KEY (object_id) REFERENCES entities(id),
    FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE(subject_id, predicate, object_id)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL,
    memory_id INTEGER NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities(id),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE(entity_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_mentions_memory ON entity_mentions(memory_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_subject ON entity_relationships(subject_id);
CREATE INDEX IF NOT EXISTS idx_entity_relationships_object ON entity_relationships(object_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE TABLE IF NOT EXISTS turn_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    turn_number INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    data_json TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_turn_events_session ON turn_events(session_id);
CREATE INDEX IF NOT EXISTS idx_turn_events_session_turn ON turn_events(session_id, turn_number);

CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_processed BOOLEAN DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_session_messages_unprocessed
    ON session_messages(is_processed) WHERE is_processed = 0;

-- FTS5 full-text search on memory summaries
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    summary, content=memories, content_rowid=id
);

-- Keep FTS index in sync with memories table
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories
WHEN NEW.summary IS NOT NULL BEGIN
    INSERT INTO memories_fts(rowid, summary) VALUES (NEW.id, NEW.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF summary ON memories
WHEN NEW.summary IS NOT NULL BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, summary) VALUES('delete', OLD.id, OLD.summary);
    INSERT INTO memories_fts(rowid, summary) VALUES (NEW.id, NEW.summary);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories
WHEN OLD.summary IS NOT NULL BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, summary) VALUES('delete', OLD.id, OLD.summary);
END;
"""


class SQLiteStore:
    """Async SQLite storage for structured data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Open connection and create schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path, isolation_level=None)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute("PRAGMA journal_mode = WAL")
        await self._db.execute("PRAGMA busy_timeout = 60000")  # wait up to 60s for write lock
        await self._db.executescript(SCHEMA_SQL)
        # Schema migrations for existing databases
        for col_sql in (
            "ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN last_accessed DATETIME",
            "ALTER TABLE memories ADD COLUMN consolidated_at DATETIME",
            "ALTER TABLE memories ADD COLUMN entities_extracted_at DATETIME",
            "ALTER TABLE projects ADD COLUMN root_path TEXT",
            "ALTER TABLE projects ADD COLUMN git_url TEXT",
            "ALTER TABLE projects ADD COLUMN language TEXT",
            "ALTER TABLE projects ADD COLUMN settings_json TEXT",
            # Project-scoped lessons
            "ALTER TABLE lessons ADD COLUMN project TEXT",
            # Bi-temporal edge tracking (Feature 4)
            "ALTER TABLE entity_relationships ADD COLUMN valid_from DATETIME",
            "ALTER TABLE entity_relationships ADD COLUMN expired_at DATETIME",
            "ALTER TABLE entity_relationships ADD COLUMN expired_by INTEGER",
        ):
            try:
                await self._db.execute(col_sql)
            except Exception:
                pass  # column already exists
        # Entity aliases table (Feature 5: Entity Resolution)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS entity_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alias_name TEXT NOT NULL,
                canonical_entity_id INTEGER NOT NULL,
                merge_method TEXT DEFAULT 'exact',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (canonical_entity_id) REFERENCES entities(id),
                UNIQUE(alias_name)
            )
        """)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical "
            "ON entity_aliases(canonical_entity_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_relationships_valid "
            "ON entity_relationships(expired_at)"
        )
        # Backfill valid_from from created_at for existing relationships
        await self._db.execute(
            "UPDATE entity_relationships SET valid_from = created_at "
            "WHERE valid_from IS NULL"
        )
        # Backfill FTS5 index with existing summaries
        await self._db.execute(
            """INSERT OR IGNORE INTO memories_fts(rowid, summary)
               SELECT id, summary FROM memories WHERE summary IS NOT NULL"""
        )
        await self._db.commit()

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # --- Sessions ---

    async def create_session(self, title: str = "New Session", project: Optional[str] = None,
                             created_at: Optional[datetime] = None) -> int:
        """Create a new session and return its ID."""
        ts = (created_at or datetime.now(timezone.utc)).isoformat()
        cursor = await self._db.execute(
            "INSERT INTO sessions (title, project, created_at, last_active) VALUES (?, ?, ?, ?)",
            (title, project, ts, ts),
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

    async def update_session_project(self, session_id: int, project: str) -> None:
        """Update a session's project tag."""
        await self._db.execute(
            "UPDATE sessions SET project = ? WHERE id = ?", (project, session_id),
        )
        await self._db.commit()

    async def get_session_ids_for_project(self, project: str) -> set[int]:
        """Get all session IDs tagged with a project name."""
        cursor = await self._db.execute(
            "SELECT id FROM sessions WHERE project = ?", (project,),
        )
        return {row[0] for row in await cursor.fetchall()}

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

    async def session_exists(self, title: str) -> bool:
        """Check if a session with the given title already exists."""
        cursor = await self._db.execute(
            "SELECT 1 FROM sessions WHERE title = ? LIMIT 1", (title,)
        )
        return await cursor.fetchone() is not None

    async def get_session_message_count(self, title: str) -> tuple[int | None, int]:
        """Get the session ID and message_count for a session by title.

        Returns (session_id, message_count) or (None, 0) if not found.
        """
        cursor = await self._db.execute(
            "SELECT id, message_count FROM sessions WHERE title = ? LIMIT 1",
            (title,),
        )
        row = await cursor.fetchone()
        if row:
            return row[0], row[1] or 0
        return None, 0

    async def save_session_message(
        self, session_id: int, role: str, content: str,
        timestamp: str | None = None,
    ) -> int:
        """Persist a session message. Returns the row ID."""
        cursor = await self._db.execute(
            """INSERT INTO session_messages (session_id, role, content, timestamp)
               VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))""",
            (session_id, role, content, timestamp),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def mark_message_processed(self, message_id: int):
        """Mark a session message as successfully processed into memory."""
        await self._db.execute(
            "UPDATE session_messages SET is_processed = 1 WHERE id = ?",
            (message_id,),
        )
        await self._db.commit()

    async def get_unprocessed_messages(
        self, limit: int = 100,
    ) -> list[dict]:
        """Get session messages that failed to process (for startup sweep).

        Returns dicts with id, session_id, role, content, timestamp.
        Only returns user/assistant messages (not system/tool).
        """
        cursor = await self._db.execute(
            """SELECT id, session_id, role, content, timestamp
               FROM session_messages
               WHERE is_processed = 0 AND role IN ('user', 'assistant')
               ORDER BY id ASC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    async def delete_session_cascade(
        self, session_id: int, memory_ids: list[int] | None = None,
    ):
        """Delete a session and all its memories, lessons, and tags.

        If memory_ids is not provided, queries them from the DB.
        """
        if memory_ids is None:
            cursor = await self._db.execute(
                "SELECT id FROM memories WHERE session_id = ?", (session_id,)
            )
            memory_ids = [row[0] for row in await cursor.fetchall()]

        # memory_tags cleaned by ON DELETE CASCADE
        if memory_ids:
            placeholders = ",".join("?" * len(memory_ids))
            await self._db.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                memory_ids,
            )

        # Clean up lessons from this session
        await self._db.execute(
            "DELETE FROM lessons WHERE source_session_id = ?", (session_id,)
        )

        # Delete the session itself
        await self._db.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        await self._db.commit()

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

    async def list_sessions_chronological(
        self, project: str, limit: int = 200,
    ) -> list[Session]:
        """List sessions for a project in chronological order (oldest first).

        Only returns sessions that have a summary (needed for digest generation).
        """
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE project = ? AND summary IS NOT NULL "
            "ORDER BY created_at ASC LIMIT ?",
            (project, limit),
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

    async def search_sessions_by_keyword(
        self, keyword: str, limit: int = 50,
    ) -> list[Session]:
        """Search sessions whose title or summary mentions a keyword.

        Used as a fallback when no sessions are tagged with a project name
        (e.g., imported sessions that predate project creation).
        Splits hyphenated/underscored names and searches for any component
        (e.g., "cozmo-explorer" searches for "cozmo-explorer", "cozmo", "explorer").
        """
        import re
        # Split project name into searchable parts
        parts = re.split(r'[-_\s]+', keyword)
        # Search for the full name + each individual word (3+ chars)
        keywords = [keyword] + [p for p in parts if len(p) >= 3]
        # Build OR conditions for each keyword
        conditions = []
        params = []
        for kw in keywords:
            pattern = f"%{kw}%"
            conditions.append("(title LIKE ? OR summary LIKE ?)")
            params.extend([pattern, pattern])
        where = " OR ".join(conditions)
        params.append(limit)
        cursor = await self._db.execute(
            f"SELECT * FROM sessions WHERE summary IS NOT NULL "
            f"AND ({where}) "
            f"ORDER BY created_at ASC LIMIT ?",
            params,
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

    async def search_memories_by_keyword(
        self, keyword: str, limit: int = 100,
    ) -> list[dict]:
        """Search memory summaries mentioning a keyword.

        Returns dicts with {summary, timestamp, importance} sorted by
        importance DESC so the most relevant memories come first.
        Splits hyphenated/underscored names into components.
        """
        import re
        parts = re.split(r'[-_\s]+', keyword)
        keywords = [keyword] + [p for p in parts if len(p) >= 3]
        conditions = []
        params = []
        for kw in keywords:
            pattern = f"%{kw}%"
            conditions.append("summary LIKE ?")
            params.append(pattern)
        where = " OR ".join(conditions)
        params.append(limit)
        cursor = await self._db.execute(
            f"SELECT summary, timestamp, importance FROM memories "
            f"WHERE summary IS NOT NULL AND ({where}) "
            f"ORDER BY importance DESC, timestamp ASC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [
            {"summary": r["summary"], "timestamp": r["timestamp"], "importance": r["importance"]}
            for r in rows
        ]

    async def get_sessions_without_summaries(self, limit: int = 100) -> list[dict]:
        """Get sessions that have memories but no summary (need backfill).

        Returns dicts with {id, title, message_count} ordered oldest first.
        """
        cursor = await self._db.execute(
            """SELECT s.id, s.title, s.message_count
               FROM sessions s
               WHERE s.summary IS NULL
                 AND s.message_count > 0
               ORDER BY s.created_at ASC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {"id": r["id"], "title": r["title"], "message_count": r["message_count"]}
            for r in rows
        ]

    async def get_lessons_by_project(self, project: str) -> list[Lesson]:
        """Get all lessons tagged with a project."""
        cursor = await self._db.execute(
            "SELECT * FROM lessons WHERE project = ? ORDER BY timestamp ASC",
            (project,),
        )
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
                project=r["project"] if "project" in r.keys() else None,
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
        allowed = {"summary", "rank", "importance", "is_archived", "metadata_json", "memory_type"}
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

    async def get_memories_batch(self, memory_ids: list[int]) -> dict[int, "Memory"]:
        """Get multiple memories by ID in a single query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        cursor = await self._db.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids,
        )
        rows = await cursor.fetchall()
        return {row["id"]: self._row_to_memory(row) for row in rows}

    def _row_to_memory(self, row) -> Memory:
        # access_count / last_accessed may not exist in very old DBs
        try:
            access_count = row["access_count"] or 0
        except (IndexError, KeyError):
            access_count = 0
        try:
            last_accessed = row["last_accessed"]
        except (IndexError, KeyError):
            last_accessed = None
        try:
            consolidated_at = row["consolidated_at"]
        except (IndexError, KeyError):
            consolidated_at = None
        try:
            entities_extracted_at = row["entities_extracted_at"]
        except (IndexError, KeyError):
            entities_extracted_at = None
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
            access_count=access_count,
            last_accessed=last_accessed,
            consolidated_at=consolidated_at,
            entities_extracted_at=entities_extracted_at,
        )

    async def get_distinct_memory_session_ids(self) -> list[int]:
        """Get all distinct session_id values from the memories table."""
        cursor = await self._db.execute(
            "SELECT DISTINCT session_id FROM memories WHERE session_id IS NOT NULL ORDER BY session_id"
        )
        rows = await cursor.fetchall()
        return [r["session_id"] for r in rows]

    async def record_memory_access(self, memory_ids: list[int]):
        """Increment access_count and set last_accessed for retrieved memories."""
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        await self._db.executemany(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        await self._db.commit()

    async def get_unconsolidated_memory_ids(self, limit: int = 100) -> list[int]:
        """Get memory IDs that haven't been consolidation-checked yet."""
        cursor = await self._db.execute(
            """SELECT id FROM memories
               WHERE consolidated_at IS NULL AND is_archived = 0 AND summary IS NOT NULL
               ORDER BY timestamp ASC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [r["id"] for r in rows]

    async def mark_memories_consolidated(self, memory_ids: list[int]):
        """Set consolidated_at timestamp for checked memories."""
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        await self._db.executemany(
            "UPDATE memories SET consolidated_at = ? WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        await self._db.commit()

    async def delete_memory(self, memory_id: int):
        """Delete a single memory by ID (tags cleaned by ON DELETE CASCADE)."""
        await self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await self._db.commit()

    async def transfer_memory_tags(self, from_id: int, to_id: int):
        """Copy tags from one memory to another (union, skip duplicates)."""
        await self._db.execute(
            """INSERT OR IGNORE INTO memory_tags (memory_id, tag_id, timestamp)
               SELECT ?, tag_id, timestamp FROM memory_tags WHERE memory_id = ?""",
            (to_id, from_id),
        )
        await self._db.commit()

    _FTS5_RESERVED = {'AND', 'OR', 'NOT', 'NEAR'}

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip FTS5 special characters and reserved keywords to prevent syntax errors."""
        # Remove characters that FTS5 interprets as operators
        sanitized = query.replace('"', ' ').replace("'", ' ')
        for ch in '?*(){}[]^~:\\/<>!@#$%&+=|,;.-':
            sanitized = sanitized.replace(ch, ' ')
        # Remove FTS5 reserved keywords (AND, OR, NOT, NEAR)
        tokens = sanitized.split()
        tokens = [t for t in tokens if t.upper() not in SQLiteStore._FTS5_RESERVED]
        return ' '.join(tokens)

    async def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search on memory summaries using FTS5.

        Returns list of {id, fts_rank} dicts sorted by relevance.
        """
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []
        try:
            cursor = await self._db.execute(
                """SELECT rowid AS id, rank AS fts_rank
                   FROM memories_fts WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?""",
                (sanitized, limit),
            )
            rows = await cursor.fetchall()
            return [{"id": r["id"], "fts_rank": r["fts_rank"]} for r in rows]
        except Exception as e:
            logger.warning("FTS5 search failed: %s", e)
            return []

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
               source_session_id, added_by, project)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                lesson.content,
                lesson.summary,
                lesson.timestamp.isoformat(),
                lesson.rank,
                lesson.importance,
                lesson.source_session_id,
                "system",
                lesson.project,
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
                project=r["project"] if "project" in r.keys() else None,
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

    async def _ensure_tags_exist(self, tag_names: list[str], category: str = "topic") -> dict[str, int]:
        """Batch-ensure tags exist and return {name: id} mapping.

        Uses INSERT OR IGNORE to create missing tags in bulk, then a single
        SELECT to retrieve all IDs.
        """
        if not tag_names:
            return {}
        await self._db.executemany(
            "INSERT OR IGNORE INTO tags (name, category) VALUES (?, ?)",
            [(name, category) for name in tag_names],
        )
        placeholders = ",".join("?" * len(tag_names))
        cursor = await self._db.execute(
            f"SELECT id, name FROM tags WHERE name IN ({placeholders}) AND category = ?",
            [*tag_names, category],
        )
        rows = await cursor.fetchall()
        return {r["name"]: r["id"] for r in rows}

    async def tag_memory(self, memory_id: int, tag_names: list[str]):
        """Associate tags with a memory."""
        tag_ids = await self._ensure_tags_exist(tag_names)
        await self._db.executemany(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
            [(memory_id, tid) for tid in tag_ids.values()],
        )
        await self._db.commit()

    async def tag_core_memory(self, core_memory_id: int, tag_names: list[str]):
        """Associate tags with a core memory."""
        tag_ids = await self._ensure_tags_exist(tag_names)
        await self._db.executemany(
            "INSERT OR IGNORE INTO core_memory_tags (core_memory_id, tag_id) VALUES (?, ?)",
            [(core_memory_id, tid) for tid in tag_ids.values()],
        )
        await self._db.commit()

    async def tag_lesson(self, lesson_id: int, tag_names: list[str]):
        """Associate tags with a lesson."""
        tag_ids = await self._ensure_tags_exist(tag_names)
        await self._db.executemany(
            "INSERT OR IGNORE INTO lesson_tags (lesson_id, tag_id) VALUES (?, ?)",
            [(lesson_id, tid) for tid in tag_ids.values()],
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

    async def get_tags_for_memories(self, memory_ids: list[int]) -> dict[int, list[str]]:
        """Batch-load tag names for multiple memories in a single query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        cursor = await self._db.execute(
            f"""SELECT mt.memory_id, t.name FROM tags t
                INNER JOIN memory_tags mt ON mt.tag_id = t.id
                WHERE mt.memory_id IN ({placeholders})""",
            memory_ids,
        )
        rows = await cursor.fetchall()
        result: dict[int, list[str]] = {mid: [] for mid in memory_ids}
        for r in rows:
            result[r["memory_id"]].append(r["name"])
        return result

    async def get_tag_count_for_memory(self, memory_id: int) -> int:
        """Get number of tags for a memory (used in importance calculation)."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) as cnt FROM memory_tags WHERE memory_id = ?", (memory_id,)
        )
        row = await cursor.fetchone()
        return row["cnt"]

    # --- Archiving / Pruning ---

    async def archive_old_memories(
        self,
        days_old: int = 90,
        max_importance: float = 0.3,
        max_rank: int = 2,
    ) -> int:
        """Archive memories older than N days with low rank and importance.

        Returns count of newly archived memories.
        """
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        cursor = await self._db.execute(
            """UPDATE memories SET is_archived = 1
               WHERE is_archived = 0
               AND timestamp < ?
               AND rank <= ?
               AND importance <= ?""",
            (cutoff, max_rank, max_importance),
        )
        await self._db.commit()
        count = cursor.rowcount
        if count:
            logger.info("Archived %d old low-value memories (older than %d days)", count, days_old)
        return count

    async def get_archived_memory_ids(
        self,
        days_old: int = 90,
        max_importance: float = 0.3,
        max_rank: int = 2,
    ) -> list[int]:
        """Get IDs of memories that were just archived (for ChromaDB cleanup)."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
        cursor = await self._db.execute(
            """SELECT id FROM memories
               WHERE is_archived = 1
               AND timestamp < ?
               AND rank <= ?
               AND importance <= ?""",
            (cutoff, max_rank, max_importance),
        )
        rows = await cursor.fetchall()
        return [r["id"] for r in rows]

    async def get_archive_stats(self) -> dict:
        """Get counts of active vs archived memories."""
        cursor = await self._db.execute(
            "SELECT is_archived, COUNT(*) as cnt FROM memories GROUP BY is_archived"
        )
        rows = await cursor.fetchall()
        stats = {"active": 0, "archived": 0}
        for r in rows:
            if r["is_archived"]:
                stats["archived"] = r["cnt"]
            else:
                stats["active"] = r["cnt"]
        return stats

    # --- Paginated Queries ---

    async def get_memories_paginated(
        self,
        page: int = 1,
        limit: int = 20,
        sort: str = "recent",
        include_archived: bool = False,
    ) -> tuple[list[Memory], int]:
        """Get paginated memories. Returns (memories, total_count)."""
        offset = (page - 1) * limit
        where = "" if include_archived else "WHERE is_archived = 0"
        order = "timestamp DESC" if sort == "recent" else "rank DESC, importance DESC"

        # Count
        cursor = await self._db.execute(f"SELECT COUNT(*) as cnt FROM memories {where}")
        row = await cursor.fetchone()
        total = row["cnt"]

        # Fetch page
        cursor = await self._db.execute(
            f"SELECT * FROM memories {where} ORDER BY {order} LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows], total

    async def get_memory_with_tags(self, memory_id: int) -> dict | None:
        """Get a memory with its tags."""
        memory = await self.get_memory(memory_id)
        if not memory:
            return None
        tags = await self.get_memory_tags(memory_id)
        return {**memory.model_dump(), "tags": tags}

    # --- Lesson Management ---

    async def get_lesson(self, lesson_id: int) -> Optional[Lesson]:
        """Get a single lesson by ID."""
        cursor = await self._db.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return Lesson(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            timestamp=row["timestamp"],
            rank=row["rank"],
            importance=row["importance"],
            source_session_id=row["source_session_id"],
            project=row["project"] if "project" in row.keys() else None,
        )

    async def delete_lesson(self, lesson_id: int):
        """Delete a lesson."""
        await self._db.execute("DELETE FROM lesson_tags WHERE lesson_id = ?", (lesson_id,))
        await self._db.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
        await self._db.commit()

    # --- Core Memory Management ---

    async def get_core_memory(self, core_memory_id: int) -> Optional[CoreMemory]:
        """Get a single core memory by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM core_memories WHERE id = ?", (core_memory_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return CoreMemory(
            id=row["id"],
            content=row["content"],
            category=row["category"],
            timestamp=row["timestamp"],
            importance=row["importance"],
            source_session_id=row["source_session_id"],
        )

    async def update_core_memory(self, core_memory_id: int, **kwargs):
        """Update core memory fields."""
        allowed = {"content", "category", "importance", "is_active"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [core_memory_id]
        await self._db.execute(
            f"UPDATE core_memories SET {set_clause} WHERE id = ?", values
        )
        await self._db.commit()

    # --- Projects ---

    async def create_project(
        self,
        name: str,
        description: str = "",
        root_path: Optional[str] = None,
        git_url: Optional[str] = None,
        language: Optional[str] = None,
    ) -> int:
        """Create a named project."""
        ts = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            """INSERT OR IGNORE INTO projects
               (name, description, root_path, git_url, language, created_at, last_active)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, description, root_path, git_url, language, ts, ts),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_project(self, name: str) -> Optional[dict]:
        """Get a project by name."""
        cursor = await self._db.execute(
            "SELECT * FROM projects WHERE name = ?", (name,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_project(self, name: str, **fields) -> None:
        """Update project fields by name."""
        allowed = {"description", "root_path", "git_url", "language", "settings_json", "metadata_json"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [name]
        await self._db.execute(
            f"UPDATE projects SET {set_clause} WHERE name = ?", values,
        )
        await self._db.commit()

    async def touch_project(self, name: str) -> None:
        """Update a project's last_active timestamp."""
        ts = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE projects SET last_active = ? WHERE name = ?", (ts, name),
        )
        await self._db.commit()

    async def delete_project(self, name: str) -> None:
        """Delete a project by name (does not touch files on disk)."""
        await self._db.execute("DELETE FROM projects WHERE name = ?", (name,))
        await self._db.commit()

    async def list_projects(self) -> list[dict]:
        """List all projects."""
        cursor = await self._db.execute(
            "SELECT * FROM projects ORDER BY last_active DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # --- Task Plans ---

    async def create_plan(self, plan: TaskPlan) -> int:
        """Create a task plan and return its ID."""
        cursor = await self._db.execute(
            """INSERT INTO task_plans (session_id, user_request, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                plan.session_id,
                plan.user_request,
                plan.status.value,
                plan.created_at.isoformat(),
                plan.updated_at.isoformat(),
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_plan(self, plan_id: int) -> Optional[TaskPlan]:
        """Get a task plan by ID, including its steps."""
        cursor = await self._db.execute(
            "SELECT * FROM task_plans WHERE id = ?", (plan_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        steps = await self.get_plan_steps(plan_id)
        return TaskPlan(
            id=row["id"],
            session_id=row["session_id"],
            user_request=row["user_request"],
            status=PlanStatus(row["status"]),
            steps=steps,
            result_summary=row["result_summary"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def update_plan(self, plan_id: int, **kwargs):
        """Update task plan fields."""
        allowed = {"status", "result_summary", "updated_at"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        # Convert enums to their values
        for k, v in fields.items():
            if hasattr(v, "value"):
                fields[k] = v.value
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [plan_id]
        await self._db.execute(f"UPDATE task_plans SET {set_clause} WHERE id = ?", values)
        await self._db.commit()

    async def list_plans(self, session_id: Optional[int] = None, limit: int = 20) -> list[TaskPlan]:
        """List task plans, optionally filtered by session."""
        if session_id:
            cursor = await self._db.execute(
                "SELECT * FROM task_plans WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM task_plans ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        plans = []
        for row in rows:
            steps = await self.get_plan_steps(row["id"])
            plans.append(TaskPlan(
                id=row["id"],
                session_id=row["session_id"],
                user_request=row["user_request"],
                status=PlanStatus(row["status"]),
                steps=steps,
                result_summary=row["result_summary"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            ))
        return plans

    async def get_active_plan(self, session_id: int) -> Optional[TaskPlan]:
        """Get the currently running or approved plan for a session."""
        cursor = await self._db.execute(
            """SELECT * FROM task_plans
               WHERE session_id = ? AND status IN ('planning', 'approved', 'running')
               ORDER BY created_at DESC LIMIT 1""",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        steps = await self.get_plan_steps(row["id"])
        return TaskPlan(
            id=row["id"],
            session_id=row["session_id"],
            user_request=row["user_request"],
            status=PlanStatus(row["status"]),
            steps=steps,
            result_summary=row["result_summary"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # --- Task Steps ---

    async def create_step(self, step: TaskStep) -> int:
        """Create a task step and return its ID."""
        cursor = await self._db.execute(
            """INSERT INTO task_steps
               (plan_id, step_number, description, status, tool_hint, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                step.plan_id,
                step.step_number,
                step.description,
                step.status.value,
                step.tool_hint,
                step.created_at.isoformat(),
                step.updated_at.isoformat(),
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_plan_steps(self, plan_id: int) -> list[TaskStep]:
        """Get all steps for a plan, ordered by step number."""
        cursor = await self._db.execute(
            "SELECT * FROM task_steps WHERE plan_id = ? ORDER BY step_number",
            (plan_id,),
        )
        rows = await cursor.fetchall()
        return [
            TaskStep(
                id=r["id"],
                plan_id=r["plan_id"],
                step_number=r["step_number"],
                description=r["description"],
                status=StepStatus(r["status"]),
                tool_hint=r["tool_hint"],
                output_result=r["output_result"],
                error_message=r["error_message"],
                retry_count=r["retry_count"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    async def update_step(self, step_id: int, **kwargs):
        """Update task step fields."""
        allowed = {"status", "output_result", "error_message", "retry_count", "updated_at"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        for k, v in fields.items():
            if hasattr(v, "value"):
                fields[k] = v.value
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [step_id]
        await self._db.execute(f"UPDATE task_steps SET {set_clause} WHERE id = ?", values)
        await self._db.commit()

    # --- Background Tasks ---

    async def create_background_task(self, task: BackgroundTask) -> int:
        """Create a background task and return its ID."""
        cursor = await self._db.execute(
            """INSERT INTO background_tasks
               (session_id, plan_id, title, task_type, prompt, status, priority,
                progress_pct, progress_message, target_endpoint, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task.session_id,
                task.plan_id,
                task.title,
                task.task_type,
                task.prompt,
                task.status.value,
                task.priority,
                task.progress_pct,
                task.progress_message,
                task.target_endpoint,
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_background_task(self, task_id: int) -> Optional[BackgroundTask]:
        """Get a background task by ID."""
        cursor = await self._db.execute(
            "SELECT * FROM background_tasks WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_background_task(row)

    async def update_background_task(self, task_id: int, **kwargs):
        """Update background task fields."""
        allowed = {
            "status", "progress_pct", "progress_message", "result",
            "error_message", "target_endpoint", "updated_at",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        for k, v in fields.items():
            if hasattr(v, "value"):
                fields[k] = v.value
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [task_id]
        await self._db.execute(
            f"UPDATE background_tasks SET {set_clause} WHERE id = ?", values
        )
        await self._db.commit()

    async def list_background_tasks(
        self,
        session_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[BackgroundTask]:
        """List background tasks with optional filters."""
        conditions = []
        params = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        cursor = await self._db.execute(
            f"SELECT * FROM background_tasks {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_background_task(r) for r in rows]

    async def get_pending_background_tasks(
        self, endpoint_name: Optional[str] = None,
    ) -> list[BackgroundTask]:
        """Get pending background tasks, optionally filtered by target endpoint."""
        if endpoint_name:
            cursor = await self._db.execute(
                """SELECT * FROM background_tasks
                   WHERE status = 'pending' AND target_endpoint = ?
                   ORDER BY priority DESC, created_at ASC""",
                (endpoint_name,),
            )
        else:
            cursor = await self._db.execute(
                """SELECT * FROM background_tasks
                   WHERE status = 'pending'
                   ORDER BY priority DESC, created_at ASC"""
            )
        rows = await cursor.fetchall()
        return [self._row_to_background_task(r) for r in rows]

    # --- Tag Discovery ---

    async def get_discovered_tag_patterns(self) -> dict[str, list[str]]:
        """Load active discovered tag patterns as {tag_name: [regex_str, ...]}."""
        cursor = await self._db.execute(
            "SELECT tag_name, pattern FROM discovered_tag_patterns WHERE is_active = 1"
        )
        rows = await cursor.fetchall()
        result: dict[str, list[str]] = {}
        for r in rows:
            result.setdefault(r["tag_name"], []).append(r["pattern"])
        return result

    async def save_discovered_tag_patterns(self, patterns: dict[str, list[str]]):
        """Persist newly discovered patterns (skip duplicates)."""
        rows = [
            (tag_name, pattern)
            for tag_name, pattern_list in patterns.items()
            for pattern in pattern_list
        ]
        if not rows:
            return
        await self._db.executemany(
            "INSERT OR IGNORE INTO discovered_tag_patterns (tag_name, pattern) VALUES (?, ?)",
            rows,
        )
        await self._db.commit()

    async def get_poorly_tagged_memory_summaries(
        self, max_tags: int = 1, limit: int = 20,
    ) -> list[str]:
        """Get summaries of non-archived memories with few tags for LLM review."""
        cursor = await self._db.execute(
            """SELECT m.summary FROM memories m
               LEFT JOIN memory_tags mt ON mt.memory_id = m.id
               WHERE m.is_archived = 0 AND m.summary IS NOT NULL
               GROUP BY m.id
               HAVING COUNT(mt.id) <= ?
               ORDER BY m.timestamp DESC
               LIMIT ?""",
            (max_tags, limit),
        )
        rows = await cursor.fetchall()
        return [r["summary"] for r in rows]

    async def get_tag_member_counts(self, min_members: int = 10) -> dict[str, int]:
        """Get tags with at least min_members non-archived memories."""
        cursor = await self._db.execute(
            """SELECT t.name, COUNT(mt.memory_id) as cnt
               FROM tags t
               INNER JOIN memory_tags mt ON mt.tag_id = t.id
               INNER JOIN memories m ON m.id = mt.memory_id
               WHERE m.is_archived = 0
               GROUP BY t.id
               HAVING cnt >= ?
               ORDER BY cnt DESC""",
            (min_members,),
        )
        rows = await cursor.fetchall()
        return {r["name"]: r["cnt"] for r in rows}

    async def get_memory_ids_for_tag(self, tag_name: str, limit: int = 200) -> list[int]:
        """Get IDs of non-archived memories with a specific tag."""
        cursor = await self._db.execute(
            """SELECT mt.memory_id FROM memory_tags mt
               INNER JOIN tags t ON t.id = mt.tag_id
               INNER JOIN memories m ON m.id = mt.memory_id
               WHERE t.name = ? AND m.is_archived = 0
               ORDER BY m.timestamp DESC
               LIMIT ?""",
            (tag_name, limit),
        )
        rows = await cursor.fetchall()
        return [r["memory_id"] for r in rows]

    async def get_poorly_tagged_memory_ids(
        self, max_tags: int = 1, limit: int = 500,
    ) -> list[int]:
        """Get IDs of non-archived memories with few tags."""
        cursor = await self._db.execute(
            """SELECT m.id FROM memories m
               LEFT JOIN memory_tags mt ON mt.memory_id = m.id
               WHERE m.is_archived = 0 AND m.summary IS NOT NULL
               GROUP BY m.id
               HAVING COUNT(mt.id) <= ?
               ORDER BY m.timestamp DESC
               LIMIT ?""",
            (max_tags, limit),
        )
        rows = await cursor.fetchall()
        return [r["id"] for r in rows]

    async def get_well_tagged_memory_sample(
        self, min_tags: int = 3, limit: int = 30,
    ) -> list[dict]:
        """Get a sample of well-tagged memories for benchmarking.

        Returns list of {id, summary, tags: [str]}.
        """
        cursor = await self._db.execute(
            """SELECT m.id, m.summary, GROUP_CONCAT(t.name) as tag_names
               FROM memories m
               INNER JOIN memory_tags mt ON mt.memory_id = m.id
               INNER JOIN tags t ON t.id = mt.tag_id
               WHERE m.is_archived = 0 AND m.summary IS NOT NULL
               GROUP BY m.id
               HAVING COUNT(mt.id) >= ?
               ORDER BY RANDOM()
               LIMIT ?""",
            (min_tags, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "summary": r["summary"],
                "tags": r["tag_names"].split(",") if r["tag_names"] else [],
            }
            for r in rows
        ]

    async def get_all_tag_names(self) -> list[str]:
        """Get all tag names from the tags table."""
        cursor = await self._db.execute("SELECT name FROM tags ORDER BY name")
        rows = await cursor.fetchall()
        return [r["name"] for r in rows]

    # --- Entity Graph ---

    async def get_or_create_entity(self, name: str, entity_type: str = "concept") -> int:
        """Get existing entity ID or create a new one. Names are lowercased."""
        name = name.strip().lower()
        cursor = await self._db.execute(
            "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
            (name, entity_type),
        )
        row = await cursor.fetchone()
        if row:
            return row["id"]
        cursor = await self._db.execute(
            "INSERT INTO entities (name, entity_type) VALUES (?, ?)",
            (name, entity_type),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def create_entity_relationship(
        self, subject_id: int, predicate: str, object_id: int, memory_id: int | None,
    ) -> int | None:
        """Create a relationship triple. Returns ID or None if duplicate."""
        try:
            cursor = await self._db.execute(
                """INSERT OR IGNORE INTO entity_relationships
                   (subject_id, predicate, object_id, source_memory_id)
                   VALUES (?, ?, ?, ?)""",
                (subject_id, predicate.strip().lower(), object_id, memory_id),
            )
            await self._db.commit()
            return cursor.lastrowid if cursor.lastrowid else None
        except Exception as e:
            logger.warning("Failed to create relationship: %s", e)
            return None

    async def create_entity_mention(self, entity_id: int, memory_id: int):
        """Record that an entity is mentioned in a memory."""
        await self._db.execute(
            "INSERT OR IGNORE INTO entity_mentions (entity_id, memory_id) VALUES (?, ?)",
            (entity_id, memory_id),
        )
        await self._db.commit()

    async def get_unextracted_memory_ids(self, limit: int = 50) -> list[int]:
        """Get memory IDs that haven't had entities extracted yet."""
        cursor = await self._db.execute(
            """SELECT id FROM memories
               WHERE entities_extracted_at IS NULL AND is_archived = 0
               AND summary IS NOT NULL
               ORDER BY timestamp ASC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [r["id"] for r in rows]

    async def mark_entities_extracted(self, memory_ids: list[int]):
        """Set entities_extracted_at timestamp for processed memories."""
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        await self._db.executemany(
            "UPDATE memories SET entities_extracted_at = ? WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        await self._db.commit()

    async def get_entity_ids_by_names(self, names: list[str]) -> list[int]:
        """Get entity IDs matching any of the given names (case-insensitive)."""
        if not names:
            return []
        lowered = [n.strip().lower() for n in names]
        placeholders = ",".join("?" * len(lowered))
        cursor = await self._db.execute(
            f"SELECT id FROM entities WHERE name IN ({placeholders})",
            lowered,
        )
        rows = await cursor.fetchall()
        return [r["id"] for r in rows]

    async def get_connected_entity_ids(self, entity_ids: list[int]) -> list[int]:
        """Get entity IDs connected to the given entities via active relationships.

        Only returns connections through non-expired relationships.
        """
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        cursor = await self._db.execute(
            f"""SELECT DISTINCT object_id AS eid FROM entity_relationships
                WHERE subject_id IN ({placeholders}) AND expired_at IS NULL
                UNION
                SELECT DISTINCT subject_id AS eid FROM entity_relationships
                WHERE object_id IN ({placeholders}) AND expired_at IS NULL""",
            entity_ids + entity_ids,
        )
        rows = await cursor.fetchall()
        # Return connected entities that aren't in the original set
        original = set(entity_ids)
        return [r["eid"] for r in rows if r["eid"] not in original]

    async def get_memory_ids_for_entities(self, entity_ids: list[int]) -> list[int]:
        """Get memory IDs that mention any of the given entities."""
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        cursor = await self._db.execute(
            f"""SELECT DISTINCT memory_id FROM entity_mentions
                WHERE entity_id IN ({placeholders})""",
            entity_ids,
        )
        rows = await cursor.fetchall()
        return [r["memory_id"] for r in rows]

    async def get_all_entity_names(self) -> list[str]:
        """Get all entity names (for fast in-memory matching at search time)."""
        cursor = await self._db.execute("SELECT DISTINCT name FROM entities")
        rows = await cursor.fetchall()
        return [r["name"] for r in rows]

    # --- Bi-temporal Edge Tracking (Feature 4) ---

    async def create_entity_relationship_temporal(
        self, subject_id: int, predicate: str, object_id: int,
        memory_id: int | None, valid_from: str | None = None,
    ) -> int | None:
        """Create a temporal relationship triple. Returns ID or None if duplicate.

        Automatically sets valid_from to now if not provided.
        """
        predicate = predicate.strip().lower()
        if not valid_from:
            valid_from = datetime.now(timezone.utc).isoformat()
        try:
            cursor = await self._db.execute(
                """INSERT OR IGNORE INTO entity_relationships
                   (subject_id, predicate, object_id, source_memory_id, valid_from)
                   VALUES (?, ?, ?, ?, ?)""",
                (subject_id, predicate, object_id, memory_id, valid_from),
            )
            await self._db.commit()
            return cursor.lastrowid if cursor.lastrowid else None
        except Exception as e:
            logger.warning("Failed to create temporal relationship: %s", e)
            return None

    async def expire_contradicting_relationships(
        self, subject_id: int, object_id: int,
        contradicting_predicates: dict[str, list[str]],
        new_predicate: str, new_relationship_id: int,
    ) -> int:
        """Expire old relationships that contradict the new one.

        Args:
            subject_id: The subject entity ID
            object_id: The object entity ID
            contradicting_predicates: Dict mapping predicate to list of contradicting predicates
            new_predicate: The predicate of the new relationship
            new_relationship_id: The ID of the new relationship (for expired_by)

        Returns:
            Count of expired relationships.
        """
        contradicts = contradicting_predicates.get(new_predicate, [])
        if not contradicts:
            return 0

        placeholders = ",".join("?" * len(contradicts))
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            f"""UPDATE entity_relationships
                SET expired_at = ?, expired_by = ?
                WHERE subject_id = ? AND object_id = ?
                AND predicate IN ({placeholders})
                AND expired_at IS NULL
                AND id != ?""",
            [now, new_relationship_id, subject_id, object_id]
            + contradicts + [new_relationship_id],
        )
        await self._db.commit()
        return cursor.rowcount

    async def get_active_relationships_for_entity(
        self, entity_id: int,
    ) -> list[dict]:
        """Get all active (non-expired) relationships for an entity."""
        cursor = await self._db.execute(
            """SELECT er.*, e1.name as subject_name, e2.name as object_name
               FROM entity_relationships er
               JOIN entities e1 ON er.subject_id = e1.id
               JOIN entities e2 ON er.object_id = e2.id
               WHERE (er.subject_id = ? OR er.object_id = ?)
               AND er.expired_at IS NULL
               ORDER BY er.created_at DESC""",
            (entity_id, entity_id),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_expired_relationships_for_entity(
        self, entity_id: int,
    ) -> list[dict]:
        """Get all expired relationships for an entity."""
        cursor = await self._db.execute(
            """SELECT er.*, e1.name as subject_name, e2.name as object_name
               FROM entity_relationships er
               JOIN entities e1 ON er.subject_id = e1.id
               JOIN entities e2 ON er.object_id = e2.id
               WHERE (er.subject_id = ? OR er.object_id = ?)
               AND er.expired_at IS NOT NULL
               ORDER BY er.expired_at DESC""",
            (entity_id, entity_id),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # --- Entity Resolution (Feature 5) ---

    async def get_entity_id_by_name(self, name: str) -> int | None:
        """Get entity ID by exact name match (case-insensitive)."""
        cursor = await self._db.execute(
            "SELECT id FROM entities WHERE name = ?", (name.strip().lower(),),
        )
        row = await cursor.fetchone()
        return row["id"] if row else None

    async def record_entity_alias(
        self, alias_name: str, canonical_entity_id: int,
        merge_method: str = "exact",
    ) -> None:
        """Record an entity alias for audit trail."""
        try:
            await self._db.execute(
                """INSERT OR IGNORE INTO entity_aliases
                   (alias_name, canonical_entity_id, merge_method)
                   VALUES (?, ?, ?)""",
                (alias_name.strip().lower(), canonical_entity_id, merge_method),
            )
            await self._db.commit()
        except Exception as e:
            logger.warning("Failed to record entity alias: %s", e)

    async def merge_entity(
        self, old_entity_id: int, canonical_entity_id: int,
    ) -> None:
        """Merge an entity into another: reassign all mentions and relationships.

        Does NOT delete the old entity (it becomes an alias).
        """
        # Reassign entity mentions
        await self._db.execute(
            """UPDATE OR IGNORE entity_mentions
               SET entity_id = ? WHERE entity_id = ?""",
            (canonical_entity_id, old_entity_id),
        )
        # Delete any duplicate mentions that couldn't be updated
        await self._db.execute(
            "DELETE FROM entity_mentions WHERE entity_id = ?",
            (old_entity_id,),
        )
        # Reassign relationships (subject side)
        await self._db.execute(
            """UPDATE OR IGNORE entity_relationships
               SET subject_id = ? WHERE subject_id = ?""",
            (canonical_entity_id, old_entity_id),
        )
        # Reassign relationships (object side)
        await self._db.execute(
            """UPDATE OR IGNORE entity_relationships
               SET object_id = ? WHERE object_id = ?""",
            (canonical_entity_id, old_entity_id),
        )
        # Clean up any orphaned relationships from the old entity
        await self._db.execute(
            """DELETE FROM entity_relationships
               WHERE subject_id = ? OR object_id = ?""",
            (old_entity_id, old_entity_id),
        )
        await self._db.commit()

    async def get_metadata(self, key: str) -> Optional[str]:
        """Get an app metadata value by key."""
        cursor = await self._db.execute(
            "SELECT value FROM app_metadata WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    async def set_metadata(self, key: str, value: str):
        """Set an app metadata value (upsert)."""
        await self._db.execute(
            "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        await self._db.commit()

    # --- Background Tasks ---

    def _row_to_background_task(self, row) -> BackgroundTask:
        return BackgroundTask(
            id=row["id"],
            session_id=row["session_id"],
            plan_id=row["plan_id"],
            title=row["title"],
            task_type=row["task_type"],
            prompt=row["prompt"],
            status=BackgroundTaskStatus(row["status"]),
            priority=row["priority"],
            progress_pct=row["progress_pct"],
            progress_message=row["progress_message"] or "",
            result=row["result"],
            error_message=row["error_message"],
            target_endpoint=row["target_endpoint"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # --- Turn Events (Conversation Flow Observability) ---

    async def log_turn_event(self, session_id: int, turn_number: int,
                             event_type: str, data: dict):
        """Log a conversation flow event. Fire-and-forget safe."""
        try:
            await self._db.execute(
                """INSERT INTO turn_events (session_id, turn_number, event_type, data_json)
                   VALUES (?, ?, ?, ?)""",
                (session_id, turn_number, event_type, json.dumps(data)),
            )
            await self._db.commit()
        except Exception as e:
            logger.debug("Failed to log turn event: %s", e)

    async def get_turn_events(self, session_id: int, limit: int = 20) -> list[dict]:
        """Get recent turn events for a session."""
        cursor = await self._db.execute(
            """SELECT id, session_id, turn_number, event_type, timestamp, data_json
               FROM turn_events WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "turn_number": r["turn_number"],
                "event_type": r["event_type"],
                "timestamp": r["timestamp"],
                "data": json.loads(r["data_json"]) if r["data_json"] else {},
            }
            for r in reversed(rows)  # chronological order
        ]

    async def get_turn_events_for_turn(self, session_id: int,
                                       turn_number: int) -> list[dict]:
        """Get all events for a specific turn."""
        cursor = await self._db.execute(
            """SELECT id, session_id, turn_number, event_type, timestamp, data_json
               FROM turn_events
               WHERE session_id = ? AND turn_number = ?
               ORDER BY id""",
            (session_id, turn_number),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "turn_number": r["turn_number"],
                "event_type": r["event_type"],
                "timestamp": r["timestamp"],
                "data": json.loads(r["data_json"]) if r["data_json"] else {},
            }
            for r in rows
        ]
