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
            (title, project, datetime.now(timezone.utc).isoformat(), datetime.now(timezone.utc).isoformat()),
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
