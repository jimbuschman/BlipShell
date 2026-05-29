"""SQLite storage for structured data (port of MemoryDB.cs schema)."""

import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

# Regex for valid SQL column identifiers — prevents injection via dynamic column names.
_VALID_COLUMN_RE = re.compile(r"^[a-z][a-z0-9_]*$")

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
    is_processed BOOLEAN DEFAULT 1,
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
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_session_archived ON memories(session_id, is_archived);

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

CREATE TABLE IF NOT EXISTS chroma_retry_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,           -- 'upsert' or 'delete'
    collection TEXT NOT NULL,          -- 'memories', 'core_memories', 'lessons', 'entities'
    item_id INTEGER NOT NULL,
    document TEXT,                     -- text to embed (for upserts)
    metadata_json TEXT,                -- JSON metadata (for upserts)
    error TEXT,                        -- last error message
    retry_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(operation, collection, item_id)
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

-- FTS5 full-text search on memory summaries AND raw content.
-- Indexes both columns so keyword search finds facts in raw messages
-- (summaries often abstract away specific names, terms, and details).
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    summary, content, content=memories, content_rowid=id
);

-- Keep FTS index in sync with memories table
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories
BEGIN
    INSERT INTO memories_fts(rowid, summary, content)
    VALUES (NEW.id, COALESCE(NEW.summary, ''), COALESCE(NEW.content, ''));
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF summary, content ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, summary, content)
    VALUES('delete', OLD.id, COALESCE(OLD.summary, ''), COALESCE(OLD.content, ''));
    INSERT INTO memories_fts(rowid, summary, content)
    VALUES (NEW.id, COALESCE(NEW.summary, ''), COALESCE(NEW.content, ''));
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories
BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, summary, content)
    VALUES('delete', OLD.id, COALESCE(OLD.summary, ''), COALESCE(OLD.content, ''));
END;

CREATE TABLE IF NOT EXISTS session_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL UNIQUE,
    effectiveness TEXT,
    reflection_text TEXT NOT NULL,
    technical_insights TEXT,
    process_insights TEXT,
    what_worked TEXT,
    what_didnt_work TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_session_reflections_session ON session_reflections(session_id);

CREATE TABLE IF NOT EXISTS follow_ups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source_session INTEGER,
    project TEXT,
    due_hint TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    resolved_session INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME,
    FOREIGN KEY (source_session) REFERENCES sessions(id),
    FOREIGN KEY (resolved_session) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_follow_ups_status ON follow_ups(status);

CREATE TABLE IF NOT EXISTS friction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    source TEXT NOT NULL DEFAULT 'nightly',
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_reviewed BOOLEAN DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_friction_log_session ON friction_log(session_id);
CREATE INDEX IF NOT EXISTS idx_friction_log_reviewed ON friction_log(is_reviewed);
"""


def _safe_set_clause(fields: dict, allowed: set[str]) -> tuple[str, list]:
    """Build a parameterized SET clause from validated column names.

    Filters to allowed columns and validates each name matches [a-z][a-z0-9_]*.
    Returns (set_clause_str, values_list).
    """
    safe = {k: v for k, v in fields.items() if k in allowed}
    for col in safe:
        if not _VALID_COLUMN_RE.match(col):
            raise ValueError(f"Invalid column name: {col!r}")
    clause = ", ".join(f"{k} = ?" for k in safe)
    return clause, list(safe.values())


class SQLiteStore:
    """Async SQLite storage for structured data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    @asynccontextmanager
    async def transaction(self):
        """Explicit transaction context manager for multi-step atomic operations.

        Usage:
            async with sqlite.transaction():
                await sqlite.update_memory(mid, is_archived=True)
                await sqlite.delete_lesson(lid)
                # Both committed atomically, or both rolled back on error.

        Under normal operation, each store method commits its own work.
        Use this only when multiple methods must succeed or fail together.
        """
        await self._db.execute("BEGIN")
        try:
            yield
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    async def initialize(self):
        """Open connection and create schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path, timeout=60, isolation_level="DEFERRED")
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
            # Lesson hit tracking
            "ALTER TABLE lessons ADD COLUMN hit_count INTEGER DEFAULT 0",
            "ALTER TABLE lessons ADD COLUMN last_accessed DATETIME",
            # Bi-temporal edge tracking (Feature 4)
            "ALTER TABLE entity_relationships ADD COLUMN valid_from DATETIME",
            "ALTER TABLE entity_relationships ADD COLUMN expired_at DATETIME",
            "ALTER TABLE entity_relationships ADD COLUMN expired_by INTEGER",
            # Unify crash recovery — memories.is_processed replaces session_messages
            "ALTER TABLE memories ADD COLUMN is_processed BOOLEAN DEFAULT 1",
            # External-source dedup for periodic imports (ChatGPT/Claude conversation id)
            "ALTER TABLE sessions ADD COLUMN external_id TEXT",
            "ALTER TABLE sessions ADD COLUMN external_updated_at REAL",
            "CREATE INDEX IF NOT EXISTS idx_sessions_external_id ON sessions(external_id)",
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
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_unprocessed "
            "ON memories(is_processed) WHERE is_processed = 0"
        )
        # Tool approval audit trail
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS tool_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                tool_name TEXT NOT NULL,
                arguments_json TEXT,
                approved BOOLEAN NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        # Backfill valid_from from created_at for existing relationships
        await self._db.execute(
            "UPDATE entity_relationships SET valid_from = created_at "
            "WHERE valid_from IS NULL"
        )
        # Backfill FTS5 index — only run once (tracks completion in app_metadata)
        cursor = await self._db.execute(
            "SELECT value FROM app_metadata WHERE key = 'fts5_backfill_done'"
        )
        row = await cursor.fetchone()
        if not row:
            await self._db.execute(
                """INSERT OR IGNORE INTO memories_fts(rowid, summary)
                   SELECT id, summary FROM memories WHERE summary IS NOT NULL"""
            )
            await self._db.execute(
                "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('fts5_backfill_done', '1')"
            )

        # FTS5 v2 migration: index both summary AND content (raw messages).
        # Summary-only FTS missed actual facts — summaries say "User asked about X"
        # but the content has the real information. Keyword search on content finds
        # specific names, terms, and facts that summaries abstract away.
        cursor = await self._db.execute(
            "SELECT value FROM app_metadata WHERE key = 'fts5_v2_done'"
        )
        row = await cursor.fetchone()
        if not row:
            logger.info("Migrating FTS5 to index both summary and content...")
            try:
                # Drop old FTS table and triggers
                await self._db.execute("DROP TRIGGER IF EXISTS memories_fts_insert")
                await self._db.execute("DROP TRIGGER IF EXISTS memories_fts_update")
                await self._db.execute("DROP TRIGGER IF EXISTS memories_fts_delete")
                await self._db.execute("DROP TABLE IF EXISTS memories_fts")
                # Create new FTS5 table with both columns
                await self._db.execute("""
                    CREATE VIRTUAL TABLE memories_fts USING fts5(
                        summary, content, content=memories, content_rowid=id
                    )
                """)
                # Create new triggers for both columns
                await self._db.execute("""
                    CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories
                    BEGIN
                        INSERT INTO memories_fts(rowid, summary, content)
                        VALUES (NEW.id, COALESCE(NEW.summary, ''), COALESCE(NEW.content, ''));
                    END
                """)
                await self._db.execute("""
                    CREATE TRIGGER memories_fts_update AFTER UPDATE OF summary, content ON memories
                    BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, summary, content)
                        VALUES('delete', OLD.id, COALESCE(OLD.summary, ''), COALESCE(OLD.content, ''));
                        INSERT INTO memories_fts(rowid, summary, content)
                        VALUES (NEW.id, COALESCE(NEW.summary, ''), COALESCE(NEW.content, ''));
                    END
                """)
                await self._db.execute("""
                    CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories
                    BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, summary, content)
                        VALUES('delete', OLD.id, COALESCE(OLD.summary, ''), COALESCE(OLD.content, ''));
                    END
                """)
                # Backfill all existing rows
                await self._db.execute("""
                    INSERT INTO memories_fts(rowid, summary, content)
                    SELECT id, COALESCE(summary, ''), COALESCE(content, '')
                    FROM memories
                """)
                await self._db.execute(
                    "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('fts5_v2_done', '1')"
                )
                logger.info("FTS5 v2 migration complete — both summary and content indexed")
            except Exception as e:
                logger.error("FTS5 v2 migration failed: %s", e)

        # v3: Fix FTS5 column name (raw_content → content). The v2 migration
        # may have run with the wrong column name in the schema. Force rebuild.
        cursor = await self._db.execute(
            "SELECT value FROM app_metadata WHERE key = 'fts5_v3_done'"
        )
        if not (await cursor.fetchone()):
            logger.info("FTS5 v3: rebuilding to fix column name...")
            try:
                await self._db.execute("DROP TRIGGER IF EXISTS memories_fts_insert")
                await self._db.execute("DROP TRIGGER IF EXISTS memories_fts_update")
                await self._db.execute("DROP TRIGGER IF EXISTS memories_fts_delete")
                await self._db.execute("DROP TABLE IF EXISTS memories_fts")
                await self._db.execute("""
                    CREATE VIRTUAL TABLE memories_fts USING fts5(
                        summary, content, content=memories, content_rowid=id
                    )
                """)
                await self._db.execute("""
                    CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories
                    BEGIN
                        INSERT INTO memories_fts(rowid, summary, content)
                        VALUES (NEW.id, COALESCE(NEW.summary, ''), COALESCE(NEW.content, ''));
                    END
                """)
                await self._db.execute("""
                    CREATE TRIGGER memories_fts_update AFTER UPDATE OF summary, content ON memories
                    BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, summary, content)
                        VALUES('delete', OLD.id, COALESCE(OLD.summary, ''), COALESCE(OLD.content, ''));
                        INSERT INTO memories_fts(rowid, summary, content)
                        VALUES (NEW.id, COALESCE(NEW.summary, ''), COALESCE(NEW.content, ''));
                    END
                """)
                await self._db.execute("""
                    CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories
                    BEGIN
                        INSERT INTO memories_fts(memories_fts, rowid, summary, content)
                        VALUES('delete', OLD.id, COALESCE(OLD.summary, ''), COALESCE(OLD.content, ''));
                    END
                """)
                await self._db.execute("""
                    INSERT INTO memories_fts(rowid, summary, content)
                    SELECT id, COALESCE(summary, ''), COALESCE(content, '')
                    FROM memories
                """)
                await self._db.execute(
                    "INSERT OR REPLACE INTO app_metadata (key, value) VALUES ('fts5_v3_done', '1')"
                )
                logger.info("FTS5 v3 rebuild complete — keyword search restored")
            except Exception as e:
                logger.error("FTS5 v3 rebuild failed: %s", e)

        await self._db.commit()

    async def commit(self):
        """Explicitly commit the current transaction."""
        await self._db.commit()

    async def rollback(self):
        """Explicitly roll back the current transaction."""
        await self._db.rollback()

    async def close(self):
        """Close the database connection."""
        if self._db:
            try:
                await self._db.execute("PRAGMA optimize")
            except Exception:
                pass  # best-effort — don't block shutdown
            await self._db.close()
            self._db = None

    # --- Sessions ---

    async def create_session(self, title: str = "New Session", project: Optional[str] = None,
                             created_at: Optional[datetime] = None,
                             external_id: Optional[str] = None,
                             external_updated_at: Optional[float] = None) -> int:
        """Create a new session and return its ID.

        external_id / external_updated_at track the source conversation when
        a session is created from an import (ChatGPT/Claude conversation id +
        its last-updated timestamp), enabling dedup and grown-conversation
        detection on periodic re-imports.
        """
        ts = (created_at or datetime.now(timezone.utc)).isoformat()
        cursor = await self._db.execute(
            "INSERT INTO sessions (title, project, created_at, last_active, "
            "external_id, external_updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (title, project, ts, ts, external_id, external_updated_at),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_session(self, session_id: int, **kwargs):
        """Update session fields."""
        allowed = {"title", "summary", "project", "last_active", "is_archived", "message_count", "metadata_json"}
        set_clause, values = _safe_set_clause(kwargs, allowed)
        if not set_clause:
            return
        values.append(session_id)
        await self._db.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
        await self._db.commit()

    async def update_session_project(self, session_id: int, project: str) -> None:
        """Update a session's project tag."""
        await self._db.execute(
            "UPDATE sessions SET project = ? WHERE id = ?", (project, session_id),
        )
        await self._db.commit()

    async def get_session_notes(self, session_id: int) -> dict[str, str]:
        """Load session notes from metadata_json. Returns {name: content}."""
        cursor = await self._db.execute(
            "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,),
        )
        row = await cursor.fetchone()
        if not row or not row["metadata_json"]:
            return {}
        try:
            import json
            metadata = json.loads(row["metadata_json"])
            return metadata.get("notes", {})
        except (json.JSONDecodeError, TypeError):
            return {}

    async def save_session_notes(self, session_id: int, notes: dict[str, str]) -> None:
        """Save notes dict to sessions.metadata_json, merging with existing metadata."""
        import json
        # Read existing metadata
        cursor = await self._db.execute(
            "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,),
        )
        row = await cursor.fetchone()
        metadata = {}
        if row and row["metadata_json"]:
            try:
                metadata = json.loads(row["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        metadata["notes"] = notes
        await self._db.execute(
            "UPDATE sessions SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata), session_id),
        )
        await self._db.commit()

    async def clear_session_notes(self, session_id: int) -> None:
        """Remove notes from session metadata."""
        import json
        cursor = await self._db.execute(
            "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,),
        )
        row = await cursor.fetchone()
        if not row or not row["metadata_json"]:
            return
        try:
            metadata = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            return
        metadata.pop("notes", None)
        await self._db.execute(
            "UPDATE sessions SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata) if metadata else None, session_id),
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

    async def get_session_by_external_id(
        self, external_id: str,
    ) -> tuple[int | None, int, float | None]:
        """Look up a session by its source conversation id.

        Returns (session_id, message_count, external_updated_at) or
        (None, 0, None) if no session was imported from that conversation.
        """
        cursor = await self._db.execute(
            "SELECT id, message_count, external_updated_at FROM sessions "
            "WHERE external_id = ? LIMIT 1",
            (external_id,),
        )
        row = await cursor.fetchone()
        if row:
            return row[0], row[1] or 0, row[2]
        return None, 0, None

    async def save_raw_memory(
        self, session_id: int, role: str, content: str,
        timestamp: str | None = None,
    ) -> int:
        """Persist a raw message as an unprocessed memory. Returns the memory ID.

        Creates a memories row with is_processed=0 and summary=NULL.
        The memory pipeline will later update this row with summary, rank,
        importance, and set is_processed=1.  If the app crashes before
        processing, get_unprocessed_memories() finds these on next startup.
        """
        cursor = await self._db.execute(
            """INSERT INTO memories
               (session_id, role, content, summary, timestamp, is_processed)
               VALUES (?, ?, ?, NULL, COALESCE(?, CURRENT_TIMESTAMP), 0)""",
            (session_id, role, content, timestamp),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def mark_memory_processed(self, memory_id: int):
        """Mark a memory as successfully processed through the pipeline."""
        await self._db.execute(
            "UPDATE memories SET is_processed = 1 WHERE id = ?",
            (memory_id,),
        )
        await self._db.commit()

    async def get_unprocessed_memories(
        self, limit: int = 100,
    ) -> list[dict]:
        """Get memories that failed to process (for startup sweep).

        Only returns user/assistant messages (skips system messages).
        """
        cursor = await self._db.execute(
            """SELECT id, session_id, role, content, timestamp
               FROM memories
               WHERE is_processed = 0 AND role IN ('user', 'assistant')
               ORDER BY id ASC LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in await cursor.fetchall()]

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

    async def delete_empty_sessions(self, min_age_hours: int = 24) -> int:
        """Delete sessions with zero memories that are older than min_age_hours.

        Also cleans up any orphaned session_messages rows for those sessions.
        Returns count of deleted sessions.
        """
        cursor = await self._db.execute("""
            SELECT s.id FROM sessions s
            WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.session_id = s.id)
              AND s.created_at < datetime('now', ?)
        """, (f'-{min_age_hours} hours',))
        ids = [r[0] for r in await cursor.fetchall()]
        if not ids:
            return 0

        placeholders = ",".join("?" * len(ids))
        # Delete from all child tables that reference sessions(id)
        child_tables = [
            ("turn_events", "session_id"),
            ("session_messages", "session_id"),
            ("session_reflections", "session_id"),
            ("friction_log", "session_id"),
            ("task_plans", "session_id"),
            ("background_tasks", "session_id"),
            ("tool_approvals", "session_id"),
            ("follow_ups", "source_session"),
        ]
        for table, col in child_tables:
            try:
                await self._db.execute(
                    f"DELETE FROM {table} WHERE {col} IN ({placeholders})", ids,
                )
            except Exception:
                pass  # table may not exist in older schemas
        await self._db.execute(
            f"DELETE FROM sessions WHERE id IN ({placeholders})", ids,
        )
        await self._db.commit()
        return len(ids)

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
        """Get sessions that have data but no summary (need backfill).

        Returns dicts with {id, title, message_count} ordered oldest first.
        """
        cursor = await self._db.execute(
            """SELECT s.id, s.title,
                      (SELECT COUNT(*) FROM memories m
                       WHERE m.session_id = s.id AND m.is_archived = 0) as message_count
               FROM sessions s
               WHERE (s.summary IS NULL OR s.summary = '')
                 AND s.is_archived = 0
               GROUP BY s.id
               HAVING message_count > 0
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
        allowed = {"summary", "rank", "importance", "is_archived", "metadata_json", "memory_type", "is_processed"}
        set_clause, values = _safe_set_clause(kwargs, allowed)
        if not set_clause:
            return
        values.append(memory_id)
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
            rank=max(row["rank"] or 1, 1),  # clamp to min 1 (valid range 1-5)
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
        """Increment access_count and set last_accessed for retrieved memories.

        Best-effort: uses BEGIN IMMEDIATE to fail fast if the write lock is
        held (e.g. by the background worker), rather than blocking the search
        response for up to 60s waiting for the lock.
        """
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute("BEGIN IMMEDIATE")
            await self._db.executemany(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                [(now, mid) for mid in memory_ids],
            )
            await self._db.commit()
        except Exception:
            try:
                await self._db.rollback()
            except Exception:
                pass
            raise

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

    # Stop words — too common to be useful in FTS5 keyword search.
    # FTS5 defaults to implicit AND, so "how does search work" requires ALL
    # words in one summary. With OR semantics + stop word removal, each
    # meaningful keyword contributes independently.
    _FTS5_STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "their", "this", "that", "these", "those",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
        "how", "what", "when", "where", "who", "why", "which",
        "not", "no", "nor", "but", "if", "so", "just", "very",
        "also", "than", "then", "too", "up", "out", "some", "all", "any",
    })

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize query for FTS5 with OR semantics.

        FTS5 defaults to implicit AND — "how does search work" requires ALL
        four words in one document. This produced zero results on virtually
        every conversational query. Fix: join tokens with OR so each keyword
        contributes independently. Stop words are removed to reduce noise.
        """
        # Whitelist approach: keep only alphanumeric characters, replace
        # everything else with a space. A blacklist inevitably misses an
        # FTS5 operator char (e.g. the backtick), which produces a
        # "syntax error near ..." failure. Keeping only alnum (Unicode-aware,
        # so non-Latin scripts survive) eliminates that entire class of bug.
        sanitized = ''.join(c if c.isalnum() else ' ' for c in query)
        # Remove FTS5 reserved keywords and stop words
        tokens = sanitized.split()
        tokens = [
            t for t in tokens
            if t.upper() not in SQLiteStore._FTS5_RESERVED
            and t.lower() not in SQLiteStore._FTS5_STOP_WORDS
            and len(t) >= 2
        ]
        # OR semantics — each keyword contributes independently
        return ' OR '.join(tokens) if tokens else ''

    async def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search on memory summaries using FTS5.

        Returns list of {id, fts_rank} dicts sorted by relevance.
        Uses OR semantics so partial keyword matches contribute.
        """
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []
        try:
            cursor = await self._db.execute(
                """SELECT rowid AS id, rank AS fts_rank
                   FROM memories_fts WHERE memories_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
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

    async def get_unsummarized_memories(self, limit: int = 0) -> list[Memory]:
        """Get memories where summary = content (LLM summarization failed during import).

        Only returns active memories with content > 200 chars (short messages
        don't benefit from summarization). limit=0 means no limit.
        """
        query = (
            "SELECT * FROM memories WHERE summary = content AND is_archived = 0 "
            "AND length(content) > 200 ORDER BY RANDOM()"
        )
        if limit > 0:
            query += f" LIMIT {limit}"
        cursor = await self._db.execute(query)
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def update_lesson_scores(self, lesson_id: int, rank: int, importance: float):
        """Update rank and importance for a lesson."""
        await self._db.execute(
            "UPDATE lessons SET rank = ?, importance = ? WHERE id = ?",
            (rank, importance, lesson_id),
        )
        await self._db.commit()

    async def get_unscored_lessons(self) -> list[Lesson]:
        """Get lessons still at default scores (rank=3, importance=0.5)."""
        cursor = await self._db.execute(
            "SELECT * FROM lessons WHERE rank = 3 AND importance = 0.5 "
            "ORDER BY timestamp DESC"
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

    async def increment_lesson_hits(self, lesson_ids: list[int]):
        """Increment hit_count and set last_accessed for used lessons."""
        if not lesson_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" * len(lesson_ids))
        await self._db.execute(
            f"UPDATE lessons SET hit_count = hit_count + 1, last_accessed = ? "
            f"WHERE id IN ({placeholders})",
            [now] + lesson_ids,
        )
        await self._db.commit()

    async def get_sessions_missing_lessons(self, limit: int = 50) -> list[dict]:
        """Find sessions with 5+ messages but no lessons extracted."""
        cursor = await self._db.execute("""
            SELECT s.id, s.project, COUNT(m.id) as msg_count
            FROM sessions s
            JOIN memories m ON m.session_id = s.id AND m.is_archived = 0
            LEFT JOIN lessons l ON l.source_session_id = s.id
            WHERE l.id IS NULL
              AND s.is_archived = 0
            GROUP BY s.id
            HAVING COUNT(m.id) >= 5
            ORDER BY s.id DESC
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in await cursor.fetchall()]

    async def get_session_messages_for_lesson(self, session_id: int, include_archived: bool = False) -> list[dict]:
        """Get conversation messages for lesson/reflection extraction.

        Reads from the memories table (which stores raw content for all sessions,
        both live and imported) rather than session_messages (which is only for
        crash recovery and may not exist for imported sessions).

        Args:
            include_archived: If True, also return archived memories. Useful for
                reflections on sessions whose memories were consolidated.
        """
        if include_archived:
            cursor = await self._db.execute("""
                SELECT role, content FROM memories
                WHERE session_id = ?
                ORDER BY id
            """, (session_id,))
        else:
            cursor = await self._db.execute("""
                SELECT role, content FROM memories
                WHERE session_id = ? AND is_archived = 0
                ORDER BY id
            """, (session_id,))
        return [dict(r) for r in await cursor.fetchall()]

    # --- Session Reflections ---

    async def create_session_reflection(
        self,
        session_id: int,
        effectiveness: str,
        reflection_text: str,
        technical_insights: str | None = None,
        process_insights: str | None = None,
        what_worked: str | None = None,
        what_didnt_work: str | None = None,
    ) -> int:
        """Store a session reflection. Returns the reflection ID.

        UNIQUE constraint on session_id makes this resume-safe — re-running
        on an already-reflected session raises IntegrityError.
        """
        cursor = await self._db.execute(
            """INSERT INTO session_reflections
               (session_id, effectiveness, reflection_text,
                technical_insights, process_insights, what_worked, what_didnt_work)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, effectiveness, reflection_text,
             technical_insights, process_insights, what_worked, what_didnt_work),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_session_reflection(self, session_id: int) -> dict | None:
        """Get a session reflection by session ID."""
        cursor = await self._db.execute(
            "SELECT * FROM session_reflections WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_sessions_missing_reflections(self, limit: int = 20) -> list[dict]:
        """Find sessions eligible for reflection.

        Criteria: has a summary, not archived, no existing reflection,
        at least 1 memory (including archived — consolidated sessions still
        have valid data for reflection).
        """
        cursor = await self._db.execute("""
            SELECT s.id, s.summary, s.project, s.title,
                   (SELECT COUNT(*) FROM memories m
                    WHERE m.session_id = s.id) as msg_count
            FROM sessions s
            LEFT JOIN session_reflections sr ON sr.session_id = s.id
            WHERE sr.id IS NULL
              AND s.summary IS NOT NULL
              AND s.summary != ''
              AND s.is_archived = 0
            GROUP BY s.id
            HAVING msg_count >= 1
            ORDER BY s.id DESC
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in await cursor.fetchall()]

    async def get_recent_reflections(
        self, limit: int = 10, project: str | None = None,
    ) -> list[dict]:
        """Get recent session reflections, optionally filtered by project."""
        if project:
            cursor = await self._db.execute("""
                SELECT sr.*, s.title, s.project
                FROM session_reflections sr
                JOIN sessions s ON s.id = sr.session_id
                WHERE s.project = ?
                ORDER BY sr.created_at DESC
                LIMIT ?
            """, (project, limit))
        else:
            cursor = await self._db.execute("""
                SELECT sr.*, s.title, s.project
                FROM session_reflections sr
                JOIN sessions s ON s.id = sr.session_id
                ORDER BY sr.created_at DESC
                LIMIT ?
            """, (limit,))
        return [dict(r) for r in await cursor.fetchall()]

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
        set_clause, values = _safe_set_clause(kwargs, allowed)
        if not set_clause:
            return
        values.append(core_memory_id)
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
        set_clause, values = _safe_set_clause(fields, allowed)
        if not set_clause:
            return
        values.append(name)
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
        # Convert enums to their values
        for k, v in list(kwargs.items()):
            if hasattr(v, "value"):
                kwargs[k] = v.value
        kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause, values = _safe_set_clause(kwargs, allowed)
        if not set_clause:
            return
        values.append(plan_id)
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
        for k, v in list(kwargs.items()):
            if hasattr(v, "value"):
                kwargs[k] = v.value
        kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause, values = _safe_set_clause(kwargs, allowed)
        if not set_clause:
            return
        values.append(step_id)
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
        for k, v in list(kwargs.items()):
            if hasattr(v, "value"):
                kwargs[k] = v.value
        kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause, values = _safe_set_clause(kwargs, allowed)
        if not set_clause:
            return
        values.append(task_id)
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

    async def get_or_create_entity(self, name: str, entity_type: str = "concept", *, skip_commit: bool = False) -> int:
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
        if not skip_commit:
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

    async def create_entity_mention(self, entity_id: int, memory_id: int, *, skip_commit: bool = False):
        """Record that an entity is mentioned in a memory."""
        await self._db.execute(
            "INSERT OR IGNORE INTO entity_mentions (entity_id, memory_id) VALUES (?, ?)",
            (entity_id, memory_id),
        )
        if not skip_commit:
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

    async def mark_entities_extracted(self, memory_ids: list[int], *, skip_commit: bool = False):
        """Set entities_extracted_at timestamp for processed memories."""
        if not memory_ids:
            return
        now = datetime.now(timezone.utc).isoformat()
        await self._db.executemany(
            "UPDATE memories SET entities_extracted_at = ? WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )
        if not skip_commit:
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
        *, skip_commit: bool = False,
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
            if not skip_commit:
                await self._db.commit()
            return cursor.lastrowid if cursor.lastrowid else None
        except Exception as e:
            logger.warning("Failed to create temporal relationship: %s", e)
            return None

    async def expire_contradicting_relationships(
        self, subject_id: int, object_id: int,
        contradicting_predicates: dict[str, list[str]],
        new_predicate: str, new_relationship_id: int,
        *, skip_commit: bool = False,
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
        if not skip_commit:
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

    async def entity_id_exists(self, entity_id: int) -> bool:
        """Check if an entity ID exists in the entities table."""
        cursor = await self._db.execute(
            "SELECT 1 FROM entities WHERE id = ? LIMIT 1", (entity_id,),
        )
        return await cursor.fetchone() is not None

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

    # --- Tool Approval Audit Trail ---

    async def log_tool_approval(self, session_id: int | None, tool_name: str,
                                arguments: dict, approved: bool):
        """Record a tool approval/denial decision."""
        import json as _json
        args_str = _json.dumps(arguments, default=str)[:2000]
        await self._db.execute(
            "INSERT INTO tool_approvals (session_id, tool_name, arguments_json, approved) "
            "VALUES (?, ?, ?, ?)",
            (session_id, tool_name, args_str, approved),
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

    # --- Follow-Up Queue ---

    async def add_follow_up(
        self, content: str, session_id: int | None = None,
        project: str | None = None, due_hint: str | None = None,
    ) -> int:
        """Add a follow-up item. Returns the new ID."""
        cursor = await self._db.execute(
            "INSERT INTO follow_ups (content, source_session, project, due_hint) "
            "VALUES (?, ?, ?, ?)",
            (content, session_id, project, due_hint),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_pending_follow_ups(
        self, project: str | None = None, limit: int = 20,
    ) -> list[dict]:
        """Get pending follow-up items, optionally filtered by project."""
        if project:
            cursor = await self._db.execute(
                "SELECT * FROM follow_ups WHERE status = 'pending' "
                "AND (project = ? OR project IS NULL) "
                "ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM follow_ups WHERE status = 'pending' "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def resolve_follow_up(
        self, follow_up_id: int, session_id: int | None = None,
    ) -> bool:
        """Mark a follow-up as resolved. Returns True if found and updated."""
        cursor = await self._db.execute(
            "UPDATE follow_ups SET status = 'resolved', resolved_session = ?, "
            "resolved_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'pending'",
            (session_id, follow_up_id),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def dismiss_follow_up(self, follow_up_id: int) -> bool:
        """Dismiss a follow-up (no longer relevant). Returns True if found."""
        cursor = await self._db.execute(
            "UPDATE follow_ups SET status = 'dismissed', "
            "resolved_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'pending'",
            (follow_up_id,),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_all_follow_ups(self, limit: int = 50) -> list[dict]:
        """Get all follow-ups (any status) for display."""
        cursor = await self._db.execute(
            "SELECT * FROM follow_ups ORDER BY "
            "CASE status WHEN 'pending' THEN 0 WHEN 'resolved' THEN 1 ELSE 2 END, "
            "created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # --- Friction Log ---

    async def add_friction_entry(
        self, session_id: int | None, source: str,
        category: str, description: str,
    ) -> int:
        """Log a friction item. Returns the new ID."""
        cursor = await self._db.execute(
            "INSERT INTO friction_log (session_id, source, category, description) "
            "VALUES (?, ?, ?, ?)",
            (session_id, source, category, description),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_friction_entries(
        self, unreviewed_only: bool = False, limit: int = 50,
    ) -> list[dict]:
        """Get friction log entries, newest first."""
        where = "WHERE is_reviewed = 0" if unreviewed_only else ""
        cursor = await self._db.execute(
            f"SELECT * FROM friction_log {where} "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def mark_friction_reviewed(self, friction_ids: list[int]) -> int:
        """Mark friction entries as reviewed. Returns count updated."""
        if not friction_ids:
            return 0
        placeholders = ",".join("?" * len(friction_ids))
        cursor = await self._db.execute(
            f"UPDATE friction_log SET is_reviewed = 1 WHERE id IN ({placeholders})",
            friction_ids,
        )
        await self._db.commit()
        return cursor.rowcount

    async def get_sessions_missing_friction_analysis(
        self, limit: int = 20,
    ) -> list[dict]:
        """Get sessions that have reflections but no friction analysis yet."""
        cursor = await self._db.execute(
            """SELECT s.id, s.summary, s.project, s.message_count
               FROM sessions s
               JOIN session_reflections sr ON sr.session_id = s.id
               WHERE s.summary IS NOT NULL
               AND s.message_count >= 5
               AND sr.reflection_text != 'SKIP'
               AND s.id NOT IN (
                   SELECT DISTINCT session_id FROM friction_log
                   WHERE session_id IS NOT NULL AND source = 'nightly'
               )
               ORDER BY s.id DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
