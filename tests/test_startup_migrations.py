"""Startup migrations run against the live database on every launch.

Both of these are one-way operations found in review to have a latent
failure mode: the reflection-vector move treats ANY vec_lessons rowid over
100000 as a stranded reflection (a genuine lesson crossing that id would
have its vector eaten), and the entity_relationships rebuild autocommits its
CREATE TABLE before the copying transaction opens, so a failed copy left
debris that wedged every retry on "table already exists".
"""

import sqlite3
import struct

import pytest

from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.vector_store import VectorStore

DIM = 8


def _blob(i):
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return struct.pack(f"{DIM}f", *v)


class TestReflectionVectorMigration:
    def _open(self, path):
        vs = VectorStore(str(path), embedding_dim=DIM)
        vs.initialize()
        return vs

    def test_stranded_reflection_is_moved(self, tmp_path):
        path = tmp_path / "m.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE lessons (id INTEGER PRIMARY KEY, lesson TEXT)")
        conn.execute(
            "CREATE TABLE session_reflections (id INTEGER PRIMARY KEY, summary TEXT)"
        )
        conn.execute("INSERT INTO session_reflections (id, summary) VALUES (7, 's')")
        conn.commit()
        conn.close()

        vs = self._open(path)     # creates vec tables, no stranded rows yet
        with vs._lock:
            vs._conn.execute(
                "INSERT INTO vec_lessons(rowid, embedding) VALUES (?, ?)",
                [100007, _blob(1)],
            )
            vs._conn.commit()
        vs.close()

        vs = self._open(path)     # migration fires on this startup
        with vs._lock:
            moved = vs._conn.execute(
                "SELECT rowid FROM vec_reflections"
            ).fetchall()
            left = vs._conn.execute(
                "SELECT rowid FROM vec_lessons WHERE rowid > 100000"
            ).fetchall()
        vs.close()

        assert [r[0] for r in moved] == [7]
        assert left == []

    def test_a_genuine_lesson_over_100000_is_left_alone(self, tmp_path):
        """The migration runs at EVERY startup. The day a real lesson's id
        crosses 100000 its vector must not be eaten or moved onto a
        reflection id."""
        path = tmp_path / "m.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE lessons (id INTEGER PRIMARY KEY, lesson TEXT)")
        conn.execute(
            "CREATE TABLE session_reflections (id INTEGER PRIMARY KEY, summary TEXT)"
        )
        conn.execute("INSERT INTO lessons (id, lesson) VALUES (100001, 'L')")
        # Reflection 1 exists — without the guard, lesson 100001's vector
        # would be moved onto it, overwriting a real reflection's embedding.
        conn.execute("INSERT INTO session_reflections (id, summary) VALUES (1, 'r')")
        conn.commit()
        conn.close()

        vs = self._open(path)
        with vs._lock:
            vs._conn.execute(
                "INSERT INTO vec_lessons(rowid, embedding) VALUES (?, ?)",
                [100001, _blob(2)],
            )
            vs._conn.commit()
        vs.close()

        vs = self._open(path)
        with vs._lock:
            lesson_vec = vs._conn.execute(
                "SELECT rowid FROM vec_lessons WHERE rowid = 100001"
            ).fetchone()
            reflections = vs._conn.execute(
                "SELECT rowid FROM vec_reflections"
            ).fetchall()
        vs.close()

        assert lesson_vec is not None, "a genuine lesson's vector was eaten"
        assert reflections == [], (
            "the lesson's vector was moved onto a reflection id"
        )


class TestEntityRelationshipsRebuild:
    def _legacy_db(self, path):
        """A DB whose entity_relationships still has the inline UNIQUE.

        The referenced entities must exist: the store runs with foreign_keys
        ON, so the rebuild's INSERT..SELECT enforces them exactly as it does
        against the real database.
        """
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT DEFAULT 'concept'
            )
        """)
        conn.execute("INSERT INTO entities (id, name) VALUES (1, 'user')")
        conn.execute("INSERT INTO entities (id, name) VALUES (2, 'acme')")
        conn.execute("""
            CREATE TABLE entity_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL,
                predicate TEXT NOT NULL,
                object_id INTEGER NOT NULL,
                source_memory_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                valid_from DATETIME,
                expired_at DATETIME,
                expired_by INTEGER,
                UNIQUE(subject_id, predicate, object_id)
            )
        """)
        conn.execute(
            "INSERT INTO entity_relationships (subject_id, predicate, object_id) "
            "VALUES (1, 'works_at', 2)"
        )
        conn.commit()
        conn.close()

    async def test_rebuild_moves_uniqueness_to_active_rows(self, tmp_path):
        path = tmp_path / "r.db"
        self._legacy_db(path)

        store = SQLiteStore(str(path))
        await store.initialize()
        cursor = await store._db.execute(
            "SELECT sql FROM sqlite_master WHERE name='entity_relationships'"
        )
        sql = (await cursor.fetchone())["sql"]
        cursor = await store._db.execute(
            "SELECT COUNT(*) AS n FROM entity_relationships"
        )
        n = (await cursor.fetchone())["n"]
        await store.close()

        assert "UNIQUE(subject_id, predicate, object_id)" not in sql
        assert n == 1, "rows were lost in the rebuild"

    async def test_debris_from_a_failed_rebuild_does_not_wedge_the_retry(self, tmp_path):
        """CREATE TABLE autocommits before the copy's transaction opens, so a
        failed copy stranded entity_relationships_new — and every retry died
        on 'table already exists' while logging that it would retry."""
        path = tmp_path / "r.db"
        self._legacy_db(path)
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE entity_relationships_new (id INTEGER)")
        conn.commit()
        conn.close()

        store = SQLiteStore(str(path))
        await store.initialize()
        cursor = await store._db.execute(
            "SELECT sql FROM sqlite_master WHERE name='entity_relationships'"
        )
        sql = (await cursor.fetchone())["sql"]
        await store.close()

        assert "UNIQUE(subject_id, predicate, object_id)" not in sql, (
            "the rebuild wedged on debris from a previous failed attempt"
        )
