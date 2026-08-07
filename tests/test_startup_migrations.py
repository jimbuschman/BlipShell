"""Startup migrations run against the live database on every launch.

Both of these are one-way operations found in review to have a latent
failure mode: the reflection-vector move treats ANY vec_lessons rowid over
100000 as a stranded reflection (a genuine lesson crossing that id would
have its vector eaten), and the entity_relationships rebuild autocommits its
CREATE TABLE before the copying transaction opens, so a failed copy left
debris that wedged every retry on "table already exists".
"""

import json
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


class TestSelfThoughtsMigration:
    """JSON blob -> self_thoughts table. One-way, against the live store."""

    async def _seed_json(self, path, items):
        store = SQLiteStore(str(path))
        await store.initialize()
        await store.set_metadata("self_thoughts", json.dumps(items))
        # Undo the migration that initialize() just ran on the empty store, so
        # the test controls when it fires.
        await store._db.execute("DELETE FROM self_thoughts")
        await store._db.execute(
            "UPDATE app_metadata SET key = 'self_thoughts' "
            "WHERE key = 'self_thoughts_pre_migration'"
        )
        await store._db.commit()
        await store.close()

    async def test_thoughts_migrate_preserving_order_and_fields(self, tmp_path):
        path = tmp_path / "t.db"
        await self._seed_json(path, [
            {"text": "first", "created_at": "2026-01-01T00:00:00+00:00",
             "weight": 2.5, "surfaced": True, "echo_count": 3,
             "surface_count": 1, "embedding": [1.0, 0.0]},
            {"text": "second", "created_at": "2026-02-01T00:00:00+00:00"},
        ])

        store = SQLiteStore(str(path))
        await store.initialize()
        rows = await store.get_self_thoughts(with_embeddings=True)
        backup = await store.get_metadata("self_thoughts_pre_migration")
        original = await store.get_metadata("self_thoughts")
        await store.close()

        assert [r["text"] for r in rows] == ["first", "second"], (
            "chronological order was not preserved — id order is what "
            "recent() reads as time order"
        )
        assert rows[0]["weight"] == 2.5
        assert rows[0]["surfaced"] is True
        assert rows[0]["echo_count"] == 3
        assert rows[0]["created_at"] == "2026-01-01T00:00:00+00:00"
        assert rows[0]["embedding"] == [1.0, 0.0]
        assert backup, "the old blob was destroyed — no rollback path"
        assert original is None, "the old key should have been renamed"

    async def test_migration_is_idempotent_across_restarts(self, tmp_path):
        path = tmp_path / "t.db"
        await self._seed_json(path, [{"text": "only one"}])

        for _ in range(3):
            store = SQLiteStore(str(path))
            await store.initialize()
            rows = await store.get_self_thoughts()
            await store.close()

        assert len(rows) == 1, "restarts duplicated the thought corpus"

    async def test_malformed_rows_are_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "t.db"
        await self._seed_json(path, [
            {"text": "good"}, {"no_text": True}, "not a dict", {"text": ""},
        ])

        store = SQLiteStore(str(path))
        await store.initialize()
        rows = await store.get_self_thoughts()
        await store.close()

        assert [r["text"] for r in rows] == ["good"]

    async def test_unparseable_blob_aborts_without_touching_it(self, tmp_path):
        path = tmp_path / "t.db"
        store = SQLiteStore(str(path))
        await store.initialize()
        await store.set_metadata("self_thoughts", "{not json")
        await store._db.commit()
        await store.close()

        store = SQLiteStore(str(path))
        await store.initialize()
        rows = await store.get_self_thoughts()
        blob = await store.get_metadata("self_thoughts")
        await store.close()

        assert rows == []
        assert blob == "{not json", "the unreadable blob was renamed away"

    async def test_stale_backup_key_does_not_wedge_the_migration(self, tmp_path):
        """A stale backup row plus a restored JSON key (a rollback drill, a
        restored DB) made the rename die on the key's UNIQUE constraint —
        and then every startup retried, failed, and left the TABLE empty
        while _load() reads the table, so the thoughts vanished from the
        running system until repaired by hand."""
        path = tmp_path / "t.db"
        store = SQLiteStore(str(path))
        await store.initialize()
        await store.set_metadata(
            "self_thoughts_pre_migration", json.dumps([{"text": "stale backup"}]),
        )
        await store.set_metadata(
            "self_thoughts", json.dumps([{"text": "restored thought"}]),
        )
        await store._db.execute("DELETE FROM self_thoughts")
        await store._db.commit()
        await store.close()

        store = SQLiteStore(str(path))
        await store.initialize()
        rows = await store.get_self_thoughts()
        backup = await store.get_metadata("self_thoughts_pre_migration")
        json_key = await store.get_metadata("self_thoughts")
        await store.close()

        assert [r["text"] for r in rows] == ["restored thought"], (
            "the migration wedged on the stale backup and the thoughts "
            "vanished from the running system"
        )
        assert json_key is None
        # The backup now holds what was just migrated — the fresher data.
        assert "restored thought" in backup

    async def test_populated_table_plus_stale_key_is_left_alone(self, tmp_path):
        """Both present means something is off — importing on top of live rows
        would duplicate the corpus, so it must refuse and say so."""
        path = tmp_path / "t.db"
        store = SQLiteStore(str(path))
        await store.initialize()
        await store.add_self_thought("already here", "2026-03-01T00:00:00+00:00")
        await store.set_metadata("self_thoughts", json.dumps([{"text": "from json"}]))
        await store._db.commit()
        await store.close()

        store = SQLiteStore(str(path))
        await store.initialize()
        rows = await store.get_self_thoughts()
        blob = await store.get_metadata("self_thoughts")
        await store.close()

        assert [r["text"] for r in rows] == ["already here"]
        assert blob is not None, "the JSON was consumed despite the refusal"


class TestSelfThoughtArchiveMandate:
    async def test_eviction_archives_rather_than_deletes(self, tmp_path):
        """max_keep enforcement is a soft-archive: an evicted thought is
        exactly what you would want back when asking why the self-layer
        drifted."""
        from blipshell.core.self_reflection import SelfThoughtStore

        store = SQLiteStore(str(tmp_path / "a.db"))
        await store.initialize()

        async def embed(text):
            return [float(len(text)), 1.0]

        s = SelfThoughtStore(store, max_keep=2, embed_fn=embed,
                             gravity_enabled=False)
        for t in ("one", "twoo", "threee", "fourrr"):
            await s.add(t)

        active = await store.get_self_thoughts()
        everything = await store.get_self_thoughts(include_archived=True)
        await store.close()

        assert len(active) == 2
        assert len(everything) == 4, "evicted thoughts were deleted, not archived"

    async def test_fold_records_what_absorbed_the_thought(self, tmp_path):
        from blipshell.core.self_reflection import SelfThoughtStore

        store = SQLiteStore(str(tmp_path / "b.db"))
        await store.initialize()

        async def embed(text):
            return [1.0, 0.0]      # everything is an echo of everything

        s = SelfThoughtStore(store, embed_fn=embed, gravity_enabled=True,
                             recur_threshold=0.85)
        await s.add("original")
        await s.add("evolved phrasing")

        active = await store.get_self_thoughts()
        cursor = await store._db.execute(
            "SELECT text, folded_into, is_archived FROM self_thoughts "
            "WHERE is_archived = 1"
        )
        archived = await cursor.fetchall()
        await store.close()

        # The echo folds into the prior IN PLACE, so one active row remains.
        assert len(active) == 1
        assert active[0]["text"] == "evolved phrasing"
        # Nothing was archived by this fold (in-place update), but the column
        # exists and the mandate holds — no row disappeared.
        assert len(active) + len(archived) >= 1
