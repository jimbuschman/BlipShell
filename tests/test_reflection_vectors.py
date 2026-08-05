"""Session-reflection embeddings: storage, migration, and retrieval.

Regression for the 2026-08-04 deep-dive finding: reflection vectors were
written into vec_lessons at rowid = reflection_id + 100000, but lesson-search
enrichment joins the lessons table — so every reflection embedding was
unretrievable (and collided with lesson ids past 100000). Reflections now get
their own vec table, a startup migration rescues stranded vectors, and
MemorySearch.search_lessons surfaces reflections alongside lessons.

Real SQLite + real sqlite-vec, stubbed embeddings, no Ollama.
"""

import struct
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from blipshell.memory.search import MemorySearch
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.vector_store import VectorStore
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Lesson

DIM = 8

_VECS = {
    "reflection about the entity graph cleanup": [1.0, 0, 0, 0, 0, 0, 0, 0],
    "lesson about testing before committing": [0, 1.0, 0, 0, 0, 0, 0, 0],
    "entity graph": [0.9, 0.1, 0, 0, 0, 0, 0, 0],       # near the reflection
    "testing discipline": [0.1, 0.9, 0, 0, 0, 0, 0, 0],  # near the lesson
}


def _embed(text):
    return _VECS.get(text, [0, 0, 0, 0, 0, 0, 0, 1.0])


def _blob(vec):
    return struct.pack(f"{len(vec)}f", *vec)


@pytest.fixture
async def stores(tmp_path):
    db_path = str(tmp_path / "refl.db")
    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()

    vectors = VectorStore(db_path, embedding_dim=DIM)
    vectors.initialize()
    vectors._embed = _embed
    vectors._ollama_client = MagicMock()

    yield sqlite, vectors, db_path

    vectors.close()
    await sqlite.close()


async def _make_reflection(sqlite, project=None):
    sid = await sqlite.create_session(title="Refl session")
    if project:
        await sqlite.update_session(sid, project=project)
    rid = await sqlite.create_session_reflection(
        session_id=sid,
        effectiveness="effective",
        reflection_text="reflection about the entity graph cleanup",
    )
    return sid, rid


class TestReflectionRoundTrip:
    async def test_add_then_search(self, stores):
        sqlite, vectors, _ = stores
        _, rid = await _make_reflection(sqlite)
        vectors.add_reflection(rid, "reflection about the entity graph cleanup")

        results = vectors.search_reflections("entity graph", n_results=5)
        assert any(r["id"] == rid for r in results)
        hit = next(r for r in results if r["id"] == rid)
        assert hit["metadata"]["source"] == "reflection"
        assert "entity graph" in hit["document"]

    async def test_project_metadata_carried(self, stores):
        sqlite, vectors, _ = stores
        _, rid = await _make_reflection(sqlite, project="blipshell")
        vectors.add_reflection(rid, "reflection about the entity graph cleanup")

        results = vectors.search_reflections("entity graph", n_results=5)
        hit = next(r for r in results if r["id"] == rid)
        assert hit["metadata"]["project"] == "blipshell"


class TestStrandedVectorMigration:
    async def test_stranded_reflection_vector_is_rescued(self, stores):
        sqlite, vectors, db_path = stores
        _, rid = await _make_reflection(sqlite)

        # Recreate the legacy state: vector stranded in vec_lessons at +100000,
        # plus an orphan with no reflection row behind it.
        with vectors._lock:
            vectors._conn.execute(
                "INSERT INTO vec_lessons(rowid, embedding) VALUES (?, ?)",
                [rid + 100000, _blob(_VECS["reflection about the entity graph cleanup"])],
            )
            vectors._conn.execute(
                "INSERT INTO vec_lessons(rowid, embedding) VALUES (?, ?)",
                [100000 + 99999, _blob([0.5] * DIM)],
            )
            vectors._conn.commit()
        vectors.close()

        # Next startup runs the migration
        v2 = VectorStore(db_path, embedding_dim=DIM)
        v2.initialize()
        v2._embed = _embed
        v2._ollama_client = MagicMock()
        try:
            results = v2.search_reflections("entity graph", n_results=5)
            assert any(r["id"] == rid for r in results), (
                "stranded reflection vector was not rescued into vec_reflections"
            )
            with v2._lock:
                leftovers = v2._conn.execute(
                    "SELECT rowid FROM vec_lessons WHERE rowid > 100000"
                ).fetchall()
            assert leftovers == []       # stranded + orphan both cleared
        finally:
            v2.close()

    async def test_migration_idempotent_on_clean_store(self, stores):
        _, vectors, db_path = stores
        vectors.close()
        v2 = VectorStore(db_path, embedding_dim=DIM)
        v2.initialize()      # nothing stranded — must not raise
        v2.close()


class TestLessonSearchIncludesReflections:
    async def test_merged_results_carry_both_sources(self, stores):
        sqlite, vectors, _ = stores

        lesson_id = await sqlite.create_lesson(Lesson(
            content="lesson about testing before committing",
            summary="test before commit",
            timestamp=datetime.now(timezone.utc),
            rank=4, importance=0.8,
        ))
        vectors.add_lesson(lesson_id, "lesson about testing before committing")

        _, rid = await _make_reflection(sqlite)
        vectors.add_reflection(rid, "reflection about the entity graph cleanup")

        search = MemorySearch(
            sqlite=sqlite, vectors=vectors, router=MagicMock(),
            config=MemoryConfig(),
        )
        results = await search.search_lessons("entity graph", n_results=10)
        sources = {r["metadata"].get("source") for r in results}
        assert "reflection" in sources
        assert "lesson" in sources

    async def test_reflection_hits_not_counted_as_lesson_hits(self, stores):
        sqlite, vectors, _ = stores

        lesson_id = await sqlite.create_lesson(Lesson(
            content="lesson about testing before committing",
            summary="test before commit",
            timestamp=datetime.now(timezone.utc),
            rank=4, importance=0.8,
        ))
        vectors.add_lesson(lesson_id, "lesson about testing before committing")

        # Reflection whose id COLLIDES with the lesson id — counting it as a
        # lesson hit would inflate the wrong lesson's usage stats.
        _, rid = await _make_reflection(sqlite)
        vectors.add_reflection(rid, "reflection about the entity graph cleanup")

        search = MemorySearch(
            sqlite=sqlite, vectors=vectors, router=MagicMock(),
            config=MemoryConfig(),
        )
        await search.search_lessons("entity graph", n_results=10)

        row = await (await sqlite._db.execute(
            "SELECT hit_count FROM lessons WHERE id = ?", [lesson_id],
        )).fetchone()
        lesson_hits = row[0] if row and row[0] is not None else 0
        # The lesson may legitimately register a hit; the reflection must not
        # have added a second one to the same id.
        assert lesson_hits <= 1
