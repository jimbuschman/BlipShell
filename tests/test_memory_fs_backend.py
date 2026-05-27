"""Unit tests for MemoryFSBackend — real SQLiteStore + mocked VectorStore.

Lessons are read-only; core memories are writable with vector-store sync.
"""

from unittest.mock import MagicMock

import pytest

from blipshell.memory.fs_backend import FSError, MemoryFSBackend
from blipshell.memory.fs_paths import Tier, parse
from blipshell.models.memory import CoreMemory, Lesson


@pytest.fixture
def mock_vectors():
    v = MagicMock()
    v.add_core_memory = MagicMock()
    v.delete_core_memory = MagicMock()
    return v


@pytest.fixture
async def backend(sqlite_store, mock_vectors):
    return MemoryFSBackend(sqlite_store, mock_vectors)


class TestListRoot:
    async def test_lists_six_tiers(self, backend):
        entries = await backend.list_root()
        paths = {e.path for e in entries}
        assert paths == {
            "/memories/lessons/",
            "/memories/core/",
            "/memories/digests/",
            "/memories/sessions/",
            "/memories/friction/",
            "/memories/notes/",
        }


class TestLessonsReadOnly:
    async def test_read_existing_lesson(self, backend, sqlite_store):
        lid = await sqlite_store.create_lesson(
            Lesson(content="Never dig straight down", project="mc")
        )
        path = parse(f"/memories/lessons/mc/{lid}-never.md")
        content = await backend.read(path)
        assert content == "Never dig straight down"

    async def test_list_lessons(self, backend, sqlite_store):
        await sqlite_store.create_lesson(Lesson(content="L1", project="mc"))
        await sqlite_store.create_lesson(Lesson(content="L2", project="mc"))
        entries = await backend.list_directory(parse("/memories/lessons/mc/"))
        assert len(entries) == 2

    async def test_create_blocked(self, backend):
        with pytest.raises(FSError, match="read-only"):
            await backend.create(parse("/memories/lessons/mc/"), "new lesson")

    async def test_edit_blocked(self, backend, sqlite_store):
        lid = await sqlite_store.create_lesson(Lesson(content="abc xyz", project="mc"))
        with pytest.raises(FSError, match="read-only"):
            await backend.replace_text(
                parse(f"/memories/lessons/mc/{lid}-x.md"), "xyz", "123"
            )

    async def test_delete_blocked(self, backend, sqlite_store):
        lid = await sqlite_store.create_lesson(Lesson(content="x", project="mc"))
        with pytest.raises(FSError, match="read-only"):
            await backend.delete(parse(f"/memories/lessons/mc/{lid}-x.md"))


class TestCoreWithVectorSync:
    async def test_create_embeds(self, backend, mock_vectors):
        canonical = await backend.create(parse("/memories/core/"), "User prefers terse.")
        assert canonical.tier == Tier.CORE
        assert canonical.file_id is not None
        # Vector sync: add_core_memory called with (id, content)
        mock_vectors.add_core_memory.assert_called_once()
        args = mock_vectors.add_core_memory.call_args[0]
        assert args[0] == canonical.file_id
        assert args[1] == "User prefers terse."

    async def test_read_back(self, backend):
        canonical = await backend.create(parse("/memories/core/"), "Likes coffee")
        assert await backend.read(canonical) == "Likes coffee"

    async def test_edit_reembeds(self, backend, mock_vectors):
        canonical = await backend.create(parse("/memories/core/"), "Likes coffee")
        mock_vectors.add_core_memory.reset_mock()
        await backend.replace_text(canonical, "coffee", "tea")
        assert await backend.read(canonical) == "Likes tea"
        # Re-embed on edit
        mock_vectors.add_core_memory.assert_called_once()
        assert mock_vectors.add_core_memory.call_args[0][1] == "Likes tea"

    async def test_delete_deactivates_and_unembeds(
        self, backend, mock_vectors, sqlite_store
    ):
        canonical = await backend.create(parse("/memories/core/"), "Soft delete me")
        cid = canonical.file_id
        await backend.delete(canonical)
        # Vector sync: delete_core_memory called
        mock_vectors.delete_core_memory.assert_called_once_with(cid)
        # SQLite: deactivated, not hard-deleted
        actives = await sqlite_store.get_active_core_memories()
        assert all(c.id != cid for c in actives)

    async def test_create_empty_rejected(self, backend):
        with pytest.raises(FSError, match="empty"):
            await backend.create(parse("/memories/core/"), "  ")

    async def test_works_without_vectors(self, sqlite_store):
        # Backend tolerates vectors=None (e.g., headless/degraded mode).
        b = MemoryFSBackend(sqlite_store, vectors=None)
        canonical = await b.create(parse("/memories/core/"), "no vectors")
        assert await b.read(canonical) == "no vectors"


class TestReadOnlyTiers:
    async def test_digest_create_blocked(self, backend):
        with pytest.raises(FSError, match="read-only"):
            await backend.create(parse("/memories/digests/blipshell.md"), "x")

    async def test_session_create_blocked(self, backend):
        with pytest.raises(FSError, match="read-only"):
            await backend.create(parse("/memories/sessions/1-fake.md"), "x")

    async def test_friction_delete_blocked(self, backend):
        with pytest.raises(FSError, match="read-only"):
            await backend.delete(parse("/memories/friction/1-x.md"))


class TestDigests:
    async def test_read_missing_digest(self, backend, sqlite_store):
        await sqlite_store.create_project("blipshell")
        with pytest.raises(FSError, match="No digest exists"):
            await backend.read(parse("/memories/digests/blipshell.md"))

    async def test_read_existing_digest(self, backend, sqlite_store):
        import json
        await sqlite_store.create_project("blipshell")
        await sqlite_store.update_project(
            "blipshell",
            metadata_json=json.dumps({"digest": "Working on memory FS tool."}),
        )
        content = await backend.read(parse("/memories/digests/blipshell.md"))
        assert "memory FS tool" in content


class TestSessions:
    async def test_list_and_read(self, backend, sqlite_store):
        sid = await sqlite_store.create_session(title="Debug", project="blipshell")
        await sqlite_store.update_session(sid, summary="Found a bug.")
        entries = await backend.list_directory(parse("/memories/sessions/"))
        assert any(f"{sid}-" in e.path for e in entries)
        content = await backend.read(parse(f"/memories/sessions/{sid}-anything.md"))
        assert "Found a bug." in content
