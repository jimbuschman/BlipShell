"""Tests for scripts/consolidation_status.py.

The script exists to answer "did consolidation hold the archive-never-delete
mandate?", so the case that matters most is that it actually FAILS when the
mandate is broken.
"""

import sqlite3

import pytest

from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.memory import Memory
from scripts.consolidation_status import collect


@pytest.fixture
async def db(tmp_path):
    path = tmp_path / "status.db"
    store = SQLiteStore(str(path))
    await store.initialize()
    session_id = await store.create_session("s")
    ids = [
        await store.create_memory(Memory(
            session_id=session_id, role="user",
            content=f"content {i}", summary=f"summary {i}",
        ))
        for i in range(5)
    ]
    entity = await store.get_or_create_entity("acme", "organization")
    await store.create_entity_relationship(entity, "employs", entity, ids[1])
    await store.close()
    return path, ids


class TestCollect:
    async def test_counts_progress(self, db):
        path, ids = db
        store = SQLiteStore(str(path))
        await store.initialize()
        await store.mark_memories_consolidated(ids[:3])
        await store.close()

        s = collect(path)
        assert s["memories_total"] == 5
        assert s["checked"] == 3
        assert s["unchecked"] == 2

    async def test_separates_consolidation_archives(self, db):
        """A memory archived by something other than consolidation must not be
        counted as a near-duplicate merge."""
        path, ids = db
        conn = sqlite3.connect(path)
        conn.execute("UPDATE memories SET is_archived = 1 WHERE id = ?", (ids[4],))
        conn.execute(
            "UPDATE memories SET is_archived = 1, consolidated_at = '2026-01-01' "
            "WHERE id = ?", (ids[3],),
        )
        conn.commit()
        conn.close()

        s = collect(path)
        assert s["memories_archived"] == 2
        assert s["archived_by_consolidation"] == 1

    async def test_reads_the_dry_run_cursor(self, db):
        path, _ = db
        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
            ("consolidation_dry_run_cursor", "1234"),
        )
        conn.commit()
        conn.close()

        assert collect(path)["dry_run_cursor"] == 1234

    async def test_missing_cursor_is_zero(self, db):
        path, _ = db
        assert collect(path)["dry_run_cursor"] == 0


class TestOrphanDetection:
    """The check that justifies the script. Deleting a memory used to cascade
    its edges away; if that ever comes back this must catch it."""

    async def test_clean_db_reports_no_orphans(self, db):
        path, _ = db
        s = collect(path)
        assert s["orphan_edges"] == 0
        assert s["orphan_mentions"] == 0
        assert s["entity_edges"] == 1

    async def test_a_deleted_memory_leaves_a_detectable_orphan(self, db):
        """Delete the memory but leave its edge behind — exactly the state a
        non-cascading hard delete would produce."""
        path, ids = db
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM memories WHERE id = ?", (ids[1],))
        conn.commit()
        conn.close()

        s = collect(path)
        assert s["orphan_edges"] == 1, (
            "an edge pointing at a deleted memory went unreported"
        )

    async def test_orphan_mentions_are_detected(self, db):
        path, ids = db
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, memory_id) VALUES (1, 999999)"
        )
        conn.commit()
        conn.close()

        assert collect(path)["orphan_mentions"] == 1


class TestExitCode:
    async def test_orphans_exit_nonzero(self, db, monkeypatch, capsys):
        """A red banner nobody scripts against is not a check — the exit code
        has to carry it too."""
        from scripts import consolidation_status

        path, ids = db
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM memories WHERE id = ?", (ids[1],))
        conn.commit()
        conn.close()

        monkeypatch.setattr(
            "sys.argv", ["consolidation_status.py", "--db", str(path)],
        )
        assert consolidation_status.main() == 2

    async def test_clean_db_exits_zero(self, db, monkeypatch):
        from scripts import consolidation_status

        path, _ = db
        monkeypatch.setattr(
            "sys.argv", ["consolidation_status.py", "--db", str(path)],
        )
        assert consolidation_status.main() == 0

    async def test_finished_sweep_reads_as_100_percent(self, db, monkeypatch, capsys):
        """Memories archived by other paths are not consolidation candidates.
        Counting them as unchecked made a finished sweep report 75.7% while the
        summary line said complete."""
        from scripts import consolidation_status

        path, ids = db
        store = SQLiteStore(str(path))
        await store.initialize()
        # One archived by something else, the rest active and all checked.
        conn_ids = [i for i in ids if i != ids[4]]
        await store.mark_memories_consolidated(conn_ids)
        await store.close()

        conn = sqlite3.connect(path)
        conn.execute("UPDATE memories SET is_archived = 1 WHERE id = ?", (ids[4],))
        conn.commit()
        conn.close()

        monkeypatch.setattr(
            "sys.argv", ["consolidation_status.py", "--db", str(path)],
        )
        assert consolidation_status.main() == 0

        out = capsys.readouterr().out
        assert "100.0%" in out, out
        assert "Full corpus has been checked" in out

    async def test_missing_db_exits_one(self, tmp_path, monkeypatch):
        from scripts import consolidation_status

        monkeypatch.setattr(
            "sys.argv",
            ["consolidation_status.py", "--db", str(tmp_path / "nope.db")],
        )
        assert consolidation_status.main() == 1
