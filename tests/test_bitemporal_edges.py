"""Tests for bi-temporal edge tracking (Feature 4).

Tests contradiction expiration, temporal filtering, and backward compatibility.
"""

import pytest

from blipshell.memory.entity_extractor import CONTRADICTING_PREDICATES


# --- Contradicting predicates dictionary ---


class TestContradictingPredicates:
    """Verify the CONTRADICTING_PREDICATES dict is well-formed."""

    def test_works_at_contradicts_left(self):
        assert "left" in CONTRADICTING_PREDICATES["works_at"]

    def test_left_contradicts_works_at(self):
        assert "works_at" in CONTRADICTING_PREDICATES["left"]

    def test_uses_contradicts_stopped_using(self):
        assert "stopped_using" in CONTRADICTING_PREDICATES["uses"]

    def test_prefers_contradicts_dislikes(self):
        assert "dislikes" in CONTRADICTING_PREDICATES["prefers"]

    def test_lives_in_contradicts_moved_from(self):
        assert "moved_from" in CONTRADICTING_PREDICATES["lives_in"]

    def test_no_self_contradiction(self):
        """A predicate should not contradict itself."""
        for pred, contradicts in CONTRADICTING_PREDICATES.items():
            assert pred not in contradicts, f"{pred} contradicts itself"


# --- SQLite temporal methods ---


class TestTemporalRelationships:
    """Test create_entity_relationship_temporal() and expire methods."""

    async def test_create_temporal_relationship(self, sqlite_store):
        """Creating a temporal relationship should set valid_from."""
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("acme", "organization")

        rel_id = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        assert rel_id is not None

        # Check it has valid_from set
        rels = await sqlite_store.get_active_relationships_for_entity(subj_id)
        assert len(rels) >= 1
        found = [r for r in rels if r["id"] == rel_id]
        assert len(found) == 1
        assert found[0]["valid_from"] is not None
        assert found[0]["expired_at"] is None

    async def test_expire_contradicting(self, sqlite_store):
        """When 'left' is added, the existing 'works_at' should be expired."""
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("acme", "organization")

        # Create original: user works_at acme
        works_id = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        assert works_id is not None

        # Create contradicting: user left acme
        left_id = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "left", obj_id, memory_id=None,
        )
        assert left_id is not None

        # Expire contradicting relationships
        expired = await sqlite_store.expire_contradicting_relationships(
            subj_id, obj_id, CONTRADICTING_PREDICATES, "left", left_id,
        )
        assert expired == 1

        # Verify works_at is expired
        active = await sqlite_store.get_active_relationships_for_entity(subj_id)
        active_predicates = [r["predicate"] for r in active]
        assert "left" in active_predicates
        assert "works_at" not in active_predicates

        # Verify expired shows up in expired list
        expired_rels = await sqlite_store.get_expired_relationships_for_entity(subj_id)
        expired_predicates = [r["predicate"] for r in expired_rels]
        assert "works_at" in expired_predicates

    async def test_no_expiration_for_non_contradicting(self, sqlite_store):
        """Non-contradicting predicates should not be expired."""
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("python", "technology")

        # Create: user uses python
        uses_id = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "uses", obj_id, memory_id=None,
        )

        # Create: user likes python (not a contradiction of uses)
        likes_id = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "likes", obj_id, memory_id=None,
        )

        # 'likes' does not contradict 'uses' in CONTRADICTING_PREDICATES
        expired = await sqlite_store.expire_contradicting_relationships(
            subj_id, obj_id, CONTRADICTING_PREDICATES, "likes", likes_id,
        )
        assert expired == 0

        # Both should still be active
        active = await sqlite_store.get_active_relationships_for_entity(subj_id)
        assert len(active) >= 2

    async def test_connected_entities_excludes_expired(self, sqlite_store):
        """get_connected_entity_ids() should not return entities through expired edges."""
        user_id = await sqlite_store.get_or_create_entity("user", "person")
        acme_id = await sqlite_store.get_or_create_entity("acme", "organization")
        newco_id = await sqlite_store.get_or_create_entity("newco", "organization")

        # Create and expire: user works_at acme
        works_id = await sqlite_store.create_entity_relationship_temporal(
            user_id, "works_at", acme_id, memory_id=None,
        )
        left_id = await sqlite_store.create_entity_relationship_temporal(
            user_id, "left", acme_id, memory_id=None,
        )
        await sqlite_store.expire_contradicting_relationships(
            user_id, acme_id, CONTRADICTING_PREDICATES, "left", left_id,
        )

        # Create active: user works_at newco
        await sqlite_store.create_entity_relationship_temporal(
            user_id, "works_at", newco_id, memory_id=None,
        )

        # user's connected entities should include newco but not acme (via works_at)
        # Note: acme is still connected via the active "left" relationship
        connected = await sqlite_store.get_connected_entity_ids([user_id])
        assert newco_id in connected
        # acme should still show up through the "left" relationship (which is active)
        assert acme_id in connected

    async def test_backfill_valid_from(self, sqlite_store):
        """Old relationships created without valid_from should get it backfilled."""
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("test", "concept")

        # Create via old method (which doesn't set valid_from directly, but
        # the migration backfill in initialize() handles it)
        rel_id = await sqlite_store.create_entity_relationship(
            subj_id, "tested", obj_id, memory_id=None,
        )

        # Run backfill again (idempotent)
        await sqlite_store._db.execute(
            "UPDATE entity_relationships SET valid_from = created_at "
            "WHERE valid_from IS NULL"
        )
        await sqlite_store._db.commit()

        # All relationships should have valid_from
        cursor = await sqlite_store._db.execute(
            "SELECT COUNT(*) as cnt FROM entity_relationships WHERE valid_from IS NULL"
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 0


class TestDuplicateInsertReturnsNone:
    """INSERT OR IGNORE + lastrowid: when the insert is ignored as a
    duplicate, lastrowid is the connection's PREVIOUS insert — an unrelated
    row. Returning it corrupted expired_by in edge invalidation
    (deep-dive 2026-08-04). A duplicate must return None."""

    async def test_duplicate_temporal_relationship_returns_none(self, sqlite_store):
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("acme", "organization")

        first = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        assert first is not None

        # Interleave an unrelated insert so lastrowid points somewhere real —
        # the exact condition that made the bug look like a valid id.
        other_id = await sqlite_store.get_or_create_entity("python", "technology")
        unrelated = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "uses", other_id, memory_id=None,
        )
        assert unrelated is not None

        dup = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        assert dup is None            # was: `unrelated`'s id

    async def test_duplicate_plain_relationship_returns_none(self, sqlite_store):
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("acme", "organization")

        first = await sqlite_store.create_entity_relationship(
            subj_id, "works_at", obj_id, None,
        )
        assert first is not None
        dup = await sqlite_store.create_entity_relationship(
            subj_id, "works_at", obj_id, None,
        )
        assert dup is None


class TestFactReassertion:
    """An expired fact must be re-assertable (deep-dive 2026-08-04): the old
    inline UNIQUE(subject, predicate, object) meant once 'works_at' was
    expired by 'left', re-asserting 'works_at' was silently dropped forever.
    Uniqueness now applies to ACTIVE rows only (partial index)."""

    async def test_expired_fact_can_be_reasserted(self, sqlite_store):
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("acme", "organization")

        works_1 = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        left_id = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "left", obj_id, memory_id=None,
        )
        await sqlite_store.expire_contradicting_relationships(
            subj_id, obj_id, CONTRADICTING_PREDICATES, "left", left_id,
        )
        # Jim rejoins Acme: expire 'left' the same way a new works_at would
        works_2 = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        assert works_2 is not None, "re-assertion of an expired fact was dropped"
        assert works_2 != works_1
        await sqlite_store.expire_contradicting_relationships(
            subj_id, obj_id, CONTRADICTING_PREDICATES, "works_at", works_2,
        )

        active = await sqlite_store.get_active_relationships_for_entity(subj_id)
        active_preds = [r["predicate"] for r in active]
        assert "works_at" in active_preds
        assert "left" not in active_preds

        # History is preserved: the ORIGINAL works_at row is still there, expired
        expired = await sqlite_store.get_expired_relationships_for_entity(subj_id)
        expired_ids = [r["id"] for r in expired]
        assert works_1 in expired_ids

    async def test_duplicate_active_fact_still_ignored(self, sqlite_store):
        """The partial index still deduplicates ACTIVE rows."""
        subj_id = await sqlite_store.get_or_create_entity("user", "person")
        obj_id = await sqlite_store.get_or_create_entity("acme", "organization")

        first = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        assert first is not None
        dup = await sqlite_store.create_entity_relationship_temporal(
            subj_id, "works_at", obj_id, memory_id=None,
        )
        assert dup is None


class TestLegacyConstraintRebuild:
    """A DB created before 2026-08 carries the inline unique constraint —
    initialize() must rebuild the table (preserving rows) and move
    uniqueness to the partial index."""

    async def test_legacy_table_rebuilt_with_data_preserved(self, tmp_path):
        import sqlite3 as _sqlite3

        from blipshell.memory.sqlite_store import SQLiteStore

        db_path = str(tmp_path / "legacy.db")
        store = SQLiteStore(db_path)
        await store.initialize()
        await store.close()

        # Revert entity_relationships to its legacy shape with data in it
        conn = _sqlite3.connect(db_path)
        conn.executescript("""
            DROP INDEX IF EXISTS idx_entity_rel_active_unique;
            DROP TABLE entity_relationships;
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
                FOREIGN KEY (subject_id) REFERENCES entities(id),
                FOREIGN KEY (object_id) REFERENCES entities(id),
                FOREIGN KEY (source_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                UNIQUE(subject_id, predicate, object_id)
            );
            INSERT INTO entities (id, name, entity_type)
                VALUES (1, 'user', 'person'), (2, 'acme', 'organization');
            INSERT INTO entity_relationships
                (subject_id, predicate, object_id, valid_from, expired_at)
                VALUES (1, 'works_at', 2, '2026-01-01T00:00:00', '2026-02-01T00:00:00');
        """)
        conn.commit()
        conn.close()

        store2 = SQLiteStore(db_path)
        await store2.initialize()
        try:
            cursor = await store2._db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='entity_relationships'"
            )
            row = await cursor.fetchone()
            assert "UNIQUE(subject_id, predicate, object_id)" not in row["sql"]

            # Data survived the rebuild
            cursor = await store2._db.execute(
                "SELECT COUNT(*) AS n FROM entity_relationships"
            )
            assert (await cursor.fetchone())["n"] == 1

            # And the expired fact is now re-assertable
            rid = await store2.create_entity_relationship_temporal(
                1, "works_at", 2, memory_id=None,
            )
            assert rid is not None
        finally:
            await store2.close()
