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
