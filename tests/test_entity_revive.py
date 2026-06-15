"""Tests for revive-on-re-mention.

When a pruned (soft-archived) entity gains new activity, it should un-archive
so a thing that becomes relevant again returns to the live graph. But a
MERGED-away husk (its name recorded in entity_aliases) must NOT revive —
resurrecting it would recreate the duplicate the merge eliminated.
"""

import pytest


async def _is_archived(store, eid: int) -> bool:
    cur = await store._db.execute("SELECT is_archived FROM entities WHERE id = ?", (eid,))
    row = await cur.fetchone()
    return bool(row["is_archived"])


class TestReviveEntities:
    async def test_pruned_entity_revives(self, sqlite_store):
        eid = await sqlite_store.get_or_create_entity("raymarching", "concept")
        await sqlite_store.archive_entities([eid])
        assert await _is_archived(sqlite_store, eid) is True

        revived = await sqlite_store.revive_entities([eid])
        assert revived == 1
        assert await _is_archived(sqlite_store, eid) is False

    async def test_merged_husk_not_revived(self, sqlite_store):
        """An archived entity whose name is a merge alias stays archived."""
        canonical = await sqlite_store.get_or_create_entity("chromadb", "technology")
        husk = await sqlite_store.get_or_create_entity("chroma", "technology")
        await sqlite_store.archive_entities([husk])
        await sqlite_store.record_entity_alias("chroma", canonical, merge_method="test")

        revived = await sqlite_store.revive_entities([husk])
        assert revived == 0
        assert await _is_archived(sqlite_store, husk) is True

    async def test_active_entity_is_noop(self, sqlite_store):
        eid = await sqlite_store.get_or_create_entity("python", "technology")
        revived = await sqlite_store.revive_entities([eid])
        assert revived == 0
        assert await _is_archived(sqlite_store, eid) is False

    async def test_empty_input(self, sqlite_store):
        assert await sqlite_store.revive_entities([]) == 0
        assert await sqlite_store.revive_entities([None]) == 0


class TestReviveWiring:
    async def test_new_relationship_revives_both_endpoints(self, sqlite_store):
        a = await sqlite_store.get_or_create_entity("esp32", "technology")
        b = await sqlite_store.get_or_create_entity("i2c", "technology")
        await sqlite_store.archive_entities([a, b])

        await sqlite_store.create_entity_relationship(a, "uses", b, None)

        assert await _is_archived(sqlite_store, a) is False
        assert await _is_archived(sqlite_store, b) is False

    async def test_relationship_does_not_revive_merged_husk(self, sqlite_store):
        canonical = await sqlite_store.get_or_create_entity("ws2812b", "technology")
        husk = await sqlite_store.get_or_create_entity("ws2812", "technology")
        live = await sqlite_store.get_or_create_entity("neopixel", "technology")
        await sqlite_store.archive_entities([husk, live])
        await sqlite_store.record_entity_alias("ws2812", canonical, merge_method="test")

        await sqlite_store.create_entity_relationship(husk, "related_to", live, None)

        # live (pruned) revives; husk (merged alias) stays archived
        assert await _is_archived(sqlite_store, live) is False
        assert await _is_archived(sqlite_store, husk) is True
