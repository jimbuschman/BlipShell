"""repair_husk_references: drain the references that already landed on husks.

Before Stage 2 husk routing and the entity vector sweep, creation-time
resolution could match a merged-away entity's leftover vector and record
the mention, alias and relationships on it (46 cases after the June 2026
merge). Search excludes archived entities, so those references were
invisible. The repair moves them to the terminal canonical. Dormant (pruned)
entities are untouched: the mention they were pruned with is normal.
"""

from blipshell.models.memory import Memory
from tests.test_entity_husk_vectors import graph, vectors  # noqa: F401  (fixtures)


async def _memory(sqlite_store, text):
    return await sqlite_store.create_memory(Memory(role="user", content=text))


async def _strand_on_husk(sqlite_store, graph):
    """Reproduce the live defect: a post-merge mention resolved onto the husk."""
    mid = await _memory(sqlite_store, "the emotion engine is display-only")
    await sqlite_store._db.execute(
        "INSERT INTO entity_mentions (entity_id, memory_id) VALUES (?, ?)", (graph["husk"], mid),
    )
    await sqlite_store.record_entity_alias("the emotion engine", graph["husk"], "embedding_auto")
    other = await sqlite_store.get_or_create_entity("cubes", "project")
    await sqlite_store._db.execute(
        "INSERT INTO entity_relationships (subject_id, predicate, object_id, source_memory_id) "
        "VALUES (?, 'part_of', ?, ?)", (graph["husk"], other, mid),
    )
    await sqlite_store._db.commit()
    return mid, other


async def _refs(sqlite_store, eid):
    async def one(sql, args):
        cursor = await sqlite_store._db.execute(sql, args)
        return (await cursor.fetchone())[0]
    m = await one("SELECT COUNT(*) FROM entity_mentions WHERE entity_id = ?", (eid,))
    r = await one("SELECT COUNT(*) FROM entity_relationships WHERE subject_id = ? OR object_id = ?", (eid, eid))
    a = await one("SELECT COUNT(*) FROM entity_aliases WHERE canonical_entity_id = ?", (eid,))
    return m, r, a


async def test_repair_moves_stranded_references_to_the_canonical(sqlite_store, graph):
    await _strand_on_husk(sqlite_store, graph)
    assert await _refs(sqlite_store, graph["husk"]) == (1, 1, 1)

    result = await sqlite_store.repair_husk_references()

    assert result["husks"] == 1 and result["repointed"] == 1 and result["unresolved"] == 0
    assert (result["mentions_moved"], result["relationships_moved"], result["aliases_repointed"]) == (1, 1, 1)
    assert await _refs(sqlite_store, graph["husk"]) == (0, 0, 0)
    m, r, a = await _refs(sqlite_store, graph["canonical"])
    assert m == 1 and r == 1
    assert a == 2  # the husk's own alias row + the repointed stray one
    assert await sqlite_store.resolve_alias("the emotion engine") == graph["canonical"]


async def test_repair_leaves_dormant_entities_alone(sqlite_store, graph):
    """A pruned entity keeps the mention it was pruned with — normal, not stranded."""
    mid = await _memory(sqlite_store, "mood states")
    await sqlite_store._db.execute(
        "INSERT INTO entity_mentions (entity_id, memory_id) VALUES (?, ?)", (graph["dormant"], mid),
    )
    await sqlite_store._db.commit()
    result = await sqlite_store.repair_husk_references()
    assert result["husks"] == 0
    assert await _refs(sqlite_store, graph["dormant"]) == (1, 0, 0)


async def test_repair_dry_run_reports_but_does_not_write(sqlite_store, graph):
    await _strand_on_husk(sqlite_store, graph)
    result = await sqlite_store.repair_husk_references(dry_run=True)
    assert result["dry_run"] is True
    assert (result["mentions_moved"], result["relationships_moved"], result["aliases_repointed"]) == (1, 1, 1)
    assert await _refs(sqlite_store, graph["husk"]) == (1, 1, 1)


async def test_repair_is_idempotent_and_clean_db_is_zero(sqlite_store, graph):
    await _strand_on_husk(sqlite_store, graph)
    await sqlite_store.repair_husk_references()
    again = await sqlite_store.repair_husk_references()
    assert again["husks"] == 0 and again["repointed"] == 0


async def test_repair_follows_alias_chain_to_terminal_canonical(sqlite_store, graph):
    """x merged into husk before husk merged into canonical: a mention stranded
    on x must end on the canonical, not on the intermediate husk."""
    x = await sqlite_store.get_or_create_entity("emo engine", "concept")
    await sqlite_store.record_entity_alias("emo engine", graph["husk"], "embedding_auto")
    await sqlite_store.archive_entities([x])
    mid = await _memory(sqlite_store, "emo engine again")
    await sqlite_store._db.execute(
        "INSERT INTO entity_mentions (entity_id, memory_id) VALUES (?, ?)", (x, mid),
    )
    await sqlite_store._db.commit()
    result = await sqlite_store.repair_husk_references()
    # Two husks are referenced: x (the mention) and husk (x's alias row names
    # it as canonical). Both drain to the terminal canonical.
    assert result["repointed"] == 2
    assert await _refs(sqlite_store, x) == (0, 0, 0)
    assert await _refs(sqlite_store, graph["husk"]) == (0, 0, 0)
    assert (await _refs(sqlite_store, graph["canonical"]))[0] == 1
    assert await sqlite_store.resolve_alias("emo engine") == graph["canonical"]


async def test_repair_counts_dead_end_chain_as_unresolved(sqlite_store):
    """A husk whose alias names a deleted canonical cannot be repaired
    automatically — report it, do not guess."""
    husk = await sqlite_store.get_or_create_entity("ghost", "concept")
    # The FK on entity_aliases blocks this today; the old hard-deleting
    # cleanup_entities job is how such rows came to exist. Reproduce that.
    await sqlite_store._db.execute("PRAGMA foreign_keys = OFF")
    await sqlite_store._db.execute(
        "INSERT INTO entity_aliases (alias_name, canonical_entity_id, merge_method) "
        "VALUES ('ghost', 777777, 'retroactive_embedding')",
    )
    await sqlite_store._db.commit()
    await sqlite_store._db.execute("PRAGMA foreign_keys = ON")
    await sqlite_store.archive_entities([husk])
    mid = await _memory(sqlite_store, "ghost")
    await sqlite_store._db.execute(
        "INSERT INTO entity_mentions (entity_id, memory_id) VALUES (?, ?)", (husk, mid),
    )
    await sqlite_store._db.commit()
    result = await sqlite_store.repair_husk_references()
    assert result["husks"] == 1 and result["unresolved"] == 1 and result["repointed"] == 0
    assert (await _refs(sqlite_store, husk))[0] == 1
