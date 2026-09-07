"""Archived entity vectors: husks are dead, dormant entities are asleep.

The June 2026 merge archived 22,775 entities and their vectors stayed in
vec_entities. Two very different populations were hiding under one flag:

  * 7,557 HUSKS (merged away; name in entity_aliases). Their vectors kept
    winning KNN in creation-time resolution, so new mentions were merged
    INTO dead entities the graph can never reach (46 measured post-merge).
  * 15,218 DORMANT (pruned; no alias). Re-mention is supposed to revive
    them, so they must remain candidates and keep their vectors.

These tests pin the split at every layer that touches it: the similarity
search, the orphan sweep and its dry-run counter, the nightly backfill, and
the extractor's routing when a husk still comes back as a candidate.
"""

import pytest

from blipshell.memory.vector_store import VectorStore


# Distinct, deterministic 8-dim directions. The three "emotion" names are
# near-identical so all three land in the KNN top-k; the decoy is far away.
_VECS = {
    "emotion engine": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "emotionengine": [0.99, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "mood states": [0.95, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "the emotion engine": [0.98, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "postgresql": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


def _fake_embed(text: str) -> list[float]:
    return _VECS.get(text, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])


@pytest.fixture
async def vectors(sqlite_store, temp_db_path):
    v = VectorStore(db_path=temp_db_path, embedding_model="fake",
                    ollama_url="http://localhost:1", embedding_dim=8)
    v.initialize()
    v._ollama_client = object()  # backfill refuses to run without a client
    v._embed = _fake_embed
    v._embed_batch = lambda texts: [_fake_embed(t) for t in texts]
    yield v
    v.close()


@pytest.fixture
async def graph(sqlite_store, vectors):
    """canonical (active) <- husk (archived + alias); dormant (archived, no alias)."""
    canonical = await sqlite_store.get_or_create_entity("emotion engine", "concept")
    husk = await sqlite_store.get_or_create_entity("emotionengine", "concept")
    dormant = await sqlite_store.get_or_create_entity("mood states", "concept")
    decoy = await sqlite_store.get_or_create_entity("postgresql", "technology")
    for eid, name in ((canonical, "emotion engine"), (husk, "emotionengine"),
                      (dormant, "mood states"), (decoy, "postgresql")):
        vectors.upsert_entity(eid, name)
    # The merge: husk -> canonical, then archived. The vector delete "failed".
    await sqlite_store.merge_entity(husk, canonical)
    await sqlite_store.record_entity_alias("emotionengine", canonical, "retroactive_embedding")
    await sqlite_store.archive_entities([husk])
    # The prune: archived with no alias.
    await sqlite_store.archive_entities([dormant])
    return {"canonical": canonical, "husk": husk, "dormant": dormant, "decoy": decoy}


# --- search_similar_entities -------------------------------------------------


async def test_similarity_search_never_returns_a_husk(vectors, graph):
    ids = {r["id"] for r in vectors.search_similar_entities("the emotion engine", n_results=5)}
    assert graph["husk"] not in ids
    assert graph["canonical"] in ids


async def test_similarity_search_still_returns_dormant_entities(vectors, graph):
    """Pruned entities are legitimate targets — re-mention revives them."""
    ids = {r["id"] for r in vectors.search_similar_entities("the emotion engine", n_results=5)}
    assert graph["dormant"] in ids


async def test_husk_filter_does_not_eat_the_result_budget(vectors, graph):
    """The husk is the NEAREST vector. With k=1 and no over-fetch the filter
    would leave nothing; the caller must still get its one live result."""
    results = vectors.search_similar_entities("emotionengine", n_results=1)
    assert len(results) == 1
    assert results[0]["id"] == graph["canonical"]


async def test_similarity_search_respects_n_results_after_filter(vectors, graph):
    results = vectors.search_similar_entities("the emotion engine", n_results=2)
    assert len(results) == 2
    assert graph["husk"] not in {r["id"] for r in results}


# --- cleanup_orphan_vectors / count_orphan_vectors ---------------------------


async def test_sweep_deletes_husk_vectors_and_keeps_dormant(vectors, graph):
    result = vectors.cleanup_orphan_vectors()
    assert result["entities_husks"] == 1
    remaining = vectors.get_all_ids("entities")
    assert graph["husk"] not in remaining
    assert graph["dormant"] in remaining
    assert graph["canonical"] in remaining


async def test_sweep_deletes_vectors_of_missing_entities(vectors, graph, sqlite_store):
    vectors.upsert_entity(99999, "postgresql")  # no entities row
    result = vectors.cleanup_orphan_vectors()
    assert result["entities_missing"] == 1
    assert 99999 not in vectors.get_all_ids("entities")


async def test_sweep_keeps_memory_counts_under_their_old_keys(vectors, graph):
    """The repair CLI prints result['archived'] / ['missing'] — those stay."""
    result = vectors.cleanup_orphan_vectors()
    assert set(result) == {"archived", "missing", "entities_husks", "entities_missing"}


async def test_dry_run_count_matches_what_the_sweep_would_delete(vectors, graph):
    vectors.upsert_entity(99999, "postgresql")
    before = vectors.count_orphan_vectors()
    swept = vectors.cleanup_orphan_vectors()
    assert before == swept
    assert before["entities_husks"] == 1 and before["entities_missing"] == 1
    after = vectors.count_orphan_vectors()
    assert all(v == 0 for v in after.values())
    # counting is read-only: the sweep still had something to do
    assert swept["entities_husks"] == 1


async def test_sweep_is_idempotent(vectors, graph):
    vectors.cleanup_orphan_vectors()
    second = vectors.cleanup_orphan_vectors()
    assert second["entities_husks"] == 0 and second["entities_missing"] == 0


# --- backfill must not undo the sweep ----------------------------------------


async def test_backfill_does_not_reembed_husks(vectors, graph):
    """With no filter the nightly backfill put every swept husk vector back
    the same night — sweep and backfill fighting forever."""
    vectors.cleanup_orphan_vectors()
    stats = vectors.backfill_missing_vectors("entities", limit=50)
    assert stats.get("error") is None
    assert graph["husk"] not in vectors.get_all_ids("entities")


async def test_backfill_does_reembed_dormant_and_active(vectors, graph):
    # Drop every entity vector, then let the backfill rebuild.
    for eid in list(vectors.get_all_ids("entities")):
        vectors.delete_entity(eid)
    vectors.backfill_missing_vectors("entities", limit=50)
    ids = vectors.get_all_ids("entities")
    assert graph["canonical"] in ids
    assert graph["dormant"] in ids
    assert graph["husk"] not in ids


# --- SQLiteStore.resolve_husk ------------------------------------------------


async def test_resolve_husk_routes_a_husk_to_its_canonical(sqlite_store, graph):
    assert await sqlite_store.resolve_husk(graph["husk"]) == (graph["canonical"], "emotion engine")


async def test_resolve_husk_is_none_for_active_and_dormant(sqlite_store, graph):
    assert await sqlite_store.resolve_husk(graph["canonical"]) is None
    assert await sqlite_store.resolve_husk(graph["dormant"]) is None
    assert await sqlite_store.resolve_husk(424242) is None


async def test_resolve_husk_follows_a_chain(sqlite_store, graph):
    """x merged into husk BEFORE husk merged into canonical: x -> husk -> canonical."""
    x = await sqlite_store.get_or_create_entity("emo engine", "concept")
    await sqlite_store.record_entity_alias("emo engine", graph["husk"], "embedding_auto")
    await sqlite_store.archive_entities([x])
    assert await sqlite_store.resolve_husk(x) == (graph["canonical"], "emotion engine")
