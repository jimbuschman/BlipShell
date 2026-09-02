"""backfill_missing_vectors: every collection's filter SQL must be valid.

Regression for the 2026-09-02 audit find: the backfill query prefixed the
active filter with `s.`, which worked for bare-column filters by luck and
crashed with a syntax error on reflections' parenthesized filter — which is
why 1,720 of 1,808 reflections had no embeddings. Filters are now written
self-qualified against the `s.` alias; these tests run the real query for
every collection against a real store, with the embedder faked.
"""

import pytest

from blipshell.memory.vector_store import _SOURCE_TABLES, VectorStore


@pytest.fixture
async def vectors(sqlite_store, temp_db_path):
    v = VectorStore(db_path=temp_db_path, embedding_model="fake",
                    ollama_url="http://localhost:1", embedding_dim=8)
    v.initialize()
    # Fake the embedder: backfill must exercise SQL, not the network.
    v._ollama_client = object()
    v._embed_batch = lambda texts: [[0.0] * 8 for _ in texts]
    yield v
    v.close()


async def test_all_collection_filters_execute(vectors):
    """The audit bug was a crash, not a wrong answer — every collection's
    backfill query must at least run."""
    for collection in _SOURCE_TABLES:
        stats = vectors.backfill_missing_vectors(collection, limit=5)
        assert "error" not in stats, (collection, stats)


async def test_reflections_backfill_embeds_and_skips_placeholders(
    vectors, sqlite_store,
):
    sid1 = await sqlite_store.create_session(title="real")
    sid2 = await sqlite_store.create_session(title="empty")
    await sqlite_store.create_session_reflection(
        session_id=sid1, effectiveness="effective",
        reflection_text="Real insight about the session.")
    await sqlite_store.create_session_reflection(
        session_id=sid2, effectiveness="skipped",
        reflection_text="Session skipped - insufficient conversation data.")

    stats = vectors.backfill_missing_vectors("reflections", limit=50)
    # The real reflection is embedded; the skipped placeholder is filtered out.
    assert stats["succeeded"] == 1
    assert stats["failed"] == 0
    # Second run: nothing left to do (the filter keeps excluding the placeholder).
    stats = vectors.backfill_missing_vectors("reflections", limit=50)
    assert stats == {"processed": 0, "succeeded": 0, "failed": 0}
