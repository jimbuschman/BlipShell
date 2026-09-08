"""The batch tagger must make monotonic progress through the pool.

Real SQLite, canned model. Sep 2026: the pool was read newest-first with no
cursor and the skip marker was gated on allow_new_tags (off in the nightly),
so memories the model gave one vague tag were re-sent every batch until the
budget ran out — 11 memories touched against a pool of 17,080.
"""

from unittest.mock import AsyncMock, MagicMock

from blipshell.memory.batch_tagger import JUNK_TAG_NAMES, POOL_MAX_TAGS, BatchTagger
from blipshell.memory.sqlite_store import BATCH_TAG_SKIP_MARKER
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory


async def _seed(sqlite_store, n=5, vocab=("identity", "code", "neutral", "llm")):
    ids = []
    for i in range(n):
        mid = await sqlite_store.create_memory(Memory(
            role="user", content=f"content {i}", summary=f"summary {i}",
        ))
        ids.append(mid)
    # the vocabulary must exist before the model can pick from it
    await sqlite_store._ensure_tags_exist(list(vocab))
    await sqlite_store._db.commit()
    return ids


def _tagger(sqlite_store, response: str, batch_size=3):
    router = MagicMock()
    router.generate = AsyncMock(return_value=response)
    return BatchTagger(sqlite_store, router, MemoryConfig(batch_tag_batch_size=batch_size))


async def _pool(sqlite_store):
    return await sqlite_store.get_poorly_tagged_memory_ids(max_tags=POOL_MAX_TAGS, limit=999)


async def test_every_examined_memory_leaves_the_pool(sqlite_store):
    await _seed(sqlite_store, n=3)
    # Two good tags / one vague tag / NONE.
    tagger = _tagger(sqlite_store, "1: identity, code\n2: neutral\n3: NONE")
    assert len(await _pool(sqlite_store)) == 3
    stats = await tagger.tag_batch()
    assert stats["memories_in_batch"] == 3
    assert stats["memories_tagged"] == 2          # items 1 and 2 received tags
    assert stats["memories_marked_skip"] == 2     # item 2 (one tag) and item 3 (NONE)
    assert await _pool(sqlite_store) == []


async def test_one_vague_tag_is_kept_but_memory_is_marked(sqlite_store):
    ids = await _seed(sqlite_store, n=1)
    tagger = _tagger(sqlite_store, "1: neutral")
    await tagger.tag_batch()
    tags = await sqlite_store.get_memory_tags(ids[0])
    assert "neutral" in tags and BATCH_TAG_SKIP_MARKER in tags
    assert await _pool(sqlite_store) == []


async def test_llm_failure_marks_nothing(sqlite_store):
    """A model error says nothing about the memories; hiding them would be wrong."""
    await _seed(sqlite_store, n=2)
    tagger = _tagger(sqlite_store, "")
    tagger.router.generate = AsyncMock(side_effect=RuntimeError("ollama down"))
    stats = await tagger.tag_batch()
    assert stats["error"] and stats["memories_marked_skip"] == 0
    assert len(await _pool(sqlite_store)) == 2


async def test_consecutive_batches_are_new_work(sqlite_store):
    await _seed(sqlite_store, n=6)
    tagger = _tagger(sqlite_store, "1: NONE\n2: NONE\n3: NONE", batch_size=3)
    result = await tagger.tag_all(max_batches=10)
    # 6 memories, batches of 3, all NONE: two productive batches, a third finds nothing
    assert result["batches"] == 3
    assert result["checked"] == 6
    assert result["memories_marked_skip"] == 6
    assert result["remaining_pool"] == 0
    assert result["stopped_early"] is False


async def test_exclude_ids_keeps_a_run_from_resending_even_if_marker_failed(sqlite_store):
    await _seed(sqlite_store, n=2)
    first = await sqlite_store.get_poorly_tagged_memory_ids(max_tags=1, limit=1)
    second = await sqlite_store.get_poorly_tagged_memory_ids(
        max_tags=1, limit=1, exclude_ids=set(first),
    )
    assert first and second and first != second
    assert await sqlite_store.count_poorly_tagged_memories() == 2


async def test_junk_names_never_offered_never_stored(sqlite_store):
    ids = await _seed(sqlite_store, n=1, vocab=("identity", "nnone", BATCH_TAG_SKIP_MARKER))
    tagger = _tagger(sqlite_store, "1: nnone, none, identity")
    offered = await tagger._get_available_tags()
    assert "nnone" not in offered and BATCH_TAG_SKIP_MARKER not in offered
    assert "identity" in offered
    await tagger.tag_batch()
    tags = await sqlite_store.get_memory_tags(ids[0])
    assert "nnone" not in tags and "none" not in tags
    assert "identity" in tags


async def test_purge_tags_removes_vocabulary_and_links(sqlite_store):
    ids = await _seed(sqlite_store, n=1, vocab=("identity",))
    await sqlite_store.tag_memory(ids[0], ["nnone", "identity"])
    assert "nnone" in await sqlite_store.get_all_tag_names()
    removed = await sqlite_store.purge_tags(sorted(JUNK_TAG_NAMES))
    assert removed == 1
    assert "nnone" not in await sqlite_store.get_all_tag_names()
    assert await sqlite_store.get_memory_tags(ids[0]) == ["identity"]
    assert await sqlite_store.purge_tags(sorted(JUNK_TAG_NAMES)) == 0  # idempotent


async def test_skip_marker_alone_is_not_a_tag_for_the_pool(sqlite_store):
    """The old failure: counted, `_skip` left a memory at one tag inside a
    '<= 1' pool. Excluding by name is what actually removes it."""
    ids = await _seed(sqlite_store, n=1)
    await sqlite_store.tag_memory(ids[0], [BATCH_TAG_SKIP_MARKER])
    assert await _pool(sqlite_store) == []
    assert await sqlite_store.count_poorly_tagged_memories() == 0
