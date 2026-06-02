"""Self-layer: lingering-thought store + the idle reflection prompt.

The store holds self-originated thoughts (separate from user memory); the prompt
is given only prior thoughts (no transcript) so what surfaces can't be a recap.
"""

import json

import pytest

from blipshell.core.self_reflection import (
    NOTHING,
    SelfThoughtStore,
    lingering_thought_prompt,
)


class FakeStore:
    """Stand-in for sqlite app_metadata (get/set_metadata over a dict)."""

    def __init__(self):
        self.data: dict[str, str] = {}

    async def get_metadata(self, key):
        return self.data.get(key)

    async def set_metadata(self, key, value):
        self.data[key] = value


# --- prompt ----------------------------------------------------------------

def test_prompt_excludes_conversation_and_allows_nothing():
    system, user = lingering_thought_prompt([])
    assert NOTHING in system
    assert "summary" in system.lower()        # explicitly not a summary
    assert "private" in user.lower() or "beginning" in user.lower()


def test_prompt_feeds_prior_thoughts_for_threading():
    system, user = lingering_thought_prompt(["I keep wondering about continuity",
                                             "names vs identity"])
    assert "continuity" in user
    assert "names vs identity" in user


# --- store -----------------------------------------------------------------

async def test_add_and_recent():
    s = SelfThoughtStore(FakeStore())
    await s.add("first")
    await s.add("second")
    assert await s.recent(5) == ["first", "second"]


async def test_pending_lifecycle():
    s = SelfThoughtStore(FakeStore())
    assert await s.has_pending() is False
    await s.add("a thought")
    assert await s.has_pending() is True

    taken = await s.take_pending()
    assert taken == "a thought"
    assert await s.has_pending() is False        # consumed
    assert await s.take_pending() is None         # nothing left


async def test_surfaced_only_once_across_reload():
    store = FakeStore()
    s = SelfThoughtStore(store)
    await s.add("remember me")
    await s.take_pending()
    # A fresh store over the same backing data must see it already surfaced.
    s2 = SelfThoughtStore(store)
    assert await s2.has_pending() is False


async def test_caps_retained_thoughts():
    s = SelfThoughtStore(FakeStore(), max_keep=3)
    for i in range(6):
        await s.add(f"t{i}")
    kept = await s.recent(99)
    assert kept == ["t3", "t4", "t5"]            # only the last max_keep


async def test_corrupt_store_is_safe():
    store = FakeStore()
    store.data[SelfThoughtStore.KEY] = "{not json"
    s = SelfThoughtStore(store)
    assert await s.recent(5) == []                # falls back to empty, no crash
    await s.add("recovers")
    assert await s.recent(5) == ["recovers"]


# --- standing-context path (embeddings + relevance) ------------------------

# Deterministic toy embedder: keyword -> fixed unit vector so cosine is exact.
_VECS = {
    "robotics cube": [1.0, 0.0, 0.0],
    "continuity of self": [0.0, 1.0, 0.0],
}


async def _toy_embed(text):
    # Unknown text -> a third orthogonal axis (cosine 0 with the known two).
    return _VECS.get(text, [0.0, 0.0, 1.0])


async def test_relevant_candidates_filters_by_cosine_floor():
    s = SelfThoughtStore(FakeStore(), embed_fn=_toy_embed)
    await s.add("robotics cube")
    await s.add("continuity of self")
    # Query identical to the robotics vector -> only that thought clears the floor.
    matches = await s.relevant_candidates([1.0, 0.0, 0.0], floor=0.5, k=3)
    assert [t for t, _ in matches] == ["robotics cube"]
    assert matches[0][1] == pytest.approx(1.0)


async def test_relevant_candidates_respects_k_and_ordering():
    s = SelfThoughtStore(FakeStore(), embed_fn=_toy_embed)
    await s.add("robotics cube")          # cosine 1.0 with the query below
    await s.add("continuity of self")     # cosine ~0.0
    # A query that leans toward robotics but also overlaps continuity a little.
    matches = await s.relevant_candidates([0.9, 0.1, 0.0], floor=0.0, k=1)
    assert len(matches) == 1
    assert matches[0][0] == "robotics cube"   # higher cosine ranks first


async def test_surfaced_thought_still_resurfaces_via_relevance():
    """The keystone: take_pending (the one-shot greeting) must NOT remove a
    thought from the standing relevance path. It sticks around and comes back."""
    s = SelfThoughtStore(FakeStore(), embed_fn=_toy_embed)
    await s.add("robotics cube")
    await s.take_pending()                       # greeting consumed it
    assert await s.has_pending() is False
    matches = await s.relevant_candidates([1.0, 0.0, 0.0], floor=0.5, k=3)
    assert [t for t, _ in matches] == ["robotics cube"]   # still retrievable


async def test_peek_pending_does_not_consume():
    s = SelfThoughtStore(FakeStore(), embed_fn=_toy_embed)
    await s.add("robotics cube")
    assert await s.peek_pending() == "robotics cube"
    assert await s.has_pending() is True         # peek left it unsurfaced
    assert await s.take_pending() == "robotics cube"


async def test_missing_embedding_is_backfilled_and_persisted():
    store = FakeStore()
    # Simulate a thought written before this layer existed (no embedding field).
    store.data[SelfThoughtStore.KEY] = json.dumps(
        [{"text": "robotics cube", "surfaced": False}]
    )
    s = SelfThoughtStore(store, embed_fn=_toy_embed)
    matches = await s.relevant_candidates([1.0, 0.0, 0.0], floor=0.5, k=3)
    assert [t for t, _ in matches] == ["robotics cube"]
    # The backfilled embedding is now persisted, so a reload doesn't re-embed.
    reloaded = json.loads(store.data[SelfThoughtStore.KEY])
    assert reloaded[0]["embedding"] == [1.0, 0.0, 0.0]


async def test_no_embedder_yields_no_candidates():
    # Without an embedder, thoughts have no vectors and the standing path stays
    # silent (the one-shot greeting still works — that path needs no embedding).
    s = SelfThoughtStore(FakeStore())
    await s.add("robotics cube")
    assert await s.relevant_candidates([1.0, 0.0, 0.0], floor=0.0, k=3) == []
    assert await s.has_pending() is True
