"""Two-stage relevance filter for self-thought injection (MemorySearch).

The cosine prefilter is recall-only; the reranker is the gate. These tests pin
the gate's behavior — especially fail-closed — without needing Ollama, by
stubbing the embedder (vectors.embed_text) and the reranker (score_pair).
"""

from unittest.mock import Mock

import pytest

from blipshell.core.self_reflection import SelfThoughtStore
from blipshell.memory.search import MemorySearch


class FakeMeta:
    def __init__(self):
        self.data = {}

    async def get_metadata(self, key):
        return self.data.get(key)

    async def set_metadata(self, key, value):
        self.data[key] = value


# Toy embedder shared by the store (thoughts) and the search (query) so cosine
# is exact and controllable.
_VECS = {
    "robotics cube": [1.0, 0.0, 0.0],
    "continuity of self": [0.0, 1.0, 0.0],
}


def _embed_sync(text):
    return _VECS.get(text, [0.0, 0.0, 1.0])


async def _embed_async(text):
    return _embed_sync(text)


class StubReranker:
    """Returns preset scores per document; records calls for assertions."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    async def score_pair(self, query, document):
        self.calls.append((query, document))
        return self.scores.get(document, 0.0)


def _make_search(reranker_enabled):
    vectors = Mock()
    vectors.embed_text = _embed_sync
    search = MemorySearch(sqlite=Mock(), vectors=vectors, router=Mock(), config=None)
    search.reranker_enabled = reranker_enabled
    return search


async def _store_with(*thoughts):
    store = SelfThoughtStore(FakeMeta(), embed_fn=_embed_async)
    for t in thoughts:
        await store.add(t)
    return store


async def test_fail_closed_when_reranker_disabled():
    """With the reranker off, nothing surfaces — and it's never even called."""
    search = _make_search(reranker_enabled=False)
    search._reranker = StubReranker({"robotics cube": 1.0})  # would pass if used
    store = await _store_with("robotics cube")

    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert out == []
    assert search._reranker.calls == []   # gate was skipped entirely


async def test_reranker_gate_rejects_below_floor():
    """A thought that clears the loose cosine prefilter but not the reranker
    floor does NOT surface — the cap can't save us, so the gate must hold."""
    search = _make_search(reranker_enabled=True)
    search._reranker = StubReranker({"robotics cube": 0.6})   # below 0.8
    store = await _store_with("robotics cube")

    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert out == []
    assert search._reranker.calls            # cosine passed it to the reranker


async def test_reranker_gate_accepts_above_floor():
    search = _make_search(reranker_enabled=True)
    search._reranker = StubReranker({"robotics cube": 0.92})
    store = await _store_with("robotics cube")

    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert [t for t, _ in out] == ["robotics cube"]
    assert out[0][1] == pytest.approx(0.92)


async def test_cosine_prefilter_excludes_unrelated_before_rerank():
    """An unrelated thought never reaches the reranker (cosine floor drops it)."""
    search = _make_search(reranker_enabled=True)
    search._reranker = StubReranker({"continuity of self": 1.0})
    store = await _store_with("continuity of self")

    # Query orthogonal to the only thought -> cosine 0 -> prefiltered out.
    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert out == []
    assert search._reranker.calls == []   # nothing survived the prefilter


async def test_max_inject_caps_results():
    search = _make_search(reranker_enabled=True)
    search._reranker = StubReranker({"robotics cube": 0.95, "continuity of self": 0.9})
    store = await _store_with("robotics cube", "continuity of self")

    # A query that is similar to both (45° to each axis) so both clear cosine.
    import math
    search.vectors.embed_text = lambda text: [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0]

    out = await search.search_self_thoughts(
        "anything", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert len(out) == 1                       # capped
    assert out[0][0] == "robotics cube"        # higher reranker score wins
