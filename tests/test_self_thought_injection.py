"""Two-stage relevance filter for self-thought injection (MemorySearch).

The cosine prefilter is recall-only; a local LLM yes/no judge is the gate.
These tests pin the gate's behavior — especially fail-closed — without needing
Ollama, by stubbing the embedder (vectors.embed_text) and the judge
(search._judge_relevance).
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


def _make_search():
    vectors = Mock()
    vectors.embed_text = _embed_sync
    return MemorySearch(sqlite=Mock(), vectors=vectors, router=Mock(), config=None)


def _stub_judge(search, verdicts):
    """Make search._judge_relevance return verdicts.get(thought, 0.0), recording calls."""
    calls = []

    async def judge(query, thought):
        calls.append((query, thought))
        return verdicts.get(thought, 0.0)

    search._judge_relevance = judge
    return calls


async def _store_with(*thoughts):
    store = SelfThoughtStore(FakeMeta(), embed_fn=_embed_async)
    for t in thoughts:
        await store.add(t)
    return store


async def test_fail_closed_when_judge_errors():
    """If the judge raises, nothing surfaces — better silent than sloppy."""
    search = _make_search()

    async def boom(query, thought):
        raise RuntimeError("model down")

    search._judge_relevance = boom
    store = await _store_with("robotics cube")

    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert out == []


async def test_judge_no_rejects():
    """A thought that clears the cosine prefilter but the judge says 'no' (0.0)
    does NOT surface — the cap can't save us, so the gate must hold."""
    search = _make_search()
    _stub_judge(search, {"robotics cube": 0.0})
    store = await _store_with("robotics cube")

    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert out == []


async def test_judge_yes_accepts():
    search = _make_search()
    _stub_judge(search, {"robotics cube": 1.0})
    store = await _store_with("robotics cube")

    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert [t for t, _ in out] == ["robotics cube"]
    assert out[0][1] == pytest.approx(1.0)   # cosine of identical vectors


async def test_cosine_prefilter_excludes_unrelated_before_judge():
    """An unrelated thought never reaches the judge (cosine floor drops it)."""
    search = _make_search()
    calls = _stub_judge(search, {"continuity of self": 1.0})
    store = await _store_with("continuity of self")

    # Query orthogonal to the only thought -> cosine 0 -> prefiltered out.
    out = await search.search_self_thoughts(
        "robotics cube", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert out == []
    assert calls == []   # nothing survived the prefilter, judge never ran


async def test_max_inject_caps_and_orders_by_cosine():
    search = _make_search()
    _stub_judge(search, {"robotics cube": 1.0, "continuity of self": 1.0})
    store = await _store_with("robotics cube", "continuity of self")

    # Query closer to robotics (0.8) than continuity (0.6) — both clear cosine,
    # both judged yes, but max_inject=1 keeps the higher-cosine one.
    search.vectors.embed_text = lambda text: [0.8, 0.6, 0.0]

    out = await search.search_self_thoughts(
        "anything", store,
        cosine_floor=0.4, rerank_floor=0.8, max_inject=1, prefilter_k=3,
    )
    assert len(out) == 1
    assert out[0][0] == "robotics cube"
