"""Self-gravity (step 1): per-thought weight that recurrence reinforces and
surfacing/age decay erodes, and that picks which thought surfaces.

All deterministic, no Ollama — stubbed embedder + judge. Critically also pins
that gravity is a strict no-op when disabled (the firewall: the assistant must
behave identically with the feature off).
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from blipshell.core.self_reflection import SelfThoughtStore
from blipshell.memory.search import MemorySearch

_VECS = {
    "a thought": [1.0, 0.0, 0.0],
    "another thought": [0.0, 1.0, 0.0],
    "echo of a thought": [1.0, 0.0, 0.0],   # identical vector -> recurrence echo
    "QUERY": [1.0, 1.0, 0.0],                # equidistant from the two thoughts
}


def _embed_sync(text):
    return _VECS.get(text, [0.0, 0.0, 1.0])


async def _embed_async(text):
    return _embed_sync(text)


class FakeMeta:
    def __init__(self):
        self.data = {}

    async def get_metadata(self, key):
        return self.data.get(key)

    async def set_metadata(self, key, value):
        self.data[key] = value


def _store(meta=None, *, gravity_enabled=True, **kw):
    return SelfThoughtStore(meta or FakeMeta(), embed_fn=_embed_async,
                            gravity_enabled=gravity_enabled, **kw)


def _raw(meta):
    return json.loads(meta.data[SelfThoughtStore.KEY])


# --- weight mechanics ------------------------------------------------------

class TestWeightMechanics:
    async def test_recurrence_reinforces_prior(self):
        s = _store(recur_boost=0.5, recur_threshold=0.85)
        await s.add("a thought")
        await s.add("echo of a thought")   # identical vector -> echoes the prior
        w = await s.effective_weights(["a thought"])
        assert w["a thought"] == pytest.approx(1.5, abs=0.02)

    async def test_recurrence_is_noop_when_disabled(self):
        meta = FakeMeta()
        s = _store(meta, gravity_enabled=False, recur_boost=0.5)
        await s.add("a thought")
        await s.add("echo of a thought")
        # raw weight untouched, and effective_weights is empty when disabled
        assert _raw(meta)[0]["weight"] == 1.0
        assert await s.effective_weights(["a thought"]) == {}

    async def test_fatigue_lowers_weight(self):
        s = _store(fatigue=0.6)
        await s.add("a thought")
        await s.apply_fatigue(["a thought"])
        w = await s.effective_weights(["a thought"])
        assert w["a thought"] == pytest.approx(0.6, abs=0.02)

    async def test_fatigue_floored_at_min_weight(self):
        s = _store(fatigue=0.6, min_weight=0.1)
        await s.add("a thought")
        for _ in range(8):
            await s.apply_fatigue(["a thought"])
        w = await s.effective_weights(["a thought"])
        assert w["a thought"] == pytest.approx(0.1, abs=1e-6)

    async def test_age_decay(self):
        meta = FakeMeta()
        s = _store(meta, half_life_days=30.0)
        await s.add("a thought")
        # Backdate it 30 days -> one half-life -> weight halves.
        items = _raw(meta)
        items[0]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()
        meta.data[SelfThoughtStore.KEY] = json.dumps(items)
        w = await s.effective_weights(["a thought"])
        assert w["a thought"] == pytest.approx(0.5, abs=0.03)

    async def test_effective_weights_empty_when_disabled(self):
        s = _store(gravity_enabled=False)
        await s.add("a thought")
        assert await s.effective_weights(["a thought"]) == {}


# --- the gate picks the heaviest relevant thought --------------------------

def _make_search():
    vectors = Mock()
    vectors.embed_text = _embed_sync
    return MemorySearch(sqlite=Mock(), vectors=vectors, router=Mock(), config=None)


def _stub_judge(search, verdicts):
    async def judge(query, thought):
        return verdicts.get(thought, 0.0)
    search._judge_relevance = judge


class TestGateUsesGravity:
    async def _two_passing_thoughts(self):
        """Both thoughts clear the prefilter and the judge; 'a thought' is
        fatigued so it weighs less than 'another thought'."""
        s = _store(fatigue=0.6)
        await s.add("a thought")
        await s.add("another thought")
        await s.apply_fatigue(["a thought"])   # -> weight 0.6 vs 1.0
        return s

    async def test_gravity_on_heaviest_wins(self):
        s = await self._two_passing_thoughts()
        search = _make_search()
        _stub_judge(search, {"a thought": 1.0, "another thought": 1.0})
        out = await search.search_self_thoughts(
            "QUERY", s, cosine_floor=0.4, rerank_floor=0.8,
            max_inject=1, prefilter_k=3, gravity_enabled=True,
        )
        # equal cosine, but 'another thought' is heavier -> it takes the slot
        assert [t for t, _ in out] == ["another thought"]

    async def test_gravity_off_falls_back_to_cosine_order(self):
        s = await self._two_passing_thoughts()
        search = _make_search()
        _stub_judge(search, {"a thought": 1.0, "another thought": 1.0})
        out = await search.search_self_thoughts(
            "QUERY", s, cosine_floor=0.4, rerank_floor=0.8,
            max_inject=1, prefilter_k=3, gravity_enabled=False,
        )
        # weight ignored -> cosine-tie resolved by stable order -> 'a thought'
        assert [t for t, _ in out] == ["a thought"]
