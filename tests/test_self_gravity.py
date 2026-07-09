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
    async def test_recurrence_reinforces_and_folds(self):
        """An echo boosts the prior AND is absorbed into it — the thought
        evolves in place instead of piling up near-duplicate copies."""
        meta = FakeMeta()
        s = _store(meta, recur_boost=0.5, recur_threshold=0.85)
        await s.add("a thought")
        await s.add("echo of a thought")   # identical vector -> echoes the prior
        items = _raw(meta)
        assert len(items) == 1                          # folded, not appended
        assert items[0]["text"] == "echo of a thought"  # evolved phrasing wins
        w = await s.effective_weights(["echo of a thought"])
        assert w["echo of a thought"] == pytest.approx(1.5, abs=0.02)

    async def test_fold_keeps_original_created_at(self):
        """Decay clocks from the thought's origin — folding must not reset it,
        or a recurring thought would never have to keep earning its weight."""
        meta = FakeMeta()
        s = _store(meta)
        await s.add("a thought")
        original_created = _raw(meta)[0]["created_at"]
        await s.add("echo of a thought")
        assert _raw(meta)[0]["created_at"] == original_created

    async def test_fold_makes_evolved_thought_pending_again(self):
        s = _store()
        await s.add("a thought")
        assert await s.take_pending() == "a thought"
        assert not await s.has_pending()
        await s.add("echo of a thought")   # folds into the surfaced prior
        assert await s.peek_pending() == "echo of a thought"

    async def test_recurrence_is_noop_when_disabled(self):
        meta = FakeMeta()
        s = _store(meta, gravity_enabled=False, recur_boost=0.5)
        await s.add("a thought")
        await s.add("echo of a thought")
        # no fold, raw weight untouched, effective_weights empty when disabled
        assert len(_raw(meta)) == 2
        assert _raw(meta)[0]["weight"] == 1.0
        assert await s.effective_weights(["a thought"]) == {}

    async def test_add_backfills_missing_created_at(self):
        """Thoughts written before the gravity layer are undated and thus
        exempt from decay — add() stamps them so their clock starts."""
        meta = FakeMeta()
        meta.data[SelfThoughtStore.KEY] = json.dumps([
            {"text": "ancient thought", "surfaced": True,
             "embedding": [0.0, 0.0, 1.0], "weight": 1.0},
        ])
        s = _store(meta)
        await s.add("another thought")
        items = _raw(meta)
        assert all(it.get("created_at") for it in items)

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


# --- snapshot (observability for /thoughts) --------------------------------

class TestSnapshot:
    async def test_snapshot_rows(self):
        s = _store()
        await s.add("a thought")
        await s.add("echo of a thought")   # folds into the prior -> weight 1.5
        await s.add("another thought")     # distinct -> second row
        await s.take_pending()             # surfaces the folded thought

        rows = await s.snapshot()
        assert len(rows) == 2
        first, second = rows
        assert first["text"] == "echo of a thought"
        assert first["surfaced"] is True
        assert first["weight"] == pytest.approx(1.5)
        # Fresh thought, no meaningful age decay yet
        assert first["effective_weight"] == pytest.approx(1.5, abs=0.02)
        assert first["has_embedding"] is True
        assert second["text"] == "another thought"
        assert second["surfaced"] is False
        assert second["weight"] == pytest.approx(1.0)

    async def test_snapshot_effective_weight_none_when_disabled(self):
        s = _store(gravity_enabled=False)
        await s.add("a thought")
        rows = await s.snapshot()
        assert rows[0]["effective_weight"] is None
        assert rows[0]["weight"] == pytest.approx(1.0)  # inert metadata

    async def test_snapshot_applies_age_decay(self):
        meta = FakeMeta()
        s = _store(meta, half_life_days=30.0)
        await s.add("a thought")
        # Backdate the thought one half-life
        items = _raw(meta)
        items[0]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).isoformat()
        meta.data[SelfThoughtStore.KEY] = json.dumps(items)

        rows = await s.snapshot()
        assert rows[0]["weight"] == pytest.approx(1.0)          # stored untouched
        assert rows[0]["effective_weight"] == pytest.approx(0.5, abs=0.02)

    async def test_snapshot_flags_missing_embedding(self):
        meta = FakeMeta()
        s = SelfThoughtStore(meta, embed_fn=None, gravity_enabled=True)
        await s.add("a thought")   # embed_fn None -> no embedding stored
        rows = await s.snapshot()
        assert rows[0]["has_embedding"] is False

    async def test_snapshot_does_not_mutate(self):
        meta = FakeMeta()
        s = _store(meta)
        await s.add("a thought")
        before = meta.data[SelfThoughtStore.KEY]
        await s.snapshot()
        assert meta.data[SelfThoughtStore.KEY] == before

    async def test_snapshot_empty_store(self):
        s = _store()
        assert await s.snapshot() == []
