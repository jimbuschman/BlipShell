"""Self-thought fatigue is charged where the thought reaches the PROMPT.

Fatigue (x0.6 per surfacing) is the anti-spiral mechanism: the same thought
can't dominate turn after turn. It used to be charged the moment the thought
was queued into the Recall pool — but Recall is a 40%-of-budget pool where a
thought competes against memories whose boosted_score routinely exceeds 1.0,
so a thought could be evicted before the model ever saw it and still pay the
full fatigue (deep-dive 2026-08-04). Weight decayed for surfacings that never
happened, and recurrence — which only fires once per multi-hour idle gap —
could never catch up.

Exercises the real ChatMixin methods against a real MemoryManager.
"""

import json
from unittest.mock import MagicMock

import pytest

from blipshell.core.agent_chat import ChatMixin
from blipshell.core.self_reflection import SelfThoughtStore
from blipshell.memory.manager import MemoryManager, PoolItem
from blipshell.models.config import MemoryConfig


class _Reflection:
    """Stand-in for config.reflection with gravity on."""
    enabled = True
    inject_enabled = True
    inject_cosine_floor = 0.4
    inject_rerank_floor = 0.8
    inject_max = 2
    inject_prefilter_k = 3
    gravity_enabled = True
    gravity_marker_weight = 1.5
    gravity_fatigue = 0.6
    gravity_min_weight = 0.1


class _Config:
    def __init__(self):
        self.reflection = _Reflection()


class _Agent(ChatMixin):
    """Minimal host for the two methods under test."""

    def __init__(self, store, matches):
        self.config = _Config()
        self._self_thoughts = store
        self._relevance_injected_thoughts = set()
        self._pending_thought_fatigue = {}
        self.memory_manager = MemoryManager(MemoryConfig(), context_tokens=8000)
        self.search = MagicMock()

        async def _search_self_thoughts(*a, **kw):
            return matches
        self.search.search_self_thoughts = _search_self_thoughts


async def _store_with(meta, texts):
    async def embed(text):
        return [1.0, 0.0, 0.0]
    s = SelfThoughtStore(
        meta, embed_fn=embed, gravity_enabled=True,
        fatigue=0.6, min_weight=0.1,
        recur_threshold=2.0,   # >1 so nothing folds; keep rows distinct
    )
    for t in texts:
        await s.add(t)
    return s


async def _weight(sqlite, text):
    for it in await sqlite.get_self_thoughts():
        if it["text"] == text:
            return it["weight"]
    raise AssertionError(f"{text!r} not in store")


class TestFatigueChargedOnlyWhenRendered:
    async def test_rendered_thought_is_fatigued(self, thought_harness):
        meta = thought_harness.sqlite
        store = await _store_with(meta, ["thought A"])
        agent = _Agent(store, [("thought A", 0.9)])

        await agent._search_self_thoughts("query")
        assert await _weight(meta, "thought A") == 1.0, "fatigue charged too early"

        # The thought made it into the rendered context
        pool_text = next(iter(agent._pending_thought_fatigue))
        await agent._charge_surfaced_thought_fatigue({pool_text})

        assert await _weight(meta, "thought A") == pytest.approx(0.6)

    async def test_budget_evicted_thought_pays_nothing(self, thought_harness):
        """The actual bug: queued but never rendered must cost nothing."""
        meta = thought_harness.sqlite
        store = await _store_with(meta, ["thought A"])
        agent = _Agent(store, [("thought A", 0.9)])

        await agent._search_self_thoughts("query")
        # Nothing survived the budget
        await agent._charge_surfaced_thought_fatigue(set())

        assert await _weight(meta, "thought A") == 1.0, (
            "a thought the model never saw was still fatigued"
        )

    async def test_partial_render_charges_only_the_survivor(self, thought_harness):
        meta = thought_harness.sqlite
        store = await _store_with(meta, ["thought A", "thought B"])
        agent = _Agent(store, [("thought A", 0.9), ("thought B", 0.8)])

        await agent._search_self_thoughts("query")
        pending = dict(agent._pending_thought_fatigue)
        survivor = next(pt for pt, t in pending.items() if t == "thought A")

        await agent._charge_surfaced_thought_fatigue({survivor})

        assert await _weight(meta, "thought A") == pytest.approx(0.6)
        assert await _weight(meta, "thought B") == 1.0

    async def test_charge_is_idempotent_within_a_turn(self, thought_harness):
        """The pending set is consumed, so a second call can't double-charge."""
        meta = thought_harness.sqlite
        store = await _store_with(meta, ["thought A"])
        agent = _Agent(store, [("thought A", 0.9)])

        await agent._search_self_thoughts("query")
        pool_text = next(iter(agent._pending_thought_fatigue))
        await agent._charge_surfaced_thought_fatigue({pool_text})
        await agent._charge_surfaced_thought_fatigue({pool_text})

        assert await _weight(meta, "thought A") == pytest.approx(0.6)

    async def test_gravity_off_never_charges(self, thought_harness):
        """Firewall: with gravity disabled the store must be untouched."""
        meta = thought_harness.sqlite
        store = await _store_with(meta, ["thought A"])
        agent = _Agent(store, [("thought A", 0.9)])
        agent.config.reflection.gravity_enabled = False

        await agent._search_self_thoughts("query")
        pending = dict(agent._pending_thought_fatigue)
        await agent._charge_surfaced_thought_fatigue(set(pending))

        assert await _weight(meta, "thought A") == 1.0

    async def test_thought_still_reaches_the_recall_pool(self, thought_harness):
        """The fix must not stop thoughts from surfacing at all."""
        meta = thought_harness.sqlite
        store = await _store_with(meta, ["thought A"])
        agent = _Agent(store, [("thought A", 0.9)])

        await agent._search_self_thoughts("query")

        gathered = agent.memory_manager.gather_memory()
        assert any("thought A" in i.text for i in gathered)
        assert any(i.text.startswith("[Thought") for i in gathered)
