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


# --- write-path embed failure + duplicate repair ---------------------------

class _FlakyEmbedder:
    """Raises on the first `fail_times` calls, then embeds normally.

    Models the live failure: reflection fires after hours of idle, the embed
    model is cold, the first call times out.
    """

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    async def __call__(self, text):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("embed backend cold")
        return _embed_sync(text)


def _dup_row(text, *, weight=1.0, surfaced=True, vec=(1.0, 0.0, 0.0),
             created_at=None):
    row = {"text": text, "surfaced": surfaced, "weight": weight,
           "embedding": list(vec)}
    if created_at:
        row["created_at"] = created_at
    return row


def _seed(meta, rows):
    meta.data[SelfThoughtStore.KEY] = json.dumps(rows)


class TestEmbedRetry:
    async def test_transient_failure_is_retried_and_thought_still_folds(self):
        """The bug: one failed embed made the thought unfoldable forever, so
        an echo landed as a permanent duplicate row."""
        meta = FakeMeta()
        flaky = _FlakyEmbedder(fail_times=1)
        s = SelfThoughtStore(meta, embed_fn=flaky, gravity_enabled=True,
                             embed_retry_delay=0.0)
        await s.add("a thought")
        assert flaky.calls == 2          # failed once, retried, succeeded
        assert _raw(meta)[0]["embedding"] == [1.0, 0.0, 0.0]

        await s.add("echo of a thought")
        assert len(_raw(meta)) == 1      # folded, not duplicated

    async def test_exhausted_retries_still_store_the_thought(self):
        """Embedding is best-effort — a total backend outage must never lose
        the thought, only leave it unembedded for later backfill."""
        meta = FakeMeta()
        flaky = _FlakyEmbedder(fail_times=99)
        s = SelfThoughtStore(meta, embed_fn=flaky, gravity_enabled=True,
                             embed_attempts=3, embed_retry_delay=0.0)
        await s.add("a thought")
        assert flaky.calls == 3
        rows = _raw(meta)
        assert len(rows) == 1
        assert rows[0]["embedding"] is None

    async def test_structurally_unavailable_embedder_is_not_retried(self):
        """A returned None means there's no vector store at all — retrying
        that is pure latency for a guaranteed-identical answer."""
        calls = []

        async def none_embedder(text):
            calls.append(text)
            return None

        s = SelfThoughtStore(FakeMeta(), embed_fn=none_embedder,
                             gravity_enabled=True, embed_attempts=3,
                             embed_retry_delay=0.0)
        await s.add("a thought")
        assert len(calls) == 1


class TestDuplicateRepair:
    async def test_add_collapses_duplicates_left_by_earlier_failures(self):
        """Three identical rows (the live 2026-07-30 state) become one, with
        the weight the thought would have had if folding had never failed."""
        meta = FakeMeta()
        _seed(meta, [_dup_row("cubes v1"), _dup_row("cubes v2"), _dup_row("cubes v3")])
        s = _store(meta, recur_boost=0.5)

        await s.add("another thought")   # distinct vector -> triggers repair

        rows = _raw(meta)
        assert len(rows) == 2
        assert rows[0]["text"] == "cubes v3"          # newest phrasing wins
        # 3 emissions == base 1.0 + two boosts
        assert rows[0]["weight"] == pytest.approx(2.0)
        assert rows[1]["text"] == "another thought"

    async def test_repair_carries_boosts_the_duplicate_had_accumulated(self):
        """Two rows that each folded one echo represent four emissions, so the
        merged weight must be 1.0 + 3 boosts — not a flat single boost."""
        meta = FakeMeta()
        _seed(meta, [_dup_row("mirror a", weight=1.5),
                     _dup_row("mirror b", weight=1.5)])
        s = _store(meta, recur_boost=0.5)

        await s.add("another thought")

        assert _raw(meta)[0]["weight"] == pytest.approx(2.5)

    async def test_repair_never_invents_weight_from_a_fatigued_duplicate(self):
        """Fatigue can push a duplicate below its base weight; the carried
        term floors at zero so repair can't manufacture gravity."""
        meta = FakeMeta()
        _seed(meta, [_dup_row("x", weight=1.0), _dup_row("y", weight=0.36)])
        s = _store(meta, recur_boost=0.5)

        await s.add("another thought")

        assert _raw(meta)[0]["weight"] == pytest.approx(1.5)

    async def test_repair_keeps_the_earliest_created_at(self):
        old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        new = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        meta = FakeMeta()
        _seed(meta, [_dup_row("first", created_at=old),
                     _dup_row("second", created_at=new)])
        s = _store(meta)

        await s.add("another thought")

        assert _raw(meta)[0]["created_at"] == old

    async def test_repair_does_not_resurrect_surfaced_thoughts(self):
        """add() makes an evolved thought pending again — but that's for a
        thought just formed. Repairing a backlog must not fire off a queue of
        old thoughts as unprompted greetings."""
        meta = FakeMeta()
        _seed(meta, [_dup_row("a", surfaced=True), _dup_row("b", surfaced=True)])
        s = _store(meta)

        await s.add("another thought")

        merged = _raw(meta)[0]
        assert merged["surfaced"] is True

    async def test_repair_keeps_pending_when_a_duplicate_was_unsurfaced(self):
        meta = FakeMeta()
        _seed(meta, [_dup_row("a", surfaced=True), _dup_row("b", surfaced=False)])
        s = _store(meta)

        await s.add("another thought")

        assert _raw(meta)[0]["surfaced"] is False

    async def test_repair_is_a_noop_when_gravity_disabled(self):
        """The firewall: with the feature off the store must behave exactly as
        it did before, duplicates and all."""
        meta = FakeMeta()
        _seed(meta, [_dup_row("a"), _dup_row("b"), _dup_row("c")])
        s = _store(meta, gravity_enabled=False)

        await s.add("another thought")

        assert len(_raw(meta)) == 4

    async def test_unembedded_prior_is_backfilled_so_an_echo_can_fold(self):
        """The second half of the bug: even with a good incoming vector, a
        prior missing its own vector scored 0.0 and so could never be echoed.
        """
        meta = FakeMeta()
        _seed(meta, [{"text": "a thought", "surfaced": True,
                      "weight": 1.0, "embedding": None}])
        s = _store(meta)

        await s.add("echo of a thought")   # same vector as "a thought"

        rows = _raw(meta)
        assert len(rows) == 1
        assert rows[0]["weight"] == pytest.approx(1.5)


class TestRelevancePathRepair:
    async def test_backfill_on_relevance_check_also_folds_the_duplicate(self):
        meta = FakeMeta()
        _seed(meta, [_dup_row("a thought"),
                     {"text": "echo of a thought", "surfaced": True,
                      "weight": 1.0, "embedding": None}])
        s = _store(meta)

        out = await s.relevant_candidates([1.0, 0.0, 0.0], floor=0.4, k=5)

        assert len(_raw(meta)) == 1
        assert [t for t, _ in out] == ["echo of a thought"]

    async def test_relevance_check_does_not_fold_when_nothing_was_backfilled(self):
        """The per-turn path stays cheap: fully-embedded stores skip the
        O(n^2) sweep entirely and leave repair to the idle write path."""
        meta = FakeMeta()
        _seed(meta, [_dup_row("a"), _dup_row("b")])
        s = _store(meta)

        await s.relevant_candidates([1.0, 0.0, 0.0], floor=0.4, k=5)

        assert len(_raw(meta)) == 2


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
