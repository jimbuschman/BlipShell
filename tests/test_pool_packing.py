"""Pool packing skips an oversized item instead of stopping (V3 Stage B2).

Pool.get_top_entries walked items by priority and BROKE at the first one
that did not fit, so a single long top-priority memory emptied the whole
pool: with a 100-token cap, a 101-token first item followed by a 10-token
second item selected nothing (review F5, reproduced). Items are independent
evidence; the right rule is to skip what does not fit and keep packing, and
to say why each rejected item was left out.
"""

from blipshell.memory.manager import MemoryManager, Pool, PoolItem


def _item(text_tokens: int, prio: float, memory_id: int = 0) -> PoolItem:
    # estimated_tokens is set explicitly so the test does not depend on the tokenizer
    return PoolItem(text=f"item-{prio}-{memory_id}", estimated_tokens=text_tokens,
                    priority_score=prio, memory_id=memory_id)


class TestSkipNotBreak:

    def test_oversized_top_item_does_not_empty_the_pool(self):
        pool = Pool("Recall", max_tokens=100)
        pool.add(_item(101, prio=9.0, memory_id=1))
        pool.add(_item(10, prio=1.0, memory_id=2))
        got = pool.get_top_entries(100)
        assert [i.memory_id for i in got] == [2]

    def test_packing_continues_past_several_oversized_items(self):
        pool = Pool("Recall", max_tokens=50)
        for mid, tokens in enumerate([80, 40, 70, 9, 60, 1], start=1):
            pool.add(_item(tokens, prio=10.0 - mid, memory_id=mid))
        got = pool.get_top_entries(50)
        assert [i.memory_id for i in got] == [2, 4, 6]  # 40 + 9 + 1 = 50
        assert sum(i.estimated_tokens for i in got) <= 50

    def test_omissions_are_recorded_with_a_reason(self):
        pool = Pool("Recall", max_tokens=100)
        pool.add(_item(101, prio=9.0, memory_id=1))
        pool.add(_item(10, prio=1.0, memory_id=2))
        pool.get_top_entries(100)
        assert [(i.memory_id, why) for i, why in pool.last_omitted] == [(1, "over budget")]

    def test_excluded_memory_ids_are_recorded_too(self):
        pool = Pool("RecentHistory", max_tokens=100)
        pool.add(_item(10, prio=2.0, memory_id=7))
        pool.get_top_entries(100, exclude_keys={("memory", 7)})
        assert [(i.memory_id, why) for i, why in pool.last_omitted] == [(7, "already sent via Recall")]

    def test_dossier_carried_ids_are_recorded_under_their_own_reason(self):
        pool = Pool("Recall", max_tokens=100)
        pool.add(_item(10, prio=2.0, memory_id=7))
        pool.add(_item(10, prio=1.0, memory_id=8))
        got = pool.get_top_entries(100, rendered_elsewhere={("memory", 7)})
        assert [i.memory_id for i in got] == [8]
        assert [(i.memory_id, why) for i, why in pool.last_omitted] == [(7, "already in the project dossier")]

    def test_max_items_still_caps(self):
        pool = Pool("Lessons", max_tokens=1000, max_items=2)
        for mid in range(1, 5):
            pool.add(_item(5, prio=10.0 - mid, memory_id=mid))
        got = pool.get_top_entries(1000)
        assert [i.memory_id for i in got] == [1, 2]
        assert {why for _, why in pool.last_omitted} == {"item cap"}

    def test_hard_cap_still_limits(self):
        pool = Pool("test", max_tokens=1000, hard_cap=10)
        pool.add(_item(25, prio=1.0, memory_id=1))
        assert pool.get_top_entries(1000) == []


class TestManagerLevel:

    def test_gather_keeps_small_recall_items_behind_a_giant_one(self, memory_config):
        mm = MemoryManager(memory_config, context_tokens=8000)
        mm.add_memory("Recall", PoolItem(text="g " * 900, priority_score=9.0, memory_id=1))
        mm.add_memory("Recall", PoolItem(text="small useful fact", priority_score=1.0, memory_id=2))
        items = mm.gather_memory(token_budget=4000, pool_budgets={"Recall": 100, "Core": 50, "Lessons": 50,
                                                                  "RecentHistory": 50, "ActiveSession": 0})
        recall = [i.memory_id for i in items if i.pool_name == "Recall"]
        assert recall == [2]
        omitted = mm.last_omitted()
        assert ("Recall", 1, "over budget") in omitted
