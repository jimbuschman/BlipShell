"""Tests for the token budget pool system (memory/manager.py)."""

import pytest

from blipshell.memory.manager import MemoryManager, Pool, PoolItem, estimate_tokens


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_none_string(self):
        assert estimate_tokens(None) == 0

    def test_normal_text(self):
        # ~4 chars per token
        assert estimate_tokens("hello world test") == 4

    def test_long_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100


class TestPool:
    def test_add_and_count(self):
        pool = Pool("test", max_tokens=1000)
        pool.add(PoolItem(text="hello world"))
        assert pool.item_count == 1
        assert pool.used_tokens > 0

    def test_no_duplicates(self):
        pool = Pool("test", max_tokens=1000)
        pool.add(PoolItem(text="same text"))
        pool.add(PoolItem(text="same text"))
        assert pool.item_count == 1

    def test_sorted_by_priority(self):
        pool = Pool("test", max_tokens=1000)
        pool.add(PoolItem(text="low priority", priority_score=1.0))
        pool.add(PoolItem(text="high priority", priority_score=5.0))
        entries = pool.get_top_entries(1000)
        assert entries[0].text == "high priority"

    def test_get_top_entries_respects_budget(self):
        pool = Pool("test", max_tokens=10)
        pool.add(PoolItem(text="a" * 100, priority_score=5.0))  # 25 tokens
        pool.add(PoolItem(text="b" * 20, priority_score=3.0))  # 5 tokens
        entries = pool.get_top_entries(10)
        # Only the smaller item should fit
        assert len(entries) <= 1

    def test_hard_cap_limits_entries(self):
        pool = Pool("test", max_tokens=1000, hard_cap=10)
        pool.add(PoolItem(text="a" * 100))  # 25 tokens > hard cap
        entries = pool.get_top_entries(1000)
        assert len(entries) == 0

    def test_remove_items(self):
        pool = Pool("test", max_tokens=1000)
        item = PoolItem(text="removable")
        pool.add(item)
        pool.remove_items([item])
        assert pool.item_count == 0

    def test_clear(self):
        pool = Pool("test", max_tokens=1000)
        pool.add(PoolItem(text="item1"))
        pool.add(PoolItem(text="item2"))
        pool.clear()
        assert pool.item_count == 0

    def test_get_oldest_items(self):
        pool = Pool("test", max_tokens=1000)
        from datetime import datetime, timedelta, timezone
        old = PoolItem(text="old", timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc))
        new = PoolItem(text="new", timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))
        pool.add(old)
        pool.add(new)
        oldest = pool.get_oldest_items(1)
        assert oldest[0].text == "old"


class TestMemoryManager:
    def test_initialization(self, memory_manager):
        usage = memory_manager.get_usage()
        assert "Core" in usage
        assert "ActiveSession" in usage
        assert "Recall" in usage
        assert "RecentHistory" in usage
        assert "Buffer" in usage

    def test_add_memory(self, memory_manager):
        memory_manager.add_memory("Recall", PoolItem(text="test memory"))
        usage = memory_manager.get_usage()
        assert usage["Recall"]["items"] == 1

    def test_add_to_unknown_pool(self, memory_manager):
        # Should log warning but not crash
        memory_manager.add_memory("NonExistent", PoolItem(text="test"))

    def test_gather_memory(self, memory_manager):
        memory_manager.add_memory("Core", PoolItem(
            text="core fact", session_role="system", priority_score=5.0,
        ))
        memory_manager.add_memory("Recall", PoolItem(
            text="recalled memory", session_role="system", priority_score=3.0,
        ))
        items = memory_manager.gather_memory(token_budget=1000)
        texts = [i.text for i in items]
        assert "core fact" in texts
        assert "recalled memory" in texts

    def test_lessons_labeled_correctly(self, memory_manager):
        memory_manager.add_memory("Core", PoolItem(
            text="lesson text", session_role="system2", priority_score=3.0,
        ))
        items = memory_manager.gather_memory(token_budget=1000)
        assert items[0].pool_name == "Lessons"

    def test_trim_on_overflow(self, memory_config):
        # Create a manager with very small budget
        memory_config.total_context_tokens = 512
        manager = MemoryManager(memory_config)
        # Add many items to overflow
        for i in range(50):
            manager.add_memory("ActiveSession", PoolItem(
                text=f"message {i} " * 10,
                session_role="user",
            ))
        # Should not crash; pool should be trimmed
        usage = manager.get_usage()
        assert usage["ActiveSession"]["used"] <= usage["ActiveSession"]["max"] + 100

    def test_get_pool(self, memory_manager):
        pool = memory_manager.get_pool("Core")
        assert pool is not None
        assert pool.name == "Core"

    def test_get_nonexistent_pool(self, memory_manager):
        assert memory_manager.get_pool("NonExistent") is None
