"""Token budget pool system (direct port of MemoryManager.cs).

5 pools: Core (10%/cap 2048), ActiveSession (35%), RecentHistory (15%),
Recall (30%/cap 8192), Buffer (10%).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from blipshell.models.config import MemoryConfig

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (text.length / 4). Port of TokenEstimator.cs."""
    if not text:
        return 0
    return len(text) // 4


@dataclass
class PoolItem:
    """An item in a memory pool."""
    text: str
    estimated_tokens: int = 0
    priority_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_role: str = "user"  # user, assistant, system, system2 (lessons)
    pool_name: str = ""
    session_id: int = 0

    def __post_init__(self):
        if self.estimated_tokens == 0:
            self.estimated_tokens = estimate_tokens(self.text)


class Pool:
    """A single memory token budget pool."""

    def __init__(self, name: str, max_tokens: int, hard_cap: int | None = None):
        self.name = name
        self.max_tokens = max_tokens
        self.hard_cap = hard_cap
        self._items: list[PoolItem] = []

    @property
    def used_tokens(self) -> int:
        return sum(item.estimated_tokens for item in self._items)

    @property
    def item_count(self) -> int:
        return len(self._items)

    def add(self, item: PoolItem):
        """Add item, avoiding duplicates by text content."""
        if any(existing.text == item.text for existing in self._items):
            return
        self._items.append(item)
        self._items.sort(key=lambda x: x.priority_score, reverse=True)

    def get_top_entries(self, available_tokens: int) -> list[PoolItem]:
        """Get top entries that fit within available tokens."""
        selected = []
        used = 0
        effective_cap = min(available_tokens, self.hard_cap or self.max_tokens)

        for item in self._items:
            if used + item.estimated_tokens <= effective_cap:
                selected.append(item)
                used += item.estimated_tokens
            else:
                break
        return selected

    def get_oldest_items(self, count: int) -> list[PoolItem]:
        """Get the oldest N items."""
        return sorted(self._items, key=lambda x: x.timestamp)[:count]

    def remove_items(self, items_to_remove: list[PoolItem]):
        """Remove specified items from the pool."""
        remove_set = {id(item) for item in items_to_remove}
        self._items = [item for item in self._items if id(item) not in remove_set]

    def clear(self):
        """Remove all items."""
        self._items.clear()


class MemoryManager:
    """Token budget pool management system.

    Port of MemoryManager.cs:
    - 5 pools with configurable percentages and hard caps
    - Rollover: unused tokens redistribute by priority
    - Overflow trimming: oldest items summarized and moved to RecentHistory
    """

    OVERHEAD_TOKENS = 1000

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.global_budget = config.total_context_tokens - config.system_prompt_reserve
        self._pools: dict[str, Pool] = {}
        self._pool_configs: dict[str, dict] = {}
        self._summarize_callback = None

        self._configure_pools()

    def set_summarize_callback(self, callback):
        """Set callback for summarizing overflow items: async (text) -> str."""
        self._summarize_callback = callback

    def _configure_pools(self):
        """Configure pools from config."""
        pools_cfg = self.config.pools
        pool_defs = {
            "Core": (pools_cfg.core.percentage, pools_cfg.core.priority, pools_cfg.core.max_tokens),
            "ActiveSession": (pools_cfg.active_session.percentage, pools_cfg.active_session.priority, pools_cfg.active_session.max_tokens),
            "RecentHistory": (pools_cfg.recent_history.percentage, pools_cfg.recent_history.priority, pools_cfg.recent_history.max_tokens),
            "Recall": (pools_cfg.recall.percentage, pools_cfg.recall.priority, pools_cfg.recall.max_tokens),
            "Buffer": (pools_cfg.buffer.percentage, pools_cfg.buffer.priority, pools_cfg.buffer.max_tokens),
        }

        total_allocated = 0
        for name, (pct, priority, hard_cap) in pool_defs.items():
            base_budget = int(self.global_budget * pct)
            capped = min(base_budget, hard_cap) if hard_cap else base_budget
            self._pools[name] = Pool(name, capped, hard_cap)
            self._pool_configs[name] = {"priority": priority, "percentage": pct}
            total_allocated += capped

        # Rollover: distribute unused tokens by priority
        unused = self.global_budget - total_allocated
        if unused > 0:
            expandable = sorted(
                [(name, cfg["priority"]) for name, cfg in self._pool_configs.items() if cfg["priority"] > 0],
                key=lambda x: x[1],
                reverse=True,
            )
            if expandable:
                bonus = unused // len(expandable)
                for name, _ in expandable:
                    self._pools[name].max_tokens += bonus

    def add_memory(self, pool_name: str, item: PoolItem):
        """Add a memory item to a pool, trimming if over budget."""
        pool = self._pools.get(pool_name)
        if not pool:
            logger.warning("Unknown pool: %s", pool_name)
            return

        if pool.used_tokens + item.estimated_tokens > pool.max_tokens:
            self._trim_pool(pool_name)

        pool.add(item)

    def gather_memory(self, token_budget: int | None = None) -> list[PoolItem]:
        """Gather memory items from all pools within budget."""
        if token_budget is None:
            token_budget = self.global_budget

        remaining = token_budget
        result = []

        for pool in self._pools.values():
            entries = pool.get_top_entries(remaining)
            for entry in entries:
                if remaining >= entry.estimated_tokens:
                    entry.pool_name = "Lessons" if entry.session_role == "system2" else pool.name
                    result.append(entry)
                    remaining -= entry.estimated_tokens

        return result

    def _trim_pool(self, pool_name: str):
        """Trim a pool by removing oldest items and optionally summarizing."""
        pool = self._pools.get(pool_name)
        if not pool:
            return

        batch_size = self.config.overflow_batch_size
        while pool.used_tokens > pool.max_tokens:
            oldest = pool.get_oldest_items(batch_size)
            if not oldest:
                break

            # For ActiveSession, summarize overflow into RecentHistory
            if pool_name == "ActiveSession" and self._summarize_callback:
                combined = " ".join(item.text for item in oldest)
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._summarize_and_store(combined))
                except RuntimeError:
                    pass  # No event loop running

            pool.remove_items(oldest)

    async def _summarize_and_store(self, text: str):
        """Summarize overflow text and add to RecentHistory."""
        if not self._summarize_callback or not text.strip():
            return
        try:
            summary = await self._summarize_callback(text)
            if summary:
                self.add_memory("RecentHistory", PoolItem(
                    text=summary,
                    session_role="system",
                    priority_score=1.0,
                ))
                logger.info("Summarized overflow → RecentHistory: %s", summary[:80])
        except Exception as e:
            logger.error("Failed to summarize overflow: %s", e)

    def get_pool(self, name: str) -> Pool | None:
        return self._pools.get(name)

    def get_usage(self) -> dict[str, dict]:
        """Get usage stats for all pools."""
        return {
            name: {
                "used": pool.used_tokens,
                "max": pool.max_tokens,
                "items": pool.item_count,
                "hard_cap": pool.hard_cap,
            }
            for name, pool in self._pools.items()
        }

    def print_usage(self):
        """Log memory usage for debugging."""
        logger.info("=== MEMORY USAGE ===")
        for name, stats in self.get_usage().items():
            logger.info("  %-15s: %d / %d tokens (%d items)",
                        name, stats["used"], stats["max"], stats["items"])
