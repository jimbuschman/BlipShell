"""Token budget pool system.

5 pools: Core (5%, personal facts), Lessons (5%, top insights),
ActiveSession (30%), RecentHistory (20%), Recall (40%, search results).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from blipshell.models.config import MemoryConfig

logger = logging.getLogger(__name__)


_tokenizer = None
_tokenizer_loaded = False


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses tiktoken (cl100k_base encoding) when available for ~10-15% accuracy
    on Ollama models. Falls back to len/4 heuristic if tiktoken is not installed.
    """
    if not text:
        return 0

    global _tokenizer, _tokenizer_loaded
    if not _tokenizer_loaded:
        _tokenizer_loaded = True
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.debug("tiktoken not installed, using len//4 fallback for token estimation")
        except Exception as e:
            logger.debug("tiktoken init failed, using len//4 fallback: %s", e)

    if _tokenizer is not None:
        return len(_tokenizer.encode(text))
    return len(text) // 4


@dataclass
class PoolItem:
    """An item in a memory pool."""
    text: str
    estimated_tokens: int = 0
    priority_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_role: str = "user"  # user, assistant, system
    pool_name: str = ""
    session_id: int = 0
    # Evidence identity (V3 B3/B4): which stored record this text came from,
    # what kind of record it is, and who said it. memory_id > 0 lets the
    # builder collapse the same memory arriving via two pools; `source` and
    # `speaker` feed the rendered label and the /why trace.
    memory_id: int = 0
    source: str = ""     # memory | core | lesson | history | thought | user_model | summary
    speaker: str = ""    # user | assistant | "" (system-authored)

    def __post_init__(self):
        if self.estimated_tokens == 0:
            self.estimated_tokens = estimate_tokens(self.text)


class Pool:
    """A single memory token budget pool."""

    def __init__(self, name: str, max_tokens: int, hard_cap: int | None = None,
                 max_items: int | None = None):
        self.name = name
        self.max_tokens = max_tokens
        self.hard_cap = hard_cap
        self.max_items = max_items  # Hard cap on number of items
        self._items: list[PoolItem] = []
        # (item, reason) for everything the last get_top_entries left out -
        # "over budget" | "item cap" | "already sent via Recall". Read by the
        # /why trace so an omission is explained, not silent (V3 B2).
        self.last_omitted: list[tuple[PoolItem, str]] = []

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

    def get_top_entries(self, available_tokens: int, max_override: int | None = None,
                        exclude_memory_ids: set[int] | None = None,
                        rendered_elsewhere: set[int] | None = None) -> list[PoolItem]:
        """Get top entries that fit within available tokens and item count cap.

        `exclude_memory_ids`: items carrying one of these memory_ids are
        skipped without charging the budget - the same memory already reached
        the request through another pool (V3 B1: Recall beats RecentHistory).
        `rendered_elsewhere`: same, for memories a block outside the pools
        carries (the active project's dossier, V3 E2). Each is recorded in
        `last_omitted` under its own reason so the /why trace says which.
        """
        selected = []
        omitted: list[tuple[PoolItem, str]] = []
        used = 0
        effective_cap = min(available_tokens, max_override or self.hard_cap or self.max_tokens)

        for item in self._items:
            if self.max_items and len(selected) >= self.max_items:
                omitted.append((item, "item cap"))
                continue
            if exclude_memory_ids and item.memory_id and item.memory_id in exclude_memory_ids:
                omitted.append((item, "already sent via Recall"))
                continue
            if rendered_elsewhere and item.memory_id and item.memory_id in rendered_elsewhere:
                omitted.append((item, "already in the project dossier"))
                continue
            if used + item.estimated_tokens <= effective_cap:
                selected.append(item)
                used += item.estimated_tokens
            else:
                # SKIP and keep packing. This used to `break`: one long
                # top-priority item emptied the whole pool (101-token first
                # item under a 100-token cap selected nothing - review F5).
                omitted.append((item, "over budget"))
        self.last_omitted = omitted
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

    def __init__(self, config: MemoryConfig, context_tokens: int = 0):
        self.config = config
        # Use explicit context_tokens if provided, otherwise fall back to config
        effective_tokens = context_tokens or config.total_context_tokens
        self.global_budget = effective_tokens - config.system_prompt_reserve
        self._pools: dict[str, Pool] = {}
        self._pool_configs: dict[str, dict] = {}
        self._summarize_callback = None
        # Memory ids that reach the request through a block OUTSIDE the pools
        # (the active project's dossier, V3 E2). Every pool skips them, Recall
        # included: the dossier is their canonical, structured place.
        self.rendered_elsewhere: set[int] = set()

        self._configure_pools()

    def set_summarize_callback(self, callback):
        """Set callback for summarizing overflow items: async (text) -> str."""
        self._summarize_callback = callback

    def _configure_pools(self):
        """Configure pools from config.

        Pool contracts:
        - Core: Stable personal facts (max 20 items). Always present.
        - Lessons: Top extracted insights (max 30 items). Always present.
        - ActiveSession: Current conversation messages.
        - RecentHistory: Previous session memories + summaries.
        - Recall: Search results — largest pool, most relevant content per query.
        """
        pools_cfg = self.config.pools
        pool_defs = {
            "Core": (pools_cfg.core.percentage, pools_cfg.core.priority, pools_cfg.core.max_tokens, pools_cfg.core.max_items),
            "Lessons": (pools_cfg.lessons.percentage, pools_cfg.lessons.priority, pools_cfg.lessons.max_tokens, pools_cfg.lessons.max_items),
            "ActiveSession": (pools_cfg.active_session.percentage, pools_cfg.active_session.priority, pools_cfg.active_session.max_tokens, pools_cfg.active_session.max_items),
            "RecentHistory": (pools_cfg.recent_history.percentage, pools_cfg.recent_history.priority, pools_cfg.recent_history.max_tokens, pools_cfg.recent_history.max_items),
            "Recall": (pools_cfg.recall.percentage, pools_cfg.recall.priority, pools_cfg.recall.max_tokens, pools_cfg.recall.max_items),
        }

        total_allocated = 0
        for name, (pct, priority, hard_cap, max_items) in pool_defs.items():
            base_budget = int(self.global_budget * pct)
            capped = min(base_budget, hard_cap) if hard_cap else base_budget
            self._pools[name] = Pool(name, capped, hard_cap, max_items=max_items)
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

    def gather_memory(self, token_budget: int | None = None,
                      pool_budgets: dict[str, int] | None = None) -> list[PoolItem]:
        """Gather memory items from all pools within budget.

        If pool_budgets is provided, each pool's effective cap is overridden
        for this call only (used by dynamic query profiles).
        """
        if token_budget is None:
            token_budget = self.global_budget

        remaining = token_budget
        result = []

        # Recall first: it is query-relevant and carries the longer excerpt, so
        # when RecentHistory offers the same memory (last session's content is
        # both "recent" and "relevant"), the history copy is skipped. Before
        # this every recalled memory of the previous session rendered twice
        # (continuity baseline 2026-09-09: 16 duplicated renders / 13 cases).
        order = sorted(self._pools.values(), key=lambda p: 0 if p.name == "Recall" else 1)
        recalled_ids: set[int] = set()
        elsewhere = self.rendered_elsewhere or None
        for pool in order:
            cap = pool_budgets.get(pool.name) if pool_budgets else None
            exclude = recalled_ids if pool.name != "Recall" else None
            entries = pool.get_top_entries(remaining, max_override=cap, exclude_memory_ids=exclude,
                                           rendered_elsewhere=elsewhere)
            for entry in entries:
                if remaining >= entry.estimated_tokens:
                    entry.pool_name = pool.name
                    result.append(entry)
                    remaining -= entry.estimated_tokens
                    if pool.name == "Recall" and entry.memory_id:
                        recalled_ids.add(entry.memory_id)

        return result

    def schedule_overflow_summary(self, text: str):
        """Summarise conversation turns that fell out of the window into
        RecentHistory (V3 B1). Fire-and-forget; returns the task, or None when
        there is nothing to do or no callback / event loop."""
        if not self._summarize_callback or not text or not text.strip():
            return None
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No event loop - conversation overflow not summarised")
            return None
        task = loop.create_task(self._summarize_and_store(text))
        task.add_done_callback(
            lambda t: logger.error("Overflow summarization failed: %s", t.exception())
            if not t.cancelled() and t.exception() else None
        )
        return task

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

            # (The ActiveSession special case - summarise evicted turns into
            # RecentHistory - moved to _build_messages + schedule_overflow_summary
            # when the conversation stopped being mirrored into a pool, V3 B1.)
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

    def last_omitted(self) -> list[tuple[str, int, str]]:
        """(pool, memory_id, reason) for everything the last gather left out."""
        out = []
        for pool in self._pools.values():
            for item, reason in pool.last_omitted:
                out.append((pool.name, item.memory_id, reason))
        return out

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

    def get_hard_caps(self) -> dict[str, int | None]:
        """Get hard caps for all pools (used by query profile budget computation)."""
        return {name: pool.hard_cap for name, pool in self._pools.items()}

    def print_usage(self):
        """Log memory usage for debugging."""
        logger.info("=== MEMORY USAGE ===")
        for name, stats in self.get_usage().items():
            logger.info("  %-15s: %d / %d tokens (%d items)",
                        name, stats["used"], stats["max"], stats["items"])
