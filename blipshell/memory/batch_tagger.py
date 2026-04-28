"""LLM-powered batch tag assignment.

Sends batches of poorly-tagged memories to an LLM for direct tag assignment.
Designed for overnight/nightly runs when the GPU is free.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.models.config import MemoryConfig

from blipshell.llm.prompts import batch_assign_tags
from blipshell.llm.router import TaskType

logger = logging.getLogger(__name__)


class BatchTagger:
    """Assigns tags to memories via LLM batch calls."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        router: LLMRouter,
        config: MemoryConfig,
        allow_new_tags: bool = False,
    ):
        self.sqlite = sqlite
        self.router = router
        self.config = config
        self.allow_new_tags = allow_new_tags

    async def _get_available_tags(self) -> list[str]:
        """Get all known tag names for the prompt."""
        return await self.sqlite.get_all_tag_names()

    async def _load_batch(self, batch_size: int) -> list[tuple[int, str]]:
        """Load a batch of poorly-tagged memories (id, summary)."""
        memory_ids = await self.sqlite.get_poorly_tagged_memory_ids(
            max_tags=1, limit=batch_size,
        )
        if not memory_ids:
            return []

        result = []
        for mid in memory_ids:
            cursor = await self.sqlite._db.execute(
                "SELECT summary FROM memories WHERE id = ? AND summary IS NOT NULL",
                (mid,),
            )
            row = await cursor.fetchone()
            if row and row["summary"]:
                result.append((mid, row["summary"]))
        return result

    def _parse_response(
        self,
        text: str,
        summaries: list[tuple[int, str]],
        valid_tags: set[str],
        allow_new_tags: bool = False,
    ) -> dict[int, list[str]]:
        """Parse LLM response into {memory_id: [tags]}.

        Expected format: "1: tag1, tag2, tag3"
        Lenient parsing: handles extra whitespace.

        When allow_new_tags is False (default), only tags in valid_tags are kept.
        When True, any well-formed tag is accepted (enables vocabulary growth).
        """
        assignments: dict[int, list[str]] = {}
        # Strip thinking tokens if present (qwen3 and some cloud models)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        lines = cleaned.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip markdown bold/formatting
            line = re.sub(r"\*+", "", line)
            line = line.strip()

            # Match "N: tag1, tag2" or "N. tag1, tag2" or "N) tag1, tag2"
            match = re.match(r"^(\d+)[.:\-)]\s*(.+)$", line)
            if not match:
                continue

            idx = int(match.group(1)) - 1  # 1-indexed to 0-indexed
            tags_str = match.group(2).strip()

            if idx < 0 or idx >= len(summaries):
                continue
            if tags_str.upper() == "NONE":
                continue

            memory_id = summaries[idx][0]
            tags = []
            for tag in re.split(r"[,;]+", tags_str):
                tag = tag.strip().lower()
                # Sanitize: only keep reasonable tag names
                tag = re.sub(r"[^a-z0-9\-_]", "", tag)
                if not tag or len(tag) < 2 or len(tag) > 40:
                    continue
                if allow_new_tags or tag in valid_tags:
                    tags.append(tag)

            if tags:
                assignments[memory_id] = tags[:5]  # cap at 5 per memory

        return assignments

    async def tag_batch(self) -> dict:
        """Process one batch of poorly-tagged memories via LLM.

        Returns stats: {memories_in_batch, memories_tagged, tags_assigned, error}.
        """
        batch_size = self.config.batch_tag_batch_size
        summaries = await self._load_batch(batch_size)
        stats = {
            "memories_in_batch": len(summaries),
            "memories_tagged": 0,
            "tags_assigned": 0,
            "error": None,
        }

        if not summaries:
            return stats

        available_tags = await self._get_available_tags()
        if not available_tags:
            stats["error"] = "No tags in database"
            return stats

        valid_tags = set(available_tags)
        system_prompt, user_prompt = batch_assign_tags(summaries, available_tags)

        try:
            response = await self.router.generate(
                TaskType.RANKING,  # routes to qwen2.5:14b — 20x faster, similar quality
                user_prompt,
                system=system_prompt,
            )
        except Exception as e:
            logger.error("Batch tagger LLM call failed: %s", e)
            stats["error"] = str(e)
            return stats

        assignments = self._parse_response(response, summaries, valid_tags, allow_new_tags=self.allow_new_tags)

        tagged_ids = set()
        for memory_id, tags in assignments.items():
            try:
                await self.sqlite.tag_memory(memory_id, tags)
                stats["memories_tagged"] += 1
                stats["tags_assigned"] += len(tags)
                tagged_ids.add(memory_id)
            except Exception as e:
                logger.error("Failed to tag memory %d: %s", memory_id, e)

        # Mark untagged memories with _skip so they leave the poorly-tagged
        # pool and don't cycle through every batch forever. These are either
        # too short/generic to tag or the model returned NONE for them.
        if self.allow_new_tags:
            for mid, _ in summaries:
                if mid not in tagged_ids:
                    try:
                        await self.sqlite.tag_memory(mid, ["_skip"])
                    except Exception:
                        pass

        return stats

    async def tag_all(
        self,
        max_batches: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
        time_budget_seconds: Optional[float] = None,
    ) -> dict:
        """Process multiple batches until no poorly-tagged memories remain.

        Args:
            max_batches: hard cap on batches per run (defaults to config).
            on_status: progress callback.
            time_budget_seconds: if set, exit cleanly before this many seconds
                elapse. Reserves a margin for the next batch + safety so we
                don't get killed by an outer wait_for. Stats include
                ``stopped_early`` and ``stop_reason`` when triggered.

        Returns combined stats.
        """
        if max_batches is None:
            max_batches = self.config.batch_tag_max_batches

        total_stats = {
            "batches": 0,
            "memories_tagged": 0,
            "tags_assigned": 0,
            "errors": 0,
            "stopped_early": False,
            "stop_reason": None,
        }

        deadline: Optional[float] = None
        if time_budget_seconds is not None:
            deadline = time.monotonic() + time_budget_seconds

        # Running estimate of per-batch cost; refined each iteration so the
        # budget check adapts to whichever endpoint actually serves the job.
        avg_batch_seconds = 0.0  # unset until first real measurement
        safety_margin_seconds = 5.0

        for batch_num in range(max_batches):
            # Time-budget gate: leave room for ~one more batch + safety.
            # Skip on the first iteration — we always run at least one
            # batch so we have a real measurement and make some progress
            # even on tight budgets.
            if deadline is not None and total_stats["batches"] > 0:
                remaining = deadline - time.monotonic()
                if remaining < avg_batch_seconds + safety_margin_seconds:
                    total_stats["stopped_early"] = True
                    total_stats["stop_reason"] = (
                        f"time budget reached after {total_stats['batches']} "
                        f"batches ({remaining:.1f}s remaining < "
                        f"{avg_batch_seconds:.1f}s avg + {safety_margin_seconds:.1f}s margin)"
                    )
                    if on_status:
                        on_status(total_stats["stop_reason"])
                    break

            if on_status and batch_num % 10 == 0:
                on_status(
                    f"Batch {batch_num + 1}/{max_batches}: "
                    f"{total_stats['memories_tagged']} tagged so far..."
                )

            batch_start = time.monotonic()
            batch_stats = await self.tag_batch()
            batch_elapsed = time.monotonic() - batch_start
            total_stats["batches"] += 1

            if batch_stats["memories_in_batch"] == 0:
                if on_status:
                    on_status("No more poorly-tagged memories.")
                break

            total_stats["memories_tagged"] += batch_stats["memories_tagged"]
            total_stats["tags_assigned"] += batch_stats["tags_assigned"]
            if batch_stats["error"]:
                total_stats["errors"] += 1

            # Refine running average: first measurement seeds the estimate,
            # then EWMA so rate-limit slowdowns reflect quickly in the gate.
            if avg_batch_seconds == 0.0:
                avg_batch_seconds = batch_elapsed
            else:
                avg_batch_seconds = 0.5 * avg_batch_seconds + 0.5 * batch_elapsed

        if on_status:
            on_status(
                f"Batch tagging complete: {total_stats['batches']} batches, "
                f"{total_stats['memories_tagged']} memories tagged, "
                f"{total_stats['tags_assigned']} tags assigned, "
                f"{total_stats['errors']} errors."
                + (f" (stopped early: {total_stats['stop_reason']})"
                   if total_stats["stopped_early"] else "")
            )
        return total_stats
