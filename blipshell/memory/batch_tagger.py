"""LLM-powered batch tag assignment.

Sends batches of poorly-tagged memories to an LLM for direct tag assignment.
Designed for overnight/nightly runs when the GPU is free.
"""

from __future__ import annotations

import logging
import re
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
        lines = text.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match "N: tag1, tag2" or "N. tag1, tag2"
            match = re.match(r"^(\d+)[.:]\s*(.+)$", line)
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

        for memory_id, tags in assignments.items():
            try:
                await self.sqlite.tag_memory(memory_id, tags)
                stats["memories_tagged"] += 1
                stats["tags_assigned"] += len(tags)
            except Exception as e:
                logger.error("Failed to tag memory %d: %s", memory_id, e)

        return stats

    async def tag_all(
        self,
        max_batches: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Process multiple batches until no poorly-tagged memories remain.

        Returns combined stats.
        """
        if max_batches is None:
            max_batches = self.config.batch_tag_max_batches

        total_stats = {
            "batches": 0,
            "memories_tagged": 0,
            "tags_assigned": 0,
            "errors": 0,
        }

        for batch_num in range(max_batches):
            if on_status and batch_num % 10 == 0:
                on_status(
                    f"Batch {batch_num + 1}/{max_batches}: "
                    f"{total_stats['memories_tagged']} tagged so far..."
                )

            batch_stats = await self.tag_batch()
            total_stats["batches"] += 1

            if batch_stats["memories_in_batch"] == 0:
                if on_status:
                    on_status("No more poorly-tagged memories.")
                break

            total_stats["memories_tagged"] += batch_stats["memories_tagged"]
            total_stats["tags_assigned"] += batch_stats["tags_assigned"]
            if batch_stats["error"]:
                total_stats["errors"] += 1

        if on_status:
            on_status(
                f"Batch tagging complete: {total_stats['batches']} batches, "
                f"{total_stats['memories_tagged']} memories tagged, "
                f"{total_stats['tags_assigned']} tags assigned, "
                f"{total_stats['errors']} errors."
            )
        return total_stats
