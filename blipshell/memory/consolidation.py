"""Memory consolidation — merge near-duplicate memories.

After importing thousands of conversations the store holds clusters of
near-duplicates (same topic, slightly different phrasing). This finds them by
cosine similarity over the vectors already in `vec_memories` and folds each
cluster into its best member.

No LLM calls, and as of 2026-08-06 no embedding calls either: it queries with
each memory's STORED vector. The previous version re-embedded every candidate
through Ollama, one HTTP round trip per memory checked, which together with a
batch size of 20 meant roughly 40 memories a day against a 17K corpus — about
fourteen months for a single sweep. The deep dive called it "effectively
decorative" and that was fair.

Losers are ARCHIVED, never deleted. Deleting cascaded `entity_relationships`
and `entity_mentions` away via ON DELETE CASCADE, which quietly violated the
archive-never-delete mandate at the edge level: a memory merge could destroy
graph structure that had nothing to do with the duplication.
"""

import asyncio
import logging
import time
from typing import Optional

from blipshell.memory.vector_store import VectorStore
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Finds and merges near-duplicate memories."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        vectors: VectorStore,
        config: Optional[MemoryConfig] = None,
    ):
        self.sqlite = sqlite
        self.vectors = vectors
        self.similarity_threshold = (
            config.consolidation_similarity if config else 0.85
        )
        self.batch_size = (
            config.consolidation_batch_size if config else 500
        )
        self.dry_run = bool(getattr(config, "consolidation_dry_run", False)) if config else False
        self.neighbors_k = 5

    async def consolidate_batch(
        self, time_budget_seconds: float | None = None,
    ) -> dict:
        """Process a batch of unchecked memories. Returns stats dict.

        Work is resumable: memories are marked consolidated as they're checked,
        so running out of budget just means the rest are picked up next night
        (same pattern as merge_entities and batch_tag).
        """
        stats = {
            "checked": 0, "merged": 0, "errors": 0,
            "no_vector": 0, "stopped_early": False,
        }
        if self.dry_run:
            stats["dry_run"] = True

        memory_ids = await self.sqlite.get_unconsolidated_memory_ids(
            limit=self.batch_size,
        )
        if not memory_ids:
            return stats

        # All vector work in ONE executor hop: the store is synchronous, and a
        # KNN scan per memory would otherwise block the event loop for the
        # whole batch.
        loop = asyncio.get_running_loop()
        try:
            neighbors = await loop.run_in_executor(
                None, self.vectors.find_neighbors, memory_ids, self.neighbors_k,
            )
        except Exception as e:
            logger.error("Consolidation neighbor lookup failed: %s", e)
            stats["errors"] += 1
            return stats

        archived_ids: set[int] = set()
        checked_ids: list[int] = []
        deadline = (
            time.monotonic() + time_budget_seconds if time_budget_seconds else None
        )

        for memory_id in memory_ids:
            if deadline and time.monotonic() >= deadline:
                stats["stopped_early"] = True
                logger.info(
                    "Consolidation stopped early at %d/%d (time budget)",
                    stats["checked"], len(memory_ids),
                )
                break
            if memory_id in archived_ids:
                continue

            found = neighbors.get(memory_id)
            if found is None:
                # Never embedded — nothing to compare against. Do NOT mark it
                # consolidated, or a backfilled vector would never be checked.
                stats["no_vector"] += 1
                continue

            try:
                stats["checked"] += 1
                checked_ids.append(memory_id)
                for dup_id, similarity in found:
                    if dup_id in archived_ids or similarity < self.similarity_threshold:
                        continue
                    winner_id, loser_id = await self._pick_winner(memory_id, dup_id)
                    if self.dry_run:
                        logger.info(
                            "[dry-run] would merge %d into %d (similarity=%.3f)",
                            loser_id, winner_id, similarity,
                        )
                    else:
                        await self._merge_memories(winner_id, loser_id)
                    archived_ids.add(loser_id)
                    stats["merged"] += 1
                    logger.debug(
                        "Merged memory %d into %d (similarity=%.3f)",
                        loser_id, winner_id, similarity,
                    )
            except Exception as e:
                stats["errors"] += 1
                logger.warning(
                    "Consolidation error for memory %d: %s", memory_id, e,
                )

        final_checked = [mid for mid in checked_ids if mid not in archived_ids]
        if final_checked and not self.dry_run:
            await self.sqlite.mark_memories_consolidated(final_checked)

        return stats

    async def _pick_winner(self, id_a: int, id_b: int) -> tuple[int, int]:
        """Decide which memory to keep (winner) and which to archive (loser).

        Winner = higher importance (tiebreak: higher rank, then newer timestamp).
        Returns (winner_id, loser_id).
        """
        mem_a = await self.sqlite.get_memory(id_a)
        mem_b = await self.sqlite.get_memory(id_b)

        if mem_a is None:
            return id_b, id_a
        if mem_b is None:
            return id_a, id_b

        key_a = (mem_a.importance, mem_a.rank, mem_a.timestamp)
        key_b = (mem_b.importance, mem_b.rank, mem_b.timestamp)
        return (id_a, id_b) if key_a >= key_b else (id_b, id_a)

    async def _merge_memories(self, winner_id: int, loser_id: int):
        """Fold the loser into the winner, then archive the loser.

        - Transfers tags (union)
        - Sums access counts
        - Keeps the longer summary
        - ARCHIVES the loser. Deleting it would cascade its entity edges and
          mentions away, destroying graph structure unrelated to the merge.
          Archived memories are already excluded from every search path, and
          the nightly orphan sweep reclaims their vectors.
        """
        winner = await self.sqlite.get_memory(winner_id)
        loser = await self.sqlite.get_memory(loser_id)
        if not winner or not loser:
            return

        await self.sqlite.transfer_memory_tags(from_id=loser_id, to_id=winner_id)

        combined_access = (winner.access_count or 0) + (loser.access_count or 0)
        updates: dict = {"access_count": combined_access}
        if len(loser.summary or "") > len(winner.summary or ""):
            updates["summary"] = loser.summary
        await self.sqlite.update_memory(winner_id, **updates)

        await self.sqlite.update_memory(loser_id, is_archived=True)
