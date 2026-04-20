"""Memory consolidation — merge near-duplicate memories.

After importing thousands of conversations, the memory store may contain
clusters of near-duplicate memories (same topic phrased slightly differently).
This module finds and merges them using ChromaDB cosine similarity, keeping the
higher-quality memory and transferring tags/access counts from the loser.

No LLM calls needed — pure vector similarity + SQLite operations.
"""

import logging
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
            config.consolidation_batch_size if config else 100
        )

    async def consolidate_batch(self) -> dict:
        """Process a batch of unchecked memories. Returns stats dict.

        For each unchecked memory, queries ChromaDB for top-5 neighbors.
        If any neighbor (excluding self) exceeds the similarity threshold,
        merges the pair (keep winner, delete loser).
        """
        stats = {"checked": 0, "merged": 0, "errors": 0}

        # 1. Get batch of unconsolidated memory IDs
        memory_ids = await self.sqlite.get_unconsolidated_memory_ids(
            limit=self.batch_size,
        )
        if not memory_ids:
            return stats

        # Track IDs that have been deleted (losers) so we skip them
        deleted_ids: set[int] = set()
        # Track IDs that were successfully checked (for marking consolidated)
        checked_ids: list[int] = []

        for memory_id in memory_ids:
            if memory_id in deleted_ids:
                continue

            try:
                duplicates = await self._find_duplicates(memory_id)
                stats["checked"] += 1
                checked_ids.append(memory_id)

                for dup_id, similarity in duplicates:
                    if dup_id in deleted_ids:
                        continue
                    winner_id, loser_id = await self._pick_winner(
                        memory_id, dup_id,
                    )
                    await self._merge_memories(winner_id, loser_id)
                    deleted_ids.add(loser_id)
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

        # Mark all checked (non-deleted) memories as consolidated
        final_checked = [mid for mid in checked_ids if mid not in deleted_ids]
        if final_checked:
            await self.sqlite.mark_memories_consolidated(final_checked)

        return stats

    async def _find_duplicates(self, memory_id: int) -> list[tuple[int, float]]:
        """Query ChromaDB for neighbors above the similarity threshold.

        Returns list of (neighbor_id, similarity) excluding self.
        """
        memory = await self.sqlite.get_memory(memory_id)
        if not memory or not memory.summary:
            return []

        results = self.vectors.search_memories(
            query=memory.summary,
            n_results=5,
        )

        duplicates = []
        for r in results:
            neighbor_id = r["id"]
            similarity = r["similarity"]
            if neighbor_id == memory_id:
                continue
            if similarity >= self.similarity_threshold:
                duplicates.append((neighbor_id, similarity))

        return duplicates

    async def _pick_winner(self, id_a: int, id_b: int) -> tuple[int, int]:
        """Decide which memory to keep (winner) and which to delete (loser).

        Winner = higher importance (tiebreak: higher rank, then newer timestamp).
        Returns (winner_id, loser_id).
        """
        mem_a = await self.sqlite.get_memory(id_a)
        mem_b = await self.sqlite.get_memory(id_b)

        if mem_a is None:
            return id_b, id_a
        if mem_b is None:
            return id_a, id_b

        # Compare: importance > rank > timestamp (newer wins)
        key_a = (mem_a.importance, mem_a.rank, mem_a.timestamp)
        key_b = (mem_b.importance, mem_b.rank, mem_b.timestamp)

        if key_a >= key_b:
            return id_a, id_b
        return id_b, id_a

    async def _merge_memories(self, winner_id: int, loser_id: int):
        """Transfer metadata from loser to winner, then delete loser.

        - Transfers all tags (union, skip duplicates)
        - Sums access_count values
        - Keeps the longer summary
        - Deletes loser from SQLite (cascade cleans tags), ChromaDB, and FTS5
        """
        winner = await self.sqlite.get_memory(winner_id)
        loser = await self.sqlite.get_memory(loser_id)
        if not winner or not loser:
            return

        # Transfer tags from loser to winner
        await self.sqlite.transfer_memory_tags(from_id=loser_id, to_id=winner_id)

        # Sum access counts
        combined_access = (winner.access_count or 0) + (loser.access_count or 0)

        # Keep the longer summary (more informative)
        update_fields: dict = {"access_count": combined_access}
        winner_summary_len = len(winner.summary) if winner.summary else 0
        loser_summary_len = len(loser.summary) if loser.summary else 0
        if loser_summary_len > winner_summary_len:
            update_fields["summary"] = loser.summary

        # Update winner — access_count isn't in the allowed set of update_memory,
        # so we do a direct SQL update
        if "summary" in update_fields:
            await self.sqlite.update_memory(winner_id, summary=update_fields["summary"])
        await self.sqlite._db.execute(
            "UPDATE memories SET access_count = ? WHERE id = ?",
            (combined_access, winner_id),
        )
        await self.sqlite._db.commit()

        # Delete loser from SQLite (FTS5 trigger handles FTS cleanup).
        # The orphan vector will be cleaned up by cleanup_orphan_vectors()
        # in the prune job — no per-ID vector delete here to avoid lock
        # contention between async SQLite and sync vector connections.
        await self.sqlite.delete_memory(loser_id)
