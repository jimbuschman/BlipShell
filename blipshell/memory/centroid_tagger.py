"""Centroid-based tag assignment using embedding vectors.

Computes centroid vectors for each well-populated tag, then assigns tags
to poorly-tagged memories based on cosine similarity — zero LLM cost.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.models.config import MemoryConfig

logger = logging.getLogger(__name__)


class CentroidTagger:
    """Assigns tags to memories via embedding centroid similarity."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        config: MemoryConfig,
    ):
        self.sqlite = sqlite
        self.chroma = chroma
        self.config = config

    async def build_centroids(
        self,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> dict[str, np.ndarray]:
        """Compute centroid embedding vector for each qualifying tag.

        Returns dict mapping tag_name to centroid vector (numpy array).
        """
        min_members = self.config.centroid_tag_min_members
        tag_counts = await self.sqlite.get_tag_member_counts(min_members)
        if not tag_counts:
            if on_status:
                on_status("No tags with enough members for centroid computation.")
            return {}

        if on_status:
            on_status(f"Computing centroids for {len(tag_counts)} tags...")

        centroids: dict[str, np.ndarray] = {}
        for tag_name, count in tag_counts.items():
            memory_ids = await self.sqlite.get_memory_ids_for_tag(
                tag_name, limit=200,
            )
            if not memory_ids:
                continue

            embeddings = self.chroma.get_embeddings_by_ids(memory_ids)
            if not embeddings:
                continue

            vectors = list(embeddings.values())
            centroid = np.mean(vectors, axis=0)
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[tag_name] = centroid

        if on_status:
            on_status(f"Built {len(centroids)} tag centroids.")
        return centroids

    async def tag_poorly_tagged(
        self,
        centroids: dict[str, np.ndarray],
        on_status: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Assign tags to poorly-tagged memories using centroid similarity.

        Returns stats dict with tagged count and tags assigned.
        """
        threshold = self.config.centroid_tag_similarity
        batch_size = self.config.centroid_tag_batch_size
        stats = {"memories_checked": 0, "memories_tagged": 0, "tags_assigned": 0}

        if not centroids:
            return stats

        # Pre-stack centroid vectors for vectorized similarity
        tag_names = list(centroids.keys())
        centroid_matrix = np.stack([centroids[t] for t in tag_names])  # (N_tags, dim)

        memory_ids = await self.sqlite.get_poorly_tagged_memory_ids(
            max_tags=1, limit=batch_size,
        )
        if not memory_ids:
            if on_status:
                on_status("No poorly-tagged memories to process.")
            return stats

        if on_status:
            on_status(f"Processing {len(memory_ids)} poorly-tagged memories...")

        # Get existing tags so we don't re-assign
        existing_tags = await self.sqlite.get_tags_for_memories(memory_ids)

        # Retrieve embeddings in batches to control memory
        chunk_size = 500
        for i in range(0, len(memory_ids), chunk_size):
            chunk_ids = memory_ids[i:i + chunk_size]
            embeddings = self.chroma.get_embeddings_by_ids(chunk_ids)

            for mid in chunk_ids:
                if mid not in embeddings:
                    continue
                stats["memories_checked"] += 1

                vec = np.array(embeddings[mid])
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                # Cosine similarity against all centroids (dot product of unit vectors)
                similarities = centroid_matrix @ vec  # (N_tags,)

                # Find tags above threshold that aren't already assigned
                current_tags = set(existing_tags.get(mid, []))
                new_tags = []
                for idx in np.where(similarities >= threshold)[0]:
                    tag = tag_names[idx]
                    if tag not in current_tags:
                        new_tags.append(tag)

                if new_tags:
                    # Cap at 5 new tags per memory
                    new_tags = new_tags[:5]
                    await self.sqlite.tag_memory(mid, new_tags)
                    stats["memories_tagged"] += 1
                    stats["tags_assigned"] += len(new_tags)

        if on_status:
            on_status(
                f"Centroid tagging: {stats['memories_tagged']} memories tagged, "
                f"{stats['tags_assigned']} tags assigned."
            )
        return stats

    async def run(
        self,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Build centroids and tag poorly-tagged memories. Returns combined stats."""
        centroids = await self.build_centroids(on_status=on_status)
        stats = await self.tag_poorly_tagged(centroids, on_status=on_status)
        stats["centroids_built"] = len(centroids)
        return stats
