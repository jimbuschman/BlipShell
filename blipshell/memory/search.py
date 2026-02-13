"""Semantic memory search (port of MemoryDB.SearchMemoriesAsync).

Pipeline: noise filter → rephrase query → ChromaDB search → filter by rank → importance boost → sort.
"""

import logging
from dataclasses import dataclass

from blipshell.llm.prompts import rephrase_as_memory_style
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.noise import contains_signal_words, should_skip_memory
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_topics
from blipshell.models.memory import MemorySearchResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result with boosted score."""
    memory_id: int
    text: str
    summary: str
    similarity: float
    boosted_score: float
    rank: int
    importance: float


class MemorySearch:
    """Semantic memory search with importance boosting.

    Port of MemoryDB.SearchMemoriesAsync:
    1. Noise filter (skip noise queries)
    2. Rephrase query as memory-style declarative sentence
    3. ChromaDB semantic search
    4. Filter by rank >= min_threshold
    5. Importance boost based on rank
    6. Sort by boosted score
    """

    def __init__(
        self,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        router: LLMRouter,
        min_rank: int = 3,
        search_limit: int = 20,
    ):
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router
        self.min_rank = min_rank
        self.search_limit = search_limit

    async def search(
        self,
        query: str,
        current_session_id: int | None = None,
        n_results: int | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity.

        Args:
            query: The search query
            current_session_id: Exclude memories from this session
            n_results: Max results to return (defaults to search_limit)

        Returns:
            Sorted list of SearchResult with boosted scores
        """
        if n_results is None:
            n_results = self.search_limit

        # Step 1: Noise filter
        if should_skip_memory(query, max_length=10):
            return []
        if should_skip_memory(query, max_length=20) and not contains_signal_words(query):
            return []

        # Step 2: Rephrase query for better semantic matching
        try:
            memory_query = await self.router.generate(
                TaskType.SUMMARIZATION,
                rephrase_as_memory_style(query),
            )
        except Exception as e:
            logger.warning("Query rephrase failed, using original: %s", e)
            memory_query = query

        # Step 3: ChromaDB semantic search
        chroma_results = self.chroma.search_memories(
            query=memory_query,
            n_results=n_results * 2,  # fetch extra for post-filtering
        )

        if not chroma_results:
            return []

        # Step 4+5: Filter and boost
        results = []
        for cr in chroma_results:
            memory_id = cr["id"]
            similarity = cr["similarity"]

            # Skip if similarity too low
            if similarity < 0.5:
                continue

            # Skip current session memories
            metadata = cr.get("metadata", {})
            if current_session_id and metadata.get("session_id") == str(current_session_id):
                continue

            # Load full memory from SQLite for rank check
            memory = await self.sqlite.get_memory(memory_id)
            if not memory:
                continue

            # Filter by rank
            if memory.rank < self.min_rank:
                continue

            # Importance boost based on rank (port of C# logic)
            normalized_importance = (memory.rank - 1) / 4.0  # 1→0.0, 5→1.0
            importance_boost = normalized_importance * 0.2
            boosted_score = similarity + importance_boost

            results.append(SearchResult(
                memory_id=memory_id,
                text=memory.content,
                summary=memory.summary or memory.content,
                similarity=similarity,
                boosted_score=boosted_score,
                rank=memory.rank,
                importance=memory.importance,
            ))

        # Step 6: Sort by boosted score
        results.sort(key=lambda r: r.boosted_score, reverse=True)
        return results[:n_results]

    async def search_core_memories(self, query: str, n_results: int = 10) -> list[dict]:
        """Search core memories by semantic similarity."""
        return self.chroma.search_core_memories(query, n_results)

    async def search_lessons(self, query: str, n_results: int = 10) -> list[dict]:
        """Search lessons by semantic similarity."""
        return self.chroma.search_lessons(query, n_results)
