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
from blipshell.memory.tagger import tag_message
from blipshell.models.config import MemoryConfig
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
    tags: list[str] = None
    tag_boost: float = 0.0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


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
        config: MemoryConfig | None = None,
        min_rank: int = 3,
        search_limit: int = 20,
    ):
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router
        # Use config values if provided, else fall back to explicit params
        if config:
            self.min_rank = config.min_rank_threshold
            self.search_limit = config.recall_search_limit
            self.similarity_threshold = config.similarity_threshold
            self.importance_boost_weight = config.importance_boost_weight
            self.search_overfetch_multiplier = config.search_overfetch_multiplier
            self.tag_overlap_boost = config.tag_overlap_boost
        else:
            self.min_rank = min_rank
            self.search_limit = search_limit
            self.similarity_threshold = 0.5
            self.importance_boost_weight = 0.2
            self.search_overfetch_multiplier = 2
            self.tag_overlap_boost = 0.1

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

        # Step 2: ChromaDB semantic search (nomic-embed-text handles
        # question-to-statement matching well enough without rephrasing)
        chroma_results = self.chroma.search_memories(
            query=query,
            n_results=n_results * self.search_overfetch_multiplier,
        )

        if not chroma_results:
            return []

        # Tag the query (pure regex, <1ms)
        query_tags = set(tag_message(query))

        # Collect candidate memory IDs for batch tag loading
        candidate_ids = [cr["id"] for cr in chroma_results]
        tags_by_memory = await self.sqlite.get_tags_for_memories(candidate_ids)

        # Step 4+5: Filter and boost
        results = []
        for cr in chroma_results:
            memory_id = cr["id"]
            similarity = cr["similarity"]

            # Skip if similarity too low
            if similarity < self.similarity_threshold:
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

            # Importance boost — uses the stored importance score (0.0-1.0)
            # which already factors in LLM assessment + recency/tag bonuses
            importance_boost = memory.importance * self.importance_boost_weight

            # Tag overlap boost
            memory_tags = tags_by_memory.get(memory_id, [])
            tag_boost = 0.0
            if query_tags and memory_tags:
                overlap_count = len(query_tags & set(memory_tags))
                tag_boost = (overlap_count / len(query_tags)) * self.tag_overlap_boost

            boosted_score = similarity + importance_boost + tag_boost

            results.append(SearchResult(
                memory_id=memory_id,
                text=memory.content,
                summary=memory.summary or memory.content,
                similarity=similarity,
                boosted_score=boosted_score,
                rank=memory.rank,
                importance=memory.importance,
                tags=memory_tags,
                tag_boost=tag_boost,
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
