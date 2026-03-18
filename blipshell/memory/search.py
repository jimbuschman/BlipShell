"""Semantic memory search (port of MemoryDB.SearchMemoriesAsync).

Pipeline: noise filter → rephrase query → ChromaDB search → filter by rank → importance boost → sort.
"""

import asyncio
import functools
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, tanh

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
    timestamp: datetime | None = None

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
            self.decay_rate = config.decay_rate
            self.decay_rates = config.decay_rates
            self.fts_weight = config.fts_weight
            self.entity_boost = config.entity_boost
            self.project_boost = getattr(config, "project_boost", 0.15)
            self.recency_boost_weight = getattr(config, "recency_boost_weight", 0.15)
            self.min_importance = getattr(config, "min_importance", 0.25)
            self.fts_baseline_similarity = getattr(config, "fts_baseline_similarity", 0.4)
            self.dedup_jaccard_threshold = getattr(config, "dedup_jaccard_threshold", 0.65)
            self.project_session_limit = getattr(config, "project_session_limit", 50)
        else:
            self.min_rank = min_rank
            self.search_limit = search_limit
            self.similarity_threshold = 0.5
            self.importance_boost_weight = 0.2
            self.search_overfetch_multiplier = 2
            self.tag_overlap_boost = 0.1
            self.decay_rate = 0.001
            self.decay_rates = None
            self.fts_weight = 0.3
            self.entity_boost = 0.15
            self.project_boost = 0.15
            self.recency_boost_weight = 0.15
            self.min_importance = 0.25
            self.fts_baseline_similarity = 0.4
            self.dedup_jaccard_threshold = 0.65
            self.project_session_limit = 50
        self.last_search_stats: dict | None = None

    async def search(
        self,
        query: str,
        current_session_id: int | None = None,
        n_results: int | None = None,
        active_project: str | None = None,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity.

        Args:
            query: The search query
            current_session_id: Exclude memories from this session
            n_results: Max results to return (defaults to search_limit)
            active_project: Boost memories from sessions tagged with this project

        Returns:
            Sorted list of SearchResult with boosted scores
        """
        if n_results is None:
            n_results = self.search_limit

        # Step 1: Noise filter
        if should_skip_memory(query, max_length=10):
            self.last_search_stats = {"chroma_hits": 0, "fts_hits": 0, "entity_hits": 0, "post_filter": 0, "final_returned": 0, "skipped": "noise_filter"}
            return []
        if should_skip_memory(query, max_length=20) and not contains_signal_words(query):
            self.last_search_stats = {"chroma_hits": 0, "fts_hits": 0, "entity_hits": 0, "post_filter": 0, "final_returned": 0, "skipped": "noise_filter"}
            return []

        # Pre-load project session IDs for boosting / two-pass search
        project_session_ids: set[int] = set()
        project_hits = 0
        if active_project:
            all_project_sids = await self.sqlite.get_session_ids_for_project(active_project)
            # Limit to most recent N sessions for performant ChromaDB $in filter
            if len(all_project_sids) > self.project_session_limit:
                sorted_sids = sorted(all_project_sids, reverse=True)
                project_session_ids = set(sorted_sids[:self.project_session_limit])
            else:
                project_session_ids = all_project_sids

        overfetch = n_results * self.search_overfetch_multiplier

        # Step 2: ChromaDB semantic search — two-pass when project is active
        # ChromaDB calls are sync + gated (OllamaGate serializes embedding calls).
        # Run in executor so the event loop stays responsive (Esc cancel, etc.).
        loop = asyncio.get_running_loop()
        chroma_results: list[dict] = []
        if project_session_ids:
            # Pass 1: Project-only memories
            project_filter = {"session_id": {"$in": [str(sid) for sid in project_session_ids]}}
            project_chroma = await loop.run_in_executor(
                None, functools.partial(
                    self.chroma.search_memories,
                    query=query, n_results=overfetch, where=project_filter,
                ),
            )
            # Mark project hits for later boosting
            project_chroma_ids = set()
            for cr in project_chroma:
                cr["_project_hit"] = True
                project_chroma_ids.add(cr["id"])
            chroma_results.extend(project_chroma)
            project_hits = len(project_chroma)

            # Pass 2: General (unfiltered) — backfill, dedup by ID
            general_chroma = await loop.run_in_executor(
                None, functools.partial(
                    self.chroma.search_memories,
                    query=query, n_results=overfetch,
                ),
            )
            for cr in general_chroma:
                if cr["id"] not in project_chroma_ids:
                    chroma_results.append(cr)
        else:
            # No project — single-pass search
            chroma_results = await loop.run_in_executor(
                None, functools.partial(
                    self.chroma.search_memories,
                    query=query, n_results=overfetch,
                ),
            )

        # Step 2b: FTS5 keyword search
        fts_results = await self.sqlite.search_fts(
            query, limit=overfetch,
        )

        # Build RRF (Reciprocal Rank Fusion) scores from both result lists
        rrf_k = 60
        rrf_scores: dict[int, float] = {}
        for rank_pos, cr in enumerate(chroma_results):
            rrf_scores[cr["id"]] = 1.0 / (rrf_k + rank_pos)
        for rank_pos, fr in enumerate(fts_results):
            fts_id = fr["id"]
            rrf_scores[fts_id] = rrf_scores.get(fts_id, 0.0) + 1.0 / (rrf_k + rank_pos)

        # Merge FTS-only hits — keyword match gets baseline similarity so they
        # can compete on other scoring signals (importance, recency, tags)
        chroma_ids = {cr["id"] for cr in chroma_results}
        for fr in fts_results:
            if fr["id"] not in chroma_ids:
                chroma_results.append({"id": fr["id"], "similarity": self.fts_baseline_similarity, "metadata": {}})

        if not chroma_results:
            self.last_search_stats = {"chroma_hits": 0, "fts_hits": len(fts_results), "entity_hits": 0, "project_hits": 0, "post_filter": 0, "floor_dropped": 0, "dedup_dropped": 0, "final_returned": 0}
            return []

        # Tag the query (pure regex, <1ms)
        query_tags = set(tag_message(query))

        # Collect candidate memory IDs for batch loading
        candidate_ids = [cr["id"] for cr in chroma_results]
        tags_by_memory = await self.sqlite.get_tags_for_memories(candidate_ids)

        # Batch-load all candidate memories (single query instead of N individual ones)
        memories_batch = await self.sqlite.get_memories_batch(candidate_ids)

        # Step 4+5: Filter and boost
        results = []
        filtered_by_similarity = 0
        filtered_by_session = 0
        filtered_by_importance = 0
        now = datetime.now(timezone.utc)
        for cr in chroma_results:
            memory_id = cr["id"]
            similarity = cr["similarity"]

            # Skip if similarity too low
            if similarity < self.similarity_threshold:
                filtered_by_similarity += 1
                continue

            # Skip current session memories
            metadata = cr.get("metadata", {})
            if current_session_id and metadata.get("session_id") == str(current_session_id):
                filtered_by_session += 1
                continue

            # Load full memory from batch
            memory = memories_batch.get(memory_id)
            if not memory:
                continue

            # Filter by importance (replaces rank filter — continuous, better at scale)
            if memory.importance < self.min_importance:
                filtered_by_importance += 1
                continue

            # Temporal decay — per-type rates so facts persist longer than events
            mem_ts = memory.timestamp if memory.timestamp.tzinfo else memory.timestamp.replace(tzinfo=timezone.utc)
            hours_age = (now - mem_ts).total_seconds() / 3600
            mem_type = memory.memory_type.value if memory.memory_type else "conversation"
            decay = self.decay_rates.get(mem_type) if self.decay_rates else self.decay_rate
            recency_factor = exp(-decay * hours_age)
            # Consolidation — frequently accessed memories resist decay
            consolidation = 1.0 + 0.1 * tanh(memory.access_count / 5)
            importance_boost = memory.importance * self.importance_boost_weight * recency_factor * consolidation

            # Direct recency boost — recent memories get a score bonus that decays over days
            # 0.15 for <1 hour old, ~0.10 for 1 day old, ~0.05 for 3 days, ~0 for 7+ days
            recency_boost = self.recency_boost_weight * exp(-hours_age / 48)

            # Tag overlap boost
            memory_tags = tags_by_memory.get(memory_id, [])
            tag_boost = 0.0
            if query_tags and memory_tags:
                overlap_count = len(query_tags & set(memory_tags))
                tag_boost = (overlap_count / len(query_tags)) * self.tag_overlap_boost

            # RRF boost from hybrid search fusion
            rrf_boost = rrf_scores.get(memory_id, 0.0) * self.fts_weight

            # Project boost — memories from the active project's sessions score higher
            project_boost = 0.0
            if project_session_ids and memory.session_id in project_session_ids:
                project_boost = self.project_boost

            boosted_score = similarity + importance_boost + tag_boost + rrf_boost + project_boost + recency_boost

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
                timestamp=memory.timestamp,
            ))

        # Step 6: Entity graph expansion — find memories connected via entities
        existing_ids = {r.memory_id for r in results}
        entity_memory_ids = []
        matched_entity_names: list[str] = []
        connected_entity_count = 0
        try:
            entity_memory_ids, matched_entity_names, connected_entity_count = await self._expand_via_entities(query)
            for eid in entity_memory_ids:
                if eid in existing_ids:
                    continue
                emem = await self.sqlite.get_memory(eid)
                if not emem or emem.importance < self.min_importance or emem.is_archived:
                    continue
                if current_session_id and emem.session_id == current_session_id:
                    continue
                # Score entity hits dynamically using importance/recency, not a fixed score
                mem_ts = emem.timestamp if emem.timestamp.tzinfo else emem.timestamp.replace(tzinfo=timezone.utc)
                e_hours_age = (now - mem_ts).total_seconds() / 3600
                e_mem_type = emem.memory_type.value if emem.memory_type else "conversation"
                e_decay = self.decay_rates.get(e_mem_type) if self.decay_rates else self.decay_rate
                e_recency_factor = exp(-e_decay * e_hours_age)
                e_importance_boost = emem.importance * self.importance_boost_weight * e_recency_factor
                e_recency_boost = self.recency_boost_weight * exp(-e_hours_age / 48)
                entity_score = self.entity_boost + e_importance_boost + e_recency_boost
                results.append(SearchResult(
                    memory_id=eid,
                    text=emem.content,
                    summary=emem.summary or emem.content,
                    similarity=0.0,
                    boosted_score=entity_score,
                    rank=emem.rank,
                    importance=emem.importance,
                    timestamp=emem.timestamp,
                ))
                existing_ids.add(eid)
        except Exception as e:
            logger.warning("Entity expansion failed: %s", e)

        # Step 7: Sort by boosted score
        results.sort(key=lambda r: r.boosted_score, reverse=True)

        # Score floor removed — similarity threshold + importance filter provide
        # sufficient quality filtering. Score floor caused compounding loss at scale.
        floor_dropped = 0

        # Step 8: Jaccard dedup on summaries — remove near-duplicate results
        dedup_dropped = 0
        if results and self.dedup_jaccard_threshold < 1.0:
            deduped = []
            accepted_word_sets: list[set[str]] = []
            for r in results:
                words = set(r.summary.lower().split())
                is_dup = False
                for accepted_ws in accepted_word_sets:
                    union = len(words | accepted_ws)
                    if union > 0:
                        jaccard = len(words & accepted_ws) / union
                        if jaccard > self.dedup_jaccard_threshold:
                            is_dup = True
                            break
                if is_dup:
                    dedup_dropped += 1
                else:
                    deduped.append(r)
                    accepted_word_sets.append(words)
            results = deduped

        final = results[:n_results]

        # Record access for returned memories (reinforces frequently recalled items)
        accessed_ids = [r.memory_id for r in final]
        if accessed_ids:
            try:
                await self.sqlite.record_memory_access(accessed_ids)
            except Exception as e:
                logger.warning("Failed to record memory access: %s", e)

        # Populate search stats for observability
        self.last_search_stats = {
            "chroma_hits": len(chroma_results),
            "fts_hits": len(fts_results),
            "entity_hits": len(entity_memory_ids),
            "entity_names": matched_entity_names,
            "connected_entities": connected_entity_count,
            "project_hits": project_hits,
            "filtered_by_similarity": filtered_by_similarity,
            "filtered_by_importance": filtered_by_importance,
            "filtered_by_session": filtered_by_session,
            "post_filter": len(results) + floor_dropped + dedup_dropped,
            "floor_dropped": floor_dropped,
            "dedup_dropped": dedup_dropped,
            "final_returned": len(final),
        }

        return final

    # Common words that exist as entities but are too generic for query matching
    ENTITY_STOP_WORDS = frozenset({
        "and", "the", "you", "her", "his", "she", "him", "they", "them",
        "this", "that", "what", "how", "who", "why", "when", "where",
        "was", "were", "has", "had", "have", "are", "not", "but", "for",
        "with", "from", "can", "will", "all", "any", "our", "its",
        "conversation", "something", "anything", "everything", "someone",
    })

    async def _expand_via_entities(self, query: str) -> tuple[list[int], list[str], int]:
        """Find memory IDs connected to entities mentioned in the query.

        1. Load all known entity names (fast — typically hundreds, not millions)
        2. Find entity names that appear in the query (word-boundary match, min 3 chars)
        3. Get their entity IDs
        4. Get connected entity IDs via relationships (capped)
        5. Get memory IDs mentioning any of these entities (capped)

        Returns:
            (memory_ids, matched_entity_names, connected_entity_count)
        """
        entity_names = await self.sqlite.get_all_entity_names()
        if not entity_names:
            return [], [], 0

        # Find entity names present in the query using word-boundary matching.
        # Skip short names (< 3 chars) to prevent "i", "a", "go" matching everything.
        # Skip common stop words that match too broadly.
        # Use \b word boundaries to prevent "time" matching "sometimes".
        query_lower = query.lower()
        matched_names = []
        for name in entity_names:
            if len(name) < 3:
                continue
            if name in self.ENTITY_STOP_WORDS:
                continue
            # Fast substring pre-filter, then word-boundary regex on candidates only
            if name not in query_lower:
                continue
            if re.search(r'\b' + re.escape(name) + r'\b', query_lower):
                matched_names.append(name)
        if not matched_names:
            return [], [], 0

        # Cap matched entities to prevent fan-out from overly generic names
        MAX_MATCHED_ENTITIES = 10
        if len(matched_names) > MAX_MATCHED_ENTITIES:
            # Prefer longer (more specific) entity names
            matched_names.sort(key=len, reverse=True)
            matched_names = matched_names[:MAX_MATCHED_ENTITIES]

        # Get entity IDs for matched names
        entity_ids = await self.sqlite.get_entity_ids_by_names(matched_names)
        if not entity_ids:
            return [], matched_names, 0

        # Get connected entities via relationships (cap expansion)
        MAX_CONNECTED = 20
        connected_ids = await self.sqlite.get_connected_entity_ids(entity_ids)
        if len(connected_ids) > MAX_CONNECTED:
            connected_ids = connected_ids[:MAX_CONNECTED]
        all_entity_ids = entity_ids + connected_ids

        # Get memory IDs mentioning any of these entities (cap results)
        MAX_ENTITY_MEMORIES = 50
        memory_ids = await self.sqlite.get_memory_ids_for_entities(all_entity_ids)
        if len(memory_ids) > MAX_ENTITY_MEMORIES:
            memory_ids = memory_ids[:MAX_ENTITY_MEMORIES]
        return memory_ids, matched_names, len(connected_ids)

    async def search_core_memories(self, query: str, n_results: int = 10) -> list[dict]:
        """Search core memories by semantic similarity."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.chroma.search_core_memories, query, n_results),
        )

    async def search_lessons(
        self, query: str, n_results: int = 10,
        active_project: str | None = None,
    ) -> list[dict]:
        """Search lessons by semantic similarity with optional project boost.

        Lessons from the active project get a similarity boost, but lessons
        from other projects still appear (they may be universally relevant).
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, functools.partial(self.chroma.search_lessons, query, n_results),
        )
        if active_project and results:
            for r in results:
                meta = r.get("metadata", {})
                if meta.get("project") == active_project:
                    r["similarity"] = r.get("similarity", 0.0) + self.project_boost

        # Track lesson usage for staleness analysis
        lesson_ids = [r["id"] for r in results if r.get("id")]
        if lesson_ids:
            try:
                await self.sqlite.increment_lesson_hits(lesson_ids)
            except Exception:
                pass  # non-critical

        return results
