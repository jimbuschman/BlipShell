"""Semantic memory search (port of MemoryDB.SearchMemoriesAsync).

Pipeline: noise filter → rephrase query → ChromaDB search → filter by rank → importance boost → sort.
"""

import asyncio
import functools
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp

from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.vector_store import VectorStore
from blipshell.memory.noise import contains_signal_words, should_skip_memory
from blipshell.memory.reranker import Reranker
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
        vectors: VectorStore,
        router: LLMRouter,
        config: MemoryConfig | None = None,
        min_rank: int = 3,
        search_limit: int = 20,
        ollama_url: str = "http://localhost:11434",
    ):
        self.sqlite = sqlite
        self.vectors = vectors
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
            self.fadem_enabled = getattr(config, "fadem_enabled", True)
            self.fadem_base_rate = getattr(config, "fadem_base_rate", 0.001)
            self.fadem_importance_factor = getattr(config, "fadem_importance_factor", 2.0)
            self.fadem_access_hours = getattr(config, "fadem_access_hours", 24.0)
            self.min_importance = getattr(config, "min_importance", 0.25)
            self.fts_baseline_similarity = getattr(config, "fts_baseline_similarity", 0.4)
            self.dedup_jaccard_threshold = getattr(config, "dedup_jaccard_threshold", 0.65)
            self.project_session_limit = getattr(config, "project_session_limit", 50)
            # Reranker config
            self.reranker_enabled = getattr(config, "reranker_enabled", False)
            self.reranker_model = getattr(config, "reranker_model", "dengcao/Qwen3-Reranker-0.6B:Q8_0")
            self.reranker_top_n = getattr(config, "reranker_top_n", 15)
            self.reranker_weight = getattr(config, "reranker_weight", 0.4)
            self.reranker_instruction = getattr(config, "reranker_instruction", "")
        else:
            self.min_rank = min_rank
            self.search_limit = search_limit
            self.similarity_threshold = 0.5
            self.importance_boost_weight = 0.2
            self.search_overfetch_multiplier = 2
            self.tag_overlap_boost = 0.1
            self.decay_rate = 0.001
            from blipshell.models.config import DecayRatesConfig
            self.decay_rates = DecayRatesConfig()
            self.fts_weight = 0.3
            self.entity_boost = 0.15
            self.project_boost = 0.15
            self.recency_boost_weight = 0.15
            self.fadem_enabled = True
            self.fadem_importance_factor = 2.0
            self.fadem_base_rate = 0.001
            self.fadem_access_hours = 24.0
            self.min_importance = 0.25
            self.fts_baseline_similarity = 0.4
            self.dedup_jaccard_threshold = 0.65
            self.project_session_limit = 50
            self.reranker_enabled = False
            self.reranker_model = "dengcao/Qwen3-Reranker-0.6B:Q8_0"
            self.reranker_top_n = 15
            self.reranker_weight = 0.4
            self.reranker_instruction = ""
        self.last_search_stats: dict | None = None
        self._ollama_url = ollama_url
        self._reranker: Reranker | None = None

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

        # Step 1: Noise filter — only skip truly empty/noise queries.
        # The old filter killed valid short queries like "where do I work" (16 chars,
        # no signal words). Search queries are user intent — they should almost always
        # proceed. Only skip single-word noise and known greetings.
        stripped = query.strip()
        if not stripped or len(stripped) < 3:
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
        _t_start = time.monotonic()

        # Step 2: ChromaDB semantic search — two-pass when project is active
        # ChromaDB calls are sync + gated (OllamaGate serializes embedding calls).
        # Run in executor so the event loop stays responsive (Esc cancel, etc.).
        _t_chroma_start = time.monotonic()
        loop = asyncio.get_running_loop()
        chroma_results: list[dict] = []
        if project_session_ids:
            # Pass 1: Project-only memories
            project_filter = {"session_id": {"$in": [str(sid) for sid in project_session_ids]}}
            project_chroma = await loop.run_in_executor(
                None, functools.partial(
                    self.vectors.search_memories,
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
                    self.vectors.search_memories,
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
                    self.vectors.search_memories,
                    query=query, n_results=overfetch,
                ),
            )

        _t_chroma_ms = (time.monotonic() - _t_chroma_start) * 1000

        # Step 2b: FTS5 keyword search
        _t_fts_start = time.monotonic()
        fts_results = await self.sqlite.search_fts(
            query, limit=overfetch,
        )
        _t_fts_ms = (time.monotonic() - _t_fts_start) * 1000

        # Build RRF (Reciprocal Rank Fusion) scores from both result lists
        rrf_k = 60
        rrf_scores: dict[int, float] = {}
        for rank_pos, cr in enumerate(chroma_results):
            rrf_scores[cr["id"]] = 1.0 / (rrf_k + rank_pos)
        for rank_pos, fr in enumerate(fts_results):
            fts_id = fr["id"]
            rrf_scores[fts_id] = rrf_scores.get(fts_id, 0.0) + 1.0 / (rrf_k + rank_pos)

        # Merge FTS-only hits — keyword match gets baseline similarity so they
        # can compete on other scoring signals (importance, recency, tags).
        # FTS hits are flagged so they bypass the similarity threshold filter —
        # a keyword match is a strong signal regardless of embedding distance.
        chroma_ids = {cr["id"] for cr in chroma_results}
        for fr in fts_results:
            if fr["id"] not in chroma_ids:
                chroma_results.append({"id": fr["id"], "similarity": self.fts_baseline_similarity, "metadata": {}, "fts_match": True})

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

            # Skip if similarity too low — but never drop FTS keyword matches,
            # which are strong recall signals regardless of embedding distance.
            if similarity < self.similarity_threshold and not cr.get("fts_match"):
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

            # Importance signal (intrinsic memory quality, no decay)
            importance_boost = memory.importance * self.importance_boost_weight

            # Recency boost — FadeMem or flat.
            # FadeMem uses per-type decay rates modulated by importance and
            # access count so important/frequently-recalled memories stay
            # relevant for weeks/months instead of dying after 48h.
            if not memory.timestamp:
                hours_age = 720.0
            else:
                mem_ts = memory.timestamp if memory.timestamp.tzinfo else memory.timestamp.replace(tzinfo=timezone.utc)
                hours_age = (now - mem_ts).total_seconds() / 3600

            if self.fadem_enabled:
                # Importance-modulated decay: imp=1.0 divides rate by 3x (with factor=2.0)
                # Uniform base rate — no type differentiation (type classification isn't
                # reliable enough to weight this heavily; importance is the cleaner signal)
                effective_rate = self.fadem_base_rate / (1.0 + memory.importance * self.fadem_importance_factor)
                # Access strengthening: each retrieval subtracts hours from effective age
                effective_hours = max(0.0, hours_age - memory.access_count * self.fadem_access_hours)
                recency_boost = self.recency_boost_weight * exp(-effective_rate * effective_hours)
            else:
                # Flat 48h half-life fallback
                recency_boost = self.recency_boost_weight * exp(-hours_age / 48)

            # Tag overlap boost
            memory_tags = tags_by_memory.get(memory_id, [])
            tag_boost = 0.0
            if query_tags and memory_tags:
                overlap_count = len(query_tags & set(memory_tags))
                tag_boost = (overlap_count / len(query_tags)) * self.tag_overlap_boost

            # RRF boost from hybrid search fusion — keyword matches are strong signal.
            # When a specific term like "OllamaGate" appears literally in content,
            # that's more reliable than semantic similarity to "contention".
            rrf_boost = rrf_scores.get(memory_id, 0.0) * self.fts_weight * 2.0

            # Project boost — memories from the active project's sessions score much higher.
            # +0.5 ensures project memories dominate over general memories with similar
            # similarity. A general memory needs ~0.5 higher semantic similarity to outrank
            # a project memory, which is the right tradeoff when project context is active.
            project_boost = 0.0
            if project_session_ids and memory.session_id in project_session_ids:
                project_boost = 0.5

            boosted_score = similarity + importance_boost + recency_boost + rrf_boost + project_boost + tag_boost

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
        # Capped at 15 results (was 50) — entity results should supplement
        # semantic search, not dominate it. Batch-loaded to avoid N+1 queries.
        _t_entity_start = time.monotonic()
        existing_ids = {r.memory_id for r in results}
        entity_memory_ids = []
        matched_entity_names: list[str] = []
        connected_entity_count = 0
        try:
            entity_memory_ids, matched_entity_names, connected_entity_count = await self._expand_via_entities(query)
            # Filter out already-found IDs before batch loading
            new_eids = [eid for eid in entity_memory_ids if eid not in existing_ids]
            entity_batch = await self.sqlite.get_memories_batch(new_eids) if new_eids else {}
            for eid in new_eids:
                emem = entity_batch.get(eid)
                if not emem or emem.importance < self.min_importance or emem.is_archived:
                    continue
                if current_session_id and emem.session_id == current_session_id:
                    continue
                # Score: entity boost + importance + recency (same FadeMem formula)
                if not emem.timestamp:
                    e_hours_age = 720.0
                else:
                    mem_ts = emem.timestamp if emem.timestamp.tzinfo else emem.timestamp.replace(tzinfo=timezone.utc)
                    e_hours_age = (now - mem_ts).total_seconds() / 3600
                if self.fadem_enabled:
                    e_eff_rate = self.fadem_base_rate / (1.0 + emem.importance * self.fadem_importance_factor)
                    e_eff_hours = max(0.0, e_hours_age - emem.access_count * self.fadem_access_hours)
                    e_recency_boost = self.recency_boost_weight * exp(-e_eff_rate * e_eff_hours)
                else:
                    e_recency_boost = self.recency_boost_weight * exp(-e_hours_age / 48)
                entity_score = self.entity_boost + (emem.importance * self.importance_boost_weight) + e_recency_boost
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

        _t_entity_ms = (time.monotonic() - _t_entity_start) * 1000

        # Step 6b: Reranker — rescore top candidates using cross-encoder model.
        # Runs after all boosting signals are applied so it can blend with them.
        _t_rerank_start = time.monotonic()
        reranker_hits = 0
        if self.reranker_enabled and results:
            try:
                reranker_hits = await self._apply_reranking(query, results)
            except Exception as e:
                logger.warning("Reranking failed, using original scores: %s", e)
        _t_rerank_ms = (time.monotonic() - _t_rerank_start) * 1000

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

        # Record access for returned memories (reinforces frequently recalled items).
        # Best-effort — don't block search or spam logs if the worker holds the lock.
        accessed_ids = [r.memory_id for r in final]
        if accessed_ids:
            try:
                await self.sqlite.record_memory_access(accessed_ids)
            except Exception:
                pass  # write lock contention — harmless, access count is non-critical

        _t_total_ms = (time.monotonic() - _t_start) * 1000

        # Populate search stats for observability (includes timing)
        self.last_search_stats = {
            "chroma_hits": len(chroma_results),
            "fts_hits": len(fts_results),
            "entity_hits": len(entity_memory_ids),
            "entity_names": matched_entity_names,
            "connected_entities": connected_entity_count,
            "project_hits": project_hits,
            "reranker_hits": reranker_hits,
            "filtered_by_similarity": filtered_by_similarity,
            "filtered_by_importance": filtered_by_importance,
            "filtered_by_session": filtered_by_session,
            "post_filter": len(results) + floor_dropped + dedup_dropped,
            "floor_dropped": floor_dropped,
            "dedup_dropped": dedup_dropped,
            "final_returned": len(final),
            # Timing (ms) per component — shows up in /flow output
            "chroma_ms": round(_t_chroma_ms, 1),
            "fts_ms": round(_t_fts_ms, 1),
            "entity_ms": round(_t_entity_ms, 1),
            "rerank_ms": round(_t_rerank_ms, 1),
            "total_ms": round(_t_total_ms, 1),
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

        # Get memory IDs mentioning any of these entities (cap results).
        # 15 (was 50) — entity results supplement semantic search, not dominate it.
        MAX_ENTITY_MEMORIES = 15
        memory_ids = await self.sqlite.get_memory_ids_for_entities(all_entity_ids)
        if len(memory_ids) > MAX_ENTITY_MEMORIES:
            memory_ids = memory_ids[:MAX_ENTITY_MEMORIES]
        return memory_ids, matched_names, len(connected_ids)

    async def _apply_reranking(self, query: str, results: list[SearchResult]) -> int:
        """Rerank the top candidates and blend scores.

        Takes the top reranker_top_n results (by boosted_score), sends them
        to the reranker model, and blends the reranker score into boosted_score:
            new_score = (1 - weight) * normalized_boosted + weight * reranker_score

        Scores are normalized to [0, 1] before blending so the reranker signal
        is on the same scale as the existing boosted_score.

        Returns the number of documents reranked.
        """
        if not results:
            return 0

        # Lazy init reranker
        if self._reranker is None:
            self._reranker = Reranker(
                ollama_url=self._ollama_url,
                model=self.reranker_model,
                instruction=self.reranker_instruction or None,
            )

        # Sort by current boosted_score to pick top candidates
        results.sort(key=lambda r: r.boosted_score, reverse=True)
        top_n = min(self.reranker_top_n, len(results))
        candidates = results[:top_n]

        # Build (memory_id, text) pairs — use summary for speed (shorter text)
        documents = [(r.memory_id, r.summary) for r in candidates]

        rerank_results = await self._reranker.rerank(query, documents)

        # Build memory_id -> reranker_score map
        reranker_scores: dict[int, float] = {}
        for rr in rerank_results:
            reranker_scores[rr.memory_id] = rr.score

        # Normalize boosted_scores to [0, 1] for blending
        max_score = max(r.boosted_score for r in candidates) if candidates else 1.0
        min_score = min(r.boosted_score for r in candidates) if candidates else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        w = self.reranker_weight
        for r in candidates:
            reranker_score = reranker_scores.get(r.memory_id, 0.5)
            normalized = (r.boosted_score - min_score) / score_range
            r.boosted_score = (1 - w) * normalized + w * reranker_score

        return len(candidates)

    async def search_self_thoughts(
        self,
        query: str,
        store,
        *,
        cosine_floor: float,
        rerank_floor: float,
        max_inject: int,
        prefilter_k: int,
        gravity_enabled: bool = False,
    ) -> list[tuple[str, float]]:
        """Two-stage relevance filter over BlipShell's own lingering thoughts.

        Stage 1 — cosine prefilter (cheap, in-process): narrow ≤max_keep thoughts
        to the prefilter_k nearest above a *loose* cosine_floor. Efficiency only;
        explicitly NOT the gate (embedding similarities sit in a high band).

        Stage 2 — LLM relevance judge (sharp): a local reasoning-model yes/no on
        each candidate is the actual decision. (This replaces a Qwen3 reranker
        that doesn't produce usable output via Ollama /api/generate; a real model
        judging relevance is the sharp gate the design wanted, on infra that
        works, and keeps self-thoughts local.) The judge returns 1.0/0.0, so
        rerank_floor keeps its threshold meaning.

        Fail-closed: if the judge errors, nothing surfaces. A self-authored
        thought reaching its own context is earned by a sharp filter or not at
        all — better silent than sloppy.

        Returns up to max_inject (text, cosine) pairs, most-similar first.
        """
        if store is None or max_inject <= 0:
            return []

        loop = asyncio.get_running_loop()
        try:
            qvec = await loop.run_in_executor(None, self.vectors.embed_text, query)
        except Exception as e:
            logger.warning("Self-thought query embed failed: %s", e)
            return []

        candidates = await store.relevant_candidates(qvec, floor=cosine_floor, k=prefilter_k)
        logger.info(
            "Self-thought prefilter: %d candidate(s) >= cosine %.2f %s",
            len(candidates), cosine_floor,
            [(t[:50], round(s, 3)) for t, s in candidates],
        )
        if not candidates:
            return []

        kept: list[tuple[str, float]] = []
        for text, sim in candidates:
            try:
                verdict = await self._judge_relevance(query, text)
            except Exception as e:
                logger.warning("Self-thought relevance judge failed: %s", e)
                continue  # fail-closed: a thought we can't judge doesn't surface
            logger.info(
                "Self-thought judge: %.1f (floor %.2f, %s) cosine %.3f for %r",
                verdict, rerank_floor, "PASS" if verdict >= rerank_floor else "drop",
                sim, text[:50],
            )
            if verdict >= rerank_floor:
                kept.append((text, sim))   # order survivors by cosine similarity

        # Among thoughts that PASSED the relevance gate, decide which one(s)
        # actually surface. Without gravity: most-relevant (cosine) wins. With
        # gravity: weight x relevance wins — so the thought that matters most to
        # BlipShell takes the (capped) slot, not merely the most lexically near.
        # Relevance was already required by the gate; gravity only re-orders it.
        if gravity_enabled:
            weights = await store.effective_weights([t for t, _ in kept])
            kept.sort(key=lambda ts: weights.get(ts[0], 1.0) * ts[1], reverse=True)
        else:
            kept.sort(key=lambda x: x[1], reverse=True)
        return kept[:max_inject]

    async def _judge_relevance(self, query: str, thought: str) -> float:
        """LLM yes/no judge: is `thought` relevant to `query` right now?

        Returns 1.0 (yes) or 0.0 (no, or no clear verdict — fail-closed). Uses
        the local reasoning model with thinking off: a relevance yes/no doesn't
        need deep reasoning, and this runs on the per-turn path where latency
        matters.
        """
        system = (
            "You decide whether one of the assistant's own past private thoughts "
            "is relevant to what the user is now saying. Reply with exactly one "
            "word: yes or no."
        )
        user = (
            f'Past private thought: "{thought}"\n\n'
            f'User just said: "{query}"\n\n'
            "Is the past thought genuinely relevant to what the user is now "
            "discussing — relevant enough that recalling it would help the "
            "conversation? Answer yes or no."
        )
        resp = await self.router.generate(
            TaskType.REASONING, user, system=system, think=False,
        )
        # Last explicit yes/no token wins; no verdict -> fail-closed (0.0).
        for tok in reversed(re.findall(r"[a-z]+", (resp or "").lower())):
            if tok == "yes":
                return 1.0
            if tok == "no":
                return 0.0
        return 0.0

    async def search_core_memories(self, query: str, n_results: int = 10) -> list[dict]:
        """Search core memories by semantic similarity."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.vectors.search_core_memories, query, n_results),
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
            None, functools.partial(self.vectors.search_lessons, query, n_results),
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
