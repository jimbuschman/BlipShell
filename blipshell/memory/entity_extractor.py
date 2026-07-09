"""Entity extraction — extract relationship triples from memories.

Batch-processes unextracted memories on startup (like consolidation).
For each memory, calls the LLM to extract (subject, predicate, object) triples,
then stores entities, relationships, and mentions in SQLite.

Extended with:
- Bi-temporal edge tracking (Feature 4): contradicting predicates expire old edges
- Entity resolution (Feature 5): 3-stage dedup before creating new entities
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from blipshell.llm.prompts import extract_entities, resolve_entity_duplicate
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore

if TYPE_CHECKING:
    from blipshell.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Predicate contradiction map — when a new predicate is created for the same
# subject-object pair, expire any active relationships with contradicting predicates.
# Key: new predicate, Value: list of predicates it contradicts.
CONTRADICTING_PREDICATES: dict[str, list[str]] = {
    "works_at": ["left", "quit", "fired_from", "resigned_from"],
    "left": ["works_at", "employed_at", "hired_by"],
    "quit": ["works_at", "employed_at", "hired_by"],
    "fired_from": ["works_at", "employed_at", "hired_by"],
    "resigned_from": ["works_at", "employed_at", "hired_by"],
    "employed_at": ["left", "quit", "fired_from", "resigned_from"],
    "hired_by": ["left", "quit", "fired_from", "resigned_from"],
    "uses": ["stopped_using", "dropped", "replaced"],
    "stopped_using": ["uses", "adopted", "switched_to"],
    "dropped": ["uses", "adopted", "switched_to"],
    "replaced": ["uses"],
    "adopted": ["stopped_using", "dropped"],
    "switched_to": ["uses", "stopped_using"],
    "prefers": ["dislikes", "avoids", "stopped_using"],
    "dislikes": ["prefers", "likes", "loves"],
    "likes": ["dislikes", "avoids"],
    "loves": ["dislikes", "hates", "avoids"],
    "hates": ["likes", "loves", "prefers"],
    "avoids": ["prefers", "likes", "uses"],
    "lives_in": ["moved_from", "left"],
    "moved_from": ["lives_in"],
    "moved_to": ["lives_in"],
}


class EntityExtractor:
    """Extracts entity relationship triples from memory summaries.

    Extended with:
    - 3-stage entity resolution (Feature 5): exact → embedding → LLM
    - Bi-temporal edges (Feature 4): contradicting predicates expire old edges
    """

    def __init__(
        self,
        sqlite: SQLiteStore,
        router: LLMRouter,
        vectors: VectorStore | None = None,
        batch_size: int = 50,
        task_type: str = TaskType.REASONING,
        model_override: str | None = None,
        entity_resolution_enabled: bool = False,
        entity_auto_merge_threshold: float = 0.85,
        entity_llm_threshold: float = 0.70,
        entity_max_candidates: int = 5,
    ):
        self.sqlite = sqlite
        self.router = router
        self.vectors = vectors
        self.batch_size = batch_size
        self.task_type = task_type
        self.model_override = model_override
        # Entity resolution config
        self._resolution_enabled = entity_resolution_enabled
        self._auto_merge_threshold = entity_auto_merge_threshold
        self._llm_threshold = entity_llm_threshold
        self._max_candidates = entity_max_candidates
        # LRU cache for resolved entities within a batch (cleared per batch)
        self._resolution_cache: dict[str, int] = {}

    async def extract_batch(self, concurrency: int = 1) -> dict:
        """Process a batch of unextracted memories. Returns stats dict.

        Args:
            concurrency: Number of LLM calls to run in parallel.
                         Use 1 for local models, higher for cloud.
        """
        stats = {"extracted": 0, "triples": 0, "errors": 0}

        memory_ids = await self.sqlite.get_unextracted_memory_ids(
            limit=self.batch_size,
        )
        if not memory_ids:
            return stats

        # Clear resolution cache for this batch
        self._resolution_cache = {}

        if concurrency <= 1:
            for memory_id in memory_ids:
                result = await self._extract_one(memory_id)
                stats["extracted"] += result["extracted"]
                stats["triples"] += result["triples"]
                stats["errors"] += result["errors"]
        else:
            semaphore = asyncio.Semaphore(concurrency)

            async def _limited(mid):
                async with semaphore:
                    return await self._extract_one(mid)

            results = await asyncio.gather(
                *[_limited(mid) for mid in memory_ids],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    stats["errors"] += 1
                else:
                    stats["extracted"] += r["extracted"]
                    stats["triples"] += r["triples"]
                    stats["errors"] += r["errors"]

        return stats

    async def _extract_one(self, memory_id: int) -> dict:
        """Extract entities from a single memory. Returns per-memory stats."""
        result = {"extracted": 0, "triples": 0, "errors": 0}
        try:
            memory = await self.sqlite.get_memory(memory_id)
            if not memory or not memory.summary:
                await self.sqlite.mark_entities_extracted([memory_id])
                return result

            system, user = extract_entities(memory.summary)
            if self.model_override:
                # Bypass router model selection, use specified model directly
                # But still respect OllamaGate to avoid concurrent Ollama access
                endpoint = await self.router._endpoint_manager.get_endpoint_for_role(self.task_type)
                if endpoint is None or endpoint.client is None:
                    raise RuntimeError(f"No endpoint available for {self.task_type}")
                if endpoint.provider == "ollama":
                    from blipshell.llm.ollama_gate import get_gate
                    gate = get_gate()
                    async with gate.async_gate(gate.infer_priority()):
                        response = await endpoint.client.generate(
                            prompt=user, model=self.model_override, system=system,
                            think=False,
                        )
                else:
                    response = await endpoint.client.generate(
                        prompt=user, model=self.model_override, system=system,
                        think=False,
                    )
            else:
                response = await self.router.generate(
                    self.task_type, user, system=system, think=False,
                )

            triples = self._parse_triples(response)

            # Phase 1: Resolve entities (may call ChromaDB/LLM — keep outside transaction)
            resolved_triples = []
            for subj, pred, obj, s_type, o_type in triples:
                try:
                    subj_id = await self._resolve_entity(subj, s_type)
                    obj_id = await self._resolve_entity(obj, o_type)
                    resolved_triples.append((subj_id, pred, obj_id, subj, obj))
                except Exception as e:
                    logger.warning(
                        "Failed to resolve entities for triple from memory %d: %s", memory_id, e,
                    )

            # Phase 2: Write all relationships/mentions in one transaction.
            # One lock acquisition instead of ~31 per memory.
            try:
                for subj_id, pred, obj_id, subj, obj in resolved_triples:
                    try:
                        rel_id = await self.sqlite.create_entity_relationship_temporal(
                            subj_id, pred, obj_id, memory_id, skip_commit=True,
                        )
                        if rel_id:
                            try:
                                expired = await self.sqlite.expire_contradicting_relationships(
                                    subj_id, obj_id,
                                    CONTRADICTING_PREDICATES,
                                    pred, rel_id, skip_commit=True,
                                )
                                if expired:
                                    logger.info(
                                        "Expired %d contradicting relationships for %s %s %s",
                                        expired, subj, pred, obj,
                                    )
                            except Exception as e:
                                logger.warning("Failed to expire contradictions: %s", e)
                        await self.sqlite.create_entity_mention(subj_id, memory_id, skip_commit=True)
                        await self.sqlite.create_entity_mention(obj_id, memory_id, skip_commit=True)
                        result["triples"] += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to store triple from memory %d: %s", memory_id, e,
                        )

                await self.sqlite.mark_entities_extracted([memory_id], skip_commit=True)
                await self.sqlite.commit()
                result["extracted"] = 1
            except Exception as e:
                try:
                    await self.sqlite.rollback()
                except Exception:
                    pass
                raise

        except Exception as e:
            result["errors"] = 1
            logger.warning(
                "Entity extraction failed for memory %d: %s", memory_id, e,
            )
            try:
                await self.sqlite.mark_entities_extracted([memory_id])
            except Exception as mark_err:
                logger.warning("Failed to mark memory %d as extracted: %s", memory_id, mark_err)

        return result

    async def _resolve_entity(self, name: str, entity_type: str = "concept") -> int:
        """4-stage entity resolution: alias → exact match → embedding → LLM.

        Stage 0: Alias routing (names merged away route to their canonical)
        Stage 1: Exact match (existing UNIQUE constraint in SQLite)
        Stage 2: Embedding similarity (auto-merge above threshold)
        Stage 3: LLM arbitration (ambiguous range)

        Falls back to get_or_create_entity() if resolution is disabled.
        """
        name = name.strip().lower()

        # Check batch-level cache first
        if name in self._resolution_cache:
            return self._resolution_cache[name]

        # Stage 0: Alias routing (always runs, before exact match — the merged
        # husk still has an entities row, so exact match would land mentions on
        # the archived husk instead of the canonical). Aliases are recorded
        # facts from past merges, so this runs even with resolution disabled.
        alias_id = await self.sqlite.resolve_alias(name)
        if alias_id is not None:
            self._resolution_cache[name] = alias_id
            return alias_id

        # Stage 1: Exact match (always runs)
        existing_id = await self.sqlite.get_entity_id_by_name(name)
        if existing_id is not None:
            self._resolution_cache[name] = existing_id
            return existing_id

        # If resolution is disabled or no ChromaDB, create new entity
        if not self._resolution_enabled or not self.vectors:
            entity_id = await self.sqlite.get_or_create_entity(name, entity_type)
            # Upsert into entity embeddings if ChromaDB available
            if self.vectors:
                try:
                    self.vectors.upsert_entity(entity_id, name, entity_type)
                except Exception as e:
                    logger.warning("Failed to upsert entity embedding: %s", e)
            self._resolution_cache[name] = entity_id
            return entity_id

        # Stage 2: Embedding similarity search
        try:
            candidates = self.vectors.search_similar_entities(
                name, n_results=self._max_candidates,
            )
        except Exception as e:
            logger.warning("Entity similarity search failed: %s", e)
            candidates = []

        for candidate in candidates:
            # Validate candidate still exists in SQLite (ChromaDB can have stale IDs)
            if not await self.sqlite.entity_id_exists(candidate["id"]):
                logger.warning(
                    "Stale entity in vector store: id=%d name='%s' — skipping",
                    candidate["id"], candidate["name"],
                )
                try:
                    self.vectors.delete_entity(candidate["id"])
                except Exception:
                    pass
                continue

            if candidate["similarity"] >= self._auto_merge_threshold:
                # Auto-merge: high confidence match
                canonical_id = candidate["id"]
                await self.sqlite.record_entity_alias(
                    name, canonical_id, merge_method="embedding_auto",
                )
                self._resolution_cache[name] = canonical_id
                logger.info(
                    "Entity auto-merged: '%s' → '%s' (sim=%.3f)",
                    name, candidate["name"], candidate["similarity"],
                )
                return canonical_id

            if candidate["similarity"] >= self._llm_threshold:
                # Stage 3: LLM arbitration for ambiguous range
                is_same = await self._llm_resolve(name, candidate["name"])
                if is_same:
                    canonical_id = candidate["id"]
                    await self.sqlite.record_entity_alias(
                        name, canonical_id, merge_method="llm_resolved",
                    )
                    self._resolution_cache[name] = canonical_id
                    logger.info(
                        "Entity LLM-merged: '%s' → '%s' (sim=%.3f)",
                        name, candidate["name"], candidate["similarity"],
                    )
                    return canonical_id

        # No match — create new entity
        entity_id = await self.sqlite.get_or_create_entity(name, entity_type)
        try:
            self.vectors.upsert_entity(entity_id, name, entity_type)
        except Exception as e:
            logger.warning("Failed to upsert entity embedding: %s", e)
        self._resolution_cache[name] = entity_id
        return entity_id

    async def _llm_resolve(self, new_name: str, existing_name: str) -> bool:
        """Ask LLM if two entity names refer to the same entity. Returns True/False."""
        try:
            system, prompt = resolve_entity_duplicate(new_name, existing_name)
            response = await self.router.generate(
                self.task_type, prompt, system=system, think=False,
            )
            return response.strip().upper().startswith("YES")
        except Exception as e:
            logger.warning("LLM entity resolution failed: %s", e)
            return False

    def _parse_triples(self, response: str) -> list[tuple[str, str, str, str, str]]:
        """Parse 'subject | predicate | object | subject_type | object_type' lines.

        Returns list of (subject, predicate, object, subject_type, object_type).
        Skips malformed lines and NONE responses.
        """
        response = response.strip()
        if not response or response.upper() == "NONE":
            return []

        triples = []
        for line in response.splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue

            subj = parts[0]
            pred = parts[1]
            obj = parts[2]
            s_type = parts[3] if len(parts) > 3 else "concept"
            o_type = parts[4] if len(parts) > 4 else "concept"

            # Skip empty or vague entries
            if not subj or not pred or not obj:
                continue

            # Validate entity names
            skip = False
            for name in (subj, obj):
                name_lower = name.lower().strip()
                # Reject pronouns and vague words
                if name_lower in (
                    "something", "it", "this", "that", "these", "those",
                    "she", "her", "he", "him", "his", "they", "them", "their",
                    "someone", "anyone", "anything", "nothing",
                    "the user", "the assistant", "assistant", "none",
                ):
                    skip = True
                    break
                # Reject single-char names
                if len(name_lower) <= 1:
                    skip = True
                    break
                # Reject names > 60 chars (sentence-length descriptions)
                if len(name) > 60:
                    skip = True
                    break
            if skip:
                continue

            # Validate entity types
            valid_types = {"person", "project", "technology", "concept",
                           "preference", "place", "organization"}
            if s_type.lower().strip() not in valid_types:
                s_type = "concept"
            if o_type.lower().strip() not in valid_types:
                o_type = "concept"

            triples.append((subj, pred, obj, s_type, o_type))

        return triples
