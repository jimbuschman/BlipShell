"""Entity extraction — extract relationship triples from memories.

Batch-processes unextracted memories on startup (like consolidation).
For each memory, calls the LLM to extract (subject, predicate, object) triples,
then stores entities, relationships, and mentions in SQLite.
"""

import asyncio
import logging

from blipshell.llm.prompts import extract_entities
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts entity relationship triples from memory summaries."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        router: LLMRouter,
        batch_size: int = 50,
        task_type: str = TaskType.REASONING,
    ):
        self.sqlite = sqlite
        self.router = router
        self.batch_size = batch_size
        self.task_type = task_type

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
            response = await self.router.generate(
                self.task_type, user, system=system, think=False,
            )

            triples = self._parse_triples(response)
            for subj, pred, obj, s_type, o_type in triples:
                try:
                    subj_id = await self.sqlite.get_or_create_entity(subj, s_type)
                    obj_id = await self.sqlite.get_or_create_entity(obj, o_type)
                    await self.sqlite.create_entity_relationship(
                        subj_id, pred, obj_id, memory_id,
                    )
                    await self.sqlite.create_entity_mention(subj_id, memory_id)
                    await self.sqlite.create_entity_mention(obj_id, memory_id)
                    result["triples"] += 1
                except Exception as e:
                    logger.warning(
                        "Failed to store triple from memory %d: %s", memory_id, e,
                    )

            await self.sqlite.mark_entities_extracted([memory_id])
            result["extracted"] = 1

        except Exception as e:
            result["errors"] = 1
            logger.warning(
                "Entity extraction failed for memory %d: %s", memory_id, e,
            )
            try:
                await self.sqlite.mark_entities_extracted([memory_id])
            except Exception:
                pass

        return result

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
            if subj.lower() in ("something", "it", "this", "that"):
                continue
            if obj.lower() in ("something", "it", "this", "that"):
                continue

            triples.append((subj, pred, obj, s_type, o_type))

        return triples
