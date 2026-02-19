"""Background memory processing pipeline.

Port of MemoryDB.CreateMemoryAsync pipeline:
noise check -> LLM summarize -> SQLite insert -> ChromaDB embed -> tag -> LLM rank+importance
"""

import logging
import re
from datetime import datetime, timedelta, timezone

from blipshell.llm.prompts import (
    detect_contradiction,
    extract_lesson,
    rank_and_importance,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.noise import should_skip_memory
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_message
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import CoreMemory, Lesson, Memory, MemoryType

logger = logging.getLogger(__name__)


class MemoryProcessor:
    """Background pipeline for processing memories.

    Pipeline steps:
    1. Noise check (skip low-value messages)
    2. LLM summarize (generate concise summary)
    3. SQLite insert (persist structured data)
    4. ChromaDB embed (store vector for semantic search)
    5. Tag (extract topic/behavior tags)
    6. LLM rank+importance (combined call: rank 1-5, importance 0.0-1.0)
    """

    def __init__(self, sqlite: SQLiteStore, chroma: ChromaStore, router: LLMRouter,
                 config: MemoryConfig | None = None, max_tags: int = 7):
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router
        self._recency_bonus = config.importance_recency_bonus if config else 0.1
        self._tag_bonus = config.importance_tag_bonus if config else 0.05
        self._contradiction_threshold = config.contradiction_similarity_threshold if config else 0.7
        self._max_tags = max_tags

    async def process_message(
        self,
        text: str,
        role: str,
        session_id: int,
        metadata: str = "{}",
        timestamp: datetime | None = None,
    ) -> int | None:
        """Full pipeline for processing a conversation message into memory.

        Returns the memory ID, or None if filtered as noise.
        """
        # Step 1: Noise check
        if should_skip_memory(text):
            logger.debug("Skipping noise: %s", text[:50])
            return None

        # Step 2: Summarize
        try:
            sum_system, sum_prompt = summarize_memory(text)
            summary = await self.router.generate(
                TaskType.SUMMARIZATION,
                sum_prompt,
                system=sum_system,
            )
            # LLM signals this is self-referential / meta content
            if summary.strip().upper() == "SKIP":
                logger.debug("Memory skipped (meta/self-referential): %s", text[:50])
                return None
        except Exception as e:
            logger.error("Summarization failed, using raw text: %s", e)
            summary = text

        # Step 3: SQLite insert
        memory = Memory(
            session_id=session_id,
            role=role,
            content=text,
            summary=summary,
            timestamp=timestamp or datetime.now(timezone.utc),
            memory_type=MemoryType.CONVERSATION,
        )
        memory_id = await self.sqlite.create_memory(memory)

        # Step 4: ChromaDB embed (use summary for better semantic matching)
        try:
            self.chroma.add_memory(memory_id, summary, {
                "session_id": str(session_id),
                "role": role,
            })
        except Exception as e:
            logger.error("ChromaDB embed failed: %s", e)

        # Step 5: Tag
        try:
            tags = tag_message(text, max_tags=self._max_tags)
            await self.sqlite.tag_memory(memory_id, tags)
        except Exception as e:
            logger.error("Tagging failed: %s", e)
            tags = []

        # Step 6+7: Combined rank (1-5) + importance (0.0-1.0) in one LLM call
        try:
            ri_system, ri_prompt = rank_and_importance(text)
            ri_text = await self.router.generate(
                TaskType.RANKING_IMPORTANCE,
                ri_prompt,
                system=ri_system,
            )
            rank, importance = self._parse_rank_and_importance(ri_text)

            # Apply bonuses
            importance += self._recency_bonus
            if len(tags) > 6:
                importance += self._tag_bonus
            importance = min(importance, 1.0)

            await self.sqlite.update_memory(memory_id, rank=rank, importance=importance)
        except Exception as e:
            logger.error("Rank+importance failed: %s", e)

        return memory_id

    async def process_core_memory(
        self, text: str, session_id: int | None = None
    ) -> int:
        """Process and store a core memory."""
        core_memory = CoreMemory(
            content=text,
            source_session_id=session_id,
        )
        mem_id = await self.sqlite.create_core_memory(core_memory)

        # Embed
        try:
            self.chroma.add_core_memory(mem_id, text)
        except Exception as e:
            logger.error("Core memory embed failed: %s", e)

        # Tag
        try:
            tags = tag_message(text, max_tags=self._max_tags)
            await self.sqlite.tag_core_memory(mem_id, tags)
        except Exception as e:
            logger.error("Core memory tagging failed: %s", e)

        # Contradiction check — deactivate stale/contradicted core memories
        try:
            deactivated = await self._check_core_memory_contradictions(
                mem_id, text,
                similarity_threshold=self._contradiction_threshold,
            )
            if deactivated:
                logger.info("Deactivated %d contradicted core memories", deactivated)
        except Exception as e:
            logger.error("Contradiction check failed: %s", e)

        return mem_id

    async def process_lesson(self, conversation_text: str, session_id: int) -> int:
        """Extract and store a lesson from a conversation."""
        # Generate lesson text via reasoning model (needs understanding, not just summarization)
        try:
            lesson_system, lesson_prompt = extract_lesson(conversation_text)
            lesson_text = await self.router.generate(
                TaskType.REASONING,
                lesson_prompt,
                system=lesson_system,
            )
        except Exception as e:
            logger.error("Lesson extraction failed: %s", e)
            lesson_text = conversation_text

        lesson = Lesson(
            content=lesson_text,
            source_session_id=session_id,
        )
        lesson_id = await self.sqlite.create_lesson(lesson)

        # Embed
        try:
            self.chroma.add_lesson(lesson_id, lesson_text)
        except Exception as e:
            logger.error("Lesson embed failed: %s", e)

        # Tag
        try:
            tags = tag_message(lesson_text, max_tags=self._max_tags)
            await self.sqlite.tag_lesson(lesson_id, tags)
        except Exception as e:
            logger.error("Lesson tagging failed: %s", e)

        return lesson_id

    async def _check_core_memory_contradictions(
        self, core_memory_id: int, text: str,
        similarity_threshold: float = 0.7,
    ) -> int:
        """Check new core memory against existing ones for contradictions.

        Searches ChromaDB for similar core memories and asks the LLM whether
        each pair contradicts. Deactivates older contradicted memories.
        Returns count of deactivated memories.
        """
        results = self.chroma.search_core_memories(text, n_results=3)

        deactivated = 0
        for r in results:
            if r["id"] == core_memory_id:
                continue
            if r["similarity"] < similarity_threshold:
                continue

            # Ask LLM if they contradict
            system, prompt = detect_contradiction(text, r["document"])
            answer = await self.router.generate(
                TaskType.REASONING, prompt, system=system, think=False,
            )

            if answer.strip().upper().startswith("YES"):
                await self.sqlite.deactivate_core_memory(r["id"])
                try:
                    self.chroma.delete_core_memory(r["id"])
                except Exception:
                    pass
                deactivated += 1
                logger.info(
                    "Deactivated contradicted core memory %d (superseded by %d)",
                    r["id"], core_memory_id,
                )

        return deactivated

    @staticmethod
    def _parse_rank_and_importance(text: str) -> tuple[int, float]:
        """Parse combined 'rank importance' from LLM response (e.g. '4 0.7')."""
        numbers = re.findall(r"(\d+\.?\d*)", text.strip())
        rank = 3
        importance = 0.3
        if len(numbers) >= 1:
            r = int(float(numbers[0]))
            if 1 <= r <= 5:
                rank = r
        if len(numbers) >= 2:
            imp = float(numbers[1])
            importance = min(max(imp, 0.0), 1.0)
        return rank, importance

    @staticmethod
    def _parse_rank(text: str) -> int:
        """Parse a rank (1-5) from LLM response."""
        text = text.strip()
        for char in text:
            if char.isdigit():
                val = int(char)
                if 1 <= val <= 5:
                    return val
        return 3  # default

    @staticmethod
    def _parse_float(text: str, default: float = 0.0) -> float:
        """Parse a float from LLM response."""
        text = text.strip()
        # Try to find a decimal number in the response
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            try:
                val = float(match.group(1))
                return min(max(val, 0.0), 1.0)
            except ValueError:
                pass
        return default
