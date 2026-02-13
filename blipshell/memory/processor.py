"""Background memory processing pipeline.

Port of MemoryDB.CreateMemoryAsync pipeline:
noise check → LLM summarize → SQLite insert → ChromaDB embed → tag → LLM rank → LLM importance
"""

import logging
from datetime import datetime, timedelta

from blipshell.llm.prompts import (
    ask_importance,
    extract_lesson,
    rank_memory,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.noise import should_skip_memory
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_message
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
    6. LLM rank (quality 1-5)
    7. LLM importance (0.0-1.0 with recency/tag bonuses)
    """

    def __init__(self, sqlite: SQLiteStore, chroma: ChromaStore, router: LLMRouter):
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router

    async def process_message(
        self,
        text: str,
        role: str,
        session_id: int,
        metadata: str = "{}",
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
            summary = await self.router.generate(
                TaskType.SUMMARIZATION,
                summarize_memory(text),
            )
        except Exception as e:
            logger.error("Summarization failed, using raw text: %s", e)
            summary = text

        # Step 3: SQLite insert
        memory = Memory(
            session_id=session_id,
            role=role,
            content=text,
            summary=summary,
            timestamp=datetime.utcnow(),
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
            tags = tag_message(text)
            await self.sqlite.tag_memory(memory_id, tags)
        except Exception as e:
            logger.error("Tagging failed: %s", e)
            tags = []

        # Step 6: Rank (1-5)
        try:
            rank_text = await self.router.generate(
                TaskType.RANKING,
                rank_memory(text),
            )
            rank = self._parse_rank(rank_text)
            await self.sqlite.update_memory(memory_id, rank=rank)
        except Exception as e:
            logger.error("Ranking failed: %s", e)

        # Step 7: Importance (0.0-1.0)
        try:
            importance = await self._calculate_importance(memory_id, text, tags)
            await self.sqlite.update_memory(memory_id, importance=importance)
        except Exception as e:
            logger.error("Importance calculation failed: %s", e)

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
            tags = tag_message(text)
            await self.sqlite.tag_core_memory(mem_id, tags)
        except Exception as e:
            logger.error("Core memory tagging failed: %s", e)

        return mem_id

    async def process_lesson(self, conversation_text: str, session_id: int) -> int:
        """Extract and store a lesson from a conversation."""
        # Generate lesson text via LLM
        try:
            lesson_text = await self.router.generate(
                TaskType.SUMMARIZATION,
                extract_lesson(conversation_text),
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
            tags = tag_message(lesson_text)
            await self.sqlite.tag_lesson(lesson_id, tags)
        except Exception as e:
            logger.error("Lesson tagging failed: %s", e)

        return lesson_id

    async def _calculate_importance(
        self, memory_id: int, text: str, tags: list[str]
    ) -> float:
        """Calculate importance score with recency and tag bonuses.

        Port of MemoryDB.CalculateImportance().
        """
        # Base importance from LLM
        try:
            importance_text = await self.router.generate(
                TaskType.RANKING,
                ask_importance(text),
            )
            importance = self._parse_float(importance_text, default=0.3)
        except Exception:
            importance = 0.3

        # Recency bonus: +0.1 if within 7 days
        importance += 0.1  # always recent at creation time

        # Tag bonus: +0.2 if many tags (>6)
        tag_count = await self.sqlite.get_tag_count_for_memory(memory_id)
        if tag_count > 6:
            importance += 0.2

        return min(importance, 1.0)

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
        import re
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            try:
                val = float(match.group(1))
                return min(max(val, 0.0), 1.0)
            except ValueError:
                pass
        return default
