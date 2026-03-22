"""Thought Engine — generates, stores, retrieves, and refines AI thoughts.

Thoughts are atomic reflections the AI generates about conversations,
memories, and its own state. They have categories (belief, opinion,
observation, question, preference, pattern) and confidence levels.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from blipshell.alive.prompts import extract_session_thoughts
from blipshell.llm.router import TaskType
from blipshell.models.alive import Thought, ThoughtCategory, ThoughtSource

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import AliveConfig

logger = logging.getLogger(__name__)

# Dedup threshold — if a new thought is this similar to an existing one, skip it
DEDUP_SIMILARITY = 0.85


class ThoughtEngine:
    """Manages the AI's thought lifecycle: generation, storage, retrieval, pruning."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        router: LLMRouter,
        config: AliveConfig,
    ):
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router
        self.config = config

    async def generate_thoughts(
        self,
        session_summary: str,
        conversation_text: str,
        session_id: int,
        current_identity: str = "",
    ) -> list[Thought]:
        """Generate thoughts from a completed session.

        Calls the reasoning model with the session context, parses the output
        into Thought objects, deduplicates against existing thoughts, and stores.
        """
        system, user = extract_session_thoughts(
            session_summary=session_summary,
            conversation_text=conversation_text,
            current_identity=current_identity,
        )

        try:
            raw = await self.router.generate(
                TaskType.REASONING, user, system=system,
            )
        except Exception as e:
            logger.error("Thought generation failed: %s", e)
            return []

        if not raw or raw.strip().upper() == "SKIP":
            return []

        thoughts = self._parse_thoughts(raw)
        stored = []

        for thought in thoughts:
            # Dedup against existing thoughts
            if await self._is_duplicate(thought.content):
                logger.debug("Skipping duplicate thought: %.60s...", thought.content)
                continue

            # Confidence floor
            if thought.confidence < self.config.thought.min_confidence:
                logger.debug("Skipping low-confidence thought (%.2f): %.60s...",
                             thought.confidence, thought.content)
                continue

            # Store
            thought.source_type = ThoughtSource.REFLECTION
            thought.source_session_id = session_id
            thought_id = await self.sqlite.add_thought(
                content=thought.content,
                category=thought.category.value,
                confidence=thought.confidence,
                source_type=thought.source_type.value,
                source_session_id=session_id,
            )
            thought.id = thought_id

            # Embed in ChromaDB
            try:
                self.chroma.add_thought(
                    thought_id, thought.content,
                    metadata={"category": thought.category.value},
                )
            except Exception as e:
                logger.warning("Failed to embed thought %d: %s", thought_id, e)

            stored.append(thought)

        logger.info("Generated %d thoughts from session %d (%d after dedup)",
                     len(thoughts), session_id, len(stored))
        return stored

    async def refine_thought(
        self,
        thought_id: int,
        new_content: str,
        new_confidence: float,
        source_type: str = "monologue",
    ) -> Thought | None:
        """Evolve an existing thought: deactivate parent, create child."""
        # Deactivate parent
        await self.sqlite.deactivate_thought(thought_id)
        try:
            self.chroma.delete_thought(thought_id)
        except Exception:
            pass

        # Get parent for metadata
        parent_thoughts = await self.sqlite.get_active_thoughts(limit=1)
        # Actually let's just create the child with the parent_thought_id
        new_id = await self.sqlite.add_thought(
            content=new_content,
            confidence=new_confidence,
            source_type=source_type,
            parent_thought_id=thought_id,
        )

        try:
            self.chroma.add_thought(new_id, new_content)
        except Exception as e:
            logger.warning("Failed to embed refined thought %d: %s", new_id, e)

        return Thought(
            id=new_id,
            content=new_content,
            confidence=new_confidence,
            source_type=ThoughtSource(source_type),
            parent_thought_id=thought_id,
        )

    async def search_thoughts(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search over thoughts via ChromaDB."""
        return self.chroma.search_thoughts(query, n_results=limit)

    async def get_active_thoughts(
        self,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get active thoughts from SQLite."""
        return await self.sqlite.get_active_thoughts(category=category, limit=limit)

    async def get_recent_thoughts(self, limit: int = 20) -> list[dict]:
        """Get most recent active thoughts."""
        return await self.sqlite.get_recent_thoughts(limit=limit)

    async def prune(self) -> int:
        """Prune low-confidence and excess thoughts."""
        return await self.sqlite.prune_thoughts(
            min_confidence=self.config.thought.min_confidence,
            max_active=self.config.thought.max_active_thoughts,
        )

    async def _is_duplicate(self, content: str) -> bool:
        """Check if a thought is too similar to an existing active thought."""
        try:
            results = self.chroma.search_thoughts(content, n_results=3)
        except Exception:
            return False

        for r in results:
            if r.get("similarity", 0) >= DEDUP_SIMILARITY:
                return True
        return False

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Strip markdown bold/italic markers from field names."""
        return re.sub(r'\*+', '', text)

    @staticmethod
    def _parse_thoughts(raw: str) -> list[Thought]:
        """Parse LLM output into Thought objects.

        Expected format:
        CATEGORY: belief
        CONFIDENCE: 0.8
        THOUGHT: I believe that...

        Also handles markdown-formatted variants like **CATEGORY:** belief
        """
        # Strip markdown from field names before parsing
        cleaned = ThoughtEngine._strip_markdown(raw)

        thoughts = []
        # Split on CATEGORY: (with optional leading - or whitespace)
        blocks = re.split(r'\n(?=-?\s*CATEGORY:)', cleaned.strip())

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            cat_match = re.search(r'CATEGORY:\s*(\w+)', block, re.IGNORECASE)
            conf_match = re.search(r'CONFIDEN(?:CE|T):\s*([\d.]+)', block, re.IGNORECASE)
            thought_match = re.search(r'THOUGHT:\s*(.+)', block, re.IGNORECASE | re.DOTALL)

            if not thought_match:
                continue

            content = thought_match.group(1).strip()
            # Clean up — remove any trailing field markers from next block
            content = re.split(
                r'\n-?\s*(?:CATEGORY|CONFIDEN(?:CE|T)|THOUGHT|REFINE|INITIATIVE|NEXT_FOCUS):',
                content, flags=re.IGNORECASE,
            )[0].strip()

            if len(content) < 10:
                continue

            # Parse category
            category = ThoughtCategory.OBSERVATION
            if cat_match:
                try:
                    category = ThoughtCategory(cat_match.group(1).lower())
                except ValueError:
                    pass

            # Parse confidence
            confidence = 0.5
            if conf_match:
                try:
                    confidence = max(0.0, min(1.0, float(conf_match.group(1))))
                except ValueError:
                    pass

            thoughts.append(Thought(
                content=content,
                category=category,
                confidence=confidence,
            ))

        return thoughts
