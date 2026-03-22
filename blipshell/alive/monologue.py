"""Inner Monologue — between-session thinking loop.

Periodically reviews memories, generates new thoughts, refines existing ones,
and adds items to the initiative queue. Runs when no session is active.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from blipshell.alive.prompts import inner_monologue_cycle
from blipshell.alive.thought_engine import ThoughtEngine
from blipshell.llm.router import TaskType
from blipshell.models.alive import MonologueCycleResult

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import AliveConfig

logger = logging.getLogger(__name__)


class InnerMonologue:
    """One-shot monologue cycle: select memories, think, store results."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        chroma: ChromaStore,
        router: LLMRouter,
        config: AliveConfig,
        thought_engine: ThoughtEngine,
    ):
        self.sqlite = sqlite
        self.chroma = chroma
        self.router = router
        self.config = config
        self.thought_engine = thought_engine

    async def run_cycle(self) -> MonologueCycleResult:
        """Run one thinking cycle. Returns stats."""
        start = time.monotonic()
        cycle_num = await self.sqlite.get_next_monologue_cycle()

        # 1. Select memories to review
        memories = await self._select_memories()
        memory_texts = [
            m.get("summary") or m.get("content", "")
            for m in memories
        ]

        # 2. Get recent thoughts for continuity
        recent = await self.thought_engine.get_recent_thoughts(limit=5)
        thought_texts = [t.get("content", "") for t in recent]

        # 3. Get current identity
        identity_row = await self.sqlite.get_current_identity()
        identity = identity_row["content"] if identity_row else ""

        # 4. Build prompt and generate
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        system, user = inner_monologue_cycle(
            current_identity=identity,
            memories=memory_texts,
            recent_thoughts=thought_texts,
            current_datetime=now_str,
        )

        try:
            raw = await self.router.generate(TaskType.REASONING, user, system=system)
        except Exception as e:
            logger.error("Monologue generation failed: %s", e)
            elapsed = time.monotonic() - start
            result = MonologueCycleResult(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                elapsed_s=elapsed,
            )
            await self.sqlite.add_monologue_log(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                elapsed_s=elapsed,
                raw_output=f"ERROR: {e}",
            )
            return result

        if not raw or raw.strip().upper() == "SKIP":
            elapsed = time.monotonic() - start
            result = MonologueCycleResult(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                elapsed_s=elapsed,
            )
            await self.sqlite.add_monologue_log(
                cycle_number=cycle_num,
                memories_reviewed=len(memories),
                elapsed_s=elapsed,
                raw_output="SKIP",
            )
            return result

        # 5. Parse and store results
        thoughts_gen, thoughts_ref, initiative_added = await self._process_output(
            raw, cycle_num,
        )

        elapsed = time.monotonic() - start
        result = MonologueCycleResult(
            cycle_number=cycle_num,
            memories_reviewed=len(memories),
            thoughts_generated=thoughts_gen,
            thoughts_refined=thoughts_ref,
            initiative_items_added=initiative_added,
            elapsed_s=elapsed,
        )

        await self.sqlite.add_monologue_log(
            cycle_number=cycle_num,
            memories_reviewed=len(memories),
            thoughts_generated=thoughts_gen,
            thoughts_refined=thoughts_ref,
            initiative_items_added=initiative_added,
            raw_output=raw[:2000],  # cap for storage
            elapsed_s=elapsed,
        )

        logger.info(
            "Monologue cycle %d: %d memories → %d thoughts, %d refined, %d initiative (%.1fs)",
            cycle_num, len(memories), thoughts_gen, thoughts_ref, initiative_added, elapsed,
        )
        return result

    async def _select_memories(self) -> list[dict]:
        """Select diverse memories for review."""
        per_cycle = self.config.inner_monologue.memories_per_cycle
        memories = []

        # Recent memories (what just happened)
        try:
            cursor = await self.sqlite._db.execute(
                """SELECT id, summary, content, importance, timestamp
                   FROM memories
                   WHERE is_archived = 0 AND summary IS NOT NULL
                   ORDER BY timestamp DESC LIMIT ?""",
                (min(3, per_cycle),),
            )
            rows = await cursor.fetchall()
            memories.extend([dict(r) for r in rows])
        except Exception as e:
            logger.debug("Failed to get recent memories: %s", e)

        # Random high-importance (long-term reflection)
        try:
            random_mems = await self.sqlite.get_random_high_importance_memories(
                limit=min(3, per_cycle - len(memories)),
            )
            memories.extend(random_mems)
        except Exception as e:
            logger.debug("Failed to get random memories: %s", e)

        # Recently accessed (what the user was thinking about)
        try:
            accessed = await self.sqlite.get_recently_accessed_memories(
                limit=min(2, per_cycle - len(memories)),
            )
            memories.extend(accessed)
        except Exception as e:
            logger.debug("Failed to get recently accessed memories: %s", e)

        # Related to latest thought (follow the thread)
        try:
            recent_thoughts = await self.thought_engine.get_recent_thoughts(limit=1)
            if recent_thoughts and len(memories) < per_cycle:
                related = self.chroma.search_memories(
                    recent_thoughts[0].get("content", ""),
                    n_results=min(2, per_cycle - len(memories)),
                )
                for r in related:
                    memories.append({
                        "id": r["id"],
                        "summary": r.get("document", ""),
                        "importance": 0.5,
                    })
        except Exception as e:
            logger.debug("Failed to get thought-related memories: %s", e)

        # Deduplicate by ID
        seen = set()
        unique = []
        for m in memories:
            mid = m.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(m)

        return unique[:per_cycle]

    async def _process_output(
        self, raw: str, cycle_num: int,
    ) -> tuple[int, int, int]:
        """Parse monologue output and store thoughts/refinements/initiative items.

        Returns (thoughts_generated, thoughts_refined, initiative_added).
        """
        thoughts_gen = 0
        thoughts_ref = 0
        initiative_added = 0

        # Parse new thoughts (reuse ThoughtEngine parser)
        from blipshell.alive.thought_engine import ThoughtEngine
        thoughts = ThoughtEngine._parse_thoughts(raw)

        for thought in thoughts:
            if thought.confidence < self.config.thought.min_confidence:
                continue
            # Dedup
            if await self.thought_engine._is_duplicate(thought.content):
                continue

            thought_id = await self.sqlite.add_thought(
                content=thought.content,
                category=thought.category.value,
                confidence=thought.confidence,
                source_type="monologue",
            )
            try:
                self.chroma.add_thought(
                    thought_id, thought.content,
                    metadata={"category": thought.category.value},
                )
            except Exception:
                pass
            thoughts_gen += 1

        # Parse refinements: REFINE: <id>
        refine_blocks = re.findall(
            r'REFINE:\s*(\d+)\s*\nCONFIDENCE:\s*([\d.]+)\s*\nTHOUGHT:\s*(.+?)(?=\n(?:CATEGORY|REFINE|INITIATIVE):|$)',
            raw, re.IGNORECASE | re.DOTALL,
        )
        for match in refine_blocks:
            try:
                parent_id = int(match[0])
                confidence = max(0.0, min(1.0, float(match[1])))
                content = match[2].strip()
                if len(content) < 10:
                    continue
                await self.thought_engine.refine_thought(
                    parent_id, content, confidence, source_type="monologue",
                )
                thoughts_ref += 1
            except (ValueError, Exception) as e:
                logger.debug("Failed to parse refinement: %s", e)

        # Parse initiative items: INITIATIVE: <category>
        initiative_blocks = re.findall(
            r'INITIATIVE:\s*(\w+)\s*\nPRIORITY:\s*([\d.]+)\s*\nCONTENT:\s*(.+?)(?=\n(?:CATEGORY|REFINE|INITIATIVE):|$)',
            raw, re.IGNORECASE | re.DOTALL,
        )
        for match in initiative_blocks:
            try:
                category = match[0].lower()
                priority = max(0.0, min(1.0, float(match[1])))
                content = match[2].strip()
                if len(content) < 10:
                    continue
                await self.sqlite.add_initiative_item(
                    content=content,
                    category=category,
                    priority=priority,
                    source_type="monologue",
                )
                initiative_added += 1
            except (ValueError, Exception) as e:
                logger.debug("Failed to parse initiative item: %s", e)

        return thoughts_gen, thoughts_ref, initiative_added
