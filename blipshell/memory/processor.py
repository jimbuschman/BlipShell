"""Background memory processing pipeline.

Port of MemoryDB.CreateMemoryAsync pipeline:
noise check -> LLM summarize -> SQLite insert -> ChromaDB embed -> tag -> LLM rank+importance
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from blipshell.llm.prompts import (
    decide_memory_action,
    detect_contradiction,
    extract_lesson,
    merge_chunk_reflections,
    rank_and_importance,
    rank_importance_and_classify,
    reflect_on_session,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_retry import (
    OP_DELETE, OP_UPSERT,
    COLLECTION_CORE, COLLECTION_LESSONS, COLLECTION_MEMORIES,
    queue_failed_op,
)
from blipshell.memory.manager import estimate_tokens
from blipshell.memory.noise import should_skip_memory
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_message
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import CoreMemory, Lesson, Memory, MemoryType

if TYPE_CHECKING:
    from blipshell.memory.chroma_store import ChromaStore

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
        # Dedup config
        self._dedup_enabled = config.dedup.enabled if config else True
        self._dedup_similarity_threshold = config.dedup.similarity_threshold if config else 0.7

    async def process_message(
        self,
        text: str,
        role: str,
        session_id: int,
        metadata: str = "{}",
        timestamp: datetime | None = None,
        memory_id: int | None = None,
    ) -> int | None:
        """Full pipeline for processing a conversation message into memory.

        If memory_id is provided, updates an existing raw memory row
        (created by save_raw_memory during live sessions).
        Otherwise creates a new row (import path, crash recovery).

        Returns the memory ID, or None if filtered as noise/skip.
        """
        import time as _time

        # Step 1: Noise check
        if should_skip_memory(text):
            logger.debug("Skipping noise: %s", text[:50])
            if memory_id:
                await self.sqlite.update_memory(memory_id, is_archived=True, is_processed=True)
            return None

        # Step 2: Summarize
        t0 = _time.monotonic()
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
                if memory_id:
                    await self.sqlite.update_memory(memory_id, is_archived=True, is_processed=True)
                return None
        except Exception as e:
            logger.error("Summarization failed, using raw text: %s", e)
            summary = text
        t_summarize = _time.monotonic() - t0
        logger.info("process_message: summarize=%.1fs", t_summarize)

        # Step 3: SQLite insert or update
        if memory_id:
            # Update existing raw memory row with processed data
            await self.sqlite.update_memory(memory_id, summary=summary)
        else:
            # Create new row (import path, crash recovery reprocess)
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
        t0 = _time.monotonic()
        embed_meta = {"session_id": str(session_id), "role": role}
        try:
            self.chroma.add_memory(memory_id, summary, embed_meta)
        except Exception as e:
            logger.error("ChromaDB embed failed (queued for retry): %s", e)
            await queue_failed_op(
                self.sqlite, OP_UPSERT, COLLECTION_MEMORIES,
                memory_id, summary, embed_meta, str(e),
            )
        t_embed = _time.monotonic() - t0

        # Step 4b: Dedup check — find similar memories, ask LLM what to do
        t_dedup = 0.0
        if self._dedup_enabled:
            t0 = _time.monotonic()
            try:
                action = await self._decide_and_apply_action(memory_id, summary)
                if action == "NONE":
                    # Redundant — archive and skip further processing
                    await self.sqlite.update_memory(memory_id, is_archived=True)
                    try:
                        self.chroma.delete_memory(memory_id)
                    except Exception as e:
                        logger.warning("Failed to delete deduped memory %d from ChromaDB (queued): %s", memory_id, e)
                        await queue_failed_op(
                            self.sqlite, OP_DELETE, COLLECTION_MEMORIES,
                            memory_id, error=str(e),
                        )
                    logger.info("Dedup: archived redundant memory %d", memory_id)
                    return None
            except Exception as e:
                logger.error("Dedup check failed (continuing): %s", e)
            t_dedup = _time.monotonic() - t0

        # Step 5: Tag
        try:
            tags = tag_message(text, max_tags=self._max_tags)
            await self.sqlite.tag_memory(memory_id, tags)
        except Exception as e:
            logger.error("Tagging failed: %s", e)
            tags = []

        # Step 6+7: Combined rank (1-5) + importance (0.0-1.0) + type in one LLM call
        t0 = _time.monotonic()
        try:
            ri_system, ri_prompt = rank_importance_and_classify(text)
            ri_text = await self.router.generate(
                TaskType.RANKING_IMPORTANCE,
                ri_prompt,
                system=ri_system,
            )
            rank, importance, memory_type = self._parse_rank_importance_type(ri_text)
            logger.debug("Classification: raw=%r → rank=%d imp=%.2f type=%s", ri_text.strip(), rank, importance, memory_type)

            # Apply bonuses
            importance += self._recency_bonus
            if len(tags) > 6:
                importance += self._tag_bonus
            importance = min(importance, 1.0)

            await self.sqlite.update_memory(
                memory_id, rank=rank, importance=importance,
                memory_type=memory_type,
            )
        except Exception as e:
            logger.error("Rank+importance+classify failed: %s", e)
        t_rank = _time.monotonic() - t0

        # Mark as fully processed
        await self.sqlite.mark_memory_processed(memory_id)

        logger.info(
            "process_message: summarize=%.1fs embed=%.1fs dedup=%.1fs rank=%.1fs total=%.1fs",
            t_summarize, t_embed, t_dedup, t_rank,
            t_summarize + t_embed + t_dedup + t_rank,
        )
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
            logger.error("Core memory embed failed (queued for retry): %s", e)
            await queue_failed_op(
                self.sqlite, OP_UPSERT, COLLECTION_CORE,
                mem_id, text, error=str(e),
            )

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

    async def process_lesson(
        self, conversation_text: str, session_id: int,
        project: str | None = None,
        min_context_tokens: int | None = None,
    ) -> int:
        """Extract and store a lesson from a conversation."""
        # Generate lesson text via session review model (needs full-conversation understanding)
        try:
            lesson_system, lesson_prompt = extract_lesson(conversation_text)
            lesson_text = await self.router.generate(
                TaskType.SESSION_REVIEW,
                lesson_prompt,
                system=lesson_system,
                min_context_tokens=min_context_tokens,
            )
        except Exception as e:
            logger.error("Lesson extraction failed: %s", e)
            lesson_text = conversation_text

        lesson = Lesson(
            content=lesson_text,
            source_session_id=session_id,
            project=project,
        )
        lesson_id = await self.sqlite.create_lesson(lesson)

        # Embed (include project in metadata for filtered/boosted search)
        try:
            meta = {"project": project} if project else None
            self.chroma.add_lesson(lesson_id, lesson_text, metadata=meta)
        except Exception as e:
            logger.error("Lesson embed failed (queued for retry): %s", e)
            await queue_failed_op(
                self.sqlite, OP_UPSERT, COLLECTION_LESSONS,
                lesson_id, lesson_text, meta, str(e),
            )

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
                except Exception as e:
                    logger.warning("Failed to delete contradicted core memory %d from ChromaDB (queued): %s", r["id"], e)
                    await queue_failed_op(
                        self.sqlite, OP_DELETE, COLLECTION_CORE,
                        r["id"], error=str(e),
                    )
                deactivated += 1
                logger.info(
                    "Deactivated contradicted core memory %d (superseded by %d)",
                    r["id"], core_memory_id,
                )

        return deactivated

    # --- Memory Dedup (Feature 3) ---

    async def _find_similar_memories(
        self, summary: str, exclude_id: int, n_results: int = 3,
    ) -> list[dict]:
        """Find existing memories similar to a new summary via ChromaDB.

        Returns list of {id, document, similarity} dicts above threshold.
        """
        results = self.chroma.search_memories(summary, n_results=n_results + 1)
        similar = []
        for r in results:
            if r["id"] == exclude_id:
                continue
            if r["similarity"] < self._dedup_similarity_threshold:
                continue
            similar.append(r)
        return similar[:n_results]

    async def _decide_and_apply_action(
        self, new_memory_id: int, summary: str,
    ) -> str:
        """Find similar memories and ask LLM to decide: ADD/UPDATE/DELETE/NONE.

        Returns the action taken.
        """
        similar = await self._find_similar_memories(summary, exclude_id=new_memory_id)
        if not similar:
            return "ADD"

        existing_summaries = [s["document"] for s in similar]
        system, prompt = decide_memory_action(summary, existing_summaries)
        response = await self.router.generate(
            TaskType.REASONING, prompt, system=system, think=False,
        )

        action, target_idx = self._parse_memory_action(response)

        if action == "NONE":
            return "NONE"

        if action == "UPDATE" and target_idx is not None and target_idx < len(similar):
            old_id = similar[target_idx]["id"]
            # Transfer tags from old → new, archive old
            await self.sqlite.transfer_memory_tags(old_id, new_memory_id)
            await self.sqlite.update_memory(old_id, is_archived=True)
            try:
                self.chroma.delete_memory(old_id)
            except Exception as e:
                logger.warning("Failed to delete memory %d from ChromaDB during dedup (queued): %s", old_id, e)
                await queue_failed_op(
                    self.sqlite, OP_DELETE, COLLECTION_MEMORIES,
                    old_id, error=str(e),
                )
            logger.info("Dedup: UPDATE — archived old memory %d in favor of %d", old_id, new_memory_id)
            return "UPDATE"

        if action == "DELETE" and target_idx is not None and target_idx < len(similar):
            old_id = similar[target_idx]["id"]
            await self.sqlite.update_memory(old_id, is_archived=True)
            try:
                self.chroma.delete_memory(old_id)
            except Exception as e:
                logger.warning("Failed to delete memory %d from ChromaDB during dedup (queued): %s", old_id, e)
                await queue_failed_op(
                    self.sqlite, OP_DELETE, COLLECTION_MEMORIES,
                    old_id, error=str(e),
                )
            logger.info("Dedup: DELETE — archived contradicted memory %d", old_id)
            return "DELETE"

        return "ADD"

    @staticmethod
    def _parse_memory_action(text: str) -> tuple[str, int | None]:
        """Parse LLM dedup action response.

        Returns (action, target_index) where target_index is 0-based.
        Examples: "ADD" → ("ADD", None), "UPDATE 1" → ("UPDATE", 0), "DELETE 2" → ("DELETE", 1)
        """
        text = text.strip().upper()
        # Look for action word
        for action in ("NONE", "UPDATE", "DELETE", "ADD"):
            if action in text:
                # Try to extract a number for UPDATE/DELETE
                if action in ("UPDATE", "DELETE"):
                    numbers = re.findall(r"\d+", text)
                    if numbers:
                        idx = int(numbers[0]) - 1  # 1-based to 0-based
                        return action, max(idx, 0)
                    return action, 0  # default to first if no number
                return action, None
        return "ADD", None  # default to ADD if unparseable

    _VALID_MEMORY_TYPES = {"fact", "event", "preference", "skill", "conversation"}

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
    def _parse_rank_importance_type(text: str) -> tuple[int, float, str]:
        """Parse combined 'rank importance type' from LLM response (e.g. '4 0.7 fact').

        Falls back to 'conversation' if type is missing or unrecognized.
        """
        text = text.strip()
        numbers = re.findall(r"(\d+\.?\d*)", text)
        rank = 3
        importance = 0.3
        memory_type = "conversation"

        if len(numbers) >= 1:
            r = int(float(numbers[0]))
            if 1 <= r <= 5:
                rank = r
        if len(numbers) >= 2:
            imp = float(numbers[1])
            importance = min(max(imp, 0.0), 1.0)

        # Extract memory type — prefer last word (prompt format: "rank importance type")
        # then fall back to reverse scan if last word isn't a valid type
        words = text.lower().split()
        if words:
            last_cleaned = re.sub(r"[^a-z]", "", words[-1])
            if last_cleaned in MemoryProcessor._VALID_MEMORY_TYPES:
                memory_type = last_cleaned
            else:
                for word in reversed(words):
                    cleaned = re.sub(r"[^a-z]", "", word)
                    if cleaned in MemoryProcessor._VALID_MEMORY_TYPES:
                        memory_type = cleaned
                        break

        return rank, importance, memory_type

    # --- Session Reflection ---

    async def process_reflection(
        self,
        session_id: int,
        session_summary: str,
        conversation_chunks: list[str],
        project: str | None = None,
        min_context_tokens: int | None = None,
    ) -> dict | None:
        """Generate and store a session reflection.

        Accepts a list of conversation chunks (from prepare_conversation_for_reflection).
        Single chunk: reflect directly. Multiple chunks: reflect on each, then merge.

        Args:
            min_context_tokens: If set, prefer endpoints with at least this context window.
                Passed through to router so large sessions route to cloud endpoints.

        Returns the parsed reflection dict, or None if the session was SKIP-ped
        or had no conversation data.
        """
        if not conversation_chunks:
            logger.warning("No conversation chunks for session %d — skipping", session_id)
            return None

        if len(conversation_chunks) == 1:
            # Single chunk — reflect directly
            raw = await self._reflect_on_text(
                session_summary, conversation_chunks[0], project,
                min_context_tokens=min_context_tokens,
            )
        else:
            # Multiple chunks — reflect on each, then merge
            chunk_reflections = []
            for i, chunk in enumerate(conversation_chunks):
                logger.info(
                    "Reflecting on chunk %d/%d for session %d",
                    i + 1, len(conversation_chunks), session_id,
                )
                chunk_raw = await self._reflect_on_text(
                    session_summary, chunk, project,
                )
                if chunk_raw.strip().upper() != "SKIP":
                    chunk_reflections.append(chunk_raw)

            if not chunk_reflections:
                return None  # All chunks were trivial

            # Merge chunk reflections
            system, user_prompt = merge_chunk_reflections(
                session_summary, chunk_reflections, project,
            )
            try:
                raw = await self.router.generate(
                    TaskType.SESSION_REVIEW, user_prompt, system=system,
                )
            except Exception as e:
                logger.error("Reflection merge failed: %s", e)
                raise

        # Check for SKIP
        if raw.strip().upper() == "SKIP":
            return None

        parsed = self._parse_reflection(raw)

        # Store in SQLite
        reflection_id = await self.sqlite.create_session_reflection(
            session_id=session_id,
            effectiveness=parsed["effectiveness"],
            reflection_text=raw.strip(),
            technical_insights=parsed.get("technical_insights"),
            process_insights=parsed.get("process_insights"),
            what_worked=parsed.get("what_worked"),
            what_didnt_work=parsed.get("what_didnt_work"),
        )

        # Embed in ChromaDB lessons collection for search
        embed_text = self._build_reflection_embed_text(parsed)
        try:
            meta = {
                "type": "reflection",
                "session_id": str(session_id),
            }
            if project:
                meta["project"] = project
            self.chroma.add_lesson(reflection_id + 100000, embed_text, metadata=meta)
        except Exception as e:
            logger.error("Reflection embed failed (queued for retry): %s", e)
            await queue_failed_op(
                self.sqlite, OP_UPSERT, COLLECTION_LESSONS,
                reflection_id + 100000, embed_text, meta, str(e),
            )

        return parsed

    async def _reflect_on_text(
        self, session_summary: str, conversation_text: str, project: str | None,
        min_context_tokens: int | None = None,
    ) -> str:
        """Run the reflection LLM call on a single text. Returns raw output."""
        system, user_prompt = reflect_on_session(
            session_summary, conversation_text, project,
        )
        try:
            return await self.router.generate(
                TaskType.SESSION_REVIEW, user_prompt, system=system,
                min_context_tokens=min_context_tokens,
            )
        except Exception as e:
            logger.error("Session reflection LLM call failed: %s", e)
            raise

    async def prepare_conversation_for_reflection(
        self, session_id: int, session_summary: str,
    ) -> tuple[list[str], int]:
        """Build full conversation text for reflection, chunked if needed.

        Returns (chunks, estimated_tokens). Most sessions produce a single chunk.
        Large sessions that exceed the local context window are routed to a
        bigger-context endpoint; if still too big, they're chunked.
        """
        messages = await self.sqlite.get_session_messages_for_lesson(session_id)
        if not messages:
            logger.warning("Session %d has no conversation data — skipping reflection", session_id)
            return [], 0

        lines = [f"{m['role']}: {m['content']}" for m in messages]
        full_text = "\n".join(lines)
        total_tokens = estimate_tokens(full_text)

        # Ask the router for the best endpoint — if the session is large,
        # min_context_tokens steers toward a bigger-context endpoint (e.g. cloud)
        context_tokens = await self.router.get_context_tokens(
            TaskType.SESSION_REVIEW, min_context_tokens=total_tokens + 4096,
        )
        # Reserve ~4K for system prompt + response
        max_tokens = max(context_tokens - 4096, context_tokens // 2)

        if total_tokens <= max_tokens:
            return [full_text], total_tokens

        # Chunk by tokens — accumulate messages until we approach the limit
        chunks = []
        current_batch = []
        current_tokens = 0
        msg_idx = 0

        for msg in messages:
            line = f"{msg['role']}: {msg['content']}"
            line_tokens = estimate_tokens(line)

            if current_tokens + line_tokens > max_tokens and current_batch:
                # Flush current batch as a chunk
                chunk_text = f"[Part {len(chunks) + 1}, messages {msg_idx - len(current_batch) + 1}-{msg_idx}]\n"
                chunk_text += "\n".join(current_batch)
                chunks.append(chunk_text)
                current_batch = []
                current_tokens = 0

            current_batch.append(line)
            current_tokens += line_tokens
            msg_idx += 1

        # Flush remaining
        if current_batch:
            chunk_text = f"[Part {len(chunks) + 1}, messages {msg_idx - len(current_batch) + 1}-{msg_idx}]\n"
            chunk_text += "\n".join(current_batch)
            chunks.append(chunk_text)

        return chunks, total_tokens

    @staticmethod
    def _parse_reflection(text: str) -> dict:
        """Parse structured reflection output from LLM.

        Tolerant parser — extracts sections by label, handles missing sections.
        """
        result = {
            "effectiveness": "unclear",
            "what_worked": None,
            "what_didnt_work": None,
            "technical_insights": None,
            "process_insights": None,
        }

        # Map of section label → dict key
        sections = {
            "EFFECTIVENESS": "effectiveness",
            "WHAT_WORKED": "what_worked",
            "WHAT_DIDNT_WORK": "what_didnt_work",
            "TECHNICAL_INSIGHTS": "technical_insights",
            "PROCESS_INSIGHTS": "process_insights",
        }

        # Find section positions
        positions = []
        for label in sections:
            pattern = rf"^{label}\s*:?\s*"
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                positions.append((match.start(), match.end(), label))

        positions.sort(key=lambda x: x[0])

        # Extract content between positions
        for i, (start, content_start, label) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            content = text[content_start:end].strip()

            key = sections[label]
            if key == "effectiveness":
                # Extract just the keyword — check longer matches first
                # to avoid "effective" matching inside "ineffective"
                normalized = content.lower().replace(" ", "_")
                for val in ("partially_effective", "ineffective", "effective", "unclear"):
                    if val in normalized:
                        result["effectiveness"] = val
                        break
            else:
                result[key] = content if content else None

        return result

    @staticmethod
    def _build_reflection_embed_text(parsed: dict) -> str:
        """Build text for ChromaDB embedding from parsed reflection."""
        parts = []
        if parsed.get("what_worked"):
            parts.append(f"What worked: {parsed['what_worked']}")
        if parsed.get("what_didnt_work"):
            parts.append(f"What didn't work: {parsed['what_didnt_work']}")
        if parsed.get("technical_insights"):
            parts.append(f"Technical insights: {parsed['technical_insights']}")
        if parsed.get("process_insights"):
            parts.append(f"Process insights: {parsed['process_insights']}")
        return "\n".join(parts) if parts else "Session reflection"

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
