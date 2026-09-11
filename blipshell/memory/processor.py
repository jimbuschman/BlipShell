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
    DEDUP_RETRY_SUFFIX_JSON,
    DEDUP_RETRY_SUFFIX_TEXT,
    decide_memory_action,
    detect_contradiction,
    extract_lesson,
    merge_chunk_reflections,
    rank_and_importance,
    rank_importance_and_classify,
    rank_lesson,
    reflect_on_session,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory import dedup_decision, supersession
from blipshell.memory.manager import estimate_tokens
from blipshell.memory.noise import should_skip_memory
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_message
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import CoreMemory, Lesson, Memory, MemoryType

if TYPE_CHECKING:
    from blipshell.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


def summary_or_raw(summary: str | None, text: str) -> str:
    """An EMPTY summarization reply is a FAILURE, not a summary.

    Both LLM clients return "" for a reply carrying no content (a model that
    emitted only reasoning tokens, a filtered or truncated cloud response) and
    raise nothing, so the `except` fallback around the summarize call never
    fired: "" was written to the memory's summary — the text FTS indexes, that
    Recall renders, and that the dedup step embeds. Ollama returns NO vector
    for "" (it drops the input), so dedup died with the bare
    `IndexError: list index out of range` seen 2026-09-11. An empty reply is
    handled exactly like a raised one: keep the raw text.
    """
    if summary and summary.strip():
        return summary
    logger.warning(
        "Summarization returned an empty reply, using raw text: %s", text[:80],
    )
    return text


# A memory whose summary is blank: the summarizer answered "" and, before
# `summary_or_raw`, that empty string was stored. Active rows only - an
# archived row is out of every pool anyway.
BLANK_SUMMARY_SQL = "TRIM(COALESCE(summary, '')) = '' AND is_archived = 0"


async def find_blank_summaries(sqlite, limit: int = 100) -> list[dict]:
    """Active memories with no summary at all, oldest first."""
    cursor = await sqlite._db.execute(
        f"SELECT id, role, timestamp, content FROM memories "
        f"WHERE {BLANK_SUMMARY_SQL} ORDER BY id LIMIT ?",
        (limit,),
    )
    return [
        {"id": r["id"], "role": r["role"], "timestamp": r["timestamp"],
         "content": r["content"] or ""}
        for r in await cursor.fetchall()
    ]


async def repair_blank_summaries(
    sqlite, router, *, dry_run: bool = True, limit: int = 100,
    on_status=None,
) -> dict:
    """Re-summarize memories left with a blank summary (see `summary_or_raw`).

    Rows written before the empty-reply fallback existed keep their content -
    FTS and the embedding both index that, so they stayed findable - but every
    pool renders their summary, and it is empty. This asks the real summarizer
    for each one and applies the rule the pipeline now applies at write time.

    Three deliberate differences from the write path:

    - A SKIP verdict does NOT archive the row. At write time SKIP filters a
      new message; here the row has existed for months, has been retrievable
      and may have been recalled. A repair restores a field, it never removes
      a record - SKIP is counted and treated as "no summary offered", i.e.
      the content fallback.
    - A model FAILURE leaves the row blank and is reported. The repair is
      re-runnable, so an outage must not convert every row into a copy of its
      own content and call it done.
    - A row with no content has nothing to summarize from; it is reported,
      never touched. (There is no text anywhere in it to recover.)

    `dry_run` (the default) lists what would be repaired WITHOUT calling the
    model, so previewing the scope costs nothing on a shared GPU.
    """
    rows = await find_blank_summaries(sqlite, limit=limit)
    stats = {"found": len(rows), "resummarized": 0, "content_fallback": 0,
             "no_content": 0, "failed": 0, "skip_verdict": 0}

    def say(msg: str) -> None:
        if on_status:
            on_status(msg)

    if dry_run:
        for row in rows:
            preview = " ".join((row["content"] or "").split())[:60]
            say(f"  would repair memory {row['id']} ({row['role']}, "
                f"{row['timestamp']}): {preview}")
        return stats

    for row in rows:
        content = row["content"]
        if not content.strip():
            stats["no_content"] += 1
            say(f"  memory {row['id']}: no content to summarize from, left as is")
            continue
        try:
            sum_system, sum_prompt = summarize_memory(content)
            reply = await router.generate(
                TaskType.SUMMARIZATION, sum_prompt, system=sum_system,
            )
        except Exception as e:
            stats["failed"] += 1
            logger.error("Re-summarize failed for memory %d: %s", row["id"], e)
            say(f"  memory {row['id']}: summarization FAILED ({e}), left blank for a retry")
            continue

        reply = (reply or "").strip()
        if reply.upper() == "SKIP":
            stats["skip_verdict"] += 1
            reply = ""
        summary = summary_or_raw(reply, content)
        await sqlite.update_memory(row["id"], summary=summary)
        if summary == content:
            stats["content_fallback"] += 1
            say(f"  memory {row['id']}: no summary offered, using raw content")
        else:
            stats["resummarized"] += 1
            say(f"  memory {row['id']}: {summary[:70]}")

    return stats


class MemoryProcessor:
    """Background pipeline for processing memories.

    Pipeline steps:
    1. Noise check (skip low-value messages)
    2. LLM summarize (generate concise summary)
    3. SQLite insert (persist structured data)
    4. Vector embed (store vector for semantic search)
    5. Tag (extract topic/behavior tags)
    6. LLM rank+importance (combined call: rank 1-5, importance 0.0-1.0)
    """

    def __init__(self, sqlite: SQLiteStore, vectors: VectorStore, router: LLMRouter,
                 config: MemoryConfig | None = None, max_tags: int = 7):
        self.sqlite = sqlite
        self.vectors = vectors
        self.router = router
        self._recency_bonus = config.importance_recency_bonus if config else 0.1
        self._tag_bonus = config.importance_tag_bonus if config else 0.05
        self._contradiction_threshold = config.contradiction_similarity_threshold if config else 0.7
        self._max_tags = max_tags
        # Dedup config
        self._dedup_enabled = config.dedup.enabled if config else True
        self._dedup_similarity_threshold = config.dedup.similarity_threshold if config else 0.7
        self._dedup_structured = config.dedup.structured_output if config else False

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

        # `step` is updated inline as we move through the pipeline. The outer
        # try/except wraps the bubbling exception in a RuntimeError tagging
        # the failing step so worker error logs identify which DB operation
        # actually failed (e.g. FK violation on create_memory vs tag_memory).
        step = "noise_check"
        try:
            # Step 1: Noise check
            if should_skip_memory(text):
                logger.debug("Skipping noise: %s", text[:50])
                if memory_id:
                    step = "noise_archive_update"
                    await self.sqlite.update_memory(memory_id, is_archived=True, is_processed=True)
                return None

            # Step 2: Summarize
            step = "summarize"
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
                        step = "skip_archive_update"
                        await self.sqlite.update_memory(memory_id, is_archived=True, is_processed=True)
                    return None
            except Exception as e:
                logger.error("Summarization failed, using raw text: %s", e)
                summary = text
            summary = summary_or_raw(summary, text)
            t_summarize = _time.monotonic() - t0
            logger.info("process_message: summarize=%.1fs", t_summarize)

            # Step 3: SQLite insert or update
            if memory_id:
                step = "update_memory_summary"
                await self.sqlite.update_memory(memory_id, summary=summary)
            else:
                step = "create_memory"
                memory = Memory(
                    session_id=session_id,
                    role=role,
                    content=text,
                    summary=summary,
                    timestamp=timestamp or datetime.now(timezone.utc),
                    memory_type=MemoryType.CONVERSATION,
                )
                memory_id = await self.sqlite.create_memory(memory)

            # Step 4: Embed for vector search.
            # Use raw content (not summary) because summaries from cloud-routed
            # models may be PII-sanitized ([PERSON], [PII]), making names and
            # dates unsearchable. Raw content preserves the original text.
            # Fall back to summary if content is unavailable.
            step = "vector_embed"
            t0 = _time.monotonic()
            embed_text = text or summary
            embed_meta = {"session_id": str(session_id), "role": role}
            try:
                self.vectors.add_memory(memory_id, embed_text, embed_meta)
            except Exception as e:
                logger.error("Vector embed failed (will be backfilled): %s", e)
            t_embed = _time.monotonic() - t0

            # Step 4b: Dedup check — find similar memories, ask LLM what to do
            t_dedup = 0.0
            if self._dedup_enabled:
                step = "dedup"
                t0 = _time.monotonic()
                try:
                    action = await self._decide_and_apply_action(memory_id, summary)
                    if action == "NONE":
                        # Redundant — archive and skip further processing.
                        # is_processed must be set too, or the startup sweep and
                        # nightly cleanup re-pick this row forever (they filter
                        # only on is_processed = 0, not is_archived).
                        step = "dedup_archive_update"
                        await self.sqlite.update_memory(memory_id, is_archived=True, is_processed=True)
                        try:
                            self.vectors.delete_memory(memory_id)
                        except Exception as e:
                            logger.warning("Failed to delete deduped memory %d vector: %s", memory_id, e)
                        logger.info("Dedup: archived redundant memory %d", memory_id)
                        return None
                except Exception as e:
                    # exc_info: this catch is fail-open by design, but the bare
                    # message alone ("list index out of range") named neither
                    # the layer nor the input. A swallowed failure must still
                    # be diagnosable.
                    logger.error("Dedup check failed (continuing): %s", e, exc_info=True)
                t_dedup = _time.monotonic() - t0

            # Step 5: Tag
            step = "tag_memory"
            try:
                tags = tag_message(text, max_tags=self._max_tags)
                await self.sqlite.tag_memory(memory_id, tags)
            except Exception as e:
                logger.error("Tagging failed (memory_id=%s): %s", memory_id, e)
                tags = []

            # Step 6+7: Combined rank (1-5) + importance (0.0-1.0) + type in one LLM call
            step = "rank_importance_classify"
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

                step = "update_memory_scores"
                await self.sqlite.update_memory(
                    memory_id, rank=rank, importance=importance,
                    memory_type=memory_type,
                )
            except Exception as e:
                logger.error("Rank+importance+classify failed: %s", e)
            t_rank = _time.monotonic() - t0

            # Mark as fully processed
            step = "mark_processed"
            await self.sqlite.mark_memory_processed(memory_id)

            logger.info(
                "process_message: summarize=%.1fs embed=%.1fs dedup=%.1fs rank=%.1fs total=%.1fs",
                t_summarize, t_embed, t_dedup, t_rank,
                t_summarize + t_embed + t_dedup + t_rank,
            )
            return memory_id

        except Exception as e:
            raise RuntimeError(
                f"process_message failed at step={step} "
                f"memory_id={memory_id} session_id={session_id}: {e}"
            ) from e

    async def process_core_memory(
        self, text: str, session_id: int | None = None,
        source_type: str = "assistant_inference",
    ) -> int:
        """Process and store a core memory.

        `source_type` says how the fact was produced (V3 B4). The default is
        the honest one for a model-initiated save: even when the text quotes
        the user, the decision to keep it as a standing fact was the model's.
        Callers that KNOW better pass it (promotion of a user-role memory ->
        user_statement)."""
        core_memory = CoreMemory(
            content=text,
            source_session_id=session_id,
            source_type=source_type,
        )
        mem_id = await self.sqlite.create_core_memory(core_memory)

        # Embed
        try:
            self.vectors.add_core_memory(mem_id, text)
        except Exception as e:
            logger.error("Core memory embed failed (will be backfilled): %s", e)

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
    ) -> int | None:
        """Extract and store a lesson from a conversation.

        Returns lesson_id on success, None if the lesson was filtered out
        (SKIP / empty / near-duplicate). RAISES if the LLM call fails —
        "the model died" is not "there was nothing worth keeping", and
        collapsing the two hid failures from every caller (all five wrap
        this call themselves and count/report the failure).
        """
        # Generate lesson text via session review model (needs full-conversation understanding)
        lesson_system, lesson_prompt = extract_lesson(conversation_text)
        lesson_text = await self.router.generate(
            TaskType.SESSION_REVIEW,
            lesson_prompt,
            system=lesson_system,
            min_context_tokens=min_context_tokens,
        )

        # Validate: filter SKIP, empty, and junk responses
        stripped = lesson_text.strip()
        if not stripped or stripped.upper() == "SKIP" or len(stripped) < 20:
            logger.debug("Lesson filtered (SKIP/empty/short): %s", stripped[:50])
            return None

        # Dedup: check if a very similar lesson already exists
        try:
            similar = self.vectors.search_lessons(stripped, n_results=1)
            if similar and similar[0].get("similarity", 0) > 0.92:
                logger.debug(
                    "Lesson skipped (near-duplicate of lesson %s, sim=%.3f): %s",
                    similar[0].get("id"), similar[0]["similarity"], stripped[:80],
                )
                return None
        except Exception as e:
            logger.debug("Lesson dedup check failed (proceeding): %s", e)

        lesson = Lesson(
            content=lesson_text,
            source_session_id=session_id,
            project=project,
            source_type="reflection",   # a model's reading of a transcript (V3 B4)
            added_by="session_review",
        )
        lesson_id = await self.sqlite.create_lesson(lesson)

        # Embed (include project in metadata for filtered/boosted search)
        try:
            meta = {"project": project} if project else None
            self.vectors.add_lesson(lesson_id, lesson_text, metadata=meta)
        except Exception as e:
            logger.error("Lesson embed failed (will be backfilled): %s", e)

        # Tag
        try:
            tags = tag_message(lesson_text, max_tags=self._max_tags)
            await self.sqlite.tag_lesson(lesson_id, tags)
        except Exception as e:
            logger.error("Lesson tagging failed: %s", e)

        # Score: rank (1-5) + importance (0.0-1.0)
        try:
            ri_system, ri_prompt = rank_lesson(lesson_text)
            ri_text = await self.router.generate(
                TaskType.RANKING_IMPORTANCE,
                ri_prompt,
                system=ri_system,
            )
            rank, importance = self._parse_rank_and_importance(ri_text)
            await self.sqlite.update_lesson_scores(lesson_id, rank, importance)
            logger.debug("Lesson %d scored: rank=%d importance=%.2f", lesson_id, rank, importance)
        except Exception as e:
            logger.error("Lesson scoring failed (keeping defaults): %s", e)

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
        results = self.vectors.search_core_memories(text, n_results=3)

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
                    self.vectors.delete_core_memory(r["id"])
                except Exception as e:
                    logger.warning("Failed to delete contradicted core memory %d vector: %s", r["id"], e)
                # The relationship is a record, not a log line (V3 E1).
                try:
                    prov = await self.sqlite.get_provenance("core_memories", [core_memory_id])
                    await supersession.record(
                        self.sqlite,
                        old_kind="core_memory", old_id=int(r["id"]),
                        new_kind="core_memory", new_id=int(core_memory_id),
                        scope=None, relation="contradicts", detected_by="core_contradiction",
                        evidence=answer.strip()[:120],
                        source_type=prov.get(core_memory_id, ("unknown", ""))[0],
                    )
                except Exception as e:
                    logger.warning("Could not record core-memory supersession %d -> %d: %s",
                                   r["id"], core_memory_id, e)
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
        results = self.vectors.search_memories(summary, n_results=n_results + 1)
        similar = []
        for r in results:
            if r["id"] == exclude_id:
                continue
            if r["similarity"] < self._dedup_similarity_threshold:
                continue
            similar.append(r)
        return similar[:n_results]

    async def _ask_dedup_verdict(
        self, summary: str, existing_summaries: list[str], n_candidates: int,
    ) -> tuple[str, int | None, str]:
        """Ask the model for a verdict; re-ask ONCE if it cannot be parsed.

        Returns (action, 0-based index | None, last raw reply). `action` is
        RETRY when both replies were unusable or out of range - the caller
        then keeps the new memory and archives nothing.
        """
        structured = self._dedup_structured
        system, prompt = decide_memory_action(summary, existing_summaries, structured=structured)
        gen_kwargs: dict = {"system": system, "think": False}
        if structured:
            gen_kwargs["response_format"] = dedup_decision.MEMORY_ACTION_SCHEMA
        suffix = DEDUP_RETRY_SUFFIX_JSON if structured else DEDUP_RETRY_SUFFIX_TEXT

        response = await self.router.generate(TaskType.REASONING, prompt, **gen_kwargs)
        action, idx = dedup_decision.parse_action(response, structured=structured)
        if action != dedup_decision.RETRY and dedup_decision.in_range(action, idx, n_candidates):
            return action, idx, response

        logger.warning(
            "Dedup verdict unusable (%s), re-asking once: %r",
            "out of range" if action != dedup_decision.RETRY else "unparseable",
            (response or "")[:200],
        )
        response = await self.router.generate(TaskType.REASONING, prompt + suffix, **gen_kwargs)
        action, idx = dedup_decision.parse_action(response, structured=structured)
        if action != dedup_decision.RETRY and dedup_decision.in_range(action, idx, n_candidates):
            return action, idx, response
        return dedup_decision.RETRY, None, response

    async def _decide_and_apply_action(
        self, new_memory_id: int, summary: str,
    ) -> str:
        """Find similar memories and ask LLM to decide: ADD/UPDATE/DELETE/NONE.

        Returns the action taken. An undecided verdict (unparseable twice, or
        naming a candidate that does not exist) is applied as ADD: the new
        memory stays, nothing is archived, and the new row's metadata records
        `dedup_undecided` so the case can be found. Every archive stamps the
        archived row with `dedup` = {action, by, candidates, reply, at}, which
        `blipshell repair --unarchive-memory` reads to explain and reverse it.
        """
        similar = await self._find_similar_memories(summary, exclude_id=new_memory_id)
        if not similar:
            return "ADD"

        # Scope (V3 E1): a verdict reached for a memory in project A may only
        # supersede candidates in A or in no project. Two different projects
        # never supersede each other, however alike the sentences look.
        try:
            projects = await supersession.memory_projects(
                self.sqlite, [new_memory_id] + [s["id"] for s in similar],
            )
        except Exception as e:
            logger.warning("Dedup scope lookup failed (treating all as global): %s", e)
            projects = {}
        new_project = projects.get(new_memory_id)
        out_of_scope = [s["id"] for s in similar
                        if not supersession.same_scope(new_project, projects.get(s["id"]))]
        if out_of_scope:
            logger.info("Dedup: %d candidate(s) in other projects excluded for memory %d: %s",
                        len(out_of_scope), new_memory_id, out_of_scope)
            similar = [s for s in similar if s["id"] not in out_of_scope]
            if not similar:
                return "ADD"

        candidate_ids = [s["id"] for s in similar]
        existing_summaries = [s["document"] for s in similar]
        action, target_idx, response = await self._ask_dedup_verdict(
            summary, existing_summaries, len(similar),
        )

        if action == dedup_decision.RETRY:
            logger.warning(
                "Dedup undecided for memory %d (candidates %s); keeping it, archiving nothing. reply=%r",
                new_memory_id, candidate_ids, (response or "")[:200],
            )
            try:
                await dedup_decision.merge_metadata(self.sqlite, new_memory_id, {
                    "dedup_undecided": dedup_decision.undecided_record(
                        candidates=candidate_ids, reply=response, structured=self._dedup_structured,
                    ),
                })
            except Exception as e:
                logger.warning("Could not record undecided dedup on memory %d: %s", new_memory_id, e)
            return "ADD"

        if action in ("ADD", "NONE"):
            return action

        # UPDATE / DELETE: the named candidate is SUPERSEDED, not archived
        # (V3 E1). It stays in place with its vector; a supersession row says
        # the new memory replaced it, in which scope, on what evidence. Search
        # hides it for current-state questions and labels it for historical
        # ones. The `dedup` stamp on the old row keeps the verdict readable.
        old_id = similar[target_idx]["id"]
        record = dedup_decision.archive_record(
            action=action, by_memory_id=new_memory_id, candidates=candidate_ids,
            reply=response, structured=self._dedup_structured,
        )
        if action == "UPDATE":
            # The refined memory inherits the old one's tags.
            await self.sqlite.transfer_memory_tags(old_id, new_memory_id)
        await dedup_decision.merge_metadata(
            self.sqlite, old_id, {"dedup": record, "superseded_by": new_memory_id},
        )
        new_mem = await self.sqlite.get_memory(new_memory_id)
        new_role = getattr(new_mem, "role", "") or ""
        await supersession.record(
            self.sqlite,
            old_kind="memory", old_id=old_id, new_kind="memory", new_id=new_memory_id,
            scope=new_project,
            relation="refines" if action == "UPDATE" else "contradicts",
            detected_by="dedup_verdict",
            evidence=response or "",
            source_type="user_statement" if new_role == "user" else "assistant_inference",
        )
        logger.info(
            "Dedup: %s — memory %d superseded by %d (candidate %d of %s, scope %s); reply=%r",
            action, old_id, new_memory_id, target_idx + 1, candidate_ids,
            supersession.scope_of(new_project), (response or "")[:120],
        )
        return action

    @staticmethod
    def _parse_memory_action(text: str) -> tuple[str, int | None]:
        """Parse a free-text dedup verdict (strict grammar; see dedup_decision).

        Returns (action, 0-based index | None); ("RETRY", None) when the reply
        is not unambiguously one verdict. There is no default target.
        """
        return dedup_decision.parse_action_text(text)

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
            await self._save_skipped_reflection(session_id)
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
                    part=(i + 1, len(conversation_chunks)),
                )
                if chunk_raw.strip().upper() != "SKIP":
                    chunk_reflections.append(chunk_raw)

            if not chunk_reflections:
                await self._save_skipped_reflection(session_id)
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
            await self._save_skipped_reflection(session_id)
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

        # Embed for search — reflections have their own vec table, keyed by
        # the real reflection id. (They used to go into the lessons
        # collection at id+100000, which lesson-search enrichment could
        # never return.)
        embed_text = self._build_reflection_embed_text(parsed)
        try:
            meta = {
                "type": "reflection",
                "session_id": str(session_id),
            }
            if project:
                meta["project"] = project
            self.vectors.add_reflection(reflection_id, embed_text, metadata=meta)
        except Exception as e:
            logger.error("Reflection embed failed (will be backfilled): %s", e)

        return parsed

    async def _save_skipped_reflection(self, session_id: int):
        """Save a placeholder reflection so skipped sessions don't reappear."""
        try:
            await self.sqlite.create_session_reflection(
                session_id=session_id,
                effectiveness="skipped",
                reflection_text="Session skipped — insufficient conversation data.",
            )
        except Exception as e:
            if "UNIQUE constraint" not in str(e) and "IntegrityError" not in type(e).__name__:
                logger.warning("Failed to create session reflection: %s", e)

    async def _reflect_on_text(
        self, session_summary: str, conversation_text: str, project: str | None,
        min_context_tokens: int | None = None,
        part: tuple[int, int] | None = None,
    ) -> str:
        """Run the reflection LLM call on a single text. Returns raw output.

        ``part`` is (index, total) when this text is one chunk of a session
        too big for the context window — it switches to the chunk-scoped
        prompt so the model doesn't judge a fragment as a whole session.
        """
        system, user_prompt = reflect_on_session(
            session_summary, conversation_text, project, part=part,
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
        messages = await self.sqlite.get_session_messages_for_lesson(session_id, include_archived=True)
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

    async def analyze_session_friction(
        self,
        session_id: int,
        session_summary: str,
        conversation_text: str,
        project: str | None = None,
    ) -> list[dict]:
        """Analyze a session for system-level friction.

        Runs on SESSION_REVIEW (whole-session analysis on the fast, large-context
        endpoint), NOT REASONING. The conversation chunk is already sized for
        SESSION_REVIEW by ``prepare_conversation_for_reflection``; routing the
        call to REASONING (local-only qwen3:14b at 32K ctx) meant a big chunk was
        fed to a slow local model, blowing past the per-session timeout so heavy
        sessions were skipped forever. Mirrors the sibling ``_reflect_on_text``.

        Returns list of parsed friction items, or empty list if NONE.
        """
        from blipshell.llm.prompts import analyze_session_friction

        system, user_prompt = analyze_session_friction(
            session_summary, conversation_text, project,
        )
        # Route large chunks to a bigger-context endpoint (same threshold as
        # session_reflections). Below the threshold, SESSION_REVIEW already
        # prefers the cloud endpoint by priority, so leave min_ctx unset.
        est_tokens = estimate_tokens(user_prompt) + estimate_tokens(system)
        min_ctx = est_tokens + 4096 if est_tokens > 28000 else None
        try:
            raw = await self.router.generate(
                TaskType.SESSION_REVIEW, user_prompt, system=system,
                min_context_tokens=min_ctx,
            )
        except Exception as e:
            logger.error("Friction analysis LLM call failed: %s", e)
            return []

        return self._parse_friction_response(raw, session_id, source="nightly")

    async def analyze_idle_friction(
        self,
        session_id: int,
        conversation_text: str,
    ) -> list[dict]:
        """Mid-session friction probe during idle time.

        Returns list of parsed friction items, or empty list if NONE.
        """
        from blipshell.llm.prompts import idle_friction_probe

        system, user_prompt = idle_friction_probe(conversation_text)
        try:
            raw = await self.router.generate(
                TaskType.REASONING, user_prompt, system=system,
            )
        except Exception as e:
            logger.error("Idle friction probe LLM call failed: %s", e)
            return []

        return self._parse_friction_response(raw, session_id, source="idle_probe")

    @staticmethod
    def _parse_friction_response(
        raw: str, session_id: int | None, source: str,
    ) -> list[dict]:
        """Parse friction analysis output into structured items."""
        raw = raw.strip()
        if not raw or raw.upper() == "NONE":
            return []

        valid_categories = {
            "TOOL_FAILURE", "REPEATED_RETRY", "MISSING_CAPABILITY",
            "WORKFLOW_FRICTION", "CONTEXT_ISSUE",
            # idle probe categories
            "TOOL_ISSUE", "MISSING_FEATURE", "CONTEXT_PROBLEM", "WORKFLOW_ISSUE",
        }

        items = []
        for line in raw.split("\n"):
            line = line.strip().lstrip("- ")
            if not line or line.upper() == "NONE":
                continue
            # Parse "CATEGORY: description"
            if ":" in line:
                cat, _, desc = line.partition(":")
                cat = cat.strip().upper()
                desc = desc.strip()
                if cat in valid_categories and desc:
                    items.append({
                        "session_id": session_id,
                        "source": source,
                        "category": cat,
                        "description": desc,
                    })

        return items

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
