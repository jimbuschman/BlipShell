"""Shared import pipeline — format-agnostic conversation import.

Provides the dataclasses and pipeline functions used by all format-specific
importers (ChatGPT, Claude, DeepSeek). Each importer only needs to produce
a list[ParsedConversation] and then call import_conversations().
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from typing import Callable, Optional

from blipshell.llm.prompts import (
    ask_importance,
    rank_and_importance,
    rank_memory,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.noise import should_skip_memory
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_message
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for parsed conversations (format-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class ParsedMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None


@dataclass
class ParsedConversation:
    title: str
    created_at: Optional[float] = None
    messages: list[ParsedMessage] = field(default_factory=list)


@dataclass
class ImportStats:
    conversations_imported: int = 0
    conversations_skipped: int = 0
    conversations_reimported: int = 0
    messages_processed: int = 0
    messages_skipped_noise: int = 0
    lessons_extracted: int = 0


# ---------------------------------------------------------------------------
# Conversation import pipeline
# ---------------------------------------------------------------------------

async def import_conversations(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    config: MemoryConfig,
    conversations: list[ParsedConversation],
    on_progress: Optional[Callable[[int, int, str, "ImportStats"], None]] = None,
    skip_lessons: bool = False,
    max_concurrent: int = 3,
    batch_mode: bool = True,
) -> ImportStats:
    """Import parsed conversations through the BlipShell memory pipeline.

    When batch_mode is True (default), uses a global batch pipeline that
    processes ALL conversations phase-by-phase to minimize model swaps.
    When False, uses the original per-conversation concurrent approach.

    Args:
        sqlite: SQLite store instance
        chroma: ChromaDB store instance
        router: LLM router for summarization/ranking
        config: Memory configuration
        conversations: Parsed conversations to import
        on_progress: Callback(current_index, total, title, stats) for progress tracking
        skip_lessons: If True, skip lesson extraction (faster)
        max_concurrent: Number of conversations to process in parallel (non-batch mode)
        batch_mode: If True, use global batch pipeline (minimizes model swaps)

    Returns:
        ImportStats with counts of imported items
    """
    processor = MemoryProcessor(sqlite, chroma, router, config=config)
    stats = ImportStats()

    if batch_mode and len(conversations) > 1:
        await _import_global_batch(
            sqlite, chroma, router, processor, config,
            conversations, stats, skip_lessons, on_progress,
        )
        return stats

    # --- Original concurrent approach (single conversation or batch_mode=False) ---
    total = len(conversations)
    semaphore = asyncio.Semaphore(max_concurrent)
    progress_count = 0

    async def _process_one(i: int, conv: ParsedConversation):
        nonlocal progress_count

        # Resume support: skip complete conversations, re-import incomplete ones
        session_id, db_count = await sqlite.get_session_message_count(conv.title)
        if session_id is not None:
            if db_count > 0:
                logger.info("Skipping already imported (%d msgs): %s", db_count, conv.title)
                stats.conversations_skipped += 1
                progress_count += 1
                if on_progress:
                    on_progress(progress_count, total, conv.title, stats)
                return
            else:
                logger.info(
                    "Re-importing incomplete '%s' (interrupted, 0 messages saved)",
                    conv.title,
                )
                try:
                    cursor = await sqlite._db.execute(
                        "SELECT id FROM memories WHERE session_id = ?",
                        (session_id,),
                    )
                    old_ids = [row[0] for row in await cursor.fetchall()]
                    for mid in old_ids:
                        try:
                            chroma.delete_memory(mid)
                        except Exception:
                            pass
                    cursor = await sqlite._db.execute(
                        "SELECT id FROM lessons WHERE source_session_id = ?",
                        (session_id,),
                    )
                    old_lesson_ids = [row[0] for row in await cursor.fetchall()]
                    for lid in old_lesson_ids:
                        try:
                            chroma.delete_lesson(lid)
                        except Exception:
                            pass
                    await sqlite.delete_session_cascade(session_id, old_ids)
                    stats.conversations_reimported += 1
                except Exception as e:
                    logger.error("Failed to clean up incomplete '%s': %s", conv.title, e)
                    return

        async with semaphore:
            try:
                await _import_single_conversation(
                    sqlite, chroma, router, processor, config, conv, stats,
                    skip_lessons,
                )
                stats.conversations_imported += 1
            except Exception as e:
                logger.error("Failed to import conversation '%s': %s", conv.title, e)

        progress_count += 1
        if on_progress:
            on_progress(progress_count, total, conv.title, stats)

    # Launch all tasks and let the semaphore control concurrency
    tasks = [asyncio.create_task(_process_one(i, conv)) for i, conv in enumerate(conversations)]
    await asyncio.gather(*tasks)

    return stats


# ---------------------------------------------------------------------------
# Global batch pipeline — minimizes model swaps across all conversations
# ---------------------------------------------------------------------------

async def _cleanup_incomplete_session(
    sqlite: SQLiteStore, chroma: ChromaStore, session_id: int,
):
    """Clean up an incomplete session (0 messages saved) for re-import."""
    cursor = await sqlite._db.execute(
        "SELECT id FROM memories WHERE session_id = ?",
        (session_id,),
    )
    old_ids = [row[0] for row in await cursor.fetchall()]
    for mid in old_ids:
        try:
            chroma.delete_memory(mid)
        except Exception:
            pass
    cursor = await sqlite._db.execute(
        "SELECT id FROM lessons WHERE source_session_id = ?",
        (session_id,),
    )
    old_lesson_ids = [row[0] for row in await cursor.fetchall()]
    for lid in old_lesson_ids:
        try:
            chroma.delete_lesson(lid)
        except Exception:
            pass
    await sqlite.delete_session_cascade(session_id, old_ids)


async def _import_global_batch(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    processor: MemoryProcessor,
    config: MemoryConfig | None,
    conversations: list[ParsedConversation],
    stats: ImportStats,
    skip_lessons: bool,
    on_progress: Optional[Callable] = None,
):
    """Global batch pipeline — processes all conversations phase-by-phase.

    Minimizes model swaps by completing each LLM task type across ALL
    conversations before moving to the next task type. This means each
    model loads only once for the entire import instead of once per
    conversation.

    Phase order:
      0. Resume check (skip already-imported conversations)
      1. Filter + tag all (no LLM)
      2. Summarize all messages (glm4 — one model load)
      3. Rank + importance all messages (qwen2.5:14b — one model load, one call per msg)
      4. DB insert + embed all (no LLM)
      5. Lessons for all conversations (qwen3:14b — one model load)
    """
    total = len(conversations)
    recency_bonus = config.importance_recency_bonus if config else 0.1
    tag_bonus = config.importance_tag_bonus if config else 0.05

    # ── Phase 0: Resume check ─────────────────────────────────────────
    # work_convs: conversations that need processing
    work_convs: list[tuple[ParsedConversation, list]] = []

    for conv in conversations:
        session_id, db_count = await sqlite.get_session_message_count(conv.title)
        if session_id is not None:
            if db_count > 0:
                logger.info("Skipping already imported (%d msgs): %s", db_count, conv.title)
                stats.conversations_skipped += 1
                if on_progress:
                    on_progress(
                        stats.conversations_skipped + stats.conversations_imported,
                        total, conv.title, stats,
                    )
                continue
            else:
                # Clean up incomplete import
                logger.info("Re-importing incomplete '%s'", conv.title)
                try:
                    await _cleanup_incomplete_session(sqlite, chroma, session_id)
                    stats.conversations_reimported += 1
                except Exception as e:
                    logger.error("Failed to clean up incomplete '%s': %s", conv.title, e)
                    continue

        work_convs.append((conv, []))

    if not work_convs:
        return

    print(f"\nGlobal batch: {len(work_convs)} conversations to process "
          f"({stats.conversations_skipped} skipped)")

    # ── Phase 1: Filter + tag all (no LLM) ────────────────────────────
    for wi_idx in range(len(work_convs)):
        conv, _ = work_convs[wi_idx]
        valid: list[tuple[ParsedMessage, list[str]]] = []
        for msg in conv.messages:
            if should_skip_memory(msg.content):
                stats.messages_skipped_noise += 1
                continue
            tags = tag_message(msg.content)
            valid.append((msg, tags))
        work_convs[wi_idx] = (conv, valid)

    total_msgs = sum(len(msgs) for _, msgs in work_convs)
    print(f"Phase 1 complete: {total_msgs} messages to process "
          f"({stats.messages_skipped_noise} noise filtered)")

    # ── Phase 2: Summarize all (one model load) ───────────────────────
    # After this phase, surviving_convs has summaries attached to each message.
    surviving_convs: list[tuple[ParsedConversation, list[tuple[ParsedMessage, list[str], str]]]] = []
    msg_counter = 0

    for conv, valid_msgs in work_convs:
        surviving: list[tuple[ParsedMessage, list[str], str]] = []
        title_short = conv.title[:40]
        for msg, tags in valid_msgs:
            msg_counter += 1
            print(f"  Phase 2 — Summarizing {msg_counter}/{total_msgs} "
                  f"[{title_short}]...", end="\r")
            try:
                sum_system, sum_prompt = summarize_memory(msg.content)
                summary = await router.generate(
                    TaskType.SUMMARIZATION, sum_prompt, system=sum_system,
                )
                if summary.strip().upper() == "SKIP":
                    stats.messages_skipped_noise += 1
                    continue
            except Exception as e:
                logger.error("Summarization failed, using raw text: %s", e)
                summary = msg.content
            surviving.append((msg, tags, summary))
        surviving_convs.append((conv, surviving))

    total_surviving = sum(len(s) for _, s in surviving_convs)
    print(f"\nPhase 2 complete: {total_surviving} messages summarized" + " " * 30)

    # ── Phase 3: Rank + Importance all (one model load) ───────────────
    # all_scores[i][j] = (rank, importance) for surviving_convs[i][j]
    all_scores: list[list[tuple[int, float]]] = []
    score_counter = 0

    for conv, surviving in surviving_convs:
        conv_scores: list[tuple[int, float]] = []
        title_short = conv.title[:40]
        for msg, tags, _summary in surviving:
            score_counter += 1
            print(f"  Phase 3 — Scoring {score_counter}/{total_surviving} "
                  f"[{title_short}]...", end="\r")
            try:
                ri_system, ri_prompt = rank_and_importance(msg.content)
                ri_text = await router.generate(
                    TaskType.RANKING_IMPORTANCE, ri_prompt, system=ri_system,
                )
                rank, importance = MemoryProcessor._parse_rank_and_importance(ri_text)
            except Exception as e:
                logger.error("Rank+importance failed: %s", e)
                rank, importance = 3, 0.3

            importance += recency_bonus
            if len(tags) > 6:
                importance += tag_bonus
            importance = min(importance, 1.0)
            conv_scores.append((rank, importance))
        all_scores.append(conv_scores)

    print(f"\nPhase 3 complete: {total_surviving} messages scored" + " " * 30)

    # ── Phase 4: DB insert + embed all (no LLM) ──────────────────────
    session_ids: list[int] = []
    progress_count = stats.conversations_skipped

    for wi_idx, (conv, surviving) in enumerate(surviving_convs):
        conv_scores = all_scores[wi_idx]
        title_short = conv.title[:40]

        conv_ts = (
            datetime.fromtimestamp(conv.created_at, tz=UTC)
            if conv.created_at else None
        )
        session_id = await sqlite.create_session(title=conv.title, created_at=conv_ts)
        session_ids.append(session_id)

        memory_ids: list[int] = []
        for msg_idx, (msg, tags, summary) in enumerate(surviving):
            ts = (
                datetime.fromtimestamp(msg.timestamp, tz=UTC)
                if msg.timestamp else None
            )
            memory = Memory(
                session_id=session_id,
                role=msg.role,
                content=msg.content,
                summary=summary,
                timestamp=ts or datetime.now(timezone.utc),
                memory_type=MemoryType.CONVERSATION,
            )
            memory_id = await sqlite.create_memory(memory)
            memory_ids.append(memory_id)
            try:
                await sqlite.tag_memory(memory_id, tags)
            except Exception as e:
                logger.error("Tagging failed: %s", e)

            rank, importance = conv_scores[msg_idx]
            try:
                await sqlite.update_memory(memory_id, rank=rank, importance=importance)
            except Exception as e:
                logger.error("Score update failed: %s", e)

        # Batch embed
        try:
            texts = [summary for _, _, summary in surviving]
            metadatas = [
                {"session_id": str(session_id), "role": msg.role}
                for msg, _, _ in surviving
            ]
            chroma.add_memories_batch(memory_ids, texts, metadatas)
        except Exception as e:
            logger.error("ChromaDB batch embed failed: %s", e)

        msg_count = len(memory_ids)
        await sqlite.update_session(session_id, message_count=msg_count)
        stats.conversations_imported += 1
        stats.messages_processed += msg_count

        progress_count += 1
        if on_progress:
            on_progress(progress_count, total, conv.title, stats)
        print(f"  [{title_short}] Committed {msg_count} memories")

    print(f"Phase 4 complete: {stats.messages_processed} memories committed to DB")

    # ── Phase 5: Lessons (one model load) ─────────────────────────────
    if not skip_lessons:
        for wi_idx, (conv, surviving) in enumerate(surviving_convs):
            if len(conv.messages) >= 4:
                title_short = conv.title[:40]
                print(f"  Phase 5 — Lessons [{title_short}]...")
                try:
                    full_text = _build_conversation_text(conv)
                    await processor.process_lesson(full_text, session_ids[wi_idx])
                    stats.lessons_extracted += 1
                except Exception as e:
                    logger.error(
                        "Lesson extraction failed for '%s': %s", conv.title, e,
                    )
        print(f"Phase 5 complete: {stats.lessons_extracted} lessons extracted")

    print(f"\nGlobal batch complete: {stats.conversations_imported} conversations, "
          f"{stats.messages_processed} messages")


# ---------------------------------------------------------------------------
# Per-conversation pipeline (used for single imports or batch_mode=False)
# ---------------------------------------------------------------------------

async def _import_single_conversation(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    processor: MemoryProcessor,
    config: MemoryConfig | None,
    conv: ParsedConversation,
    stats: ImportStats,
    skip_lessons: bool,
):
    """Import a single conversation using batch-by-task-type pipeline.

    Instead of processing each message through the full 7-step pipeline
    (which swaps LLM models at every step), this batches all messages by
    task type so each model loads only once per conversation.

    Batch order (minimizes model swaps):
      1. Noise filter + tagging  (no LLM)
      2. Summarize all           (glm4)
      3. Rank all                (qwen2.5:14b)
      4. DB create + tag all     (no LLM)
      5. Embed all               (nomic-embed-text)
      6. Importance all + ranks  (qwen3:14b)
      7. Lessons                 (qwen3:14b — same model, no swap)
    """
    # Create a session for this conversation (preserve original date)
    conv_ts = (
        datetime.fromtimestamp(conv.created_at, tz=UTC)
        if conv.created_at else None
    )
    session_id = await sqlite.create_session(title=conv.title, created_at=conv_ts)

    recency_bonus = config.importance_recency_bonus if config else 0.1
    tag_bonus = config.importance_tag_bonus if config else 0.05

    title_short = conv.title[:40]
    total_msgs = len(conv.messages)

    # ── Step 1: Filter noise + tag (no LLM) ────────────────────────────
    valid_messages: list[tuple[ParsedMessage, list[str]]] = []
    for msg in conv.messages:
        if should_skip_memory(msg.content):
            stats.messages_skipped_noise += 1
            continue
        tags = tag_message(msg.content)
        valid_messages.append((msg, tags))
    noise_count = total_msgs - len(valid_messages)
    print(f"  [{title_short}] {len(valid_messages)}/{total_msgs} messages kept ({noise_count} noise)")

    # ── Step 2: Summarize all (one model load) ─────────────────────────
    # Messages where summary comes back as "SKIP" are filtered out.
    surviving: list[tuple[ParsedMessage, list[str], str]] = []
    for i, (msg, tags) in enumerate(valid_messages):
        print(f"  [{title_short}] Summarizing {i+1}/{len(valid_messages)}...", end="\r")
        try:
            sum_system, sum_prompt = summarize_memory(msg.content)
            summary = await router.generate(
                TaskType.SUMMARIZATION, sum_prompt, system=sum_system,
            )
            if summary.strip().upper() == "SKIP":
                logger.debug("Memory skipped (meta/self-referential): %s", msg.content[:50])
                stats.messages_skipped_noise += 1
                continue
        except Exception as e:
            logger.error("Summarization failed, using raw text: %s", e)
            summary = msg.content
        surviving.append((msg, tags, summary))
    print(f"  [{title_short}] Summarized {len(surviving)} messages" + " " * 20)

    # ── Step 3: Rank all (one model load) ──────────────────────────────
    ranks: list[int] = []
    for i, (msg, _tags, _summary) in enumerate(surviving):
        print(f"  [{title_short}] Ranking {i+1}/{len(surviving)}...", end="\r")
        try:
            rank_system, rank_prompt = rank_memory(msg.content)
            rank_text = await router.generate(
                TaskType.RANKING, rank_prompt, system=rank_system,
            )
            ranks.append(MemoryProcessor._parse_rank(rank_text))
        except Exception as e:
            logger.error("Ranking failed: %s", e)
            ranks.append(3)
    print(f"  [{title_short}] Ranked {len(ranks)} messages" + " " * 20)

    # ── Step 4: DB create + tag all (no LLM) ──────────────────────────
    memory_ids: list[int] = []
    for msg, tags, summary in surviving:
        ts = (
            datetime.fromtimestamp(msg.timestamp, tz=UTC)
            if msg.timestamp else None
        )
        memory = Memory(
            session_id=session_id,
            role=msg.role,
            content=msg.content,
            summary=summary,
            timestamp=ts or datetime.now(timezone.utc),
            memory_type=MemoryType.CONVERSATION,
        )
        memory_id = await sqlite.create_memory(memory)
        memory_ids.append(memory_id)
        try:
            await sqlite.tag_memory(memory_id, tags)
        except Exception as e:
            logger.error("Tagging failed: %s", e)
    print(f"  [{title_short}] Saved {len(memory_ids)} memories to DB")

    # ── Step 5: Embed all (single batch call) ──────────────────────────
    try:
        texts = [summary for _msg, _tags, summary in surviving]
        metadatas = [
            {"session_id": str(session_id), "role": msg.role}
            for msg, _tags, _summary in surviving
        ]
        chroma.add_memories_batch(memory_ids, texts, metadatas)
    except Exception as e:
        logger.error("ChromaDB batch embed failed: %s", e)
    print(f"  [{title_short}] Embedded {len(memory_ids)} memories")

    # ── Step 6: Importance all + update ranks (one model load) ─────────
    for i, memory_id in enumerate(memory_ids):
        print(f"  [{title_short}] Importance {i+1}/{len(memory_ids)}...", end="\r")
        _msg, tags, _summary = surviving[i]
        try:
            imp_system, imp_prompt = ask_importance(surviving[i][0].content)
            importance_text = await router.generate(
                TaskType.IMPORTANCE, imp_prompt, system=imp_system,
            )
            importance = MemoryProcessor._parse_float(importance_text, default=0.3)
        except Exception:
            importance = 0.3

        importance += recency_bonus
        if len(tags) > 6:
            importance += tag_bonus
        importance = min(importance, 1.0)

        try:
            await sqlite.update_memory(memory_id, rank=ranks[i], importance=importance)
        except Exception as e:
            logger.error("Rank/importance update failed: %s", e)
    print(f"  [{title_short}] Scored importance for {len(memory_ids)} memories" + " " * 20)

    # Update session message count
    msg_count = len(memory_ids)
    stats.messages_processed += msg_count
    await sqlite.update_session(session_id, message_count=msg_count)

    # ── Step 7: Lessons (same model as importance — no swap) ───────────
    if not skip_lessons and len(conv.messages) >= 4:
        print(f"  [{title_short}] Extracting lessons...")
        try:
            full_text = _build_conversation_text(conv)
            await processor.process_lesson(full_text, session_id)
            stats.lessons_extracted += 1
        except Exception as e:
            logger.error("Lesson extraction failed for '%s': %s", conv.title, e)

    print(f"  [{title_short}] Done ({msg_count} messages processed)")


def _build_conversation_text(conv: ParsedConversation) -> str:
    """Build a plain-text representation of a conversation for lesson extraction."""
    lines = []
    for msg in conv.messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n\n".join(lines)
