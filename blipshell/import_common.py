"""Shared import pipeline — format-agnostic conversation import.

Provides the dataclasses and pipeline functions used by all format-specific
importers (ChatGPT, Claude, DeepSeek). Each importer only needs to produce
a list[ParsedConversation] and then call import_conversations().
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from typing import Callable, Optional

from blipshell.llm.prompts import (
    ask_importance,
    generate_session_title,
    rank_and_importance,
    rank_memory,
    summarize_memory,
    summarize_session_conversation,
    summarize_session_summaries,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.vector_store import VectorStore
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
    vectors: VectorStore,
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
        vectors: Vector store instance
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
    processor = MemoryProcessor(sqlite, vectors, router, config=config)
    stats = ImportStats()

    if batch_mode and len(conversations) > 1:
        await _import_global_batch(
            sqlite, vectors, router, processor, config,
            conversations, stats, skip_lessons, on_progress,
            max_concurrent=max_concurrent,
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
                            vectors.delete_memory(mid)
                        except Exception as e:
                            logger.warning("Failed to delete memory vector %d: %s", mid, e)
                    cursor = await sqlite._db.execute(
                        "SELECT id FROM lessons WHERE source_session_id = ?",
                        (session_id,),
                    )
                    old_lesson_ids = [row[0] for row in await cursor.fetchall()]
                    for lid in old_lesson_ids:
                        try:
                            vectors.delete_lesson(lid)
                        except Exception as e:
                            logger.warning("Failed to delete lesson vector %d: %s", lid, e)
                    await sqlite.delete_session_cascade(session_id, old_ids)
                    stats.conversations_reimported += 1
                except Exception as e:
                    logger.error("Failed to clean up incomplete '%s': %s", conv.title, e)
                    return

        async with semaphore:
            try:
                await _import_single_conversation(
                    sqlite, vectors, router, processor, config, conv, stats,
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
    sqlite: SQLiteStore, vectors: VectorStore, session_id: int,
):
    """Clean up an incomplete session (0 messages saved) for re-import."""
    cursor = await sqlite._db.execute(
        "SELECT id FROM memories WHERE session_id = ?",
        (session_id,),
    )
    old_ids = [row[0] for row in await cursor.fetchall()]
    for mid in old_ids:
        try:
            vectors.delete_memory(mid)
        except Exception as e:
            logger.warning("Failed to delete memory vector %d: %s", mid, e)
    cursor = await sqlite._db.execute(
        "SELECT id FROM lessons WHERE source_session_id = ?",
        (session_id,),
    )
    old_lesson_ids = [row[0] for row in await cursor.fetchall()]
    for lid in old_lesson_ids:
        try:
            vectors.delete_lesson(lid)
        except Exception as e:
            logger.warning("Failed to delete lesson vector %d: %s", lid, e)
    await sqlite.delete_session_cascade(session_id, old_ids)


async def _import_global_batch(
    sqlite: SQLiteStore,
    vectors: VectorStore,
    router: LLMRouter,
    processor: MemoryProcessor,
    config: MemoryConfig | None,
    conversations: list[ParsedConversation],
    stats: ImportStats,
    skip_lessons: bool,
    on_progress: Optional[Callable] = None,
    max_concurrent: int = 3,
):
    """Global batch pipeline — processes each conversation fully before the next.

    Each conversation goes through all phases (filter, summarize, score,
    DB save, lessons) and is committed to DB before moving on. This means
    a crash only loses work on the current conversation — all prior ones
    are safely persisted and will be skipped on resume.

    Uses the combined rank_and_importance prompt (1 LLM call per message
    for scoring) which is more efficient than the separate rank + importance
    calls used by _import_single_conversation.

    Per-conversation steps:
      1. Filter + tag (no LLM)
      2. Summarize all messages (cloud LLM)
      3. Rank + importance all messages (local LLM, combined prompt)
      4. DB insert + tag + score + embed
      5. Lessons (if enabled)
    """
    total = len(conversations)
    recency_bonus = config.importance_recency_bonus if config else 0.1
    tag_bonus = config.importance_tag_bonus if config else 0.05

    # ── Phase 0: Resume check ─────────────────────────────────────────
    # Scan all conversations upfront to build the work list and skip count.
    work_convs: list[ParsedConversation] = []

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
                    await _cleanup_incomplete_session(sqlite, vectors, session_id)
                    stats.conversations_reimported += 1
                except Exception as e:
                    logger.error("Failed to clean up incomplete '%s': %s", conv.title, e)
                    continue

        work_convs.append(conv)

    if not work_convs:
        return

    print(f"\nGlobal batch: {len(work_convs)} conversations to process "
          f"({stats.conversations_skipped} skipped, {max_concurrent} concurrent)")

    # ── Per-conversation worker ───────────────────────────────────────
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_conversation(conv_idx: int, conv: ParsedConversation):
        async with semaphore:
            title_short = conv.title[:40]
            prefix = f"  [{conv_idx+1}/{len(work_convs)}] [{title_short}]"
            total_msgs = len(conv.messages)
            print(f"{prefix} Starting ({total_msgs} messages)", flush=True)

            # ── Step 1: Filter + tag (no LLM) ────────────────────────
            valid_msgs: list[tuple[ParsedMessage, list[str]]] = []
            for msg in conv.messages:
                if should_skip_memory(msg.content):
                    stats.messages_skipped_noise += 1
                    continue
                tags = tag_message(msg.content)
                valid_msgs.append((msg, tags))
            print(f"{prefix} Filtered: {len(valid_msgs)}/{total_msgs} kept", flush=True)

            # ── Step 2: Summarize (cloud LLM) ────────────────────────
            surviving: list[tuple[ParsedMessage, list[str], str]] = []
            step_start = time.monotonic()
            for i, (msg, tags) in enumerate(valid_msgs):
                call_start = time.monotonic()
                print(f"{prefix} Summarizing {i+1}/{len(valid_msgs)}...", flush=True)
                try:
                    sum_system, sum_prompt = summarize_memory(msg.content)
                    summary = await router.generate(
                        TaskType.SUMMARIZATION, sum_prompt, system=sum_system,
                    )
                    elapsed = time.monotonic() - call_start
                    if summary.strip().upper() == "SKIP":
                        print(f"{prefix}   -> SKIP ({elapsed:.1f}s)", flush=True)
                        stats.messages_skipped_noise += 1
                        continue
                    print(f"{prefix}   -> OK ({elapsed:.1f}s)", flush=True)
                except Exception as e:
                    elapsed = time.monotonic() - call_start
                    print(f"{prefix}   -> FAILED ({elapsed:.1f}s): {e}", flush=True)
                    logger.error("Summarization failed, using raw text: %s", e)
                    summary = msg.content
                surviving.append((msg, tags, summary))
            step_elapsed = time.monotonic() - step_start
            print(f"{prefix} Summarized: {len(surviving)} messages ({step_elapsed:.1f}s total)", flush=True)

            # ── Step 3: Rank + Importance (local LLM, combined prompt)
            scores: list[tuple[int, float]] = []
            step_start = time.monotonic()
            for i, (msg, tags, _summary) in enumerate(surviving):
                call_start = time.monotonic()
                print(f"{prefix} Scoring {i+1}/{len(surviving)}...", flush=True)
                try:
                    ri_system, ri_prompt = rank_and_importance(msg.content)
                    ri_text = await router.generate(
                        TaskType.RANKING_IMPORTANCE, ri_prompt, system=ri_system,
                    )
                    rank, importance = MemoryProcessor._parse_rank_and_importance(ri_text)
                    elapsed = time.monotonic() - call_start
                    print(f"{prefix}   -> rank={rank} imp={importance:.2f} ({elapsed:.1f}s)", flush=True)
                except Exception as e:
                    elapsed = time.monotonic() - call_start
                    print(f"{prefix}   -> FAILED ({elapsed:.1f}s): {e}", flush=True)
                    logger.error("Rank+importance failed: %s", e)
                    rank, importance = 3, 0.3

                importance += recency_bonus
                if len(tags) > 6:
                    importance += tag_bonus
                importance = min(importance, 1.0)
                scores.append((rank, importance))
            step_elapsed = time.monotonic() - step_start
            print(f"{prefix} Scored: {len(scores)} messages ({step_elapsed:.1f}s total)", flush=True)

            # ── Step 4: DB insert + tag + score + embed ──────────────
            print(f"{prefix} Saving to DB...", flush=True)
            conv_ts = (
                datetime.fromtimestamp(conv.created_at, tz=UTC)
                if conv.created_at else None
            )
            session_id = await sqlite.create_session(title=conv.title, created_at=conv_ts)

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

                rank, importance = scores[msg_idx]
                try:
                    await sqlite.update_memory(memory_id, rank=rank, importance=importance)
                except Exception as e:
                    logger.error("Score update failed: %s", e)

            # Batch embed
            if surviving:
                print(f"{prefix} Embedding...", flush=True)
                try:
                    texts = [summary for _, _, summary in surviving]
                    metadatas = [
                        {"session_id": str(session_id), "role": msg.role}
                        for msg, _, _ in surviving
                    ]
                    vectors.add_memories_batch(memory_ids, texts, metadatas)
                except Exception as e:
                    logger.error("Vector batch embed failed: %s", e)

            msg_count = len(memory_ids)
            await sqlite.update_session(session_id, message_count=msg_count)
            stats.conversations_imported += 1
            stats.messages_processed += msg_count

            # ── Step 5: Lessons ──────────────────────────────────────
            if not skip_lessons and len(conv.messages) >= 4:
                call_start = time.monotonic()
                print(f"{prefix} Extracting lessons...", flush=True)
                try:
                    full_text = _build_conversation_text(conv)
                    await processor.process_lesson(full_text, session_id)
                    stats.lessons_extracted += 1
                    elapsed = time.monotonic() - call_start
                    print(f"{prefix}   -> lesson OK ({elapsed:.1f}s)", flush=True)
                except Exception as e:
                    elapsed = time.monotonic() - call_start
                    print(f"{prefix}   -> lesson FAILED ({elapsed:.1f}s): {e}", flush=True)
                    logger.error(
                        "Lesson extraction failed for '%s': %s", conv.title, e,
                    )

            # ── Step 6: Session summary ───────────────────────────────
            if surviving:
                call_start = time.monotonic()
                print(f"{prefix} Generating session summary...", flush=True)
                try:
                    texts = [summary for _, _, summary in surviving]
                    if len(texts) > 20:
                        chunk_summaries = []
                        for ci in range(0, len(texts), 20):
                            chunk_text = "\n".join(texts[ci:ci + 20])
                            cs = await router.generate(
                                TaskType.SUMMARIZATION,
                                summarize_session_summaries(chunk_text),
                            )
                            chunk_summaries.append(cs)
                        sess_summary = await router.generate(
                            TaskType.SUMMARIZATION,
                            summarize_session_summaries("\n".join(chunk_summaries)),
                        )
                    else:
                        sess_summary = await router.generate(
                            TaskType.SUMMARIZATION,
                            summarize_session_conversation("\n".join(texts)),
                        )
                    sess_title = await router.generate(
                        TaskType.SUMMARIZATION,
                        generate_session_title(sess_summary),
                    )
                    await sqlite.update_session(
                        session_id,
                        summary=sess_summary.strip(),
                        title=sess_title.strip(),
                    )
                    elapsed = time.monotonic() - call_start
                    print(f"{prefix}   -> summary OK ({elapsed:.1f}s)", flush=True)
                except Exception as e:
                    elapsed = time.monotonic() - call_start
                    print(f"{prefix}   -> summary FAILED ({elapsed:.1f}s): {e}", flush=True)
                    logger.error(
                        "Session summary failed for '%s': %s", conv.title, e,
                    )

            # ── Progress ─────────────────────────────────────────────
            progress_count = stats.conversations_skipped + stats.conversations_imported
            if on_progress:
                on_progress(progress_count, total, conv.title, stats)
            print(f"{prefix} Done — {msg_count} memories committed", flush=True)

    # Launch all and let semaphore control concurrency
    tasks = [
        asyncio.create_task(_process_conversation(i, conv))
        for i, conv in enumerate(work_convs)
    ]
    await asyncio.gather(*tasks)

    print(f"\nGlobal batch complete: {stats.conversations_imported} conversations, "
          f"{stats.messages_processed} messages")


# ---------------------------------------------------------------------------
# Per-conversation pipeline (used for single imports or batch_mode=False)
# ---------------------------------------------------------------------------

async def _import_single_conversation(
    sqlite: SQLiteStore,
    vectors: VectorStore,
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
        vectors.add_memories_batch(memory_ids, texts, metadatas)
    except Exception as e:
        logger.error("Vector batch embed failed: %s", e)
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

    # ── Step 8: Session summary ─────────────────────────────────────────
    if surviving:
        print(f"  [{title_short}] Generating session summary...")
        try:
            texts = [summary for _msg, _tags, summary in surviving]
            if len(texts) > 20:
                chunk_summaries = []
                for ci in range(0, len(texts), 20):
                    chunk_text = "\n".join(texts[ci:ci + 20])
                    cs = await router.generate(
                        TaskType.SUMMARIZATION,
                        summarize_session_summaries(chunk_text),
                    )
                    chunk_summaries.append(cs)
                sess_summary = await router.generate(
                    TaskType.SUMMARIZATION,
                    summarize_session_summaries("\n".join(chunk_summaries)),
                )
            else:
                sess_summary = await router.generate(
                    TaskType.SUMMARIZATION,
                    summarize_session_conversation("\n".join(texts)),
                )
            sess_title = await router.generate(
                TaskType.SUMMARIZATION,
                generate_session_title(sess_summary),
            )
            await sqlite.update_session(
                session_id,
                summary=sess_summary.strip(),
                title=sess_title.strip(),
            )
        except Exception as e:
            logger.error("Session summary failed for '%s': %s", conv.title, e)

    print(f"  [{title_short}] Done ({msg_count} messages processed)")


def _build_conversation_text(conv: ParsedConversation) -> str:
    """Build a plain-text representation of a conversation for lesson extraction."""
    lines = []
    for msg in conv.messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    return "\n\n".join(lines)
