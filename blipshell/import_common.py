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

from blipshell.llm.prompts import ask_importance, rank_memory, summarize_memory
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
) -> ImportStats:
    """Import parsed conversations through the BlipShell memory pipeline.

    For each conversation:
    1. Create a BlipShell session
    2. Process each message through the memory pipeline (noise, summarize, embed, tag, rank)
    3. Optionally extract lessons from the full conversation

    Args:
        sqlite: SQLite store instance
        chroma: ChromaDB store instance
        router: LLM router for summarization/ranking
        config: Memory configuration
        conversations: Parsed conversations to import
        on_progress: Callback(current_index, total, title, stats) for progress tracking
        skip_lessons: If True, skip lesson extraction (faster)

    Returns:
        ImportStats with counts of imported items
    """
    processor = MemoryProcessor(sqlite, chroma, router, config=config)
    stats = ImportStats()
    total = len(conversations)

    for i, conv in enumerate(conversations):
        if on_progress:
            on_progress(i, total, conv.title, stats)

        # Resume support: skip complete conversations, re-import incomplete ones
        session_id, db_count = await sqlite.get_session_message_count(conv.title)
        if session_id is not None:
            # message_count is only set after all steps complete, so >0
            # means the conversation was fully processed. If 0, it was
            # interrupted mid-import and needs to be redone.
            if db_count > 0:
                logger.info("Skipping already imported (%d msgs): %s", db_count, conv.title)
                stats.conversations_skipped += 1
                continue
            else:
                # Incomplete — session exists but message_count=0 (interrupted)
                logger.info(
                    "Re-importing incomplete '%s' (interrupted, 0 messages saved)",
                    conv.title,
                )
                try:
                    # Clean up ChromaDB entries for old memories
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
                    # Clean up lessons in ChromaDB
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
                    continue

        try:
            await _import_single_conversation(
                sqlite, chroma, router, processor, config, conv, stats,
                skip_lessons,
            )
            stats.conversations_imported += 1
        except Exception as e:
            logger.error("Failed to import conversation '%s': %s", conv.title, e)

        # Small delay between conversations to avoid overwhelming the LLM endpoint
        if i < total - 1:
            await asyncio.sleep(0.5)

    return stats


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
