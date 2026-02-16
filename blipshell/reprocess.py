"""Reprocess imported memories and lessons with a better model.

Re-runs the LLM summarization/ranking/importance pipeline on existing data
that was originally processed with a weak model (e.g. gemma3:4b).
"""

import asyncio
import logging
from typing import Callable, Optional

from blipshell.llm.prompts import (
    ask_importance,
    extract_lesson,
    rank_memory,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import tag_message
from blipshell.models.memory import Lesson

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int], None]]


async def reprocess_memories(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    batch_size: int = 50,
    skip_embed: bool = False,
    no_think: bool = False,
    on_progress: ProgressCallback = None,
) -> dict:
    """Re-summarize, re-rank, re-score, and optionally re-embed all memories.

    Returns stats dict with processed/skipped/errors/total counts.
    """
    stats = {"processed": 0, "skipped": 0, "errors": 0, "total": 0}
    think = False if no_think else None  # None = let model decide, False = disable

    # Get total count
    _, total = await sqlite.get_memories_paginated(page=1, limit=1, include_archived=True)
    stats["total"] = total

    if on_progress:
        on_progress(0, total)

    page = 1
    while True:
        memories, _ = await sqlite.get_memories_paginated(
            page=page, limit=batch_size, include_archived=True,
        )
        if not memories:
            break

        for memory in memories:
            if not memory.content or not memory.content.strip():
                stats["skipped"] += 1
                if on_progress:
                    done = stats["processed"] + stats["skipped"] + stats["errors"]
                    on_progress(done, total)
                continue

            try:
                # Re-summarize
                summary = await router.generate(
                    TaskType.SUMMARIZATION,
                    summarize_memory(memory.content),
                    think=think,
                )
                if summary.strip().upper() == "SKIP":
                    summary = memory.summary or memory.content[:200]

                # Re-rank (1-5)
                rank_text = await router.generate(
                    TaskType.RANKING,
                    rank_memory(memory.content),
                    think=think,
                )
                rank = MemoryProcessor._parse_rank(rank_text)

                # Re-importance (0.0-1.0)
                importance_text = await router.generate(
                    TaskType.RANKING,
                    ask_importance(memory.content),
                    think=think,
                )
                importance = MemoryProcessor._parse_float(importance_text, default=0.3)

                # Tag bonus
                tag_count = await sqlite.get_tag_count_for_memory(memory.id)
                if tag_count > 6:
                    importance += 0.05
                importance = min(importance, 1.0)

                # Update SQLite
                await sqlite.update_memory(
                    memory.id,
                    summary=summary,
                    rank=rank,
                    importance=importance,
                )

                # Re-embed in ChromaDB (upsert handles updates)
                if not skip_embed:
                    chroma.add_memory(memory.id, summary, {
                        "session_id": str(memory.session_id or ""),
                        "role": memory.role,
                    })

                stats["processed"] += 1

            except Exception as e:
                logger.error("Error reprocessing memory %d: %s", memory.id, e)
                stats["errors"] += 1

            if on_progress:
                done = stats["processed"] + stats["skipped"] + stats["errors"]
                on_progress(done, total)

            # Rate limit to avoid overwhelming Ollama
            await asyncio.sleep(0.1)

        page += 1

    return stats


async def reprocess_lessons(
    sqlite: SQLiteStore,
    chroma: ChromaStore,
    router: LLMRouter,
    min_messages: int = 4,
    no_think: bool = False,
    on_progress: ProgressCallback = None,
) -> dict:
    """Delete all existing lessons and re-extract from conversation memories.

    Groups memories by session_id to reconstruct conversations, then uses
    the reasoning model to extract actionable lessons from each session.

    Returns stats dict with sessions_processed/lessons_extracted/errors/old_lessons_deleted.
    """
    stats = {
        "sessions_processed": 0,
        "lessons_extracted": 0,
        "errors": 0,
        "old_lessons_deleted": 0,
    }
    think = False if no_think else None

    # Delete all existing lessons from SQLite and ChromaDB
    old_lessons = await sqlite.get_all_lessons()
    stats["old_lessons_deleted"] = len(old_lessons)

    for lesson in old_lessons:
        try:
            chroma.delete_lesson(lesson.id)
        except Exception:
            pass
        await sqlite.delete_lesson(lesson.id)

    # Get distinct session IDs from memories
    session_ids = await sqlite.get_distinct_memory_session_ids()
    total = len(session_ids)

    if on_progress:
        on_progress(0, total)

    for idx, session_id in enumerate(session_ids):
        try:
            memories = await sqlite.get_memories_by_session(session_id)

            if len(memories) < min_messages:
                if on_progress:
                    on_progress(idx + 1, total)
                continue

            # Build conversation text from memories
            conv_parts = []
            for m in memories:
                role_label = "User" if m.role == "user" else "Assistant"
                text = m.content[:500] if m.content else ""
                if text:
                    conv_parts.append(f"{role_label}: {text}")

            conversation_text = "\n".join(conv_parts)

            # Truncate to avoid exceeding model context
            if len(conversation_text) > 4000:
                conversation_text = conversation_text[:4000]

            # Extract lessons via reasoning model
            lesson_system, lesson_prompt = extract_lesson(conversation_text)
            lesson_text = await router.generate(
                TaskType.REASONING,
                lesson_prompt,
                system=lesson_system,
                think=think,
            )

            # Parse individual lessons (one per line)
            lines = [
                line.strip().lstrip("- ").lstrip("* ").lstrip("• ").strip()
                for line in lesson_text.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            for line in lines:
                # Skip too-short lines (likely noise or numbering artifacts)
                if len(line) < 10:
                    continue

                # Strip leading numbers like "1. " or "1) "
                stripped = line
                if len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)" :
                    stripped = stripped[2:].strip()
                elif len(stripped) > 3 and stripped[:2].isdigit() and stripped[2] in ".)":
                    stripped = stripped[3:].strip()

                if len(stripped) < 10:
                    continue

                lesson = Lesson(
                    content=stripped,
                    summary=stripped,
                    rank=3,
                    importance=0.5,
                    source_session_id=session_id,
                )
                lesson_id = await sqlite.create_lesson(lesson)

                try:
                    chroma.add_lesson(lesson_id, stripped)
                except Exception as e:
                    logger.debug("Lesson embed failed: %s", e)

                try:
                    tags = tag_message(stripped)
                    await sqlite.tag_lesson(lesson_id, tags)
                except Exception:
                    pass

                stats["lessons_extracted"] += 1

            stats["sessions_processed"] += 1

        except Exception as e:
            logger.error("Error processing session %d: %s", session_id, e)
            stats["errors"] += 1

        if on_progress:
            on_progress(idx + 1, total)

        # Rate limit
        await asyncio.sleep(0.1)

    return stats
