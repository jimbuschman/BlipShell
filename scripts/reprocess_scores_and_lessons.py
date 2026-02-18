"""Reprocess bad scores and lessons from the import.

Fixes two issues from the initial batch import:
1. Scores: Sessions > 650 got default rank=3/importance=0.3 due to scoring
   failures. Reprocesses all memories in those sessions using the combined
   rank_and_importance prompt with qwen2.5:14b.
2. Lessons: ~275 lessons are raw conversation dumps instead of behavioral
   insights. Deletes bad lessons and regenerates them.

Usage:
    python scripts/reprocess_scores_and_lessons.py [--scores-only] [--lessons-only] [--dry-run]
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.config import ConfigManager
from blipshell.llm.prompts import extract_lesson, rank_and_importance
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.chroma_store import ChromaStore
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import get_ollama_url

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Threshold: sessions at or above this ID have bad scores
BAD_SCORE_SESSION_THRESHOLD = 650


async def reprocess_scores(sqlite: SQLiteStore, router: LLMRouter, config, dry_run: bool = False):
    """Reprocess rank+importance for all memories in sessions >= threshold."""
    recency_bonus = config.memory.importance_recency_bonus
    tag_bonus = config.memory.importance_tag_bonus

    # Get all memories that need rescoring
    cursor = await sqlite._db.execute(
        """SELECT m.id, m.content, m.session_id
           FROM memories m
           WHERE m.session_id >= ?
           ORDER BY m.session_id, m.id""",
        (BAD_SCORE_SESSION_THRESHOLD,),
    )
    rows = await cursor.fetchall()
    total = len(rows)
    print(f"\nScores: {total} memories to reprocess (sessions >= {BAD_SCORE_SESSION_THRESHOLD})")

    if dry_run:
        print("  (dry run — no changes)")
        return

    # Get tags for importance bonus calculation
    tag_counts = {}
    cursor = await sqlite._db.execute(
        """SELECT mt.memory_id, COUNT(*) FROM memory_tags mt
           JOIN memories m ON m.id = mt.memory_id
           WHERE m.session_id >= ?
           GROUP BY mt.memory_id""",
        (BAD_SCORE_SESSION_THRESHOLD,),
    )
    for row in await cursor.fetchall():
        tag_counts[row[0]] = row[1]

    updated = 0
    failed = 0
    current_session = None

    for i, (mem_id, content, session_id) in enumerate(rows):
        if session_id != current_session:
            current_session = session_id
            # Get session title for display
            cursor = await sqlite._db.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
            title_row = await cursor.fetchone()
            title = (title_row[0] if title_row else "?")[:40]

        call_start = time.monotonic()
        print(f"  [{i+1}/{total}] Scoring mem {mem_id} [{title}]...", flush=True)

        try:
            ri_system, ri_prompt = rank_and_importance(content)
            ri_text = await router.generate(
                TaskType.RANKING_IMPORTANCE, ri_prompt, system=ri_system,
            )
            rank, importance = MemoryProcessor._parse_rank_and_importance(ri_text)
            elapsed = time.monotonic() - call_start

            # Apply bonuses
            importance += recency_bonus
            num_tags = tag_counts.get(mem_id, 0)
            if num_tags > 6:
                importance += tag_bonus
            importance = min(importance, 1.0)

            await sqlite.update_memory(mem_id, rank=rank, importance=importance)
            updated += 1
            print(f"    -> rank={rank} imp={importance:.2f} ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            failed += 1
            elapsed = time.monotonic() - call_start
            print(f"    -> FAILED ({elapsed:.1f}s): {e}", flush=True)

    print(f"\nScores complete: {updated} updated, {failed} failed")


async def reprocess_lessons(
    sqlite: SQLiteStore, chroma: ChromaStore, router: LLMRouter,
    processor: MemoryProcessor, dry_run: bool = False,
):
    """Delete bad lessons and regenerate them."""
    # Find bad lessons (conversation dumps)
    cursor = await sqlite._db.execute(
        """SELECT id, source_session_id FROM lessons
           WHERE content LIKE 'User:%' AND content LIKE '%Assistant:%'"""
    )
    bad_lessons = await cursor.fetchall()
    print(f"\nLessons: {len(bad_lessons)} bad lessons to regenerate")

    if dry_run:
        print("  (dry run — no changes)")
        return

    # Delete bad lessons
    deleted = 0
    for lesson_id, session_id in bad_lessons:
        try:
            await sqlite.delete_lesson(lesson_id)
            try:
                chroma.delete_lesson(lesson_id)
            except Exception:
                pass
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete lesson {lesson_id}: {e}", flush=True)

    print(f"  Deleted {deleted} bad lessons", flush=True)

    # Get unique session IDs that need new lessons
    session_ids = sorted(set(sid for _, sid in bad_lessons))

    # Regenerate lessons
    generated = 0
    failed = 0
    for i, session_id in enumerate(session_ids):
        # Get session info
        cursor = await sqlite._db.execute(
            "SELECT title, message_count FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row or (row[1] or 0) < 4:
            continue
        title = row[0][:40]

        # Build conversation text from memories
        cursor = await sqlite._db.execute(
            "SELECT role, content FROM memories WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        messages = await cursor.fetchall()
        if len(messages) < 4:
            continue

        lines = []
        for role, content in messages:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        full_text = "\n\n".join(lines)

        call_start = time.monotonic()
        print(f"  [{i+1}/{len(session_ids)}] Lesson for [{title}]...", flush=True)
        try:
            await processor.process_lesson(full_text, session_id)
            generated += 1
            elapsed = time.monotonic() - call_start
            print(f"    -> OK ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            failed += 1
            elapsed = time.monotonic() - call_start
            print(f"    -> FAILED ({elapsed:.1f}s): {e}", flush=True)

    print(f"\nLessons complete: {generated} generated, {failed} failed")


async def main():
    parser = argparse.ArgumentParser(description="Reprocess bad scores and lessons")
    parser.add_argument("--scores-only", action="store_true", help="Only reprocess scores")
    parser.add_argument("--lessons-only", action="store_true", help="Only reprocess lessons")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without changes")
    args = parser.parse_args()

    do_scores = not args.lessons_only
    do_lessons = not args.scores_only

    # Load config
    config_mgr = ConfigManager()
    config = config_mgr.load()

    # Initialize stores
    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()

    ollama_url = get_ollama_url(config.endpoints)
    chroma = ChromaStore(
        persist_directory=config.database.chroma_path,
        ollama_url=ollama_url,
        embedding_model=config.models.embedding,
    )

    router = LLMRouter(config.models, config.endpoints, llm_config=config.llm)
    processor = MemoryProcessor(sqlite, chroma, router, config=config.memory)

    print(f"Config loaded. Scoring model: {router.get_model(TaskType.RANKING_IMPORTANCE)}")
    print(f"Reasoning model (lessons): {router.get_model(TaskType.REASONING)}")

    if do_scores:
        await reprocess_scores(sqlite, router, config, dry_run=args.dry_run)

    if do_lessons:
        await reprocess_lessons(sqlite, chroma, router, processor, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
