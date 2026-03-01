"""Backfill session summaries for imported sessions.

The import pipeline processed individual messages (summarize, rank, embed)
but never generated session-level summaries. This script retroactively
generates summaries for all sessions that have memories but no summary.

Each session gets:
1. A summary (chunked summarization of memory summaries, same as end_session)
2. A title (generated from the summary)

Resume support: skips sessions that already have summaries.

Usage:
    python scripts/backfill_session_summaries.py [--limit N] [--dry-run]
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import (
    generate_session_title,
    summarize_session_conversation,
    summarize_session_summaries,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import get_ollama_url

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SUMMARY_CHUNK_SIZE = 20


async def summarize_session(
    sqlite: SQLiteStore, router: LLMRouter, session_id: int,
) -> tuple[str, str]:
    """Generate summary and title for a session from its memories.

    Returns (summary, title).
    """
    memories = await sqlite.get_memories_by_session(session_id)
    if not memories:
        return "", ""

    texts = [m.summary or m.content for m in memories]

    if len(texts) > SUMMARY_CHUNK_SIZE:
        chunk_summaries = []
        for i in range(0, len(texts), SUMMARY_CHUNK_SIZE):
            chunk = texts[i:i + SUMMARY_CHUNK_SIZE]
            chunk_text = "\n".join(chunk)
            try:
                chunk_summary = await router.generate(
                    TaskType.SUMMARIZATION,
                    summarize_session_summaries(chunk_text),
                )
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                logger.error("Chunk summarization failed: %s", e)
                chunk_summaries.append(chunk_text[:200])

        try:
            summary = await router.generate(
                TaskType.SUMMARIZATION,
                summarize_session_summaries("\n".join(chunk_summaries)),
            )
        except Exception as e:
            logger.error("Meta-summarization failed: %s", e)
            summary = "\n".join(chunk_summaries)
    else:
        all_text = "\n".join(texts)
        try:
            summary = await router.generate(
                TaskType.SUMMARIZATION,
                summarize_session_conversation(all_text),
            )
        except Exception as e:
            logger.error("Session summarization failed: %s", e)
            summary = all_text[:500]

    try:
        title = await router.generate(
            TaskType.SUMMARIZATION,
            generate_session_title(summary),
        )
    except Exception as e:
        logger.error("Title generation failed: %s", e)
        title = f"Session {session_id}"

    return summary.strip(), title.strip()


async def backfill(limit: int = 0, dry_run: bool = False):
    """Backfill session summaries for all sessions missing them."""
    config_mgr = ConfigManager()
    config = config_mgr.config
    db_path = config_mgr.get_db_path()
    ollama_url = get_ollama_url()

    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()

    endpoint_mgr = EndpointManager(config)
    router = LLMRouter(ollama_url, endpoint_mgr)

    # Find sessions needing summaries
    query_limit = limit if limit > 0 else 10000
    sessions = await sqlite.get_sessions_without_summaries(limit=query_limit)
    print(f"Found {len(sessions)} sessions without summaries")

    if dry_run:
        for s in sessions[:20]:
            print(f"  Session {s['id']}: {s['title']} ({s['message_count']} messages)")
        if len(sessions) > 20:
            print(f"  ... and {len(sessions) - 20} more")
        await sqlite.close()
        return

    processed = 0
    failed = 0
    start_time = time.monotonic()

    for i, session in enumerate(sessions):
        sid = session["id"]
        title = session["title"] or "Untitled"
        msg_count = session["message_count"]
        prefix = f"[{i+1}/{len(sessions)}]"

        try:
            call_start = time.monotonic()
            summary, new_title = await summarize_session(sqlite, router, sid)
            elapsed = time.monotonic() - call_start

            if summary:
                await sqlite.update_session(sid, summary=summary, title=new_title)
                processed += 1
                print(f"{prefix} {title[:50]} ({msg_count} msgs) -> OK ({elapsed:.1f}s)")
            else:
                print(f"{prefix} {title[:50]} -> no memories, skipped")
        except Exception as e:
            failed += 1
            print(f"{prefix} {title[:50]} -> FAILED: {e}")

        # Progress update every 50
        if (i + 1) % 50 == 0:
            elapsed_total = time.monotonic() - start_time
            rate = (i + 1) / elapsed_total * 60
            remaining = (len(sessions) - i - 1) / rate if rate > 0 else 0
            print(f"\n--- Progress: {i+1}/{len(sessions)} "
                  f"({processed} OK, {failed} failed, "
                  f"{rate:.0f}/min, ~{remaining:.0f} min remaining) ---\n")

    elapsed_total = time.monotonic() - start_time
    print(f"\nDone: {processed} summarized, {failed} failed, "
          f"{len(sessions) - processed - failed} skipped "
          f"({elapsed_total:.0f}s total)")

    await sqlite.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill session summaries for imported sessions",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max sessions to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show sessions that need backfill without processing")
    args = parser.parse_args()
    asyncio.run(backfill(limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
