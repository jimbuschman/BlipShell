"""Backfill session reflections for existing sessions.

Generates holistic session reflections for sessions that have summaries
and messages but no reflection yet. Each session gets a structured review
of effectiveness, what worked/didn't, technical insights, and process insights.

Resume-safe: UNIQUE constraint on session_id prevents duplicates.

Usage:
    python scripts/backfill_session_reflections.py [--limit N] [--dry-run] [--project NAME]
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
from blipshell.llm.router import LLMRouter
from blipshell.memory.vector_store import VectorStore
from blipshell.memory.processor import MemoryProcessor
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import MemoryConfig, get_ollama_url

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def backfill(
    limit: int = 0,
    dry_run: bool = False,
    project: str | None = None,
):
    """Backfill session reflections for unreflected sessions."""
    config_mgr = ConfigManager()
    config = config_mgr.load()

    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()

    vectors = VectorStore(
        db_path=config.database.path,
        embedding_model=config.models.embedding,
        ollama_url=get_ollama_url(config.endpoints),
        embedding_dim=config.database.embedding_dimensions,
    )
    vectors.initialize()

    endpoint_mgr = EndpointManager(config.endpoints, llm_config=config.llm)
    router = LLMRouter(config.models, endpoint_mgr)
    processor = MemoryProcessor(sqlite, vectors, router, config=config.memory)

    # Find sessions needing reflections
    query_limit = limit if limit > 0 else 10000
    sessions = await sqlite.get_sessions_missing_reflections(limit=query_limit)

    # Filter by project if specified
    if project:
        sessions = [s for s in sessions if s.get("project") == project]

    print(f"Found {len(sessions)} sessions needing reflections")

    if dry_run:
        for s in sessions[:30]:
            title = s.get("title") or "Untitled"
            proj = s.get("project") or "-"
            print(f"  Session {s['id']}: {title[:60]} (project={proj}, {s['msg_count']} msgs)")
        if len(sessions) > 30:
            print(f"  ... and {len(sessions) - 30} more")
        await sqlite.close()
        return

    processed = 0
    skipped = 0
    failed = 0
    start_time = time.monotonic()

    for i, session in enumerate(sessions):
        sid = session["id"]
        title = session.get("title") or "Untitled"
        summary = session["summary"]
        proj = session.get("project")
        prefix = f"[{i+1}/{len(sessions)}]"

        try:
            call_start = time.monotonic()

            # Prepare full conversation (chunked if too large for context)
            chunks, total_tokens = await processor.prepare_conversation_for_reflection(
                sid, summary,
            )

            # Route large sessions to bigger-context endpoint
            min_ctx = total_tokens + 4096 if total_tokens > 28000 else None

            # Generate reflection (auto-merges if multiple chunks)
            result = await processor.process_reflection(
                session_id=sid,
                session_summary=summary,
                conversation_chunks=chunks,
                project=proj,
                min_context_tokens=min_ctx,
            )
            elapsed = time.monotonic() - call_start

            if result is None:
                skipped += 1
                print(f"{prefix} {title[:50]} -> SKIP (trivial session) ({elapsed:.1f}s)")
            else:
                processed += 1
                eff = result.get("effectiveness", "?")
                print(f"{prefix} {title[:50]} -> {eff} ({elapsed:.1f}s)")

        except Exception as e:
            failed += 1
            print(f"{prefix} {title[:50]} -> FAILED: {e}")

        # Progress update every 20
        if (i + 1) % 20 == 0:
            elapsed_total = time.monotonic() - start_time
            rate = (i + 1) / elapsed_total * 60
            remaining = (len(sessions) - i - 1) / rate if rate > 0 else 0
            print(
                f"\n--- Progress: {i+1}/{len(sessions)} "
                f"({processed} OK, {skipped} skipped, {failed} failed, "
                f"{rate:.0f}/min, ~{remaining:.0f} min remaining) ---\n"
            )

    elapsed_total = time.monotonic() - start_time
    print(
        f"\nDone: {processed} reflected, {skipped} skipped, {failed} failed "
        f"({elapsed_total:.0f}s total)"
    )

    await sqlite.close()


def main():
    parser = argparse.ArgumentParser(
        description="Backfill session reflections for existing sessions",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max sessions to process (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show sessions that need reflections without processing",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Only process sessions for this project",
    )
    args = parser.parse_args()
    asyncio.run(backfill(limit=args.limit, dry_run=args.dry_run, project=args.project))


if __name__ == "__main__":
    main()
