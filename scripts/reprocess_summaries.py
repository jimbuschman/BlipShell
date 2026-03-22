"""Re-summarize and re-embed all memories with the improved prompt.

81% of existing summaries are third-person abstractions ("User asked about X")
that lose key facts, names, and specifics. This script:
1. Re-summarizes each memory using the fixed prompt (preserves facts)
2. Re-embeds in ChromaDB using the new summary
3. Tracks progress — can be interrupted and resumed

Usage:
    python scripts/reprocess_summaries.py                     # dry run (show what would change)
    python scripts/reprocess_summaries.py --apply             # run it
    python scripts/reprocess_summaries.py --apply --limit 100 # test on 100 memories first
    python scripts/reprocess_summaries.py --apply --resume    # resume from last checkpoint
    python scripts/reprocess_summaries.py --apply --embed-only  # skip re-summarize, just re-embed with raw content

Options:
    --db PATH         SQLite database path (default: data/blipshell.db)
    --chroma PATH     ChromaDB directory (default: data/chroma)
    --ollama URL      Ollama URL (default: http://localhost:11434)
    --limit N         Process only N memories (for testing)
    --resume          Resume from last checkpoint
    --embed-only      Skip LLM re-summarization, just re-embed with existing content
    --batch N         Batch size for commits (default: 50)
    --quiet           Minimal output
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("reprocess")

CHECKPOINT_KEY = "reprocess_summaries_last_id"


async def run(args):
    import aiosqlite
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.llm.prompts import summarize_memory

    # Connect to SQLite
    db = await aiosqlite.connect(args.db)
    db.row_factory = aiosqlite.Row

    # Connect to ChromaDB
    chroma = ChromaStore(persist_dir=args.chroma, ollama_url=args.ollama)
    chroma.initialize()

    # Set up LLM router for summarization (only if not embed-only)
    router = None
    if not args.embed_only:
        from blipshell.core.config import ConfigManager
        from blipshell.llm.router import LLMRouter
        from blipshell.llm.endpoints import EndpointManager
        config = ConfigManager(args.config).load()
        ep_mgr = EndpointManager(config.endpoints, config.llm)
        router = LLMRouter(config.models, ep_mgr, pii_enabled=False)

    # Get resume checkpoint
    last_id = 0
    if args.resume:
        cursor = await db.execute(
            "SELECT value FROM app_metadata WHERE key = ?", (CHECKPOINT_KEY,)
        )
        row = await cursor.fetchone()
        if row:
            last_id = int(row[0])
            logger.info("Resuming from memory ID %d", last_id)

    # Count total work
    cursor = await db.execute(
        "SELECT COUNT(*) FROM memories WHERE is_archived = 0 AND id > ? AND content IS NOT NULL",
        (last_id,),
    )
    total = (await cursor.fetchone())[0]
    if args.limit:
        total = min(total, args.limit)
    logger.info("Memories to process: %d", total)

    if not args.apply:
        # Dry run — show sample of what would change
        cursor = await db.execute(
            """SELECT id, substr(summary, 1, 80) as old_summary, substr(content, 1, 80) as content_preview
               FROM memories WHERE is_archived = 0 AND id > ? AND content IS NOT NULL
               ORDER BY id LIMIT 10""",
            (last_id,),
        )
        rows = await cursor.fetchall()
        print(f"\nDRY RUN — would process {total} memories")
        print(f"\nSample memories:")
        for r in rows:
            print(f"  [{r['id']}] Summary: {r['old_summary']}")
            print(f"         Content: {r['content_preview']}")
            print()
        await db.close()
        return

    # Process memories in batches
    processed = 0
    errors = 0
    skipped = 0
    batch_start = time.monotonic()

    cursor = await db.execute(
        """SELECT id, content, summary, role, session_id
           FROM memories
           WHERE is_archived = 0 AND id > ? AND content IS NOT NULL
           ORDER BY id
           LIMIT ?""",
        (last_id, args.limit or 999999999),
    )
    rows = await cursor.fetchall()

    for i, row in enumerate(rows):
        mem_id = row["id"]
        content = row["content"]
        old_summary = row["summary"] or ""
        role = row["role"]
        session_id = row["session_id"]

        try:
            if args.embed_only:
                # Just re-embed with raw content (no LLM call)
                new_summary = old_summary
                embed_text = content if content else old_summary
            else:
                # Re-summarize with improved prompt
                sum_system, sum_prompt = summarize_memory(content)
                from blipshell.llm.router import TaskType
                new_summary = await router.generate(
                    TaskType.SUMMARIZATION,
                    sum_prompt,
                    system=sum_system,
                )

                if new_summary.strip().upper() == "SKIP":
                    skipped += 1
                    if not args.quiet:
                        logger.debug("Skipped %d (SKIP response)", mem_id)
                    continue

                # Update summary in SQLite
                await db.execute(
                    "UPDATE memories SET summary = ? WHERE id = ?",
                    (new_summary, mem_id),
                )
                embed_text = new_summary

            # Re-embed in ChromaDB
            metadata = {"session_id": str(session_id), "role": role, "source": "memory"}
            try:
                chroma.add_memory(mem_id, embed_text, metadata)
            except Exception as e:
                logger.warning("ChromaDB embed failed for %d: %s", mem_id, e)

            processed += 1

            # Checkpoint every batch_size memories
            if processed % args.batch == 0:
                await db.execute(
                    "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
                    (CHECKPOINT_KEY, str(mem_id)),
                )
                await db.commit()
                elapsed = time.monotonic() - batch_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                if not args.quiet:
                    logger.info(
                        "Progress: %d/%d (%.0f%%) — %.1f/s — ETA %.0fm — errors: %d",
                        processed, total, processed * 100 / total, rate, eta / 60, errors,
                    )

        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.error("Error processing memory %d: %s", mem_id, e)
            elif errors == 11:
                logger.error("Suppressing further error logs...")

            # If we're getting rate limited, wait and continue
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning("Rate limited — waiting 60s before continuing")
                await asyncio.sleep(60)

    # Final commit and checkpoint
    await db.execute(
        "INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)",
        (CHECKPOINT_KEY, str(rows[-1]["id"]) if rows else str(last_id)),
    )
    await db.commit()
    await db.close()

    elapsed = time.monotonic() - batch_start
    logger.info(
        "Done: %d processed, %d skipped, %d errors in %.1fs (%.1f/s)",
        processed, skipped, errors, elapsed, processed / elapsed if elapsed > 0 else 0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Re-summarize and re-embed all memories with improved prompt",
    )
    parser.add_argument("--db", default="data/blipshell.db")
    parser.add_argument("--chroma", default="data/chroma")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path for LLM router (default: config.yaml)")
    parser.add_argument("--apply", action="store_true", help="Actually process (default: dry run)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N memories")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--embed-only", action="store_true",
                        help="Skip re-summarization, just re-embed existing content")
    parser.add_argument("--batch", type=int, default=50, help="Checkpoint batch size")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
