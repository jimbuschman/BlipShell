"""Extract entity relationship triples from all unprocessed memories.

One-shot script to process the full backlog instead of 50-per-startup.
Resume support: tracks progress via entities_extracted_at column, so
re-running safely skips already-processed memories.

Usage:
    python scripts/extract_entities.py [--batch-size 100] [--dry-run]
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
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.entity_extractor import EntityExtractor
from blipshell.memory.sqlite_store import SQLiteStore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def run(batch_size: int, dry_run: bool):
    config_mgr = ConfigManager()
    config = config_mgr.config

    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()

    endpoint_mgr = EndpointManager(config.endpoints, config.llm)
    await endpoint_mgr.startup_health_check()
    router = LLMRouter(config.models, endpoint_mgr)

    # Count total unprocessed
    cursor = await sqlite._db.execute(
        """SELECT COUNT(*) as cnt FROM memories
           WHERE entities_extracted_at IS NULL AND is_archived = 0
           AND summary IS NOT NULL"""
    )
    row = await cursor.fetchone()
    total = row["cnt"]
    print(f"Unprocessed memories: {total}")

    if total == 0:
        print("Nothing to do.")
        await sqlite.close()
        return

    if dry_run:
        print(f"[DRY RUN] Would process {total} memories in batches of {batch_size}")
        await sqlite.close()
        return

    processed = 0
    total_triples = 0
    total_errors = 0
    start = time.time()

    extractor = EntityExtractor(sqlite, router, batch_size=batch_size)

    while processed < total:
        stats = await extractor.extract_batch()
        if stats["extracted"] == 0 and stats["errors"] == 0:
            break

        processed += stats["extracted"] + stats["errors"]
        total_triples += stats["triples"]
        total_errors += stats["errors"]

        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total - processed) / rate if rate > 0 else 0

        print(
            f"  {processed}/{total} memories "
            f"({total_triples} triples, {total_errors} errors) "
            f"— {rate:.1f}/s, ~{remaining:.0f}s remaining"
        )

    elapsed = time.time() - start
    print(f"\nDone: {processed} memories, {total_triples} triples, "
          f"{total_errors} errors in {elapsed:.1f}s")

    # Summary stats
    cursor = await sqlite._db.execute("SELECT COUNT(*) as cnt FROM entities")
    entity_count = (await cursor.fetchone())["cnt"]
    cursor = await sqlite._db.execute("SELECT COUNT(*) as cnt FROM entity_relationships")
    rel_count = (await cursor.fetchone())["cnt"]
    print(f"Total entities: {entity_count}, relationships: {rel_count}")

    await sqlite.close()


def main():
    parser = argparse.ArgumentParser(description="Extract entity triples from memories")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Memories per batch (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count unprocessed memories without extracting")
    args = parser.parse_args()

    asyncio.run(run(args.batch_size, args.dry_run))


if __name__ == "__main__":
    main()
