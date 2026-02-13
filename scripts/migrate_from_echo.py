"""One-time migration from EchoFrontendV2's MemoryDatabase.db to BlipShell.

Reads existing SQLite database, imports memories/core memories/lessons/sessions
into the new schema, and populates ChromaDB for semantic search.

Usage:
    python -m scripts.migrate_from_echo --source MemoryDatabase.db
"""

import argparse
import asyncio
import logging
import sqlite3
import struct
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def dequantize(blob: bytes) -> list[float]:
    """Dequantize byte blob back to float array (port of C# Dequantize)."""
    return [(b - 128.0) / 127.0 for b in blob]


async def migrate(source_db: str, target_db: str = "data/blipshell.db",
                  chroma_path: str = "data/chroma"):
    """Run the migration."""
    from blipshell.memory.chroma_store import ChromaStore
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.memory.tagger import tag_message
    from blipshell.models.memory import CoreMemory, Lesson, Memory, MemoryType

    # Open source
    src = sqlite3.connect(source_db)
    src.row_factory = sqlite3.Row
    logger.info("Opened source: %s", source_db)

    # Initialize target
    target = SQLiteStore(target_db)
    await target.initialize()

    chroma = ChromaStore(chroma_path)
    chroma.initialize()

    # --- Migrate Sessions ---
    logger.info("Migrating sessions...")
    session_map = {}  # old_id -> new_id
    try:
        for row in src.execute("SELECT * FROM Sessions ORDER BY ID"):
            new_id = await target.create_session(
                title=row["Title"] or "Imported Session",
            )
            session_map[row["ID"]] = new_id
            if row["Summary"]:
                await target.update_session(new_id, summary=row["Summary"])
        logger.info("  Migrated %d sessions", len(session_map))
    except Exception as e:
        logger.warning("  Session migration failed: %s", e)

    # --- Migrate Memories ---
    logger.info("Migrating memories...")
    mem_count = 0
    for row in src.execute("SELECT * FROM Memories ORDER BY ID"):
        session_id = session_map.get(row["SessionID"])
        memory = Memory(
            session_id=session_id,
            role=row["Speaker"] or "user",
            content=row["Text"],
            summary=row["SummaryText"] or row["Text"],
            timestamp=row["TimeStamp"] or datetime.utcnow().isoformat(),
            rank=int(row["Rank"]) if row["Rank"] else 3,
            importance=float(row["Importance"]) if row["Importance"] else 0.3,
            memory_type=MemoryType.CONVERSATION,
        )
        new_id = await target.create_memory(memory)

        # Embed in ChromaDB (use summary for better search)
        summary = row["SummaryText"] or row["Text"]
        try:
            chroma.add_memory(new_id, summary, {
                "session_id": str(session_id or 0),
                "role": row["Speaker"] or "user",
            })
        except Exception as e:
            logger.warning("  ChromaDB embed failed for memory %d: %s", new_id, e)

        # Migrate tags
        try:
            tags = tag_message(row["Text"])
            await target.tag_memory(new_id, tags)
        except Exception:
            pass

        mem_count += 1
    logger.info("  Migrated %d memories", mem_count)

    # --- Migrate Core Memories ---
    logger.info("Migrating core memories...")
    core_count = 0
    try:
        for row in src.execute("SELECT * FROM CoreMemory WHERE IsActive = 1"):
            cm = CoreMemory(
                content=row["Content"],
                category=row["Type"] or "general",
                timestamp=row["Created"] or datetime.utcnow().isoformat(),
                importance=float(row["Priority"]) if row["Priority"] else 0.5,
            )
            new_id = await target.create_core_memory(cm)

            try:
                chroma.add_core_memory(new_id, row["Content"])
            except Exception:
                pass

            try:
                tags = tag_message(row["Content"])
                await target.tag_core_memory(new_id, tags)
            except Exception:
                pass

            core_count += 1
        logger.info("  Migrated %d core memories", core_count)
    except Exception as e:
        logger.warning("  Core memory migration failed: %s", e)

    # --- Migrate Lessons ---
    logger.info("Migrating lessons...")
    lesson_count = 0
    try:
        for row in src.execute("SELECT * FROM Lessons"):
            lesson = Lesson(
                content=row["Text"],
                timestamp=row["TimeStamp"] or datetime.utcnow().isoformat(),
            )
            new_id = await target.create_lesson(lesson)

            try:
                chroma.add_lesson(new_id, row["Text"])
            except Exception:
                pass

            try:
                tags = tag_message(row["Text"])
                await target.tag_lesson(new_id, tags)
            except Exception:
                pass

            lesson_count += 1
        logger.info("  Migrated %d lessons", lesson_count)
    except Exception as e:
        logger.warning("  Lesson migration failed: %s", e)

    # Done
    src.close()
    await target.close()
    logger.info("Migration complete!")
    logger.info("  Sessions: %d, Memories: %d, Core: %d, Lessons: %d",
                len(session_map), mem_count, core_count, lesson_count)


def main():
    parser = argparse.ArgumentParser(description="Migrate from EchoFrontendV2 to BlipShell")
    parser.add_argument("--source", required=True, help="Path to MemoryDatabase.db")
    parser.add_argument("--target", default="data/blipshell.db", help="Target SQLite path")
    parser.add_argument("--chroma", default="data/chroma", help="ChromaDB persist path")
    args = parser.parse_args()

    asyncio.run(migrate(args.source, args.target, args.chroma))


if __name__ == "__main__":
    main()
