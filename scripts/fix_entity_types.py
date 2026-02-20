"""Fix misclassified entity types via curated corrections.

Applies a curated mapping of entity name → correct type for high-mention
entities that were misclassified during extraction. Also applies heuristic
rules for common patterns (e.g., localhost → technology, generic words → concept).

Safe to re-run — idempotent operations.

Usage:
    python scripts/fix_entity_types.py [--dry-run] [--db PATH]
"""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiosqlite

logging.basicConfig(level=logging.WARNING)

# ── Curated corrections: name → correct type ──
# Only entities with significant mention counts that are clearly wrong.
CORRECTIONS = {
    # concept → technology
    "chatgpt": "technology",
    "ollama": "technology",
    "query_messages": "technology",
    "session_id": "technology",
    "store_message": "technology",
    "dda algorithm": "technology",
    "persistent memory": "technology",

    # place → technology
    "localhost": "technology",
    "backend": "technology",
    "computer": "technology",
    "port 11434": "technology",
    "localhost:11434": "technology",
    "open_webui": "technology",

    # technology → concept (generic words misclassified as tech)
    "reflection": "concept",
    "tags": "concept",
    "systems": "concept",
    "frameworks": "concept",
    "tool": "concept",
    "models": "concept",

    # person → concept (generic words misclassified as person)
    "player": "concept",
    "person": "concept",
    "speaker": "concept",

    # project → concept (generic words)
    "project": "concept",
    "experiment": "concept",
    "projects": "concept",
    "creative projects": "concept",
    "metrics": "concept",

    # organization → technology (these are AI products, not orgs)
    "cohere": "technology",

    # preference → concept
    "autonomy": "concept",
    "walking": "concept",

    # place → concept
    "sandbox": "concept",
    "place": "concept",
    "local": "concept",

    # place → technology (file paths / tech locations)
    "github repository": "technology",
    "chroma_data": "technology",

    # technology → project (specific projects)
    "car": "concept",

    # Specific corrections from the data
    "llms": "technology",
    "ai": "technology",
    "llm": "technology",
}

# ── Heuristic rules: patterns that indicate a specific type ──
# Applied after curated corrections. Only for unambiguous patterns.
HEURISTIC_RULES = [
    # File paths are places (already mostly correct)
    (re.compile(r"^c:[/\\]", re.IGNORECASE), "place"),
    (re.compile(r"^/home/", re.IGNORECASE), "place"),
    # .py/.js/.cs files are projects
    (re.compile(r"\.(py|js|ts|cs|sh|txt|json|yaml|yml|toml|cfg|ini|md)$", re.IGNORECASE), "project"),
    # URLs are technology
    (re.compile(r"^https?://"), "technology"),
]


async def run(dry_run: bool, db_path: str | None = None):
    if db_path:
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
    else:
        from blipshell.memory.sqlite_store import SQLiteStore
        from blipshell.core.config import ConfigManager
        config_mgr = ConfigManager()
        config = config_mgr.config
        sqlite = SQLiteStore(config.database.path)
        await sqlite.initialize()
        db = sqlite._db

    # --- 1. Apply curated corrections ---
    print("=== Applying curated type corrections ===")
    corrected = 0
    merged = 0
    for name, correct_type in CORRECTIONS.items():
        cursor = await db.execute(
            "SELECT id, entity_type FROM entities WHERE lower(name) = ?",
            (name.lower(),),
        )
        rows = await cursor.fetchall()
        for row in rows:
            if row["entity_type"] != correct_type:
                if not dry_run:
                    # Check if correcting would create a duplicate
                    dup_cursor = await db.execute(
                        "SELECT id FROM entities WHERE lower(name) = ? AND entity_type = ? AND id != ?",
                        (name.lower(), correct_type, row["id"]),
                    )
                    dup = await dup_cursor.fetchone()
                    if dup:
                        await _merge_entity(db, keep_id=dup["id"], remove_id=row["id"])
                        merged += 1
                    else:
                        await db.execute(
                            "UPDATE entities SET entity_type = ? WHERE id = ?",
                            (correct_type, row["id"]),
                        )
                corrected += 1
                print(f"  {name}: {row['entity_type']} -> {correct_type}")

    if not dry_run and (corrected or merged):
        await db.commit()
    print(f"  Corrected {corrected} entities ({merged} merged as duplicates)")

    # --- 2. Apply heuristic rules ---
    print("\n=== Applying heuristic rules ===")
    cursor = await db.execute("SELECT id, name, entity_type FROM entities")
    all_rows = await cursor.fetchall()
    heuristic_fixes = 0
    for row in all_rows:
        for pattern, correct_type in HEURISTIC_RULES:
            if pattern.search(row["name"]) and row["entity_type"] != correct_type:
                if not dry_run:
                    # Check for duplicate
                    dup_cursor = await db.execute(
                        "SELECT id FROM entities WHERE lower(name) = ? AND entity_type = ? AND id != ?",
                        (row["name"].lower(), correct_type, row["id"]),
                    )
                    dup = await dup_cursor.fetchone()
                    if dup:
                        await _merge_entity(db, keep_id=dup["id"], remove_id=row["id"])
                    else:
                        await db.execute(
                            "UPDATE entities SET entity_type = ? WHERE id = ?",
                            (correct_type, row["id"]),
                        )
                heuristic_fixes += 1
                break  # Only apply first matching rule
    if not dry_run and heuristic_fixes:
        await db.commit()
    print(f"  Fixed {heuristic_fixes} entities via heuristics")

    # --- 3. Deduplicate after type changes ---
    print("\n=== Deduplicating after type corrections ===")
    cursor = await db.execute(
        """SELECT lower(name) as lname, entity_type, COUNT(*) as cnt
           FROM entities GROUP BY lower(name), entity_type HAVING cnt > 1"""
    )
    dup_groups = await cursor.fetchall()
    dedup_count = 0
    for group in dup_groups:
        cursor = await db.execute(
            "SELECT id FROM entities WHERE lower(name) = ? AND entity_type = ? ORDER BY id",
            (group["lname"], group["entity_type"]),
        )
        dupes = await cursor.fetchall()
        if len(dupes) <= 1:
            continue
        keep_id = dupes[0]["id"]
        for dupe in dupes[1:]:
            if not dry_run:
                await _merge_entity(db, keep_id=keep_id, remove_id=dupe["id"])
            dedup_count += 1
    if not dry_run and dedup_count:
        await db.commit()
    print(f"  Merged {dedup_count} duplicates")

    # --- Summary ---
    print("\n=== Type Distribution ===")
    cursor = await db.execute(
        "SELECT entity_type, COUNT(*) as cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC"
    )
    for row in await cursor.fetchall():
        print(f"  {row['entity_type']:15s} {row['cnt']:6d}")

    print("\n=== Top 30 Entities ===")
    cursor = await db.execute("""
        SELECT e.name, e.entity_type, COUNT(em.id) as mentions
        FROM entities e
        LEFT JOIN entity_mentions em ON e.id = em.entity_id
        GROUP BY e.id ORDER BY mentions DESC LIMIT 30
    """)
    for row in await cursor.fetchall():
        print(f"  {row['mentions']:5d}  [{row['entity_type']:12s}] {row['name']}")

    total = (await (await db.execute("SELECT COUNT(*) as cnt FROM entities")).fetchone())["cnt"]
    print(f"\nTotal entities: {total}")

    if db_path:
        await db.close()
    else:
        await sqlite.close()


async def _merge_entity(db, keep_id: int, remove_id: int):
    """Merge remove_id into keep_id: transfer relationships and mentions, then delete."""
    await db.execute(
        "UPDATE OR IGNORE entity_mentions SET entity_id = ? WHERE entity_id = ?",
        (keep_id, remove_id),
    )
    await db.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (remove_id,))
    await db.execute(
        "UPDATE OR IGNORE entity_relationships SET subject_id = ? WHERE subject_id = ?",
        (keep_id, remove_id),
    )
    await db.execute(
        "UPDATE OR IGNORE entity_relationships SET object_id = ? WHERE object_id = ?",
        (keep_id, remove_id),
    )
    await db.execute(
        "DELETE FROM entity_relationships WHERE subject_id = ? OR object_id = ?",
        (remove_id, remove_id),
    )
    await db.execute("DELETE FROM entities WHERE id = ?", (remove_id,))


def main():
    parser = argparse.ArgumentParser(description="Fix misclassified entity types")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show corrections without modifying the DB")
    parser.add_argument("--db", default=None,
                        help="Direct path to DB file (bypasses config)")
    args = parser.parse_args()
    asyncio.run(run(args.dry_run, args.db))


if __name__ == "__main__":
    main()
