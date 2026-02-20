"""Clean up dirty entity data from LLM extraction artifacts.

Fixes:
1. Entity types with leaked thinking tokens (</think>, commentary, etc.)
2. Duplicate "user" entities from numbered/bulleted list formatting
3. Pronoun entities that should have been skipped (she, her, he, him, etc.)

Safe to re-run — idempotent operations.

Usage:
    python scripts/cleanup_entities.py [--dry-run]
"""

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.core.config import ConfigManager

logging.basicConfig(level=logging.WARNING)

# Valid entity types
VALID_TYPES = {"person", "project", "technology", "concept", "preference", "place", "organization"}

# Pronouns and vague words to delete entirely
DELETE_NAMES = {
    "she", "her", "he", "him", "his", "they", "them", "their",
    "it", "its", "this", "that", "these", "those",
    "something", "someone", "anything", "nothing",
    "the user", "the assistant",
}

# Patterns that indicate a "user" variant to merge
USER_PATTERN = re.compile(r"^[\-\d\.\*\#\s]*user$", re.IGNORECASE)


def clean_entity_type(raw_type: str) -> str:
    """Normalize a dirty entity type to a valid one."""
    t = raw_type.strip().lower()

    # Strip </think> and everything after it
    if "</think>" in t:
        t = t.split("</think>")[0].strip()

    # Strip trailing commentary after " - " or " ("
    t = re.split(r"\s*[\-\(]", t)[0].strip()

    # Strip trailing punctuation
    t = t.rstrip("?.!:,;")

    # Map to valid type
    if t in VALID_TYPES:
        return t

    # Fuzzy matches
    if "tech" in t:
        return "technology"
    if "person" in t:
        return "person"
    if "project" in t:
        return "project"
    if "prefer" in t:
        return "preference"
    if "place" in t or "location" in t:
        return "place"
    if "org" in t:
        return "organization"

    return "concept"


async def run(dry_run: bool):
    config_mgr = ConfigManager()
    config = config_mgr.config

    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()
    db = sqlite._db

    # --- 1. Fix dirty entity types ---
    print("=== Fixing dirty entity types ===")
    cursor = await db.execute(
        "SELECT id, entity_type FROM entities WHERE entity_type NOT IN (?, ?, ?, ?, ?, ?, ?)",
        tuple(VALID_TYPES),
    )
    dirty_rows = await cursor.fetchall()
    print(f"  Found {len(dirty_rows)} entities with invalid types")

    type_fixes = 0
    for row in dirty_rows:
        eid, raw_type = row["id"], row["entity_type"]
        cleaned = clean_entity_type(raw_type)
        if cleaned != raw_type:
            if not dry_run:
                # Check if this would create a duplicate (same name + cleaned type)
                name_cursor = await db.execute("SELECT name FROM entities WHERE id = ?", (eid,))
                name_row = await name_cursor.fetchone()
                if name_row:
                    dup_cursor = await db.execute(
                        "SELECT id FROM entities WHERE name = ? AND entity_type = ? AND id != ?",
                        (name_row["name"], cleaned, eid),
                    )
                    dup = await dup_cursor.fetchone()
                    if dup:
                        # Merge into existing entity
                        await _merge_entity(db, keep_id=dup["id"], remove_id=eid)
                    else:
                        await db.execute(
                            "UPDATE entities SET entity_type = ? WHERE id = ?",
                            (cleaned, eid),
                        )
            type_fixes += 1
    if not dry_run:
        await db.commit()
    print(f"  Fixed {type_fixes} entity types")

    # --- 2. Merge duplicate user entities ---
    print("\n=== Merging duplicate 'user' entities ===")
    cursor = await db.execute("SELECT id, name, entity_type FROM entities")
    all_entities = await cursor.fetchall()

    # Find the canonical "user" entity (person type)
    canonical_user = None
    user_variants = []
    for row in all_entities:
        name = row["name"]
        if name == "user" and row["entity_type"] == "person":
            canonical_user = row["id"]
        elif USER_PATTERN.match(name) and name != "user":
            user_variants.append((row["id"], name))

    if canonical_user and user_variants:
        print(f"  Found {len(user_variants)} user variants to merge into entity {canonical_user}")
        for variant_id, variant_name in user_variants:
            if not dry_run:
                await _merge_entity(db, keep_id=canonical_user, remove_id=variant_id)
        if not dry_run:
            await db.commit()
        print(f"  Merged {len(user_variants)} variants")
    else:
        print("  No user variants to merge")

    # --- 3. Delete pronoun entities ---
    print("\n=== Deleting pronoun/vague entities ===")
    pronoun_ids = []
    for row in all_entities:
        if row["name"] in DELETE_NAMES:
            pronoun_ids.append((row["id"], row["name"]))

    if pronoun_ids:
        print(f"  Found {len(pronoun_ids)} pronoun/vague entities to delete")
        for pid, pname in pronoun_ids:
            if not dry_run:
                await _delete_entity(db, pid)
        if not dry_run:
            await db.commit()
        print(f"  Deleted {len(pronoun_ids)} entities")
    else:
        print("  No pronoun entities found")

    # --- 4. Delete entities with very long names (LLM rambling) ---
    print("\n=== Deleting entities with excessively long names ===")
    cursor = await db.execute("SELECT id, name FROM entities WHERE length(name) > 80")
    long_rows = await cursor.fetchall()
    if long_rows:
        print(f"  Found {len(long_rows)} entities with names > 80 chars")
        for row in long_rows:
            if not dry_run:
                await _delete_entity(db, row["id"])
        if not dry_run:
            await db.commit()
        print(f"  Deleted {len(long_rows)} entities")
    else:
        print("  None found")

    # --- Summary ---
    print("\n=== Final Stats ===")
    r = (await (await db.execute("SELECT COUNT(*) as cnt FROM entities")).fetchone())["cnt"]
    print(f"  Entities: {r}")
    r = (await (await db.execute("SELECT COUNT(*) as cnt FROM entity_relationships")).fetchone())["cnt"]
    print(f"  Relationships: {r}")
    r = (await (await db.execute("SELECT COUNT(*) as cnt FROM entity_mentions")).fetchone())["cnt"]
    print(f"  Mentions: {r}")

    await sqlite.close()


async def _merge_entity(db, keep_id: int, remove_id: int):
    """Merge remove_id entity into keep_id: transfer relationships and mentions, then delete."""
    # Transfer mentions
    await db.execute(
        "UPDATE OR IGNORE entity_mentions SET entity_id = ? WHERE entity_id = ?",
        (keep_id, remove_id),
    )
    # Delete remaining mentions (duplicates that couldn't be updated)
    await db.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (remove_id,))

    # Transfer relationships (subject side)
    await db.execute(
        "UPDATE OR IGNORE entity_relationships SET subject_id = ? WHERE subject_id = ?",
        (keep_id, remove_id),
    )
    # Transfer relationships (object side)
    await db.execute(
        "UPDATE OR IGNORE entity_relationships SET object_id = ? WHERE object_id = ?",
        (keep_id, remove_id),
    )
    # Delete remaining relationships that couldn't be transferred (duplicates)
    await db.execute(
        "DELETE FROM entity_relationships WHERE subject_id = ? OR object_id = ?",
        (remove_id, remove_id),
    )

    # Delete the entity itself
    await db.execute("DELETE FROM entities WHERE id = ?", (remove_id,))


async def _delete_entity(db, entity_id: int):
    """Delete an entity and all its mentions and relationships."""
    await db.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (entity_id,))
    await db.execute(
        "DELETE FROM entity_relationships WHERE subject_id = ? OR object_id = ?",
        (entity_id, entity_id),
    )
    await db.execute("DELETE FROM entities WHERE id = ?", (entity_id,))


def main():
    parser = argparse.ArgumentParser(description="Clean up entity extraction artifacts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be changed without modifying the DB")
    args = parser.parse_args()
    asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    main()
