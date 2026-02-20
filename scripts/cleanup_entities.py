"""Clean up dirty entity data from LLM extraction artifacts.

Fixes:
1. Delete entities with </think> leaked into names
2. Delete entities with LLM commentary/reasoning in names
3. Strip formatting prefixes (-, *, #, 1., 2., etc.) and merge with canonical
4. Delete single-char, numeric-only, and blank entity names
5. Delete quote-wrapped commentary names
6. Merge all "user" variants into canonical user entity
7. Delete pronoun/vague entities
8. Fix invalid entity types
9. Delete remaining long names (> 60 chars)

Safe to re-run — idempotent operations.

Usage:
    python scripts/cleanup_entities.py [--dry-run] [--db PATH]
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

# Valid entity types
VALID_TYPES = {"person", "project", "technology", "concept", "preference", "place", "organization"}

# Pronouns and vague words to delete entirely
DELETE_NAMES = {
    "she", "her", "he", "him", "his", "they", "them", "their",
    "it", "its", "this", "that", "these", "those",
    "something", "someone", "anything", "nothing",
    "the user", "the assistant", "assistant", "none",
    "subject", "object", "predicate",
}

# LLM commentary patterns — if a name matches any of these, delete it
COMMENTARY_PATTERNS = [
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"<think>", re.IGNORECASE),
    re.compile(r"\bi think\b", re.IGNORECASE),
    re.compile(r"\blet me\b", re.IGNORECASE),
    re.compile(r"\bhere are\b", re.IGNORECASE),
    re.compile(r"\bthese seem\b", re.IGNORECASE),
    re.compile(r"\bactually[,.]", re.IGNORECASE),
    re.compile(r"\bhmm\b", re.IGNORECASE),
    re.compile(r"\bfinal answer", re.IGNORECASE),
    re.compile(r"\bfinal output", re.IGNORECASE),
    re.compile(r"\bthe memory\b", re.IGNORECASE),
    re.compile(r"\bthis memory\b", re.IGNORECASE),
    re.compile(r"\bthe summary\b", re.IGNORECASE),
    re.compile(r"\bcould be[:\s]", re.IGNORECASE),
    re.compile(r"\bcould say[:\s]", re.IGNORECASE),
    re.compile(r"\bmaybe[:\s]", re.IGNORECASE),
    re.compile(r"\bor better[:\s]", re.IGNORECASE),
    re.compile(r"\btriple[s]?\b", re.IGNORECASE),
    re.compile(r"\bextract\b", re.IGNORECASE),
    re.compile(r"\bpredicate\b", re.IGNORECASE),
    re.compile(r"\bisn't really\b", re.IGNORECASE),
    re.compile(r"\bnot really\b", re.IGNORECASE),
    re.compile(r"\bnothing\b", re.IGNORECASE),
    re.compile(r"\u2192"),  # → arrow
    re.compile(r"->"),       # ASCII arrow
    re.compile(r"\beven if\b", re.IGNORECASE),
    re.compile(r"\bassumed\b", re.IGNORECASE),
    re.compile(r"\bi'll\b", re.IGNORECASE),
    re.compile(r"\bi could\b", re.IGNORECASE),
    re.compile(r"\bi realize\b", re.IGNORECASE),
    re.compile(r"\blooking at\b", re.IGNORECASE),
    re.compile(r"\bexample[s]?\b", re.IGNORECASE),
    re.compile(r"\bisn't\b", re.IGNORECASE),
    re.compile(r"\bdoesn't\b", re.IGNORECASE),
    re.compile(r"\bhasn't\b", re.IGNORECASE),
    re.compile(r"\bwouldn't\b", re.IGNORECASE),
    re.compile(r"\bcouldn't\b", re.IGNORECASE),
    re.compile(r"\bshouldn't\b", re.IGNORECASE),
    re.compile(r"\btoo vague\b", re.IGNORECASE),
    re.compile(r"\bnot allowed\b", re.IGNORECASE),
    re.compile(r"\bnot valid\b", re.IGNORECASE),
    re.compile(r"\bapproved list\b", re.IGNORECASE),
    re.compile(r"\ballowed list\b", re.IGNORECASE),
    re.compile(r"\bdoes not\b", re.IGNORECASE),
    re.compile(r"\bbut this\b", re.IGNORECASE),
    re.compile(r"\bbut again\b", re.IGNORECASE),
    re.compile(r"\bbut \"", re.IGNORECASE),
    re.compile(r"\bno,\s", re.IGNORECASE),
    re.compile(r"\balso mentioned\b", re.IGNORECASE),
    re.compile(r"\bthe format\b", re.IGNORECASE),
    re.compile(r"\bthe entity\b", re.IGNORECASE),
    re.compile(r"\bentity types?\b", re.IGNORECASE),
    re.compile(r"\bstill awkward\b", re.IGNORECASE),
    re.compile(r"\bnot quite\b", re.IGNORECASE),
    re.compile(r"\bskip\b", re.IGNORECASE),
    re.compile(r"\babstract\b", re.IGNORECASE),
    re.compile(r"\bmetaphorical\b", re.IGNORECASE),
    re.compile(r"\bwho is\b", re.IGNORECASE),
    re.compile(r"\bawkward\b", re.IGNORECASE),
    re.compile(r"\bso when\b", re.IGNORECASE),
    re.compile(r"\bthere's no\b", re.IGNORECASE),
    re.compile(r"^\?", re.IGNORECASE),  # starts with ?
    re.compile(r"^so \"", re.IGNORECASE),  # starts with so "
    re.compile(r"\bhome\" or\b", re.IGNORECASE),
    re.compile(r"\btopic\]", re.IGNORECASE),
    re.compile(r"\btriggering\b", re.IGNORECASE),
    re.compile(r"\bchecking in\b", re.IGNORECASE),
    re.compile(r"\bthe user is\b", re.IGNORECASE),
    re.compile(r"\(but\s", re.IGNORECASE),      # (but ...
    re.compile(r"\(this\s", re.IGNORECASE),      # (this ...
    re.compile(r"\(not\s", re.IGNORECASE),       # (not ...
    re.compile(r"\basking about\b", re.IGNORECASE),
    re.compile(r"\bcould work\b", re.IGNORECASE),
    re.compile(r"\bmeaningful\?\b", re.IGNORECASE),
    re.compile(r"\bthat's accurate\b", re.IGNORECASE),
    re.compile(r"\bgets extracted\b", re.IGNORECASE),
    re.compile(r"\bis extracting\b", re.IGNORECASE),
    re.compile(r"\bso a specific\b", re.IGNORECASE),
    re.compile(r"\bcode execution - no\b", re.IGNORECASE),
]

# Regex for names that are LLM commentary appended after a dash
# e.g., "ai - but this is vague", "concept - no, doesn't work"
# Matches any name with " - " separator (min 3 chars after dash)
DASH_COMMENTARY = re.compile(r"^(.{2,60})\s+-\s+(.{3,})$")

def _name_has_dash_commentary(name: str) -> bool:
    """Check if name has LLM commentary appended after ' - '."""
    # Skip file paths (C:\..., /home/...)
    if name.startswith(("c:", "C:", "/", "\\")):
        return False
    m = DASH_COMMENTARY.match(name)
    if not m:
        return False
    before = m.group(1).strip()
    after = m.group(2).lower()

    # If the part after dash is 4+ words, it's almost certainly commentary
    if len(after.split()) >= 4:
        return True

    # If the after-dash part is just a single word that's an entity type
    # or repeats a word from before, it's LLM noting the type
    after_words = after.strip().rstrip("?").split()
    if len(after_words) == 1:
        single = after_words[0].lower().rstrip("\"'")
        # Entity type names used as commentary
        if single in {"person", "technology", "concept", "project",
                       "preference", "place", "organization", "vague",
                       "person", "script", "files", "yes", "maybe",
                       "similar"}:
            return True
        # Repeats a word from the entity part
        if single in before.lower():
            return True

    # Also catch specific commentary words
    commentary_words = [
        "but", "this", "vague", "abstract", "possible", "reasonable",
        "doesn't", "isn't", "not", "could", "too", "maybe", "also",
        "related", "describes", "mentioned", "allowed", "valid",
        "work", "fit", "sense", "great", "approved",
        "selenium", "being used", "interacts", "the script",
        "the assistant", "the user", "somewhat", "concept",
        "a bit", "if this", "movie", "discussing", "person",
        "stated", "explicitly", "visible", "considering",
        "meaningful", "quite", "exploring", "same issue",
        "indirectly", "specific", "similar", "activity",
        "character", "develops", "or \"", "or built",
        "yes", "same", "so user", "technology uses",
    ]
    return any(w in after for w in commentary_words)

# Formatting prefix pattern — strip these and try to salvage the entity name
FORMAT_PREFIX = re.compile(
    r"^(?:"
    r"[\-\*\#]+\s+"            # - item, * item, # item, ## item
    r"|\d{1,2}\.\s+"           # 1. item, 12. item
    r"|`"                       # `backtick wrapped`
    r"|\"|\'"                   # quote-wrapped
    r")",
)


def has_commentary(name: str) -> bool:
    """Check if an entity name contains LLM commentary/reasoning."""
    for pattern in COMMENTARY_PATTERNS:
        if pattern.search(name):
            return True
    if _name_has_dash_commentary(name):
        return True
    return False


def strip_formatting(name: str) -> str:
    """Strip list formatting prefixes from an entity name."""
    cleaned = name
    # Repeatedly strip formatting prefixes
    for _ in range(3):
        m = FORMAT_PREFIX.match(cleaned)
        if m:
            cleaned = cleaned[m.end():]
        else:
            break
    # Strip trailing backticks/quotes
    cleaned = cleaned.strip("`\"' \t")
    return cleaned.strip()


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


async def run(dry_run: bool, db_path: str | None = None):
    if db_path:
        # Direct DB access (for testing on a copy)
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

    # --- Initial stats ---
    r = (await (await db.execute("SELECT COUNT(*) as cnt FROM entities")).fetchone())["cnt"]
    print(f"Starting entities: {r}")

    # --- 1. Delete entities with LLM commentary in names ---
    print("\n=== Pass 1: Deleting entities with LLM commentary in names ===")
    cursor = await db.execute("SELECT id, name FROM entities")
    all_rows = await cursor.fetchall()
    commentary_count = 0
    for row in all_rows:
        if has_commentary(row["name"]):
            if not dry_run:
                await _delete_entity(db, row["id"])
            commentary_count += 1
    if not dry_run and commentary_count:
        await db.commit()
    print(f"  Deleted {commentary_count} entities with LLM commentary")

    # --- 2. Strip formatting prefixes and merge/rename ---
    print("\n=== Pass 2: Stripping formatting prefixes ===")
    cursor = await db.execute("SELECT id, name, entity_type FROM entities")
    all_rows = await cursor.fetchall()
    stripped = 0
    merged = 0
    deleted_empty = 0
    for row in all_rows:
        name = row["name"]
        cleaned = strip_formatting(name)
        if cleaned == name:
            continue
        if not cleaned or len(cleaned) <= 1:
            # Stripping left nothing useful — delete
            if not dry_run:
                await _delete_entity(db, row["id"])
            deleted_empty += 1
            continue

        cleaned_lower = cleaned.lower()
        if not dry_run:
            # Check if cleaned name already exists
            dup_cursor = await db.execute(
                "SELECT id FROM entities WHERE lower(name) = ? AND entity_type = ? AND id != ?",
                (cleaned_lower, row["entity_type"], row["id"]),
            )
            dup = await dup_cursor.fetchone()
            if dup:
                await _merge_entity(db, keep_id=dup["id"], remove_id=row["id"])
                merged += 1
            else:
                # Also check without type constraint (merge across types)
                dup_cursor = await db.execute(
                    "SELECT id FROM entities WHERE lower(name) = ? AND id != ?",
                    (cleaned_lower, row["id"]),
                )
                dup = await dup_cursor.fetchone()
                if dup:
                    await _merge_entity(db, keep_id=dup["id"], remove_id=row["id"])
                    merged += 1
                else:
                    await db.execute(
                        "UPDATE entities SET name = ? WHERE id = ?",
                        (cleaned, row["id"]),
                    )
                    stripped += 1
        else:
            stripped += 1
    if not dry_run:
        await db.commit()
    print(f"  Renamed {stripped}, merged {merged}, deleted {deleted_empty} empty")

    # --- 3. Delete single-char, numeric-only, and short junk names ---
    print("\n=== Pass 3: Deleting single-char and numeric-only names ===")
    cursor = await db.execute("SELECT id, name FROM entities")
    all_rows = await cursor.fetchall()
    junk_count = 0
    for row in all_rows:
        name = row["name"].strip()
        # Single char, numeric, or common junk
        if len(name) <= 1 or name.replace(".", "").replace("-", "").isdigit():
            if not dry_run:
                await _delete_entity(db, row["id"])
            junk_count += 1
    if not dry_run and junk_count:
        await db.commit()
    print(f"  Deleted {junk_count} junk entities")

    # --- 4. Merge all user variants ---
    print("\n=== Pass 4: Merging user variants ===")
    cursor = await db.execute("SELECT id, name, entity_type FROM entities")
    all_rows = await cursor.fetchall()

    # Find canonical "user" entity
    canonical_user = None
    user_variants = []
    for row in all_rows:
        name = row["name"].strip().lower()
        if name == "user" and row["entity_type"] == "person":
            canonical_user = row["id"]
            break
    if not canonical_user:
        # Try any "user" entity
        for row in all_rows:
            if row["name"].strip().lower() == "user":
                canonical_user = row["id"]
                break

    if canonical_user:
        # Find variants: anything with "user" that isn't the canonical
        user_variant_re = re.compile(
            r"^[\-\*\#\d\.\s`\"']*(?:the\s+)?user[\s`\"']*$",
            re.IGNORECASE,
        )
        for row in all_rows:
            if row["id"] == canonical_user:
                continue
            name = row["name"]
            if user_variant_re.match(name):
                user_variants.append((row["id"], name))

        if user_variants:
            for vid, vname in user_variants:
                if not dry_run:
                    await _merge_entity(db, keep_id=canonical_user, remove_id=vid)
            if not dry_run:
                await db.commit()
        print(f"  Merged {len(user_variants)} user variants into entity {canonical_user}")
    else:
        print("  No canonical user entity found")

    # --- 5. Delete pronoun/vague entities ---
    print("\n=== Pass 5: Deleting pronoun/vague entities ===")
    cursor = await db.execute("SELECT id, name FROM entities")
    all_rows = await cursor.fetchall()
    pronoun_count = 0
    for row in all_rows:
        if row["name"].strip().lower() in DELETE_NAMES:
            if not dry_run:
                await _delete_entity(db, row["id"])
            pronoun_count += 1
    if not dry_run and pronoun_count:
        await db.commit()
    print(f"  Deleted {pronoun_count} pronoun/vague entities")

    # --- 6. Fix invalid entity types ---
    print("\n=== Pass 6: Fixing invalid entity types ===")
    cursor = await db.execute(
        "SELECT id, name, entity_type FROM entities WHERE entity_type NOT IN (?, ?, ?, ?, ?, ?, ?)",
        tuple(VALID_TYPES),
    )
    dirty_rows = await cursor.fetchall()
    type_fixes = 0
    for row in dirty_rows:
        eid, name, raw_type = row["id"], row["name"], row["entity_type"]
        cleaned = clean_entity_type(raw_type)
        if not dry_run:
            # Check for duplicate after type fix
            dup_cursor = await db.execute(
                "SELECT id FROM entities WHERE name = ? AND entity_type = ? AND id != ?",
                (name, cleaned, eid),
            )
            dup = await dup_cursor.fetchone()
            if dup:
                await _merge_entity(db, keep_id=dup["id"], remove_id=eid)
            else:
                await db.execute(
                    "UPDATE entities SET entity_type = ? WHERE id = ?",
                    (cleaned, eid),
                )
        type_fixes += 1
    if not dry_run and type_fixes:
        await db.commit()
    print(f"  Fixed {type_fixes} entity types")

    # --- 7. Deduplicate entities (same name, different case/type) ---
    print("\n=== Pass 7: Deduplicating entities by name ===")
    cursor = await db.execute(
        """SELECT lower(name) as lname, COUNT(*) as cnt
           FROM entities GROUP BY lower(name) HAVING cnt > 1"""
    )
    dup_groups = await cursor.fetchall()
    dedup_count = 0
    for group in dup_groups:
        lname = group["lname"]
        cursor = await db.execute(
            "SELECT id, name, entity_type FROM entities WHERE lower(name) = ? ORDER BY id",
            (lname,),
        )
        dupes = await cursor.fetchall()
        if len(dupes) <= 1:
            continue
        # Keep the first one, merge the rest into it
        keep_id = dupes[0]["id"]
        for dupe in dupes[1:]:
            if not dry_run:
                await _merge_entity(db, keep_id=keep_id, remove_id=dupe["id"])
            dedup_count += 1
    if not dry_run and dedup_count:
        await db.commit()
    print(f"  Merged {dedup_count} duplicate entities")

    # --- 8. Delete remaining long names (> 60 chars) ---
    print("\n=== Pass 8: Deleting long names (> 60 chars) ===")
    cursor = await db.execute("SELECT id, name FROM entities WHERE length(name) > 60")
    long_rows = await cursor.fetchall()
    if long_rows:
        for row in long_rows:
            if not dry_run:
                await _delete_entity(db, row["id"])
        if not dry_run:
            await db.commit()
    print(f"  Deleted {len(long_rows)} entities with long names")

    # --- 9. Clean up orphaned relationships and mentions ---
    print("\n=== Pass 9: Cleaning orphaned relationships and mentions ===")
    if not dry_run:
        cursor = await db.execute(
            """DELETE FROM entity_relationships WHERE
               subject_id NOT IN (SELECT id FROM entities) OR
               object_id NOT IN (SELECT id FROM entities)"""
        )
        orphan_rels = cursor.rowcount
        cursor = await db.execute(
            """DELETE FROM entity_mentions WHERE
               entity_id NOT IN (SELECT id FROM entities)"""
        )
        orphan_mentions = cursor.rowcount
        await db.commit()
        print(f"  Removed {orphan_rels} orphaned relationships, {orphan_mentions} orphaned mentions")
    else:
        cursor = await db.execute(
            """SELECT COUNT(*) as cnt FROM entity_relationships WHERE
               subject_id NOT IN (SELECT id FROM entities) OR
               object_id NOT IN (SELECT id FROM entities)"""
        )
        orphan_rels = (await cursor.fetchone())["cnt"]
        cursor = await db.execute(
            """SELECT COUNT(*) as cnt FROM entity_mentions WHERE
               entity_id NOT IN (SELECT id FROM entities)"""
        )
        orphan_mentions = (await cursor.fetchone())["cnt"]
        print(f"  Would remove {orphan_rels} orphaned relationships, {orphan_mentions} orphaned mentions")

    # --- Final stats ---
    print("\n=== Final Stats ===")
    r = (await (await db.execute("SELECT COUNT(*) as cnt FROM entities")).fetchone())["cnt"]
    print(f"  Entities: {r}")
    r = (await (await db.execute("SELECT COUNT(*) as cnt FROM entity_relationships")).fetchone())["cnt"]
    print(f"  Relationships: {r}")
    r = (await (await db.execute("SELECT COUNT(*) as cnt FROM entity_mentions")).fetchone())["cnt"]
    print(f"  Mentions: {r}")

    # Top entities by mention count
    print("\n=== Top 20 Entities ===")
    cursor = await db.execute("""
        SELECT e.name, e.entity_type, COUNT(em.id) as mentions
        FROM entities e
        LEFT JOIN entity_mentions em ON e.id = em.entity_id
        GROUP BY e.id
        ORDER BY mentions DESC LIMIT 20
    """)
    for row in await cursor.fetchall():
        print(f"  {row['mentions']:4d}  [{row['entity_type']}] {row['name']}")

    if db_path:
        await db.close()
    else:
        await sqlite.close()


async def _merge_entity(db, keep_id: int, remove_id: int):
    """Merge remove_id entity into keep_id: transfer relationships and mentions, then delete."""
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
    parser.add_argument("--db", default=None,
                        help="Direct path to DB file (bypasses config)")
    args = parser.parse_args()
    asyncio.run(run(args.dry_run, args.db))


if __name__ == "__main__":
    main()
