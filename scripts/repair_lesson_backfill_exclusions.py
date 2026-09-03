"""One-shot repair: stop backfill_lessons re-creating consolidated lessons.

The 2026-09-02 cleanup deleted 358 lessons (321 paraphrase-family duplicates,
37 defective rows). Their source sessions then matched backfill_lessons'
"has messages, no lessons" query, so the next nightly started faithfully
re-extracting them — timing out on the backlog and slowly re-creating the
duplicates the consolidation removed.

This script reads the cleanup's own receipt files (written next to the
database) and registers the affected sessions in the lesson-backfill
exclusion set:

  - ALL sessions from the consolidation receipts (their insight survives in
    a keeper lesson from another session), and
  - the false-anti-pattern sessions from the patch receipts (their "lesson"
    was a false accusation; nothing worth re-extracting),
  - but NOT the raw-assistant-dump sessions (those never had a real lesson
    extracted — one genuine extraction by the current model is a feature).

Run once on the box holding the live DB (BlipShell may be running; this is
one metadata write):
    python scripts/repair_lesson_backfill_exclusions.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The 9 quoted-dialogue false anti-patterns (their sessions get excluded);
# the other 28 deleted-by-patch rows were assistant-reply dumps (sessions
# stay eligible for one real extraction).
FALSE_ANTIPATTERN_LESSON_IDS = {1297, 1298, 1299, 1301, 1308, 1351, 1372,
                                1396, 1406}


async def run() -> int:
    from blipshell.core.config import ConfigManager
    from blipshell.memory.sqlite_store import SQLiteStore

    config = ConfigManager().load()
    db_dir = Path(config.database.path).parent

    session_ids: set[int] = set()
    found_files = []

    consolidation = db_dir / "lesson_consolidation_receipts_2026-09-02.json"
    if consolidation.exists():
        data = json.loads(consolidation.read_text(encoding="utf-8"))
        for row in data.get("deleted_lessons", []):
            if row.get("source_session_id") is not None:
                session_ids.add(int(row["source_session_id"]))
        found_files.append(consolidation.name)

    patch = db_dir / "db_patch_receipts_2026-09-02.json"
    if patch.exists():
        data = json.loads(patch.read_text(encoding="utf-8"))
        for row in data.get("deleted_lessons", []):
            if (row.get("id") in FALSE_ANTIPATTERN_LESSON_IDS
                    and row.get("source_session_id") is not None):
                session_ids.add(int(row["source_session_id"]))
        found_files.append(patch.name)

    if not found_files:
        print(f"No receipt files found next to the database ({db_dir}) — "
              f"nothing to repair. (Were they moved?)")
        return 1
    if not session_ids:
        print("Receipt files found but no session ids in them — nothing to do.")
        return 0

    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()
    try:
        total = await sqlite.add_lesson_backfill_exclusions(session_ids)
        remaining = await sqlite.get_sessions_missing_lessons(limit=500)
        print(f"read: {', '.join(found_files)}")
        print(f"excluded {len(session_ids)} sessions "
              f"(exclusion set now {total})")
        print(f"sessions still legitimately missing lessons: {len(remaining)}")
        return 0
    finally:
        await sqlite.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
