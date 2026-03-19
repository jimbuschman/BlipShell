"""One-time memory cleanup + project backfill.

Fixes data quality issues found during search benchmarking:
1. Backfill project tags — tag sessions with BlipShell content as project=blipshell
2. Archive duplicate memories — keep newest, archive older copies
3. Archive stress test artifacts — benchmark noise polluting search
4. Report stats before/after

Usage:
    python scripts/cleanup_memory.py                    # dry run (report only)
    python scripts/cleanup_memory.py --apply            # apply changes
    python scripts/cleanup_memory.py --apply --backup   # backup first, then apply
    python scripts/cleanup_memory.py --db data/blipshell.db  # custom DB path
"""

import argparse
import asyncio
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project backfill — tag sessions that contain BlipShell work
# ---------------------------------------------------------------------------

async def backfill_project_tags(db, dry_run=True) -> dict:
    """Tag sessions whose content mentions BlipShell code/terms."""

    # BlipShell-specific terms that indicate project work (not just casual mention)
    code_terms = [
        'blipshell/', 'blipshell\\',  # file paths
        'agent_chat', 'agent_session', 'agent_project',
        'sqlite_store', 'chroma_store', 'chat_loop',
        'executor.py', 'search.py', 'processor.py', 'worker.py',
        'config.yaml', 'ToolRegistry', 'TaskExecutor',
        'MemorySearch', 'MemoryProcessor', 'ChatLoop',
        'ChromaStore', 'OllamaGate', 'ollama_gate',
        'LoopConfig', 'LoopResult', 'EndpointManager',
        'nightly.py', 'benchmark_', 'test_executor',
    ]

    # Find sessions with >= 3 mentions of BlipShell or any code term
    cursor = await db.execute("""
        SELECT m.session_id, COUNT(*) as hits
        FROM memories m
        JOIN sessions s ON s.id = m.session_id
        WHERE s.project IS NULL
          AND (m.content LIKE '%blipshell%' OR m.content LIKE '%BlipShell%')
        GROUP BY m.session_id
        HAVING hits >= 3
    """)
    blipshell_sessions = {r[0]: r[1] for r in await cursor.fetchall()}

    # Also find sessions with code-specific terms (even if "blipshell" isn't mentioned)
    for term in code_terms:
        cursor = await db.execute("""
            SELECT DISTINCT m.session_id
            FROM memories m
            JOIN sessions s ON s.id = m.session_id
            WHERE s.project IS NULL
              AND m.content LIKE ?
        """, (f'%{term}%',))
        for r in await cursor.fetchall():
            sid = r[0]
            blipshell_sessions[sid] = blipshell_sessions.get(sid, 0) + 1

    # Filter to sessions with strong signal (>= 3 total hits across all terms)
    to_tag = {sid: hits for sid, hits in blipshell_sessions.items() if hits >= 3}

    if not dry_run and to_tag:
        for sid in to_tag:
            await db.execute(
                "UPDATE sessions SET project = 'blipshell' WHERE id = ? AND project IS NULL",
                (sid,),
            )
        await db.commit()

    # Count memories now covered
    if to_tag:
        placeholders = ','.join('?' * len(to_tag))
        cursor = await db.execute(
            f"SELECT COUNT(*) as c FROM memories WHERE session_id IN ({placeholders})",
            list(to_tag.keys()),
        )
        mem_count = (await cursor.fetchone())[0]
    else:
        mem_count = 0

    return {
        "sessions_tagged": len(to_tag),
        "memories_covered": mem_count,
        "action": "applied" if not dry_run else "dry_run",
    }


# ---------------------------------------------------------------------------
# Duplicate cleanup — archive older copies of identical content
# ---------------------------------------------------------------------------

async def cleanup_duplicates(db, dry_run=True) -> dict:
    """Archive duplicate memories, keeping the newest copy."""

    cursor = await db.execute("""
        SELECT content, GROUP_CONCAT(id) as ids, COUNT(*) as cnt
        FROM memories
        WHERE is_archived = 0
        GROUP BY content
        HAVING cnt > 1
        ORDER BY cnt DESC
    """)
    rows = await cursor.fetchall()

    total_archived = 0
    for r in rows:
        ids = [int(x) for x in r[1].split(',')]
        # Keep the newest (highest ID), archive the rest
        ids.sort()
        to_archive = ids[:-1]  # all except last (newest)
        total_archived += len(to_archive)

        if not dry_run:
            placeholders = ','.join('?' * len(to_archive))
            await db.execute(
                f"UPDATE memories SET is_archived = 1 WHERE id IN ({placeholders})",
                to_archive,
            )

    if not dry_run and total_archived > 0:
        await db.commit()

    return {
        "duplicate_groups": len(rows),
        "memories_archived": total_archived,
        "action": "applied" if not dry_run else "dry_run",
    }


# ---------------------------------------------------------------------------
# Stress test artifact cleanup
# ---------------------------------------------------------------------------

async def cleanup_stress_tests(db, dry_run=True) -> dict:
    """Archive memories that are stress test / benchmark artifacts."""

    # Find memories with stress test content
    cursor = await db.execute("""
        SELECT COUNT(*) as c FROM memories
        WHERE is_archived = 0
          AND (content LIKE '%stress_test_%'
               OR content LIKE '%canned_test_%'
               OR content LIKE '%simple_chat_test%'
               OR content LIKE '%stress_output%'
               OR content LIKE 'Create a Python file called stress_%'
               OR content LIKE 'Create a file called stress_%'
               OR content LIKE 'Run ''python --version'' using the shell command tool%'
               OR content LIKE '%stress_test_consistent%'
               OR content LIKE '%stress_test_buggy%'
               OR content LIKE '%stress_test_calculator%')
    """)
    count = (await cursor.fetchone())[0]

    if not dry_run and count > 0:
        await db.execute("""
            UPDATE memories SET is_archived = 1
            WHERE is_archived = 0
              AND (content LIKE '%stress_test_%'
                   OR content LIKE '%canned_test_%'
                   OR content LIKE '%simple_chat_test%'
                   OR content LIKE '%stress_output%'
                   OR content LIKE 'Create a Python file called stress_%'
                   OR content LIKE 'Create a file called stress_%'
                   OR content LIKE 'Run ''python --version'' using the shell command tool%'
                   OR content LIKE '%stress_test_consistent%'
                   OR content LIKE '%stress_test_buggy%'
                   OR content LIKE '%stress_test_calculator%')
        """)
        await db.commit()

    return {
        "memories_archived": count,
        "action": "applied" if not dry_run else "dry_run",
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

async def report(db):
    """Print database stats."""
    stats = {}
    for label, sql in [
        ("total_memories", "SELECT COUNT(*) FROM memories"),
        ("archived", "SELECT COUNT(*) FROM memories WHERE is_archived = 1"),
        ("active", "SELECT COUNT(*) FROM memories WHERE is_archived = 0"),
        ("low_importance", "SELECT COUNT(*) FROM memories WHERE importance < 0.25 AND is_archived = 0"),
        ("no_summary", "SELECT COUNT(*) FROM memories WHERE (summary IS NULL OR summary = '') AND is_archived = 0"),
        ("total_sessions", "SELECT COUNT(*) FROM sessions"),
        ("blipshell_sessions", "SELECT COUNT(*) FROM sessions WHERE project = 'blipshell'"),
        ("blipshell_memories", "SELECT COUNT(*) FROM memories m JOIN sessions s ON s.id = m.session_id WHERE s.project = 'blipshell' AND m.is_archived = 0"),
        ("core_memories", "SELECT COUNT(*) FROM core_memories WHERE is_active = 1"),
        ("lessons", "SELECT COUNT(*) FROM lessons"),
        ("duplicates", "SELECT SUM(cnt-1) FROM (SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0 GROUP BY content HAVING cnt > 1)"),
    ]:
        cursor = await db.execute(sql)
        r = await cursor.fetchone()
        stats[label] = r[0] or 0

    print("\n=== Database Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    import aiosqlite

    parser = argparse.ArgumentParser(description="Memory cleanup + project backfill")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite database path")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--backup", action="store_true", help="Backup DB before applying")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    if args.backup and args.apply:
        backup_path = args.db + f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(args.db, backup_path)
        print(f"Backup saved to {backup_path}")

    db = await aiosqlite.connect(args.db)
    db.row_factory = aiosqlite.Row

    mode = "APPLYING" if args.apply else "DRY RUN"
    print(f"\n{'='*50}")
    print(f"Memory Cleanup ({mode})")
    print(f"{'='*50}")

    # Before stats
    print("\n--- BEFORE ---")
    await report(db)

    # Run cleanup steps
    print(f"\n--- Step 1: Backfill project tags ---")
    r1 = await backfill_project_tags(db, dry_run=not args.apply)
    print(f"  Sessions to tag as blipshell: {r1['sessions_tagged']}")
    print(f"  Memories covered: {r1['memories_covered']}")

    print(f"\n--- Step 2: Archive duplicates ---")
    r2 = await cleanup_duplicates(db, dry_run=not args.apply)
    print(f"  Duplicate groups: {r2['duplicate_groups']}")
    print(f"  Memories to archive: {r2['memories_archived']}")

    print(f"\n--- Step 3: Archive stress test artifacts ---")
    r3 = await cleanup_stress_tests(db, dry_run=not args.apply)
    print(f"  Memories to archive: {r3['memories_archived']}")

    if args.apply:
        print("\n--- AFTER ---")
        await report(db)

    await db.close()

    print(f"\n{'='*50}")
    if not args.apply:
        print("DRY RUN — no changes made. Use --apply to execute.")
    else:
        print("DONE — changes applied.")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
