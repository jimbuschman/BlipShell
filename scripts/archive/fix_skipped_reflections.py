"""Delete wrongly-skipped session reflections so they get regenerated.

Finds reflections marked 'skipped' where the session actually has 10+
memories (enough conversation data for a real reflection). Deletes them
so the nightly job or backfill script picks them up.

Usage:
    python scripts/fix_skipped_reflections.py [--dry-run] [--min-memories 10]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

console = Console()


async def main():
    parser = argparse.ArgumentParser(description="Fix wrongly-skipped reflections")
    parser.add_argument("--db", default="data/blipshell.db")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--min-memories", type=int, default=10,
                        help="Min memories to consider a session wrongly skipped (default: 10)")
    args = parser.parse_args()

    from blipshell.memory.sqlite_store import SQLiteStore

    sqlite = SQLiteStore(args.db)
    await sqlite.initialize()

    cursor = await sqlite._db.execute("""
        SELECT sr.id, sr.session_id, s.message_count, s.title,
            (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id) as actual_memories
        FROM session_reflections sr
        JOIN sessions s ON sr.session_id = s.id
        WHERE sr.effectiveness = 'skipped'
        AND (SELECT COUNT(*) FROM memories m WHERE m.session_id = s.id) >= ?
        ORDER BY sr.session_id DESC
    """, (args.min_memories,))
    rows = await cursor.fetchall()

    if not rows:
        console.print("[green]No wrongly-skipped reflections found.[/green]")
        await sqlite.close()
        return

    console.print(f"Found [bold]{len(rows)}[/bold] skipped reflections with {args.min_memories}+ memories:")
    for r in rows:
        title = (r["title"] or "")[:50]
        console.print(f"  session #{r['session_id']} ({r['actual_memories']} memories) \"{title}\"")

    if args.dry_run:
        console.print(f"\n[yellow]Dry run — would delete {len(rows)} reflections.[/yellow]")
    else:
        ids = [r["id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        await sqlite._db.execute(
            f"DELETE FROM session_reflections WHERE id IN ({placeholders})", ids,
        )
        await sqlite._db.commit()
        console.print(f"\n[green]Deleted {len(rows)} skipped reflections.[/green]")
        console.print("Run the nightly or backfill script to regenerate them.")

    await sqlite.close()


if __name__ == "__main__":
    asyncio.run(main())
