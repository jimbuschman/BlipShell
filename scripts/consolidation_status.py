"""Read-only report on where consolidation has got to.

The per-pass table (`checked`, `merged`, `integrity_ok`) only exists in the
scrollback of the run that printed it, so after a long `--loop` there was no
way to ask "how far did it get, and did it hold the archive mandate?" short of
opening the DB by hand.

Read-only: opens SQLite in immutable mode and never writes.

Usage:
    python scripts/consolidation_status.py
    python scripts/consolidation_status.py --db data/blipshell.db
    python scripts/consolidation_status.py --json
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

DRY_CURSOR_KEY = "consolidation_dry_run_cursor"


def _scalar(conn: sqlite3.Connection, sql: str, default: int = 0) -> int:
    try:
        row = conn.execute(sql).fetchone()
        return row[0] if row and row[0] is not None else default
    except sqlite3.Error:
        return default


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Read-only connection that is actually safe against a live writer.

    mode=ro takes real read locks, so a concurrent writer (the agent) can't
    hand us a torn page. immutable=1 — the first version of this script —
    skips locking entirely on the promise that the file never changes; against
    a live WAL database that promise is false and SQLite documents the result
    as possible SQLITE_CORRUPT or silently wrong answers. immutable stays only
    as a fallback for the odd case where mode=ro can't open (stale -wal with
    no -shm), and says so out loud.
    """
    try:
        return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        console.print(
            "[yellow]mode=ro open failed; falling back to immutable=1. "
            "Do not trust these numbers if BlipShell is writing right now.[/yellow]"
        )
        return sqlite3.connect(f"file:{db_path.as_posix()}?immutable=1", uri=True)


def collect(db_path: Path) -> dict:
    conn = _connect_readonly(db_path)
    try:
        stats = {
            "db": str(db_path),
            "memories_total": _scalar(conn, "SELECT COUNT(*) FROM memories"),
            "memories_active": _scalar(
                conn, "SELECT COUNT(*) FROM memories WHERE is_archived = 0"),
            "memories_archived": _scalar(
                conn, "SELECT COUNT(*) FROM memories WHERE is_archived = 1"),
            "checked": _scalar(
                conn,
                "SELECT COUNT(*) FROM memories WHERE consolidated_at IS NOT NULL"),
            # Progress is measured DIRECTLY (active rows that carry the mark),
            # never derived by subtraction from other counts — the subtraction
            # version broke the first time the nightly age/rank prune archived
            # an already-checked memory.
            "checked_active": _scalar(
                conn,
                "SELECT COUNT(*) FROM memories "
                "WHERE consolidated_at IS NOT NULL AND is_archived = 0"),
            "unchecked": _scalar(
                conn,
                "SELECT COUNT(*) FROM memories "
                "WHERE consolidated_at IS NULL AND is_archived = 0"),
            "entity_edges": _scalar(
                conn, "SELECT COUNT(*) FROM entity_relationships"),
            "entity_mentions": _scalar(conn, "SELECT COUNT(*) FROM entity_mentions"),
        }

        # Memories archived BY the merge specifically. Counted by the
        # merged_into provenance stamp _merge_memories writes, NOT by
        # `is_archived AND consolidated_at` — several jobs archive memories
        # (the nightly age/rank prune, write-time dedup), and once the sweep
        # has touched the whole corpus every one of their victims carries
        # consolidated_at too, so that heuristic inflates forever after the
        # first prune. LIKE rather than json_extract because a single
        # malformed metadata_json row would abort json_extract for the whole
        # query. Merges from before 2026-08-07 predate the stamp and are
        # undercounted (≤59 rows on the live corpus).
        stats["archived_by_consolidation"] = _scalar(
            conn,
            "SELECT COUNT(*) FROM memories "
            "WHERE is_archived = 1 AND metadata_json LIKE '%\"merged_into\"%'",
        )

        row = conn.execute(
            "SELECT value FROM app_metadata WHERE key = ?", (DRY_CURSOR_KEY,),
        ).fetchone()
        stats["dry_run_cursor"] = int(row[0]) if row and row[0] else 0

        # The mandate: consolidation archives, never deletes. A memory row whose
        # edges vanished would be the signature of the old cascading delete.
        stats["orphan_edges"] = _scalar(
            conn,
            "SELECT COUNT(*) FROM entity_relationships "
            "WHERE source_memory_id IS NOT NULL AND source_memory_id NOT IN "
            "(SELECT id FROM memories)",
        )
        stats["orphan_mentions"] = _scalar(
            conn,
            "SELECT COUNT(*) FROM entity_mentions "
            "WHERE memory_id NOT IN (SELECT id FROM memories)",
        )
        return stats
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/blipshell.db")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        console.print(f"[red]No such database: {db_path}[/red]")
        return 1

    s = collect(db_path)

    if args.json:
        print(json.dumps(s, indent=2))
        return 0

    # Against the ACTIVE corpus, not the total. Archived memories are never
    # consolidation candidates, so dividing by the total made a finished sweep
    # read as "75.7% done" while the summary below it said complete.
    eligible = s["memories_active"] or 1
    checked_active = s["checked_active"]
    pct = 100.0 * checked_active / eligible

    t = Table(title="Consolidation status", show_header=False)
    t.add_column("", style="cyan")
    t.add_column("", justify="right")
    t.add_row("Database", s["db"])
    t.add_row("Memories (total)", f"{s['memories_total']:,}")
    t.add_row("  active", f"{s['memories_active']:,}")
    t.add_row("  archived", f"{s['memories_archived']:,}")
    t.add_row(
        "Checked (of active)", f"{checked_active:,} / {eligible:,}  ({pct:.1f}%)",
    )
    t.add_row("Still unchecked", f"{s['unchecked']:,}")
    t.add_row("Archived as near-duplicates", f"{s['archived_by_consolidation']:,}")
    t.add_row("Dry-run cursor", f"{s['dry_run_cursor']:,}")
    t.add_row("Entity edges", f"{s['entity_edges']:,}")
    t.add_row("Entity mentions", f"{s['entity_mentions']:,}")
    console.print(t)

    orphans = s["orphan_edges"] + s["orphan_mentions"]
    if orphans:
        console.print(
            f"[red]FAIL: {s['orphan_edges']} edges and {s['orphan_mentions']} "
            f"mentions point at memories that no longer exist. Consolidation is "
            f"meant to archive, never delete.[/red]"
        )
        return 2

    console.print("[green]OK: no orphaned edges or mentions - nothing was deleted.[/green]")
    if s["unchecked"] == 0:
        console.print("[green]Full corpus has been checked.[/green]")
    else:
        console.print(
            f"{s['unchecked']:,} memories still unchecked - "
            f"run 'blipshell nightly --job consolidate --loop' to continue."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
