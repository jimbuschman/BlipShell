"""Backup BlipShell databases — SQLite + ChromaDB.

Uses SQLite's online backup API (safe even if blipshell is running)
and copies the ChromaDB directory.

Usage:
    python scripts/backup_db.py
    python scripts/backup_db.py --db data/blipshell.db --chroma data/chroma
    python scripts/backup_db.py --out backups/
    python scripts/backup_db.py --keep 3          # rotate, keep last 3

Programmatic use (pre-operation backup):
    from scripts.backup_db import backup_before_destructive
    backup_before_destructive("cleanup_entities")
"""

import argparse
import logging
import re
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)

# Project root (scripts/ is one level down)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "blipshell.db"
_DEFAULT_CHROMA = _PROJECT_ROOT / "data" / "chroma"
_DEFAULT_BACKUP_DIR = _PROJECT_ROOT / "backups"

# Regex to detect timestamped backup dirs: YYYYMMDD_HHMMSS or pre_<name>_YYYYMMDD_HHMMSS
_BACKUP_DIR_PATTERN = re.compile(r"^(?:pre_\w+_)?\d{8}_\d{6}$")


def backup_sqlite(src_path: str, dst_path: str):
    """Use SQLite's backup API for a consistent copy."""
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)
    with dst:
        src.backup(dst)
    dst.close()
    src.close()


def run_backup(
    db_path: str | Path = _DEFAULT_DB,
    chroma_path: str | Path = _DEFAULT_CHROMA,
    out_dir: str | Path = _DEFAULT_BACKUP_DIR,
    prefix: str = "",
    quiet: bool = False,
) -> Path | None:
    """Run a full backup and return the backup directory path.

    Args:
        db_path: SQLite database file.
        chroma_path: ChromaDB directory.
        out_dir: Parent directory for backups.
        prefix: Optional prefix for the backup dir name (e.g. "pre_cleanup_").
        quiet: Suppress console output.

    Returns:
        Path to the created backup directory, or None on total failure.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}{timestamp}" if prefix else timestamp
    backup_dir = Path(out_dir) / dir_name
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        console.print(f"[bold]BlipShell Backup[/bold] → {backup_dir}")
        console.print()

    ok = False

    # SQLite backup (uses backup API — safe while blipshell is running)
    db_src = Path(db_path)
    if db_src.exists():
        db_dst = backup_dir / db_src.name
        if not quiet:
            console.print(f"SQLite: {db_src} → {db_dst}")
        start = time.perf_counter()
        try:
            backup_sqlite(str(db_src), str(db_dst))
            size_mb = db_dst.stat().st_size / (1024 * 1024)
            elapsed = time.perf_counter() - start
            if not quiet:
                console.print(f"  [green]OK[/green] ({size_mb:.1f} MB, {elapsed:.1f}s)")
            ok = True
        except Exception as e:
            if not quiet:
                console.print(f"  [red]FAILED: {e}[/red]")
            logger.error("SQLite backup failed: %s", e)
    else:
        if not quiet:
            console.print(f"[yellow]SQLite not found: {db_src}[/yellow]")

    # ChromaDB backup (directory copy)
    chroma_src = Path(chroma_path)
    if chroma_src.exists():
        chroma_dst = backup_dir / chroma_src.name
        if not quiet:
            console.print(f"ChromaDB: {chroma_src} → {chroma_dst}")
        start = time.perf_counter()
        try:
            shutil.copytree(str(chroma_src), str(chroma_dst))
            total = sum(f.stat().st_size for f in chroma_dst.rglob("*") if f.is_file())
            size_mb = total / (1024 * 1024)
            elapsed = time.perf_counter() - start
            if not quiet:
                console.print(f"  [green]OK[/green] ({size_mb:.1f} MB, {elapsed:.1f}s)")
            ok = True
        except Exception as e:
            if not quiet:
                console.print(f"  [red]FAILED: {e}[/red]")
            logger.error("ChromaDB backup failed: %s", e)
    else:
        if not quiet:
            console.print(f"[yellow]ChromaDB not found: {chroma_src} (skip)[/yellow]")

    if not quiet:
        console.print()
        console.print(f"[bold green]Backup complete:[/bold green] {backup_dir}")

    return backup_dir if ok else None


def rotate_backups(out_dir: str | Path = _DEFAULT_BACKUP_DIR, keep: int = 5) -> list[Path]:
    """Delete old backups, keeping the most recent `keep` directories.

    Only considers directories matching the backup naming pattern.

    Returns:
        List of deleted directory paths.
    """
    out = Path(out_dir)
    if not out.exists():
        return []

    # Find all backup dirs matching the pattern
    backup_dirs = sorted(
        (d for d in out.iterdir() if d.is_dir() and _BACKUP_DIR_PATTERN.match(d.name)),
        key=lambda d: d.name,
    )

    if len(backup_dirs) <= keep:
        return []

    to_delete = backup_dirs[: len(backup_dirs) - keep]
    deleted = []
    for d in to_delete:
        try:
            shutil.rmtree(d)
            deleted.append(d)
            logger.info("Rotated old backup: %s", d)
        except Exception as e:
            logger.error("Failed to delete backup %s: %s", d, e)

    return deleted


def get_last_backup_time(out_dir: str | Path = _DEFAULT_BACKUP_DIR) -> datetime | None:
    """Get the timestamp of the most recent backup, or None if no backups exist."""
    out = Path(out_dir)
    if not out.exists():
        return None

    backup_dirs = sorted(
        (d for d in out.iterdir() if d.is_dir() and _BACKUP_DIR_PATTERN.match(d.name)),
        key=lambda d: d.name,
    )
    if not backup_dirs:
        return None

    # Extract timestamp from the directory name (last 15 chars: YYYYMMDD_HHMMSS)
    name = backup_dirs[-1].name
    ts_str = name[-15:]  # YYYYMMDD_HHMMSS
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def backup_before_destructive(
    operation_name: str,
    db_path: str | Path = _DEFAULT_DB,
    chroma_path: str | Path = _DEFAULT_CHROMA,
    out_dir: str | Path = _DEFAULT_BACKUP_DIR,
) -> Path | None:
    """Create a pre-operation backup before a destructive script runs.

    Args:
        operation_name: Short name for the operation (e.g. "cleanup_entities").
        db_path: SQLite database file.
        chroma_path: ChromaDB directory.
        out_dir: Parent directory for backups.

    Returns:
        Path to the backup directory, or None on failure.
    """
    console.print(f"[dim]Creating pre-operation backup ({operation_name})...[/dim]")
    result = run_backup(
        db_path=db_path,
        chroma_path=chroma_path,
        out_dir=out_dir,
        prefix=f"pre_{operation_name}_",
        quiet=False,
    )
    if result:
        console.print(f"[dim]Pre-op backup saved: {result}[/dim]\n")
    else:
        console.print("[yellow]Pre-op backup failed — proceeding anyway[/yellow]\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Backup BlipShell databases")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--chroma", default="data/chroma", help="ChromaDB directory")
    parser.add_argument("--out", default="backups", help="Backup output directory")
    parser.add_argument("--keep", type=int, default=None,
                        help="Rotate backups, keeping only the last N (default: no rotation)")
    args = parser.parse_args()

    run_backup(db_path=args.db, chroma_path=args.chroma, out_dir=args.out)

    if args.keep is not None:
        deleted = rotate_backups(out_dir=args.out, keep=args.keep)
        if deleted:
            console.print(f"\n[dim]Rotated {len(deleted)} old backup(s):[/dim]")
            for d in deleted:
                console.print(f"  [dim]deleted: {d.name}[/dim]")
        else:
            console.print(f"\n[dim]No rotation needed (≤{args.keep} backups exist)[/dim]")


if __name__ == "__main__":
    main()
