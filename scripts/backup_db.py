"""Backup BlipShell databases — SQLite + ChromaDB.

Uses SQLite's online backup API (safe even if blipshell is running)
and copies the ChromaDB directory.

Usage:
    python scripts/backup_db.py
    python scripts/backup_db.py --db data/blipshell.db --chroma data/chroma
    python scripts/backup_db.py --out backups/
"""

import argparse
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()


def backup_sqlite(src_path: str, dst_path: str):
    """Use SQLite's backup API for a consistent copy."""
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)
    with dst:
        src.backup(dst)
    dst.close()
    src.close()


def main():
    parser = argparse.ArgumentParser(description="Backup BlipShell databases")
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--chroma", default="data/chroma", help="ChromaDB directory")
    parser.add_argument("--out", default="backups", help="Backup output directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(args.out) / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]BlipShell Backup[/bold] → {backup_dir}")
    console.print()

    # SQLite backup (uses backup API — safe while blipshell is running)
    db_src = Path(args.db)
    if db_src.exists():
        db_dst = backup_dir / db_src.name
        console.print(f"SQLite: {db_src} → {db_dst}")
        start = time.perf_counter()
        try:
            backup_sqlite(str(db_src), str(db_dst))
            size_mb = db_dst.stat().st_size / (1024 * 1024)
            elapsed = time.perf_counter() - start
            console.print(f"  [green]OK[/green] ({size_mb:.1f} MB, {elapsed:.1f}s)")
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
    else:
        console.print(f"[yellow]SQLite not found: {db_src}[/yellow]")

    # ChromaDB backup (directory copy)
    chroma_src = Path(args.chroma)
    if chroma_src.exists():
        chroma_dst = backup_dir / chroma_src.name
        console.print(f"ChromaDB: {chroma_src} → {chroma_dst}")
        start = time.perf_counter()
        try:
            shutil.copytree(str(chroma_src), str(chroma_dst))
            # Calculate total size
            total = sum(f.stat().st_size for f in chroma_dst.rglob("*") if f.is_file())
            size_mb = total / (1024 * 1024)
            elapsed = time.perf_counter() - start
            console.print(f"  [green]OK[/green] ({size_mb:.1f} MB, {elapsed:.1f}s)")
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
    else:
        console.print(f"[yellow]ChromaDB not found: {chroma_src} (skip)[/yellow]")

    console.print()
    console.print(f"[bold green]Backup complete:[/bold green] {backup_dir}")


if __name__ == "__main__":
    main()
