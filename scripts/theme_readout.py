"""Theme-diversity readout over the self-thought corpus. READ-ONLY.

Answers, with a number, the question the self-gravity step-2 gate currently
asks subjectively: is the self-layer thinking about many things or rewording
one? Uses blipshell/memory/themes.py (ported from Wisp, hand-validated there).

Quote `distinct_themes` and `domination` (no-chain). Everything is
deterministic content-word Jaccard: the same corpus scores the same on any
machine, so runs are comparable across days and across the two-PC split.

Usage (from repo root):
    python -m scripts.theme_readout                # active thoughts
    python -m scripts.theme_readout --include-archived
    python -m scripts.theme_readout --db PATH      # a specific database

Opens the database with mode=ro: this script can never create a phantom DB
or mutate the store, no matter how it is launched.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.memory.themes import family_sizes, theme_diversity  # noqa: E402


def load_thoughts(db_path: str, include_archived: bool) -> list[dict]:
    """Self-thought rows, oldest first, via a read-only connection."""
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        con.row_factory = sqlite3.Row
        where = "" if include_archived else " WHERE is_archived = 0"
        rows = con.execute(
            f"SELECT id, text, created_at, is_archived, echo_count, "
            f"surface_count FROM self_thoughts{where} ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def ascii_safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def report(rows: list[dict]) -> list[str]:
    texts = [r["text"] for r in rows]
    stats = theme_diversity(texts)
    lines = [
        f"thoughts scored:        {stats['texts']}",
        f"distinct themes:        {stats['distinct_themes']}   (single-link; the count to quote)",
        f"domination:             {stats['domination']}   (largest no-chain family share; the rumination index)",
        f"  largest family size:  {stats['largest_family']}",
        f"  single-link figure:   {stats['domination_single_link']}   (chains; inflates on a monoculture -- context only)",
        f"themes per thought:     {stats['themes_per_text']}",
    ]
    # Show the biggest no-chain family so the number has a face.
    if texts:
        family_of, sizes = family_sizes(texts)
        biggest = max(range(len(sizes)), key=lambda i: sizes[i])
        members = [i for i, f in enumerate(family_of) if f == biggest]
        if len(members) > 1:
            lines.append("")
            lines.append(f"largest family ({len(members)} thoughts), first 3:")
            for i in members[:3]:
                lines.append(f"  - {ascii_safe(texts[i][:120])}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", help="database path (default: config.yaml's)")
    parser.add_argument(
        "--include-archived", action="store_true",
        help="score archived thoughts too (folded duplicates, evictions)",
    )
    args = parser.parse_args()

    if args.db:
        db_path = args.db
    else:
        from blipshell.core.config import ConfigManager
        db_path = ConfigManager().load().database.path

    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1

    rows = load_thoughts(db_path, args.include_archived)
    scope = "active + archived" if args.include_archived else "active"
    print(f"Self-thought theme readout ({scope}) -- {db_path}")
    print()
    if not rows:
        print("No thoughts stored.")
        return 0
    for line in report(rows):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
