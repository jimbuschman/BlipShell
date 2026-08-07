"""Measure the near-duplicate similarity distribution to calibrate the threshold.

The full sweep archived 59 memories out of ~32,000 checked (0.18%) at
`consolidation_similarity_threshold: 0.92`. That is either an honest reading of
a corpus without many duplicates, or a threshold set too high after 0.85 was
seen merging unrelated content. Guessing between those from the merge count
alone is how the threshold got mis-set in the first place.

This samples memories, takes each one's nearest stored neighbour, and reports
where those similarities actually fall -- so the threshold can be argued from
the corpus rather than from a single pass's merge rate.

Makes NO LLM or embedding calls: every vector is already stored, so this is
pure sqlite-vec KNN and runs anywhere the DB does.

Merges nothing and updates no memory. (The vector store opens the DB
read-write because loading sqlite-vec and ensuring its vec0 tables requires
it; the memory text is read through a separate read-only handle.)

Usage:
    python scripts/consolidation_calibrate.py
    python scripts/consolidation_calibrate.py --sample 3000
    python scripts/consolidation_calibrate.py --show-pairs 15
    python scripts/consolidation_calibrate.py --band 0.85 0.92 --show-pairs 25
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from scripts.consolidation_status import _scalar

console = Console()

# Buckets chosen around the two thresholds that have actually been run: 0.85
# (merged unrelated content) and 0.92 (current).
BUCKETS = [
    (0.99, 1.01, "0.99+  near-identical"),
    (0.96, 0.99, "0.96 - 0.99"),
    (0.92, 0.96, "0.92 - 0.96  (merges today)"),
    (0.88, 0.92, "0.88 - 0.92  (0.92 blocks these)"),
    (0.85, 0.88, "0.85 - 0.88  (old threshold merged these)"),
    (0.80, 0.85, "0.80 - 0.85"),
    (0.00, 0.80, "below 0.80"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/blipshell.db")
    ap.add_argument("--sample", type=int, default=2000,
                    help="how many active memories to probe (default 2000)")
    ap.add_argument("--show-pairs", type=int, default=10,
                    help="print this many example pairs from the band")
    ap.add_argument("--band", type=float, nargs=2, default=(0.88, 0.92),
                    metavar=("LO", "HI"),
                    help="similarity band to draw examples from")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        console.print(f"[red]No such database: {db_path}[/red]")
        return 1

    # Imported late so --help works without the sqlite-vec extension present.
    from blipshell.memory.vector_store import VectorStore

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Sample at random, NOT `ORDER BY id LIMIT n`: ordering by id takes the
    # oldest memories, and a threshold read off the oldest slice of the corpus
    # is not a threshold for the corpus.
    #
    # (This is a separate matter from the empty first run, which was a k=1 bug
    # in find_neighbors returning only the query row — see its docstring.)
    rows = conn.execute(
        "SELECT id FROM memories WHERE is_archived = 0 "
        "ORDER BY RANDOM() LIMIT ?", (args.sample,),
    ).fetchall()
    memory_ids = [r["id"] for r in rows]

    if not memory_ids:
        console.print("[yellow]No active memories to sample.[/yellow]")
        return 0

    console.print(f"Probing {len(memory_ids):,} memories (no LLM calls)...")

    store = VectorStore(str(db_path))
    store.initialize()
    try:
        neighbors = store.find_neighbors(memory_ids, k=1)
    finally:
        try:
            store.close()
        except Exception:
            pass

    tops: list[tuple[int, int, float]] = []
    for mid, hits in neighbors.items():
        if hits:
            nid, sim = hits[0]
            tops.append((mid, nid, sim))

    if not tops:
        # Distinguish "this sample had no vectors" from "the store is empty".
        # Recommending a rebuild on the strength of one unlucky sample would
        # send the user into a multi-hour Ollama run for nothing.
        total_vectors = _scalar(conn, "SELECT COUNT(*) FROM vec_memories")
        total_active = _scalar(
            conn, "SELECT COUNT(*) FROM memories WHERE is_archived = 0")
        console.print(
            f"[yellow]No neighbours for any of the {len(memory_ids):,} sampled "
            f"memories.[/yellow]"
        )
        console.print(
            f"Vector store holds {total_vectors:,} memory vectors against "
            f"{total_active:,} active memories."
        )
        if total_vectors == 0:
            console.print(
                "[red]The vector store is empty — run scripts/rebuild_vectors.py"
                "[/red]"
            )
        else:
            console.print(
                "Vectors exist, so this sample simply missed them. Re-run "
                "(the sample is random), or raise --sample."
            )
        conn.close()
        return 0

    examined = len(tops)
    not_embedded = len(memory_ids) - len(neighbors)

    t = Table(title=f"Nearest-neighbour similarity ({examined:,} memories)")
    t.add_column("Band", style="cyan")
    t.add_column("Count", justify="right")
    t.add_column("Share", justify="right")
    for lo, hi, label in BUCKETS:
        n = sum(1 for _, _, s in tops if lo <= s < hi)
        t.add_row(label, f"{n:,}", f"{100.0 * n / examined:5.2f}%")
    console.print(t)

    if not_embedded:
        console.print(
            f"[yellow]{not_embedded:,} sampled memories had no stored vector "
            f"and were skipped.[/yellow]"
        )

    at_92 = sum(1 for _, _, s in tops if s >= 0.92)
    at_85 = sum(1 for _, _, s in tops if s >= 0.85)
    console.print(
        f"\nWould merge at 0.92: {at_92:,} ({100.0 * at_92 / examined:.2f}%)   "
        f"at 0.85: {at_85:,} ({100.0 * at_85 / examined:.2f}%)"
    )
    console.print(
        "[dim]These are upper bounds: consolidation also skips already-archived "
        "neighbours and stops fan-out within a pass.[/dim]"
    )

    lo, hi = args.band
    band = sorted(
        (p for p in tops if lo <= p[2] < hi), key=lambda p: -p[2],
    )[: args.show_pairs]
    if band:
        console.print(
            f"\n[bold]Examples in {lo}-{hi} -- the judgement call.[/bold] "
            f"If these read as duplicates, 0.92 is too high; if they read as "
            f"merely related, it is right."
        )
        for mid, nid, sim in band:
            a = conn.execute(
                "SELECT summary, content FROM memories WHERE id = ?", (mid,),
            ).fetchone()
            b = conn.execute(
                "SELECT summary, content FROM memories WHERE id = ?", (nid,),
            ).fetchone()
            if not a or not b:
                continue

            def text(row):
                raw = (row["summary"] or row["content"] or "").strip()
                flat = " ".join(raw.split())[:150]
                # Windows console is cp1252; keep it ASCII-safe.
                return flat.encode("ascii", "replace").decode("ascii")

            console.print(f"\n  [cyan]{sim:.4f}[/cyan]  #{mid} vs #{nid}")
            console.print(f"    A: {text(a)}")
            console.print(f"    B: {text(b)}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
