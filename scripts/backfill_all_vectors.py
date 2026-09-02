"""Run the vector backfill to completion, every collection, in one sitting.

The nightly `backfill_vectors` job is deliberately capped (500 per collection
per run) so it fits the job time budget; this standalone runner loops until
nothing is missing — for draining a known backlog in one go, e.g. the 1,720
unembedded session reflections found by the 2026-09-02 audit.

Runs on the box with local Ollama (embedding model must be reachable):
    python -m scripts.backfill_all_vectors
    python -m scripts.backfill_all_vectors --db PATH   # a specific database
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.memory.vector_store import _SOURCE_TABLES, VectorStore  # noqa: E402

BATCH = 500


def drain(vectors: VectorStore, on_status=print) -> dict:
    """Backfill every collection until empty. Returns totals per collection.

    Stops a collection early if a full batch fails outright (embedder down) —
    looping on a dead backend would spin forever.
    """
    totals: dict[str, dict] = {}
    for collection in _SOURCE_TABLES:
        done = failed = batches = 0
        while True:
            stats = vectors.backfill_missing_vectors(collection, limit=BATCH)
            done += stats.get("succeeded", 0)
            failed += stats.get("failed", 0)
            batches += 1
            if stats.get("processed", 0) == 0:
                break
            if stats.get("succeeded", 0) == 0:
                on_status(f"  {collection}: batch produced no successes — "
                          f"embedder down? stopping this collection")
                break
            on_status(f"  {collection}: +{stats['succeeded']} "
                      f"({done} total)")
        totals[collection] = {"embedded": done, "failed": failed,
                              "batches": batches}
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", help="database path (default: config.yaml's)")
    args = parser.parse_args()

    from blipshell.core.config import ConfigManager
    from blipshell.models.config import get_ollama_url

    config = ConfigManager().load()
    db_path = args.db or config.database.path
    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1

    vectors = VectorStore(
        db_path=db_path,
        embedding_model=config.models.embedding,
        ollama_url=get_ollama_url(config.endpoints),
        embedding_dim=config.database.embedding_dimensions,
    )
    vectors.initialize()
    start = time.monotonic()
    totals = drain(vectors)
    vectors.close()

    print(f"\ndone in {time.monotonic() - start:.0f}s:")
    grand = 0
    for collection, t in totals.items():
        if t["embedded"] or t["failed"]:
            print(f"  {collection}: {t['embedded']} embedded"
                  + (f", {t['failed']} FAILED" if t["failed"] else ""))
        grand += t["embedded"]
    print(f"  total: {grand}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
