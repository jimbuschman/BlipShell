"""Embedding / retrieval-quality benchmark on the current sqlite-vec stack.

The old scripts/benchmark_embeds.py is dead — it queried ChromaDB, removed in the
sqlite-vec migration. This rebuilds the measurement honestly:

  * Load the curated query->expected-memory labels from data/search_ground_truth.json.
  * Build a focused candidate set = (every expected memory across all queries) +
    N random distractors, pulled read-only from blipshell.db.
  * Embed that set AND the queries with the *candidate* embedding model (both in the
    same vector space — the only valid way to A/B an embedder), in memory.
  * Rank by cosine and score Precision@5 / Recall@10 / MRR against the labels.

In-memory + read-only: it never writes vectors back to the production DB. Embedding
models are Ollama-local, so this runs against the candidate via the Ollama client.

Metric functions are pure (ranked ids + expected set in, score out) and unit-tested.
"""

import json
import logging
import math
import random
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_GROUND_TRUTH = "data/search_ground_truth.json"
EMBED_CHUNK = 32
MAX_EMBED_CHARS = 6000


# ---------------------------------------------------------------------------
# Pure metrics
# ---------------------------------------------------------------------------

def precision_at_k(ranked_ids: list[int], expected: set[int], k: int = 5) -> float:
    if not expected:
        return 0.0
    top = ranked_ids[:k]
    denom = min(k, len(expected))
    return len(set(top) & expected) / denom if denom else 0.0


def recall_at_k(ranked_ids: list[int], expected: set[int], k: int = 10) -> float:
    if not expected:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & expected) / len(expected)


def mrr(ranked_ids: list[int], expected: set[int]) -> float:
    for i, rid in enumerate(ranked_ids):
        if rid in expected:
            return 1.0 / (i + 1)
    return 0.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def aggregate_retrieval(per_query: list[dict]) -> Optional[dict]:
    """Mean P@5/R@10/MRR + a blended 0-1 headline over scored queries."""
    if not per_query:
        return None
    n = len(per_query)
    p5 = sum(q["p_at_5"] for q in per_query) / n
    r10 = sum(q["r_at_10"] for q in per_query) / n
    m = sum(q["mrr"] for q in per_query) / n
    return {
        "p_at_5": round(p5, 4),
        "r_at_10": round(r10, 4),
        "mrr": round(m, 4),
        "headline": round((p5 + r10 + m) / 3, 4),
        "n_queries": n,
    }


# ---------------------------------------------------------------------------
# Data loading (read-only)
# ---------------------------------------------------------------------------

def load_ground_truth(path: str = DEFAULT_GROUND_TRUTH) -> dict[str, list[int]]:
    """Load query -> expected_ids, dropping queries with no labels (data gaps)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ground-truth file not found: {path}")
    raw = json.loads(p.read_text())
    return {q: v["expected_ids"] for q, v in raw.items() if v.get("expected_ids")}


def _load_memory_texts(db_path: str, ids: list[int], distractors: int) -> dict[int, str]:
    """Read summary/content for the given ids + N random distractor memories."""
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        texts: dict[int, str] = {}
        if ids:
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT id, COALESCE(NULLIF(summary,''), content) AS text "
                f"FROM memories WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            for r in rows:
                if r["text"]:
                    texts[r["id"]] = r["text"]
        if distractors > 0:
            placeholders = ",".join("?" for _ in ids) if ids else "0"
            rows = conn.execute(
                f"SELECT id, COALESCE(NULLIF(summary,''), content) AS text "
                f"FROM memories WHERE id NOT IN ({placeholders}) "
                f"AND COALESCE(NULLIF(summary,''), content) IS NOT NULL "
                f"ORDER BY RANDOM() LIMIT ?",
                (*ids, distractors) if ids else (distractors,),
            ).fetchall()
            for r in rows:
                if r["text"]:
                    texts[r["id"]] = r["text"]
        return texts
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Async runner (Ollama embedding)
# ---------------------------------------------------------------------------

async def run_embedding_benchmark(
    *,
    model: str,
    ollama_url: str,
    db_path: str,
    ground_truth_path: str = DEFAULT_GROUND_TRUTH,
    distractors: int = 300,
    seed: int = 1234,
) -> Optional[dict]:
    """Score a candidate embedding model. Returns aggregate dict or None on failure."""
    import asyncio

    try:
        gt = load_ground_truth(ground_truth_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("embedding: ground truth unavailable: %s", e)
        return None
    if not gt:
        logger.warning("embedding: no labeled queries with expected_ids")
        return None

    expected_ids = sorted({i for ids in gt.values() for i in ids})
    try:
        texts = _load_memory_texts(db_path, expected_ids, distractors)
    except Exception as e:  # noqa: BLE001
        logger.warning("embedding: could not load memory texts from %s: %s", db_path, e)
        return None
    if not texts:
        logger.warning("embedding: no memory texts loaded (empty DB?)")
        return None

    rng = random.Random(seed)  # deterministic distractor order doesn't matter; reserved
    _ = rng  # (RANDOM() in SQL drives sampling; kept for future in-python sampling)

    mem_ids = list(texts.keys())
    mem_texts = [texts[i][:MAX_EMBED_CHARS] for i in mem_ids]
    queries = list(gt.keys())

    def _embed_all() -> Optional[tuple[list[list[float]], list[list[float]]]]:
        """Blocking Ollama embedding of memories + queries (run in executor)."""
        import ollama
        client = ollama.Client(host=ollama_url)

        def embed_batch(items: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            for i in range(0, len(items), EMBED_CHUNK):
                resp = client.embed(model=model, input=items[i:i + EMBED_CHUNK])
                out.extend(resp["embeddings"])
            return out

        return embed_batch(mem_texts), embed_batch(queries)

    try:
        loop = asyncio.get_event_loop()
        mem_vecs, query_vecs = await loop.run_in_executor(None, _embed_all)
    except Exception as e:  # noqa: BLE001
        logger.warning("embedding: candidate model '%s' embed failed: %s", model, e)
        return None

    per_query = []
    for qi, query in enumerate(queries):
        qvec = query_vecs[qi]
        scored = sorted(
            ((mem_ids[mi], cosine(qvec, mem_vecs[mi])) for mi in range(len(mem_ids))),
            key=lambda x: x[1], reverse=True,
        )
        ranked = [mid for mid, _ in scored]
        # Restrict expected to ids actually present in the candidate set.
        expected = set(gt[query]) & set(mem_ids)
        if not expected:
            continue
        per_query.append({
            "query": query,
            "p_at_5": precision_at_k(ranked, expected, 5),
            "r_at_10": recall_at_k(ranked, expected, 10),
            "mrr": mrr(ranked, expected),
        })

    return aggregate_retrieval(per_query)
