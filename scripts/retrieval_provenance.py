"""Does retrieval surface the world, or the assistant's own exhaust? READ-ONLY.

Wisp's most portable diagnosis (2026-08-30): an agent that stores its own
utterances alongside what it perceives will retrieve its own past answers
instead of the perception that would correct them, because self-authored text
is phrased as an ANSWER TO THE QUESTION BEING ASKED and therefore sits closer
in the embedding than a raw percept ever can. Measured there: the percept
holding the answer ranked 47th at similarity 0.512, behind 27 of the agent's
own prior replies, while search fetched 10 — and the corpus was 89%
self-generated, so the problem compounds.

BlipShell stores both roles in `memories` and its retrieval is under a
"good-enough, do NOT tune preemptively" mandate (V2_PLAN) — so this script is
the MEASUREMENT that must come before any tuning. It runs the real search
pipeline (memory/search.py: rephrase, RRF fusion, every boost) over recent
user messages as probes and reports, per probe and in aggregate:

  * the share of top-k results authored by the assistant,
  * the rank of the first user-authored result,
  * against the corpus-wide baseline share (if assistant text is 40% of the
    corpus and 40% of retrieval, that is representation, not pathology).

If the retrieved assistant-share sits far above the corpus baseline, BlipShell
has the exhaust problem and the measured fix from Wisp (echo down-weighting
PLUS overfetch-then-rank — neither works alone) becomes worth porting.

Needs a live embedding model. Run it on the Ollama PC against the live corpus:
    python -m scripts.retrieval_provenance
    python -m scripts.retrieval_provenance --probes 50 --top-k 10
or drive it from the dev box over Tailscale (same pattern as `benchmark run
--url`; the DB probed is still the LOCAL one, so this only diagnoses the live
corpus if you point --db at a copy of it):
    python -m scripts.retrieval_provenance --url http://<ollama-pc>:11434
--url also disables every non-Ollama endpoint for the run, so no cloud
endpoint is touched by a diagnostic.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def summarize(probe_rows: list[dict], baseline_assistant_share: float) -> dict:
    """Aggregate per-probe stats. Pure, unit-testable.

    Each probe row: {"assistant_in_topk": int, "k": int,
    "first_user_rank": int | None} (rank is 1-based; None = no user-authored
    result in top-k).
    """
    if not probe_rows:
        return {"probes": 0}
    shares = [r["assistant_in_topk"] / r["k"] for r in probe_rows if r["k"]]
    top1_assistant = sum(
        1 for r in probe_rows if r.get("top1_role") == "assistant"
    )
    user_ranks = [r["first_user_rank"] for r in probe_rows
                  if r["first_user_rank"] is not None]
    no_user_at_all = sum(1 for r in probe_rows if r["first_user_rank"] is None)
    mean_share = sum(shares) / len(shares) if shares else 0.0
    return {
        "probes": len(probe_rows),
        "mean_assistant_share_topk": round(mean_share, 3),
        "corpus_assistant_share": round(baseline_assistant_share, 3),
        "over_representation": round(
            mean_share / baseline_assistant_share, 2
        ) if baseline_assistant_share else None,
        "top1_assistant_count": top1_assistant,
        "median_first_user_rank": (
            sorted(user_ranks)[len(user_ranks) // 2] if user_ranks else None
        ),
        "probes_with_no_user_result": no_user_at_all,
    }


async def run(probes: int, top_k: int, db_override: str | None,
              url_override: str | None = None) -> int:
    from blipshell.core.config import ConfigManager
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.memory.search import MemorySearch
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.memory.vector_store import VectorStore
    from blipshell.models.config import get_ollama_url

    config = ConfigManager().load()
    if url_override:
        # Point every Ollama endpoint at the given host and hide the cloud
        # ones — a read-only diagnostic must not spend cloud quota or ship
        # probe text offsite as a side effect.
        for ep in config.endpoints:
            if ep.provider == "ollama":
                ep.url = url_override
            else:
                ep.enabled = False
    db_path = db_override or config.database.path
    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1

    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()
    try:
        vectors = VectorStore(
            db_path=db_path,
            embedding_model=config.models.embedding,
            ollama_url=get_ollama_url(config.endpoints),
            embedding_dim=config.database.embedding_dimensions,
        )
        vectors.initialize()
        router = LLMRouter(config.models, EndpointManager(config.endpoints, config.llm))
        search = MemorySearch(
            sqlite, vectors, router,
            config=config.memory,
            ollama_url=get_ollama_url(config.endpoints),
        )

        # Corpus baseline: what share of the searchable corpus is
        # assistant-authored? Retrieval matching that share is representation;
        # retrieval far above it is the exhaust pathology.
        role_rows = await sqlite._db.execute_fetchall(
            "SELECT role, COUNT(*) FROM memories WHERE is_archived = 0 GROUP BY role"
        )
        by_role = {r[0]: r[1] for r in role_rows}
        total = sum(by_role.values()) or 1
        baseline = by_role.get("assistant", 0) / total
        print(f"Corpus: {total} active memories, "
              f"{by_role.get('assistant', 0)} assistant-authored "
              f"({baseline:.1%}), {by_role.get('user', 0)} user-authored")

        # Probes: the most recent user-authored memories, their raw content
        # used as the query — the same text the per-turn recall pool would
        # search on. The probe memory itself is excluded from scoring (a
        # memory finding itself is the query-echo artifact, not evidence).
        probe_mems = await sqlite._db.execute_fetchall(
            "SELECT id, content FROM memories "
            "WHERE role = 'user' AND is_archived = 0 "
            "AND length(content) BETWEEN 40 AND 600 "
            "ORDER BY id DESC LIMIT ?", (probes,)
        )
        if not probe_mems:
            print("No user-authored memories to probe with.")
            return 1
        print(f"Probing with {len(probe_mems)} recent user messages, "
              f"top-{top_k} scored per probe...")

        probe_rows: list[dict] = []
        for pid, content in probe_mems:
            results = await search.search(content, n_results=top_k + 1)
            scored = [r for r in results if r.memory_id != pid][:top_k]
            if not scored:
                continue
            ids = [r.memory_id for r in scored]
            marks = ",".join("?" for _ in ids)
            rows = await sqlite._db.execute_fetchall(
                f"SELECT id, role FROM memories WHERE id IN ({marks})", ids,
            )
            role_of = {r[0]: r[1] for r in rows}
            roles = [role_of.get(i, "?") for i in ids]
            first_user = next(
                (idx + 1 for idx, role in enumerate(roles) if role == "user"),
                None,
            )
            probe_rows.append({
                "assistant_in_topk": sum(1 for r in roles if r == "assistant"),
                "k": len(roles),
                "top1_role": roles[0],
                "first_user_rank": first_user,
            })

        stats = summarize(probe_rows, baseline)
        print()
        print("Retrieval provenance (assistant-authored = exhaust):")
        print(f"  probes scored:                 {stats['probes']}")
        print(f"  assistant share of top-{top_k}:      "
              f"{stats['mean_assistant_share_topk']:.1%}")
        print(f"  assistant share of corpus:     "
              f"{stats['corpus_assistant_share']:.1%}")
        if stats.get("over_representation") is not None:
            print(f"  over-representation:           "
                  f"{stats['over_representation']}x")
        print(f"  probes where top-1 is exhaust: "
              f"{stats['top1_assistant_count']}/{stats['probes']}")
        print(f"  median rank of first user mem: "
              f"{stats['median_first_user_rank']}")
        print(f"  probes with NO user result:    "
              f"{stats['probes_with_no_user_result']}")
        print()
        if stats.get("over_representation") and stats["over_representation"] > 1.5:
            print("VERDICT-SHAPED HINT: retrieval over-selects the assistant's own")
            print("text well beyond its corpus share. The measured fix in Wisp was")
            print("echo down-weighting PLUS overfetch-then-rank; neither alone.")
        else:
            print("Retrieval roughly tracks the corpus mix -- no exhaust pathology")
            print("demonstrated by this probe set. Do not tune (V2_PLAN mandate).")
        return 0
    finally:
        await sqlite.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probes", type=int, default=25,
                        help="recent user messages to probe with (default 25)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="results scored per probe (default 10)")
    parser.add_argument("--db", help="database path (default: config.yaml's)")
    parser.add_argument("--url", help="Ollama URL override (e.g. over Tailscale); "
                                      "also disables all cloud endpoints")
    args = parser.parse_args()
    return asyncio.run(run(args.probes, args.top_k, args.db, args.url))


if __name__ == "__main__":
    sys.exit(main())
