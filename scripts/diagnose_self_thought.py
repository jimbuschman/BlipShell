"""Diagnose why a self-thought does (or doesn't) resurface as standing context.

Prints ONLY the decisive numbers — no verbose log noise — for each stage of
the two-stage relevance filter, plus an isolated reranker sanity check on an
obvious match/non-match so we learn whether the reranker itself works.

Usage:
    python scripts/diagnose_self_thought.py
    python scripts/diagnose_self_thought.py --config path/to/config.yaml
"""

import argparse
import asyncio
import os
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.config import ConfigManager
from blipshell.core.self_reflection import SelfThoughtStore, _cosine
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter
from blipshell.memory.search import MemorySearch
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.vector_store import VectorStore
from blipshell.models.config import get_ollama_url

logging.basicConfig(level=logging.WARNING)  # keep the output clean

SEED = ("I keep wondering whether the modular cubes should express emotion "
        "through motion rather than through color.")
QUERY = "Should the modular cubes express emotion through motion rather than color?"
OFFTOPIC = "What's a good recipe for sourdough bread?"


def line(label, value):
    print(f"  {label:<34} {value}")


async def _throwaway_store():
    """A real SQLiteStore on a temp file, so the diagnostic still never
    touches real data.

    Used to be a dict faking get/set_metadata — but thoughts moved from the
    app_metadata JSON blob to a real table (2026-08-07), so SelfThoughtStore
    now calls get_self_thoughts/add_self_thought/... and a metadata-only fake
    crashes on first use. And it only crashes HERE, on the Ollama PC, where
    the dev-box suite can't see it.
    """
    import tempfile
    from blipshell.memory.sqlite_store import SQLiteStore

    path = os.path.join(tempfile.mkdtemp(prefix="blipshell_diag_"), "diag.db")
    store = SQLiteStore(path)
    await store.initialize()
    return store


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default=None)
    args = ap.parse_args()

    config = ConfigManager(args.config).load()
    ollama_url = get_ollama_url(config.endpoints)

    print("=" * 70)
    print("  SELF-THOUGHT INJECTION DIAGNOSTIC")
    print("=" * 70)
    line("reranker_enabled", config.memory.reranker_enabled)
    line("reranker_model", config.memory.reranker_model)
    line("embedding_model", config.models.embedding)
    line("inject_cosine_floor", config.reflection.inject_cosine_floor)
    line("inject_rerank_floor", config.reflection.inject_rerank_floor)
    line("ollama_url", ollama_url)
    print()

    # --- build the same components the agent uses ---
    vectors = VectorStore(
        db_path=config.database.path,
        embedding_model=config.models.embedding,
        ollama_url=ollama_url,
        embedding_dim=config.database.embedding_dimensions,
    )
    vectors.initialize()
    sqlite = SQLiteStore(config.database.path)
    await sqlite.initialize()
    endpoint_mgr = EndpointManager(config.endpoints, config.llm)
    router = LLMRouter(config.models, endpoint_mgr)
    search = MemorySearch(sqlite, vectors, router, config=config.memory, ollama_url=ollama_url)

    # --- 1. EMBEDDING: does the model produce vectors at all? ---
    print("[1] EMBEDDING")
    try:
        qv = vectors.embed_text(QUERY)
        sv = vectors.embed_text(SEED)
        line("query vector dim", len(qv))
        line("cosine(query, seed)", round(_cosine(qv, sv), 4))
        line("vs inject_cosine_floor", "PASS" if _cosine(qv, sv) >= config.reflection.inject_cosine_floor else "DROP")
    except Exception as e:
        line("EMBEDDING FAILED", repr(e))
        return
    print()

    # --- 2. RELEVANCE JUDGE (isolated): obvious match vs obvious non-match ---
    # This is the stage-2 gate: a local reasoning-model yes/no. Should say yes
    # to a clear match and no to a clear non-match.
    print("[2] RELEVANCE JUDGE SANITY (isolated)")
    try:
        v_match = await search._judge_relevance(
            "Should the cubes show emotion through motion?",
            "I keep wondering whether the cubes should express emotion through motion.",
        )
        v_nonmatch = await search._judge_relevance(
            "What's a good recipe for sourdough bread?",
            "I keep wondering whether the cubes should express emotion through motion.",
        )
        line("judge(obvious match)", v_match)
        line("judge(obvious non-match)", v_nonmatch)
        if v_match >= 1.0 and v_nonmatch <= 0.0:
            line("VERDICT", "judge works (yes to match, no to non-match)")
        elif v_match <= v_nonmatch:
            line("VERDICT", "judge not discriminating -> check the reasoning model / prompt")
        else:
            line("VERDICT", f"judge partial (match={v_match}, non-match={v_nonmatch})")
    except Exception as e:
        line("JUDGE CALL FAILED", repr(e))
    print()

    # --- 3. FULL PATH: the exact call the chat turn makes ---
    print("[3] FULL search_self_thoughts() PATH")

    async def embed_fn(text):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, vectors.embed_text, text)

    # Throwaway temp-file backing so the diagnostic never touches real data.
    diag_sqlite = await _throwaway_store()
    store = SelfThoughtStore(diag_sqlite, max_keep=50, embed_fn=embed_fn)
    await store.add(SEED)

    for label, q in [("ON-TOPIC", QUERY), ("OFF-TOPIC", OFFTOPIC)]:
        out = await search.search_self_thoughts(
            q, store,
            cosine_floor=config.reflection.inject_cosine_floor,
            rerank_floor=config.reflection.inject_rerank_floor,
            max_inject=config.reflection.inject_max,
            prefilter_k=config.reflection.inject_prefilter_k,
        )
        line(f"{label} -> injected", out if out else "(nothing)")

    for s in (sqlite, diag_sqlite):
        try:
            await s.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
