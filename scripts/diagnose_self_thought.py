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
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.config import ConfigManager
from blipshell.core.self_reflection import SelfThoughtStore, _cosine
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter
from blipshell.memory.reranker import Reranker
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


class _MemStore:
    """In-memory app_metadata stand-in so the diagnostic never touches real data."""

    def __init__(self):
        self.data = {}

    async def get_metadata(self, key):
        return self.data.get(key)

    async def set_metadata(self, key, value):
        self.data[key] = value


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

    # --- 2. RERANKER (isolated): obvious match vs obvious non-match ---
    print("[2] RERANKER SANITY (isolated, not self-thought)")
    rr = Reranker(
        ollama_url=ollama_url,
        model=config.memory.reranker_model,
        instruction=config.memory.reranker_instruction or None,
    )
    try:
        s_match = await rr.score_pair("What is Python?", "Python is a high-level programming language.")
        s_nonmatch = await rr.score_pair("What is Python?", "The Eiffel Tower is in Paris.")
        line("score(obvious match)", round(s_match, 4))
        line("score(obvious non-match)", round(s_nonmatch, 4))
        if s_match == 0.5 and s_nonmatch == 0.5:
            line("VERDICT", "reranker returns NEUTRAL 0.5 -> model not answering "
                            "(pull it? `ollama pull " + config.memory.reranker_model + "`)")
        elif s_match <= s_nonmatch:
            line("VERDICT", "reranker not discriminating (match <= non-match) -> "
                            "prompt/format mismatch for this model")
        else:
            line("VERDICT", "reranker works (match > non-match)")
    except Exception as e:
        line("RERANKER CALL FAILED", repr(e))
    await rr.close()
    print()

    # --- 3. FULL PATH: the exact call the chat turn makes ---
    print("[3] FULL search_self_thoughts() PATH")

    async def embed_fn(text):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, vectors.embed_text, text)

    # Throwaway in-memory backing so the diagnostic never touches real data.
    store = SelfThoughtStore(_MemStore(), max_keep=50, embed_fn=embed_fn)
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

    if hasattr(sqlite, "close"):
        try:
            await sqlite.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
