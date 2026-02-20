"""Benchmark embedding models for BlipShell memory search.

Embeds a sample of real memories with each model, runs test queries,
and compares quality (similarity scores, result relevance) and speed.

Usage:
    python scripts/benchmark_embeds.py
    python scripts/benchmark_embeds.py --sample 200
    python scripts/benchmark_embeds.py --models nomic-embed-text mxbai-embed-large
"""

import argparse
import asyncio
import shutil
import tempfile
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "snowflake-arctic-embed:335m",
    "bge-m3",
]

# Max chars to send per document — some models have small context (512 tokens).
# 500 chars ≈ 125 tokens, safe for all models.
EMBED_MAX_CHARS = 500

# Real queries a user would type — mix of recall, topical, and specific
TEST_QUERIES = [
    "python performance profiling",
    "what does the user prefer for programming",
    "desk robot design",
    "Minecraft performance issues",
    "docker containers and deployment",
    "memory and conversation architecture",
    "what programming languages does the user know",
    "hardware and GPU setup",
    "debugging problems and frustrations",
    "ollama and local LLM configuration",
]


async def load_sample(db_path: str, sample_size: int) -> list[dict]:
    """Load a random sample of memories from SQLite."""
    from blipshell.memory.sqlite_store import SQLiteStore

    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()

    cursor = await sqlite._db.execute(
        """SELECT id, summary, session_id, role FROM memories
           WHERE summary IS NOT NULL AND is_archived = 0
           ORDER BY RANDOM() LIMIT ?""",
        (sample_size,),
    )
    rows = await cursor.fetchall()
    await sqlite.close()

    # Filter out empty/whitespace-only summaries (cause NaN embeddings)
    return [dict(r) for r in rows if r["summary"] and r["summary"].strip()]


def benchmark_model(
    model_name: str,
    memories: list[dict],
    ollama_url: str,
    batch_size: int,
) -> dict:
    """Embed memories and run queries with a single model. Returns results dict."""
    console.print(f"\n[bold cyan]{model_name}[/bold cyan]")

    # Create temp ChromaDB
    tmp_dir = tempfile.mkdtemp(prefix="blip_bench_")
    try:
        ef = OllamaEmbeddingFunction(url=ollama_url, model_name=model_name)
        client = chromadb.PersistentClient(
            path=tmp_dir, settings=Settings(anonymized_telemetry=False),
        )
        coll = client.get_or_create_collection(
            name="bench", embedding_function=ef, metadata={"hnsw:space": "cosine"},
        )

        # Embed in batches
        console.print(f"  Embedding {len(memories)} memories...", end="")
        embed_start = time.perf_counter()

        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            coll.upsert(
                ids=[str(m["id"]) for m in batch],
                documents=[m["summary"][:EMBED_MAX_CHARS] for m in batch],
                metadatas=[{"session_id": str(m["session_id"] or "")} for m in batch],
            )

        embed_time = time.perf_counter() - embed_start
        rate = len(memories) / embed_time
        console.print(f" {embed_time:.1f}s ({rate:.0f} mem/s)")

        # Run queries
        query_results = {}
        query_start = time.perf_counter()

        for query in TEST_QUERIES:
            results = coll.query(
                query_texts=[query], n_results=5, include=["distances", "documents"],
            )
            hits = []
            for j in range(len(results["ids"][0])):
                dist = results["distances"][0][j]
                sim = 1.0 - dist
                doc = results["documents"][0][j][:120]
                hits.append({"id": results["ids"][0][j], "sim": sim, "doc": doc})
            query_results[query] = hits

        query_time = time.perf_counter() - query_start
        qps = len(TEST_QUERIES) / query_time
        console.print(f"  Queried {len(TEST_QUERIES)} queries in {query_time:.2f}s ({qps:.0f} q/s)")

        return {
            "model": model_name,
            "embed_time": embed_time,
            "embed_rate": rate,
            "query_time": query_time,
            "qps": qps,
            "results": query_results,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def print_speed_table(results: list[dict]):
    """Print speed comparison table."""
    table = Table(title="Speed Comparison")
    table.add_column("Model")
    table.add_column("Embed Time", justify="right")
    table.add_column("Embed Rate", justify="right")
    table.add_column("Query Time", justify="right")
    table.add_column("Queries/s", justify="right")

    for r in results:
        table.add_row(
            r["model"],
            f"{r['embed_time']:.1f}s",
            f"{r['embed_rate']:.0f} mem/s",
            f"{r['query_time']:.2f}s",
            f"{r['qps']:.0f} q/s",
        )
    console.print()
    console.print(table)


def print_quality_table(results: list[dict]):
    """Print per-query similarity comparison across models."""
    for query in TEST_QUERIES:
        table = Table(title=f'"{query}"')
        table.add_column("#", justify="right", width=3)
        for r in results:
            short_name = r["model"].split(":")[0].replace("-embed", "")
            table.add_column(short_name, min_width=15)

        for rank in range(5):
            row = [str(rank + 1)]
            for r in results:
                hits = r["results"][query]
                if rank < len(hits):
                    h = hits[rank]
                    sim_color = "green" if h["sim"] >= 0.5 else "yellow" if h["sim"] >= 0.3 else "red"
                    row.append(f"[{sim_color}]{h['sim']:.3f}[/{sim_color}] {h['doc'][:50]}")
                else:
                    row.append("-")
            table.add_row(*row)

        console.print()
        console.print(table)


def print_summary_table(results: list[dict]):
    """Print average similarity scores across all queries."""
    table = Table(title="Overall Quality (avg similarity of top-5 results)")
    table.add_column("Model")
    table.add_column("Avg Top-1 Sim", justify="right")
    table.add_column("Avg Top-5 Sim", justify="right")
    table.add_column("Hits >= 0.5", justify="right")
    table.add_column("Hits >= 0.3", justify="right")

    for r in results:
        top1_sims = []
        all_sims = []
        above_50 = 0
        above_30 = 0

        for query in TEST_QUERIES:
            hits = r["results"][query]
            if hits:
                top1_sims.append(hits[0]["sim"])
            for h in hits:
                all_sims.append(h["sim"])
                if h["sim"] >= 0.5:
                    above_50 += 1
                if h["sim"] >= 0.3:
                    above_30 += 1

        avg_top1 = sum(top1_sims) / len(top1_sims) if top1_sims else 0
        avg_all = sum(all_sims) / len(all_sims) if all_sims else 0
        total = len(TEST_QUERIES) * 5

        table.add_row(
            r["model"],
            f"{avg_top1:.3f}",
            f"{avg_all:.3f}",
            f"{above_50}/{total}",
            f"{above_30}/{total}",
        )

    console.print()
    console.print(table)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--db", default="data/blipshell.db")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    console.print("[bold]Embedding Model Benchmark[/bold]")
    console.print(f"Models: {', '.join(args.models)}")
    console.print(f"Sample: {args.sample} memories")
    console.print(f"Queries: {len(TEST_QUERIES)}")

    # Load sample
    console.print("\nLoading memories from SQLite...")
    memories = await load_sample(args.db, args.sample)
    console.print(f"Loaded {len(memories)} memories")

    # Benchmark each model
    results = []
    for model in args.models:
        try:
            r = benchmark_model(model, memories, args.ollama_url, args.batch_size)
            results.append(r)
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")

    if not results:
        console.print("[red]No models completed successfully[/red]")
        return

    # Print results
    print_speed_table(results)
    print_summary_table(results)
    print_quality_table(results)


if __name__ == "__main__":
    asyncio.run(main())
