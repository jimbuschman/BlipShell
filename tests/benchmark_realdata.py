"""Benchmark LLM models using real imported data from the DB.

Pulls a diverse sample of actual messages from the BlipShell database
and tests ranking, importance, and summarization across multiple models.
Uses a larger dataset than the hand-picked benchmark for more
representative results.

Usage:
    python tests/benchmark_realdata.py                              # run all default models
    python tests/benchmark_realdata.py phi4:latest qwen3:14b        # run only specified models
    python tests/benchmark_realdata.py --db path/to/blipshell.db    # custom DB path
    python tests/benchmark_realdata.py --sample 50                  # custom sample size
"""

import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import ask_importance, rank_memory, summarize_memory
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
BENCHMARK_MODELS = [
    "gemma3:4b",
    "qwen2.5:14b",
    "phi4:latest",
    "qwen3:14b",
    "llama3.1:8b",
    "mistral-small3.2:latest",
]

OLLAMA_URL = "http://localhost:11434"

console = Console()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample_messages(db_path: str, sample_size: int = 50) -> list[dict]:
    """Pull a diverse sample of real messages from the DB.

    Stratified sampling: pulls from different rank levels and roles
    to get a representative mix of content types.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if total == 0:
        conn.close()
        raise ValueError(f"No memories found in {db_path}")

    # Pull a stratified sample: mix of ranks, roles, and content lengths
    # This ensures we get greetings, filler, technical content, decisions, etc.
    messages = []

    # Get some from each existing rank bucket
    for rank in [1, 2, 3, 4, 5]:
        per_rank = max(sample_size // 5, 2)
        rows = c.execute(
            """SELECT id, content, role, rank, importance, summary
               FROM memories WHERE rank = ?
               ORDER BY RANDOM() LIMIT ?""",
            (rank, per_rank),
        ).fetchall()
        for row in rows:
            messages.append(dict(row))

    # Also grab some with rank=0 (failed) to see if models can rank them
    rows = c.execute(
        """SELECT id, content, role, rank, importance, summary
           FROM memories WHERE rank = 0
           ORDER BY RANDOM() LIMIT 5""",
    ).fetchall()
    for row in rows:
        messages.append(dict(row))

    # If we don't have enough variety, fill with random
    if len(messages) < sample_size:
        existing_ids = {m["id"] for m in messages}
        remaining = sample_size - len(messages)
        rows = c.execute(
            """SELECT id, content, role, rank, importance, summary
               FROM memories ORDER BY RANDOM() LIMIT ?""",
            (remaining + len(existing_ids),),
        ).fetchall()
        for row in rows:
            if dict(row)["id"] not in existing_ids:
                messages.append(dict(row))
                if len(messages) >= sample_size:
                    break

    conn.close()

    # Deduplicate and limit
    seen = set()
    unique = []
    for m in messages:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)
    return unique[:sample_size]


def categorize_message(text: str) -> str:
    """Simple heuristic category for display."""
    text_lower = text.lower().strip()
    word_count = len(text.split())
    if word_count <= 3:
        return "short"
    if word_count <= 8:
        return "brief"
    if any(kw in text_lower for kw in ["def ", "class ", "import ", "func", "async ", "```"]):
        return "code"
    if any(kw in text_lower for kw in ["error", "bug", "fix", "issue", "fail"]):
        return "debug"
    if any(kw in text_lower for kw in ["i think", "i'll", "decided", "going to", "plan"]):
        return "decision"
    if any(kw in text_lower for kw in ["how", "what", "why", "explain", "?"]):
        return "question"
    return "general"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_router(model_name: str) -> LLMRouter:
    """Create a LLMRouter that routes ALL task types to the given model."""
    models = ModelsConfig(
        reasoning=model_name,
        tool_calling=model_name,
        coding=model_name,
        summarization=model_name,
        ranking=model_name,
        importance=model_name,
        embedding=model_name,
    )
    endpoint_cfg = EndpointConfig(
        name="benchmark",
        url=OLLAMA_URL,
        roles=["reasoning", "tool_calling", "coding", "summarization", "ranking", "importance", "embedding"],
        priority=1,
        max_concurrent=1,
    )
    endpoint_manager = EndpointManager([endpoint_cfg], LLMConfig())
    return LLMRouter(models, endpoint_manager)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def benchmark_ranking(router: LLMRouter, messages: list[dict]) -> list[dict]:
    results = []
    for msg in messages:
        sys_prompt, user_prompt = rank_memory(msg["content"])
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.RANKING, user_prompt, system=sys_prompt, think=False,
            )
            rank = MemoryProcessor._parse_rank(raw)
        except Exception as e:
            raw = f"ERROR: {e}"
            rank = -1
        elapsed = time.perf_counter() - start
        results.append({
            "id": msg["id"],
            "original_rank": msg["rank"],
            "new_rank": rank,
            "raw": raw,
            "time": round(elapsed, 2),
        })
        await asyncio.sleep(0.05)
    return results


async def benchmark_importance(router: LLMRouter, messages: list[dict]) -> list[dict]:
    results = []
    for msg in messages:
        sys_prompt, user_prompt = ask_importance(msg["content"])
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.IMPORTANCE, user_prompt, system=sys_prompt, think=False,
            )
            score = MemoryProcessor._parse_float(raw, default=0.3)
        except Exception as e:
            raw = f"ERROR: {e}"
            score = -1.0
        elapsed = time.perf_counter() - start
        results.append({
            "id": msg["id"],
            "original_importance": msg["importance"],
            "new_importance": score,
            "raw": raw,
            "time": round(elapsed, 2),
        })
        await asyncio.sleep(0.05)
    return results


async def benchmark_summarization(router: LLMRouter, messages: list[dict]) -> list[dict]:
    results = []
    for msg in messages:
        sys_prompt, user_prompt = summarize_memory(msg["content"])
        start = time.perf_counter()
        try:
            response = await router.generate(
                TaskType.SUMMARIZATION, user_prompt, system=sys_prompt, think=False,
            )
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.perf_counter() - start
        results.append({
            "id": msg["id"],
            "original_summary": msg["summary"],
            "new_summary": response,
            "time": round(elapsed, 2),
        })
        await asyncio.sleep(0.05)
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_ranking(all_results: dict, messages: list[dict]):
    """Print ranking distribution and comparison stats."""
    console.rule("[bold]Ranking Analysis")

    # Category map for messages
    msg_map = {m["id"]: m for m in messages}

    for model, data in all_results.items():
        ranks = data["ranking"]
        dist = {}
        for r in ranks:
            dist[r["new_rank"]] = dist.get(r["new_rank"], 0) + 1

        total = len(ranks)
        avg = sum(r["new_rank"] for r in ranks if r["new_rank"] > 0) / max(1, sum(1 for r in ranks if r["new_rank"] > 0))
        errors = sum(1 for r in ranks if r["new_rank"] == -1)

        console.print(f"\n[cyan]{model}[/cyan]  avg={avg:.1f}  errors={errors}")
        for rank in sorted(dist.keys()):
            count = dist[rank]
            bar = "█" * count
            pct = count / total * 100
            console.print(f"  rank {rank}: {bar} {count} ({pct:.0f}%)")


def analyze_importance(all_results: dict):
    """Print importance distribution stats."""
    console.rule("[bold]Importance Analysis")

    for model, data in all_results.items():
        scores = data["importance"]
        valid = [s["new_importance"] for s in scores if s["new_importance"] >= 0]
        if not valid:
            continue
        avg = sum(valid) / len(valid)
        low = sum(1 for v in valid if v <= 0.3)
        mid = sum(1 for v in valid if 0.3 < v <= 0.6)
        high = sum(1 for v in valid if v > 0.6)
        errors = sum(1 for s in scores if s["new_importance"] == -1)

        console.print(
            f"\n[cyan]{model}[/cyan]  avg={avg:.2f}  "
            f"low(≤0.3)={low}  mid(0.3-0.6)={mid}  high(>0.6)={high}  "
            f"errors={errors}"
        )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_ranking_table(all_results: dict, messages: list[dict]):
    models = list(all_results.keys())
    msg_map = {m["id"]: m for m in messages}

    table = Table(title="Ranking Comparison (sample)", show_lines=True)
    table.add_column("ID", width=4)
    table.add_column("Category", width=10)
    table.add_column("Content", ratio=2)
    table.add_column("Prev", justify="center", width=5)
    for model in models:
        table.add_column(model, justify="center", width=10)

    # Show a subset for readability
    sample_ids = sorted(set(r["id"] for r in all_results[models[0]]["ranking"]))[:30]

    for msg_id in sample_ids:
        msg = msg_map.get(msg_id, {})
        content = msg.get("content", "")[:80]
        category = categorize_message(msg.get("content", ""))
        prev_rank = msg.get("rank", "?")

        row = [str(msg_id), category, escape(content), str(prev_rank)]
        for model in models:
            ranking_results = all_results[model]["ranking"]
            match = next((r for r in ranking_results if r["id"] == msg_id), None)
            if match:
                rank = match["new_rank"]
                time_s = match["time"]
                row.append(f"{rank}\n[dim]{time_s}s[/dim]")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)


def print_importance_table(all_results: dict, messages: list[dict]):
    models = list(all_results.keys())
    msg_map = {m["id"]: m for m in messages}

    table = Table(title="Importance Comparison (sample)", show_lines=True)
    table.add_column("ID", width=4)
    table.add_column("Category", width=10)
    table.add_column("Content", ratio=2)
    table.add_column("Prev", justify="center", width=5)
    for model in models:
        table.add_column(model, justify="center", width=10)

    sample_ids = sorted(set(r["id"] for r in all_results[models[0]]["importance"]))[:30]

    for msg_id in sample_ids:
        msg = msg_map.get(msg_id, {})
        content = msg.get("content", "")[:80]
        category = categorize_message(msg.get("content", ""))
        prev_imp = msg.get("importance", "?")
        if isinstance(prev_imp, float):
            prev_imp = f"{prev_imp:.1f}"

        row = [str(msg_id), category, escape(content), str(prev_imp)]
        for model in models:
            imp_results = all_results[model]["importance"]
            match = next((r for r in imp_results if r["id"] == msg_id), None)
            if match:
                score = match["new_importance"]
                time_s = match["time"]
                row.append(f"{score:.1f}\n[dim]{time_s}s[/dim]")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)


def print_summary_table(all_results: dict, messages: list[dict]):
    models = list(all_results.keys())
    msg_map = {m["id"]: m for m in messages}

    table = Table(title="Summarization Comparison (sample)", show_lines=True, expand=True)
    table.add_column("Content", ratio=1)
    table.add_column("Prev Summary", ratio=1)
    for model in models:
        table.add_column(model, ratio=1)

    # Only show a handful for summarization since they're verbose
    sample_ids = sorted(set(r["id"] for r in all_results[models[0]]["summarization"]))[:10]

    for msg_id in sample_ids:
        msg = msg_map.get(msg_id, {})
        content = escape(msg.get("content", "")[:150])
        prev_summary = escape(msg.get("summary", "")[:150])

        row = [content, prev_summary]
        for model in models:
            sum_results = all_results[model]["summarization"]
            match = next((r for r in sum_results if r["id"] == msg_id), None)
            if match:
                summary = escape(match["new_summary"][:150])
                time_s = match["time"]
                row.append(f"{summary}\n[dim]({time_s}s)[/dim]")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(models: list[str], db_path: str, sample_size: int):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "benchmark_realdata_results.json"

    # Load sample messages from DB
    console.print(f"[bold]Loading {sample_size} sample messages from {db_path}...[/bold]")
    messages = load_sample_messages(db_path, sample_size)
    console.print(f"[green]Loaded {len(messages)} messages[/green]")

    # Show category distribution
    categories = {}
    for m in messages:
        cat = categorize_message(m["content"])
        categories[cat] = categories.get(cat, 0) + 1
    console.print(f"[dim]Categories: {categories}[/dim]\n")

    # Load existing results to merge with
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        if all_results:
            console.print(f"[yellow]Loaded existing results for: {', '.join(all_results.keys())}[/yellow]")

    console.print(f"[bold]Running benchmarks for: {', '.join(models)}[/bold]\n")

    for model in models:
        console.rule(f"[bold blue]Benchmarking: {model}")
        router = make_router(model)

        console.print("  [dim]Running ranking...[/dim]")
        ranking = await benchmark_ranking(router, messages)

        console.print("  [dim]Running importance...[/dim]")
        importance = await benchmark_importance(router, messages)

        console.print("  [dim]Running summarization...[/dim]")
        summarization = await benchmark_summarization(router, messages)

        all_results[model] = {
            "ranking": ranking,
            "importance": importance,
            "summarization": summarization,
        }
        console.print(f"  [green]Done with {model}[/green]\n")

    # Save results before rendering (crash-safe)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[bold]Results saved to {output_path} ({len(all_results)} models)[/bold]")

    # Print tables and analysis
    console.rule("[bold green]Results")
    print_ranking_table(all_results, messages)
    console.print()
    print_importance_table(all_results, messages)
    console.print()
    print_summary_table(all_results, messages)
    console.print()
    analyze_ranking(all_results, messages)
    console.print()
    analyze_importance(all_results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark models with real imported data")
    parser.add_argument("models", nargs="*", default=BENCHMARK_MODELS,
                        help="Models to benchmark")
    parser.add_argument("--db", default="data/blipshell.db",
                        help="Path to blipshell.db (default: data/blipshell.db)")
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of messages to sample (default: 50)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.models, args.db, args.sample))


if __name__ == "__main__":
    main()
