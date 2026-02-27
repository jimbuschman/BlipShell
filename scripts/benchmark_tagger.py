"""Benchmark local models for tag assignment quality.

Tests precision, recall, and F1 against ground truth (memories with 3+ tags).
Use results to pick the best model for the nightly batch tagger.

Usage:
    python scripts/benchmark_tagger.py
    python scripts/benchmark_tagger.py --models qwen3:14b,qwen2.5:14b,qwen2.5:7b
    python scripts/benchmark_tagger.py --sample 50 --db data/blipshell.db
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_MODELS = ["qwen3:14b", "qwen2.5:14b", "qwen2.5:7b"]


def make_router(model_name: str, ollama_url: str):
    """Create a router that routes REASONING to a specific model."""
    from blipshell.llm.endpoints import EndpointManager
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig

    models = ModelsConfig(reasoning=model_name)
    ep = EndpointConfig(
        name="bench-local",
        url=ollama_url,
        roles=["reasoning"],
        priority=1,
        max_concurrent=1,
    )
    return LLMRouter(models, EndpointManager([ep], LLMConfig()))


def parse_tag_response(text: str, valid_tags: set[str]) -> list[str]:
    """Parse single-memory tag response (one tag per line)."""
    tags = []
    for line in text.strip().splitlines():
        line = line.strip().lower()
        # Strip bullets, dashes, numbers
        line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        if line in valid_tags:
            tags.append(line)
    return tags[:5]


async def benchmark_model(
    model_name: str,
    samples: list[dict],
    available_tags: list[str],
    ollama_url: str,
) -> dict:
    """Test a model on tag assignment. Returns metrics dict."""
    from blipshell.llm.router import TaskType

    router = make_router(model_name, ollama_url)
    valid_tags = set(available_tags)

    system_prompt = (
        "You assign tags to memories. Given a summary and available tags, "
        "assign 1-5 relevant tags.\n"
        "Output only tag names, one per line. If none fit: NONE\n\n"
        f"Available tags:\n{', '.join(sorted(available_tags))}"
    )

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_tags_assigned = 0
    total_latency = 0.0
    extra_tags_found = 0  # valid tags not in ground truth
    errors = 0

    for i, sample in enumerate(samples):
        console.print(f"  [{i + 1}/{len(samples)}] Memory {sample['id']}...", end=" ")

        t0 = time.monotonic()
        try:
            response = await router.generate(
                TaskType.REASONING,
                sample["summary"],
                system=system_prompt,
            )
            elapsed = time.monotonic() - t0
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
            errors += 1
            continue

        predicted = parse_tag_response(response, valid_tags)
        ground_truth = set(sample["tags"])
        predicted_set = set(predicted)

        # Metrics
        correct = predicted_set & ground_truth
        precision = len(correct) / len(predicted_set) if predicted_set else 0.0
        recall = len(correct) / len(ground_truth) if ground_truth else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        extra = len(predicted_set - ground_truth)

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_tags_assigned += len(predicted)
        total_latency += elapsed
        extra_tags_found += extra

        console.print(
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
            f"({len(predicted)} tags, {elapsed:.1f}s)"
        )

    n = len(samples) - errors
    if n == 0:
        return {"model": model_name, "error": "All samples failed"}

    return {
        "model": model_name,
        "samples": len(samples),
        "errors": errors,
        "avg_precision": total_precision / n,
        "avg_recall": total_recall / n,
        "avg_f1": total_f1 / n,
        "avg_tags": total_tags_assigned / n,
        "avg_latency_s": total_latency / n,
        "extra_tags": extra_tags_found,
    }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark tag assignment models")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="Comma-separated model names")
    parser.add_argument("--sample", type=int, default=30,
                        help="Number of ground-truth memories to test")
    parser.add_argument("--db", default=None, help="Path to blipshell.db")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output", default="data/benchmark_tagger_results.json")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    # Load config for DB path if not specified
    from blipshell.core.config import ConfigManager
    config_mgr = ConfigManager(args.config)
    config = config_mgr.load()
    db_path = args.db or config.database.path

    # Load ground truth
    from blipshell.memory.sqlite_store import SQLiteStore
    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()

    console.print(f"[bold]Loading {args.sample} well-tagged memories from {db_path}...[/bold]")
    samples = await sqlite.get_well_tagged_memory_sample(
        min_tags=3, limit=args.sample,
    )
    if not samples:
        console.print("[red]No well-tagged memories found (need 3+ tags).[/red]")
        return

    available_tags = await sqlite.get_all_tag_names()
    await sqlite.close()

    console.print(f"Loaded {len(samples)} samples, {len(available_tags)} available tags.\n")

    # Benchmark each model
    models = [m.strip() for m in args.models.split(",")]
    results = []

    for model in models:
        console.print(f"\n[bold cyan]Testing: {model}[/bold cyan]")
        result = await benchmark_model(model, samples, available_tags, args.ollama_url)
        results.append(result)

    # Summary table
    table = Table(title="Tag Assignment Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right", style="bold")
    table.add_column("Avg Tags", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Errors", justify="right")

    for r in results:
        if "error" in r:
            table.add_row(r["model"], "—", "—", "—", "—", "—", r["error"])
        else:
            table.add_row(
                r["model"],
                f"{r['avg_precision']:.3f}",
                f"{r['avg_recall']:.3f}",
                f"{r['avg_f1']:.3f}",
                f"{r['avg_tags']:.1f}",
                f"{r['avg_latency_s']:.1f}s",
                str(r["errors"]),
            )

    console.print()
    console.print(table)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
