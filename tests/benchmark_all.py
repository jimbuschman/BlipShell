"""Unified benchmark runner — orchestrates all benchmark suites.

Single entry point to run pipeline (summarize/rank/importance/entity/contradiction),
reasoning (coding/tool-calling), and real-data benchmarks.

Usage:
    python tests/benchmark_all.py                                   # run all suites
    python tests/benchmark_all.py --suite pipeline                  # only pipeline
    python tests/benchmark_all.py --suite reasoning                 # only reasoning/coding
    python tests/benchmark_all.py --suite realdata                  # only real-data
    python tests/benchmark_all.py --models qwen3:14b glm4:latest
    python tests/benchmark_all.py --db path/to/blipshell.db        # for realdata suite
    python tests/benchmark_all.py --sample 30                       # realdata sample size
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so "from tests.*" imports work
# when this script is run directly (python tests/benchmark_all.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

console = Console()

SUITE_CHOICES = ["pipeline", "reasoning", "realdata", "validate"]


async def run_pipeline(models: list[str]) -> dict:
    """Run the pipeline benchmark suite (benchmark_models.py)."""
    from tests.benchmark_models import run_benchmark
    console.rule("[bold magenta]Suite: Pipeline (summarize/rank/importance/entity/contradiction)")
    start = time.perf_counter()
    try:
        await run_benchmark(models)
        elapsed = round(time.perf_counter() - start, 1)
        return {"status": "ok", "time": elapsed}
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 1)
        console.print(f"[red]Pipeline suite error: {e}[/red]")
        return {"status": "error", "error": str(e), "time": elapsed}


async def run_reasoning(models: list[str]) -> dict:
    """Run the reasoning benchmark suite (benchmark_reasoning.py)."""
    from tests.benchmark_reasoning import run_benchmark
    console.rule("[bold magenta]Suite: Reasoning (reasoning/coding/tool-calling)")
    start = time.perf_counter()
    try:
        await run_benchmark(models)
        elapsed = round(time.perf_counter() - start, 1)
        return {"status": "ok", "time": elapsed}
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 1)
        console.print(f"[red]Reasoning suite error: {e}[/red]")
        return {"status": "error", "error": str(e), "time": elapsed}


async def run_realdata(models: list[str], db_path: str, sample_size: int) -> dict:
    """Run the real-data benchmark suite (benchmark_realdata.py)."""
    from tests.benchmark_realdata import run_benchmark
    console.rule("[bold magenta]Suite: Real Data (ranking/importance/summarization/entity/contradiction)")
    start = time.perf_counter()
    try:
        await run_benchmark(models, db_path, sample_size)
        elapsed = round(time.perf_counter() - start, 1)
        return {"status": "ok", "time": elapsed}
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 1)
        console.print(f"[red]Real-data suite error: {e}[/red]")
        return {"status": "error", "error": str(e), "time": elapsed}


async def run_validate(config_path: str) -> dict:
    """Run prompt validation against configured production models."""
    console.rule("[bold magenta]Suite: Validate (prompt format check with production models)")
    start = time.perf_counter()
    try:
        from scripts.validate_prompts import (
            ValidationResult,
            validate_contradiction,
            validate_entity_extraction,
            validate_importance,
            validate_lesson_extraction,
            validate_rank_and_importance,
            validate_ranking,
            validate_summarization,
        )
        from blipshell.core.config import ConfigManager
        from blipshell.llm.endpoints import EndpointManager
        from blipshell.llm.router import LLMRouter

        config_mgr = ConfigManager(config_path)
        config = config_mgr.config
        endpoint_manager = EndpointManager(config.endpoints, config.llm)
        router = LLMRouter(config.models, endpoint_manager)
        result = ValidationResult()

        await validate_ranking(router, result)
        await validate_importance(router, result)
        await validate_summarization(router, result)
        await validate_entity_extraction(router, result)
        await validate_contradiction(router, result)
        await validate_lesson_extraction(router, result)
        await validate_rank_and_importance(router, result)

        result.print_report()
        elapsed = round(time.perf_counter() - start, 1)
        status = "ok" if result.all_passed else "error"
        passed = sum(1 for r in result.results if r["passed"])
        total = len(result.results)
        return {"status": status, "time": elapsed, "passed": passed, "total": total}
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 1)
        console.print(f"[red]Validate suite error: {e}[/red]")
        return {"status": "error", "error": str(e), "time": elapsed}


async def main():
    parser = argparse.ArgumentParser(description="Unified benchmark runner for all suites")
    parser.add_argument("--suite", nargs="*", choices=SUITE_CHOICES,
                        help="Suites to run (default: all)")
    parser.add_argument("--models", nargs="*",
                        help="Models to benchmark (passed to each suite)")
    parser.add_argument("--db", default="data/blipshell.db",
                        help="Path to blipshell.db for realdata suite")
    parser.add_argument("--sample", type=int, default=50,
                        help="Sample size for realdata suite (default: 50)")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path (for validate suite)")
    args = parser.parse_args()

    suites = args.suite or SUITE_CHOICES

    console.print(f"[bold]Unified Benchmark Runner[/bold]")
    console.print(f"Suites: {', '.join(suites)}")
    if args.models:
        console.print(f"Models: {', '.join(args.models)}")
    console.print()

    summary = {}
    total_start = time.perf_counter()

    if "pipeline" in suites:
        from tests.benchmark_models import BENCHMARK_MODELS as PIPELINE_MODELS
        models = args.models or PIPELINE_MODELS
        summary["pipeline"] = await run_pipeline(models)
        console.print()

    if "reasoning" in suites:
        from tests.benchmark_reasoning import BENCHMARK_MODELS as REASONING_MODELS
        models = args.models or REASONING_MODELS
        summary["reasoning"] = await run_reasoning(models)
        console.print()

    if "realdata" in suites:
        from tests.benchmark_realdata import BENCHMARK_MODELS as REALDATA_MODELS
        models = args.models or REALDATA_MODELS
        summary["realdata"] = await run_realdata(models, args.db, args.sample)
        console.print()

    if "validate" in suites:
        summary["validate"] = await run_validate(args.config)
        console.print()

    total_elapsed = round(time.perf_counter() - total_start, 1)

    # Print final summary
    console.rule("[bold green]Summary")
    ok_count = sum(1 for s in summary.values() if s["status"] == "ok")
    err_count = sum(1 for s in summary.values() if s["status"] == "error")
    for suite_name, result in summary.items():
        status = "[green]OK[/green]" if result["status"] == "ok" else f"[red]ERROR: {result.get('error', '?')}[/red]"
        console.print(f"  {suite_name}: {status} ({result['time']}s)")
    console.print(f"\n[bold]Total: {ok_count} passed, {err_count} failed, {total_elapsed}s[/bold]")

    # Save combined summary
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    summary_path = data_dir / "benchmark_summary.json"
    summary_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suites": summary,
        "total_time": total_elapsed,
        "models_override": args.models,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    console.print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
