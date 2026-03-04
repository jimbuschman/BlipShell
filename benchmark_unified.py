#!/usr/bin/env python3
"""Unified LLM benchmark for BlipShell — one run, one table, one JSON.

Evaluates every LLM functional role to select optimal models per task.
See benchmark-spec.md for full specification.

Usage:
    python benchmark_unified.py --dry-run --suite all --models qwen3:14b,glm4:latest
    python benchmark_unified.py --suite pipeline --models qwen3:14b,qwen2.5:14b
    python benchmark_unified.py --suite all --models qwen3:14b --sample 20
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console

from blipshell.benchmark.models import BenchmarkResult, SuiteResult, TaskScore
from blipshell.benchmark.output import (
    console,
    print_role_detail,
    print_role_table,
    save_role_results,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suite registry — maps spec suite names to lazy-loaded suite classes.
# Separate from the global blipshell.benchmark.suites registry so we don't
# break the existing `blipshell benchmark` command.
# ---------------------------------------------------------------------------

UNIFIED_SUITES: dict[str, str] = {
    "pipeline":    "blipshell.benchmark.suites.unified_pipeline:UnifiedPipelineSuite",
    "extraction":  "blipshell.benchmark.suites.extraction:ExtractionSuite",
    "synthesis":   "blipshell.benchmark.suites.synthesis:SynthesisSuite",
    "interactive": "blipshell.benchmark.suites.interactive:InteractiveSuite",
}

SUITE_ORDER = ["pipeline", "extraction", "synthesis", "interactive"]

DEFAULT_MODELS = ["qwen3:14b", "qwen2.5:14b", "glm4:latest"]


def _load_suite(key: str):
    """Lazy-load and instantiate a suite class."""
    import importlib
    module_path, class_name = UNIFIED_SUITES[key].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def _resolve_suites(suite_arg: str | None) -> list[str]:
    """Resolve --suite arg to list of suite names."""
    if not suite_arg or suite_arg == "all":
        return list(SUITE_ORDER)
    names = [s.strip() for s in suite_arg.split(",")]
    for n in names:
        if n not in UNIFIED_SUITES:
            console.print(f"[red]Unknown suite: {n!r}. "
                          f"Available: {', '.join(SUITE_ORDER)}[/red]")
            sys.exit(1)
    return names


def _resolve_models(models_arg: str | None) -> list[str]:
    """Resolve --models arg to list of model names."""
    if not models_arg:
        return list(DEFAULT_MODELS)
    return [m.strip() for m in models_arg.split(",")]


# ---------------------------------------------------------------------------
# Dry-run: generate stub results without any LLM calls
# ---------------------------------------------------------------------------

def _dry_run(models: list[str], suites: list[str]) -> BenchmarkResult:
    """Generate stub BenchmarkResult with 0.0 scores for all roles."""
    all_results: list[SuiteResult] = []

    for suite_name in suites:
        suite = _load_suite(suite_name)
        for model in models:
            # Each suite class defines its roles as task_names in its ROLES constant
            # We get them by importing the module
            import importlib
            module_path, _ = UNIFIED_SUITES[suite_name].rsplit(":", 1)
            mod = importlib.import_module(module_path)
            roles = getattr(mod, "ROLES", [])

            scores = [
                TaskScore(task_name=role, quality=0.0, speed_s=0.0, samples=0)
                for role in roles
            ]
            all_results.append(SuiteResult(
                suite_name=suite_name, model=model, scores=scores,
            ))

    return BenchmarkResult(
        suite_results=all_results,
        models=models,
        suites=suites,
        timestamp=datetime.now(timezone.utc).isoformat(),
        elapsed_s=0.0,
    )


# ---------------------------------------------------------------------------
# Mock run: canned LLM responses, exercises scoring logic without Ollama
# ---------------------------------------------------------------------------

async def _mock_run(models: list[str], suites: list[str]) -> BenchmarkResult:
    """Run suites with a mock router that returns canned responses."""
    from blipshell.benchmark.mock import MockRouter

    mock_router = MockRouter()

    def mock_router_factory(model_name: str):
        return mock_router

    def on_status(msg: str):
        console.print(f"  [dim]{msg}[/dim]")

    result = BenchmarkResult(
        models=models,
        suites=suites,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    total_start = time.monotonic()

    for suite_name in suites:
        suite = _load_suite(suite_name)

        console.print(f"\n{'-' * 60}")
        console.print(f"[bold cyan]Suite: {suite.name}[/bold cyan] — {suite.description}")
        console.print(f"{'-' * 60}")

        try:
            suite_results = await suite.run(
                models,
                router_factory=mock_router_factory,
                db_path=None,  # no DB in mock mode
                thorough=False,
                on_status=on_status,
            )
            result.suite_results.extend(suite_results)
        except Exception as e:
            logger.exception("Suite %s failed", suite_name)
            console.print(f"[red]Suite {suite_name} failed: {e}[/red]")

    result.elapsed_s = round(time.monotonic() - total_start, 1)
    return result


# ---------------------------------------------------------------------------
# Live run: execute suites with real LLM calls
# ---------------------------------------------------------------------------

async def _live_run(
    models: list[str],
    suites: list[str],
    *,
    config_path: str | None = None,
    sample: int | None = None,
    output_path: str = "data/benchmark_unified.json",
) -> BenchmarkResult:
    """Run suites with real model calls."""
    from blipshell.benchmark.shared import (
        check_model_availability,
        get_config_and_db,
        get_context_tokens,
        make_router,
    )

    config, resolved_db, ollama_url = get_config_and_db(config_path)

    console.print(f"DB: {resolved_db}")
    console.print(f"Ollama: {ollama_url}")

    # Check availability
    available, unavailable = await check_model_availability(models, ollama_url)
    if unavailable:
        console.print(f"[yellow]Unavailable (skipping): {', '.join(unavailable)}[/yellow]")
    if not available:
        console.print("[red]No models available![/red]")
        return BenchmarkResult()

    # Router factory
    def router_factory(model_name: str):
        ctx_tokens = get_context_tokens(model_name)
        kwargs = {}
        if ctx_tokens:
            kwargs["context_tokens"] = min(ctx_tokens, 32768)
        return make_router(model_name, ollama_url, **kwargs)

    # Status callback
    def on_status(msg: str):
        console.print(f"  [dim]{msg}[/dim]")

    thorough = (sample or 20) > 30

    result = BenchmarkResult(
        models=available,
        suites=suites,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    total_start = time.monotonic()

    for suite_name in suites:
        suite = _load_suite(suite_name)

        console.print(f"\n{'-' * 60}")
        console.print(f"[bold cyan]Suite: {suite.name}[/bold cyan] — {suite.description}")
        console.print(f"{'-' * 60}")

        try:
            suite_results = await suite.run(
                available,
                router_factory=router_factory,
                config=config,
                db_path=resolved_db,
                ollama_url=ollama_url,
                thorough=thorough,
                on_status=on_status,
            )
            result.suite_results.extend(suite_results)
        except Exception as e:
            logger.exception("Suite %s failed", suite_name)
            console.print(f"[red]Suite {suite_name} failed: {e}[/red]")

    result.elapsed_s = round(time.monotonic() - total_start, 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified LLM benchmark for BlipShell (benchmark-spec.md)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "suites:\n"
            "  pipeline     Summarization, scoring, dedup, contradiction, lesson extraction\n"
            "  extraction   Entity extraction, entity resolution, tag discovery, batch tags\n"
            "  synthesis    Session reflection, titling, project digest, self-reflection, plan gen\n"
            "  interactive  Tool calling, coding (multi-turn harness)\n"
            "  all          Run all suites (default)\n"
        ),
    )
    parser.add_argument(
        "--suite", default="all",
        help="Suite(s) to run: pipeline, extraction, synthesis, interactive, all (default: all)",
    )
    parser.add_argument(
        "--models",
        help=f"Comma-separated model names (default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--sample", type=int, default=20,
        help="Sample size for DB-backed suites (default: 20)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--output", default="data/benchmark_unified.json",
        help="Path for JSON results (default: data/benchmark_unified.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print table format with stub scores (no LLM calls)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use canned LLM responses (exercises scoring logic without Ollama)",
    )
    parser.add_argument(
        "--detail",
        help="Show per-case breakdown for a role (e.g. --detail lesson, --detail lssn)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    suites = _resolve_suites(args.suite)
    models = _resolve_models(args.models)

    if args.dry_run:
        mode = "[dim](dry run)[/dim]"
    elif args.mock:
        mode = "[dim](mock — canned responses)[/dim]"
    else:
        mode = ""
    console.print(f"\n[bold]Unified LLM Benchmark[/bold] {mode}")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Suites: {', '.join(suites)}")
    if not args.dry_run:
        console.print(f"Sample: {args.sample}")
    console.print()

    if args.dry_run:
        result = _dry_run(models, suites)
    elif args.mock:
        result = await _mock_run(models, suites)
    else:
        result = await _live_run(
            models, suites,
            config_path=args.config,
            sample=args.sample,
            output_path=args.output,
        )

    # Print per-role table
    print_role_table(result, suites_filter=suites)

    # Print per-case detail if requested
    if args.detail:
        print_role_detail(result, args.detail)

    # Save JSON
    path = save_role_results(result, args.output)

    console.print(f"\n[dim]Time: {result.elapsed_s:.0f}s | "
                  f"{len(result.models)} models × {len(suites)} suites[/dim]")
    console.print(f"[dim]Results: {path}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
