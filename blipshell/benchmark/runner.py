"""Benchmark runner — orchestrates suites and produces results."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

from blipshell.benchmark.models import BenchmarkResult, SuiteResult
from blipshell.benchmark.output import (
    console,
    print_comparison_table,
    print_suite_detail,
    save_incremental,
    save_results,
)
from blipshell.benchmark.shared import (
    check_model_availability,
    get_config_and_db,
    get_context_tokens,
    make_cloud_router,
    make_router,
)
from blipshell.benchmark.suites import SUITES, get_suite

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODELS = ["qwen3.5:4b", "qwen3.5:9b", "qwen3:14b", "qwen3:4b"]
CLOUD_MODELS = ["glm-5:cloud", "qwen3.5:cloud"]


async def run_benchmark(
    *,
    models: list[str] | None = None,
    suite_names: list[str] | None = None,
    thorough: bool = False,
    cloud: bool = False,
    config_path: str | None = None,
    db_path: str | None = None,
    output_path: str | None = None,
    quiet: bool = False,
    on_status: Callable[[str], None] | None = None,
) -> BenchmarkResult:
    """Run the unified benchmark.

    Args:
        models: Model names to test. Defaults to DEFAULT_LOCAL_MODELS.
        suite_names: Suite names to run. Defaults to all.
        thorough: Use larger sample sizes.
        cloud: Include cloud models.
        config_path: Path to config.yaml.
        db_path: Path to SQLite DB.
        output_path: Path for JSON results.
        quiet: Suppress Rich output.
        on_status: Callback for progress updates.
    """
    # Load config
    config, resolved_db, ollama_url = get_config_and_db(config_path, db_path)

    # Resolve models
    model_list = list(models) if models else list(DEFAULT_LOCAL_MODELS)
    if cloud:
        model_list.extend(CLOUD_MODELS)

    # Resolve suites
    suites_to_run = suite_names if suite_names else list(SUITES)

    if not quiet:
        console.print(f"\n[bold]Unified Model Benchmark[/bold]")
        console.print(f"Models: {', '.join(model_list)}")
        console.print(f"Suites: {', '.join(suites_to_run)}")
        console.print(f"DB: {resolved_db}")
        console.print(f"Ollama: {ollama_url}")
        if thorough:
            console.print("[yellow]Thorough mode (larger samples)[/yellow]")
        console.print()

    # Check model availability
    available, unavailable = await check_model_availability(model_list, ollama_url)
    if unavailable and not quiet:
        console.print(f"[yellow]Unavailable models (skipping): {', '.join(unavailable)}[/yellow]")
    if not available:
        if not quiet:
            console.print("[red]No models available![/red]")
        return BenchmarkResult()

    # Router factory
    def router_factory(model_name: str):
        ctx_tokens = get_context_tokens(model_name)
        # Check if it's a cloud model — try to find matching endpoint
        if model_name.endswith(":cloud") or "cloud" in model_name:
            for ep in config.endpoints:
                if ep.enabled:
                    cloud_router = make_cloud_router(ep.name, model_name, config)
                    if cloud_router:
                        return cloud_router
        kwargs = {}
        if ctx_tokens:
            kwargs["context_tokens"] = ctx_tokens
        return make_router(model_name, ollama_url, **kwargs)

    # Run suites
    result = BenchmarkResult(
        models=available,
        suites=suites_to_run,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    total_start = time.monotonic()

    for suite_name in suites_to_run:
        try:
            suite = get_suite(suite_name)
        except ValueError as e:
            if not quiet:
                console.print(f"[red]{e}[/red]")
            continue

        if not quiet:
            console.print(f"\n{'─' * 60}")
            console.print(f"[bold cyan]Suite: {suite.name}[/bold cyan] — {suite.description}")
            console.print(f"{'─' * 60}")

        try:
            suite_results = await suite.run(
                available,
                router_factory=router_factory,
                config=config,
                db_path=resolved_db,
                ollama_url=ollama_url,
                thorough=thorough,
                on_status=on_status or (_make_console_status() if not quiet else None),
            )
            result.suite_results.extend(suite_results)

            # Print suite detail
            if not quiet and suite_results:
                print_suite_detail(suite_name, suite_results)

            # Save incrementally
            for sr in suite_results:
                save_incremental(sr.model, sr, output_path)

        except Exception as e:
            logger.exception("Suite %s failed", suite_name)
            if not quiet:
                console.print(f"[red]Suite {suite_name} failed: {e}[/red]")

    result.elapsed_s = round(time.monotonic() - total_start, 1)

    # Final comparison table
    if not quiet:
        print_comparison_table(result)
        console.print(f"\n[dim]Total time: {result.elapsed_s:.0f}s[/dim]")

    # Save full results
    path = save_results(result, output_path)
    if not quiet:
        console.print(f"[dim]Results saved to {path}[/dim]")

    return result


def _make_console_status() -> Callable[[str], None]:
    """Create a status callback that prints to console."""
    def on_status(msg: str):
        console.print(f"  [dim]{msg}[/dim]")
    return on_status
