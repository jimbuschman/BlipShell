"""Thin async entry points for `blipshell benchmark` — wires config, judge,
candidate router, harness, store, and scoreboard together. The CLI command
delegates here (lazy import + asyncio.run), matching nightly/test commands.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table

from blipshell.benchmark.discovery import (
    fetch_artificial_analysis,
    fetch_openrouter,
    shortlist,
)
from blipshell.benchmark.harness import BenchmarkHarness, build_candidate_router
from blipshell.benchmark.judge import JudgeUnavailable, build_judge
from blipshell.benchmark.scoreboard import build_scoreboard, render_scoreboard
from blipshell.benchmark.store import BenchmarkStore
from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.models.config import get_ollama_url, resolve_env_vars

logger = logging.getLogger(__name__)
console = Console()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_group(model: str, run_ts: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    return f"{slug}@{run_ts}"


def _baseline_model_desc(config) -> str:
    m = config.models
    return f"{m.tool_calling} (interactive), {m.reasoning} (background)"


async def run_benchmark(
    model: str,
    *,
    config_path: Optional[str] = None,
    tier: str = "quick",
    judge_enabled: bool = True,
    baseline: bool = False,
    provider: str = "ollama",
    url: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> None:
    """Run a candidate through the suites, store metrics, render the verdict."""
    config = ConfigManager(config_path).load()
    bench = config.benchmark

    # Judge — built from the REAL configured endpoints (looked up by name).
    judge = None
    if judge_enabled and bench.judge_model:
        if model == bench.judge_model:
            console.print(
                f"[red]Refusing to run: candidate '{model}' is the configured judge "
                f"model — that would be self-grading. Pick a different judge_endpoint/"
                f"judge_model or a different candidate.[/red]"
            )
            return
        real_manager = EndpointManager(config.endpoints, config.llm)
        try:
            judge = build_judge(config, real_manager)
            console.print(f"[dim]Judge: {bench.judge_model} via '{bench.judge_endpoint}'[/dim]")
        except JudgeUnavailable as e:
            console.print(f"[yellow]Judge disabled: {e}. Running deterministic metrics only.[/yellow]")
            judge = None
    elif judge_enabled and not bench.judge_model:
        console.print("[dim]No judge_model configured — open-ended tasks not graded.[/dim]")

    # Candidate router (its own pinned endpoint; no fallback).
    if provider == "ollama" and not url:
        url = get_ollama_url(config.endpoints)
    if provider == "openai" and not url:
        console.print("[red]--provider openai requires --url[/red]")
        return
    api_key = resolve_env_vars(f"${{{api_key_env}}}") if api_key_env else None
    router = build_candidate_router(model, provider=provider, url=url, api_key=api_key)

    run_ts = _now_iso()
    group = _run_group(model, run_ts)

    store = await BenchmarkStore(bench.db_path).initialize()
    try:
        if baseline:
            await store.clear_baseline()

        console.rule(f"[bold blue]Benchmarking {model}  ({tier} tier{', baseline' if baseline else ''})")
        harness = BenchmarkHarness(
            model=model, router=router, run_group=group, run_ts=run_ts,
            tier=tier, judge=judge, is_baseline=baseline,
        )
        rows = await harness.run(
            db_path=config.database.path,
            full_sample=bench.full_sample,
            on_status=lambda m: console.print(f"  [dim]{m}[/dim]"),
        )
        await store.record_many(rows)
        console.print(f"[green]Recorded {len(rows)} metrics to {bench.db_path}[/green]\n")

        # A --baseline run is the reference itself — render standalone (no self-compare).
        baseline_rows = [] if baseline else await store.baseline_metrics()
        catalog = await store.catalog_lookup(model)
        sb = build_scoreboard(
            rows, baseline_rows,
            task_weights=bench.task_weights, verdict_delta=bench.verdict_delta,
        )
        render_scoreboard(
            sb, model, _baseline_model_desc(config),
            catalog=catalog, console=console,
        )
    finally:
        await store.close()


async def run_scoreboard(model: str, *, config_path: Optional[str] = None) -> None:
    """Render the most recent stored run for a model vs the baseline."""
    config = ConfigManager(config_path).load()
    bench = config.benchmark
    store = await BenchmarkStore(bench.db_path).initialize()
    try:
        group = await store.latest_run_group(model)
        if not group:
            models = await store.models_with_runs()
            console.print(f"[yellow]No runs for '{model}'.[/yellow] "
                          f"Models with data: {', '.join(models) or '(none)'}")
            return
        rows = await store.metrics_for_group(group)
        baseline_rows = await store.baseline_metrics()
        # If this model IS the baseline, render standalone.
        if rows and rows[0].get("is_baseline"):
            baseline_rows = []
        catalog = await store.catalog_lookup(model)
        sb = build_scoreboard(
            rows, baseline_rows,
            task_weights=bench.task_weights, verdict_delta=bench.verdict_delta,
        )
        render_scoreboard(sb, model, _baseline_model_desc(config), catalog=catalog, console=console)
    finally:
        await store.close()


async def run_discover(
    *,
    config_path: Optional[str] = None,
    min_context: Optional[int] = None,
    max_price: Optional[float] = None,
    vision: bool = False,
) -> None:
    """Pull the candidate shortlist from OpenRouter + Artificial Analysis."""
    config = ConfigManager(config_path).load()
    bench = config.benchmark
    store = await BenchmarkStore(bench.db_path).initialize()
    try:
        known = await store.known_catalog_keys()
        fetched_ts = _now_iso()

        console.print("[dim]Fetching OpenRouter model list…[/dim]")
        or_entries = await fetch_openrouter(fetched_ts)
        console.print("[dim]Fetching Artificial Analysis (if AA_API_KEY set)…[/dim]")
        aa_entries = await fetch_artificial_analysis(fetched_ts)

        all_entries = or_entries + aa_entries
        for e in all_entries:
            await store.upsert_catalog(e)
        console.print(
            f"[green]Catalog: {len(or_entries)} OpenRouter + {len(aa_entries)} "
            f"Artificial Analysis entries[/green]"
        )

        short = shortlist(
            all_entries,
            min_context=min_context if min_context is not None else bench.discover_min_context,
            max_price=max_price if max_price is not None else bench.discover_max_price,
            vision_only=vision,
            known_keys=known,
        )
        if not short:
            console.print("[yellow]No models matched the shortlist filters.[/yellow]")
            return

        table = Table(title=f"Candidate shortlist ({len(short)} models)", show_lines=False)
        table.add_column("New", justify="center", width=3)
        table.add_column("Model", style="cyan", overflow="ellipsis", max_width=44)
        table.add_column("Src", style="dim", width=3)
        table.add_column("Ctx", justify="right", width=9)
        table.add_column("$/1M in", justify="right", width=8)
        table.add_column("II", justify="right", width=4)
        table.add_column("tok/s", justify="right", width=6)
        table.add_column("Vis", justify="center", width=4)
        for e in short[:40]:
            table.add_row(
                "[green]new[/]" if e.get("is_new") else "",
                e["model"],
                e["source"][:2],
                f"{e['context_length']:,}" if e.get("context_length") else "-",
                f"{e['price_in']:.2f}" if e.get("price_in") is not None else "-",
                f"{e['intelligence_index']:.0f}" if e.get("intelligence_index") is not None else "-",
                f"{e['tok_per_s']:.0f}" if e.get("tok_per_s") is not None else "-",
                "yes" if e.get("vision") else "",
            )
        console.print(table)
        if len(short) > 40:
            console.print(f"[dim]…and {len(short) - 40} more (filters narrow this).[/dim]")
        console.print(
            "[dim]Test one with:[/dim] blipshell benchmark run --model <id> --quick"
        )
    finally:
        await store.close()
