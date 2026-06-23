"""Thin async entry points for `blipshell benchmark` — wires config, judge,
candidate router, harness, store, and scoreboard together. The CLI command
delegates here (lazy import + asyncio.run), matching nightly/test commands.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
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
from blipshell.benchmark.report import build_report, write_report
from blipshell.benchmark.store import BenchmarkStore
from blipshell.core.config import DEFAULT_CONFIG_PATH, ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.models.config import get_ollama_url, resolve_env_vars

logger = logging.getLogger(__name__)
console = Console()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_group(model: str, run_ts: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    return f"{slug}@{run_ts}"


def _resolve_db_path(db_path: str, config_path: Optional[str]) -> str:
    """Anchor a relative benchmark db_path to the config file's directory (the
    repo root by default), NOT the current working directory.

    `blipshell` is an installed CLI run from any folder; with a cwd-relative
    path, `benchmark run` from the repo and `benchmark report` from another
    folder would open different files — the second silently showing an empty DB.
    """
    p = Path(db_path)
    if p.is_absolute():
        return str(p)
    base = Path(config_path).resolve().parent if config_path else DEFAULT_CONFIG_PATH.parent
    return str((base / p).resolve())


def _report_dir(config_path: Optional[str]) -> str:
    """data/benchmark/ next to the config file (repo root by default), cwd-independent."""
    base = Path(config_path).resolve().parent if config_path else DEFAULT_CONFIG_PATH.parent
    return str(base / "data" / "benchmark")


async def _regenerate_report(store, config, config_path: Optional[str]) -> tuple[dict, int]:
    """Build the shareable report from the latest stored run of every model.
    Returns (written_paths, model_count)."""
    models = await store.models_with_runs()
    model_rows: dict[str, list[dict]] = {}
    catalog: dict[str, dict] = {}
    for m in models:
        group = await store.latest_run_group(m)
        if not group:
            continue
        model_rows[m] = await store.metrics_for_group(group)
        cat = await store.catalog_lookup(m)
        if cat:
            catalog[m] = cat
    report = build_report(
        model_rows, catalog=catalog,
        judge_model=config.benchmark.judge_model or None,
        generated_ts=_now_iso(),
        task_weights=config.benchmark.task_weights,
    )
    paths = write_report(report, _report_dir(config_path))
    return paths, len(model_rows)


async def run_benchmark(
    model: str,
    *,
    config_path: Optional[str] = None,
    judge_enabled: bool = True,
    provider: str = "ollama",
    url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    coding_timeout: float = 300.0,
) -> None:
    """Run ONE deep test of a model across every job, store metrics, and
    regenerate the shareable report. No tiers — always full depth."""
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
    # Embedding benchmark always uses a local Ollama endpoint (embedders are local).
    ollama_url = url if provider == "ollama" else get_ollama_url(config.endpoints)

    run_ts = _now_iso()
    group = _run_group(model, run_ts)

    db_path = _resolve_db_path(bench.db_path, config_path)
    store = await BenchmarkStore(db_path).initialize()
    try:
        console.rule(f"[bold blue]Benchmarking {model} — full deep run")
        console.print("[dim]Runs every job incl. the agentic coding suite + embedding. "
                      "This is intentionally heavy (~30-90 min); shallow runs don't discriminate.[/dim]")
        harness = BenchmarkHarness(
            model=model, router=router, run_group=group, run_ts=run_ts,
            tier="deep", judge=judge,
        )
        rows = await harness.run(
            db_path=config.database.path,
            ollama_url=ollama_url,
            full_sample=bench.full_sample,
            coding_timeout=coding_timeout,
            on_status=lambda m: console.print(f"  [dim]{m}[/dim]"),
        )
        await store.record_many(rows)
        console.print(f"[green]Recorded {len(rows)} metrics for {model}.[/green]")

        paths, n = await _regenerate_report(store, config, config_path)
        console.print(
            f"[bold green]Report updated[/bold green] ({n} model{'s' if n != 1 else ''} side by side):\n"
            f"  [cyan]{paths['md']}[/cyan]   ← hand this to a stronger LLM\n"
            f"  [dim]{paths['json']}[/dim]"
        )
    finally:
        await store.close()


async def run_report(*, config_path: Optional[str] = None) -> None:
    """Regenerate the shareable report from stored runs, without re-running."""
    config = ConfigManager(config_path).load()
    store = await BenchmarkStore(_resolve_db_path(config.benchmark.db_path, config_path)).initialize()
    try:
        paths, n = await _regenerate_report(store, config, config_path)
        if n == 0:
            console.print("[yellow]No benchmarked models yet.[/yellow] "
                          "Run `blipshell benchmark run <model>` first.")
            return
        console.print(
            f"[bold green]Report regenerated[/bold green] ({n} model{'s' if n != 1 else ''}):\n"
            f"  [cyan]{paths['md']}[/cyan]\n  [dim]{paths['json']}[/dim]"
        )
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
    store = await BenchmarkStore(_resolve_db_path(bench.db_path, config_path)).initialize()
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
