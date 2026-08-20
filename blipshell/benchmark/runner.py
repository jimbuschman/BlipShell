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
from blipshell.benchmark.report import build_report, write_report
from blipshell.benchmark.results import ResultsStore, results_dir
from blipshell.benchmark.store import BenchmarkStore
from blipshell.core.config import ConfigManager, resolve_config_relative
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

    Delegates to `resolve_config_relative`, which is this same fix living at
    the config chokepoint. Keeping a second implementation here is what let
    the main `database.path` go unanchored for months while this one was safe.
    """
    return resolve_config_relative(db_path, config_path)


def _print_advice(advice_md: str) -> None:
    """Echo the assignment advice to the terminal.

    Printed as Markdown so the verdicts and the exact re-run commands are on
    screen after a run, rather than only inside a file the user then has to open.
    """
    if not advice_md.strip():
        return
    from rich.markdown import Markdown
    console.print()
    console.print(Markdown(advice_md))
    console.print()


def _candidate_context_tokens(config, url: str) -> Optional[int]:
    """Context window to give the candidate: whatever the real endpoint at `url` uses.

    Benchmarking a model in a different window than production gives it is
    measuring a different thing. Matching by URL means a local Ollama candidate
    inherits the `local` endpoint's 32768 instead of Ollama's own default.

    Takes the SMALLEST matching window, deliberately. localhost:11434 serves
    both `local` (32K, real local models) and `local-cloud` (128K, cloud
    passthrough where num_ctx is the remote provider's business anyway). Handing
    a 14B local model a 128K num_ctx allocates an enormous KV cache — on a GPU
    that has already failed CUDA init under memory pressure (2026-07-31), and
    32K is what these models genuinely run at in production.
    """
    if not url:
        return None
    target = url.rstrip("/")
    windows = [
        ep.context_tokens for ep in config.endpoints
        if (ep.url or "").rstrip("/") == target and ep.context_tokens
    ]
    return min(windows) if windows else None


def _report_dir(config_path: Optional[str]) -> str:
    """data/benchmark/ next to the config file (repo root by default), cwd-independent."""
    return resolve_config_relative("data/benchmark", config_path)


async def _regenerate_report(store, config, config_path: Optional[str]) -> tuple[dict, int]:
    """Build the shareable report from the latest committed run of every model.

    Results come from `benchmark_results/` (committed, so every model any
    machine has ever benchmarked is present). `store` is consulted only for the
    catalog cache — price/context/speed for cloud models.

    Returns (written_paths, model_count).
    """
    results = ResultsStore(results_dir(config_path))
    model_rows = results.model_rows()

    catalog: dict[str, dict] = {}
    for m in model_rows:
        cat = await store.catalog_lookup(m)
        if cat:
            catalog[m] = cat

    report = build_report(
        model_rows, catalog=catalog,
        judge_model=config.benchmark.judge_model or None,
        generated_ts=_now_iso(),
        task_weights=config.benchmark.task_weights,
        provenance=results.provenance(),
    )

    # The per-config-key advice is the part you act on, so it leads the file and
    # is echoed to the console. The per-job matrix is supporting detail.
    from blipshell.benchmark.advice import build_advice, render_advice
    advice_md = ""
    try:
        advice_md = render_advice(build_advice(report, config))
    except Exception as e:  # never let advice break the report itself
        logger.warning("Could not build assignment advice: %s", e)

    paths = write_report(report, _report_dir(config_path), advice=advice_md)
    return paths, len(model_rows), advice_md


def merge_repeat_rows(row_sets: list[list[dict]]) -> list[dict]:
    """Merge N repeats of the same run into one row set.

    Numeric metrics become their mean, with `values` (per-repeat) and
    `spread` (max-min) attached — single runs were being read as precise
    points while within-model variance measured ~40% of between-model signal
    (qwen3 lessons: 0.420 then 0.345, nothing said which to believe).
    Non-numeric or missing values keep the first occurrence. Row identity is
    (suite, task_type, metric); rows appearing in only some repeats (e.g. a
    judge that failed once) merge over the repeats they have.
    """
    if len(row_sets) == 1:
        return row_sets[0]
    order: list[tuple] = []
    grouped: dict[tuple, list[dict]] = {}
    for rows in row_sets:
        for r in rows:
            key = (r.get("suite"), r.get("task_type"), r.get("metric"))
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(r)
    merged: list[dict] = []
    for key in order:
        group = grouped[key]
        base = dict(group[0])
        vals = [r.get("value") for r in group
                if isinstance(r.get("value"), (int, float))]
        if len(vals) > 1:
            base["value"] = round(sum(vals) / len(vals), 4)
            base["values"] = vals
            base["spread"] = round(max(vals) - min(vals), 4)
        merged.append(base)
    return merged


async def run_benchmark(
    model: str,
    *,
    config_path: Optional[str] = None,
    judge_enabled: bool = True,
    provider: str = "ollama",
    url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    coding_timeout: float = 300.0,
    jobs: Optional[set] = None,
    timeout_override: Optional[float] = None,
    context_tokens: Optional[int] = None,
    repeats: int = 1,
) -> None:
    """Run a deep test of a model across the requested jobs, store metrics, and
    regenerate the shareable report. `jobs=None` = every job at full depth."""
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

    # Give the candidate the SAME per-call timeout and context window it would
    # get in production. Both were previously left at library defaults: the
    # timeout fell back to 120s (config.yaml sets 300 for slow local models) and
    # num_ctx was never sent at all. A 14B local model then timed out on the
    # reasoning suite and scored 0 on ability it has (live 2026-08-03, qwen3:14b
    # on plans/analysis).
    # Per-call timeout: config.llm.timeout unless overridden. A local 14B running
    # with thinking on can exceed 300s on generation-heavy jobs, and a timeout
    # there doesn't fail loudly — it drops that case from the judged set and
    # biases the score upward, so it's worth being generous here.
    llm_cfg = config.llm
    if timeout_override:
        llm_cfg = llm_cfg.model_copy(update={"timeout": float(timeout_override)})

    cand_ctx = _candidate_context_tokens(config, url) if context_tokens is None \
        else context_tokens
    if cand_ctx:
        console.print(f"[dim]Candidate: num_ctx={cand_ctx:,}, "
                      f"timeout={llm_cfg.timeout:.0f}s[/dim]")
    else:
        console.print(
            f"[yellow]No configured endpoint matches {url} — sending no num_ctx, "
            f"so Ollama's default window applies and long prompts may be "
            f"truncated.[/yellow] [dim]timeout={llm_cfg.timeout:.0f}s[/dim]"
        )
    from blipshell.benchmark.recording import RecordingRouter, wrap_router_clients
    router = RecordingRouter(build_candidate_router(
        model, provider=provider, url=url, api_key=api_key,
        context_tokens=cand_ctx, llm_config=llm_cfg,
    ))
    # Record at the client layer too: agentic coding and tool-calling chat
    # bypass router.generate() (pilot 2026-08-10 — the 0.19 suite produced
    # zero transcripts).
    wrap_router_clients(router)
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
        if jobs:
            console.print(f"[dim]Scoped to jobs: {', '.join(sorted(jobs))}[/dim]")
        row_sets: list[list[dict]] = []
        for rep in range(max(1, repeats)):
            if repeats > 1:
                console.print(f"[dim]Repeat {rep + 1}/{repeats}[/dim]")
            router.repeat = rep
            row_sets.append(await harness.run(
                db_path=config.database.path,
                ollama_url=ollama_url,
                full_sample=bench.full_sample,
                coding_timeout=coding_timeout,
                jobs=jobs,
                on_status=lambda m: console.print(f"  [dim]{m}[/dim]"),
            ))
        rows = merge_repeat_rows(row_sets)
        # Results go to a committed file, not the gitignored DB — so the other
        # machine inherits this run on `git pull` and the comparison corpus
        # actually accumulates.
        rstore = ResultsStore(results_dir(config_path))
        res_path = rstore.write_run(
            model=model, run_group=group, run_ts=run_ts, rows=rows,
            tier="deep", judge_model=bench.judge_model or None, jobs=jobs,
        )
        # Transcripts are the primary artifact now — scores index them.
        tr_path = rstore.write_transcripts(
            model=model, run_ts=run_ts, calls=router.calls,
        )
        console.print(f"[dim]Transcripts ({len(router.calls)} calls) -> {tr_path}[/dim]")
        console.print(
            f"[green]Recorded {len(rows)} metrics for {model}[/green] "
            f"-> [cyan]{res_path}[/cyan]\n"
            f"[dim]Commit that file so the other machine sees this run.[/dim]"
        )

        paths, n, advice_md = await _regenerate_report(store, config, config_path)
        _print_advice(advice_md)
        console.print(
            f"[bold green]Report updated[/bold green] ({n} model{'s' if n != 1 else ''} side by side):\n"
            f"  [cyan]{paths['md']}[/cyan]\n"
            f"  [dim]{paths['json']}[/dim]"
        )
    finally:
        await store.close()


async def run_report(*, config_path: Optional[str] = None) -> None:
    """Regenerate the shareable report from stored runs, without re-running."""
    config = ConfigManager(config_path).load()
    store = await BenchmarkStore(_resolve_db_path(config.benchmark.db_path, config_path)).initialize()
    try:
        paths, n, advice_md = await _regenerate_report(store, config, config_path)
        if n == 0:
            console.print("[yellow]No benchmarked models yet.[/yellow] "
                          "Run `blipshell benchmark run <model>` first.")
            return
        _print_advice(advice_md)
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
            "[dim]Test one with:[/dim] blipshell benchmark run <id>   "
            "[dim](add --jobs pipeline,reasoning to scope)[/dim]"
        )
    finally:
        await store.close()
