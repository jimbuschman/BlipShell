"""Cross-model leaderboard — raw numbers for manual comparison.

BlipShell routes a mix of local and cloud models per job, each with its own
fallback model, chosen by endpoint priority/availability. That routing is too
rich for an automated "should I switch" verdict to model honestly (it would
have to assume a single model per job and ignore endpoints + fallbacks). So
this view makes NO recommendation: it lays every benchmarked model's scores out
side by side — a quality matrix plus per-suite latency — and lets you compare by
eye and decide given what you know about your own routing.

`build_leaderboard` is pure (data in, dict out) so it's unit-testable without a
DB or models. `render_leaderboard` does the Rich output.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table

from blipshell.benchmark.scoreboard import SCORING_METRICS, _latency_map, _weighted

# Stable display order: known jobs first (pipeline + interaction order), any
# unrecognized task_type after, alphabetically.
KNOWN_JOB_ORDER = [
    "ranking", "importance", "rank_importance", "contradiction", "entity",
    "summarization", "lessons", "reasoning", "coding", "tool_calling",
    "session_review", "embedding",
]


def _model_scoring_map(rows: list[dict]) -> dict[str, dict]:
    """task_type -> {value, metric} for this model's scoring metrics."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["metric"] in SCORING_METRICS and r["value"] is not None:
            out[r["task_type"]] = {"value": float(r["value"]), "metric": r["metric"]}
    return out


def build_leaderboard(
    model_rows: dict[str, list[dict]],
    *,
    task_weights: Optional[dict[str, float]] = None,
) -> dict:
    """Lay every benchmarked model's scores out per task. Pure, no verdict.

    model_rows : model name -> its latest-run metric rows
    """
    task_weights = task_weights or {}
    models = sorted(model_rows)
    scoring = {m: _model_scoring_map(rows) for m, rows in model_rows.items()}
    latency = {m: _latency_map(rows) for m, rows in model_rows.items()}

    seen: set[str] = set()
    for sm in scoring.values():
        seen.update(sm)
    ordered = [t for t in KNOWN_JOB_ORDER if t in seen]
    ordered += sorted(t for t in seen if t not in KNOWN_JOB_ORDER)

    tasks = []
    for task_type in ordered:
        scores = {m: scoring[m][task_type]["value"] for m in models if task_type in scoring[m]}
        if not scores:
            continue
        metric = next(scoring[m][task_type]["metric"] for m in models if task_type in scoring[m])
        tasks.append({
            "task_type": task_type,
            "metric": metric,
            "scores": {m: round(v, 4) for m, v in scores.items()},
            "best_model": max(scores, key=lambda m: scores[m]),  # for display bolding only
        })

    composite = {}
    for m in models:
        comp = _weighted(scoring[m], task_weights)
        if comp is not None:
            composite[m] = round(comp, 4)

    return {
        "models": models,
        "tasks": tasks,
        "composite": composite,
        "latency": latency,  # model -> {suite: mean_s}
    }


def render_leaderboard(lb: dict, console: Optional[Console] = None) -> None:
    """Render the quality matrix + per-suite latency. Numbers only, no verdict."""
    console = console or Console()

    if not lb["tasks"]:
        console.print("[yellow]No benchmarked models with scoring metrics yet.[/yellow] "
                      "Run `blipshell benchmark run <model> --full` for a few models first.")
        return

    models = lb["models"]

    # --- Quality matrix: tasks x models (best per row bolded) ---
    q = Table(title="Quality by task (accuracy / quality / tool_pass_rate; higher is better)",
              show_lines=False)
    q.add_column("Task", style="cyan", no_wrap=True)
    for m in models:
        q.add_column(m, justify="right")
    for t in lb["tasks"]:
        row = [t["task_type"]]
        for m in models:
            v = t["scores"].get(m)
            if v is None:
                row.append("—")
            elif m == t["best_model"]:
                row.append(f"[bold green]{v:.3f}[/]")
            else:
                row.append(f"{v:.3f}")
        q.add_row(*row)
    # Composite (weighted across the tasks each model measured)
    comp = lb["composite"]
    if comp:
        best_comp = max(comp, key=lambda m: comp[m])
        q.add_section()
        crow = ["[bold]COMPOSITE[/]"]
        for m in models:
            v = comp.get(m)
            if v is None:
                crow.append("—")
            elif m == best_comp:
                crow.append(f"[bold green]{v:.3f}[/]")
            else:
                crow.append(f"{v:.3f}")
        q.add_row(*crow)
    console.print(q)

    # --- Latency by suite (lower = faster); all models side by side ---
    lat_by_model = lb.get("latency", {})
    suites = sorted({s for mlat in lat_by_model.values() for s in mlat})
    if suites:
        lt = Table(title="Latency (mean s/call, by suite; lower is faster)", show_lines=False)
        lt.add_column("Suite", style="cyan", no_wrap=True)
        for m in models:
            lt.add_column(m, justify="right")
        for suite in suites:
            vals = {m: lat_by_model.get(m, {}).get(suite) for m in models}
            present = {m: v for m, v in vals.items() if v is not None}
            fastest = min(present, key=lambda m: present[m]) if present else None
            row = [suite]
            for m in models:
                v = vals[m]
                if v is None:
                    row.append("—")
                elif m == fastest:
                    row.append(f"[bold green]{v:.2f}[/]")
                else:
                    row.append(f"{v:.2f}")
            lt.add_row(*row)
        console.print(lt)

    console.print(
        "[dim]No verdict on purpose: BlipShell's per-job local/cloud routing + fallback models "
        "aren't modeled here. Latency is a per-suite mean (jobs in the same suite share it); "
        "embedding has none. Compare the numbers and decide.[/dim]"
    )
