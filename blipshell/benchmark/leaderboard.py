"""Cross-model leaderboard — answers "for each job, is there a better model
than the one config.yaml currently assigns?"

The pairwise scoreboard (scoreboard.py) compares ONE candidate against ONE
global baseline. That can't answer the per-job model-selection question, because
production routes a *different* model to each job. This module lines every
benchmarked model up against the per-job incumbent (the model config.yaml routes
that job to) and flags where a tested model beats it.

`build_leaderboard` is pure (data in, dict out) so the decision math is
unit-testable without a DB or models. `render_leaderboard` does the Rich output.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table

from blipshell.benchmark.scoreboard import SCORING_METRICS, _latency_map, _weighted

# Benchmark task_type -> the config.yaml ModelsConfig field whose model serves
# that job in production. Tasks NOT in this map (none currently) render with no
# incumbent (rank-only). Fallback fields (ranking_importance/session_review) are
# resolved by the caller before building the incumbents dict.
TASK_TO_CONFIG_FIELD = {
    "ranking": "ranking",
    "importance": "importance",
    "rank_importance": "ranking_importance",
    "contradiction": "reasoning",
    "entity": "reasoning",
    "summarization": "summarization",
    "lessons": "session_review",
    "reasoning": "reasoning",
    "coding": "coding",
    "tool_calling": "tool_calling",
    "session_review": "session_review",
    "embedding": "embedding",
}

# Latency is measured per SUITE, not per job, so several jobs share one mean.
# Each job maps to the suite latency row(s) that cover it (first present wins —
# coding has its own latency only under --all, else it rode the reasoning suite).
# Jobs with no measured latency (embedding) map to an empty list.
JOB_LATENCY_SUITE = {
    "ranking": ["pipeline"], "importance": ["pipeline"], "rank_importance": ["pipeline"],
    "contradiction": ["pipeline"], "entity": ["pipeline"], "summarization": ["pipeline"],
    "lessons": ["pipeline"],
    "reasoning": ["reasoning_suite"], "tool_calling": ["reasoning_suite"],
    "coding": ["coding", "reasoning_suite"],
    "session_review": ["session_review"],
    "embedding": [],
}


def _job_latency(lat_map: dict[str, float], task_type: str) -> Optional[float]:
    """Mean s/call for a job, from its suite's latency (first present wins)."""
    for suite in JOB_LATENCY_SUITE.get(task_type, []):
        if suite in lat_map:
            return lat_map[suite]
    return None


def _score_for(rows: list[dict], task_type: str) -> Optional[float]:
    """The scoring-metric value a model got for one task_type, or None."""
    for r in rows:
        if r["task_type"] == task_type and r["metric"] in SCORING_METRICS and r["value"] is not None:
            return float(r["value"])
    return None


def _model_scoring_map(rows: list[dict]) -> dict[str, dict]:
    """task_type -> {value, metric} for this model's scoring metrics."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["metric"] in SCORING_METRICS and r["value"] is not None:
            out[r["task_type"]] = {"value": float(r["value"]), "metric": r["metric"]}
    return out


def build_leaderboard(
    model_rows: dict[str, list[dict]],
    incumbents: dict[str, str],
    *,
    task_weights: Optional[dict[str, float]] = None,
    verdict_delta: float = 0.05,
) -> dict:
    """Compare every benchmarked model against the per-job incumbent. Pure.

    model_rows  : model name -> its latest-run metric rows
    incumbents  : task_type -> the model config.yaml assigns to that job
    """
    task_weights = task_weights or {}
    models = sorted(model_rows)
    latency = {m: _latency_map(rows) for m, rows in model_rows.items()}

    # All task_types that anyone scored, ordered with known jobs first.
    seen: set[str] = set()
    for rows in model_rows.values():
        seen.update(t for t in _model_scoring_map(rows))
    ordered = [t for t in TASK_TO_CONFIG_FIELD if t in seen]
    ordered += sorted(t for t in seen if t not in TASK_TO_CONFIG_FIELD)

    tasks = []
    suggestions = []
    for task_type in ordered:
        scores = {m: _score_for(rows, task_type) for m, rows in model_rows.items()}
        scores = {m: v for m, v in scores.items() if v is not None}
        if not scores:
            continue

        incumbent = incumbents.get(task_type)
        inc_score = scores.get(incumbent) if incumbent else None

        best_model = max(scores, key=lambda m: scores[m])
        best_score = scores[best_model]

        # Delta vs incumbent only if the incumbent was actually benchmarked.
        delta = (best_score - inc_score) if inc_score is not None else None
        switch = bool(
            delta is not None and delta > verdict_delta and best_model != incumbent
        )
        if switch:
            suggestions.append({
                "task_type": task_type,
                "from": incumbent,
                "to": best_model,
                "delta": round(delta, 4),
            })

        inc_lat = _job_latency(latency.get(incumbent, {}), task_type) if incumbent else None
        best_lat = _job_latency(latency.get(best_model, {}), task_type)
        tasks.append({
            "task_type": task_type,
            "incumbent": incumbent,
            "incumbent_benchmarked": inc_score is not None,
            "incumbent_score": round(inc_score, 4) if inc_score is not None else None,
            "incumbent_latency": round(inc_lat, 2) if inc_lat is not None else None,
            "best_model": best_model,
            "best_score": round(best_score, 4),
            "best_latency": round(best_lat, 2) if best_lat is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "switch": switch,
            "scores": {m: round(v, 4) for m, v in scores.items()},
        })

    composite = {}
    for m, rows in model_rows.items():
        comp = _weighted(_model_scoring_map(rows), task_weights)
        if comp is not None:
            composite[m] = round(comp, 4)

    return {
        "models": models,
        "tasks": tasks,
        "composite": composite,
        "latency": latency,  # model -> {suite: mean_s}
        "switch_suggestions": suggestions,
    }


def render_leaderboard(lb: dict, console: Optional[Console] = None) -> None:
    """Render the per-job decision table, the full model x job matrix, and a
    plain-language switch summary."""
    console = console or Console()

    if not lb["tasks"]:
        console.print("[yellow]No benchmarked models with scoring metrics yet.[/yellow] "
                      "Run `blipshell benchmark run <model> --full` for a few models first.")
        return

    # --- Decision table: one row per job ---
    dec = Table(title="Per-job model selection — incumbent vs best tested", show_lines=False)
    dec.add_column("Job", style="cyan", no_wrap=True)
    dec.add_column("Incumbent (config)", style="dim")
    dec.add_column("Inc.", justify="right")
    dec.add_column("Best tested")
    dec.add_column("Best", justify="right")
    dec.add_column("Delta", justify="right")
    dec.add_column("Lat", justify="right")
    dec.add_column("Switch?", justify="center")

    for t in lb["tasks"]:
        inc = t["incumbent"] or "[dim](unmapped)[/]"
        if t["incumbent"] and not t["incumbent_benchmarked"]:
            inc = f"{t['incumbent']} [yellow](not benchmarked)[/]"
        inc_s = f"{t['incumbent_score']:.3f}" if t["incumbent_score"] is not None else "—"
        best = t["best_model"]
        # Bold the best model only when it actually beats the incumbent.
        best_disp = f"[green]{best}[/]" if t["switch"] else best
        delta = f"{t['delta']:+.3f}" if t["delta"] is not None else "—"
        # Latency of the best model for this job; on a SWITCH, show inc->best so
        # the quality/speed tradeoff is visible (speed never drives the verdict).
        if t["switch"] and t["incumbent_latency"] is not None and t["best_latency"] is not None:
            lat = f"{t['incumbent_latency']:.1f}->{t['best_latency']:.1f}s"
        elif t["best_latency"] is not None:
            lat = f"{t['best_latency']:.1f}s"
        else:
            lat = "—"
        if t["switch"]:
            verdict = "[bold green]SWITCH[/]"
        elif t["delta"] is not None:
            verdict = "[dim]keep[/]"
        else:
            verdict = "[dim]—[/]"
        dec.add_row(t["task_type"], inc, inc_s, best_disp, f"{t['best_score']:.3f}", delta, lat, verdict)

    console.print(dec)
    console.print("[dim]Lat = mean s/call, measured per suite (jobs in the same suite share it); "
                  "speed is shown, not scored.[/dim]")

    # --- Full matrix: models x jobs ---
    models = lb["models"]
    matrix = Table(title="All tested models × jobs", show_lines=False)
    matrix.add_column("Job", style="cyan", no_wrap=True)
    for m in models:
        matrix.add_column(m, justify="right")
    for t in lb["tasks"]:
        row = [t["task_type"]]
        for m in models:
            v = t["scores"].get(m)
            if v is None:
                row.append("—")
            elif m == t["best_model"]:
                row.append(f"[bold]{v:.3f}[/]")
            else:
                row.append(f"{v:.3f}")
        matrix.add_row(*row)
    # Composite row
    matrix.add_section()
    comp = lb["composite"]
    best_comp = max(comp, key=lambda m: comp[m]) if comp else None
    crow = ["[bold]COMPOSITE[/]"]
    for m in models:
        v = comp.get(m)
        if v is None:
            crow.append("—")
        elif m == best_comp:
            crow.append(f"[bold green]{v:.3f}[/]")
        else:
            crow.append(f"{v:.3f}")
    matrix.add_row(*crow)
    console.print(matrix)

    # --- Latency by suite (lower = faster); all models side by side ---
    lat_by_model = lb.get("latency", {})
    suites = sorted({s for mlat in lat_by_model.values() for s in mlat})
    if suites:
        lt = Table(title="Latency (mean s/call, by suite — lower is faster)", show_lines=False)
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

    # --- Switch summary ---
    console.print()
    sugg = lb["switch_suggestions"]
    if not sugg:
        console.print("[green]No switches suggested[/] — current config models win or tie on every "
                      "benchmarked job. (Jobs whose incumbent isn't benchmarked are excluded.)")
    else:
        console.print("[bold]Switch suggestions:[/]")
        for s in sugg:
            console.print(
                f"  - [cyan]{s['task_type']}[/]: {s['from']} -> [green]{s['to']}[/] "
                f"(+{s['delta']:.3f})"
            )
        console.print("\n[dim]Edit config.yaml `models:` to adopt — then re-run with --baseline "
                      "to lock the new reference.[/dim]")
