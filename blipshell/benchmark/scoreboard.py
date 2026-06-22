"""Scoreboard + switch-verdict.

Turns stored metric rows into the one thing the user actually wants: a per-task
comparison of a candidate model against the current production baseline, a
weighted composite, and a plain-language "switch / keep / mixed" recommendation.

`build_scoreboard` is pure (list[row] in, dict out) so the verdict math is
unit-testable without a DB or models. `render_scoreboard` does the Rich output.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table

# Metrics that count toward the composite (higher = better for all of them).
SCORING_METRICS = {"accuracy", "quality", "tool_pass_rate"}


def _scoring_map(rows: list[dict]) -> dict[str, dict]:
    """task_type -> {value, metric} for scoring metrics with a non-None value."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["metric"] in SCORING_METRICS and r["value"] is not None:
            out[r["task_type"]] = {"value": float(r["value"]), "metric": r["metric"]}
    return out


def _verdict(delta: Optional[float], threshold: float) -> str:
    if delta is None:
        return "n/a"
    if delta > threshold:
        return "better"
    if delta < -threshold:
        return "worse"
    return "tie"


def build_scoreboard(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    *,
    task_weights: Optional[dict[str, float]] = None,
    verdict_delta: float = 0.05,
) -> dict:
    """Compare a candidate's metric rows against the baseline's. Pure function."""
    task_weights = task_weights or {}
    cand = _scoring_map(candidate_rows)
    base = _scoring_map(baseline_rows)
    have_baseline = bool(base)

    tasks = []
    for task_type in sorted(cand):
        c_val = cand[task_type]["value"]
        b_val = base[task_type]["value"] if task_type in base else None
        delta = (c_val - b_val) if b_val is not None else None
        tasks.append({
            "task_type": task_type,
            "metric": cand[task_type]["metric"],
            "candidate": round(c_val, 4),
            "baseline": round(b_val, 4) if b_val is not None else None,
            "delta": round(delta, 4) if delta is not None else None,
            "weight": task_weights.get(task_type, 1.0),
            "verdict": _verdict(delta, verdict_delta),
        })

    comp_cand = _weighted(cand, task_weights)
    # Composite baseline computed only over tasks the candidate also measured,
    # so the two composites are over the same task set (fair comparison).
    shared_base = {k: v for k, v in base.items() if k in cand}
    comp_base = _weighted(shared_base, task_weights) if have_baseline else None
    comp_delta = (comp_cand - comp_base) if (comp_cand is not None and comp_base is not None) else None

    if not have_baseline:
        overall = "no_baseline"
    else:
        overall = _verdict(comp_delta, verdict_delta)

    return {
        "tasks": tasks,
        "composite_candidate": round(comp_cand, 4) if comp_cand is not None else None,
        "composite_baseline": round(comp_base, 4) if comp_base is not None else None,
        "composite_delta": round(comp_delta, 4) if comp_delta is not None else None,
        "overall": overall,
        "have_baseline": have_baseline,
        "latency": _latency_map(candidate_rows),
    }


def _weighted(score_map: dict[str, dict], weights: dict[str, float]) -> Optional[float]:
    num = den = 0.0
    for task_type, d in score_map.items():
        w = weights.get(task_type, 1.0)
        num += w * d["value"]
        den += w
    return (num / den) if den else None


def _latency_map(rows: list[dict]) -> dict[str, float]:
    return {
        r["task_type"]: r["value"]
        for r in rows
        if r["metric"] == "latency_s" and r["value"] is not None
    }


def recommendation_text(sb: dict, candidate_model: str, baseline_model_desc: str) -> str:
    """One-paragraph plain-language switch recommendation."""
    if not sb["have_baseline"]:
        return (
            f"No production baseline recorded yet — run "
            f"`blipshell benchmark run --model <current> --baseline` first, then "
            f"compare. Showing {candidate_model}'s standalone scores above."
        )
    better = [t["task_type"] for t in sb["tasks"] if t["verdict"] == "better"]
    worse = [t["task_type"] for t in sb["tasks"] if t["verdict"] == "worse"]
    parts = []
    if better:
        parts.append("better at " + ", ".join(better))
    if worse:
        parts.append("worse at " + ", ".join(worse))
    if not parts:
        parts.append("roughly on par across all measured tasks")
    cd = sb["composite_delta"]
    overall = sb["overall"]
    head = {
        "better": f"Recommend SWITCHING to {candidate_model}",
        "worse": f"Recommend KEEPING {baseline_model_desc}",
        "tie": f"{candidate_model} and current are about even",
        "n/a": f"{candidate_model} vs current",
    }.get(overall, f"{candidate_model} vs current")
    delta_str = f" (composite {cd:+.3f})" if cd is not None else ""
    return f"{head}{delta_str}: {candidate_model} is " + "; ".join(parts) + "."


def render_scoreboard(
    sb: dict,
    candidate_model: str,
    baseline_model_desc: str,
    *,
    catalog: Optional[dict] = None,
    console: Optional[Console] = None,
) -> None:
    """Render the scoreboard + recommendation to the terminal."""
    console = console or Console()

    title = f"Benchmark: {candidate_model}"
    if sb["have_baseline"]:
        title += f"  vs baseline ({baseline_model_desc})"
    table = Table(title=title, show_lines=False)
    table.add_column("Task", style="cyan", no_wrap=True)
    table.add_column("Metric", style="dim")
    table.add_column("Candidate", justify="right")
    if sb["have_baseline"]:
        table.add_column("Baseline", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Verdict", justify="center")

    color = {"better": "green", "worse": "red", "tie": "yellow", "n/a": "dim"}
    for t in sb["tasks"]:
        row = [t["task_type"], t["metric"], f"{t['candidate']:.3f}"]
        if sb["have_baseline"]:
            base = f"{t['baseline']:.3f}" if t["baseline"] is not None else "—"
            delta = f"{t['delta']:+.3f}" if t["delta"] is not None else "—"
            v = t["verdict"]
            row += [base, delta, f"[{color.get(v, 'dim')}]{v}[/]"]
        table.add_row(*row)

    # Composite row
    cc = sb["composite_candidate"]
    if sb["have_baseline"]:
        cb = sb["composite_baseline"]
        cd = sb["composite_delta"]
        ov = sb["overall"]
        table.add_section()
        table.add_row(
            "[bold]COMPOSITE[/]", "weighted",
            f"[bold]{cc:.3f}[/]" if cc is not None else "—",
            f"{cb:.3f}" if cb is not None else "—",
            f"{cd:+.3f}" if cd is not None else "—",
            f"[bold {color.get(ov, 'dim')}]{ov}[/]",
        )
    else:
        table.add_section()
        table.add_row("[bold]COMPOSITE[/]", "weighted", f"[bold]{cc:.3f}[/]" if cc is not None else "—")

    console.print(table)

    # Cost / speed line from the discovery catalog, if available.
    if catalog:
        pin = catalog.get("price_in")
        pout = catalog.get("price_out")
        tps = catalog.get("tok_per_s")
        ii = catalog.get("intelligence_index")
        bits = []
        if pin is not None or pout is not None:
            bits.append(f"cost ${pin or 0:.2f}/${pout or 0:.2f} per 1M (in/out)")
        if tps:
            bits.append(f"{tps:.0f} tok/s")
        if ii is not None:
            bits.append(f"Intelligence Index {ii:.0f}")
        if catalog.get("context_length"):
            bits.append(f"{catalog['context_length']:,} ctx")
        if bits:
            console.print(f"[dim]Catalog:[/] {'  ·  '.join(bits)}")

    lat = sb.get("latency") or {}
    if lat:
        lat_str = ", ".join(f"{k}={v:.2f}s" for k, v in sorted(lat.items()))
        console.print(f"[dim]Latency (mean/call):[/] {lat_str}")

    console.print()
    console.print(recommendation_text(sb, candidate_model, baseline_model_desc))
