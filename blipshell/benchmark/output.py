"""Output formatting for the unified benchmark system."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from blipshell.benchmark.models import BenchmarkResult, SuiteResult, TaskScore

console = Console()

DEFAULT_OUTPUT_PATH = "data/benchmark_results.json"


def print_comparison_table(result: BenchmarkResult) -> None:
    """Print the unified comparison table: model x suite -> quality + speed."""
    table = Table(title="Unified Model Benchmark")
    table.add_column("Model", style="bold")

    # One column pair per suite
    for suite_name in result.suites:
        table.add_column(f"{suite_name}\nQ", justify="right", style="cyan")
        table.add_column(f"{suite_name}\nSpd", justify="right", style="dim")

    for model in result.models:
        row = [model]
        for suite_name in result.suites:
            # Find the suite result for this model
            sr = _find_suite_result(result, model, suite_name)
            if sr and sr.scores:
                # Average quality and speed across tasks in this suite
                avg_q = sum(s.quality for s in sr.scores) / len(sr.scores)
                avg_spd = sum(s.speed_s for s in sr.scores) / len(sr.scores)
                row.append(f"{avg_q:.2f}")
                row.append(f"{avg_spd:.1f}s")
            else:
                row.append("-")
                row.append("-")
        table.add_row(*row)

    console.print()
    console.print(table)


def print_suite_detail(suite_name: str, results: list[SuiteResult]) -> None:
    """Print detailed results for a single suite across models."""
    if not results:
        return

    # Get all task names from the first result
    task_names = []
    for sr in results:
        for score in sr.scores:
            if score.task_name not in task_names:
                task_names.append(score.task_name)

    if not task_names:
        return

    table = Table(title=f"Suite: {suite_name}")
    table.add_column("Model", style="bold")

    for task_name in task_names:
        table.add_column(f"{task_name}\nQ", justify="right", style="cyan")
        table.add_column(f"{task_name}\nSpd", justify="right", style="dim")
        table.add_column(f"{task_name}\nN", justify="right", style="dim")

    for sr in results:
        row = [sr.model]
        for task_name in task_names:
            score = _find_task_score(sr, task_name)
            if score:
                err_str = f" ({score.errors}err)" if score.errors else ""
                row.append(f"{score.quality:.2f}")
                row.append(f"{score.speed_s:.1f}s")
                row.append(f"{score.samples}{err_str}")
            else:
                row.extend(["-", "-", "-"])
        table.add_row(*row)

    console.print()
    console.print(table)


def save_results(result: BenchmarkResult, output_path: str | None = None) -> str:
    """Merge this run's results into the accumulated JSON file. Returns the path."""
    path = output_path or DEFAULT_OUTPUT_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing accumulated data
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Update metadata
    data["last_updated"] = result.timestamp or datetime.now(timezone.utc).isoformat()
    data.setdefault("results", {})

    # Merge: new results overwrite per model+suite, but prior model+suite combos are kept
    for sr in result.suite_results:
        if sr.model not in data["results"]:
            data["results"][sr.model] = {}
        suite_data = {}
        for score in sr.scores:
            suite_data[score.task_name] = {
                "quality": round(score.quality, 3),
                "speed_s": round(score.speed_s, 2),
                "samples": score.samples,
                "errors": score.errors,
                "detail": score.detail,
            }
        data["results"][sr.model][sr.suite_name] = suite_data

    # Rebuild models/suites lists from accumulated data
    data["models"] = sorted(data["results"].keys())
    all_suites: set[str] = set()
    for model_data in data["results"].values():
        all_suites.update(model_data.keys())
    data["suites"] = sorted(all_suites)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_accumulated(output_path: str | None = None) -> BenchmarkResult:
    """Load all accumulated results from JSON into a BenchmarkResult."""
    path = output_path or DEFAULT_OUTPUT_PATH
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return BenchmarkResult()

    results = data.get("results", {})
    all_models = sorted(results.keys())
    all_suites: set[str] = set()
    suite_results: list[SuiteResult] = []

    for model, suites in results.items():
        for suite_name, tasks in suites.items():
            all_suites.add(suite_name)
            scores = []
            for task_name, task_data in tasks.items():
                scores.append(TaskScore(
                    task_name=task_name,
                    quality=task_data.get("quality", 0),
                    speed_s=task_data.get("speed_s", 0),
                    samples=task_data.get("samples", 0),
                    errors=task_data.get("errors", 0),
                    detail=task_data.get("detail", {}),
                ))
            suite_results.append(SuiteResult(
                suite_name=suite_name, model=model, scores=scores,
            ))

    return BenchmarkResult(
        suite_results=suite_results,
        models=all_models,
        suites=sorted(all_suites),
        timestamp=data.get("last_updated", ""),
    )


def save_incremental(
    model: str, suite_result: SuiteResult, output_path: str | None = None,
) -> None:
    """Append one model's suite result to the JSON file (crash-safe)."""
    path = output_path or DEFAULT_OUTPUT_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Load existing or start fresh
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"timestamp": datetime.now(timezone.utc).isoformat(), "results": {}}

    if model not in data["results"]:
        data["results"][model] = {}

    suite_data = {}
    for score in suite_result.scores:
        suite_data[score.task_name] = {
            "quality": round(score.quality, 3),
            "speed_s": round(score.speed_s, 2),
            "samples": score.samples,
            "errors": score.errors,
            "detail": score.detail,
        }
    data["results"][model][suite_result.suite_name] = suite_data

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Per-role table (Section 4.5 of benchmark-spec.md)
# ---------------------------------------------------------------------------

# Ordered list of (column_header, task_name, suite_name) for the role table.
# Column headers are short labels matching the spec's terminal table format.
ROLE_COLUMNS: list[tuple[str, str, str]] = [
    ("summ",  "summarization",      "pipeline"),
    ("score", "scoring",            "pipeline"),
    ("dedup", "dedup",              "pipeline"),
    ("contr", "contradiction",      "pipeline"),
    ("lssn",  "lesson",             "pipeline"),
    ("ent",   "entity_extraction",  "extraction"),
    ("e_res", "entity_resolution",  "extraction"),
    ("tag_d", "tag_discovery",      "extraction"),
    ("b_tag", "batch_tags",         "extraction"),
    ("refl",  "reflection",         "synthesis"),
    ("title", "titling",            "synthesis"),
    ("dgst",  "digest",             "synthesis"),
    ("s_rfl", "self_reflection",    "synthesis"),
    ("plan",  "plan_gen",           "synthesis"),
    ("chat",  "tool_calling",       "interactive"),
    ("code",  "coding",             "interactive"),
]


def print_role_table(
    result: BenchmarkResult,
    *,
    suites_filter: list[str] | None = None,
) -> None:
    """Print per-role comparison table matching benchmark-spec.md Section 4.5.

    Each column is one LLM role. Rows are models. Final columns are AVG and avg_s.
    If suites_filter is set, only show roles from those suites.
    """
    # Filter columns to requested suites
    if suites_filter:
        cols = [c for c in ROLE_COLUMNS if c[2] in suites_filter]
    else:
        cols = list(ROLE_COLUMNS)

    if not cols:
        return

    table = Table(title="Unified LLM Benchmark — Per-Role Scores")
    table.add_column("Model", style="bold", min_width=14)

    for header, _, _ in cols:
        table.add_column(header, justify="right", style="cyan", min_width=5)

    table.add_column("AVG", justify="right", style="bold green", min_width=5)
    table.add_column("avg_s", justify="right", style="dim", min_width=5)

    # Build a lookup: (model, task_name) -> TaskScore
    score_map: dict[tuple[str, str], TaskScore] = {}
    for sr in result.suite_results:
        for sc in sr.scores:
            score_map[(sr.model, sc.task_name)] = sc

    for model in result.models:
        row = [model]
        qualities = []
        speeds = []

        for _, task_name, _ in cols:
            sc = score_map.get((model, task_name))
            if sc is not None and sc.samples > 0:
                row.append(f"{sc.quality:.2f}")
                qualities.append(sc.quality)
                speeds.append(sc.speed_s)
            elif sc is not None:
                # Stub (0 samples) — show the score but dimmed
                row.append(f"[dim]{sc.quality:.2f}[/dim]")
                qualities.append(sc.quality)
                speeds.append(sc.speed_s)
            else:
                row.append("—")

        # AVG and avg_s
        if qualities:
            avg_q = sum(qualities) / len(qualities)
            avg_s = sum(speeds) / len(speeds)
            row.append(f"{avg_q:.2f}")
            row.append(f"{avg_s:.1f}")
        else:
            row.append("—")
            row.append("—")

        table.add_row(*row)

    console.print()
    console.print(table)


def print_role_detail(
    result: BenchmarkResult,
    role: str,
) -> None:
    """Print per-case breakdown for a single role across all models.

    The role arg can be a task_name (e.g. 'lesson') or a column header
    (e.g. 'lssn'). Prints each case's checks and score.
    """
    # Resolve role: accept either column header or task_name
    task_name = role
    for header, tname, _ in ROLE_COLUMNS:
        if role == header:
            task_name = tname
            break

    # Find matching TaskScores across models
    found_any = False
    for model in result.models:
        # Find the TaskScore for this model + task_name
        sc = None
        for sr in result.suite_results:
            if sr.model != model:
                continue
            for s in sr.scores:
                if s.task_name == task_name:
                    sc = s
                    break
            if sc:
                break

        if not sc or "cases" not in sc.detail:
            continue

        cases = sc.detail["cases"]
        if not cases:
            continue

        found_any = True

        table = Table(title=f"Detail: {task_name} ({model}) - {len(cases)} cases, "
                      f"avg={sc.quality:.2f}")
        table.add_column("#", style="dim", width=3)
        table.add_column("Case", style="bold", max_width=45, overflow="fold")
        table.add_column("Output", max_width=50, overflow="fold")
        table.add_column("Checks", style="cyan", max_width=40, overflow="fold")
        table.add_column("Score", justify="right", style="green", width=5)

        for i, case in enumerate(cases, 1):
            # Case identifier: use id, summary, or project — whatever is available
            case_id = (case.get("id") or case.get("summary")
                       or case.get("project") or f"case_{i}")
            if isinstance(case_id, str) and len(case_id) > 45:
                case_id = case_id[:42] + "..."

            # Output snippet
            output = case.get("output") or case.get("title") or ""
            if len(output) > 50:
                output = output[:47] + "..."

            # Checks: format as key=val pairs
            checks = case.get("checks", {})
            checks_str = " ".join(f"{k}={v}" for k, v in checks.items())

            score = case.get("score", 0)

            # Color score red if 0, yellow if partial, green if perfect
            if score >= 1.0:
                score_str = f"[green]{score:.2f}[/green]"
            elif score > 0:
                score_str = f"[yellow]{score:.2f}[/yellow]"
            else:
                score_str = f"[red]{score:.2f}[/red]"

            table.add_row(str(i), str(case_id), output, checks_str, score_str)

        console.print()
        console.print(table)

    if not found_any:
        console.print(f"[yellow]No detail data found for role '{role}'. "
                      f"Valid roles: {', '.join(h for h, _, _ in ROLE_COLUMNS)}[/yellow]")


def save_role_results(
    result: BenchmarkResult,
    output_path: str = "data/benchmark_unified.json",
) -> str:
    """Save results in the Section 4.6 JSON format: model -> role -> {score, avg_time, ...}."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "meta": {
            "timestamp": result.timestamp or datetime.now(timezone.utc).isoformat(),
            "suites": result.suites,
        },
        "models": {},
    }

    for model in result.models:
        model_data: dict = {}
        for sr in result.suite_results:
            if sr.model != model:
                continue
            for sc in sr.scores:
                model_data[sc.task_name] = {
                    "score": round(sc.quality, 3),
                    "avg_time": round(sc.speed_s, 2),
                    "cases": sc.samples,
                    "errors": sc.errors,
                    "detail": sc.detail,
                }
        data["models"][model] = model_data

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def _find_suite_result(
    result: BenchmarkResult, model: str, suite_name: str,
) -> SuiteResult | None:
    for sr in result.suite_results:
        if sr.model == model and sr.suite_name == suite_name:
            return sr
    return None


def _find_task_score(sr: SuiteResult, task_name: str) -> TaskScore | None:
    for s in sr.scores:
        if s.task_name == task_name:
            return s
    return None
