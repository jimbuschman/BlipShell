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
