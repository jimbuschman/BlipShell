"""Coding Model Benchmark — compare Ollama models on real codebase tasks.

Copies the real BlipShell codebase to a sandbox, runs each model through
realistic coding tasks via TaskExecutor.execute_dynamic(), and verifies
results with automated checks (syntax, diff, pytest, functional tests).

Reuses the tested infrastructure from tests/benchmark_coding.py. Adds:
  - Auto-detect available Ollama models
  - Resume support (skip completed models)
  - Scorecard + recommendations
  - Proper argparse CLI

Usage:
    python scripts/benchmark_coding.py                              # test all available models
    python scripts/benchmark_coding.py --models qwen3:14b glm-5:cloud
    python scripts/benchmark_coding.py --resume                     # skip completed models
    python scripts/benchmark_coding.py --timeout 600                # longer timeout
    python scripts/benchmark_coding.py --dry-run-verify             # test checks on unmodified code
    python scripts/benchmark_coding.py --tasks stats_command dry_run_edit  # specific tasks only
"""

import argparse
import asyncio
import io
import json
import shutil
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
elif not sys.stdout.isatty():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from blipshell.llm.client import LLMClient

# Import tested infrastructure from existing benchmark
from tests.benchmark_coding import (
    CODING_TASKS,
    TaskMetrics,
    build_project_context,
    create_project_sandbox,
    dry_run_verify,
    make_router,
    reset_sandbox,
    run_task,
)

console = Console()

OLLAMA_URL = "http://localhost:11434"
OUTPUT_PATH = Path("data") / "benchmark_coding_results.json"


# ============================================================================
# MODEL DETECTION
# ============================================================================

async def list_available_models(ollama_url: str = OLLAMA_URL) -> list[str]:
    """Get list of models available on Ollama."""
    client = LLMClient(host=ollama_url)
    try:
        return await client.list_models()
    except Exception as e:
        console.print(f"[red]Could not list models from {ollama_url}: {e}[/red]")
        return []


async def warmup_model(model_name: str):
    """Send a throwaway request to load the model into memory."""
    from blipshell.llm.router import TaskType
    router = make_router(model_name)
    try:
        await asyncio.wait_for(
            router.generate(TaskType.CODING, "Say hello.", system="Be brief.", think=False),
            timeout=120,
        )
    except Exception:
        pass


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_results(all_raw: dict, path: Path):
    """Save results to JSON (atomic write)."""
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, indent=2, default=str)
    tmp.replace(path)


def load_results(path: Path) -> dict:
    """Load previous results. Returns {model_name: [task_dicts]}."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]Warning: could not load previous results: {e}[/yellow]")
        return {}


def get_completed_models(results: dict, task_names: list[str]) -> set[str]:
    """Return model names that have completed all requested tasks."""
    completed = set()
    for model_name, task_results in results.items():
        if model_name.startswith("_"):
            continue
        completed_tasks = {r.get("task_name") for r in task_results if isinstance(r, dict)}
        if all(t in completed_tasks for t in task_names):
            completed.add(model_name)
    return completed


# ============================================================================
# DISPLAY
# ============================================================================

def print_summary_table(all_raw: dict, task_names: list[str]):
    """Print comparison table across all models and tasks."""
    table = Table(title="CODING BENCHMARK — SUMMARY", show_lines=True, title_style="bold magenta")
    table.add_column("Model", style="cyan", min_width=25)
    table.add_column("Task", min_width=18)
    table.add_column("Checks", justify="center", min_width=8)
    table.add_column("Tools", justify="center", min_width=6)
    table.add_column("Edit Fail", justify="center", min_width=9)
    table.add_column("Time", justify="right", min_width=8)
    table.add_column("Pytest", justify="center", min_width=7)
    table.add_column("Error", min_width=10)

    for model_name, task_results in all_raw.items():
        if model_name.startswith("_"):
            continue
        for i, result in enumerate(task_results):
            if not isinstance(result, dict):
                continue
            if result.get("task_name") not in task_names:
                continue

            model_col = model_name if i == 0 else ""
            checks = f"{result.get('checks_passed', 0)}/{result.get('checks_total', 0)}"
            tools = str(result.get("total_tool_calls", 0))
            edit_fail = result.get("total_edit_failures", 0)
            edit_str = f"[red]{edit_fail}[/red]" if edit_fail > 0 else "[green]0[/green]"
            total_time = f"{result.get('total_time', 0):.0f}s"
            error = result.get("error", "")

            # Pytest from check details
            pytest_str = "[dim]n/a[/dim]"
            for check in result.get("check_details", []):
                if isinstance(check, dict) and "pytest" in check.get("check", ""):
                    pytest_str = "[green]PASS[/green]" if check.get("passed") else "[red]FAIL[/red]"
                    break

            error_str = f"[red]{error[:30]}[/red]" if error else ""

            table.add_row(model_col, result.get("task_name", "?"), checks, tools,
                          edit_str, total_time, pytest_str, error_str)

    console.print(table)


def print_scorecard(all_raw: dict, task_names: list[str]):
    """Print per-model scorecard: check pass rate, speed, efficiency."""
    table = Table(title="CODING SCORECARD", show_lines=True, title_style="bold green")
    table.add_column("Model", style="cyan", min_width=25)
    table.add_column("Check Rate", justify="right", min_width=10)
    table.add_column("Avg Time", justify="right", min_width=9)
    table.add_column("Avg Tools", justify="right", min_width=9)
    table.add_column("Edit Fails", justify="right", min_width=10)
    table.add_column("Errors", justify="right", min_width=7)

    model_scores = {}  # for recommendations

    for model_name, task_results in all_raw.items():
        if model_name.startswith("_"):
            continue

        relevant = [r for r in task_results
                    if isinstance(r, dict) and r.get("task_name") in task_names]
        if not relevant:
            continue

        checks_passed = sum(r.get("checks_passed", 0) for r in relevant)
        checks_total = sum(r.get("checks_total", 0) for r in relevant)
        check_rate = checks_passed / checks_total if checks_total > 0 else 0.0
        avg_time = sum(r.get("total_time", 0) for r in relevant) / len(relevant)
        avg_tools = sum(r.get("total_tool_calls", 0) for r in relevant) / len(relevant)
        edit_fails = sum(r.get("total_edit_failures", 0) for r in relevant)
        errors = sum(1 for r in relevant if r.get("error"))

        model_scores[model_name] = {
            "check_rate": check_rate,
            "avg_time": avg_time,
            "avg_tools": avg_tools,
            "edit_fails": edit_fails,
            "errors": errors,
        }

        rate_str = f"{check_rate * 100:.0f}%"
        if check_rate >= 0.8:
            rate_str = f"[green]{rate_str}[/green]"
        elif check_rate >= 0.5:
            rate_str = f"[yellow]{rate_str}[/yellow]"
        else:
            rate_str = f"[red]{rate_str}[/red]"

        error_str = f"[red]{errors}[/red]" if errors > 0 else "[green]0[/green]"
        edit_str = f"[red]{edit_fails}[/red]" if edit_fails > 0 else "[green]0[/green]"

        table.add_row(
            model_name,
            rate_str,
            f"{avg_time:.0f}s",
            f"{avg_tools:.0f}",
            edit_str,
            error_str,
        )

    console.print(table)
    return model_scores


def print_recommendations(model_scores: dict):
    """Print recommendations based on scores."""
    if not model_scores:
        return

    console.print("\n[bold]RECOMMENDATIONS[/bold]")
    console.print("-" * 60)

    # Best check rate
    best_quality = max(model_scores.items(), key=lambda x: x[1]["check_rate"])
    console.print(
        f"  Best quality:  [cyan]{best_quality[0]}[/cyan]  "
        f"({best_quality[1]['check_rate'] * 100:.0f}% checks passed)"
    )

    # Fastest (among models with >50% check rate)
    viable = {k: v for k, v in model_scores.items() if v["check_rate"] >= 0.5}
    if viable:
        fastest = min(viable.items(), key=lambda x: x[1]["avg_time"])
        console.print(
            f"  Fastest (>50%): [cyan]{fastest[0]}[/cyan]  "
            f"({fastest[1]['avg_time']:.0f}s avg, {fastest[1]['check_rate'] * 100:.0f}% checks)"
        )

    # Most efficient (fewest tool calls among >50% check rate)
    if viable:
        efficient = min(viable.items(), key=lambda x: x[1]["avg_tools"])
        console.print(
            f"  Most efficient: [cyan]{efficient[0]}[/cyan]  "
            f"({efficient[1]['avg_tools']:.0f} avg tool calls, "
            f"{efficient[1]['check_rate'] * 100:.0f}% checks)"
        )

    # Best overall: highest check rate, tie-break by speed
    if viable:
        best = max(
            viable.items(),
            key=lambda x: (x[1]["check_rate"], -x[1]["avg_time"]),
        )
        console.print(
            f"\n  [bold green]Best overall: {best[0]} "
            f"({best[1]['check_rate'] * 100:.0f}%, {best[1]['avg_time']:.0f}s avg)[/bold green]"
        )


def print_check_details(all_raw: dict, task_names: list[str]):
    """Print per-check pass/fail for each model×task."""
    for model_name, task_results in all_raw.items():
        if model_name.startswith("_"):
            continue
        for result in task_results:
            if not isinstance(result, dict):
                continue
            if result.get("task_name") not in task_names:
                continue
            console.print(f"\n  [cyan]{model_name}[/cyan] / {result['task_name']}:")
            for check in result.get("check_details", []):
                if isinstance(check, dict):
                    mark = "[green]PASS[/green]" if check.get("passed") else "[red]FAIL[/red]"
                    console.print(f"    {mark} {check.get('check', '?')}: {check.get('reason', '')[:80]}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Coding model benchmark — compare Ollama models on real codebase tasks"
    )
    parser.add_argument("--models", nargs="*", help="Models to test (default: all available)")
    parser.add_argument("--tasks", nargs="*",
                        choices=[t["name"] for t in CODING_TASKS],
                        help="Tasks to run (default: all)")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Timeout per LLM call in seconds (default: 300)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results (skip completed models)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output JSON path")
    parser.add_argument("--dry-run-verify", action="store_true",
                        help="Test verification checks on unmodified sandbox (no LLM)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-check details in output")
    args = parser.parse_args()

    output_path = Path(args.output)

    # Dry-run mode: just test the verification checks
    if args.dry_run_verify:
        await dry_run_verify()
        return

    # Determine tasks
    if args.tasks:
        tasks = [t for t in CODING_TASKS if t["name"] in args.tasks]
    else:
        tasks = CODING_TASKS
    task_names = [t["name"] for t in tasks]

    console.rule("[bold]Coding Model Benchmark[/bold]")
    console.print(f"Tasks: {', '.join(task_names)}")
    console.print(f"Timeout: {args.timeout}s per LLM call")
    console.print(f"Output: {output_path}")

    # Load previous results for resume
    all_raw = load_results(output_path) if args.resume else {}
    if args.resume and all_raw:
        completed = get_completed_models(all_raw, task_names)
        if completed:
            console.print(f"[yellow]Resuming: {len(completed)} model(s) already completed: "
                          f"{', '.join(completed)}[/yellow]")

    # Determine models
    if args.models:
        models_to_test = args.models
    else:
        console.print(f"\nQuerying available models on {OLLAMA_URL}...")
        available = await list_available_models()
        if not available:
            console.print("[red]No models found. Is Ollama running?[/red]")
            return
        console.print(f"Found {len(available)} models: {', '.join(available[:10])}")
        if len(available) > 10:
            console.print(f"  ... and {len(available) - 10} more")
        models_to_test = available

    # Filter completed models
    if args.resume:
        completed = get_completed_models(all_raw, task_names)
        skip = [m for m in models_to_test if m in completed]
        if skip:
            console.print(f"[yellow]Skipping {len(skip)} completed: {', '.join(skip)}[/yellow]")
        models_to_test = [m for m in models_to_test if m not in completed]

    if not models_to_test:
        console.print("[green]All models already completed! Use without --resume to re-run.[/green]")
    else:
        console.print(f"\nTesting {len(models_to_test)} model(s): {', '.join(models_to_test)}")
        console.print(f"Tasks per model: {len(tasks)} ({', '.join(task_names)})")

        est_minutes = len(models_to_test) * len(tasks) * 3  # rough: ~3 min per task
        console.print(f"Estimated time: ~{est_minutes} minutes\n")

        # Create sandbox once (reset between tasks)
        console.print("[dim]Creating project sandbox (copying BlipShell source)...[/dim]")
        sandbox_path = create_project_sandbox()
        console.print(f"[dim]Sandbox: {sandbox_path}[/dim]")

        console.print("[dim]Building project context...[/dim]")
        project_context = build_project_context(sandbox_path)
        console.print(f"[dim]Project context: {len(project_context)} chars[/dim]\n")

        try:
            for i, model_name in enumerate(models_to_test, 1):
                console.rule(f"[bold cyan]{model_name}[/bold cyan] ({i}/{len(models_to_test)})")

                # Warmup
                console.print("  Warming up...", end=" ")
                await warmup_model(model_name)
                console.print("[green]ready[/green]")

                model_results = []
                model_ok = True

                for task in tasks:
                    console.print(f"\n  [bold]{task['name']}[/bold]: {task['description']}")

                    try:
                        metrics = await run_task(
                            model_name, task, sandbox_path, project_context,
                            timeout=args.timeout,
                        )

                        if metrics.error:
                            console.print(f"    [red]ERROR: {metrics.error[:100]}[/red]")
                        else:
                            console.print(
                                f"    [green]Done[/green] — "
                                f"checks {metrics.checks_passed}/{metrics.checks_total}, "
                                f"{metrics.total_tool_calls} tools, "
                                f"{metrics.total_edit_failures} edit fails, "
                                f"{metrics.total_time:.0f}s"
                            )
                            # Show check results
                            for label, ok, reason in metrics.check_details:
                                mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
                                console.print(f"      {mark} {label}: {reason[:70]}")

                        model_results.append(metrics.to_dict())

                    except Exception as e:
                        console.print(f"    [red]CRASHED: {e}[/red]")
                        model_results.append({
                            "task_name": task["name"],
                            "error": str(e),
                            "checks_passed": 0,
                            "checks_total": len(task.get("verify_checks", [])),
                            "total_tool_calls": 0,
                            "total_edit_failures": 0,
                            "total_time": 0,
                            "check_details": [],
                        })

                # Save after each model
                all_raw[model_name] = model_results
                save_results(all_raw, output_path)
                console.print(f"\n  [dim]Saved to {output_path}[/dim]")

        finally:
            # Always clean up sandbox
            try:
                shutil.rmtree(sandbox_path, ignore_errors=True)
                console.print(f"\n[dim]Sandbox cleaned up[/dim]")
            except Exception:
                pass

    # Print results (including previously loaded)
    if all_raw:
        console.print()
        console.rule("[bold]Results[/bold]")
        console.print()
        print_summary_table(all_raw, task_names)
        console.print()
        scores = print_scorecard(all_raw, task_names)
        print_recommendations(scores)

        if args.verbose:
            console.print()
            print_check_details(all_raw, task_names)

        console.print(f"\n[dim]Full results: {output_path}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
