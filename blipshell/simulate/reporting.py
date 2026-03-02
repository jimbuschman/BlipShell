"""Report formatting for simulation results — Rich console + JSON export."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import IO

from rich.console import Console
from rich.table import Table

from blipshell.simulate.models import (
    ResultStatus,
    SimScenarioResult,
    SimStepResult,
    SimSuiteResult,
)


def _status_style(status: ResultStatus) -> str:
    if status == ResultStatus.PASS:
        return "green"
    elif status == ResultStatus.WARN:
        return "yellow"
    return "red"


def _status_icon(status: ResultStatus) -> str:
    if status == ResultStatus.PASS:
        return "PASS"
    elif status == ResultStatus.WARN:
        return "WARN"
    return "FAIL"


def print_step_result(con: Console, sr: SimStepResult):
    """Print one step result with details on failure."""
    icon = _status_icon(sr.status)
    style = _status_style(sr.status)
    desc = sr.description or f"step {sr.step_index}"
    con.print(f"  [{style}]{icon}[/{style}] {desc} ({sr.elapsed_seconds:.1f}s)")

    if sr.hard_failures:
        for f in sr.hard_failures:
            con.print(f"    [red]HARD: {f}[/red]")
    if sr.soft_failures:
        for f in sr.soft_failures:
            con.print(f"    [yellow]SOFT: {f}[/yellow]")
    if sr.error:
        con.print(f"    [red]ERROR: {sr.error}[/red]")


def print_scenario_result(con: Console, result: SimScenarioResult):
    """Print full scenario result with all steps."""
    icon = _status_icon(result.status)
    style = _status_style(result.status)
    con.print(f"\n[{style} bold]{icon}: {result.name}[/{style} bold] ({result.elapsed_seconds:.1f}s)")

    if result.error:
        con.print(f"  [red]Scenario error: {result.error}[/red]")

    for sr in result.step_results:
        print_step_result(con, sr)


def print_suite_summary(con: Console, suite: SimSuiteResult):
    """Print suite-level summary table."""
    con.print()

    table = Table(title="Simulation Results")
    table.add_column("Scenario", style="bold")
    table.add_column("Category")
    table.add_column("Status")
    table.add_column("Steps")
    table.add_column("Time", justify="right")
    table.add_column("Failures", justify="right")

    for r in suite.scenario_results:
        style = _status_style(r.status)
        failures = sum(len(sr.hard_failures) for sr in r.step_results)
        warns = sum(len(sr.soft_failures) for sr in r.step_results)
        fail_str = ""
        if failures:
            fail_str = f"{failures} hard"
        if warns:
            fail_str += f"{', ' if fail_str else ''}{warns} soft"

        table.add_row(
            r.name,
            r.category,
            f"[{style}]{_status_icon(r.status)}[/{style}]",
            str(len(r.step_results)),
            f"{r.elapsed_seconds:.1f}s",
            fail_str or "-",
        )

    con.print(table)

    # Summary line
    con.print(
        f"\n[bold]Total: {suite.total} scenarios | "
        f"[green]{suite.passed} passed[/green] | "
        f"[yellow]{suite.warned} warned[/yellow] | "
        f"[red]{suite.failed} failed[/red] | "
        f"{suite.elapsed_seconds:.1f}s[/bold]"
    )


def export_json(suite: SimSuiteResult, file: IO[str] | None = None) -> str:
    """Export suite result as JSON. Returns JSON string."""

    def _serialize(obj):
        """Custom serializer for non-serializable types."""
        if hasattr(obj, "value"):
            return obj.value  # enums
        raise TypeError(f"Not serializable: {type(obj)}")

    data = {
        "elapsed_seconds": suite.elapsed_seconds,
        "summary": {
            "total": suite.total,
            "passed": suite.passed,
            "warned": suite.warned,
            "failed": suite.failed,
        },
        "scenarios": [],
    }

    for r in suite.scenario_results:
        scenario_data = {
            "name": r.name,
            "category": r.category,
            "status": r.status.value,
            "elapsed_seconds": r.elapsed_seconds,
            "error": r.error,
            "steps": [],
        }
        for sr in r.step_results:
            scenario_data["steps"].append({
                "index": sr.step_index,
                "description": sr.description,
                "action": sr.action.value,
                "status": sr.status.value,
                "elapsed_seconds": sr.elapsed_seconds,
                "tools_called": sr.tools_called,
                "tool_call_count": sr.tool_call_count,
                "hard_failures": sr.hard_failures,
                "soft_failures": sr.soft_failures,
                "error": sr.error,
            })
        data["scenarios"].append(scenario_data)

    json_str = json.dumps(data, indent=2, default=_serialize)

    if file:
        file.write(json_str)

    return json_str
