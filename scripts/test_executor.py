"""Headless test harness for BlipShell executor.

Bootstraps the full Agent, runs a task through agent.chat(force_plan=True),
captures tool calls / errors / timing, and outputs a structured JSON report.

Usage:
    python scripts/test_executor.py "create a hello world script"
    python scripts/test_executor.py "task" --project blipshell
    python scripts/test_executor.py "task" --output results.json
    python scripts/test_executor.py --canned               # built-in test suite
    python scripts/test_executor.py --canned --quiet        # JSON only, no streaming
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console

from blipshell.core.agent import Agent
from blipshell.core.config import ConfigManager

console = Console(stderr=True)  # Rich output to stderr so JSON goes to stdout cleanly

# ---------------------------------------------------------------------------
# Built-in canned tests
# ---------------------------------------------------------------------------

CANNED_TESTS = [
    {
        "name": "file_creation",
        "task": (
            "Create a Python file called test_hello.py in the project root "
            "that prints 'Hello from BlipShell test harness'. "
            "Then verify it has correct syntax by running: python -c \"import ast; ast.parse(open('test_hello.py').read()); print('OK')\""
        ),
        "expect_tools": ["write_file"],
        "expect_complete": True,
    },
    {
        "name": "read_and_grep",
        "task": (
            "Find all Python files that contain the string 'TaskExecutor' "
            "and report which files contain it and on which lines."
        ),
        "expect_tools": ["grep_files"],
        "expect_complete": True,
    },
    {
        "name": "read_edit_flow",
        "task": (
            "Read the file blipshell/core/tools/base.py, then add a one-line "
            "comment '# test harness marker' at the very end of the file."
        ),
        "expect_tools": ["read_file", "edit_file"],
        "expect_complete": True,
    },
]


# ---------------------------------------------------------------------------
# Streaming collector — captures on_token output and parses events
# ---------------------------------------------------------------------------

class StreamCollector:
    """Captures streaming output and extracts structured events."""

    def __init__(self, quiet: bool = False):
        self.raw_output: list[str] = []
        self.tool_calls: list[dict] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.quiet = quiet
        self._current_tool: str | None = None
        self._first_tool_time: float | None = None
        self._start_time: float = time.monotonic()

    def on_token(self, chunk: str):
        """Callback for agent.chat(on_token=...)."""
        self.raw_output.append(chunk)

        # Parse tool call markers
        tool_match = re.search(r"\[Tool: (\w+)\]", chunk)
        if tool_match:
            name = tool_match.group(1)
            self._current_tool = name
            if self._first_tool_time is None:
                self._first_tool_time = time.monotonic()
            self.tool_calls.append({"name": name, "success": True})

        # Parse tool results for errors
        result_match = re.search(r"\[Result: (.+?)\]", chunk)
        if result_match:
            result_text = result_match.group(1)
            if result_text.startswith("Error:"):
                self.errors.append(result_text[:200])
                if self.tool_calls:
                    self.tool_calls[-1]["success"] = False

        # Parse warnings
        if "[Budget warning" in chunk:
            self.warnings.append("Budget warning injected")
        if "[Forced completion" in chunk:
            self.warnings.append("Forced completion triggered")
        if "[Context compacted" in chunk:
            self.warnings.append(chunk.strip().strip("[]"))
        if "[Duplicate call blocked]" in chunk:
            self.warnings.append("Duplicate call blocked")
            if self.tool_calls:
                self.tool_calls[-1]["success"] = False

        # Parse errors in streaming output
        if "Error:" in chunk and "[Result:" not in chunk and "[Tool:" not in chunk:
            err = chunk.strip()
            if err and err not in self.errors:
                self.errors.append(err[:200])

        # Stream to stderr if not quiet
        if not self.quiet:
            console.print(chunk, end="", highlight=False)

    @property
    def first_tool_seconds(self) -> float:
        if self._first_tool_time is None:
            return 0.0
        return round(self._first_tool_time - self._start_time, 2)


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

async def run_test(
    task: str,
    project: str | None = None,
    output_path: str | None = None,
    config_path: str | None = None,
    quiet: bool = False,
) -> dict:
    """Run a single headless test and return structured results.

    Args:
        task: The task description to send to the agent.
        project: Optional project name to activate.
        output_path: Optional file path to write JSON results.
        config_path: Optional config.yaml path.
        quiet: If True, suppress streaming output (JSON only).

    Returns:
        Dict with structured test results.
    """
    # 1. Bootstrap
    if not quiet:
        console.print(f"[bold cyan]Bootstrapping agent...[/bold cyan]")

    config_manager = ConfigManager(config_path)
    config = config_manager.load()
    agent = Agent(config, config_manager)

    def on_status(msg: str):
        if not quiet:
            console.print(f"  [dim]{msg}[/dim]")

    await agent.initialize(on_status=on_status)

    # 2. Start session
    session_id = await agent.start_session(project=project)
    if not quiet:
        console.print(f"[bold cyan]Session #{session_id} started[/bold cyan]")

    # 3. Activate project if requested
    if project:
        try:
            await agent.activate_project(project)
            if not quiet:
                console.print(f"[bold cyan]Project '{project}' activated[/bold cyan]")
        except KeyError:
            console.print(f"[bold red]Project '{project}' not found — running without project context[/bold red]")
            project = None

    # 4. Set headless ask_user callback
    async def _headless_ask_user(question: str) -> str:
        return "Make your best judgment."

    agent.set_ask_user_callback(_headless_ask_user)

    # 5. Run the task
    collector = StreamCollector(quiet=quiet)
    start_time = time.monotonic()

    if not quiet:
        console.print(f"\n[bold green]Running task:[/bold green] {task}\n")
        console.print("[dim]" + "─" * 60 + "[/dim]")

    try:
        result = await agent.chat(
            user_message=task,
            on_token=collector.on_token,
            force_plan=True,
        )
    except Exception as e:
        result = f"FATAL ERROR: {e}"
        collector.errors.append(str(e))

    elapsed = round(time.monotonic() - start_time, 2)

    if not quiet:
        console.print("\n[dim]" + "─" * 60 + "[/dim]")

    # 6. Collect flow events from DB
    flow_events = []
    try:
        events = await agent.sqlite.get_turn_events(session_id, limit=50)
        for evt in events:
            flow_events.append({
                "turn": evt["turn_number"],
                "type": evt["event_type"],
                "data": evt["data"],
            })
    except Exception:
        pass

    # 7. Extract executor state
    executor = agent.task_executor
    files_read = sorted(executor.files_read) if executor else []
    files_created = list(executor._step_files_created) if executor else []
    files_edited = list(executor._step_files_edited) if executor else []

    # Determine completion method
    completed = False
    completion_method = "unknown"
    if "[Task complete signal received]" in "".join(collector.raw_output):
        completed = True
        completion_method = "task_complete"
    elif "[No tool calls — treating as complete]" in "".join(collector.raw_output):
        completed = True
        completion_method = "no_tool_calls"
    elif "FATAL ERROR" in (result or ""):
        completion_method = "error"

    # Find transcript path
    transcript_path = ""
    if executor and executor.last_messages:
        # Look for the most recent transcript file
        transcript_dir = Path("data/project_transcripts")
        if transcript_dir.exists():
            transcripts = sorted(transcript_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if transcripts:
                transcript_path = str(transcripts[-1])

    # Get model/endpoint from flow events
    model = "unknown"
    endpoint = "unknown"
    for evt in flow_events:
        if evt["type"] == "llm_complete":
            model = evt["data"].get("model", model)
            endpoint = evt["data"].get("endpoint", endpoint)
            break

    # 8. Build report
    report = {
        "task": task,
        "project": project,
        "model": model,
        "endpoint": endpoint,
        "session_id": session_id,
        "timing": {
            "total_seconds": elapsed,
            "first_tool_call_seconds": collector.first_tool_seconds,
        },
        "tool_calls": collector.tool_calls,
        "tool_call_count": len(collector.tool_calls),
        "errors": collector.errors,
        "warnings": collector.warnings,
        "files_read": files_read,
        "files_created": files_created,
        "files_edited": files_edited,
        "result": (result or "")[:2000],
        "completed": completed,
        "completion_method": completion_method,
        "flow_events": flow_events,
        "transcript_path": transcript_path,
    }

    # 9. Output
    report_json = json.dumps(report, indent=2, default=str)

    if output_path:
        Path(output_path).write_text(report_json, encoding="utf-8")
        if not quiet:
            console.print(f"\n[bold green]Report written to {output_path}[/bold green]")
    else:
        # Print JSON to stdout (Rich output goes to stderr)
        print(report_json)

    # 10. Summary to stderr
    if not quiet:
        console.print(f"\n[bold]Test Summary:[/bold]")
        console.print(f"  Completed: {'YES' if completed else 'NO'} ({completion_method})")
        console.print(f"  Tool calls: {len(collector.tool_calls)}")
        console.print(f"  Errors: {len(collector.errors)}")
        console.print(f"  Time: {elapsed}s")
        if files_created:
            console.print(f"  Files created: {', '.join(files_created)}")
        if files_edited:
            console.print(f"  Files edited: {', '.join(files_edited)}")

    # 11. Cleanup
    try:
        await agent.end_session()
    except Exception:
        pass
    try:
        if agent.sqlite:
            await agent.sqlite.close()
    except Exception:
        pass

    return report


# ---------------------------------------------------------------------------
# Canned test suite
# ---------------------------------------------------------------------------

async def run_canned_tests(
    project: str | None = None,
    config_path: str | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Run all built-in canned tests and return results."""
    results = []

    for i, test in enumerate(CANNED_TESTS, 1):
        console.print(f"\n[bold yellow]{'=' * 60}[/bold yellow]")
        console.print(f"[bold yellow]Canned Test {i}/{len(CANNED_TESTS)}: {test['name']}[/bold yellow]")
        console.print(f"[bold yellow]{'=' * 60}[/bold yellow]\n")

        report = await run_test(
            task=test["task"],
            project=project,
            config_path=config_path,
            quiet=quiet,
        )

        # Check expectations
        passed = True
        checks = []

        if test.get("expect_complete"):
            ok = report["completed"]
            checks.append(("completed", ok))
            if not ok:
                passed = False

        for expected_tool in test.get("expect_tools", []):
            found = any(tc["name"] == expected_tool for tc in report["tool_calls"])
            checks.append((f"used_{expected_tool}", found))
            if not found:
                passed = False

        no_errors = len(report["errors"]) == 0
        checks.append(("no_errors", no_errors))
        if not no_errors:
            passed = False

        report["_test_name"] = test["name"]
        report["_checks"] = checks
        report["_passed"] = passed
        results.append(report)

        status = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        console.print(f"\n{status} — {test['name']}")
        for check_name, check_ok in checks:
            icon = "[green]OK[/green]" if check_ok else "[red]FAIL[/red]"
            console.print(f"  {icon} {check_name}")

    # Final summary
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    total = len(results)
    passed = sum(1 for r in results if r["_passed"])
    console.print(f"[bold]Results: {passed}/{total} passed[/bold]")
    for r in results:
        status = "[green]PASS[/green]" if r["_passed"] else "[red]FAIL[/red]"
        console.print(f"  {status} {r['_test_name']} ({r['timing']['total_seconds']}s, {r['tool_call_count']} tools)")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Headless test harness for BlipShell executor",
    )
    parser.add_argument("task", nargs="?", help="Task description to execute")
    parser.add_argument("--project", "-p", default=None, help="Project to activate")
    parser.add_argument("--output", "-o", default=None, help="Write JSON report to file")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--canned", action="store_true", help="Run built-in test suite")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress streaming output")

    args = parser.parse_args()

    if args.canned:
        asyncio.run(run_canned_tests(
            project=args.project,
            config_path=args.config,
            quiet=args.quiet,
        ))
    elif args.task:
        asyncio.run(run_test(
            task=args.task,
            project=args.project,
            output_path=args.output,
            config_path=args.config,
            quiet=args.quiet,
        ))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
