"""Headless test harness for BlipShell executor.

Bootstraps the full Agent, runs a task through agent.chat(force_plan=True),
captures tool calls / errors / timing, and outputs a structured JSON report.

Usage:
    python scripts/test_executor.py "create a hello world script"
    python scripts/test_executor.py "task" --project blipshell
    python scripts/test_executor.py "task" --output results.json
    python scripts/test_executor.py --canned               # quick built-in tests (~5 min)
    python scripts/test_executor.py --stress               # full stress suite (~1-2 hours)
    python scripts/test_executor.py --stress --quiet        # overnight, JSON only
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
# Comprehensive stress test suite — exercises every tool and edge case
# ---------------------------------------------------------------------------

STRESS_TESTS = [
    # ── Category 1: Tool Coverage ──────────────────────────────────────────
    {
        "name": "tool_write_file",
        "category": "tool_coverage",
        "task": (
            "Create a Python file called stress_test_output.py that contains a "
            "function called greet(name) which returns f'Hello, {name}!'. "
            "Include a __main__ block that calls greet('World') and prints the result."
        ),
        "expect_tools": ["write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "tool_read_file",
        "category": "tool_coverage",
        "task": (
            "Read the file pyproject.toml and tell me the project name, version, "
            "and what build system it uses."
        ),
        "expect_tools": ["read_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "tool_edit_file",
        "category": "tool_coverage",
        "task": (
            "Read the file blipshell/__init__.py, then use edit_file to add a "
            "comment '# Stress test edit marker' as the very last line."
        ),
        "expect_tools": ["read_file", "edit_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "tool_list_directory",
        "category": "tool_coverage",
        "task": (
            "List the contents of the blipshell/core/tools/ directory and tell me "
            "how many Python files are in it and what each file likely does based on its name."
        ),
        "expect_tools": ["list_directory"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "tool_grep_files",
        "category": "tool_coverage",
        "task": (
            "Search the codebase for all files that contain 'async def execute' "
            "and list each file with the line numbers where it appears."
        ),
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "tool_glob_files",
        "category": "tool_coverage",
        "task": (
            "Find all Python test files (matching the pattern tests/test_*.py) "
            "and list them with their sizes."
        ),
        "expect_tools": ["glob_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "tool_run_command",
        "category": "tool_coverage",
        "task": (
            "Run 'python --version' using the shell command tool and report "
            "what Python version is installed."
        ),
        "expect_tools": ["run_command"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "tool_git_status",
        "category": "tool_coverage",
        "task": (
            "Check the git status of this project and tell me what branch we're on "
            "and if there are any uncommitted changes."
        ),
        "expect_tools": ["git_status"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ── Category 2: Multi-step Tasks ───────────────────────────────────────
    {
        "name": "multi_read_then_create",
        "category": "multi_step",
        "task": (
            "Read blipshell/core/tools/shell.py and blipshell/core/tools/code_tools.py. "
            "Then create a new file called stress_test_summary.txt that lists: "
            "1) The class names defined in each file, "
            "2) How many methods each class has, "
            "3) Whether each file imports asyncio."
        ),
        "expect_tools": ["read_file", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "multi_grep_read_edit",
        "category": "multi_step",
        "task": (
            "First, use grep_files to find which file contains the class 'StreamCollector'. "
            "Then read that file. Then use edit_file to add a one-line comment "
            "'# Found by stress test' right above the class definition line."
        ),
        "expect_tools": ["grep_files", "read_file", "edit_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "multi_explore_and_report",
        "category": "multi_step",
        "task": (
            "Explore the blipshell/core/ directory structure: list the directory, "
            "then read the first 30 lines of at least 3 different .py files in it. "
            "Create a file called stress_test_exploration.txt summarizing what each "
            "file does based on its docstring or first few lines."
        ),
        "expect_tools": ["list_directory", "read_file", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ── Category 3: Error Recovery ─────────────────────────────────────────
    {
        "name": "error_nonexistent_file",
        "category": "error_recovery",
        "task": (
            "Try to read the file 'this_file_does_not_exist_12345.py'. "
            "When it fails, create the file instead with the content "
            "'# Created because the original was missing'."
        ),
        "expect_tools": ["read_file", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "error_syntax_fix",
        "category": "error_recovery",
        "task": (
            "Create a Python file called stress_test_broken.py with this exact content:\n"
            "def add(a, b)\n"
            "    return a + b\n\n"
            "Then run 'python -c \"import ast; ast.parse(open(\\\"stress_test_broken.py\\\").read())\"' "
            "to check syntax. It will fail because the colon is missing. "
            "Fix the syntax error using edit_file and verify it passes."
        ),
        "expect_tools": ["write_file", "run_command", "edit_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "error_edit_wrong_text",
        "category": "error_recovery",
        "task": (
            "Read the file blipshell/models/tools.py. Then try to edit it by replacing "
            "the text 'THIS_STRING_DOES_NOT_EXIST_IN_THE_FILE' with 'replaced'. "
            "When the edit fails (because that text isn't in the file), "
            "report what happened and call task_complete."
        ),
        "expect_tools": ["read_file", "edit_file"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ── Category 4: Scaffolding & State Awareness ──────────────────────────
    {
        "name": "scaffold_no_reread",
        "category": "scaffolding",
        "task": (
            "Read the file blipshell/core/config.py. Then WITHOUT re-reading it, "
            "answer these questions based on what you already read: "
            "1) What class does it define? "
            "2) Does it use pydantic? "
            "3) What is the default config file name? "
            "Do NOT read the file again — use what you already have."
        ),
        "expect_tools": ["read_file"],
        "expect_complete": True,
        "force_plan": True,
        # Success: should only have 1 read_file call
        "expect_max_tool_calls": 5,
    },
    {
        "name": "scaffold_tool_selection",
        "category": "scaffolding",
        "task": (
            "Find all Python files in the project that import 'logging'. "
            "Use the most efficient tool for this — do NOT use run_command with grep."
        ),
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "scaffold_budget_awareness",
        "category": "scaffolding",
        "task": (
            "This is a test of your budget management. Read exactly 3 files: "
            "pyproject.toml, blipshell/__init__.py, and CLAUDE.md. "
            "For each file, report the first line. Then immediately call task_complete. "
            "Do NOT explore further or read any other files."
        ),
        "expect_tools": ["read_file"],
        "expect_complete": True,
        "force_plan": True,
        "expect_max_tool_calls": 8,
    },

    # ── Category 5: Simple Chat Path (force_plan=False) ───────────────────
    {
        "name": "simple_chat_greeting",
        "category": "simple_chat",
        "task": "Hello, how are you today?",
        "expect_tools": [],
        "expect_complete": True,
        "force_plan": False,
    },
    {
        "name": "simple_chat_question",
        "category": "simple_chat",
        "task": "What is Python's GIL and why does it matter for multithreading?",
        "expect_tools": [],
        "expect_complete": True,
        "force_plan": False,
    },
    {
        "name": "simple_chat_with_tools",
        "category": "simple_chat",
        "task": "What files are in the project root directory?",
        "expect_tools": ["list_directory"],
        "expect_complete": True,
        "force_plan": False,
    },

    # ── Category 6: Complex Real-World Tasks ──────────────────────────────
    {
        "name": "real_add_function",
        "category": "real_world",
        "task": (
            "Create a new Python file called stress_test_utils.py with these functions:\n"
            "1. word_count(text: str) -> int — returns number of words\n"
            "2. char_count(text: str) -> int — returns number of non-whitespace characters\n"
            "3. line_count(text: str) -> int — returns number of lines\n\n"
            "Then create stress_test_utils_test.py that tests all 3 functions with "
            "at least 2 test cases each. Run the tests with: "
            "python -m pytest stress_test_utils_test.py -v"
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "real_codebase_analysis",
        "category": "real_world",
        "task": (
            "Analyze the blipshell/core/tools/ directory. For each .py file:\n"
            "1. List the file name\n"
            "2. List all tool classes defined in it (classes that inherit from Tool)\n"
            "3. For each tool class, note the tool name (from the definition() method)\n\n"
            "Create a file called stress_test_tool_inventory.json with a JSON array "
            "of objects, each with keys: file, classes (list of {class_name, tool_name})."
        ),
        "expect_tools": ["list_directory", "read_file", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "real_multi_file_refactor",
        "category": "real_world",
        "task": (
            "Create two files:\n"
            "1. stress_test_calculator.py — a Calculator class with methods: "
            "add(a, b), subtract(a, b), multiply(a, b), divide(a, b). "
            "divide should raise ValueError on division by zero.\n"
            "2. stress_test_calculator_test.py — pytest tests covering all 4 operations "
            "plus the division-by-zero error case.\n\n"
            "Run the tests and make sure they pass. If any test fails, fix it."
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ── Category 7: Edge Cases ────────────────────────────────────────────
    {
        "name": "edge_empty_file",
        "category": "edge_case",
        "task": (
            "Create an empty file called stress_test_empty.py (no content at all). "
            "Then read it back and confirm it's empty."
        ),
        "expect_tools": ["write_file", "read_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "edge_large_grep",
        "category": "edge_case",
        "task": (
            "Search the entire codebase for the pattern 'import' using grep_files "
            "with max_results set to 10. Report how many results were returned "
            "and whether it was truncated."
        ),
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "edge_special_characters",
        "category": "edge_case",
        "task": (
            "Create a file called stress_test_special.py with this content:\n"
            "# Special chars: quotes \" and ' and backslash \\ and newline \\n\n"
            "data = {'key': 'value with \"quotes\"', 'path': 'C:\\\\Users\\\\test'}\n"
            "print(data)\n\n"
            "Then verify it has valid syntax."
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "edge_long_task_budget",
        "category": "edge_case",
        "task": (
            "This task tests budget management. Read these files one by one:\n"
            "1. blipshell/core/agent.py (first 50 lines only)\n"
            "2. blipshell/core/executor.py (first 50 lines only)\n"
            "3. blipshell/core/config.py (first 50 lines only)\n"
            "4. blipshell/core/planner.py (first 50 lines only)\n"
            "5. blipshell/core/repo_map.py (first 50 lines only)\n"
            "6. blipshell/llm/router.py (first 50 lines only)\n"
            "7. blipshell/llm/client.py (first 50 lines only)\n"
            "8. blipshell/memory/manager.py (first 50 lines only)\n"
            "9. blipshell/memory/search.py (first 50 lines only)\n"
            "10. blipshell/memory/processor.py (first 50 lines only)\n\n"
            "After reading all 10, create stress_test_architecture.txt listing "
            "the module name and its one-line docstring for each."
        ),
        "expect_tools": ["read_file", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ── Category 8: Flow Observability ────────────────────────────────────
    {
        "name": "flow_events_logged",
        "category": "observability",
        "task": (
            "Read the file blipshell/core/tools/base.py and summarize "
            "what the ToolRegistry class does."
        ),
        "expect_tools": ["read_file"],
        "expect_complete": True,
        "force_plan": True,
        # We check flow_events are populated in the report
        "expect_flow_events": ["turn_start", "llm_complete"],
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
    force_plan: bool = True,
) -> dict:
    """Run a single headless test and return structured results.

    Args:
        task: The task description to send to the agent.
        project: Optional project name to activate.
        output_path: Optional file path to write JSON results.
        config_path: Optional config.yaml path.
        quiet: If True, suppress streaming output (JSON only).
        force_plan: If True, always use executor path. If False, let classifier decide.

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
            force_plan=force_plan,
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
# Test suite runner — shared by canned and stress modes
# ---------------------------------------------------------------------------

async def run_test_suite(
    tests: list[dict],
    suite_name: str,
    project: str | None = None,
    config_path: str | None = None,
    output_path: str | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Run a list of test definitions and return results."""
    results = []
    total = len(tests)
    suite_start = time.monotonic()

    for i, test in enumerate(tests, 1):
        category = test.get("category", "")
        cat_label = f" [{category}]" if category else ""
        console.print(f"\n[bold yellow]{'=' * 70}[/bold yellow]")
        console.print(f"[bold yellow]Test {i}/{total}{cat_label}: {test['name']}[/bold yellow]")
        console.print(f"[bold yellow]{'=' * 70}[/bold yellow]\n")

        fp = test.get("force_plan", True)

        report = await run_test(
            task=test["task"],
            project=project,
            config_path=config_path,
            quiet=quiet,
            force_plan=fp,
        )

        # Check expectations
        passed = True
        checks = []

        # Completion check
        if test.get("expect_complete"):
            ok = report["completed"]
            checks.append(("completed", ok))
            if not ok:
                passed = False

        # Expected tools used
        for expected_tool in test.get("expect_tools", []):
            found = any(tc["name"] == expected_tool for tc in report["tool_calls"])
            checks.append((f"used_{expected_tool}", found))
            if not found:
                passed = False

        # Max tool calls budget check
        max_tc = test.get("expect_max_tool_calls")
        if max_tc is not None:
            ok = report["tool_call_count"] <= max_tc
            checks.append((f"tool_calls<={max_tc} (got {report['tool_call_count']})", ok))
            if not ok:
                passed = False

        # Flow events check
        for expected_event in test.get("expect_flow_events", []):
            found = any(evt["type"] == expected_event for evt in report.get("flow_events", []))
            checks.append((f"flow_{expected_event}", found))
            if not found:
                passed = False

        # No fatal errors (tool errors are OK for error_recovery tests)
        if test.get("category") != "error_recovery":
            no_errors = len(report["errors"]) == 0
            checks.append(("no_errors", no_errors))
            if not no_errors:
                passed = False

        report["_test_name"] = test["name"]
        report["_category"] = test.get("category", "")
        report["_checks"] = checks
        report["_passed"] = passed
        results.append(report)

        status = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        console.print(f"\n{status} — {test['name']}")
        for check_name, check_ok in checks:
            icon = "[green]OK[/green]" if check_ok else "[red]FAIL[/red]"
            console.print(f"  {icon} {check_name}")

    # Final summary
    suite_elapsed = round(time.monotonic() - suite_start, 1)
    console.print(f"\n[bold]{'=' * 70}[/bold]")
    total_tests = len(results)
    passed_count = sum(1 for r in results if r["_passed"])
    console.print(f"[bold]{suite_name}: {passed_count}/{total_tests} passed in {suite_elapsed}s[/bold]\n")

    # Group by category
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("_category", "uncategorized")
        categories.setdefault(cat, []).append(r)

    for cat, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r["_passed"])
        cat_total = len(cat_results)
        cat_status = "[green]" if cat_passed == cat_total else "[red]"
        console.print(f"  {cat_status}{cat}: {cat_passed}/{cat_total}[/{cat_status.strip('[')}]")
        for r in cat_results:
            status = "[green]PASS[/green]" if r["_passed"] else "[red]FAIL[/red]"
            console.print(f"    {status} {r['_test_name']} ({r['timing']['total_seconds']}s, {r['tool_call_count']} tools)")

    # Save results if output path given
    if output_path:
        # Strip non-serializable check tuples for JSON
        serializable = []
        for r in results:
            r_copy = {k: v for k, v in r.items() if not k.startswith("_")}
            r_copy["test_name"] = r.get("_test_name", "")
            r_copy["category"] = r.get("_category", "")
            r_copy["passed"] = r.get("_passed", False)
            r_copy["checks"] = [{"name": c[0], "ok": c[1]} for c in r.get("_checks", [])]
            serializable.append(r_copy)

        report_data = {
            "suite": suite_name,
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": total_tests - passed_count,
            "elapsed_seconds": suite_elapsed,
            "results": serializable,
        }
        Path(output_path).write_text(
            json.dumps(report_data, indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"\n[bold green]Full results written to {output_path}[/bold green]")

    return results


async def run_canned_tests(
    project: str | None = None,
    config_path: str | None = None,
    output_path: str | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Run the quick canned test suite."""
    return await run_test_suite(
        CANNED_TESTS, "Canned Tests",
        project=project, config_path=config_path,
        output_path=output_path, quiet=quiet,
    )


async def run_stress_tests(
    project: str | None = None,
    config_path: str | None = None,
    output_path: str | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Run the full stress test suite."""
    return await run_test_suite(
        STRESS_TESTS, "Stress Tests",
        project=project, config_path=config_path,
        output_path=output_path or "data/stress_test_results.json",
        quiet=quiet,
    )


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
    parser.add_argument("--canned", action="store_true", help="Quick test suite (~5 min)")
    parser.add_argument("--stress", action="store_true", help="Full stress suite (~1-2 hours)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress streaming output")

    args = parser.parse_args()

    if args.stress:
        asyncio.run(run_stress_tests(
            project=args.project,
            config_path=args.config,
            output_path=args.output,
            quiet=args.quiet,
        ))
    elif args.canned:
        asyncio.run(run_canned_tests(
            project=args.project,
            config_path=args.config,
            output_path=args.output,
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
