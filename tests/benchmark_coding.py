"""Benchmark coding models on real plan-and-execute tasks.

Tests the full coding agent flow: plan generation → step execution → tool calling.
Compares models on speed, quality, tool discipline, and code correctness.

Metrics tracked per task per model:
  - Plan generation time + step count
  - Per-step: execution time, tool calls, edit successes/failures
  - Total time (plan + all steps + summary)
  - Bad behaviors: .md files created, file re-reads, excessive tool calls
  - Code correctness (pytest pass/fail on sandbox)

Usage:
    python tests/benchmark_coding.py                                              # all default models
    python tests/benchmark_coding.py qwen3-coder:480b-cloud gpt-oss:120b-cloud   # specific models
    python tests/benchmark_coding.py --timeout 600                                # longer timeout
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from blipshell.core.tools.base import ToolRegistry
from blipshell.core.tools.code_tools import GlobTool, GrepTool
from blipshell.core.tools.filesystem import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from blipshell.core.tools.shell import ShellTool
from blipshell.core.executor import TaskExecutor
from blipshell.core.planner import TaskPlanner
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig, PlannerConfig

# ---------------------------------------------------------------------------
# Models to benchmark (all routed through local Ollama which proxies to cloud)
# ---------------------------------------------------------------------------
BENCHMARK_MODELS = [
    # Current coding model (baseline)
    "qwen3-coder:480b-cloud",
    # Ollama cloud models to evaluate
    "qwen3-coder-next:cloud",
    "devstral-2:123b-cloud",
    "deepseek-v3.2:cloud",
    "kimi-k2.5:cloud",
    "cogito-2.1:671b-cloud",
    "qwen3-next:80b-cloud",
    "minimax-m2.5:cloud",
    "glm-5:cloud",
]

OLLAMA_URL = "http://localhost:11434"

console = Console()

# ---------------------------------------------------------------------------
# Sandbox file contents — pre-built Python files for tasks to operate on
# ---------------------------------------------------------------------------

CALCULATOR_PY = '''\
"""Simple calculator module."""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a // b  # BUG: should be a / b (true division, not floor division)
'''

TEST_CALCULATOR_PY = '''\
"""Existing tests for calculator — only covers add/subtract."""

from calculator import add, subtract


def test_add():
    assert add(2, 3) == 5


def test_add_negative():
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(10, 3) == 7
'''

CONFIG_MODULE_PY = '''\
"""Simple key-value configuration store."""


class Config:
    """In-memory configuration with dot-notation access."""

    def __init__(self, data: dict | None = None):
        self._data: dict = data or {}

    def get(self, key: str) -> str | None:
        """Get a config value by key, returns None if missing."""
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        """Set a config value."""
        self._data[key] = value

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._data

    def all_keys(self) -> list[str]:
        """Return all configuration keys."""
        return list(self._data.keys())
'''

DATA_PROCESSOR_PY = '''\
"""Data processing pipeline with duplicated validation logic."""

import re
from typing import Any


def process_users(users: list[dict[str, Any]]) -> list[dict]:
    """Process and validate user records."""
    results = []
    for user in users:
        # Validate name
        name = user.get("name", "")
        if not name or not isinstance(name, str):
            continue
        name = name.strip()
        if len(name) < 2 or len(name) > 100:
            continue
        if not re.match(r"^[a-zA-Z\\s\\-\\']+$", name):
            continue

        # Validate email
        email = user.get("email", "")
        if not email or not isinstance(email, str):
            continue
        email = email.strip().lower()
        if not re.match(r"^[^@]+@[^@]+\\.[^@]+$", email):
            continue

        results.append({"name": name, "email": email})
    return results


def process_products(products: list[dict[str, Any]]) -> list[dict]:
    """Process and validate product records."""
    results = []
    for product in products:
        # Validate name (same logic as users!)
        name = product.get("name", "")
        if not name or not isinstance(name, str):
            continue
        name = name.strip()
        if len(name) < 2 or len(name) > 100:
            continue
        if not re.match(r"^[a-zA-Z\\s\\-\\']+$", name):
            continue

        # Validate price
        price = product.get("price", 0)
        if not isinstance(price, (int, float)) or price < 0:
            continue

        results.append({"name": name, "price": round(price, 2)})
    return results
'''

# ---------------------------------------------------------------------------
# Coding tasks
# ---------------------------------------------------------------------------

CODING_TASKS = [
    {
        "name": "fix_bug_add_test",
        "description": "Fix a division bug and add tests",
        "request": (
            "Fix the divide function in calculator.py — it uses floor division (//) "
            "instead of true division (/). Also add tests for multiply and divide "
            "(including a test that divide raises ValueError for zero) to test_calculator.py."
        ),
        "files": {
            "calculator.py": CALCULATOR_PY,
            "test_calculator.py": TEST_CALCULATOR_PY,
        },
        "verify_command": "python -m pytest test_calculator.py -v",
        "verify_checks": [
            # (check_type, target) — what to verify after task completes
            ("file_contains", ("calculator.py", "a / b")),            # bug fixed
            ("file_not_contains", ("calculator.py", "a // b")),       # old bug gone
            ("file_contains", ("test_calculator.py", "test_divide")), # test added
            ("file_contains", ("test_calculator.py", "test_multiply")), # test added
            ("pytest_passes", "test_calculator.py"),
        ],
    },
    {
        "name": "add_feature",
        "description": "Add a method to an existing class",
        "request": (
            "Add a `get_or_default(key, default=None)` method to the Config class "
            "in config_module.py that returns the value for key if it exists, otherwise "
            "returns the default value. Also add a `delete(key)` method that removes "
            "a key if it exists (no error if missing). Write tests for both new methods "
            "in a new test_config.py file."
        ),
        "files": {
            "config_module.py": CONFIG_MODULE_PY,
        },
        "verify_command": "python -m pytest test_config.py -v",
        "verify_checks": [
            ("file_contains", ("config_module.py", "get_or_default")),
            ("file_contains", ("config_module.py", "def delete")),
            ("file_exists", "test_config.py"),
            ("pytest_passes", "test_config.py"),
        ],
    },
    {
        "name": "refactor",
        "description": "Extract duplicated validation logic",
        "request": (
            "Refactor data_processor.py — the name validation logic is duplicated "
            "between process_users() and process_products(). Extract it into a "
            "_validate_name(name) helper that returns the cleaned name or None if "
            "invalid. Update both functions to use the helper. Write tests in "
            "test_data_processor.py to verify the refactoring didn't break anything."
        ),
        "files": {
            "data_processor.py": DATA_PROCESSOR_PY,
        },
        "verify_command": "python -m pytest test_data_processor.py -v",
        "verify_checks": [
            ("file_contains", ("data_processor.py", "_validate_name")),
            ("file_exists", "test_data_processor.py"),
            ("pytest_passes", "test_data_processor.py"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """Record of a single tool call during step execution."""
    name: str
    arguments: dict
    success: bool
    result_preview: str  # first 200 chars
    time_ms: float = 0.0


@dataclass
class StepMetrics:
    """Metrics for a single step execution."""
    step_number: int
    description: str
    execution_time: float  # seconds
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    output_preview: str = ""

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def edit_attempts(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.name == "edit_file")

    @property
    def edit_failures(self) -> int:
        return sum(
            1 for tc in self.tool_calls
            if tc.name == "edit_file" and not tc.success
        )

    @property
    def file_reads(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.name in ("read_file", "list_directory"))

    @property
    def unique_files_read(self) -> set[str]:
        return {
            tc.arguments.get("path", "")
            for tc in self.tool_calls
            if tc.name == "read_file"
        }


@dataclass
class TaskMetrics:
    """Metrics for a complete coding task (plan + all steps)."""
    task_name: str
    model: str
    plan_time: float = 0.0           # seconds for plan generation
    plan_steps: int = 0              # number of steps in plan
    plan_text: str = ""              # raw plan output
    steps: list[StepMetrics] = field(default_factory=list)
    summary_time: float = 0.0       # seconds for final summary
    summary_text: str = ""
    total_time: float = 0.0         # end-to-end wall time
    error: str = ""

    # Verification results
    checks_passed: int = 0
    checks_total: int = 0
    pytest_passed: bool = False
    pytest_output: str = ""

    @property
    def total_tool_calls(self) -> int:
        return sum(s.tool_call_count for s in self.steps)

    @property
    def total_edit_failures(self) -> int:
        return sum(s.edit_failures for s in self.steps)

    @property
    def total_file_reads(self) -> int:
        return sum(s.file_reads for s in self.steps)

    @property
    def files_created(self) -> set[str]:
        """Files written by write_file tool."""
        return {
            tc.arguments.get("path", "")
            for s in self.steps
            for tc in s.tool_calls
            if tc.name == "write_file"
        }

    @property
    def unwanted_files(self) -> set[str]:
        """Files created that shouldn't be (docs, READMEs, etc.)."""
        return {
            f for f in self.files_created
            if f.lower().endswith((".md", ".rst", ".txt"))
            and "test" not in f.lower()
        }

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "task_name": self.task_name,
            "model": self.model,
            "plan_time": round(self.plan_time, 2),
            "plan_steps": self.plan_steps,
            "plan_text": self.plan_text,
            "total_time": round(self.total_time, 2),
            "summary_time": round(self.summary_time, 2),
            "total_tool_calls": self.total_tool_calls,
            "total_edit_failures": self.total_edit_failures,
            "total_file_reads": self.total_file_reads,
            "unwanted_files": list(self.unwanted_files),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "pytest_passed": self.pytest_passed,
            "pytest_output": self.pytest_output[:1000],
            "error": self.error,
            "steps": [
                {
                    "step_number": s.step_number,
                    "description": s.description,
                    "execution_time": round(s.execution_time, 2),
                    "tool_call_count": s.tool_call_count,
                    "edit_attempts": s.edit_attempts,
                    "edit_failures": s.edit_failures,
                    "file_reads": s.file_reads,
                    "tool_calls": [
                        {"name": tc.name, "success": tc.success}
                        for tc in s.tool_calls
                    ],
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Instrumented tool registry — wraps ToolRegistry to capture metrics
# ---------------------------------------------------------------------------

class InstrumentedToolRegistry(ToolRegistry):
    """ToolRegistry wrapper that captures tool call metrics."""

    def __init__(self):
        super().__init__()
        self.call_log: list[ToolCallRecord] = []

    def reset_log(self):
        self.call_log = []

    async def execute_tool_call(self, tool_call):
        """Execute and log the tool call."""
        start = time.perf_counter()
        result = await super().execute_tool_call(tool_call)
        elapsed_ms = (time.perf_counter() - start) * 1000

        record = ToolCallRecord(
            name=tool_call.name,
            arguments=dict(tool_call.arguments),
            success=result.success,
            result_preview=result.result[:200] if result.result else "",
            time_ms=round(elapsed_ms, 1),
        )
        self.call_log.append(record)
        return result


# ---------------------------------------------------------------------------
# Sandbox setup
# ---------------------------------------------------------------------------

def create_sandbox(task: dict) -> str:
    """Create a temp directory with the task's files. Returns the path."""
    sandbox = tempfile.mkdtemp(prefix="blip_bench_")
    for filename, content in task["files"].items():
        filepath = os.path.join(sandbox, filename)
        Path(filepath).write_text(content, encoding="utf-8")
    return sandbox


def create_tool_registry(sandbox_path: str) -> InstrumentedToolRegistry:
    """Create a tool registry pointing at the sandbox directory."""
    registry = InstrumentedToolRegistry()

    registry.register(ReadFileTool(
        max_file_size=1048576,
        root_path=sandbox_path,
    ), group="filesystem")
    registry.register(WriteFileTool(
        root_path=sandbox_path,
    ), group="filesystem")
    registry.register(EditFileTool(root_path=sandbox_path), group="filesystem")
    registry.register(ListDirectoryTool(root_path=sandbox_path), group="filesystem")
    registry.register(ShellTool(
        timeout=30,
        allowed_commands=["python", "pip", "pytest", "type", "dir", "echo"],
        cwd=sandbox_path,
    ), group="shell")
    registry.register(GrepTool(root_path=sandbox_path), group="coding")
    registry.register(GlobTool(root_path=sandbox_path), group="coding")

    return registry


# ---------------------------------------------------------------------------
# Router / planner / executor setup
# ---------------------------------------------------------------------------

def make_router(model_name: str, timeout: float = 300.0) -> LLMRouter:
    """Create an LLMRouter that routes CODING and TOOL_CALLING to the given model."""
    models = ModelsConfig(
        reasoning=model_name,
        tool_calling=model_name,
        coding=model_name,
        summarization=model_name,
        ranking=model_name,
        importance=model_name,
        embedding=model_name,
    )
    endpoint_cfg = EndpointConfig(
        name="benchmark",
        url=OLLAMA_URL,
        roles=["reasoning", "tool_calling", "coding", "summarization",
               "ranking", "importance", "embedding"],
        priority=1,
        max_concurrent=1,
        context_tokens=131072,
    )
    llm_config = LLMConfig(timeout=timeout)
    endpoint_manager = EndpointManager([endpoint_cfg], llm_config)
    return LLMRouter(models, endpoint_manager)


async def create_sqlite(sandbox_path: str) -> SQLiteStore:
    """Create a temporary SQLiteStore for plan/step persistence."""
    db_path = os.path.join(sandbox_path, "_benchmark.db")
    store = SQLiteStore(db_path)
    await store.initialize()
    return store


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

async def run_verification(sandbox_path: str, task: dict) -> tuple[int, int, bool, str]:
    """Run verification checks on the sandbox after task completion.

    Returns (checks_passed, checks_total, pytest_passed, pytest_output).
    """
    checks = task.get("verify_checks", [])
    passed = 0
    total = len(checks)
    pytest_passed = False
    pytest_output = ""

    for check_type, target in checks:
        if check_type == "file_contains":
            filepath, needle = target
            full_path = os.path.join(sandbox_path, filepath)
            if os.path.isfile(full_path):
                content = Path(full_path).read_text(encoding="utf-8", errors="replace")
                if needle in content:
                    passed += 1

        elif check_type == "file_not_contains":
            filepath, needle = target
            full_path = os.path.join(sandbox_path, filepath)
            if os.path.isfile(full_path):
                content = Path(full_path).read_text(encoding="utf-8", errors="replace")
                if needle not in content:
                    passed += 1

        elif check_type == "file_exists":
            full_path = os.path.join(sandbox_path, target)
            if os.path.isfile(full_path):
                passed += 1

        elif check_type == "pytest_passes":
            test_file = os.path.join(sandbox_path, target)
            if os.path.isfile(test_file):
                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "pytest", test_file, "-v",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=sandbox_path,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=30,
                    )
                    pytest_output = stdout.decode("utf-8", errors="replace")
                    if stderr:
                        pytest_output += "\n" + stderr.decode("utf-8", errors="replace")
                    if proc.returncode == 0:
                        passed += 1
                        pytest_passed = True
                except asyncio.TimeoutError:
                    pytest_output = "pytest timed out after 30s"
                except Exception as e:
                    pytest_output = f"pytest error: {e}"

    return passed, total, pytest_passed, pytest_output


# ---------------------------------------------------------------------------
# Run a single task for a single model
# ---------------------------------------------------------------------------

async def run_task(model_spec: str, task: dict, timeout: float = 300.0) -> TaskMetrics:
    """Run a coding task with a given model and return metrics."""
    metrics = TaskMetrics(task_name=task["name"], model=model_spec)
    sandbox_path = create_sandbox(task)

    try:
        router = make_router(model_spec, timeout=timeout)
        sqlite = await create_sqlite(sandbox_path)
        tool_registry = create_tool_registry(sandbox_path)

        planner_config = PlannerConfig(
            enabled=True,
            auto_approve=True,
            max_steps=7,
            max_retries_per_step=1,  # 1 retry max — don't waste time
        )

        planner = TaskPlanner(router, sqlite, planner_config)
        # Simulate project mode so it routes to CODING task type
        planner.active_project = {"name": "benchmark", "root_path": sandbox_path}

        executor = TaskExecutor(
            router=router,
            sqlite=sqlite,
            tool_registry=tool_registry,
            config=planner_config,
            system_prompt=(
                "You are a coding assistant. Execute the task step by step using the tools available. "
                "Be concise and efficient."
            ),
            max_tool_iterations=15,  # reasonable cap per step
        )
        executor.active_project = {"name": "benchmark", "root_path": sandbox_path}

        wall_start = time.perf_counter()

        # Phase 1: Plan generation
        plan_start = time.perf_counter()
        try:
            plan = await planner.create_plan(task["request"])
            metrics.plan_time = time.perf_counter() - plan_start
            metrics.plan_steps = len(plan.steps)
            metrics.plan_text = "\n".join(
                f"  {s.step_number}. {s.description}" for s in plan.steps
            )
        except Exception as e:
            metrics.plan_time = time.perf_counter() - plan_start
            metrics.error = f"Plan generation failed: {e}"
            metrics.total_time = time.perf_counter() - wall_start
            return metrics

        # Phase 2: Step execution (with metrics capture per step)
        for step in plan.steps:
            tool_registry.reset_log()
            step_start = time.perf_counter()

            try:
                result = await executor._execute_step(
                    plan=plan,
                    step_number=step.step_number,
                    step_description=step.description,
                    total_steps=len(plan.steps),
                    completed_summaries=[
                        f"{s.description}: done" for s in metrics.steps
                    ],
                )
            except Exception as e:
                result = f"ERROR: {e}"

            step_time = time.perf_counter() - step_start

            step_metrics = StepMetrics(
                step_number=step.step_number,
                description=step.description,
                execution_time=step_time,
                tool_calls=list(tool_registry.call_log),
                output_preview=result[:500] if result else "",
            )
            metrics.steps.append(step_metrics)

        # Phase 3: Summary generation
        summary_start = time.perf_counter()
        try:
            step_results = [s.output_preview for s in metrics.steps]
            summary = await executor._generate_summary(task["request"], step_results)
            metrics.summary_text = summary[:1000] if summary else ""
        except Exception:
            metrics.summary_text = ""
        metrics.summary_time = time.perf_counter() - summary_start

        metrics.total_time = time.perf_counter() - wall_start

        # Phase 4: Verification
        (
            metrics.checks_passed,
            metrics.checks_total,
            metrics.pytest_passed,
            metrics.pytest_output,
        ) = await run_verification(sandbox_path, task)

        # Close DB
        try:
            await sqlite.close()
        except Exception:
            pass

    except Exception as e:
        metrics.error = str(e)
    finally:
        # Cleanup sandbox
        try:
            shutil.rmtree(sandbox_path, ignore_errors=True)
        except Exception:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Display — Rich tables
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict[str, list[TaskMetrics]]):
    """Print a summary comparison table across all models and tasks."""
    table = Table(
        title="Coding Model Benchmark — Summary",
        show_lines=True,
        expand=True,
    )
    table.add_column("Model", style="cyan", width=26, no_wrap=True)
    table.add_column("Task", width=18)
    table.add_column("Plan", width=8, justify="center")
    table.add_column("Steps", width=6, justify="center")
    table.add_column("Tools", width=6, justify="center")
    table.add_column("Edit Fail", width=9, justify="center")
    table.add_column("Reads", width=6, justify="center")
    table.add_column("Plan(s)", width=8, justify="right")
    table.add_column("Exec(s)", width=8, justify="right")
    table.add_column("Total(s)", width=8, justify="right")
    table.add_column("Checks", width=8, justify="center")
    table.add_column("Pytest", width=7, justify="center")

    for model, task_results in all_results.items():
        for i, m in enumerate(task_results):
            model_col = model if i == 0 else ""
            exec_time = sum(s.execution_time for s in m.steps)
            checks_str = f"{m.checks_passed}/{m.checks_total}"
            pytest_str = (
                "[green]PASS[/green]" if m.pytest_passed
                else "[red]FAIL[/red]" if m.checks_total > 0
                else "[dim]N/A[/dim]"
            )
            edit_fail_str = (
                f"[red]{m.total_edit_failures}[/red]" if m.total_edit_failures > 0
                else "[green]0[/green]"
            )
            unwanted_str = ""
            if m.unwanted_files:
                unwanted_str = f" [red]+{len(m.unwanted_files)}md[/red]"

            table.add_row(
                model_col,
                m.task_name,
                f"{m.plan_steps}",
                f"{len(m.steps)}",
                f"{m.total_tool_calls}{unwanted_str}",
                edit_fail_str,
                f"{m.total_file_reads}",
                f"{m.plan_time:.1f}",
                f"{exec_time:.1f}",
                f"{m.total_time:.1f}",
                checks_str,
                pytest_str,
            )

    console.print(table)


def print_plan_table(all_results: dict[str, list[TaskMetrics]]):
    """Print the generated plans for comparison."""
    for task_name in [t["name"] for t in CODING_TASKS]:
        table = Table(
            title=f"Plans — {task_name}",
            show_lines=True,
            expand=True,
        )
        table.add_column("Model", style="cyan", width=26, no_wrap=True)
        table.add_column("Plan", ratio=1)
        table.add_column("Time", width=8, justify="right")

        for model, task_results in all_results.items():
            for m in task_results:
                if m.task_name == task_name:
                    plan_text = escape(m.plan_text[:600]) if m.plan_text else "[dim]error[/dim]"
                    table.add_row(model, plan_text, f"{m.plan_time:.1f}s")

        console.print(table)
        console.print()


def print_step_detail_table(all_results: dict[str, list[TaskMetrics]]):
    """Print per-step tool call details for each model/task combo."""
    for model, task_results in all_results.items():
        for m in task_results:
            table = Table(
                title=f"Step Details — {model} / {m.task_name}",
                show_lines=True,
            )
            table.add_column("Step", width=6, justify="center")
            table.add_column("Description", width=40)
            table.add_column("Tools", width=6, justify="center")
            table.add_column("Edits", width=10, justify="center")
            table.add_column("Time(s)", width=8, justify="right")
            table.add_column("Tool Calls", ratio=1)

            for s in m.steps:
                tool_names = [
                    f"{'[green]' if tc.success else '[red]'}{tc.name}{'[/green]' if tc.success else '[/red]'}"
                    for tc in s.tool_calls
                ]
                tools_str = ", ".join(tool_names) if tool_names else "[dim]none[/dim]"
                edit_str = f"{s.edit_attempts - s.edit_failures}/{s.edit_attempts}"

                table.add_row(
                    f"{s.step_number}",
                    escape(s.description[:40]),
                    f"{s.tool_call_count}",
                    edit_str,
                    f"{s.execution_time:.1f}",
                    tools_str,
                )

            console.print(table)
            console.print()


def print_totals_table(all_results: dict[str, list[TaskMetrics]]):
    """Print aggregated totals per model."""
    table = Table(
        title="Model Totals (across all tasks)",
        show_lines=True,
    )
    table.add_column("Model", style="cyan", width=26, no_wrap=True)
    table.add_column("Tasks", width=6, justify="center")
    table.add_column("Total Time", width=10, justify="right")
    table.add_column("Avg Time", width=10, justify="right")
    table.add_column("Total Tools", width=10, justify="center")
    table.add_column("Edit Failures", width=12, justify="center")
    table.add_column("Pytest Pass", width=10, justify="center")
    table.add_column("Checks", width=10, justify="center")

    for model, task_results in all_results.items():
        total_time = sum(m.total_time for m in task_results)
        avg_time = total_time / len(task_results) if task_results else 0
        total_tools = sum(m.total_tool_calls for m in task_results)
        total_edit_fail = sum(m.total_edit_failures for m in task_results)
        pytest_pass = sum(1 for m in task_results if m.pytest_passed)
        checks_pass = sum(m.checks_passed for m in task_results)
        checks_total = sum(m.checks_total for m in task_results)

        edit_str = (
            f"[red]{total_edit_fail}[/red]" if total_edit_fail > 0
            else "[green]0[/green]"
        )

        table.add_row(
            model,
            f"{len(task_results)}",
            f"{total_time:.1f}s",
            f"{avg_time:.1f}s",
            f"{total_tools}",
            edit_str,
            f"{pytest_pass}/{len(task_results)}",
            f"{checks_pass}/{checks_total}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(models: list[str], timeout: float = 300.0):
    """Run the full coding benchmark for all models."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "benchmark_coding_results.json"

    # Load existing results to merge
    all_raw: dict[str, list[dict]] = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                all_raw = json.load(f)
            existing = [m for m in models if m in all_raw]
            if existing:
                console.print(
                    f"[yellow]Loaded existing results for: {', '.join(existing)}[/yellow]"
                )
        except json.JSONDecodeError:
            pass

    console.print(f"\n[bold]Coding Model Benchmark[/bold]")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Tasks: {', '.join(t['name'] for t in CODING_TASKS)}")
    console.print(f"Timeout: {timeout}s per LLM call\n")

    all_results: dict[str, list[TaskMetrics]] = {}

    for model_spec in models:
        console.rule(f"[bold blue]Benchmarking: {model_spec}")
        model_results: list[TaskMetrics] = []

        for task in CODING_TASKS:
            console.print(f"\n  [dim]Task: {task['name']} — {task['description']}[/dim]")
            console.print(f"  [dim]Request: {task['request'][:80]}...[/dim]")

            metrics = await run_task(model_spec, task, timeout=timeout)

            if metrics.error:
                console.print(f"  [red]ERROR: {metrics.error}[/red]")
            else:
                pytest_str = "[green]PASS[/green]" if metrics.pytest_passed else "[red]FAIL[/red]"
                console.print(
                    f"  [green]Done[/green] — "
                    f"{metrics.plan_steps} steps, "
                    f"{metrics.total_tool_calls} tool calls, "
                    f"{metrics.total_edit_failures} edit failures, "
                    f"checks {metrics.checks_passed}/{metrics.checks_total}, "
                    f"pytest {pytest_str}, "
                    f"time {metrics.total_time:.1f}s"
                )

            model_results.append(metrics)

        all_results[model_spec] = model_results
        all_raw[model_spec] = [m.to_dict() for m in model_results]

        console.print(f"\n  [green]Completed all tasks for {model_spec}[/green]")

    # Save results
    with open(output_path, "w") as f:
        json.dump(all_raw, f, indent=2)
    console.print(f"\n[bold]Results saved to {output_path}[/bold]")

    # Print comparison tables
    console.print()
    console.rule("[bold green]Results")
    console.print()
    print_summary_table(all_results)
    console.print()
    print_totals_table(all_results)
    console.print()
    print_plan_table(all_results)
    print_step_detail_table(all_results)


def main():
    """CLI entry point."""
    models = []
    timeout = 300.0

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--timeout" and i + 1 < len(args):
            timeout = float(args[i + 1])
            i += 2
        else:
            models.append(args[i])
            i += 1

    if not models:
        models = BENCHMARK_MODELS

    asyncio.run(run_benchmark(models, timeout=timeout))


if __name__ == "__main__":
    main()
