"""Standalone model benchmark — no Agent, no memory, no SQLite, no ChromaDB.

Directly uses LLMClient/OpenAICompatClient → ChatLoop → ToolRegistry to
benchmark models on executor-style tasks. Designed for A/B comparison.

Supports both Ollama (local) and OpenAI-compatible APIs (OpenRouter, Groq, etc.).

Usage (Ollama):
    python scripts/benchmark_standalone.py model_a model_b
    python scripts/benchmark_standalone.py model_a model_b --host http://192.168.1.100:11434

Usage (OpenRouter):
    python scripts/benchmark_standalone.py openai/gpt-4o anthropic/claude-sonnet-4 \\
        --provider openai --api-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY

Usage (Groq):
    python scripts/benchmark_standalone.py llama-3.3-70b-versatile mixtral-8x7b-32768 \\
        --provider openai --api-url https://api.groq.com/openai/v1 --api-key $GROQ_API_KEY

Usage (mixed — model A on Ollama, model B on OpenRouter):
    python scripts/benchmark_standalone.py qwen3:14b openai/gpt-4o \\
        --provider-b openai --api-url-b https://openrouter.ai/api/v1 --api-key-b $OPENROUTER_API_KEY

Options:
    --suite canned|stress|all     Test suite (default: stress)
    --budget 30                   Tool call budget per test
    --timeout 180                 LLM timeout in seconds
    --context 131072              Context window size
    --resume results.json         Skip already-completed tests
    --quiet                       JSON output only
"""

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from blipshell.llm.client import LLMClient
from blipshell.core.chat_loop import ChatLoop, LoopConfig, LoopResult
from blipshell.core.tools.base import ToolRegistry
from blipshell.core.tools.filesystem import (
    ReadFileTool, WriteFileTool, EditFileTool, ListDirectoryTool,
)
from blipshell.core.tools.code_tools import GrepTool, GlobTool
from blipshell.core.tools.shell import ShellTool
from blipshell.core.tools.interaction_tools import TaskCompleteTool, AskUserTool
from blipshell.models.tools import ToolResult

console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Client factory — creates LLMClient (Ollama) or OpenAICompatClient
# ---------------------------------------------------------------------------

def create_client(
    provider: str,
    host: str = "http://localhost:11434",
    api_url: str = "",
    api_key: str = "",
    timeout: float = 180.0,
):
    """Create an LLM client based on provider type.

    Args:
        provider: "ollama" or "openai"
        host: Ollama host URL (only for provider="ollama")
        api_url: Base URL for OpenAI-compatible API (e.g. https://openrouter.ai/api/v1)
        api_key: API key (only for provider="openai")
        timeout: Request timeout in seconds

    Returns:
        LLMClient or OpenAICompatClient (duck-typed, same interface)
    """
    if provider == "openai":
        if not api_url:
            raise ValueError("--api-url is required when --provider is openai")
        # Lazy import — openai package only needed when actually using OpenAI provider
        from blipshell.llm.openai_client import OpenAICompatClient
        return OpenAICompatClient(
            base_url=api_url,
            api_key=api_key,
            timeout=timeout,
        )
    else:
        return LLMClient(
            host=host,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# System prompt — copied from executor_system_prompt() to avoid imports
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Executor-style system prompt. Self-contained, no BlipShell imports."""
    os_name = platform.system()
    os_note = ""
    if os_name == "Windows":
        os_note = (
            "\n# Platform\n"
            "This is Windows. Do NOT use Unix commands (ls, cat, grep, head, tail, wc, find) "
            "in run_command. Use the dedicated tools: list_directory, read_file, grep_files, glob_files.\n"
        )

    return (
        "You are a coding agent. You complete tasks autonomously using tools.\n"
        + os_note +
        "\n# Rules\n"
        "1. PLAN first — state your approach in 1-3 sentences before writing code.\n"
        "2. Read before editing. NEVER re-read a file you already read.\n"
        "3. Make MINIMAL changes — no refactoring or extras beyond the task.\n"
        "4. If something fails twice, use ask_user instead of retrying blindly.\n"
        "5. Call task_complete when DONE. Do NOT just stop responding.\n"
        "6. Each tool's description explains when/how to use it — follow that guidance.\n"
        "7. Do NOT narrate your thinking — just call tools or answer directly.\n"
    )


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

CANNED_TESTS = [
    {
        "name": "file_creation",
        "task": (
            "Create a Python file called test_hello.py in the current directory "
            "that prints 'Hello from benchmark test'. "
            "Then verify it has correct syntax by running: python -c \"import ast; ast.parse(open('test_hello.py').read()); print('OK')\""
        ),
        "expect_tools": ["write_file"],
    },
    {
        "name": "read_and_grep",
        "task": (
            "Create a file called sample_code.py with 3 Python functions: add, subtract, multiply. "
            "Then use grep_files to find which files in the current directory contain 'def add'. "
            "Report what you find."
        ),
        "expect_tools": ["write_file"],
    },
    {
        "name": "read_edit_flow",
        "task": (
            "Create a file called scratch.py with the content:\n"
            "def hello():\n"
            "    return 'hello'\n\n"
            "Then read it back, and use edit_file to add a second function "
            "farewell() that returns 'goodbye'. Verify the file has valid syntax."
        ),
        "expect_tools": ["write_file", "read_file", "edit_file"],
    },
]

STRESS_TESTS = [
    # ── Tool Coverage ──
    {
        "name": "tool_write_file",
        "category": "tool_coverage",
        "task": (
            "Create a Python file called stress_output.py that contains a "
            "function called greet(name) which returns f'Hello, {name}!'. "
            "Include a __main__ block that calls greet('World') and prints the result."
        ),
        "expect_tools": ["write_file"],
    },
    {
        "name": "tool_list_directory",
        "category": "tool_coverage",
        "task": (
            "List the contents of the current directory and tell me "
            "how many files are in it."
        ),
        "expect_tools": ["list_directory"],
    },
    {
        "name": "tool_run_command",
        "category": "tool_coverage",
        "task": (
            "Run 'python --version' using the shell command tool and report "
            "what Python version is installed."
        ),
        "expect_tools": ["run_command"],
    },
    {
        "name": "tool_grep_files",
        "category": "tool_coverage",
        "task": (
            "Create a file called search_target.py with 'class MyWidget:' and "
            "'class MyButton:' in it. Then use grep_files to search the current "
            "directory for all files containing 'class My'."
        ),
        "expect_tools": ["write_file"],
    },
    {
        "name": "tool_glob_files",
        "category": "tool_coverage",
        "task": (
            "Create three files: a.py, b.py, c.txt. Then use glob_files to find "
            "all *.py files in the current directory and report them."
        ),
        "expect_tools": ["write_file"],
    },

    # ── Multi-step ──
    {
        "name": "multi_read_then_create",
        "category": "multi_step",
        "task": (
            "Create a file called module_a.py with a class called ServiceA that has "
            "a run() method. Then read it back. Then create module_b.py that imports "
            "ServiceA and has a class ServiceB that wraps ServiceA."
        ),
        "expect_tools": ["write_file", "read_file"],
    },
    {
        "name": "multi_grep_read_edit",
        "category": "multi_step",
        "task": (
            "Create a file called edit_target.py with:\n"
            "class MyService:\n"
            "    def run(self):\n"
            "        pass\n\n"
            "Then use grep_files to find which file contains 'class MyService'. "
            "Then read that file. Then use edit_file to add a one-line comment "
            "'# Found by benchmark' right above the class definition."
        ),
        "expect_tools": ["write_file", "read_file", "edit_file"],
    },
    {
        "name": "multi_five_step_chain",
        "category": "multi_step",
        "task": (
            "Complete these steps in order:\n"
            "1. Create step1.txt with 'step 1 done'\n"
            "2. Create step2.txt with 'step 2 done'\n"
            "3. List the current directory\n"
            "4. Read step1.txt back\n"
            "5. Create results.txt listing the files you found in step 3"
        ),
        "expect_tools": ["write_file", "list_directory", "read_file"],
    },

    # ── Error Recovery ──
    {
        "name": "error_nonexistent_file",
        "category": "error_recovery",
        "task": (
            "Try to read the file 'does_not_exist_12345.py'. "
            "When it fails, create the file instead with the content "
            "'# Created because the original was missing'."
        ),
        "expect_tools": ["read_file", "write_file"],
    },
    {
        "name": "error_syntax_fix",
        "category": "error_recovery",
        "task": (
            "Create a Python file called broken.py with this exact content:\n"
            "def add(a, b)\n"
            "    return a + b\n\n"
            "Then run 'python -c \"import ast; ast.parse(open(\\\"broken.py\\\").read())\"' "
            "to check syntax. It will fail because the colon is missing. "
            "Fix the syntax error using edit_file and verify it passes."
        ),
        "expect_tools": ["write_file", "run_command", "edit_file"],
    },
    {
        "name": "error_edit_wrong_text",
        "category": "error_recovery",
        "task": (
            "Create a file called target.py with:\n"
            "x = 42\n\n"
            "Read it. Then try to edit it by replacing "
            "the text 'THIS_DOES_NOT_EXIST' with 'replaced'. "
            "When the edit fails, report what happened and call task_complete."
        ),
        "expect_tools": ["write_file", "read_file", "edit_file"],
    },

    # ── Real-world Coding ──
    {
        "name": "real_calculator",
        "category": "real_world",
        "task": (
            "Create two files:\n"
            "1. calculator.py — a Calculator class with methods: "
            "add(a, b), subtract(a, b), multiply(a, b), divide(a, b). "
            "divide should raise ValueError on division by zero.\n"
            "2. test_calculator.py — pytest tests covering all 4 operations "
            "plus the division-by-zero error case.\n\n"
            "Run the tests and make sure they pass. If any test fails, fix it."
        ),
        "expect_tools": ["write_file", "run_command"],
    },
    {
        "name": "real_dataclass_module",
        "category": "real_world",
        "task": (
            "Create a file models.py with:\n"
            "1. A dataclass 'User' with fields: name (str), email (str), age (int)\n"
            "2. A dataclass 'Team' with fields: name (str), members (list[User])\n"
            "3. A function create_team(name, *users) -> Team\n"
            "4. A function team_average_age(team) -> float\n\n"
            "Then create test_models.py that tests:\n"
            "- Creating a User and Team\n"
            "- create_team with multiple users\n"
            "- team_average_age with known values\n"
            "- team_average_age with empty team (should return 0.0)\n\n"
            "Run the tests."
        ),
        "expect_tools": ["write_file", "run_command"],
    },
    {
        "name": "real_transform_functions",
        "category": "real_world",
        "task": (
            "Create transforms.py with these functions:\n"
            "1. reverse_words(s) — reverses word order in a string\n"
            "2. count_vowels(s) — counts vowels (a, e, i, o, u, case-insensitive)\n"
            "3. remove_duplicates(lst) — removes duplicates preserving order\n"
            "4. flatten(nested) — flattens a nested list of any depth\n\n"
            "Create test_transforms.py with thorough tests. "
            "Run the tests. If any fail, fix the implementation and re-run."
        ),
        "expect_tools": ["write_file", "run_command"],
    },
    {
        "name": "real_bug_hunt",
        "category": "real_world",
        "task": (
            "Create buggy.py with this deliberately buggy code:\n\n"
            "def fibonacci(n):\n"
            "    if n <= 0:\n"
            "        return []\n"
            "    if n == 1:\n"
            "        return [0]\n"
            "    result = [0, 1]\n"
            "    for i in range(2, n):\n"
            "        result.append(result[i] + result[i-1])\n"
            "    return result\n\n"
            "def find_max(lst):\n"
            "    max_val = 0\n"
            "    for item in lst:\n"
            "        if item > max_val:\n"
            "            max_val = item\n"
            "    return max_val\n\n"
            "Create test_buggy.py that tests:\n"
            "- fibonacci(7) should return [0, 1, 1, 2, 3, 5, 8]\n"
            "- find_max([-5, -3, -1]) should return -1 (not 0)\n\n"
            "Run the tests, find the failures, fix the bugs, and re-run until they pass."
        ),
        "expect_tools": ["write_file", "run_command", "edit_file"],
    },

    # ── Edge Cases ──
    {
        "name": "edge_empty_file",
        "category": "edge_case",
        "task": (
            "Create an empty file called empty.py (no content at all). "
            "Then read it back and confirm it's empty."
        ),
        "expect_tools": ["write_file", "read_file"],
    },
    {
        "name": "edge_overwrite_file",
        "category": "edge_case",
        "task": (
            "Create a file called overwrite.txt with content 'version 1'. "
            "Then read it back. Then create the same file with content 'version 2' "
            "(overwriting it). Read it back again and confirm it says 'version 2'."
        ),
        "expect_tools": ["write_file", "read_file"],
    },

    # ── Instruction Following ──
    {
        "name": "instruct_exact_content",
        "category": "instruction",
        "task": (
            "Create a file called exact.py with EXACTLY this content "
            "(no additions, no modifications):\n\n"
            "# Exact content test\n"
            "x = 42\n"
            "print(x)\n"
        ),
        "expect_tools": ["write_file"],
    },
    {
        "name": "instruct_multi_constraint",
        "category": "instruction",
        "task": (
            "Create constrained.py following ALL of these constraints:\n"
            "1. Must have exactly 3 functions\n"
            "2. Each function must have a docstring\n"
            "3. No function may be longer than 5 lines (including the def line)\n"
            "4. All functions must take exactly 2 parameters\n"
            "5. File must pass python syntax check\n\n"
            "Verify syntax after creating the file."
        ),
        "expect_tools": ["write_file", "run_command"],
    },

    # ── Heavy / Budget Pressure ──
    {
        "name": "heavy_create_test_suite",
        "category": "heavy",
        "task": (
            "Create a mini test suite with 3 modules and their tests:\n\n"
            "1. stack.py — a Stack class with push, pop, peek, is_empty, size\n"
            "2. queue_ds.py — a Queue class with enqueue, dequeue, peek, is_empty, size\n"
            "3. linked_list.py — a LinkedList class with append, prepend, find, delete, to_list\n\n"
            "For each, create a corresponding test_*.py file with pytest tests covering "
            "all methods including edge cases (empty collection operations). "
            "Run ALL tests together: python -m pytest test_stack.py "
            "test_queue_ds.py test_linked_list.py -v"
        ),
        "expect_tools": ["write_file", "run_command"],
    },

    # ── Simple Chat (no tools needed) ──
    {
        "name": "chat_greeting",
        "category": "simple_chat",
        "task": "Hello! How are you today?",
        "expect_tools": [],
        "force_plan": False,
    },
    {
        "name": "chat_code_question",
        "category": "simple_chat",
        "task": "Write me a Python function that checks if a string is a palindrome. Just show the code, don't create any files.",
        "expect_tools": [],
        "force_plan": False,
    },
]


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Result of running one test against one model."""
    test_name: str
    model: str
    task: str = ""
    completed: bool = False
    completion_method: str = "unknown"
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    first_tool_seconds: float = 0.0
    response_preview: str = ""


# ---------------------------------------------------------------------------
# Stream collector — captures tool call events from on_token output
# ---------------------------------------------------------------------------

class StreamCollector:
    """Captures streaming output and extracts tool call markers."""

    def __init__(self, quiet: bool = False):
        self.raw_output: list[str] = []
        self.tool_calls: list[dict] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.quiet = quiet
        self._first_tool_time: float | None = None
        self._start_time: float = time.monotonic()

    def on_token(self, chunk: str):
        self.raw_output.append(chunk)

        # Tool call markers from ChatLoop
        tool_match = re.search(r"▸ (\w+)", chunk)
        if tool_match:
            name = tool_match.group(1)
            if name != "Running":  # skip "Running N tools:" message
                if self._first_tool_time is None:
                    self._first_tool_time = time.monotonic()
                self.tool_calls.append({"name": name, "success": True})

        # Error markers
        if "✘" in chunk or "Error:" in chunk:
            err = chunk.strip()[:200]
            if err and err not in self.errors:
                self.errors.append(err)
            if self.tool_calls:
                self.tool_calls[-1]["success"] = False

        # Warnings
        if "[Budget warning" in chunk:
            self.warnings.append("Budget warning")
        if "[Forced completion" in chunk:
            self.warnings.append("Forced completion")
        if "[duplicate blocked]" in chunk or "[Duplicate call blocked]" in chunk:
            self.warnings.append("Duplicate blocked")

        if not self.quiet:
            console.print(chunk, end="", highlight=False)

    @property
    def first_tool_seconds(self) -> float:
        if self._first_tool_time is None:
            return 0.0
        return round(self._first_tool_time - self._start_time, 2)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def create_tool_registry(sandbox_path: str) -> ToolRegistry:
    """Create a fresh ToolRegistry with all tools pointed at the sandbox."""
    registry = ToolRegistry()

    registry.register(ReadFileTool(root_path=sandbox_path), group="filesystem")
    registry.register(WriteFileTool(root_path=sandbox_path), group="filesystem")
    registry.register(EditFileTool(root_path=sandbox_path), group="filesystem")
    registry.register(ListDirectoryTool(root_path=sandbox_path), group="filesystem")
    registry.register(GrepTool(root_path=sandbox_path), group="code")
    registry.register(GlobTool(root_path=sandbox_path), group="code")
    registry.register(ShellTool(timeout=30, cwd=sandbox_path), group="shell")
    registry.register(TaskCompleteTool(), group="general")

    # Headless ask_user — returns a canned response
    ask_user = AskUserTool()
    async def _headless_ask(question: str) -> str:
        return "Name it user_file.py with content: print('hello from ask_user')"
    ask_user.callback = _headless_ask
    registry.register(ask_user, group="general")

    return registry


async def run_one_test(
    client,
    model: str,
    test: dict,
    sandbox_path: str,
    budget: int = 30,
    context_tokens: int = 131072,
    quiet: bool = False,
    is_openai: bool = False,
) -> TestResult:
    """Run a single test against a model. Returns structured result."""

    test_name = test["name"]
    task = test["task"]
    force_plan = test.get("force_plan", True)

    # Fresh tool registry per test — clean state
    registry = create_tool_registry(sandbox_path)

    # Build messages
    sys_prompt = build_system_prompt()
    task_prompt = f"Task: {task}" if force_plan else task
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": task_prompt},
    ]

    # Configure loop
    config = LoopConfig(
        budget=budget if force_plan else 10,
        enable_dedup=True,
        enable_compaction=True,
        compaction_threshold=0.85,
        context_limit=context_tokens,
        completion_tool="task_complete" if force_plan else None,
        capture_inline_text=True,
        auto_continue_on_exhaustion=True,
        enable_parallel=True,
        max_parallel=8,
    )

    collector = StreamCollector(quiet=quiet)
    # Ollama needs num_ctx; OpenAI-compatible APIs ignore it (stripped by client)
    chat_kwargs = {"options": {"num_ctx": context_tokens}} if not is_openai else {}

    if not quiet:
        console.print(f"\n[bold cyan]─── {test_name} ({model}) ───[/bold cyan]")
        console.print(f"[dim]{task[:120]}{'...' if len(task)>120 else ''}[/dim]\n")

    loop = ChatLoop(registry, collector.on_token)
    start = time.monotonic()

    try:
        result = await loop.run(
            client=client,
            messages=messages,
            model=model,
            tools=registry.get_all_ollama_tools(),
            chat_kwargs=chat_kwargs,
            config=config,
        )
    except Exception as e:
        elapsed = round(time.monotonic() - start, 2)
        return TestResult(
            test_name=test_name,
            model=model,
            task=task,
            completed=False,
            completion_method="error",
            errors=[str(e)],
            elapsed_seconds=elapsed,
            response_preview=f"FATAL: {e}",
        )

    elapsed = round(time.monotonic() - start, 2)

    # Determine completion
    completed = False
    completion_method = result.completion_method

    raw_joined = "".join(collector.raw_output)
    if "[Task complete signal received]" in raw_joined:
        completed = True
        completion_method = "task_complete"
    elif result.completion_method == "tool":
        completed = True
        completion_method = "task_complete"
    elif result.completion_method == "text" and result.response:
        completed = True
        completion_method = "text"
    elif result.completion_method == "nudge":
        completed = True
        completion_method = "nudge"
    elif not force_plan and result.response and len(result.response.strip()) > 0:
        completed = True
        completion_method = "simple_chat"
    elif force_plan and result.response and len(result.response.strip()) > 50 and result.tool_call_count > 0:
        completed = True
        completion_method = "text_after_tools"

    if not quiet:
        status = "[bold green]PASS[/bold green]" if completed else "[bold red]FAIL[/bold red]"
        console.print(f"\n{status} — {completion_method} — {result.tool_call_count} tools — {elapsed}s")

    return TestResult(
        test_name=test_name,
        model=model,
        task=task,
        completed=completed,
        completion_method=completion_method,
        tool_calls=collector.tool_calls,
        tool_call_count=result.tool_call_count,
        errors=collector.errors,
        warnings=collector.warnings,
        elapsed_seconds=elapsed,
        first_tool_seconds=collector.first_tool_seconds,
        response_preview=(result.response or "")[:500],
    )


async def run_suite(
    client,
    model: str,
    tests: list[dict],
    sandbox_base: str,
    budget: int = 30,
    context_tokens: int = 131072,
    quiet: bool = False,
    completed_keys: set[str] | None = None,
    is_openai: bool = False,
) -> list[TestResult]:
    """Run a test suite against one model. Returns list of results."""

    results = []
    total = len(tests)

    for i, test in enumerate(tests, 1):
        test_name = test["name"]
        key = f"{model}:{test_name}"

        if completed_keys and key in completed_keys:
            if not quiet:
                console.print(f"[dim]  Skipping {test_name} (already done)[/dim]")
            continue

        if not quiet:
            console.print(f"\n[bold]═══ Test {i}/{total}: {test_name} ═══[/bold]")

        # Fresh sandbox per test — sanitize model name for path
        safe_model = model.replace(":", "_").replace("/", "__")
        sandbox = os.path.join(sandbox_base, f"{safe_model}__{test_name}")
        os.makedirs(sandbox, exist_ok=True)

        try:
            result = await run_one_test(
                client=client,
                model=model,
                test=test,
                sandbox_path=sandbox,
                budget=budget,
                context_tokens=context_tokens,
                quiet=quiet,
                is_openai=is_openai,
            )
            results.append(result)

            # Track token usage for OpenAI clients
            if is_openai and hasattr(client, 'get_usage'):
                usage = client.get_usage()
                if not quiet:
                    console.print(
                        f"[dim]  Tokens: {usage['prompt_tokens']}p + "
                        f"{usage['completion_tokens']}c = "
                        f"{usage['prompt_tokens'] + usage['completion_tokens']}t "
                        f"({usage['requests']} requests)[/dim]"
                    )

        except Exception as e:
            console.print(f"[bold red]Test {test_name} crashed: {e}[/bold red]")
            results.append(TestResult(
                test_name=test_name,
                model=model,
                task=test["task"],
                completed=False,
                completion_method="crash",
                errors=[str(e)],
            ))

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(model_a: str, model_b: str,
                     results_a: list[TestResult], results_b: list[TestResult]):
    """Print a Rich comparison table of results."""

    # Index by test name
    by_name_a = {r.test_name: r for r in results_a}
    by_name_b = {r.test_name: r for r in results_b}

    all_names = list(dict.fromkeys(
        [r.test_name for r in results_a] + [r.test_name for r in results_b]
    ))

    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Test", style="cyan", min_width=25)
    table.add_column(model_a, min_width=20)
    table.add_column(model_b, min_width=20)
    table.add_column("Winner", min_width=10)

    wins_a = wins_b = ties = 0
    time_a = time_b = 0.0
    tools_a = tools_b = 0

    for name in all_names:
        ra = by_name_a.get(name)
        rb = by_name_b.get(name)

        def fmt(r: TestResult | None) -> str:
            if r is None:
                return "[dim]skipped[/dim]"
            status = "✓" if r.completed else "✗"
            return f"{status} {r.completion_method} | {r.tool_call_count}tc | {r.elapsed_seconds}s"

        # Determine winner
        winner = ""
        if ra and rb:
            time_a += ra.elapsed_seconds
            time_b += rb.elapsed_seconds
            tools_a += ra.tool_call_count
            tools_b += rb.tool_call_count

            if ra.completed and not rb.completed:
                winner = f"[green]{model_a}[/green]"
                wins_a += 1
            elif rb.completed and not ra.completed:
                winner = f"[green]{model_b}[/green]"
                wins_b += 1
            elif ra.completed and rb.completed:
                # Both completed — compare efficiency
                if ra.elapsed_seconds < rb.elapsed_seconds * 0.8:
                    winner = f"[green]{model_a}[/green]"
                    wins_a += 1
                elif rb.elapsed_seconds < ra.elapsed_seconds * 0.8:
                    winner = f"[green]{model_b}[/green]"
                    wins_b += 1
                else:
                    winner = "tie"
                    ties += 1
            else:
                winner = "both failed"
                ties += 1

        table.add_row(name, fmt(ra), fmt(rb), winner)

    console.print(table)

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  {model_a}: {sum(1 for r in results_a if r.completed)}/{len(results_a)} passed, "
                  f"{tools_a} total tools, {time_a:.1f}s total")
    console.print(f"  {model_b}: {sum(1 for r in results_b if r.completed)}/{len(results_b)} passed, "
                  f"{tools_b} total tools, {time_b:.1f}s total")
    console.print(f"  Wins: {model_a}={wins_a}, {model_b}={wins_b}, ties={ties}")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed(path: str) -> tuple[list[dict], set[str]]:
    """Load previous results and extract completed test keys."""
    if not os.path.exists(path):
        return [], set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    keys = set()
    for r in results:
        if r.get("completed"):
            keys.add(f"{r['model']}:{r['test_name']}")
    return results, keys


def save_results(path: str, model_a: str, model_b: str,
                 results: list[TestResult], previous: list[dict] | None = None):
    """Save results to JSON."""
    all_results = list(previous or [])
    for r in results:
        all_results.append(asdict(r))

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": [model_a, model_b],
        "results": all_results,
        "summary": {
            model_a: {
                "passed": sum(1 for r in results if r.model == model_a and r.completed),
                "total": sum(1 for r in results if r.model == model_a),
                "total_time": round(sum(r.elapsed_seconds for r in results if r.model == model_a), 1),
                "total_tools": sum(r.tool_call_count for r in results if r.model == model_a),
            },
            model_b: {
                "passed": sum(1 for r in results if r.model == model_b and r.completed),
                "total": sum(1 for r in results if r.model == model_b),
                "total_time": round(sum(r.elapsed_seconds for r in results if r.model == model_b), 1),
                "total_tools": sum(r.tool_call_count for r in results if r.model == model_b),
            },
        },
    }

    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_env(val: str) -> str:
    """Resolve ${ENV_VAR} or $ENV_VAR references in a string."""
    if not val:
        return val
    if val.startswith("${") and val.endswith("}"):
        return os.environ.get(val[2:-1], val)
    if val.startswith("$") and not val.startswith("${"):
        return os.environ.get(val[1:], val)
    return val


async def main():
    parser = argparse.ArgumentParser(
        description="Standalone model benchmark — no Agent/memory/DB required",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Two Ollama models\n"
            "  python scripts/benchmark_standalone.py qwen3:14b glm-5:cloud\n\n"
            "  # Two OpenRouter models\n"
            "  python scripts/benchmark_standalone.py openai/gpt-4o anthropic/claude-sonnet-4 \\\n"
            "      --provider openai --api-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY\n\n"
            "  # Mixed: Ollama model A vs OpenRouter model B\n"
            "  python scripts/benchmark_standalone.py qwen3:14b openai/gpt-4o \\\n"
            "      --provider-b openai --api-url-b https://openrouter.ai/api/v1 --api-key-b $OPENROUTER_API_KEY\n"
        ),
    )
    parser.add_argument("model_a", help="First model name (e.g. qwen3:14b or openai/gpt-4o)")
    parser.add_argument("model_b", help="Second model name")

    # Shared defaults (apply to both models unless overridden)
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"],
                        help="LLM provider for both models (default: ollama)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama host URL (default: localhost:11434)")
    parser.add_argument("--api-url", default="",
                        help="OpenAI-compatible API base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--api-key", default="",
                        help="API key (supports $ENV_VAR syntax)")

    # Per-model overrides (model B can use a different provider)
    parser.add_argument("--provider-b", default=None, choices=["ollama", "openai"],
                        help="Override provider for model B (default: same as --provider)")
    parser.add_argument("--host-b", default=None,
                        help="Override Ollama host for model B")
    parser.add_argument("--api-url-b", default=None,
                        help="Override API URL for model B")
    parser.add_argument("--api-key-b", default=None,
                        help="Override API key for model B")

    # Test configuration
    parser.add_argument("--suite", choices=["canned", "stress", "all"],
                        default="stress", help="Test suite to run (default: stress)")
    parser.add_argument("--output", "-o", default="data/benchmark_standalone_results.json",
                        help="Output JSON path")
    parser.add_argument("--resume", metavar="PATH",
                        help="Resume from previous results file (skip completed tests)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output (JSON results only)")
    parser.add_argument("--budget", type=int, default=30,
                        help="Tool call budget per test (default: 30)")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="LLM timeout in seconds (default: 180)")
    parser.add_argument("--context", type=int, default=131072,
                        help="Context window size (default: 131072)")
    parser.add_argument("--sandbox", default=None,
                        help="Sandbox directory (default: temp dir)")

    args = parser.parse_args()

    # Resolve per-model settings (B inherits from shared unless overridden)
    provider_a = args.provider
    provider_b = args.provider_b or args.provider
    host_a = args.host
    host_b = args.host_b or args.host
    api_url_a = args.api_url
    api_url_b = args.api_url_b or args.api_url
    api_key_a = resolve_env(args.api_key)
    api_key_b = resolve_env(args.api_key_b or args.api_key)

    # Select test suite
    if args.suite == "canned":
        tests = CANNED_TESTS
    elif args.suite == "stress":
        tests = STRESS_TESTS
    elif args.suite == "all":
        tests = CANNED_TESTS + STRESS_TESTS

    if not args.quiet:
        console.print(f"[bold]Standalone Model Benchmark[/bold]")
        console.print(f"  Model A: {args.model_a} ({provider_a})")
        console.print(f"  Model B: {args.model_b} ({provider_b})")
        if provider_a == "openai":
            console.print(f"  API URL A: {api_url_a}")
        else:
            console.print(f"  Host A: {host_a}")
        if provider_b == "openai":
            console.print(f"  API URL B: {api_url_b}")
        elif provider_b != provider_a or host_b != host_a:
            console.print(f"  Host B: {host_b}")
        console.print(f"  Suite: {args.suite} ({len(tests)} tests)")
        console.print(f"  Budget: {args.budget} tools/test")
        console.print(f"  Timeout: {args.timeout}s")
        console.print(f"  Context: {args.context}")

    # Resume support
    previous_results = []
    completed_keys = set()
    if args.resume:
        previous_results, completed_keys = load_completed(args.resume)
        if not args.quiet:
            console.print(f"  Resuming: {len(completed_keys)} tests already done")

    # Create clients for each model
    try:
        client_a = create_client(
            provider=provider_a, host=host_a,
            api_url=api_url_a, api_key=api_key_a,
            timeout=args.timeout,
        )
    except ValueError as e:
        console.print(f"[bold red]Model A client error: {e}[/bold red]")
        sys.exit(1)

    try:
        client_b = create_client(
            provider=provider_b, host=host_b,
            api_url=api_url_b, api_key=api_key_b,
            timeout=args.timeout,
        )
    except ValueError as e:
        console.print(f"[bold red]Model B client error: {e}[/bold red]")
        sys.exit(1)

    # Health check
    for label, client, provider in [("A", client_a, provider_a), ("B", client_b, provider_b)]:
        try:
            healthy = await client.check_health()
            if not healthy:
                console.print(f"[bold yellow]Warning: Model {label} endpoint health check failed[/bold yellow]")
        except Exception as e:
            console.print(f"[bold yellow]Warning: Model {label} health check error: {e}[/bold yellow]")

    # For Ollama clients, verify models exist
    if provider_a == "ollama":
        try:
            model_names = await client_a.list_models()
            if not any(args.model_a in n or n.startswith(args.model_a) for n in model_names):
                console.print(f"[bold yellow]Warning: '{args.model_a}' not found in Ollama. "
                              f"Available: {', '.join(model_names[:10])}[/bold yellow]")
        except Exception:
            pass
    if provider_b == "ollama":
        try:
            model_names = await client_b.list_models()
            if not any(args.model_b in n or n.startswith(args.model_b) for n in model_names):
                console.print(f"[bold yellow]Warning: '{args.model_b}' not found in Ollama. "
                              f"Available: {', '.join(model_names[:10])}[/bold yellow]")
        except Exception:
            pass

    # Create sandbox
    if args.sandbox:
        sandbox_base = args.sandbox
        os.makedirs(sandbox_base, exist_ok=True)
        cleanup_sandbox = False
    else:
        sandbox_base = tempfile.mkdtemp(prefix="blip_bench_")
        cleanup_sandbox = True

    if not args.quiet:
        console.print(f"  Sandbox: {sandbox_base}")
        console.print()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_results: list[TestResult] = []

    try:
        # Run model A
        if not args.quiet:
            console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
            console.print(f"[bold magenta]  MODEL A: {args.model_a} ({provider_a})[/bold magenta]")
            console.print(f"[bold magenta]{'='*60}[/bold magenta]")

        results_a = await run_suite(
            client=client_a,
            model=args.model_a,
            tests=tests,
            sandbox_base=sandbox_base,
            budget=args.budget,
            context_tokens=args.context,
            quiet=args.quiet,
            completed_keys=completed_keys,
            is_openai=(provider_a == "openai"),
        )
        all_results.extend(results_a)

        # Save intermediate results
        save_results(args.output, args.model_a, args.model_b,
                     all_results, previous_results)

        # Run model B
        if not args.quiet:
            console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
            console.print(f"[bold magenta]  MODEL B: {args.model_b} ({provider_b})[/bold magenta]")
            console.print(f"[bold magenta]{'='*60}[/bold magenta]")

        results_b = await run_suite(
            client=client_b,
            model=args.model_b,
            tests=tests,
            sandbox_base=sandbox_base,
            budget=args.budget,
            context_tokens=args.context,
            quiet=args.quiet,
            completed_keys=completed_keys,
            is_openai=(provider_b == "openai"),
        )
        all_results.extend(results_b)

        # Final save
        save_results(args.output, args.model_a, args.model_b,
                     all_results, previous_results)

        # Token usage summary for OpenAI clients
        if not args.quiet:
            for label, client, provider in [("A", client_a, provider_a), ("B", client_b, provider_b)]:
                if provider == "openai" and hasattr(client, 'get_usage'):
                    usage = client.get_usage()
                    total_tokens = usage['prompt_tokens'] + usage['completion_tokens']
                    console.print(
                        f"\n[bold]Token usage ({label}):[/bold] "
                        f"{usage['prompt_tokens']:,}p + {usage['completion_tokens']:,}c "
                        f"= {total_tokens:,} total ({usage['requests']} requests)"
                    )

        # Comparison table
        if not args.quiet and results_a and results_b:
            console.print(f"\n[bold]{'='*60}[/bold]")
            print_comparison(args.model_a, args.model_b, results_a, results_b)

    finally:
        if cleanup_sandbox:
            try:
                shutil.rmtree(sandbox_base, ignore_errors=True)
            except Exception:
                pass

    if not args.quiet:
        console.print(f"\n[bold green]Results saved to {args.output}[/bold green]")
    else:
        # Quiet mode — print JSON to stdout
        with open(args.output, "r", encoding="utf-8") as f:
            print(f.read())


if __name__ == "__main__":
    asyncio.run(main())
