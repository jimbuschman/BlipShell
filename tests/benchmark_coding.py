"""Benchmark coding models on real coding tasks against the BlipShell codebase.

Copies the real BlipShell source to a temp directory, sets up project context
(repo map, git tools, system prompt) matching what the agent uses in /project mode,
and runs realistic coding tasks through TaskExecutor.execute_dynamic().

Captures both raw transcripts (for manual side-by-side comparison) and structured
metrics JSON (for Rich tables).

Output files:
    data/benchmark_coding_results.json              — structured metrics (incremental merge)
    data/benchmark_coding_transcripts/              — raw output per model+task
        qwen3-coder_480b-cloud__stats_command__2026-02-22T14-30-00.txt

Usage:
    python tests/benchmark_coding.py                                              # all default models
    python tests/benchmark_coding.py qwen3-coder:480b-cloud devstral-2:123b-cloud # specific models
    python tests/benchmark_coding.py --timeout 600                                # longer timeout
    python tests/benchmark_coding.py --hard                                       # hard tasks only
    python tests/benchmark_coding.py --expert                                     # expert tasks only
    python tests/benchmark_coding.py --all                                        # standard + hard + expert
    python tests/benchmark_coding.py --dry-run-verify                             # test checks on unmodified code
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from blipshell.core.repo_map import RepoMap
from blipshell.core.tools.base import ToolRegistry
from blipshell.core.tools.code_tools import GlobTool, GrepTool
from blipshell.core.tools.filesystem import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from blipshell.core.tools.git_tools import (
    GitAddTool,
    GitCommitTool,
    GitDiffTool,
    GitStatusTool,
)
from blipshell.core.tools.shell import ShellTool
from blipshell.core.executor import TaskExecutor
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.model_settings import ModelSettings, ModelSettingsRegistry
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig, PlannerConfig

# ---------------------------------------------------------------------------
# Models to benchmark (all routed through local Ollama which proxies to cloud)
# ---------------------------------------------------------------------------
BENCHMARK_MODELS = [
    # Current coding model (baseline)
    "glm-5:cloud",
    # Ollama cloud models to evaluate
    "qwen3-coder:480b-cloud",
    "qwen3-coder-next:cloud",
    "devstral-2:123b-cloud",
    "deepseek-v3.2:cloud",
    "kimi-k2.5:cloud",
    "cogito-2.1:671b-cloud",
    "qwen3-next:80b-cloud",
    "minimax-m2.5:cloud",
    "qwen3.5:397b-cloud",
    # Local models
    "qwen3.5:9b",
    "qwen3.5:4b",
    "qwen3:14b",
    "qwen2.5-coder:7b",
]

OLLAMA_URL = "http://localhost:11434"

# Directories/files to exclude when copying the project
COPY_EXCLUDES = {".git", "data", "__pycache__", "backups", ".vs", "NUL",
                 ".pytest_cache", ".mypy_cache", "dist", "build", ".eggs",
                 ".venv", "venv", ".tox", "node_modules"}

# Model settings loaded from config.yaml (same as production)
_MODEL_SETTINGS_CONFIG = {
    "qwen3-coder": {
        "max_tool_calls": 20,
        "use_repo_map": True,
        "extra_instructions": "Prefer editing existing files over creating new ones. Keep changes minimal.",
    },
    "devstral": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "extra_instructions": "Prefer editing existing files over creating new ones.",
    },
    "gpt-oss": {
        "max_tool_calls": 10,
        "use_repo_map": True,
        "think": False,
        "extra_instructions": "Be direct. Skip exploration and go straight to implementation.",
    },
    "glm-5": {
        "max_tool_calls": 15,
        "use_repo_map": True,
    },
    # MiniMax via Ollama cloud (:cloud forms) — identical settings across all
    # three versions so the model is the only variable in the A/B. Matches the
    # OpenRouter minimax profiles below (25 calls, repo map, no think).
    "minimax-m2.5": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "minimax-m2.7": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "minimax-m3": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    # OpenRouter models (matched via prefix scan, e.g. "anthropic/claude-sonnet-4-6")
    "anthropic/claude-sonnet": {
        "max_tool_calls": 30,
        "use_repo_map": True,
        "think": False,
    },
    "anthropic/claude-haiku": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "google/gemini-2.5-flash": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "google/gemini-2.5-pro": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "z-ai/glm-5": {
        "max_tool_calls": 15,
        "use_repo_map": True,
    },
    "minimax/minimax-m2.5": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "minimax/minimax-m3": {
        # Same profile as m2.5 baseline so the only variable is the model.
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "moonshotai/kimi-k2.5": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "qwen/qwen3-coder-next": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "mistralai/devstral-2512": {
        "max_tool_calls": 25,
        "use_repo_map": True,
        "think": False,
    },
    "z-ai/glm-4.7-flash": {
        "max_tool_calls": 20,
        "use_repo_map": True,
        "think": False,
    },
}

console = Console()

# Project root (real BlipShell source to copy from)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Sandbox setup — full codebase copy
# ---------------------------------------------------------------------------

def create_project_sandbox() -> str:
    """Copy the real BlipShell source to a temp directory.

    Excludes .git/, data/, __pycache__/, backups/, .vs/, NUL.
    Initializes a fresh git repo + initial commit so git tools work.
    Returns the sandbox path.
    """
    sandbox = tempfile.mkdtemp(prefix="blip_bench_")

    def _ignore(directory: str, contents: list[str]) -> set[str]:
        return {c for c in contents if c in COPY_EXCLUDES}

    shutil.copytree(str(PROJECT_ROOT), sandbox, ignore=_ignore, dirs_exist_ok=True)

    # Initialize git repo so git tools work
    subprocess.run(
        ["git", "init"], cwd=sandbox,
        capture_output=True, timeout=10,
    )
    subprocess.run(
        ["git", "add", "."], cwd=sandbox,
        capture_output=True, timeout=30,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial benchmark snapshot"],
        cwd=sandbox, capture_output=True, timeout=30,
        env={**os.environ, "GIT_AUTHOR_NAME": "benchmark",
             "GIT_AUTHOR_EMAIL": "bench@test", "GIT_COMMITTER_NAME": "benchmark",
             "GIT_COMMITTER_EMAIL": "bench@test"},
    )

    return sandbox


def reset_sandbox(sandbox_path: str):
    """Reset sandbox to initial commit state (faster than re-copying)."""
    subprocess.run(
        ["git", "checkout", "."], cwd=sandbox_path,
        capture_output=True, timeout=10,
    )
    subprocess.run(
        ["git", "clean", "-fd"], cwd=sandbox_path,
        capture_output=True, timeout=10,
    )


# ---------------------------------------------------------------------------
# Project context (standalone — ports Agent._scan_project_context)
# ---------------------------------------------------------------------------

def build_project_context(sandbox_path: str) -> str:
    """Build project context string matching what Agent._scan_project_context() produces.

    Includes: project metadata, git info, repo map, file tree, key files.
    """
    root = Path(sandbox_path)
    parts = [
        "Project: blipshell",
        f"Root: {sandbox_path}",
        "Language: Python",
    ]

    # Git info
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=sandbox_path, capture_output=True, text=True, timeout=5,
        )
        if branch.returncode == 0:
            parts.append(f"Branch: {branch.stdout.strip()}")

        log = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=sandbox_path, capture_output=True, text=True, timeout=5,
        )
        if log.returncode == 0 and log.stdout.strip():
            parts.append(f"\nRecent commits:\n{log.stdout.strip()}")
    except Exception:
        pass

    # Code map (AST-based structure)
    repo_map = RepoMap(sandbox_path)
    code_map = repo_map.build(max_lines=120)
    if code_map:
        parts.append(f"\nCode structure (classes, functions):\n{code_map}")

    # Compact file tree (top level)
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv",
                 ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
                 ".vs", ".idea", ".vscode", "backups"}
    tree_lines = []
    for entry in sorted(root.iterdir()):
        if entry.name in skip_dirs:
            continue
        prefix = "[DIR] " if entry.is_dir() else "      "
        tree_lines.append(f"  {prefix}{entry.name}")
    if tree_lines:
        parts.append("\nTop-level layout:\n" + "\n".join(tree_lines[:40]))

    # Key files
    key_files = ["pyproject.toml", "CLAUDE.md"]
    for fname in key_files:
        fpath = root / fname
        if fpath.is_file():
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()[:60]
                truncated = "\n".join(lines)
                if len(content.splitlines()) > 60:
                    truncated += "\n... (truncated)"
                parts.append(f"\n=== {fname} ===\n{truncated}")
            except Exception:
                pass

    return "\n".join(parts)


def build_executor_system_prompt(
    sandbox_path: str,
    project_context: str,
    tool_limit: int = 15,
    extra_instructions: str = "",
) -> str:
    """Build the system prompt matching what Agent injects in project mode.

    Ports agent.py:1444-1477 — PROJECT CONTEXT header, execution mode,
    tool discipline rules, platform info.
    """
    base = (
        "You are BlipShell, a helpful local AI assistant with persistent memory.\n"
        "You remember previous conversations and learn from interactions.\n"
        "Be concise and helpful. Use your memory to provide personalized assistance.\n\n"
        "IMPORTANT: You have tools available. You MUST use them when appropriate:\n"
        "- Use read_file, write_file, edit_file, list_directory for file operations.\n"
        "- Use run_command to execute shell commands.\n"
        "- Use grep_files, glob_files for searching code.\n"
        "- Use git_status, git_diff, git_add, git_commit for git operations.\n"
    )

    project_block = (
        "\n\n--- PROJECT CONTEXT ---\n"
        f'You are working on the project "blipshell".\n'
        f"Project root: {sandbox_path}\n"
        "All file tools (read_file, write_file, edit_file, list_directory) resolve "
        "relative paths against this project root. Use relative paths like "
        "'blipshell/ui/cli.py', NOT absolute paths.\n"
        "The run_command tool also runs from the project root.\n\n"
        "EXECUTION MODE: You are an autonomous coding agent.\n"
        "- Execute tasks fully without asking for permission or confirmation.\n"
        "- Make decisions yourself. Do NOT ask 'should I?', 'want me to?', 'which approach?'.\n"
        "- Only ask the user if you genuinely cannot determine a critical requirement.\n"
        "- When finished, summarize what you built in 2-3 sentences.\n\n"
        "TOOL DISCIPLINE:\n"
        "- Read a file ONCE, then use what you learned. Do NOT re-read files.\n"
        "- List a directory ONCE. Do NOT re-list directories.\n"
        "- Do NOT run the same grep or glob search twice.\n"
        "- Always read a file before editing it.\n"
        "- Use grep_files/glob_files tools, NOT shell grep/find/wc.\n"
        "- NEVER launch interactive or full-screen apps via run_command (TUI, curses, Textual .run()). They destroy the terminal.\n"
        "- NEVER create documentation files (.md, README) unless explicitly asked.\n"
        f"- Target UNDER {tool_limit} tool calls per task. Read, write, test — do not explore endlessly.\n\n"
        "PLATFORM: Windows.\n"
        f"- Project root: {sandbox_path}\n"
        "- Do NOT use Linux commands (ls, cat, grep, head, tail, find, wc) in shell.\n"
        "- Use 'dir' not 'ls', 'type' not 'cat'. Or better: use the file/grep/glob tools.\n"
        "- Do NOT use 'cd' in run_command — it already runs from the project root.\n\n"
    )

    if extra_instructions:
        project_block += f"MODEL-SPECIFIC INSTRUCTIONS:\n{extra_instructions}\n\n"

    project_block += project_context

    return base + project_block


# ---------------------------------------------------------------------------
# Coding tasks — real BlipShell modifications
# ---------------------------------------------------------------------------

CODING_TASKS = [
    {
        "name": "stats_command",
        "description": "Add /stats CLI command showing memory statistics",
        "request": (
            "Add a /stats CLI command to BlipShell that shows memory statistics. "
            "It should display: total memory count, average rank (rounded to 1 decimal), "
            "top 5 most common tags with counts, and a breakdown of memory types "
            "(fact, preference, skill, event, conversation) with counts. "
            "Follow the existing command patterns in blipshell/ui/cli.py — "
            "register the command the same way other slash commands are registered. "
            "Get the data from SQLiteStore methods (check what's available)."
        ),
        "verify_checks": [
            # Model actually added new code (not pre-existing matches)
            ("diff_contains", ("blipshell/ui/cli.py", r"\+.*stats")),
            ("diff_contains", ("blipshell/", r"\+.*(SELECT|count|memories)")),
            # Modified files still parse
            ("syntax_check", "blipshell/ui/cli.py"),
            # No junk files
            ("no_unwanted_files", None),
            # Actually creates new code (not just narrating)
            ("files_changed", 1),
        ],
    },
    {
        "name": "dry_run_edit",
        "description": "Add dry_run parameter to edit_file tool",
        "request": (
            "Add a `dry_run` boolean parameter to the edit_file tool in "
            "blipshell/core/tools/filesystem.py. When dry_run=True, the tool "
            "should return a unified diff preview (using difflib.unified_diff) "
            "showing what would change, WITHOUT actually modifying the file. "
            "Update the tool's definition to include the new parameter. "
            "Also write a small test in tests/test_dry_run_edit.py that creates "
            "a temp file, calls edit_file with dry_run=True, and asserts the "
            "file was NOT modified and the output contains diff markers (--- and +++). "
            "Keep the test simple and self-contained."
        ),
        "verify_checks": [
            # Model added dry_run to the tool (in the diff, not pre-existing)
            ("diff_contains", ("blipshell/core/tools/filesystem.py", r"\+.*dry_run")),
            # Model added unified_diff usage (not just the existing import)
            ("diff_contains", ("blipshell/core/tools/filesystem.py", r"\+.*unified_diff")),
            # Modified file still compiles
            ("syntax_check", "blipshell/core/tools/filesystem.py"),
            # Test file exists and has assertions
            ("file_exists", "tests/test_dry_run_edit.py"),
            ("grep_in_sandbox", ("tests/test_dry_run_edit.py", r"assert")),
            # Test file compiles
            ("syntax_check", "tests/test_dry_run_edit.py"),
            # Run the model's test
            ("pytest_in_sandbox", "tests/test_dry_run_edit.py"),
            # Functional: dry_run param exists in tool definition
            ("functional_test",
             "from blipshell.core.tools.filesystem import EditFileTool; "
             "t = EditFileTool(); assert 'dry_run' in [p.name for p in t.definition().parameters]"),
        ],
    },
    {
        "name": "new_module_with_test",
        "description": "Create a new utility module and a test that imports it",
        "request": (
            "Create a new module blipshell/core/rate_tracker.py that tracks API call "
            "rates. It should contain:\n"
            "1. A class RateTracker with:\n"
            "   - __init__(self, window_seconds: int = 60) — time window for rate calc\n"
            "   - record_call(self) — records a call at current time\n"
            "   - get_rate(self) -> int — returns number of calls within the window\n"
            "   - is_limited(self, max_calls: int) -> bool — True if rate >= max_calls\n"
            "   - reset(self) — clears all recorded calls\n"
            "2. Use time.monotonic() for timestamps (not datetime).\n"
            "3. Store calls in a list and prune expired entries in get_rate().\n\n"
            "Then create tests/test_rate_tracker.py that tests:\n"
            "- Recording calls increments the rate\n"
            "- is_limited returns True when at/over limit\n"
            "- reset clears the rate to 0\n"
            "The test must import RateTracker from the correct module path "
            "(blipshell.core.rate_tracker). Keep both files simple and focused."
        ),
        "verify_checks": [
            # Module exists and compiles
            ("file_exists", "blipshell/core/rate_tracker.py"),
            ("syntax_check", "blipshell/core/rate_tracker.py"),
            # Has the RateTracker class
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"class RateTracker")),
            # Has the required methods
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"def record_call")),
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"def get_rate")),
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"def is_limited")),
            # Test file exists, compiles, and imports correctly
            ("file_exists", "tests/test_rate_tracker.py"),
            ("syntax_check", "tests/test_rate_tracker.py"),
            ("grep_in_sandbox", ("tests/test_rate_tracker.py", r"from blipshell\.core\.rate_tracker import")),
            # Test actually passes
            ("pytest_in_sandbox", "tests/test_rate_tracker.py"),
            # Functional: class is importable and works
            ("functional_test",
             "from blipshell.core.rate_tracker import RateTracker; "
             "rt = RateTracker(); rt.record_call(); "
             "assert rt.get_rate() == 1; "
             "assert not rt.is_limited(5); "
             "rt.reset(); assert rt.get_rate() == 0"),
        ],
    },
    {
        "name": "fix_shell_output_truncation",
        "description": "Fix shell tool to truncate long output (bug fix task)",
        "request": (
            "The ShellTool in blipshell/core/tools/shell.py has a problem: when a "
            "command produces very long output (e.g. 50,000+ characters), the entire "
            "output is returned to the LLM, wasting context window budget.\n\n"
            "Fix this by adding output truncation to the tool's execute method:\n"
            "- If stdout exceeds 5,000 characters, truncate to the first 2,000 chars "
            "+ a marker line '[... N characters truncated ...]' + the last 2,000 chars.\n"
            "- Apply the same truncation to stderr.\n"
            "- Do NOT truncate if the output is 5,000 characters or fewer.\n\n"
            "Also write a test in tests/test_shell_truncation.py that:\n"
            "1. Creates a ShellTool instance\n"
            "2. Runs a command that produces long output (e.g. python -c \"print('x' * 10000)\")\n"
            "3. Asserts the result contains the truncation marker\n"
            "4. Asserts the result length is less than the original output length\n"
            "5. Also tests that short output is NOT truncated\n"
            "Keep the test simple and self-contained."
        ),
        "verify_checks": [
            # Model added truncation logic
            ("diff_contains", ("blipshell/core/tools/shell.py", r"\+.*truncat")),
            # Model preserved the marker format
            ("diff_contains", ("blipshell/core/tools/shell.py", r"\+.*characters truncated")),
            # Modified file still compiles
            ("syntax_check", "blipshell/core/tools/shell.py"),
            # Test file exists and compiles
            ("file_exists", "tests/test_shell_truncation.py"),
            ("syntax_check", "tests/test_shell_truncation.py"),
            # Test has assertions
            ("grep_in_sandbox", ("tests/test_shell_truncation.py", r"assert")),
            # Test passes
            ("pytest_in_sandbox", "tests/test_shell_truncation.py"),
            # No junk files
            ("no_unwanted_files", None),
        ],
    },
    {
        "name": "multi_file_text_analyzer",
        "description": "Create a utility package with module, __init__ re-export, and tests (multi-file coordination)",
        "request": (
            "Create a text analysis utility package with three coordinated files:\n\n"
            "1. blipshell/utils/text_analyzer.py — a class TextAnalyzer with:\n"
            "   - word_count(text: str) -> int — count words (split on whitespace)\n"
            "   - extract_urls(text: str) -> list[str] — extract all http/https URLs using regex\n"
            "   - estimate_tokens(text: str) -> int — approximate token count (word_count * 1.3, rounded to int)\n\n"
            "2. blipshell/utils/__init__.py — import and re-export TextAnalyzer so it can be "
            "imported as 'from blipshell.utils import TextAnalyzer'\n\n"
            "3. tests/test_text_analyzer.py — tests for all three methods:\n"
            "   - word_count: empty string returns 0, normal sentence returns correct count\n"
            "   - extract_urls: text with URLs returns them, text without URLs returns empty list\n"
            "   - estimate_tokens: returns approximately word_count * 1.3\n"
            "   - The test MUST import from blipshell.utils (not blipshell.utils.text_analyzer)\n\n"
            "Note: blipshell/utils/ directory may not exist yet — create it if needed."
        ),
        "verify_checks": [
            # All three files exist
            ("file_exists", "blipshell/utils/text_analyzer.py"),
            ("file_exists", "blipshell/utils/__init__.py"),
            ("file_exists", "tests/test_text_analyzer.py"),
            # All compile
            ("syntax_check", "blipshell/utils/text_analyzer.py"),
            ("syntax_check", "blipshell/utils/__init__.py"),
            ("syntax_check", "tests/test_text_analyzer.py"),
            # Module has the class and methods
            ("grep_in_sandbox", ("blipshell/utils/text_analyzer.py", r"class TextAnalyzer")),
            ("grep_in_sandbox", ("blipshell/utils/text_analyzer.py", r"def word_count")),
            ("grep_in_sandbox", ("blipshell/utils/text_analyzer.py", r"def extract_urls")),
            ("grep_in_sandbox", ("blipshell/utils/text_analyzer.py", r"def estimate_tokens")),
            # __init__.py re-exports TextAnalyzer
            ("grep_in_sandbox", ("blipshell/utils/__init__.py", r"TextAnalyzer")),
            # Test imports from blipshell.utils (not blipshell.utils.text_analyzer)
            ("grep_in_sandbox", ("tests/test_text_analyzer.py", r"from blipshell\.utils import")),
            # Tests pass
            ("pytest_in_sandbox", "tests/test_text_analyzer.py"),
            # Functional: re-export actually works
            ("functional_test",
             "from blipshell.utils import TextAnalyzer; "
             "t = TextAnalyzer(); "
             "assert t.word_count('hello world') == 2; "
             "assert t.word_count('') == 0; "
             "urls = t.extract_urls('visit https://example.com today'); "
             "assert 'https://example.com' in urls; "
             "assert t.estimate_tokens('one two three') == 4"),
            # Created enough files
            ("files_changed", 3),
        ],
    },
]


# ---------------------------------------------------------------------------
# Hard tasks — codebase comprehension, diagnosis, edge cases, refactoring
# ---------------------------------------------------------------------------

HARD_TASKS = [
    {
        "name": "add_method_existing_class",
        "description": "Add method to existing class (requires reading + understanding existing code)",
        "request": (
            "Add a `get_memory_stats()` async method to `SQLiteStore` in "
            "blipshell/memory/sqlite_store.py. It should return a dict with:\n"
            "- 'total_memories': int — count of non-archived memories\n"
            "- 'avg_rank': float — average rank rounded to 1 decimal (0.0 if no ranked memories)\n"
            "- 'type_breakdown': dict — mapping memory_type string to count\n"
            "- 'oldest_date': str — created_at of oldest memory (ISO format), or '' if empty\n\n"
            "You MUST read sqlite_store.py first to understand the DB schema, table names, "
            "and coding patterns (async methods, aiosqlite usage, etc). Do NOT guess the "
            "schema — look at existing methods.\n\n"
            "Then write a test in tests/test_memory_stats.py that:\n"
            "1. Creates an in-memory SQLiteStore (see existing test fixtures for how)\n"
            "2. Inserts a couple of test memories\n"
            "3. Calls get_memory_stats() and asserts the structure is correct\n"
            "4. Asserts total_memories matches what was inserted"
        ),
        "verify_checks": [
            # Method added to the right file
            ("diff_contains", ("blipshell/memory/sqlite_store.py", r"\+.*async def get_memory_stats")),
            # Uses SQL queries (not just returning dummy data)
            ("diff_contains", ("blipshell/memory/sqlite_store.py", r"\+.*(SELECT|COUNT|AVG)")),
            # Modified file still compiles
            ("syntax_check", "blipshell/memory/sqlite_store.py"),
            # Test file exists and compiles
            ("file_exists", "tests/test_memory_stats.py"),
            ("syntax_check", "tests/test_memory_stats.py"),
            # Test has assertions
            ("grep_in_sandbox", ("tests/test_memory_stats.py", r"assert")),
            # Test imports from the right place
            ("grep_in_sandbox", ("tests/test_memory_stats.py", r"from blipshell\.memory\.sqlite_store")),
            # Test passes
            ("pytest_in_sandbox", "tests/test_memory_stats.py"),
            # Functional: method exists and returns a dict
            ("functional_test",
             "import asyncio; "
             "from blipshell.memory.sqlite_store import SQLiteStore; "
             "async def check(): "
             "  s = SQLiteStore(':memory:'); await s.initialize(); "
             "  stats = await s.get_memory_stats(); "
             "  assert isinstance(stats, dict); "
             "  assert 'total_memories' in stats; "
             "  assert 'avg_rank' in stats; "
             "  assert 'type_breakdown' in stats; "
             "  assert 'oldest_date' in stats; "
             "  assert stats['total_memories'] == 0; "
             "asyncio.run(check())"),
        ],
    },
    {
        "name": "edge_case_parser",
        "description": "Implement parser with many edge cases (algorithmic thinking)",
        "request": (
            "Create blipshell/utils/config_parser.py with a function:\n\n"
            "  parse_value(raw: str) -> int | float | bool | str\n\n"
            "Rules (applied in this priority order):\n"
            "1. If raw is empty or only whitespace → return ''\n"
            "2. Env vars: '${VAR_NAME}' → os.environ['VAR_NAME']. Raise KeyError if not set. "
            "The ENTIRE string must be '${...}' — not embedded in other text.\n"
            "3. Booleans: 'true'/'True'/'TRUE' → True, 'false'/'False'/'FALSE' → False "
            "(only exact case variants, not 'tRuE')\n"
            "4. Integers: string of optional '-' + digits → int (e.g., '42', '-7', '0')\n"
            "5. Floats: string matching optional '-' + digits + '.' + digits → float "
            "(e.g., '3.14', '-0.5'). Must have digits on both sides of the dot.\n"
            "6. Quoted strings: if starts AND ends with matching '\"' or \"'\", "
            "strip the outer quotes (e.g., '\"hello\"' → 'hello'). Single layer only.\n"
            "7. Everything else: return as-is (stripped of leading/trailing whitespace)\n\n"
            "Also create blipshell/utils/__init__.py if it doesn't exist (can be empty).\n\n"
            "Write tests/test_config_parser.py covering ALL these cases plus edge cases:\n"
            "- '' and '   ' return ''\n"
            "- '42' → 42, '-7' → -7, '0' → 0\n"
            "- '3.14' → 3.14, '-0.5' → -0.5\n"
            "- '.5' stays as string (no digit before dot)\n"
            "- '3.' stays as string (no digit after dot)\n"
            "- 'true' → True, 'FALSE' → False, 'tRuE' stays as string\n"
            "- '\"hello world\"' → 'hello world'\n"
            "- \"'single'\" → 'single'\n"
            "- '\"mismatched' stays as string\n"
            "- '${PATH}' returns os.environ['PATH']\n"
            "- '${NONEXISTENT_XYZ}' raises KeyError\n"
            "- 'prefix${VAR}' stays as string (not a pure env var)\n"
            "- '  42  ' → 42 (whitespace stripped before parsing)"
        ),
        "verify_checks": [
            # Module exists and compiles
            ("file_exists", "blipshell/utils/config_parser.py"),
            ("syntax_check", "blipshell/utils/config_parser.py"),
            # Has the function
            ("grep_in_sandbox", ("blipshell/utils/config_parser.py", r"def parse_value")),
            # Test file exists and compiles
            ("file_exists", "tests/test_config_parser.py"),
            ("syntax_check", "tests/test_config_parser.py"),
            # Test passes
            ("pytest_in_sandbox", "tests/test_config_parser.py"),
            # Functional: integers
            ("functional_test",
             "from blipshell.utils.config_parser import parse_value; "
             "assert parse_value('42') == 42; "
             "assert parse_value('-7') == -7; "
             "assert parse_value('0') == 0; "
             "assert type(parse_value('42')) is int"),
            # Functional: floats and non-floats
            ("functional_test",
             "from blipshell.utils.config_parser import parse_value; "
             "assert parse_value('3.14') == 3.14; "
             "assert type(parse_value('3.14')) is float; "
             "assert parse_value('.5') == '.5'; "
             "assert parse_value('3.') == '3.'"),
            # Functional: booleans
            ("functional_test",
             "from blipshell.utils.config_parser import parse_value; "
             "assert parse_value('true') is True; "
             "assert parse_value('FALSE') is False; "
             "assert parse_value('True') is True; "
             "assert parse_value('tRuE') == 'tRuE'"),
            # Functional: quoted strings
            ("functional_test",
             "from blipshell.utils.config_parser import parse_value; "
             "assert parse_value('\"hello world\"') == 'hello world'; "
             "assert parse_value(\"'single'\") == 'single'; "
             "assert parse_value('\"mismatched') == '\"mismatched'"),
            # Functional: env vars
            ("functional_test",
             "import os; "
             "from blipshell.utils.config_parser import parse_value; "
             "assert parse_value('${PATH}') == os.environ['PATH']; "
             "try: parse_value('${NONEXISTENT_BENCHMARK_XYZ_123}'); assert False\n"
             "except KeyError: pass; "
             "assert parse_value('prefix${PATH}') == 'prefix${PATH}'"),
            # Functional: whitespace and empty
            ("functional_test",
             "from blipshell.utils.config_parser import parse_value; "
             "assert parse_value('') == ''; "
             "assert parse_value('   ') == ''; "
             "assert parse_value('  42  ') == 42"),
        ],
    },
    {
        "name": "cross_file_cli_feature",
        "description": "Add feature spanning two existing files (codebase navigation + integration)",
        "request": (
            "Add a `/recent` CLI command that shows the N most recently created memories "
            "(default 5). This requires changes to TWO existing files:\n\n"
            "1. blipshell/memory/sqlite_store.py — add an async method:\n"
            "   get_recent_memories(n: int = 5) -> list[dict]\n"
            "   Returns dicts with keys: id, summary, memory_type, rank, created_at\n"
            "   Ordered by created_at DESC, only non-archived memories.\n\n"
            "2. blipshell/ui/cli.py — add a /recent command handler:\n"
            "   - Follow the EXACT pattern of existing slash commands (look at how /core, "
            "   /flow, /health, etc. are registered and dispatched)\n"
            "   - Accept optional number argument: /recent or /recent 10\n"
            "   - Display results in a Rich table with columns: ID, Type, Rank, Summary (truncated), Date\n\n"
            "You MUST read both files first to understand the patterns. Do not guess — "
            "the CLI has a specific command dispatch pattern you must follow."
        ),
        "verify_checks": [
            # Method added to sqlite_store
            ("diff_contains", ("blipshell/memory/sqlite_store.py", r"\+.*async def get_recent_memories")),
            ("diff_contains", ("blipshell/memory/sqlite_store.py", r"\+.*(SELECT|ORDER BY)")),
            ("syntax_check", "blipshell/memory/sqlite_store.py"),
            # Command added to cli.py
            ("diff_contains", ("blipshell/ui/cli.py", r"\+.*recent")),
            ("syntax_check", "blipshell/ui/cli.py"),
            # Both files changed
            ("files_changed", 2),
            # Functional: method exists and returns a list
            ("functional_test",
             "import asyncio; "
             "from blipshell.memory.sqlite_store import SQLiteStore; "
             "async def check(): "
             "  s = SQLiteStore(':memory:'); await s.initialize(); "
             "  result = await s.get_recent_memories(5); "
             "  assert isinstance(result, list); "
             "asyncio.run(check())"),
            # No junk files
            ("no_unwanted_files", None),
        ],
    },
    {
        "name": "inject_and_diagnose_bug",
        "description": "Find and fix an injected bug from symptoms only (diagnosis)",
        "request": (
            "There's a bug in blipshell/core/tools/filesystem.py — the edit_file tool. "
            "Users report that edits silently fail: the tool returns success but the file "
            "content doesn't actually change. The bug was introduced recently.\n\n"
            "Your task:\n"
            "1. Read the edit_file tool's execute method in filesystem.py\n"
            "2. Find the bug (hint: look at the write-back logic)\n"
            "3. Fix it\n"
            "4. Write a test in tests/test_edit_bug.py that:\n"
            "   - Creates a temp file with known content\n"
            "   - Calls edit_file to change some text\n"
            "   - Reads the file back and asserts the change was applied\n"
            "   - Also tests that edit_file returns a success message"
        ),
        # This task requires a sandbox setup hook to inject the bug
        "setup_hook": "inject_edit_bug",
        "verify_checks": [
            # Model changed filesystem.py (fixed the bug)
            ("diff_contains", ("blipshell/core/tools/filesystem.py", r"\+.*write|open")),
            ("syntax_check", "blipshell/core/tools/filesystem.py"),
            # Test file exists and passes
            ("file_exists", "tests/test_edit_bug.py"),
            ("syntax_check", "tests/test_edit_bug.py"),
            ("grep_in_sandbox", ("tests/test_edit_bug.py", r"assert")),
            ("pytest_in_sandbox", "tests/test_edit_bug.py"),
            # Functional: edit_file actually works after the fix
            ("functional_test",
             "import asyncio, tempfile, os; "
             "from blipshell.core.tools.filesystem import EditFileTool; "
             "async def check(): "
             "  t = EditFileTool(); "
             "  f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False); "
             "  f.write('hello world'); f.close(); "
             "  t._root_path = os.path.dirname(f.name); "
             "  from blipshell.core.tools.base import ToolCall; "
             "  tc = ToolCall(id='t1', name='edit_file', arguments={'path': f.name, 'old_text': 'hello', 'new_text': 'goodbye'}); "
             "  r = await t.execute(tc); "
             "  content = open(f.name).read(); "
             "  os.unlink(f.name); "
             "  assert 'goodbye' in content, f'Edit did not apply: {content}'; "
             "asyncio.run(check())"),
            # No junk
            ("no_unwanted_files", None),
        ],
    },
    {
        "name": "multi_file_feature_integration",
        "description": "Add feature spanning 4 interdependent files (config + tool + registration + CLI)",
        "request": (
            "Add a 'session notes' feature to BlipShell — a simple scratchpad where the "
            "user can save and retrieve named notes within a session. This requires "
            "coordinated changes to FOUR existing files:\n\n"

            "1. **blipshell/models/config.py** — Add a `NotesConfig` class (Pydantic BaseModel) with:\n"
            "   - `enabled: bool = True`\n"
            "   - `max_notes: int = 50` (max notes per session)\n"
            "   - `max_note_length: int = 2000` (max chars per note)\n"
            "   Add a `notes: NotesConfig = NotesConfig()` field to `BlipShellConfig`.\n"
            "   You MUST read the file first to understand the existing config class patterns.\n\n"

            "2. **blipshell/core/tools/note_tools.py** — Create a NEW file with two tools:\n"
            "   a) `SaveNoteTool(Tool)` — saves a named note\n"
            "      - Parameters: `name` (string, required), `content` (string, required)\n"
            "      - Stores notes in a dict `self._notes: dict[str, str]`\n"
            "      - Validates against `max_notes` and `max_note_length` from config\n"
            "      - Returns confirmation message with note name\n"
            "   b) `GetNoteTool(Tool)` — retrieves a note by name\n"
            "      - Parameters: `name` (string, required)\n"
            "      - Returns the note content, or an error message if not found\n"
            "      - Mark as `read_only = True`\n"
            "   Both tools must follow the EXACT Tool base class pattern — read "
            "   blipshell/core/tools/base.py and another tool file (e.g. memory_tools.py) "
            "   to understand the ToolDefinition, ToolParameter, ToolParameterType imports "
            "   and the execute() method signature.\n\n"

            "3. **blipshell/core/agent_tools.py** — Register both note tools in `_register_tools()`.\n"
            "   You MUST read this file to see how other tools are registered:\n"
            "   - Tools are instantiated with config values passed to constructors\n"
            "   - Registered with `self.tool_registry.register(tool, group='group_name')`\n"
            "   - Pass `self.config.notes.max_notes` and `self.config.notes.max_note_length` "
            "   to the tool constructors\n"
            "   - Use group name 'notes'\n\n"

            "4. **blipshell/ui/cli.py** — Add a `/notes` slash command:\n"
            "   - `/notes` with no args — list all saved notes (name + first 50 chars preview)\n"
            "   - `/notes get <name>` — show full content of a note\n"
            "   - `/notes clear` — clear all notes\n"
            "   You MUST read cli.py to find the slash command dispatch section and follow "
            "   the EXACT elif pattern used by other commands like /health, /flow, etc.\n"
            "   The notes dict lives on the tool instances — access it via:\n"
            "   `agent.tool_registry.get_tool('save_note')._notes`\n\n"

            "CRITICAL: All four files must be consistent:\n"
            "- Config field names must match what agent_tools.py passes to constructors\n"
            "- Tool names in definitions must match what cli.py looks up via get_tool()\n"
            "- Parameter names in ToolDefinition must match execute() method signatures\n"
            "- All existing code in modified files must continue to work"
        ),
        "verify_checks": [
            # --- Config layer ---
            ("diff_contains", ("blipshell/models/config.py", r"\+.*class NotesConfig")),
            ("diff_contains", ("blipshell/models/config.py", r"\+.*max_notes.*int.*=.*50")),
            ("diff_contains", ("blipshell/models/config.py", r"\+.*notes.*:.*NotesConfig")),
            ("syntax_check", "blipshell/models/config.py"),

            # --- Tool layer ---
            ("file_exists", "blipshell/core/tools/note_tools.py"),
            ("syntax_check", "blipshell/core/tools/note_tools.py"),
            ("grep_in_sandbox", ("blipshell/core/tools/note_tools.py", r"class SaveNoteTool")),
            ("grep_in_sandbox", ("blipshell/core/tools/note_tools.py", r"class GetNoteTool")),
            ("grep_in_sandbox", ("blipshell/core/tools/note_tools.py", r"read_only.*=.*True")),
            # Uses proper base class imports
            ("grep_in_sandbox", ("blipshell/core/tools/note_tools.py",
                                 r"from.*tools\.(base|tools).*import.*(Tool|ToolDefinition)")),

            # --- Registration layer ---
            ("diff_contains", ("blipshell/core/agent_tools.py", r"\+.*SaveNoteTool|save_note|note_tools")),
            ("diff_contains", ("blipshell/core/agent_tools.py", r"\+.*register.*group.*notes")),
            ("syntax_check", "blipshell/core/agent_tools.py"),

            # --- CLI layer ---
            ("diff_contains", ("blipshell/ui/cli.py", r"\+.*notes")),
            ("syntax_check", "blipshell/ui/cli.py"),

            # --- Integration: tools actually work ---
            ("functional_test",
             "from blipshell.core.tools.note_tools import SaveNoteTool, GetNoteTool; "
             "import asyncio; "
             "async def check(): "
             "  save = SaveNoteTool(max_notes=10, max_note_length=100); "
             "  get = GetNoteTool(save._notes); "
             "  r = await save.execute(name='todo', content='buy milk'); "
             "  assert 'todo' in r.lower(); "
             "  r2 = await get.execute(name='todo'); "
             "  assert 'buy milk' in r2; "
             "  r3 = await get.execute(name='nonexistent'); "
             "  assert 'not found' in r3.lower() or 'error' in r3.lower(); "
             "asyncio.run(check())"),

            # --- Integration: max_notes limit enforced ---
            ("functional_test",
             "from blipshell.core.tools.note_tools import SaveNoteTool; "
             "import asyncio; "
             "async def check(): "
             "  save = SaveNoteTool(max_notes=2, max_note_length=100); "
             "  await save.execute(name='a', content='one'); "
             "  await save.execute(name='b', content='two'); "
             "  r = await save.execute(name='c', content='three'); "
             "  assert 'limit' in r.lower() or 'max' in r.lower() or 'full' in r.lower(); "
             "asyncio.run(check())"),

            # --- 4 files changed ---
            ("files_changed", 4),
            # No junk files
            ("no_unwanted_files", None),
        ],
    },
]


# ---------------------------------------------------------------------------
# Expert tasks — multi-step reasoning, refactoring, architecture, debugging
# These separate 84% models from 90%+ models.
# ---------------------------------------------------------------------------

EXPERT_TASKS = [
    {
        "name": "refactor_preserve_behavior",
        "description": "Refactor class into two classes while preserving all existing behavior",
        "request": (
            "The `RateTracker` concept doesn't exist yet. Create it, then immediately "
            "refactor it. Specifically:\n\n"
            "STEP 1: Create blipshell/core/rate_tracker.py with a class `RateTracker` "
            "that has these methods:\n"
            "  - __init__(self, window_seconds=60, max_calls=100)\n"
            "  - record_call(self, endpoint: str) → None — records a call for an endpoint\n"
            "  - get_rate(self, endpoint: str) → int — calls in window for that endpoint\n"
            "  - is_limited(self, endpoint: str) → bool — True if rate >= max_calls\n"
            "  - get_all_rates(self) → dict[str, int] — rates for all endpoints\n"
            "  - reset(self, endpoint: str | None = None) → None — reset one or all\n"
            "Use time.monotonic() and store per-endpoint call lists.\n\n"
            "STEP 2: Write tests/test_rate_tracker.py that tests:\n"
            "  - Recording calls increments rate for that endpoint only\n"
            "  - Different endpoints track independently\n"
            "  - is_limited returns True at max_calls\n"
            "  - get_all_rates shows all tracked endpoints\n"
            "  - reset(endpoint) clears only that endpoint\n"
            "  - reset() with no args clears everything\n\n"
            "STEP 3: Now refactor — split RateTracker into TWO classes:\n"
            "  a) `EndpointCounter` — tracks calls for a SINGLE endpoint\n"
            "     (window_seconds, max_calls, record_call, get_rate, is_limited, reset)\n"
            "  b) `RateTracker` — orchestrates multiple EndpointCounters\n"
            "     (same public API as before, but delegates to EndpointCounter instances)\n"
            "  The refactored RateTracker must pass ALL the same tests unchanged.\n\n"
            "STEP 4: Run the tests to confirm they still pass after refactoring.\n\n"
            "CRITICAL: Do NOT modify the tests during refactoring. The whole point is "
            "that the public API stays the same."
        ),
        "verify_checks": [
            ("file_exists", "blipshell/core/rate_tracker.py"),
            ("syntax_check", "blipshell/core/rate_tracker.py"),
            # Both classes exist after refactor
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"class EndpointCounter")),
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"class RateTracker")),
            # EndpointCounter has core methods
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"def record_call")),
            ("grep_in_sandbox", ("blipshell/core/rate_tracker.py", r"def is_limited")),
            # Tests exist and pass
            ("file_exists", "tests/test_rate_tracker.py"),
            ("syntax_check", "tests/test_rate_tracker.py"),
            ("pytest_in_sandbox", "tests/test_rate_tracker.py"),
            # Functional: refactored API still works
            ("functional_test",
             "from blipshell.core.rate_tracker import RateTracker, EndpointCounter; "
             "rt = RateTracker(window_seconds=60, max_calls=2); "
             "rt.record_call('ep1'); rt.record_call('ep1'); "
             "assert rt.is_limited('ep1'); "
             "assert not rt.is_limited('ep2'); "
             "assert rt.get_all_rates() == {'ep1': 2}; "
             "rt.reset('ep1'); assert rt.get_rate('ep1') == 0"),
            # Functional: EndpointCounter works standalone
            ("functional_test",
             "from blipshell.core.rate_tracker import EndpointCounter; "
             "ec = EndpointCounter(window_seconds=60, max_calls=3); "
             "ec.record_call(); ec.record_call(); "
             "assert ec.get_rate() == 2; "
             "assert not ec.is_limited(); "
             "ec.record_call(); assert ec.is_limited()"),
            ("no_unwanted_files", None),
        ],
    },
    {
        "name": "debug_from_traceback",
        "description": "Fix a real bug given only the error traceback (no hints about location)",
        "request": (
            "A user reported this error when running BlipShell:\n\n"
            "```\n"
            "Traceback (most recent call last):\n"
            "  File \"blipshell/memory/search.py\", line 142, in search\n"
            "    boosted = self._apply_boosts(results, query)\n"
            "  File \"blipshell/memory/search.py\", line 198, in _apply_boosts\n"
            "    for r in results:\n"
            "  File \"blipshell/memory/search.py\", line 203, in _apply_boosts\n"
            "    tag_overlap = len(set(r['tags']) & query_tags)\n"
            "TypeError: unhashable type: 'list'\n"
            "```\n\n"
            "The tags field is sometimes a JSON string (like '[\"python\", \"coding\"]') "
            "instead of a parsed list. When tags is already a list of strings, set() works. "
            "But when a tag itself is a list (malformed data), set() fails.\n\n"
            "Your task:\n"
            "1. Read search.py to find the _apply_boosts method and understand the code\n"
            "2. Fix it to handle both cases:\n"
            "   - If r['tags'] is a string, parse it as JSON first\n"
            "   - If any individual tag is not a string, skip it\n"
            "   - Never crash — gracefully handle malformed tags\n"
            "3. Write tests/test_search_boost_fix.py that verifies:\n"
            "   - Normal list tags work: ['python', 'coding']\n"
            "   - JSON string tags work: '[\"python\", \"coding\"]'\n"
            "   - Nested list tags don't crash: [['python'], 'coding']\n"
            "   - Empty/None tags don't crash\n\n"
            "IMPORTANT: Only fix the specific bug. Do NOT refactor surrounding code."
        ),
        "setup_hook": "inject_tag_bug",
        "verify_checks": [
            # Fixed the bug in search.py
            ("diff_contains", ("blipshell/memory/search.py", r"\+.*(json\.loads|isinstance|str)")),
            ("syntax_check", "blipshell/memory/search.py"),
            # Test exists and passes
            ("file_exists", "tests/test_search_boost_fix.py"),
            ("syntax_check", "tests/test_search_boost_fix.py"),
            ("pytest_in_sandbox", "tests/test_search_boost_fix.py"),
            # Didn't over-modify (fewer than 20 lines changed in search.py)
            ("max_diff_lines", ("blipshell/memory/search.py", 20)),
            ("no_unwanted_files", None),
        ],
    },
    {
        "name": "async_retry_decorator",
        "description": "Implement a reusable async retry decorator with tests (design + correctness)",
        "request": (
            "Create blipshell/utils/retry.py with an async retry decorator:\n\n"
            "```python\n"
            "def async_retry(\n"
            "    max_retries: int = 3,\n"
            "    base_delay: float = 1.0,\n"
            "    max_delay: float = 60.0,\n"
            "    backoff_factor: float = 2.0,\n"
            "    retry_on: tuple[type[Exception], ...] = (Exception,),\n"
            ") -> Callable:\n"
            "```\n\n"
            "Requirements:\n"
            "- Exponential backoff: delay = min(base_delay * backoff_factor^attempt, max_delay)\n"
            "- Only retry exceptions that are instances of retry_on types\n"
            "- Non-matching exceptions propagate immediately\n"
            "- After max_retries exhausted, raise the last exception\n"
            "- The decorator must preserve the function's name and docstring (use functools.wraps)\n"
            "- Must work with async functions only (not sync)\n"
            "- Use asyncio.sleep for delays\n\n"
            "Also create blipshell/utils/__init__.py if it doesn't exist.\n\n"
            "Write tests/test_retry.py:\n"
            "- Function succeeds on first try → no retries, returns result\n"
            "- Function fails twice then succeeds → retries and returns result\n"
            "- Function always fails → raises after max_retries\n"
            "- retry_on filter works: ValueError retried, TypeError not retried\n"
            "- Decorator preserves __name__ and __doc__\n"
            "- Backoff delays increase correctly (mock asyncio.sleep to capture delay values)\n"
            "Use pytest and pytest-asyncio (or asyncio.run) for async tests."
        ),
        "verify_checks": [
            ("file_exists", "blipshell/utils/retry.py"),
            ("syntax_check", "blipshell/utils/retry.py"),
            ("grep_in_sandbox", ("blipshell/utils/retry.py", r"def async_retry")),
            ("grep_in_sandbox", ("blipshell/utils/retry.py", r"functools\.wraps")),
            ("grep_in_sandbox", ("blipshell/utils/retry.py", r"asyncio\.sleep")),
            ("file_exists", "tests/test_retry.py"),
            ("syntax_check", "tests/test_retry.py"),
            ("pytest_in_sandbox", "tests/test_retry.py"),
            # Functional: basic retry works
            ("functional_test",
             "import asyncio; "
             "from blipshell.utils.retry import async_retry; "
             "call_count = 0; "
             "@async_retry(max_retries=3, base_delay=0.01) \n"
             "async def flaky(): \n"
             "  global call_count; call_count += 1; \n"
             "  if call_count < 3: raise ValueError('not yet'); \n"
             "  return 'ok'; \n"
             "assert asyncio.run(flaky()) == 'ok'; "
             "assert call_count == 3"),
            # Functional: non-matching exception propagates immediately
            ("functional_test",
             "import asyncio; "
             "from blipshell.utils.retry import async_retry; "
             "@async_retry(max_retries=3, base_delay=0.01, retry_on=(ValueError,)) \n"
             "async def wrong_error(): raise TypeError('nope'); \n"
             "try: asyncio.run(wrong_error()); assert False \n"
             "except TypeError: pass"),
            # Functional: preserves function metadata
            ("functional_test",
             "from blipshell.utils.retry import async_retry; "
             "@async_retry() \n"
             "async def my_func(): \n"
             "  '''My docstring'''; \n"
             "  pass; \n"
             "assert my_func.__name__ == 'my_func'; "
             "assert 'My docstring' in (my_func.__doc__ or '')"),
            ("no_unwanted_files", None),
        ],
    },
    {
        "name": "read_existing_and_extend",
        "description": "Read a complex existing file, understand its patterns, and extend it correctly",
        "request": (
            "Add a new tool `SearchMemoryTool` to the existing tool system. This requires "
            "you to deeply understand the existing patterns before writing code.\n\n"
            "1. Read blipshell/core/tools/base.py to understand: Tool base class, "
            "   ToolDefinition, ToolParameter, ToolParameterType, ToolCall, and the "
            "   execute() method signature.\n"
            "2. Read one existing tool file (e.g., blipshell/core/tools/memory_tools.py) "
            "   to see a real implementation pattern.\n"
            "3. Create blipshell/core/tools/search_tool.py with `SearchMemoryTool(Tool)`:\n"
            "   - Parameters: query (string, required), limit (integer, optional, default 5), "
            "     memory_type (string, optional, one of: fact/preference/skill/event/conversation)\n"
            "   - read_only = True\n"
            "   - execute() should return a formatted string listing matching memories\n"
            "   - Since we can't connect to a real DB in the benchmark, the execute method "
            "     should accept an optional `search_fn` callback in __init__ that takes "
            "     (query, limit, memory_type) and returns list[dict]. If no callback, "
            "     return 'Search not available (no backend connected)'.\n"
            "4. Write tests/test_search_tool.py:\n"
            "   - Test that tool definition has correct parameters and types\n"
            "   - Test execute with a mock search_fn that returns sample results\n"
            "   - Test execute with no search_fn returns the fallback message\n"
            "   - Test that memory_type parameter is optional\n"
            "   - Test that limit defaults to 5 in the definition\n\n"
            "CRITICAL: Your ToolDefinition, ToolParameter usage, and execute() signature "
            "must EXACTLY match the patterns in the existing code. Don't invent your own."
        ),
        "verify_checks": [
            ("file_exists", "blipshell/core/tools/search_tool.py"),
            ("syntax_check", "blipshell/core/tools/search_tool.py"),
            ("grep_in_sandbox", ("blipshell/core/tools/search_tool.py", r"class SearchMemoryTool")),
            ("grep_in_sandbox", ("blipshell/core/tools/search_tool.py", r"read_only.*=.*True")),
            # Uses proper imports from base
            ("grep_in_sandbox", ("blipshell/core/tools/search_tool.py",
                                 r"from.*tools\.base.*import")),
            ("file_exists", "tests/test_search_tool.py"),
            ("syntax_check", "tests/test_search_tool.py"),
            ("pytest_in_sandbox", "tests/test_search_tool.py"),
            # Functional: tool definition is correct
            ("functional_test",
             "from blipshell.core.tools.search_tool import SearchMemoryTool; "
             "t = SearchMemoryTool(); "
             "d = t.definition(); "
             "names = [p.name for p in d.parameters]; "
             "assert 'query' in names; "
             "assert 'limit' in names; "
             "assert 'memory_type' in names"),
            # Functional: execute with mock callback
            ("functional_test",
             "import asyncio; "
             "from blipshell.core.tools.search_tool import SearchMemoryTool; "
             "from blipshell.core.tools.base import ToolCall; "
             "async def mock_search(q, limit=5, memory_type=None): "
             "  return [{'summary': 'test memory', 'rank': 3}]; "
             "t = SearchMemoryTool(search_fn=mock_search); "
             "tc = ToolCall(id='t1', name='search_memory', arguments={'query': 'test'}); "
             "r = asyncio.run(t.execute(tc)); "
             "assert 'test memory' in r"),
            # Functional: execute without callback
            ("functional_test",
             "import asyncio; "
             "from blipshell.core.tools.search_tool import SearchMemoryTool; "
             "from blipshell.core.tools.base import ToolCall; "
             "t = SearchMemoryTool(); "
             "tc = ToolCall(id='t1', name='search_memory', arguments={'query': 'test'}); "
             "r = asyncio.run(t.execute(tc)); "
             "assert 'not available' in r.lower() or 'no backend' in r.lower()"),
            ("no_unwanted_files", None),
        ],
    },
    {
        "name": "performance_optimization",
        "description": "Identify and fix a performance problem (profiling mindset)",
        "request": (
            "Create blipshell/utils/text_index.py with a `TextIndex` class for fast "
            "substring search across many documents:\n\n"
            "```python\n"
            "class TextIndex:\n"
            "    def __init__(self):\n"
            "        '''Initialize empty index.'''\n"
            "    \n"
            "    def add(self, doc_id: str, text: str) -> None:\n"
            "        '''Add a document to the index.'''\n"
            "    \n"
            "    def search(self, query: str) -> list[str]:\n"
            "        '''Return doc_ids containing query as substring (case-insensitive).'''\n"
            "    \n"
            "    def remove(self, doc_id: str) -> None:\n"
            "        '''Remove a document from the index.'''\n"
            "    \n"
            "    def __len__(self) -> int:\n"
            "        '''Number of indexed documents.'''\n"
            "```\n\n"
            "NAIVE approach would scan all documents on every search. Instead, implement "
            "a trigram index:\n"
            "- On add(): extract all 3-character subsequences (trigrams) from lowercased text\n"
            "- Store a mapping: trigram → set of doc_ids\n"
            "- On search(): extract trigrams from query, intersect their doc_id sets, "
            "  then verify with actual substring check (trigram index can have false positives)\n"
            "- On remove(): clean up trigram entries\n\n"
            "Also create blipshell/utils/__init__.py if needed.\n\n"
            "Write tests/test_text_index.py:\n"
            "- Basic add + search works\n"
            "- Case-insensitive search\n"
            "- Remove actually removes from results\n"
            "- Empty query returns empty list\n"
            "- Short queries (< 3 chars) fall back to brute-force scan\n"
            "- Performance test: add 1000 documents, search should complete in < 0.1s\n"
            "  (import time, measure wall clock — this verifies the trigram index is used)\n"
            "- __len__ returns correct count"
        ),
        "verify_checks": [
            ("file_exists", "blipshell/utils/text_index.py"),
            ("syntax_check", "blipshell/utils/text_index.py"),
            ("grep_in_sandbox", ("blipshell/utils/text_index.py", r"class TextIndex")),
            # Uses trigram approach (not just brute force)
            ("grep_in_sandbox", ("blipshell/utils/text_index.py", r"trigram|ngram|3")),
            ("file_exists", "tests/test_text_index.py"),
            ("syntax_check", "tests/test_text_index.py"),
            ("pytest_in_sandbox", "tests/test_text_index.py"),
            # Functional: basic operations
            ("functional_test",
             "from blipshell.utils.text_index import TextIndex; "
             "idx = TextIndex(); "
             "idx.add('d1', 'Hello World Python'); "
             "idx.add('d2', 'Goodbye World Java'); "
             "idx.add('d3', 'Hello Python Again'); "
             "r = idx.search('python'); "
             "assert 'd1' in r; assert 'd3' in r; assert 'd2' not in r; "
             "assert len(idx) == 3"),
            # Functional: case insensitive
            ("functional_test",
             "from blipshell.utils.text_index import TextIndex; "
             "idx = TextIndex(); "
             "idx.add('d1', 'Hello WORLD'); "
             "assert 'd1' in idx.search('world'); "
             "assert 'd1' in idx.search('HELLO')"),
            # Functional: remove works
            ("functional_test",
             "from blipshell.utils.text_index import TextIndex; "
             "idx = TextIndex(); "
             "idx.add('d1', 'test document'); "
             "idx.remove('d1'); "
             "assert 'd1' not in idx.search('test'); "
             "assert len(idx) == 0"),
            # Functional: performance — 1000 docs, search under 0.1s
            ("functional_test",
             "import time; "
             "from blipshell.utils.text_index import TextIndex; "
             "idx = TextIndex(); "
             "for i in range(1000): "
             "  idx.add(f'd{i}', f'document number {i} with some text about topic {i % 50}'); "
             "start = time.monotonic(); "
             "[idx.search('topic 25') for _ in range(100)]; "
             "elapsed = time.monotonic() - start; "
             "assert elapsed < 1.0, f'100 searches took {elapsed:.2f}s — too slow'"),
            ("no_unwanted_files", None),
        ],
    },
]


# Setup hook for the tag bug injection
def inject_tag_bug(sandbox_path: str):
    """Inject a tag-handling bug into search.py's _apply_boosts method.

    Makes tags sometimes appear as JSON strings or nested lists instead
    of clean string lists, then removes the defensive parsing.
    """
    search_path = os.path.join(sandbox_path, "blipshell", "memory", "search.py")
    if not os.path.exists(search_path):
        return
    with open(search_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find tag_overlap line and ensure it uses raw set() without defensive parsing
    # If there's already defensive code, strip it to create the bug
    import re as _re
    # Look for any json.loads or isinstance checks near tag_overlap and remove them
    # Then ensure the line does: tag_overlap = len(set(r['tags']) & query_tags)
    # which will crash on non-hashable tags

    # Simple approach: find _apply_boosts and inject a known-buggy version
    if "_apply_boosts" in content and "tag_overlap" in content:
        # Add a comment marking the bug location for verification
        content = content.replace(
            "tag_overlap = len(set(r.get('tags', []))",
            "tag_overlap = len(set(r['tags'])",
            1,
        )
        # Also try the other common pattern
        content = content.replace(
            "tags = r.get('tags', [])",
            "tags = r['tags']  # BUG: no defensive parsing",
            1,
        )
        with open(search_path, "w", encoding="utf-8") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# A/B System Configurations — test which BlipShell features help
# ---------------------------------------------------------------------------

SYSTEM_CONFIGS = {
    "full": {
        "description": "All systems ON (production defaults)",
        # No overrides — uses all defaults
    },
    "bare": {
        "description": "Minimal prompt, no scaffolding — raw LLM capability",
        "use_minimal_prompt": True,
        "enable_dedup": False,
        "enable_compaction": False,
        "disable_state_block": True,
        "disable_winddown": True,
    },
    "no_state": {
        "description": "No [STATE] block injection — can the model track its own state?",
        "disable_state_block": True,
    },
    "no_dedup": {
        "description": "No dedup — does the model waste calls re-reading files?",
        "enable_dedup": False,
    },
    "no_winddown": {
        "description": "No budget wind-down message — does the model run out gracefully?",
        "disable_winddown": True,
    },
    "guardrails": {
        "description": "Full + guardrails (trajectory monitor + completion audit)",
        "guardrails_enabled": True,
    },
    "no_scaffolding": {
        "description": "No state block, no wind-down, no compaction — just dedup and prompt",
        "disable_state_block": True,
        "disable_winddown": True,
        "enable_compaction": False,
    },
    "stale_detection": {
        "description": "Full + stale-read detection (already on by default — this is a control)",
        # Stale detection is wired into executor by default, no override needed.
        # This config exists to label it explicitly in A/B results.
    },
}


def build_minimal_system_prompt(sandbox_path: str, tool_limit: int = 15) -> str:
    """Bare-bones system prompt — no scaffolding rules, just the essential context."""
    return (
        "You are a coding assistant. Complete the task using the available tools.\n"
        f"Project root: {sandbox_path}\n"
        "Use relative paths. Call task_complete when done.\n"
        "Platform: Windows — use the file/grep/glob tools, not shell commands.\n"
    )


def inject_edit_bug(sandbox_path: str):
    """Inject a subtle bug into edit_file: comment out the write-back so edits silently fail.

    The actual code has `resolved.write_text(new_content, encoding="utf-8")` in multiple
    places (Layer 1 exact match, Layer 2 whitespace match, etc.). We break the FIRST
    occurrence so the most common edit path silently succeeds without writing.
    """
    fs_path = os.path.join(sandbox_path, "blipshell", "core", "tools", "filesystem.py")
    with open(fs_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace first write_text(new_content...) with a no-op
    # The actual line is: resolved.write_text(new_content, encoding="utf-8")
    content = content.replace(
        'resolved.write_text(new_content, encoding="utf-8")',
        '# BUG: write_text accidentally removed during refactor\n'
        '            _ = new_content  # noqa: F841 — computed but never written',
        1,  # only replace first occurrence
    )

    with open(fs_path, "w", encoding="utf-8") as f:
        f.write(content)


# Setup hooks — called before a task runs to modify the sandbox
SETUP_HOOKS = {
    "inject_edit_bug": inject_edit_bug,
    "inject_tag_bug": inject_tag_bug,
}


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


@dataclass
class TranscriptMetrics:
    """Behavioral metrics parsed from the raw transcript."""
    questions_asked: int = 0       # "should I" / "want me to" patterns
    linux_cmds_on_windows: int = 0  # shell calls using ls/cat/grep
    llm_word_count: int = 0        # total words in LLM responses (verbosity)


@dataclass
class TaskMetrics:
    """Metrics for a complete coding task (plan + all steps)."""
    task_name: str
    model: str
    plan_time: float = 0.0
    plan_steps: int = 0
    plan_text: str = ""
    steps: list[StepMetrics] = field(default_factory=list)
    summary_time: float = 0.0
    summary_text: str = ""
    total_time: float = 0.0
    error: str = ""

    # Verification results
    checks_passed: int = 0
    checks_total: int = 0
    check_details: list[tuple[str, bool, str]] = field(default_factory=list)
    pytest_passed: bool = False
    pytest_output: str = ""

    # Transcript behavioral metrics
    transcript_metrics: TranscriptMetrics = field(default_factory=TranscriptMetrics)

    # Transcript file path
    transcript_path: str = ""

    # Token usage and cost (OpenRouter / OpenAI-compatible)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_requests: int = 0
    estimated_cost_usd: float = 0.0

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
            "check_details": [
                {"check": label, "passed": ok, "reason": reason}
                for label, ok, reason in self.check_details
            ],
            "pytest_passed": self.pytest_passed,
            "pytest_output": self.pytest_output,
            "transcript_metrics": {
                "questions_asked": self.transcript_metrics.questions_asked,
                "linux_cmds_on_windows": self.transcript_metrics.linux_cmds_on_windows,
                "llm_word_count": self.transcript_metrics.llm_word_count,
            },
            "transcript_path": self.transcript_path,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_requests": self.api_requests,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
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
# Transcript capture
# ---------------------------------------------------------------------------

class TranscriptCapture:
    """Accumulates streaming output as a raw transcript.

    Acts as on_token callback for execute_plan(). Adds benchmark annotations
    (model name, timing, verification results) and saves to disk.
    """

    def __init__(self, model: str, task_name: str, output_dir: Path):
        self.model = model
        self.task_name = task_name
        self.output_dir = output_dir
        self._lines: list[str] = []
        self._current_step: int = 0

    def on_token(self, token: str):
        """Streaming token callback — accumulates raw output."""
        self._lines.append(token)

    def add_header(self, plan_text: str):
        """Add benchmark header at the top of transcript."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._lines.insert(0, (
            f"{'=' * 70}\n"
            f"BENCHMARK TRANSCRIPT\n"
            f"Model: {self.model}\n"
            f"Task: {self.task_name}\n"
            f"Timestamp: {ts}\n"
            f"{'=' * 70}\n\n"
            f"--- PLAN ---\n{plan_text}\n\n"
            f"--- EXECUTION ---\n"
        ))

    def add_step_header(self, step_num: int, total: int, description: str):
        """Mark step boundaries in the transcript."""
        self._current_step = step_num
        self._lines.append(
            f"\n{'─' * 50}\n"
            f"Step {step_num}/{total}: {description}\n"
            f"{'─' * 50}\n"
        )

    def add_step_footer(self, step_num: int, total: int, result_summary: str):
        """Mark step completion."""
        self._lines.append(f"\n  [OK] Step {step_num}/{total} complete: {result_summary}\n")

    def add_verification(self, checks_passed: int, checks_total: int):
        """Add verification results at the end."""
        self._lines.append(
            f"\n{'=' * 70}\n"
            f"VERIFICATION: {checks_passed}/{checks_total} checks passed\n"
            f"{'=' * 70}\n"
        )

    def add_timing(self, plan_time: float, total_time: float):
        """Add timing summary."""
        self._lines.append(
            f"\nTIMING: plan={plan_time:.1f}s, total={total_time:.1f}s\n"
        )

    def get_text(self) -> str:
        """Return the full transcript as a string."""
        return "".join(self._lines)

    def save(self) -> str:
        """Save transcript to disk. Returns the file path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        # Sanitize model name for filename
        safe_model = self.model.replace(":", "_").replace("/", "_")
        filename = f"{safe_model}__{self.task_name}__{ts}.txt"
        filepath = self.output_dir / filename
        filepath.write_text(self.get_text(), encoding="utf-8")
        return str(filepath)

    def parse_metrics(self) -> TranscriptMetrics:
        """Parse behavioral metrics from the raw transcript."""
        text = self.get_text()
        metrics = TranscriptMetrics()

        # Questions asked (bad in autonomous mode)
        question_patterns = re.compile(
            r"\b(should I|want me to|shall I|would you like|do you want)\b", re.IGNORECASE,
        )
        metrics.questions_asked = len(question_patterns.findall(text))

        # Linux commands on Windows (platform violation)
        linux_cmd_patterns = re.compile(
            r"\[Tool: run_command\].*?\b(ls\b|cat\b|grep\b|head\b|tail\b|find\b|wc\b)",
            re.IGNORECASE | re.DOTALL,
        )
        metrics.linux_cmds_on_windows = len(linux_cmd_patterns.findall(text))

        # LLM word count (verbosity)
        # Count words outside of tool call/result blocks
        metrics.llm_word_count = len(text.split())

        return metrics


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
# Tool registry creation (with git tools)
# ---------------------------------------------------------------------------

def create_tool_registry(sandbox_path: str) -> InstrumentedToolRegistry:
    """Create a tool registry pointing at the sandbox, including git tools."""
    registry = InstrumentedToolRegistry()

    # Filesystem tools
    registry.register(ReadFileTool(
        max_file_size=1048576,
        root_path=sandbox_path,
    ), group="filesystem")
    registry.register(WriteFileTool(
        root_path=sandbox_path,
    ), group="filesystem")
    registry.register(EditFileTool(root_path=sandbox_path), group="filesystem")
    registry.register(ListDirectoryTool(root_path=sandbox_path), group="filesystem")

    # Shell tool
    registry.register(ShellTool(
        timeout=30,
        allowed_commands=["python", "pip", "pytest", "type", "dir", "echo", "git"],
        cwd=sandbox_path,
    ), group="shell")

    # Code search tools
    registry.register(GrepTool(root_path=sandbox_path), group="coding")
    registry.register(GlobTool(root_path=sandbox_path), group="coding")

    # Git tools (matching Agent.activate_project)
    registry.register(GitStatusTool(root_path=sandbox_path), group="coding")
    registry.register(GitDiffTool(root_path=sandbox_path), group="coding")
    registry.register(GitAddTool(root_path=sandbox_path), group="coding")
    registry.register(GitCommitTool(root_path=sandbox_path), group="coding")

    return registry


# ---------------------------------------------------------------------------
# Router / planner / executor setup
# ---------------------------------------------------------------------------

def make_router(model_name: str, timeout: float = 300.0) -> LLMRouter:
    """Create an LLMRouter that routes CODING and TOOL_CALLING to the given model.

    If model_name contains '/' (e.g. 'google/gemini-2.5-flash'), routes through
    OpenRouter (requires OPENROUTER_API_KEY env var). Otherwise uses local Ollama.
    """
    models = ModelsConfig(
        reasoning=model_name,
        tool_calling=model_name,
        coding=model_name,
        summarization=model_name,
        ranking=model_name,
        importance=model_name,
        embedding=model_name,
    )

    if "/" in model_name:
        # OpenRouter — OpenAI-compatible API
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY env var required for OpenRouter models. "
                "Get one at https://openrouter.ai/keys"
            )
        # Debug: show key is loaded (mask middle)
        masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "TOO_SHORT"
        console.print(f"[dim]OpenRouter API key: {masked} ({len(api_key)} chars)[/dim]")
        # Strip any accidental quotes/whitespace from the key
        api_key = api_key.strip().strip('"').strip("'")
        # Per-model pricing ($/million tokens) — from OpenRouter pricing pages
        _OPENROUTER_PRICING = {
            "anthropic/claude-sonnet":   (3.0, 15.0),
            "anthropic/claude-haiku":    (0.80, 4.0),
            "google/gemini-2.5-flash":   (0.15, 0.60),
            "google/gemini-2.5-pro":     (1.25, 10.0),
            "z-ai/glm-5":               (0.80, 2.56),
            "minimax/minimax-m2.5":      (0.27, 0.95),
            # m3: promo (50% off) is 0.30/1.20; regular list price is 0.60/2.40.
            "minimax/minimax-m3":        (0.30, 1.20),
            "moonshotai/kimi-k2.5":      (0.45, 2.20),
            "qwen/qwen3-coder-next":     (0.12, 0.75),
            "mistralai/devstral-2512":   (0.40, 2.00),
            "z-ai/glm-4.7-flash":        (0.06, 0.40),
        }
        cost_prompt, cost_completion = 0.0, 0.0
        for prefix, (cp, cc) in _OPENROUTER_PRICING.items():
            if model_name.startswith(prefix):
                cost_prompt, cost_completion = cp, cc
                break
        endpoint_cfg = EndpointConfig(
            name="openrouter",
            url="https://openrouter.ai/api/v1",
            provider="openai",
            api_key=api_key,
            roles=["reasoning", "tool_calling", "coding", "summarization",
                   "ranking", "importance", "embedding"],
            priority=1,
            max_concurrent=3,
            context_tokens=200000,
            cost_per_1m_prompt=cost_prompt,
            cost_per_1m_completion=cost_completion,
        )
    else:
        # Local Ollama
        endpoint_cfg = EndpointConfig(
            name="benchmark",
            url=OLLAMA_URL,
            roles=["reasoning", "tool_calling", "coding", "summarization",
                   "ranking", "importance", "embedding"],
            priority=1,
            max_concurrent=2,
            context_tokens=131072,
        )

    # No retries in benchmark — one timeout = fail, move on
    llm_config = LLMConfig(timeout=timeout, max_retries=0)
    endpoint_manager = EndpointManager([endpoint_cfg], llm_config)
    return LLMRouter(models, endpoint_manager)


def get_model_settings(model_name: str) -> ModelSettings:
    """Get model settings for a benchmark model, matching production config."""
    registry = ModelSettingsRegistry()
    registry.load(_MODEL_SETTINGS_CONFIG)
    return registry.get(model_name)


async def create_sqlite(sandbox_path: str) -> SQLiteStore:
    """Create a temporary SQLiteStore for plan/step persistence."""
    db_path = os.path.join(sandbox_path, "_benchmark.db")
    store = SQLiteStore(db_path)
    await store.initialize()
    return store


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

async def run_verification(sandbox_path: str, task: dict) -> tuple[int, int, list[tuple[str, bool, str]]]:
    """Run verification checks on the sandbox after task completion.

    Returns (checks_passed, checks_total, check_details).
    check_details is a list of (check_label, passed, reason) for logging.
    """
    checks = task.get("verify_checks", [])
    passed = 0
    total = len(checks)
    details: list[tuple[str, bool, str]] = []

    for check_type, target in checks:
        label = f"{check_type}({target})" if target else check_type
        check_passed = False
        reason = ""

        try:
            if check_type == "grep_in_sandbox":
                # (path_prefix, regex_pattern) — search for pattern in .py files under path
                path_prefix, pattern = target
                search_path = os.path.join(sandbox_path, path_prefix)
                found = False
                regex = re.compile(pattern)
                # If path points to a specific file, only search that file (don't fall back to parent dir)
                if os.path.isfile(search_path):
                    content = Path(search_path).read_text(encoding="utf-8", errors="replace")
                    found = bool(regex.search(content))
                elif os.path.isdir(search_path):
                    for dirpath, dirs, files in os.walk(search_path):
                        for fname in files:
                            if not fname.endswith(".py"):
                                continue
                            fpath = os.path.join(dirpath, fname)
                            try:
                                content = Path(fpath).read_text(encoding="utf-8", errors="replace")
                                if regex.search(content):
                                    found = True
                                    break
                            except Exception:
                                continue
                        if found:
                            break
                # else: path doesn't exist → found stays False
                check_passed = found
                reason = "pattern found" if found else "pattern not found"

            elif check_type == "file_exists":
                full_path = os.path.join(sandbox_path, target)
                exists = os.path.isfile(full_path)
                check_passed = exists
                reason = "exists" if exists else "not found"

            elif check_type == "file_contains":
                filepath, needle = target
                full_path = os.path.join(sandbox_path, filepath)
                if os.path.isfile(full_path):
                    content = Path(full_path).read_text(encoding="utf-8", errors="replace")
                    if needle in content:
                        check_passed = True
                        reason = "needle found"
                    else:
                        reason = "needle not in file"
                else:
                    reason = "file not found"

            elif check_type == "no_unwanted_files":
                proc = await asyncio.create_subprocess_exec(
                    "git", "status", "--porcelain",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_path,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                output = stdout.decode("utf-8", errors="replace")
                unwanted = []
                for line in output.splitlines():
                    if len(line) < 4:
                        continue
                    fpath = line[3:].strip()
                    if fpath.lower().endswith((".md", ".rst")) and "test" not in fpath.lower():
                        unwanted.append(fpath)
                check_passed = not unwanted
                reason = "clean" if not unwanted else f"unwanted: {unwanted}"

            elif check_type == "diff_contains":
                # (path_prefix, regex_pattern) — check that git diff output contains pattern
                # This proves the model actually changed something (can't match pre-existing code)
                diff_path, pattern = target
                proc = await asyncio.create_subprocess_exec(
                    "git", "diff", "--", diff_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_path,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                diff_output = stdout.decode("utf-8", errors="replace")
                # Also check untracked files (new files won't show in git diff)
                if not diff_output.strip():
                    proc2 = await asyncio.create_subprocess_exec(
                        "git", "diff", "--cached", "--", diff_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=sandbox_path,
                    )
                    stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=10)
                    diff_output = stdout2.decode("utf-8", errors="replace")
                # For new untracked files, generate a pseudo-diff.
                # ONLY for files that git status shows as untracked (??) or newly added (A).
                # Do NOT generate pseudo-diff for committed files with no changes.
                if not diff_output.strip():
                    proc3 = await asyncio.create_subprocess_exec(
                        "git", "status", "--porcelain", "--", diff_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=sandbox_path,
                    )
                    stdout3, _ = await asyncio.wait_for(proc3.communicate(), timeout=10)
                    status_output = stdout3.decode("utf-8", errors="replace")
                    diff_lines = []
                    for line in status_output.splitlines():
                        if line.startswith("?") or line.startswith("A"):
                            fpath = line[3:].strip()
                            fp = os.path.join(sandbox_path, fpath)
                            if os.path.isfile(fp):
                                try:
                                    fc = Path(fp).read_text(encoding="utf-8", errors="replace")
                                    diff_lines.extend(f"+{l}" for l in fc.splitlines())
                                except Exception:
                                    pass
                    diff_output = "\n".join(diff_lines)

                regex = re.compile(pattern)
                found = bool(regex.search(diff_output))
                check_passed = found
                reason = "pattern in diff" if found else f"pattern not in diff ({len(diff_output)} chars)"

            elif check_type == "syntax_check":
                # Run python -m py_compile on the file
                full_path = os.path.join(sandbox_path, target)
                if not os.path.isfile(full_path):
                    reason = "file not found"
                else:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "py_compile", full_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=sandbox_path,
                    )
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
                    if proc.returncode == 0:
                        check_passed = True
                        reason = "compiles OK"
                    else:
                        reason = f"syntax error: {stderr.decode('utf-8', errors='replace')[:200]}"

            elif check_type == "pytest_in_sandbox":
                # Run pytest on a specific test file
                full_path = os.path.join(sandbox_path, target)
                if not os.path.isfile(full_path):
                    reason = "test file not found"
                else:
                    sandbox_env = {
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": sandbox_path,
                    }
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "pytest", full_path, "-v", "--tb=short",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=sandbox_path,
                        env=sandbox_env,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                    output = stdout.decode("utf-8", errors="replace")
                    if proc.returncode == 0:
                        check_passed = True
                        reason = "tests passed"
                    else:
                        # Extract last few lines for failure summary
                        lines = output.strip().splitlines()
                        tail = "\n".join(lines[-5:]) if len(lines) > 5 else output
                        reason = f"tests failed (rc={proc.returncode}): {tail[:200]}"

            elif check_type == "functional_test":
                # Run a Python snippet in the sandbox — passes if exit code 0
                sandbox_env = {
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": sandbox_path,
                }
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", target,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_path,
                    env=sandbox_env,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
                if proc.returncode == 0:
                    check_passed = True
                    reason = "snippet passed"
                else:
                    err = stderr.decode("utf-8", errors="replace")
                    reason = f"snippet failed (rc={proc.returncode}): {err[:200]}"

            elif check_type == "files_changed":
                # Check git diff --name-only has at least N files changed
                min_files = target
                proc = await asyncio.create_subprocess_exec(
                    "git", "diff", "--name-only",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_path,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                changed = [l for l in stdout.decode("utf-8", errors="replace").splitlines() if l.strip()]
                # Also count untracked files
                proc2 = await asyncio.create_subprocess_exec(
                    "git", "status", "--porcelain",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_path,
                )
                stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=10)
                for line in stdout2.decode("utf-8", errors="replace").splitlines():
                    if line.startswith("?"):
                        changed.append(line[3:].strip())
                n_changed = len(set(changed))
                check_passed = n_changed >= min_files
                reason = f"{n_changed} files changed (need >={min_files})"

        except Exception as e:
            reason = f"error: {e}"

        if check_passed:
            passed += 1
        details.append((label, check_passed, reason))

    return passed, total, details


# ---------------------------------------------------------------------------
# Run a single task for a single model
# ---------------------------------------------------------------------------

async def run_task(
    model_spec: str,
    task: dict,
    sandbox_path: str,
    project_context: str,
    timeout: float = 300.0,
    system_config: dict | None = None,
) -> TaskMetrics:
    """Run a coding task with a given model against the real codebase and return metrics.

    Uses execute_dynamic() — iterative dynamic execution — with transcript capture.

    Args:
        system_config: Optional A/B config dict from SYSTEM_CONFIGS.
            Controls scaffolding features (state block, dedup, compaction, etc.).
    """
    config_name = system_config.get("_name", "full") if system_config else "full"
    metrics = TaskMetrics(task_name=task["name"], model=model_spec)

    # Set up transcript capture (include config name in filename)
    transcript_dir = Path("data") / "benchmark_coding_transcripts"
    transcript = TranscriptCapture(model_spec, task["name"], transcript_dir)

    try:
        # Reset sandbox to clean state
        reset_sandbox(sandbox_path)

        # Run task-specific setup hook (e.g., inject a bug)
        setup_hook = task.get("setup_hook")
        if setup_hook and setup_hook in SETUP_HOOKS:
            SETUP_HOOKS[setup_hook](sandbox_path)

        router = make_router(model_spec, timeout=timeout)
        sqlite = await create_sqlite(sandbox_path)
        tool_registry = create_tool_registry(sandbox_path)

        # Get model-specific settings
        ms = get_model_settings(model_spec)

        sc = system_config or {}

        # Build system prompt — minimal or full based on config
        if sc.get("use_minimal_prompt"):
            system_prompt = build_minimal_system_prompt(
                sandbox_path=sandbox_path,
                tool_limit=ms.max_tool_calls,
            )
            system_prompt += "\n\n" + project_context
        else:
            system_prompt = build_executor_system_prompt(
                sandbox_path=sandbox_path,
                project_context=project_context,
                tool_limit=ms.max_tool_calls,
                extra_instructions=ms.extra_instructions,
            )

        planner_config = PlannerConfig(
            enabled=True,
            auto_approve=True,
            max_steps=5,
            max_retries_per_step=1,
        )

        executor = TaskExecutor(
            router=router,
            sqlite=sqlite,
            tool_registry=tool_registry,
            config=planner_config,
            system_prompt=system_prompt,
            max_tool_iterations=ms.max_tool_calls,
        )
        executor.active_project = {"name": "blipshell", "root_path": sandbox_path}
        executor.project_context = ""  # Already baked into system_prompt

        # Apply A/B system config overrides to executor
        if sc.get("disable_state_block"):
            executor._disable_state_block = True
        if sc.get("disable_winddown"):
            executor._disable_winddown = True
        if "enable_dedup" in sc:
            executor._override_enable_dedup = sc["enable_dedup"]
        if "enable_compaction" in sc:
            executor._override_enable_compaction = sc["enable_compaction"]
        if sc.get("guardrails_enabled"):
            from blipshell.models.config import GuardrailsConfig
            executor.guardrails_config = GuardrailsConfig(
                enabled=True,
                trajectory_monitor=True,
                completion_audit=True,
                monitor_interval=5,
                max_audit_retries=1,
            )

        wall_start = time.perf_counter()

        # Add transcript header
        config_label = f" [config: {config_name}]" if config_name != "full" else ""
        transcript.add_header(f"[Dynamic execution — single continuous conversation{config_label}]")

        # Capture metrics as one step (the entire task)
        tool_registry.reset_log()
        task_start_time = time.perf_counter()

        def on_step_complete(step_num: int, result_summary: str):
            step_time = time.perf_counter() - task_start_time
            step_metrics = StepMetrics(
                step_number=step_num,
                description=result_summary[:200],
                execution_time=step_time,
                tool_calls=list(tool_registry.call_log),
                output_preview=result_summary[:500],
            )
            metrics.steps.append(step_metrics)

        try:
            summary = await executor.execute_dynamic(
                task["request"],
                on_step_complete=on_step_complete,
                on_token=transcript.on_token,
            )
            metrics.summary_text = summary[:1000] if summary else ""
        except Exception as e:
            metrics.error = f"Execution failed: {e}"

        metrics.total_time = time.perf_counter() - wall_start
        metrics.plan_steps = len(metrics.steps)  # iterations completed

        # Verification
        metrics.checks_passed, metrics.checks_total, metrics.check_details = (
            await run_verification(sandbox_path, task)
        )

        # Extract pytest results from check details
        for label, ok, reason in metrics.check_details:
            if label.startswith("pytest_in_sandbox"):
                metrics.pytest_passed = ok
                metrics.pytest_output = reason
                break

        # Add verification + timing to transcript (with per-check details)
        transcript.add_verification(metrics.checks_passed, metrics.checks_total)
        for label, ok, reason in metrics.check_details:
            transcript._lines.append(f"  {'PASS' if ok else 'FAIL'} {label}: {reason}\n")
        transcript.add_timing(0.0, metrics.total_time)

        # Parse transcript behavioral metrics
        metrics.transcript_metrics = transcript.parse_metrics()

        # Capture token usage and estimate cost (OpenRouter / OpenAI-compatible)
        try:
            from blipshell.llm.openai_client import OpenAICompatClient
            for ep in router._endpoint_manager._endpoints:
                client = ep._client
                if isinstance(client, OpenAICompatClient):
                    metrics.prompt_tokens = client.total_prompt_tokens
                    metrics.completion_tokens = client.total_completion_tokens
                    metrics.api_requests = client.total_requests
                    # Estimate cost from endpoint config
                    cfg = ep._config
                    cost_prompt = cfg.cost_per_1m_prompt * metrics.prompt_tokens / 1_000_000
                    cost_completion = cfg.cost_per_1m_completion * metrics.completion_tokens / 1_000_000
                    metrics.estimated_cost_usd = cost_prompt + cost_completion
                    transcript._lines.append(
                        f"\nTOKEN USAGE: {metrics.prompt_tokens:,} prompt + "
                        f"{metrics.completion_tokens:,} completion = "
                        f"{metrics.prompt_tokens + metrics.completion_tokens:,} total "
                        f"({metrics.api_requests} requests)\n"
                        f"ESTIMATED COST: ${metrics.estimated_cost_usd:.4f}\n"
                    )
                    break
        except Exception:
            pass  # non-critical — cost tracking is best-effort

        # Save transcript
        metrics.transcript_path = transcript.save()

        # Close DB
        try:
            await sqlite.close()
        except Exception:
            pass

    except Exception as e:
        metrics.error = str(e)

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
    table.add_column("Task", width=16)
    table.add_column("Iters", width=6, justify="center")
    table.add_column("Tools", width=6, justify="center")
    table.add_column("Edit Fail", width=9, justify="center")
    table.add_column("Reads", width=6, justify="center")
    table.add_column("Total(s)", width=8, justify="right")
    table.add_column("Checks", width=8, justify="center")
    table.add_column("Pytest", width=7, justify="center")
    table.add_column("?s Asked", width=8, justify="center")
    table.add_column("Cost", width=8, justify="right")

    for model, task_results in all_results.items():
        for i, m in enumerate(task_results):
            model_col = model if i == 0 else ""
            checks_str = f"{m.checks_passed}/{m.checks_total}"
            edit_fail_str = (
                f"[red]{m.total_edit_failures}[/red]" if m.total_edit_failures > 0
                else "[green]0[/green]"
            )
            unwanted_str = ""
            if m.unwanted_files:
                unwanted_str = f" [red]+{len(m.unwanted_files)}md[/red]"

            questions_str = (
                f"[red]{m.transcript_metrics.questions_asked}[/red]"
                if m.transcript_metrics.questions_asked > 0
                else "[green]0[/green]"
            )

            # Pytest column: show pass/fail/n-a
            if any(label.startswith("pytest_in_sandbox") for label, _, _ in m.check_details):
                pytest_str = "[green]PASS[/green]" if m.pytest_passed else "[red]FAIL[/red]"
            else:
                pytest_str = "[dim]n/a[/dim]"

            cost_str = f"${m.estimated_cost_usd:.3f}" if m.estimated_cost_usd > 0 else "[dim]local[/dim]"

            table.add_row(
                model_col,
                m.task_name,
                f"{len(m.steps)}",
                f"{m.total_tool_calls}{unwanted_str}",
                edit_fail_str,
                f"{m.total_file_reads}",
                f"{m.total_time:.1f}",
                checks_str,
                pytest_str,
                questions_str,
                cost_str,
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
    table.add_column("Checks", width=10, justify="center")
    table.add_column("Questions", width=10, justify="center")
    table.add_column("Linux Cmds", width=10, justify="center")
    table.add_column("Total Cost", width=10, justify="right")

    for model, task_results in all_results.items():
        total_time = sum(m.total_time for m in task_results)
        avg_time = total_time / len(task_results) if task_results else 0
        total_tools = sum(m.total_tool_calls for m in task_results)
        total_edit_fail = sum(m.total_edit_failures for m in task_results)
        checks_pass = sum(m.checks_passed for m in task_results)
        checks_total = sum(m.checks_total for m in task_results)
        total_questions = sum(m.transcript_metrics.questions_asked for m in task_results)
        total_linux = sum(m.transcript_metrics.linux_cmds_on_windows for m in task_results)
        total_cost = sum(m.estimated_cost_usd for m in task_results)

        edit_str = (
            f"[red]{total_edit_fail}[/red]" if total_edit_fail > 0
            else "[green]0[/green]"
        )
        questions_str = (
            f"[red]{total_questions}[/red]" if total_questions > 0
            else "[green]0[/green]"
        )
        linux_str = (
            f"[red]{total_linux}[/red]" if total_linux > 0
            else "[green]0[/green]"
        )
        cost_str = f"${total_cost:.3f}" if total_cost > 0 else "[dim]local[/dim]"

        table.add_row(
            model,
            f"{len(task_results)}",
            f"{total_time:.1f}s",
            f"{avg_time:.1f}s",
            f"{total_tools}",
            edit_str,
            f"{checks_pass}/{checks_total}",
            questions_str,
            linux_str,
            cost_str,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(
    models: list[str],
    timeout: float = 300.0,
    tasks: list[dict] | None = None,
    system_config: dict | None = None,
):
    """Run the coding benchmark for specified models.

    Args:
        tasks: Task list to run. Defaults to CODING_TASKS.
        system_config: A/B config dict from SYSTEM_CONFIGS. None = "full" (defaults).
    """
    task_list = tasks or CODING_TASKS
    config_name = system_config.get("_name", "full") if system_config else "full"

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

    console.print(f"\n[bold]Coding Model Benchmark (Real Codebase)[/bold]")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Tasks: {', '.join(t['name'] for t in task_list)}")
    console.print(f"Config: [bold]{config_name}[/bold] — {(system_config or {}).get('description', 'all systems ON')}")
    console.print(f"Timeout: {timeout}s per LLM call\n")

    # Create a single sandbox for the entire benchmark run
    console.print("[dim]Creating project sandbox (copying BlipShell source)...[/dim]")
    sandbox_path = create_project_sandbox()
    console.print(f"[dim]Sandbox: {sandbox_path}[/dim]")

    # Build project context once (same for all models)
    console.print("[dim]Building project context (repo map, file tree)...[/dim]")
    project_context = build_project_context(sandbox_path)
    console.print(f"[dim]Project context: {len(project_context)} chars[/dim]\n")

    all_results: dict[str, list[TaskMetrics]] = {}

    try:
        for model_spec in models:
            # Results key includes config for A/B comparison
            result_key = f"{model_spec}:{config_name}" if config_name != "full" else model_spec
            console.rule(f"[bold blue]Benchmarking: {result_key}")
            model_results: list[TaskMetrics] = []

            for task in task_list:
                console.print(f"\n  [dim]Task: {task['name']} — {task['description']}[/dim]")
                console.print(f"  [dim]Request: {task['request'][:80]}...[/dim]")

                metrics = await run_task(
                    model_spec, task, sandbox_path, project_context,
                    timeout=timeout, system_config=system_config,
                )

                if metrics.error:
                    console.print(f"  [red]ERROR: {metrics.error}[/red]")
                else:
                    console.print(
                        f"  [green]Done[/green] — "
                        f"{metrics.plan_steps} steps, "
                        f"{metrics.total_tool_calls} tool calls, "
                        f"{metrics.total_edit_failures} edit failures, "
                        f"checks {metrics.checks_passed}/{metrics.checks_total}, "
                        f"time {metrics.total_time:.1f}s"
                    )
                    # Show per-check results
                    for label, ok, reason in metrics.check_details:
                        mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
                        console.print(f"    {mark} {label}: {reason}")
                    if metrics.transcript_path:
                        console.print(f"  [dim]Transcript: {metrics.transcript_path}[/dim]")

                model_results.append(metrics)

            all_results[result_key] = model_results
            all_raw[result_key] = [m.to_dict() for m in model_results]

            # Save incrementally after each model
            with open(output_path, "w") as f:
                json.dump(all_raw, f, indent=2)

            console.print(f"\n  [green]Completed all tasks for {result_key}[/green]")

    finally:
        # Cleanup sandbox
        try:
            shutil.rmtree(sandbox_path, ignore_errors=True)
        except Exception:
            pass

    # Final save
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
    print_step_detail_table(all_results)


async def dry_run_verify():
    """Create sandbox, skip LLM, run verification against unmodified code.

    All diff-based checks should fail (0/N) since no code was changed.
    Useful for validating that checks don't pass trivially on the baseline.
    """
    console.print("\n[bold]Dry-Run Verification (no LLM, unmodified sandbox)[/bold]")
    console.print("Expected: diff-based checks should FAIL, syntax checks should PASS\n")

    sandbox_path = create_project_sandbox()
    console.print(f"[dim]Sandbox: {sandbox_path}[/dim]\n")

    try:
        for task in CODING_TASKS:
            console.print(f"[bold]{task['name']}[/bold] — {task['description']}")
            passed, total, details = await run_verification(sandbox_path, task)
            for label, ok, reason in details:
                mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
                console.print(f"  {mark} {label}: {reason}")
            color = "green" if passed == 0 else "red"
            console.print(f"  [{color}]Result: {passed}/{total} passed[/{color}]\n")
    finally:
        shutil.rmtree(sandbox_path, ignore_errors=True)


def main():
    """CLI entry point.

    Usage:
        python tests/benchmark_coding.py                              # all models, standard tasks
        python tests/benchmark_coding.py glm-5:cloud --hard           # one model, hard tasks
        python tests/benchmark_coding.py glm-5:cloud --expert         # one model, expert tasks
        python tests/benchmark_coding.py glm-5:cloud --config bare    # A/B test with bare config
        python tests/benchmark_coding.py --ab glm-5:cloud             # run all configs for one model
        python tests/benchmark_coding.py --tasks stats_command,edge_case_parser  # specific tasks
        python tests/benchmark_coding.py --list-configs               # show available A/B configs
        python tests/benchmark_coding.py --dry-run-verify             # test checks on unmodified code
    """
    models = []
    timeout = 300.0
    dry_run = False
    use_hard = False
    use_expert = False
    use_all = False
    config_name = "full"
    ab_mode = False
    task_filter: list[str] = []

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--timeout" and i + 1 < len(args):
            timeout = float(args[i + 1])
            i += 2
        elif args[i] == "--dry-run-verify":
            dry_run = True
            i += 1
        elif args[i] == "--hard":
            use_hard = True
            i += 1
        elif args[i] == "--expert":
            use_expert = True
            i += 1
        elif args[i] == "--all":
            use_all = True
            i += 1
        elif args[i] == "--config" and i + 1 < len(args):
            config_name = args[i + 1]
            i += 2
        elif args[i] == "--ab":
            ab_mode = True
            i += 1
        elif args[i] == "--tasks" and i + 1 < len(args):
            task_filter = [t.strip() for t in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--list-configs":
            console.print("\n[bold]Available A/B System Configurations[/bold]\n")
            for name, cfg in SYSTEM_CONFIGS.items():
                desc = cfg.get("description", "")
                overrides = {k: v for k, v in cfg.items() if k != "description"}
                console.print(f"  [cyan]{name:16s}[/cyan] {desc}")
                if overrides:
                    console.print(f"                   [dim]{overrides}[/dim]")
            console.print()
            return
        else:
            models.append(args[i])
            i += 1

    if dry_run:
        asyncio.run(dry_run_verify())
        return

    if not models:
        models = BENCHMARK_MODELS

    # Build task list
    if use_all:
        task_list = CODING_TASKS + HARD_TASKS + EXPERT_TASKS
    elif use_expert:
        task_list = EXPERT_TASKS
    elif use_hard:
        task_list = HARD_TASKS
    else:
        task_list = CODING_TASKS

    # Apply task filter
    if task_filter:
        task_list = [t for t in task_list if t["name"] in task_filter]
        if not task_list:
            all_names = [t["name"] for t in CODING_TASKS + HARD_TASKS + EXPERT_TASKS]
            console.print(f"[red]No tasks matched filter: {task_filter}[/red]")
            console.print(f"[dim]Available: {', '.join(all_names)}[/dim]")
            return

    # A/B mode: run all configs for each model
    if ab_mode:
        console.print(f"\n[bold]A/B Benchmark — testing {len(SYSTEM_CONFIGS)} configs[/bold]\n")
        for cfg_name, cfg in SYSTEM_CONFIGS.items():
            cfg_with_name = {**cfg, "_name": cfg_name}
            console.rule(f"[bold magenta]Config: {cfg_name} — {cfg.get('description', '')}")
            asyncio.run(run_benchmark(
                models, timeout=timeout, tasks=task_list, system_config=cfg_with_name,
            ))
        return

    # Single config mode
    system_config = None
    if config_name != "full":
        if config_name not in SYSTEM_CONFIGS:
            console.print(f"[red]Unknown config: {config_name}. Use --list-configs to see options.[/red]")
            return
        system_config = {**SYSTEM_CONFIGS[config_name], "_name": config_name}

    asyncio.run(run_benchmark(
        models, timeout=timeout, tasks=task_list, system_config=system_config,
    ))


if __name__ == "__main__":
    main()
