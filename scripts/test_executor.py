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
    # ══════════════════════════════════════════════════════════════════════
    # Category 1: Tool Coverage — every tool exercised individually
    # ══════════════════════════════════════════════════════════════════════
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
    {
        "name": "tool_git_diff",
        "category": "tool_coverage",
        "task": (
            "Run git diff to see if there are any unstaged changes in the project. "
            "Report what you find."
        ),
        "expect_tools": ["git_diff"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 2: Multi-step Tasks -- chains of 3+ tools
    # ======================================================================
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
    {
        "name": "multi_glob_grep_read",
        "category": "multi_step",
        "task": (
            "Use glob_files to find all *.py files under blipshell/llm/. "
            "Then use grep_files to search those files for 'class.*Client'. "
            "Then read the first 50 lines of any file that has a Client class. "
            "Report what client classes exist and their constructors."
        ),
        "expect_tools": ["glob_files", "grep_files", "read_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "multi_five_step_chain",
        "category": "multi_step",
        "task": (
            "Complete these steps in order:\n"
            "1. List the blipshell/models/ directory\n"
            "2. Read blipshell/models/tools.py\n"
            "3. Search for all files that import from blipshell.models.tools\n"
            "4. Create stress_test_dependency_map.txt listing which files depend on models/tools.py\n"
            "5. Verify the file was created by reading it back"
        ),
        "expect_tools": ["list_directory", "read_file", "grep_files", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 3: Error Recovery -- handle failures gracefully
    # ======================================================================
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
    {
        "name": "error_blocked_shell_cmd",
        "category": "error_recovery",
        "task": (
            "Try to run the command 'cat blipshell/__init__.py' using run_command. "
            "When it gets blocked, use the correct tool (read_file) instead."
        ),
        "expect_tools": ["read_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "error_wrong_directory_grep",
        "category": "error_recovery",
        "task": (
            "Try to use grep_files on the file blipshell/core/agent.py (not a directory). "
            "When it fails, adjust and search the parent directory blipshell/core/ instead "
            "for the pattern 'async def chat'."
        ),
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "error_multi_syntax_fix",
        "category": "error_recovery",
        "task": (
            "Create a file stress_test_multi_error.py with this content:\n"
            "def foo()\n"
            "    x = [1, 2, 3\n"
            "    return x\n\n"
            "Check syntax with python. There are TWO errors: missing colon and unclosed bracket. "
            "Fix BOTH errors and verify the file passes syntax check."
        ),
        "expect_tools": ["write_file", "run_command", "edit_file"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 4: Scaffolding & State Awareness
    # ======================================================================
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
    {
        "name": "scaffold_prefer_edit_over_write",
        "category": "scaffolding",
        "task": (
            "Read stress_test_output.py (created by an earlier test). "
            "Add a new function called farewell(name) that returns f'Goodbye, {name}!'. "
            "Use edit_file, NOT write_file — the file already exists."
        ),
        "expect_tools": ["read_file", "edit_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "scaffold_glob_vs_grep",
        "category": "scaffolding",
        "task": (
            "I need two things:\n"
            "1. Find all .yaml files in the project (use the right tool for finding files by pattern)\n"
            "2. Find all files that contain the string 'endpoint' (use the right tool for searching content)\n"
            "Use the correct specialized tool for each — not run_command."
        ),
        "expect_tools": ["glob_files", "grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 5: Simple Chat Path (force_plan=False)
    # ======================================================================
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
    {
        "name": "simple_chat_followup",
        "category": "simple_chat",
        "task": "Can you explain what asyncio is and when to use it in Python?",
        "expect_tools": [],
        "expect_complete": True,
        "force_plan": False,
    },
    {
        "name": "simple_chat_code_question",
        "category": "simple_chat",
        "task": "Write me a Python function that checks if a string is a palindrome. Just show the code, don't create any files.",
        "expect_tools": [],
        "expect_complete": True,
        "force_plan": False,
    },

    # ======================================================================
    # Category 6: Complex Real-World Tasks
    # ======================================================================
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
        "name": "real_multi_file_create_test",
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
    {
        "name": "real_dataclass_module",
        "category": "real_world",
        "task": (
            "Create a file stress_test_models.py with:\n"
            "1. A dataclass 'User' with fields: name (str), email (str), age (int)\n"
            "2. A dataclass 'Team' with fields: name (str), members (list[User])\n"
            "3. A function create_team(name, *users) -> Team\n"
            "4. A function team_average_age(team) -> float\n\n"
            "Then create stress_test_models_test.py that tests:\n"
            "- Creating a User and Team\n"
            "- create_team with multiple users\n"
            "- team_average_age with known values\n"
            "- team_average_age with empty team (should return 0.0)\n\n"
            "Run the tests."
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "real_config_parser",
        "category": "real_world",
        "task": (
            "Create stress_test_config_parser.py with a class SimpleConfig that:\n"
            "1. __init__(self, path: str) — takes a file path\n"
            "2. load(self) -> dict — reads the file as key=value pairs (one per line), "
            "ignoring blank lines and lines starting with #\n"
            "3. get(self, key, default=None) — returns value for key\n"
            "4. set(self, key, value) — sets a key\n"
            "5. save(self) — writes back to the file\n\n"
            "Create stress_test_config_parser_test.py that:\n"
            "- Creates a temp file with known content\n"
            "- Tests load(), get(), set(), save()\n"
            "- Verifies comments and blank lines are preserved\n\n"
            "Run the tests."
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "real_cli_tool",
        "category": "real_world",
        "task": (
            "Create stress_test_cli.py — a simple command-line tool using argparse that:\n"
            "1. Accepts a positional argument 'filename'\n"
            "2. Has --count flag that prints the number of lines\n"
            "3. Has --upper flag that prints the content in uppercase\n"
            "4. Has --search PATTERN option that prints matching lines\n"
            "5. Without flags, just prints the file content\n\n"
            "Create a test file stress_test_cli_input.txt with 5 lines of text. "
            "Then test the CLI tool by running:\n"
            "- python stress_test_cli.py stress_test_cli_input.txt --count\n"
            "- python stress_test_cli.py stress_test_cli_input.txt --upper\n"
            "Verify the output looks correct."
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "real_find_and_document",
        "category": "real_world",
        "task": (
            "Find all Python files under blipshell/memory/ using glob_files. "
            "For each file, read the first 20 lines to get the module docstring. "
            "Create stress_test_memory_docs.md with a summary table:\n"
            "| File | Purpose |\n"
            "| --- | --- |\n"
            "for each file in the memory module."
        ),
        "expect_tools": ["glob_files", "read_file", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "real_bug_hunt",
        "category": "real_world",
        "task": (
            "Create a file stress_test_buggy.py with this deliberately buggy code:\n\n"
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
            "Create stress_test_buggy_test.py that tests both functions and finds the bugs:\n"
            "- fibonacci(7) should return [0, 1, 1, 2, 3, 5, 8]\n"
            "- find_max([-5, -3, -1]) should return -1 (not 0)\n\n"
            "Run the tests, find the failures, fix the bugs, and re-run until they pass."
        ),
        "expect_tools": ["write_file", "run_command", "edit_file"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 7: Edge Cases
    # ======================================================================
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
            "# Special chars: quotes and backslash\n"
            "data = {'key': 'value', 'path': 'C:\\\\Users\\\\test'}\n"
            "message = \"it's a \\\"test\\\"\"\n"
            "print(data, message)\n\n"
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
    {
        "name": "edge_deeply_nested_path",
        "category": "edge_case",
        "task": (
            "List the directory blipshell/core/tools/ then list blipshell/memory/ "
            "then list blipshell/ui/. Report the total number of Python files "
            "across all three directories."
        ),
        "expect_tools": ["list_directory"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "edge_regex_grep",
        "category": "edge_case",
        "task": (
            "Use grep_files with the regex pattern 'def __init__\\(self.*config' "
            "to find all classes whose __init__ takes a config parameter. "
            "Report what you find."
        ),
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "edge_overwrite_file",
        "category": "edge_case",
        "task": (
            "Create a file stress_test_overwrite.txt with content 'version 1'. "
            "Then read it back. Then create the same file with content 'version 2' "
            "(overwriting it). Read it back again and confirm it says 'version 2'."
        ),
        "expect_tools": ["write_file", "read_file"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 8: Flow Observability
    # ======================================================================
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
        "expect_flow_events": ["turn_start", "llm_complete"],
    },
    {
        "name": "flow_events_simple_path",
        "category": "observability",
        "task": "What time is it?",
        "expect_tools": [],
        "expect_complete": True,
        "force_plan": False,
        "expect_flow_events": ["turn_start"],
    },

    # ======================================================================
    # Category 9: Consistency / Repeat Tests -- same task multiple times
    # ======================================================================
    {
        "name": "consistency_write_1",
        "category": "consistency",
        "task": (
            "Create a Python file called stress_test_consistent_1.py with a function "
            "is_even(n) that returns True if n is even, False otherwise. "
            "Include a docstring. Verify syntax."
        ),
        "expect_tools": ["write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "consistency_write_2",
        "category": "consistency",
        "task": (
            "Create a Python file called stress_test_consistent_2.py with a function "
            "is_odd(n) that returns True if n is odd, False otherwise. "
            "Include a docstring. Verify syntax."
        ),
        "expect_tools": ["write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "consistency_write_3",
        "category": "consistency",
        "task": (
            "Create a Python file called stress_test_consistent_3.py with a function "
            "is_positive(n) that returns True if n > 0, False otherwise. "
            "Include a docstring. Verify syntax."
        ),
        "expect_tools": ["write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "consistency_grep_1",
        "category": "consistency",
        "task": "Search for all files containing 'class Agent' and report the results.",
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "consistency_grep_2",
        "category": "consistency",
        "task": "Search for all files containing 'class TaskExecutor' and report the results.",
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "consistency_grep_3",
        "category": "consistency",
        "task": "Search for all files containing 'class MemoryManager' and report the results.",
        "expect_tools": ["grep_files"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 10: Heavy / Long-running Tasks (budget pressure)
    # ======================================================================
    {
        "name": "heavy_full_codebase_survey",
        "category": "heavy",
        "task": (
            "Survey the entire blipshell/ package. List every subdirectory, "
            "count the Python files in each, and for each directory read at least "
            "one file's first 20 lines. Create stress_test_full_survey.txt with:\n"
            "- Directory name\n"
            "- Number of .py files\n"
            "- One-line summary of what the directory does\n"
            "Cover: blipshell/core/, blipshell/llm/, blipshell/memory/, "
            "blipshell/models/, blipshell/session/, blipshell/ui/"
        ),
        "expect_tools": ["list_directory", "read_file", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "heavy_create_test_suite",
        "category": "heavy",
        "task": (
            "Create a mini test suite with 3 modules and their tests:\n\n"
            "1. stress_test_stack.py — a Stack class with push, pop, peek, is_empty, size\n"
            "2. stress_test_queue.py — a Queue class with enqueue, dequeue, peek, is_empty, size\n"
            "3. stress_test_linked_list.py — a LinkedList class with append, prepend, find, delete, to_list\n\n"
            "For each, create a corresponding _test.py file with pytest tests covering "
            "all methods including edge cases (empty collection operations). "
            "Run ALL tests together: python -m pytest stress_test_stack_test.py "
            "stress_test_queue_test.py stress_test_linked_list_test.py -v"
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "heavy_cross_reference",
        "category": "heavy",
        "task": (
            "Build a cross-reference of the BlipShell codebase:\n"
            "1. Find all classes that inherit from Tool (grep for 'class.*Tool.*:')\n"
            "2. Find all classes that inherit from ABC\n"
            "3. Find all files that import from blipshell.core.agent\n"
            "4. Find all files that import from blipshell.llm.router\n"
            "5. Find all files that import from blipshell.memory.sqlite_store\n\n"
            "Create stress_test_xref.json with the results organized by category."
        ),
        "expect_tools": ["grep_files", "write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "heavy_read_modify_verify",
        "category": "heavy",
        "task": (
            "Create stress_test_transform.py with these functions:\n"
            "1. reverse_words(s) — reverses word order in a string\n"
            "2. count_vowels(s) — counts vowels in a string\n"
            "3. title_case(s) — converts to title case without using .title()\n"
            "4. remove_duplicates(lst) — removes duplicates preserving order\n"
            "5. flatten(nested) — flattens a nested list of any depth\n\n"
            "Create stress_test_transform_test.py with thorough tests. "
            "Run the tests. If any fail, fix the implementation and re-run."
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 11: Git Operations
    # ======================================================================
    {
        "name": "git_status_and_diff",
        "category": "git",
        "task": (
            "Check the git status and git diff of the project. "
            "Report: what branch, how many files modified, any untracked files."
        ),
        "expect_tools": ["git_status", "git_diff"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "git_log_analysis",
        "category": "git",
        "task": (
            "Run 'git log --oneline -10' to see the last 10 commits. "
            "Report the commit messages and whether they follow a consistent format."
        ),
        "expect_tools": ["run_command"],
        "expect_complete": True,
        "force_plan": True,
    },

    # ======================================================================
    # Category 12: Instruction Following
    # ======================================================================
    {
        "name": "instruct_exact_content",
        "category": "instruction",
        "task": (
            "Create a file called stress_test_exact.py with EXACTLY this content "
            "(no additions, no modifications):\n\n"
            "# Exact content test\n"
            "x = 42\n"
            "print(x)\n"
        ),
        "expect_tools": ["write_file"],
        "expect_complete": True,
        "force_plan": True,
    },
    {
        "name": "instruct_do_not_modify",
        "category": "instruction",
        "task": (
            "Read the file blipshell/core/executor.py. "
            "Tell me what the _compact_messages function does. "
            "Do NOT modify the file in any way — just read and report."
        ),
        "expect_tools": ["read_file"],
        "expect_complete": True,
        "force_plan": True,
        "expect_max_tool_calls": 5,
    },
    {
        "name": "instruct_specific_line_range",
        "category": "instruction",
        "task": (
            "Read lines 1-30 of blipshell/core/agent.py. "
            "List all the imports you see in those 30 lines."
        ),
        "expect_tools": ["read_file"],
        "expect_complete": True,
        "force_plan": True,
        "expect_max_tool_calls": 5,
    },
    {
        "name": "instruct_multi_constraint",
        "category": "instruction",
        "task": (
            "Create stress_test_constrained.py following ALL of these constraints:\n"
            "1. Must have exactly 3 functions\n"
            "2. Each function must have a docstring\n"
            "3. No function may be longer than 5 lines (including the def line)\n"
            "4. All functions must take exactly 2 parameters\n"
            "5. File must pass python syntax check\n\n"
            "Verify syntax after creating the file."
        ),
        "expect_tools": ["write_file", "run_command"],
        "expect_complete": True,
        "force_plan": True,
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
