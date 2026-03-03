"""Coding suite — code generation, tool calling, and chat quality benchmark.

Tests interactive model quality: can it generate correct code, call tools
accurately, and produce helpful conversational responses?
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded test cases
# ---------------------------------------------------------------------------

CODE_GENERATION_TESTS = [
    {
        "prompt": "Write a Python function that checks if a string is a palindrome. Return only the function.",
        "checks": [
            lambda r: "def " in r,
            lambda r: "palindrome" in r.lower() or "[::-1]" in r or "reversed" in r,
            lambda r: "return" in r,
        ],
        "label": "palindrome",
    },
    {
        "prompt": "Write a Python function that finds the two numbers in a list that add up to a target. Return only the function.",
        "checks": [
            lambda r: "def " in r,
            lambda r: "return" in r,
            lambda r: "for " in r or "while " in r or "enumerate" in r,
        ],
        "label": "two_sum",
    },
    {
        "prompt": "Write a Python function to flatten a nested list (e.g., [1, [2, [3, 4]], 5] -> [1, 2, 3, 4, 5]). Return only the function.",
        "checks": [
            lambda r: "def " in r,
            lambda r: "list" in r.lower() or "isinstance" in r or "flatten" in r.lower(),
            lambda r: "return" in r or "yield" in r,
        ],
        "label": "flatten",
    },
    {
        "prompt": "Write a Python function that converts a Roman numeral string to an integer. Return only the function.",
        "checks": [
            lambda r: "def " in r,
            lambda r: any(x in r for x in ["IV", "IX", "XL", "'I'", '"I"', "dict", "map"]),
            lambda r: "return" in r,
        ],
        "label": "roman_to_int",
    },
]

TOOL_CALLING_TESTS = [
    {
        "system": (
            "You have access to these tools:\n"
            "- read_file(path: str) -> str: Read a file's contents\n"
            "- write_file(path: str, content: str): Write content to a file\n"
            "- search(query: str) -> list[str]: Search for files\n\n"
            "When you need to use a tool, respond with a JSON object: "
            '{{"tool": "tool_name", "args": {{"param": "value"}}}}'
        ),
        "prompt": "Read the file config.yaml",
        "checks": [
            lambda r: "read_file" in r.lower(),
            lambda r: "config.yaml" in r or "config" in r.lower(),
        ],
        "label": "read_file",
    },
    {
        "system": (
            "You have access to these tools:\n"
            "- read_file(path: str) -> str\n"
            "- write_file(path: str, content: str)\n"
            "- search(query: str) -> list[str]\n\n"
            "When you need to use a tool, respond with a JSON object: "
            '{{"tool": "tool_name", "args": {{"param": "value"}}}}'
        ),
        "prompt": "Find all Python files that contain the word 'database'",
        "checks": [
            lambda r: "search" in r.lower(),
            lambda r: "database" in r.lower() or "python" in r.lower() or ".py" in r,
        ],
        "label": "search",
    },
    {
        "system": (
            "You have access to these tools:\n"
            "- read_file(path: str) -> str\n"
            "- write_file(path: str, content: str)\n"
            "- run_command(cmd: str) -> str: Run a shell command\n\n"
            "When you need to use a tool, respond with a JSON object: "
            '{{"tool": "tool_name", "args": {{"param": "value"}}}}'
        ),
        "prompt": "Write 'hello world' to the file output.txt",
        "checks": [
            lambda r: "write_file" in r.lower(),
            lambda r: "output.txt" in r or "output" in r.lower(),
            lambda r: "hello" in r.lower(),
        ],
        "label": "write_file",
    },
]

CHAT_QUALITY_TESTS = [
    {
        "prompt": "What's the difference between a list and a tuple in Python?",
        "checks": [
            lambda r: "mutable" in r.lower() or "immutable" in r.lower(),
            lambda r: "list" in r.lower() and "tuple" in r.lower(),
            lambda r: len(r.split()) > 20,  # should give a real explanation
        ],
        "label": "list_vs_tuple",
    },
    {
        "prompt": "Explain what a deadlock is in concurrent programming, in 2-3 sentences.",
        "checks": [
            lambda r: "lock" in r.lower() or "wait" in r.lower() or "block" in r.lower(),
            lambda r: "thread" in r.lower() or "process" in r.lower() or "resource" in r.lower(),
            lambda r: len(r.split()) < 200,  # should be concise
        ],
        "label": "deadlock",
    },
    {
        "prompt": "I have a SQLite database that's getting slow with 100K rows. What should I check first?",
        "checks": [
            lambda r: "index" in r.lower(),
            lambda r: any(x in r.lower() for x in ["query", "explain", "analyze", "vacuum", "wal"]),
            lambda r: len(r.split()) > 15,
        ],
        "label": "sqlite_perf",
    },
    {
        "prompt": "What does this error mean: 'TypeError: cannot unpack non-iterable NoneType object'?",
        "checks": [
            lambda r: "none" in r.lower() or "None" in r,
            lambda r: "return" in r.lower() or "unpack" in r.lower() or "tuple" in r.lower(),
        ],
        "label": "error_explain",
    },
]


class CodingSuite(BenchmarkSuite):
    name = "coding"
    description = "Code generation, tool calling accuracy, and chat quality"
    task_types = ["coding", "tool_calling", "reasoning"]
    needs_db = False
    needs_router = True
    quick_samples = 0  # fixed data
    thorough_samples = 0

    async def run(
        self,
        models: list[str],
        *,
        router_factory: Callable[[str], LLMRouter] | None = None,
        config: BlipShellConfig | None = None,
        db_path: str | None = None,
        ollama_url: str = "http://localhost:11434",
        thorough: bool = False,
        on_status: Callable[[str], None] | None = None,
    ) -> list[SuiteResult]:
        results = []
        for model in models:
            if on_status:
                on_status(f"[coding] Testing {model}")
            router = router_factory(model) if router_factory else None
            if not router:
                continue
            sr = await self._benchmark_model(model, router, on_status)
            results.append(sr)
        return results

    async def _benchmark_model(
        self, model: str, router: LLMRouter, on_status: Callable | None,
    ) -> SuiteResult:
        scores = []
        total_start = time.monotonic()

        code_score = await self._bench_code_generation(router, on_status)
        scores.append(code_score)

        tool_score = await self._bench_tool_calling(router, on_status)
        scores.append(tool_score)

        chat_score = await self._bench_chat_quality(router, on_status)
        scores.append(chat_score)

        elapsed = time.monotonic() - total_start
        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(elapsed, 1),
        )

    async def _bench_code_generation(
        self, router: LLMRouter, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.router import TaskType

        times = []
        checks_passed = 0
        total_checks = 0
        errors = 0

        system = "You are a coding assistant. Write clean, correct Python code. Return only the code, no explanations."

        for test in CODE_GENERATION_TESTS:
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.CODING, test["prompt"], system=system, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                for check in test["checks"]:
                    total_checks += 1
                    if check(raw):
                        checks_passed += 1
            except Exception as e:
                logger.debug("code gen error (%s): %s", test["label"], e)
                errors += 1
                total_checks += len(test["checks"])

        avg_speed = sum(times) / len(times) if times else 0
        quality = checks_passed / total_checks if total_checks else 0

        return TaskScore(
            task_name="code_generation",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(CODE_GENERATION_TESTS),
            errors=errors,
            detail={"checks_passed": checks_passed, "total_checks": total_checks},
        )

    async def _bench_tool_calling(
        self, router: LLMRouter, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.router import TaskType

        times = []
        checks_passed = 0
        total_checks = 0
        errors = 0

        for test in TOOL_CALLING_TESTS:
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.TOOL_CALLING, test["prompt"],
                    system=test["system"], think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                for check in test["checks"]:
                    total_checks += 1
                    if check(raw):
                        checks_passed += 1
            except Exception as e:
                logger.debug("tool calling error (%s): %s", test["label"], e)
                errors += 1
                total_checks += len(test["checks"])

        avg_speed = sum(times) / len(times) if times else 0
        quality = checks_passed / total_checks if total_checks else 0

        return TaskScore(
            task_name="tool_calling",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(TOOL_CALLING_TESTS),
            errors=errors,
            detail={"checks_passed": checks_passed, "total_checks": total_checks},
        )

    async def _bench_chat_quality(
        self, router: LLMRouter, on_status: Callable | None,
    ) -> TaskScore:
        from blipshell.llm.router import TaskType

        times = []
        checks_passed = 0
        total_checks = 0
        errors = 0

        system = (
            "You are a helpful programming assistant. "
            "Give clear, concise, accurate answers."
        )

        for test in CHAT_QUALITY_TESTS:
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.REASONING, test["prompt"], system=system, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                for check in test["checks"]:
                    total_checks += 1
                    if check(raw):
                        checks_passed += 1
            except Exception as e:
                logger.debug("chat quality error (%s): %s", test["label"], e)
                errors += 1
                total_checks += len(test["checks"])

        avg_speed = sum(times) / len(times) if times else 0
        quality = checks_passed / total_checks if total_checks else 0

        return TaskScore(
            task_name="chat_quality",
            quality=round(quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(CHAT_QUALITY_TESTS),
            errors=errors,
            detail={"checks_passed": checks_passed, "total_checks": total_checks},
        )
