"""Coding suite — code generation quality on real-world patterns.

Tests model ability to generate, fix, and refactor Python code using
patterns from the BlipShell codebase. Validates syntax + structural
correctness, not just "does it contain def".
"""

from __future__ import annotations

import ast
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
# Test tasks — based on real BlipShell patterns
# ---------------------------------------------------------------------------

CODE_TASKS: list[dict] = [
    # 1. Dataclass with validation (BlipShell uses these everywhere)
    {
        "name": "dataclass_config",
        "prompt": (
            "Write a Python dataclass called EndpointHealth with fields: "
            "name (str), url (str), is_healthy (bool, default True), "
            "failure_count (int, default 0), last_checked (float, default 0.0). "
            "Add a method `record_failure()` that increments failure_count and "
            "sets is_healthy to False if failure_count >= 3. "
            "Add a method `record_success()` that resets failure_count to 0 and "
            "sets is_healthy to True. "
            "Use the @dataclass decorator. Only output the code, no explanation."
        ),
        "checks": [
            ("has_dataclass_decorator", lambda c: "@dataclass" in c),
            ("has_class", lambda c: "class EndpointHealth" in c),
            ("has_fields", lambda c: "name" in c and "url" in c and "is_healthy" in c and "failure_count" in c),
            ("has_record_failure", lambda c: "def record_failure" in c),
            ("has_record_success", lambda c: "def record_success" in c),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
    # 2. Async retry function (common pattern in LLM client code)
    {
        "name": "async_retry",
        "prompt": (
            "Write an async Python function called `retry_call` that takes an "
            "async callable `fn`, `max_retries` (int, default 3), and "
            "`base_delay` (float, default 1.0). It should call fn(), and if it "
            "raises an exception, retry up to max_retries times with exponential "
            "backoff (base_delay * 2**attempt). If all retries fail, raise the "
            "last exception. Use asyncio.sleep for delays. "
            "Only output the code, no explanation."
        ),
        "checks": [
            ("has_async_def", lambda c: "async def retry_call" in c),
            ("has_await", lambda c: "await" in c),
            ("has_retry_loop", lambda c: "for" in c or "while" in c),
            ("has_sleep", lambda c: "asyncio.sleep" in c or "sleep(" in c),
            ("has_except", lambda c: "except" in c),
            ("has_raise", lambda c: "raise" in c),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
    # 3. Bug fix — off-by-one in pagination (real executor pattern)
    {
        "name": "fix_pagination_bug",
        "prompt": (
            "Fix the bug in this Python function. The function is supposed to "
            "return lines start_line through start_line+max_lines-1 (0-indexed), "
            "but it has an off-by-one error and also doesn't handle the case "
            "where start_line is past the end of the file:\n\n"
            "```python\n"
            "def read_lines(content: str, start_line: int = 0, max_lines: int = 200) -> str:\n"
            '    lines = content.split("\\n")\n'
            "    selected = lines[start_line:start_line + max_lines + 1]\n"
            '    return "\\n".join(selected)\n'
            "```\n\n"
            "Return ONLY the fixed function. No explanation."
        ),
        "checks": [
            ("has_def", lambda c: "def read_lines" in c),
            ("fixes_off_by_one", lambda c: "max_lines + 1" not in c),
            ("handles_bounds", lambda c: "len(lines)" in c or "min(" in c or "max(" in c or "if start_line" in c),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
    # 4. SQL query builder (sqlite_store pattern)
    {
        "name": "sql_query_builder",
        "prompt": (
            "Write a Python function called `build_memory_query` that builds a "
            "SQL SELECT query for a 'memories' table. It takes optional keyword "
            "arguments: session_id (int), is_archived (bool), memory_type (str), "
            "min_rank (int), limit (int, default 100). "
            "Build the WHERE clause dynamically based on which arguments are not None. "
            "Return a tuple of (query_string, params_list). "
            "Use parameterized queries (? placeholders) to prevent SQL injection. "
            "Only output the code, no explanation."
        ),
        "checks": [
            ("has_def", lambda c: "def build_memory_query" in c),
            ("uses_params", lambda c: "?" in c),
            ("has_where", lambda c: "WHERE" in c or "where" in c),
            ("returns_tuple", lambda c: "return" in c and ("tuple" in c or ", params" in c or ", args" in c or "(query" in c)),
            ("dynamic_build", lambda c: "append" in c or "+=" in c or "join" in c or "conditions" in c or "clauses" in c),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
    # 5. Batch processor with error isolation (memory pipeline pattern)
    {
        "name": "batch_processor",
        "prompt": (
            "Write an async Python function called `process_batch` that takes a "
            "list of items and an async callable `processor`. It should process "
            "items in batches of `batch_size` (default 10). Each item should be "
            "processed individually within the batch — if one item fails, log the "
            "error and continue with the next (don't abort the batch). "
            "Return a dict with keys 'processed' (int), 'failed' (int), "
            "and 'errors' (list of error strings). "
            "Only output the code, no explanation."
        ),
        "checks": [
            ("has_async_def", lambda c: "async def process_batch" in c),
            ("has_batch_logic", lambda c: "batch_size" in c and ("range(" in c or "for" in c)),
            ("has_error_isolation", lambda c: "try" in c and "except" in c and "continue" in c),
            ("tracks_results", lambda c: "processed" in c and "failed" in c),
            ("has_await", lambda c: "await" in c),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
    # 6. Config parser with env var expansion (config.py pattern)
    {
        "name": "config_env_vars",
        "prompt": (
            "Write a Python function called `resolve_env_vars` that takes a string "
            "value and replaces any ${ENV_VAR} patterns with the actual environment "
            "variable values using os.environ. If the env var doesn't exist, leave "
            "the ${...} pattern unchanged. Handle multiple ${...} patterns in a "
            "single string. Return the resolved string. "
            "Only output the code, no explanation."
        ),
        "checks": [
            ("has_def", lambda c: "def resolve_env_vars" in c),
            ("uses_os_environ", lambda c: "os.environ" in c or "os.getenv" in c),
            ("has_pattern_match", lambda c: "${" in c or "re." in c or "replace" in c),
            ("has_return", lambda c: "return" in c),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
    # 7. LRU cache with max size (response cache pattern)
    {
        "name": "lru_cache",
        "prompt": (
            "Write a Python class called `LRUCache` that implements a "
            "least-recently-used cache with a configurable max_size (default 100). "
            "It should have methods: `get(key)` that returns the value or None, "
            "`put(key, value)` that adds/updates an entry and evicts the oldest "
            "if over max_size, and `__len__` that returns current size. "
            "Use OrderedDict from collections. "
            "Only output the code, no explanation."
        ),
        "checks": [
            ("has_class", lambda c: "class LRUCache" in c),
            ("uses_ordered_dict", lambda c: "OrderedDict" in c),
            ("has_get", lambda c: "def get" in c),
            ("has_put", lambda c: "def put" in c),
            ("has_len", lambda c: "def __len__" in c),
            ("has_eviction", lambda c: "pop" in c or "del " in c),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
    # 8. Refactor — extract method (executor refactoring task)
    {
        "name": "refactor_extract",
        "prompt": (
            "Refactor this function by extracting the scoring logic into a "
            "separate helper function called `_calculate_score`:\n\n"
            "```python\n"
            "def process_memory(content: str, rank: int, importance: float, tags: list[str]) -> dict:\n"
            "    summary = content[:100] if len(content) > 100 else content\n"
            "    # Scoring logic that should be extracted\n"
            "    score = rank * 0.4 + importance * 0.3\n"
            "    if len(tags) >= 5:\n"
            "        score += 0.1\n"
            "    if rank >= 4:\n"
            "        score += 0.2\n"
            "    score = min(score, 1.0)\n"
            "    return {'summary': summary, 'score': score, 'tags': tags}\n"
            "```\n\n"
            "Return both functions. Only output the code, no explanation."
        ),
        "checks": [
            ("has_helper", lambda c: "def _calculate_score" in c),
            ("has_main", lambda c: "def process_memory" in c),
            ("calls_helper", lambda c: "_calculate_score(" in c),
            ("helper_has_params", lambda c: "rank" in c.split("def _calculate_score")[1].split(")")[0] if "def _calculate_score" in c else False),
            ("valid_syntax", lambda c: _check_syntax(c)),
        ],
    },
]


def _check_syntax(code: str) -> bool:
    """Check if code is valid Python syntax."""
    # Strip markdown code fences if present
    cleaned = code.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```python) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        ast.parse(cleaned)
        return True
    except SyntaxError:
        return False


def _extract_code(raw: str) -> str:
    """Extract code from LLM response, stripping markdown fences and explanation."""
    text = raw.strip()
    # If wrapped in code fence, extract
    if "```" in text:
        blocks = text.split("```")
        for i in range(1, len(blocks), 2):
            block = blocks[i]
            # Remove language identifier (python, py, etc.)
            lines = block.split("\n")
            if lines and lines[0].strip().lower() in ("python", "py", ""):
                lines = lines[1:]
            return "\n".join(lines).strip()
    # Otherwise return as-is (model might have just output code)
    return text


class CodingSuite(BenchmarkSuite):
    name = "coding"
    description = "Code generation quality on real-world Python patterns"
    task_types = ["coding"]
    needs_db = False
    needs_router = True
    quick_samples = 0  # fixed test set
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
        from blipshell.llm.router import TaskType

        times = []
        total_checks = 0
        passed_checks = 0
        syntax_pass = 0
        syntax_total = 0
        errors = 0
        task_details: dict[str, dict] = {}

        for task in CODE_TASKS:
            name = task["name"]
            prompt = task["prompt"]
            checks = task["checks"]

            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.CODING, prompt, think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                code = _extract_code(raw)
                task_passed = 0
                task_total = len(checks)
                task_results = {}

                for check_name, check_fn in checks:
                    try:
                        result = check_fn(code)
                        task_results[check_name] = result
                        if result:
                            task_passed += 1
                            passed_checks += 1
                        if check_name == "valid_syntax":
                            syntax_total += 1
                            if result:
                                syntax_pass += 1
                    except Exception:
                        task_results[check_name] = False
                    total_checks += 1

                task_details[name] = {
                    "passed": task_passed,
                    "total": task_total,
                    "time_s": round(elapsed, 2),
                    "checks": task_results,
                }
            except Exception as e:
                logger.debug("coding task %s error: %s", name, e)
                errors += 1
                total_checks += len(checks)
                task_details[name] = {"error": str(e)}

        avg_speed = sum(times) / len(times) if times else 0
        quality = passed_checks / total_checks if total_checks else 0

        scores = [
            TaskScore(
                task_name="code_generation",
                quality=round(quality, 3),
                speed_s=round(avg_speed, 2),
                samples=len(CODE_TASKS),
                errors=errors,
                detail={
                    "checks_passed": passed_checks,
                    "total_checks": total_checks,
                    "syntax_valid": f"{syntax_pass}/{syntax_total}",
                    "tasks": task_details,
                },
            ),
        ]

        total_time = sum(times)
        return SuiteResult(
            suite_name=self.name, model=model,
            scores=scores, elapsed_s=round(total_time, 1),
        )
