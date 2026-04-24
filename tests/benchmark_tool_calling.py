"""Benchmark LLM models for tool-calling quality on interactive chat.

Tests cover:
- Clear tool selection (which tool, right args)
- Negative cases (when NOT to call any tool)
- Trap selection (right tool among similar ones)
- Argument quality (not just "did it call X" — did it pass sensible args)

Each test is scored:
- tool_correct: right tool picked (or correctly no tool for negatives)
- args_ok: argument values are sensible per test-specific checker

Results saved to data/tool_calling_benchmark_YYYY-MM-DD_HHMMSS.json
for cross-run comparison.

Usage:
    python tests/benchmark_tool_calling.py                      # default candidate list
    python tests/benchmark_tool_calling.py glm-5.1:cloud         # one or more specific models
    python tests/benchmark_tool_calling.py --models list.txt     # newline-separated file
    python tests/benchmark_tool_calling.py --compare prior.json  # diff vs prior run
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import httpx
import ollama
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()

OLLAMA_HOST = "http://localhost:11434"

DEFAULT_CANDIDATES = [
    # Cloud — new
    "kimi-k2.6:cloud",
    "deepseek-v4-flash:cloud",
    "gemma4:31b-cloud",
    "nemotron-3-super:cloud",
    "minimax-m2.7:cloud",
    "ministral-3:14b-cloud",
    "ministral-3:8b-cloud",
    # Cloud — previously benched / baselines
    "glm-5.1:cloud",
    "gpt-oss:120b-cloud",
    "qwen3.5:397b-cloud",
    "qwen3-coder:480b-cloud",
    "kimi-k2.5:cloud",
    "glm-4.7:cloud",
    "nemotron-3-nano:30b-cloud",
    "devstral-small-2:24b-cloud",
    "gemini-3-flash-preview:cloud",
    "minimax-m2.5:cloud",
    # Local — RTX 3060 12GB friendly (pull first with `ollama pull <name>`)
    "qwen3:14b",
    "gpt-oss:latest",
    "ministral-3:14b",
    "gemma4:e4b",
]

# ---------------------------------------------------------------------------
# Tool definitions — representative of BlipShell interactive chat use
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for facts, news, current events, pricing, or anything that changes over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the user's filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command on the user's local machine. Use for system info, running scripts, or local inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Search the user's past conversations, notes, and saved memories. Use for anything related to what the user previously said, did, or saved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for memories"},
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass
class ToolTest:
    id: str
    message: str
    expected_tool: Optional[str]  # None = should not call any tool
    description: str = ""
    argcheck: Optional[Callable[[dict], bool]] = None


def _has_any(args: dict, key: str, terms: list[str]) -> bool:
    v = str(args.get(key, "")).lower()
    return any(t in v for t in terms)


TESTS: list[ToolTest] = [
    # Clear tool selection
    ToolTest(
        id="web_search_factual",
        message="Search the web for current ESP32 MAX98357 I2S wiring diagrams",
        expected_tool="web_search",
        description="Explicit request to search the web",
        argcheck=lambda a: "esp32" in str(a.get("query", "")).lower(),
    ),
    ToolTest(
        id="read_file_explicit",
        message="Read the contents of config.yaml and tell me what's in it",
        expected_tool="read_file",
        description="File named explicitly",
        argcheck=lambda a: "config" in str(a.get("path", "")).lower() and "yaml" in str(a.get("path", "")).lower(),
    ),
    ToolTest(
        id="memory_recall_explicit",
        message="What did I tell you yesterday about the desk robot connectors?",
        expected_tool="search_memories",
        description="Personal history reference",
        argcheck=lambda a: _has_any(a, "query", ["desk", "robot", "connector"]),
    ),
    ToolTest(
        id="shell_system_info",
        message="Run `python --version` and tell me which Python is installed",
        expected_tool="run_command",
        description="Explicit shell command",
        argcheck=lambda a: "python" in str(a.get("command", "")).lower(),
    ),
    # Negative — should NOT call a tool
    ToolTest(
        id="casual_math",
        message="Quick question: what's 17 * 23?",
        expected_tool=None,
        description="Arithmetic — no tool needed",
    ),
    ToolTest(
        id="explanation_no_tool",
        message="In 2 sentences, explain what a Merkle tree is.",
        expected_tool=None,
        description="Pure explanation — no tool needed",
    ),
    ToolTest(
        id="greeting",
        message="hey, how's it going?",
        expected_tool=None,
        description="Casual greeting — no tool needed",
    ),
    # Ambiguity / judgment
    ToolTest(
        id="memory_vs_web_personal",
        message="Remind me what I was working on last week related to audio hardware",
        expected_tool="search_memories",
        description="Personal context — memory over web",
        argcheck=lambda a: _has_any(a, "query", ["audio"]),
    ),
    ToolTest(
        id="web_not_memory_current",
        message="What's the current stable release of Python?",
        expected_tool="web_search",
        description="Current external info — web not memory",
        argcheck=lambda a: "python" in str(a.get("query", "")).lower(),
    ),
    # Traps
    ToolTest(
        id="no_shell_for_web",
        message="Look up the current price of a Raspberry Pi 5",
        expected_tool="web_search",
        description="Price lookup — web, not run_command",
        argcheck=lambda a: _has_any(a, "query", ["raspberry", "pi"]),
    ),
    # Argument quality
    ToolTest(
        id="query_quality_full_context",
        message="Find me some good beginner tutorials for teaching Python to kids",
        expected_tool="web_search",
        description="Query should include useful context, not one word",
        argcheck=lambda a: len(str(a.get("query", ""))) > 15 and _has_any(a, "query", ["python", "kid", "beginner", "tutorial", "teach"]),
    ),
    ToolTest(
        id="specific_memory_query",
        message="Did I save anything about Docker volume permissions?",
        expected_tool="search_memories",
        description="Specific memory with key terms in args",
        argcheck=lambda a: _has_any(a, "query", ["docker", "volume", "permission"]),
    ),
    # Follow-through
    ToolTest(
        id="multi_clue_web",
        message="I've been trying to understand the differences between TCP and QUIC protocols for a project. Can you pull up some comparisons?",
        expected_tool="web_search",
        description="Comparison research — web search",
        argcheck=lambda a: _has_any(a, "query", ["tcp", "quic", "protocol", "comparison", "diff"]),
    ),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    test_id: str
    expected_tool: Optional[str]
    tool_called: Optional[str]
    tool_args: dict = field(default_factory=dict)
    tool_correct: bool = False
    args_ok: Optional[bool] = None  # None if no argcheck
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ModelResult:
    model: str
    elapsed_total_seconds: float = 0.0
    tests: list[TestResult] = field(default_factory=list)
    load_error: Optional[str] = None

    @property
    def tool_pass_rate(self) -> float:
        if not self.tests:
            return 0.0
        return sum(1 for t in self.tests if t.tool_correct) / len(self.tests)

    @property
    def args_pass_rate(self) -> float:
        relevant = [t for t in self.tests if t.args_ok is not None]
        if not relevant:
            return 0.0
        return sum(1 for t in relevant if t.args_ok) / len(relevant)


# ---------------------------------------------------------------------------
# Ollama response parsing
# ---------------------------------------------------------------------------

def _extract_tool_calls(response) -> list[tuple[str, dict]]:
    """Return list of (name, arguments) extracted from an Ollama response."""
    msg = getattr(response, "message", None)
    if msg is not None:
        tc_list = getattr(msg, "tool_calls", None) or []
    elif isinstance(response, dict):
        tc_list = response.get("message", {}).get("tool_calls") or []
    else:
        tc_list = []

    out = []
    for tc in tc_list:
        fn = getattr(tc, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "") or ""
            args = getattr(fn, "arguments", {}) or {}
        elif isinstance(tc, dict):
            fn_d = tc.get("function", {}) or {}
            name = fn_d.get("name", "") or ""
            args = fn_d.get("arguments", {}) or {}
        else:
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        out.append((name, args))
    return out


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

async def run_one_test(client: ollama.AsyncClient, model: str, test: ToolTest) -> TestResult:
    start = time.perf_counter()
    try:
        response = await client.chat(
            model=model,
            messages=[{"role": "user", "content": test.message}],
            tools=TOOLS,
            stream=False,
            options={"num_predict": 200},
        )
    except Exception as e:
        return TestResult(
            test_id=test.id,
            expected_tool=test.expected_tool,
            tool_called=None,
            elapsed_seconds=time.perf_counter() - start,
            error=str(e)[:300],
        )
    elapsed = time.perf_counter() - start

    calls = _extract_tool_calls(response)
    tool_called = calls[0][0] if calls else None
    tool_args = calls[0][1] if calls else {}

    # Scoring
    if test.expected_tool is None:
        tool_correct = (tool_called is None)
    else:
        tool_correct = (tool_called == test.expected_tool)

    args_ok: Optional[bool] = None
    if test.argcheck is not None and tool_correct and tool_args:
        try:
            args_ok = bool(test.argcheck(tool_args))
        except Exception:
            args_ok = False

    return TestResult(
        test_id=test.id,
        expected_tool=test.expected_tool,
        tool_called=tool_called,
        tool_args=tool_args,
        tool_correct=tool_correct,
        args_ok=args_ok,
        elapsed_seconds=elapsed,
    )


async def run_model(client: ollama.AsyncClient, model: str, tests: list[ToolTest]) -> ModelResult:
    mr = ModelResult(model=model)
    start = time.perf_counter()
    console.print(f"\n[bold cyan]{model}[/bold cyan]")
    for t in tests:
        r = await run_one_test(client, model, t)
        mr.tests.append(r)
        _render_test_line(t, r)
    mr.elapsed_total_seconds = time.perf_counter() - start
    console.print(
        f"  [dim]total {mr.elapsed_total_seconds:.1f}s, "
        f"tool pass {mr.tool_pass_rate:.0%}, args {mr.args_pass_rate:.0%}[/dim]"
    )
    return mr


def _render_test_line(test: ToolTest, result: TestResult) -> None:
    if result.error:
        console.print(f"  [red]{test.id:28}[/red] ERROR {result.error[:60]}")
        return
    if result.tool_correct:
        color = "green"
        marker = "ok  "
    else:
        color = "red"
        marker = "FAIL"
    args_note = ""
    if result.args_ok is False:
        args_note = " [yellow](args weak)[/yellow]"
    elif result.args_ok is True:
        args_note = " [dim](args ok)[/dim]"

    called = result.tool_called or "(none)"
    expected = test.expected_tool or "(none)"
    console.print(
        f"  [{color}]{marker}[/{color}] {test.id:28} "
        f"exp={expected:17} got={called:17} {result.elapsed_seconds:5.1f}s{args_note}"
    )


# ---------------------------------------------------------------------------
# Summary rendering + persistence
# ---------------------------------------------------------------------------

def render_summary(models: list[ModelResult]) -> None:
    rows = sorted(models, key=lambda m: m.tool_pass_rate, reverse=True)
    table = Table(title="Tool-Calling Benchmark Summary")
    table.add_column("Model", justify="left")
    table.add_column("Tool %", justify="right")
    table.add_column("Args %", justify="right")
    table.add_column("Avg time", justify="right")
    table.add_column("Errors", justify="right")

    for m in rows:
        if m.load_error:
            table.add_row(m.model, "-", "-", "-", f"[red]{m.load_error[:40]}[/red]")
            continue
        avg = sum(t.elapsed_seconds for t in m.tests) / max(len(m.tests), 1)
        errors = sum(1 for t in m.tests if t.error)
        table.add_row(
            m.model,
            f"{m.tool_pass_rate:.0%}",
            f"{m.args_pass_rate:.0%}" if any(t.args_ok is not None for t in m.tests) else "-",
            f"{avg:.1f}s",
            str(errors) if errors else "-",
        )
    console.print(table)


def save_results(models: list[ModelResult], out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out_path = out_dir / f"tool_calling_benchmark_{timestamp}.json"
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": [{"id": t.id, "message": t.message, "expected_tool": t.expected_tool,
                   "description": t.description} for t in TESTS],
        "models": [
            {
                "model": m.model,
                "elapsed_total_seconds": m.elapsed_total_seconds,
                "tool_pass_rate": m.tool_pass_rate,
                "args_pass_rate": m.args_pass_rate,
                "load_error": m.load_error,
                "tests": [asdict(t) for t in m.tests],
            }
            for m in models
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"\n[dim]Saved to {out_path}[/dim]")
    return out_path


def compare_runs(prior_path: Path, current: list[ModelResult]) -> None:
    with open(prior_path) as f:
        prior = json.load(f)
    prior_by_model = {m["model"]: m for m in prior.get("models", [])}

    console.print(f"\n[bold]Δ vs {prior_path.name}[/bold]")
    table = Table()
    table.add_column("Model")
    table.add_column("Prior tool%", justify="right")
    table.add_column("New tool%", justify="right")
    table.add_column("Δ", justify="right")

    for m in current:
        p = prior_by_model.get(m.model)
        if not p:
            table.add_row(m.model, "-", f"{m.tool_pass_rate:.0%}", "[dim](new)[/dim]")
            continue
        prior_rate = p.get("tool_pass_rate", 0.0)
        delta = m.tool_pass_rate - prior_rate
        color = "green" if delta > 0 else "red" if delta < 0 else "white"
        table.add_row(m.model, f"{prior_rate:.0%}", f"{m.tool_pass_rate:.0%}",
                      f"[{color}]{delta:+.0%}[/{color}]")
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all(host: str, models: list[str]) -> list[ModelResult]:
    client = ollama.AsyncClient(host=host, timeout=httpx.Timeout(180.0, connect=10.0))
    results: list[ModelResult] = []
    for model in models:
        try:
            results.append(await run_model(client, model, TESTS))
        except Exception as e:
            mr = ModelResult(model=model, load_error=str(e)[:200])
            console.print(f"[red]Load failure on {model}: {e}[/red]")
            results.append(mr)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark tool-calling quality")
    parser.add_argument("models", nargs="*", help="Specific models to benchmark (overrides defaults)")
    parser.add_argument("--models-file", help="Path to newline-separated model list")
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument("--output", default="data", help="Output dir for JSON results")
    parser.add_argument("--compare", help="Path to prior results JSON to diff against")
    args = parser.parse_args()

    if args.models:
        models = args.models
    elif args.models_file:
        with open(args.models_file) as f:
            models = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        models = DEFAULT_CANDIDATES

    console.print(f"[cyan]Benchmarking {len(models)} models on {len(TESTS)} tool-calling tests[/cyan]")

    results = asyncio.run(run_all(args.host, models))
    render_summary(results)
    out_path = save_results(results, Path(args.output))

    if args.compare:
        compare_runs(Path(args.compare), results)

    console.print(f"\n[green]Done.[/green] Re-run anytime with --compare {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
