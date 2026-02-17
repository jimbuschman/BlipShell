"""Benchmark LLM models for reasoning, coding, and tool-calling tasks.

Compares cloud reasoning/coding models side-by-side on analytical thinking,
code generation/review, and tool selection. Prints Rich comparison tables
and saves results to JSON.

Usage:
    python tests/benchmark_reasoning.py                                    # run all default models
    python tests/benchmark_reasoning.py glm-5:cloud qwen3-coder:480b-cloud  # run only specified models
    python -m pytest tests/benchmark_reasoning.py -s                       # run all via pytest
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import generate_plan, reflect_on_response
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
BENCHMARK_MODELS = [
    "glm-5:cloud",
    "qwen3-coder:480b-cloud",
    "gpt-oss:120b-cloud",
]

OLLAMA_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Test data — Reasoning
# ---------------------------------------------------------------------------
REASONING_TESTS = [
    {
        "name": "plan_generation",
        "description": "Generate a plan for a multi-step task",
        "prompt_fn": "generate_plan",
        "input": "Add retry logic with exponential backoff to the HTTP client, write tests for it, and update the README",
    },
    {
        "name": "technical_analysis",
        "description": "Analyze a technical problem",
        "system": "You are a helpful technical assistant. Be concise and specific.",
        "prompt": "Why might an ESP32 with a MAX98357 I2S DAC produce noise or garbage audio? List the 3 most likely causes and how to debug each one.",
    },
    {
        "name": "self_reflection",
        "description": "Catch errors in a flawed response",
        "prompt_fn": "reflect_on_response",
        "user_message": "How do I connect an SSD1306 OLED to an ESP32?",
        "response": "Connect VCC to 3.3V, GND to GND, SDA to GPIO 21, and SCL to GPIO 22. Use the Adafruit_SSD1306 library. Call display.begin(SSD1306_SWITCHCAPVCC, 0x3D) to initialize.",
        # Error: default I2C address is 0x3C not 0x3D for most SSD1306
    },
]

REASONING_LABELS = [t["name"] for t in REASONING_TESTS]

# ---------------------------------------------------------------------------
# Test data — Coding
# ---------------------------------------------------------------------------
CODING_TESTS = [
    {
        "name": "bug_spotting",
        "description": "Find bugs in a function",
        "system": "You are a code reviewer. Find bugs in the following code. List each bug concisely.",
        "prompt": (
            "Find the bugs in this Python function:\n\n"
            "def retry_with_backoff(func, max_retries=3, base_delay=1.0):\n"
            "    for attempt in range(max_retries):\n"
            "        try:\n"
            "            return func()\n"
            "        except Exception as e:\n"
            "            if attempt == max_retries:\n"
            "                raise\n"
            "            delay = base_delay * (2 ** attempt)\n"
            "            time.sleep(delay)\n"
        ),
        # Bugs: 1) off-by-one: attempt never equals max_retries (range stops before)
        #        2) missing import time
        #        3) never raises on final attempt (falls through silently)
    },
    {
        "name": "code_generation",
        "description": "Generate a small function",
        "system": "Write only the function code. No explanation, no markdown fences.",
        "prompt": "Write a Python function `parse_rank(text: str) -> int` that extracts a rank (1-5) from LLM output text. It should find the first digit 1-5 in the text and return it, defaulting to 3 if none found.",
    },
    {
        "name": "code_review",
        "description": "Review working but messy code",
        "system": "You are a senior developer reviewing code. Give 3-5 actionable suggestions. Be concise.",
        "prompt": (
            "Review this code:\n\n"
            "async def process_memory(text, router, db):\n"
            "    s, u = summarize_memory(text)\n"
            '    summary = await router.generate("summarization", u, system=s, think=False)\n'
            "    s2, u2 = rank_memory(text)\n"
            '    r = await router.generate("ranking", u2, system=s2, think=False)\n'
            "    rank = 3\n"
            "    for c in r:\n"
            "        if c.isdigit() and 1 <= int(c) <= 5:\n"
            "            rank = int(c)\n"
            "            break\n"
            "    s3, u3 = ask_importance(text)\n"
            '    imp = await router.generate("ranking", u3, system=s3, think=False)\n'
            "    importance = 0.3\n"
            "    try:\n"
            "        importance = float(imp.strip())\n"
            "    except:\n"
            "        pass\n"
            "    await db.update(summary=summary, rank=rank, importance=importance)\n"
            '    return {"summary": summary, "rank": rank, "importance": importance}\n'
        ),
    },
]

CODING_LABELS = [t["name"] for t in CODING_TESTS]

# ---------------------------------------------------------------------------
# Test data — Tool calling
# ---------------------------------------------------------------------------
MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Command to run"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Search past conversations and memories",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query for memories"}},
                "required": ["query"],
            },
        },
    },
]

TOOL_CALLING_TESTS = [
    {
        "name": "web_search",
        "description": "Should call web_search",
        "message": "Search the web for ESP32 MAX98357 I2S wiring diagram",
        "expected_tool": "web_search",
    },
    {
        "name": "read_file",
        "description": "Should call read_file",
        "message": "Read the contents of worker.py",
        "expected_tool": "read_file",
    },
    {
        "name": "memory_recall",
        "description": "Should call search_memories",
        "message": "What did we discuss last time about the desk robot connectors?",
        "expected_tool": "search_memories",
    },
    {
        "name": "tool_selection",
        "description": "Should pick the right tool (not run_command)",
        "message": "Look up the current price of a Raspberry Pi 5",
        "expected_tool": "web_search",
    },
]

TOOL_LABELS = [t["name"] for t in TOOL_CALLING_TESTS]

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_model_spec(spec: str) -> tuple[str, dict]:
    """Parse a model spec like 'gpt-oss:latest/low' into (model_name, extra_options).

    Supports reasoning_effort suffix: /low, /medium, /high
    Returns (model_name, options_dict) where options_dict may contain reasoning_effort.
    """
    if "/" in spec:
        parts = spec.rsplit("/", 1)
        model_name = parts[0]
        effort = parts[1].lower()
        if effort in ("low", "medium", "high"):
            return model_name, {"reasoning_effort": effort}
        # Not a known effort level — treat whole string as model name
    return spec, {}


def make_router(model_name: str, timeout: float = 300.0) -> LLMRouter:
    """Create a LLMRouter that routes ALL task types to the given model."""
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
        roles=["reasoning", "tool_calling", "coding", "summarization", "ranking", "importance", "embedding"],
        priority=1,
        max_concurrent=1,
    )
    llm_config = LLMConfig(timeout=timeout)
    endpoint_manager = EndpointManager([endpoint_cfg], llm_config)
    return LLMRouter(models, endpoint_manager)


async def _generate_with_options(
    router: LLMRouter,
    task_type: str,
    prompt: str,
    system: str | None = None,
    extra_options: dict | None = None,
) -> str:
    """Call router.generate(), merging extra_options into the Ollama options dict.

    When extra_options contains keys like reasoning_effort, we go through the
    client directly so we can pass them in the options dict.
    """
    if not extra_options:
        return await router.generate(task_type, prompt, system=system, think=False)

    # Bypass router to inject extra options
    model, client = await router.get_model_and_client(task_type)
    if not client:
        raise RuntimeError(f"No available endpoint for task type: {task_type}")

    gen_kwargs: dict = {"options": {**extra_options}}
    # Don't pass think=False when using reasoning_effort (they conflict)
    if "reasoning_effort" not in extra_options:
        gen_kwargs["think"] = False

    result = await client.generate(prompt=prompt, model=model, system=system, **gen_kwargs)
    return result


def extract_response(response) -> tuple[str, list | None]:
    """Extract content and tool_calls from an Ollama response.

    Handles both dict responses (old ollama) and object responses (ollama 0.4+).
    """
    # Try object attribute access first (ollama 0.4+)
    msg = getattr(response, "message", None)
    if msg is not None:
        content = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", None)
        return content, tool_calls

    # Fallback to dict access (older ollama)
    if isinstance(response, dict):
        msg = response.get("message", {})
        return msg.get("content", ""), msg.get("tool_calls", None)

    return "", None


def extract_tool_call_info(tc) -> tuple[str, dict]:
    """Extract name and arguments from a tool call object or dict."""
    # Object access (ollama 0.4+)
    fn = getattr(tc, "function", None)
    if fn is not None:
        name = getattr(fn, "name", "") or ""
        args = getattr(fn, "arguments", {}) or {}
        return name, args

    # Dict access
    if isinstance(tc, dict):
        fn = tc.get("function", {})
        return fn.get("name", ""), fn.get("arguments", {})

    return "", {}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def benchmark_reasoning(router: LLMRouter, extra_options: dict | None = None) -> list[dict]:
    results = []
    for test in REASONING_TESTS:
        start = time.perf_counter()
        try:
            if test.get("prompt_fn") == "generate_plan":
                prompt = generate_plan(test["input"])
                response = await _generate_with_options(
                    router, TaskType.REASONING, prompt, extra_options=extra_options,
                )
            elif test.get("prompt_fn") == "reflect_on_response":
                prompt = reflect_on_response(test["user_message"], test["response"])
                response = await _generate_with_options(
                    router, TaskType.REASONING, prompt, extra_options=extra_options,
                )
            else:
                response = await _generate_with_options(
                    router, TaskType.REASONING, test["prompt"],
                    system=test.get("system"), extra_options=extra_options,
                )
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.perf_counter() - start
        results.append({"response": response, "time": round(elapsed, 2)})
        await asyncio.sleep(0.1)
    return results


async def benchmark_coding(router: LLMRouter, extra_options: dict | None = None) -> list[dict]:
    results = []
    for test in CODING_TESTS:
        start = time.perf_counter()
        try:
            response = await _generate_with_options(
                router, TaskType.CODING, test["prompt"],
                system=test.get("system"), extra_options=extra_options,
            )
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.perf_counter() - start
        results.append({"response": response, "time": round(elapsed, 2)})
        await asyncio.sleep(0.1)
    return results


async def benchmark_tool_calling(router: LLMRouter, extra_options: dict | None = None) -> list[dict]:
    model, client = await router.get_model_and_client(TaskType.TOOL_CALLING)
    if not client:
        return [{"error": "No client available", "time": 0}] * len(TOOL_CALLING_TESTS)

    results = []
    for test in TOOL_CALLING_TESTS:
        messages = [{"role": "user", "content": test["message"]}]
        start = time.perf_counter()
        chat_kwargs = {}
        if extra_options:
            chat_kwargs["options"] = {**extra_options}
        try:
            response = await client.chat(messages=messages, model=model, tools=MOCK_TOOLS, **chat_kwargs)
            content, tool_calls = extract_response(response)

            called_tools = []
            if tool_calls:
                for tc in tool_calls:
                    name, args = extract_tool_call_info(tc)
                    called_tools.append({"name": name, "args": args})

            correct = any(t["name"] == test["expected_tool"] for t in called_tools)
            result = {
                "content": content,
                "tool_calls": called_tools,
                "expected": test["expected_tool"],
                "correct": correct,
                "time": round(time.perf_counter() - start, 2),
            }
        except Exception as e:
            result = {
                "content": f"ERROR: {e}",
                "tool_calls": [],
                "expected": test["expected_tool"],
                "correct": False,
                "time": round(time.perf_counter() - start, 2),
            }
        results.append(result)
        await asyncio.sleep(0.1)
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_reasoning_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Reasoning", show_lines=True, expand=True)
    table.add_column("Model", style="cyan", width=24, no_wrap=True)
    for label in REASONING_LABELS:
        table.add_column(label, ratio=1)

    for model in models:
        row = [model]
        for i, label in enumerate(REASONING_LABELS):
            r = all_results[model]["reasoning"][i]
            cell = f"{escape(r['response'][:500])}\n[dim]({r['time']}s)[/dim]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def print_coding_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Coding", show_lines=True, expand=True)
    table.add_column("Model", style="cyan", width=24, no_wrap=True)
    for label in CODING_LABELS:
        table.add_column(label, ratio=1)

    for model in models:
        row = [model]
        for i, label in enumerate(CODING_LABELS):
            r = all_results[model]["coding"][i]
            cell = f"{escape(r['response'][:500])}\n[dim]({r['time']}s)[/dim]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def print_tool_calling_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Tool Calling", show_lines=True, expand=True)
    table.add_column("Model", style="cyan", width=24, no_wrap=True)
    for label in TOOL_LABELS:
        table.add_column(label, ratio=1)

    for model in models:
        row = [model]
        for i, label in enumerate(TOOL_LABELS):
            r = all_results[model]["tool_calling"][i]
            # Format tool calls
            if r.get("tool_calls"):
                tools_str = ", ".join(
                    f"{t['name']}({escape(json.dumps(t['args'], ensure_ascii=False)[:100])})"
                    for t in r["tool_calls"]
                )
            else:
                tools_str = "[dim]no tool called[/dim]"

            correct_mark = "[green]PASS[/green]" if r.get("correct") else "[red]FAIL[/red]"
            expected = r.get("expected", "?")
            content_preview = escape(r.get("content", "")[:100])

            cell = f"{correct_mark} (expected: {expected})\n{tools_str}"
            if content_preview:
                cell += f"\n[dim]{content_preview}[/dim]"
            cell += f"\n[dim]({r['time']}s)[/dim]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(models: list[str]):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "benchmark_reasoning_results.json"

    # Load existing results to merge with
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        existing = [m for m in models if m in all_results]
        if existing:
            console.print(f"[yellow]Loaded existing results for: {', '.join(all_results.keys())}[/yellow]")

    # Only run models that were requested
    console.print(f"[bold]Running benchmarks for: {', '.join(models)}[/bold]\n")

    for model_spec in models:
        model_name, extra_options = parse_model_spec(model_spec)
        # Use the full spec as the results key (e.g. "gpt-oss:latest/low")
        result_key = model_spec
        if extra_options:
            console.print(f"  [yellow]Extra options: {extra_options}[/yellow]")

        console.rule(f"[bold blue]Benchmarking: {result_key}")
        router = make_router(model_name)

        console.print("  [dim]Running reasoning tests...[/dim]")
        reasoning = await benchmark_reasoning(router, extra_options)

        console.print("  [dim]Running coding tests...[/dim]")
        coding = await benchmark_coding(router, extra_options)

        console.print("  [dim]Running tool calling tests...[/dim]")
        tool_calling = await benchmark_tool_calling(router, extra_options)

        all_results[result_key] = {
            "reasoning": reasoning,
            "coding": coding,
            "tool_calling": tool_calling,
        }
        console.print(f"  [green]Done with {result_key}[/green]\n")

    # Save merged results to JSON first (so data isn't lost if table rendering fails)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[bold]Results saved to {output_path} ({len(all_results)} models)[/bold]")

    # Print comparison tables (all results, including previously loaded)
    console.rule("[bold green]Results")
    print_reasoning_table(all_results)
    console.print()
    print_coding_table(all_results)
    console.print()
    print_tool_calling_table(all_results)


def test_benchmark():
    """Entry point for pytest -s."""
    asyncio.run(run_benchmark(BENCHMARK_MODELS))


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else BENCHMARK_MODELS
    asyncio.run(run_benchmark(models))
