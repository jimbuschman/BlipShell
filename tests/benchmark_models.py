"""Benchmark LLM models for BlipShell memory pipeline tasks.

Runs summarization, ranking, importance, and lesson extraction across
multiple models side-by-side, then prints Rich comparison tables and
saves results to JSON.

Usage:
    python tests/benchmark_models.py                          # run all default models
    python tests/benchmark_models.py phi4:14b qwen3:14b       # run only specified models
    python -m pytest tests/benchmark_models.py -s             # run all via pytest
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
from blipshell.llm.prompts import (
    ask_importance,
    detect_contradiction,
    extract_entities,
    extract_lesson,
    rank_and_importance,
    rank_memory,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.entity_extractor import EntityExtractor
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
BENCHMARK_MODELS = [
    "gemma3:4b",
    "gpt-oss:latest",
    "llama3.1:8b",
    "qwen2.5:14b",
    "phi4:latest",
    "mistral-small3.2:latest",
    "qwen3:14b",
    "dolphin3:latest",
    "olmo2:latest",
    "glm-5:cloud",
    "gpt-oss:120b-cloud",
]

OLLAMA_URL = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
TEST_MESSAGES = [
    # Greeting/filler - should be rank 1, low importance, SKIP summary
    {"role": "user", "content": "hey"},
    # Short filler
    {"role": "user", "content": "ok thanks"},
    # System noise - should be rank 1, SKIP
    {
        "role": "user",
        "content": (
            "[System: <important_rules>\nYou are in agent mode.\n"
            "If you need to use multiple tools...</important_rules>]\ntest"
        ),
    },
    # Technical question - should be rank 3-4, medium importance
    {
        "role": "user",
        "content": (
            "I'm using a MAX98357 with the esp32 and the sound it produces "
            "is terrible. Mostly sounds like just noise/garbage."
        ),
    },
    # Personal fact - should be rank 4-5, high importance
    {
        "role": "user",
        "content": (
            "for some reason my daughters laptop when we got in dec only "
            "wants to load 10 blocks in minecraft, thats pretty terrbile"
        ),
    },
    # Code review (assistant) - should be rank 3-4, medium-high importance
    {
        "role": "assistant",
        "content": (
            "Now I can see the worker.py file! Let me provide a code review:\n\n"
            "## Code Review: worker.py\n\n"
            "### Issues Found:\n"
            "1. Missing stop() call on shutdown\n"
            "2. No retry logic for failed HTTP requests\n"
            "3. No model fallback mechanism"
        ),
    },
    # Decision/preference - should be rank 4-5, high importance
    {
        "role": "user",
        "content": (
            "yeah i think ill go with the two-module design for the desk robot. "
            "main board plus a sidecar for the sensors. JST connectors between them."
        ),
    },
    # Casual/minor - should be rank 2, low importance
    {"role": "user", "content": "sanding paint when its still tacky?"},
]

TEST_CONVERSATIONS = [
    # Session for lesson extraction
    [
        {
            "role": "user",
            "content": "man this desk robot is just being a pain. the connectors are too loose.",
        },
        {
            "role": "assistant",
            "content": (
                "Yeah that's frustrating. For the ESP32 connections, JST-PH 2-pin "
                "connectors are much more reliable than dupont wires."
            ),
        },
        {
            "role": "user",
            "content": "yeah ill try those. also the speaker got really hot during testing",
        },
        {
            "role": "assistant",
            "content": (
                "That's a critical warning sign - could be a short or wrong impedance "
                "speaker. Check the speaker ohms matches what the MAX98357 expects."
            ),
        },
    ],
]

# Short labels for table rows
MESSAGE_LABELS = [
    "hey (greeting)",
    "ok thanks (filler)",
    "system noise",
    "ESP32 audio issue",
    "daughter's minecraft",
    "code review (asst)",
    "desk robot decision",
    "sanding paint",
]

# ---------------------------------------------------------------------------
# Test data — Entity Extraction
# ---------------------------------------------------------------------------
ENTITY_TEST_SUMMARIES = [
    "User asked about Python performance tuning for data analysis.",
    "User decided to use a two-module design for the desk robot with JST connectors.",
    "Assistant explained how to configure Ollama with GPU acceleration.",
    "User's daughter has a Minecraft performance issue on her HP laptop.",
    "User said hello.",
]

ENTITY_LABELS = [
    "python perf tuning",
    "desk robot design",
    "ollama gpu config",
    "daughter minecraft",
    "hello (expect NONE)",
]

# ---------------------------------------------------------------------------
# Test data — Contradiction Detection
# ---------------------------------------------------------------------------
CONTRADICTION_PAIRS = [
    ("User prefers dark mode", "User prefers light mode", True),
    ("User uses Windows 10", "User upgraded to Windows 11", True),
    ("User likes Python", "User dislikes Python", True),
    ("User likes coffee", "User likes tea", False),
    ("User has a cat named Luna", "User works at Acme", False),
    ("User knows Python", "User also knows Rust", False),
]

CONTRADICTION_LABELS = [
    "dark/light mode",
    "Win10/Win11",
    "likes/dislikes Python",
    "coffee & tea",
    "cat & job",
    "Python & Rust",
]

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_router(model_name: str) -> LLMRouter:
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
    endpoint_manager = EndpointManager([endpoint_cfg], LLMConfig())
    return LLMRouter(models, endpoint_manager)


def build_conversation_text(messages: list[dict]) -> str:
    """Format messages into User:/Assistant: conversation text."""
    parts = []
    for m in messages:
        label = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{label}: {m['content']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def benchmark_summarization(router: LLMRouter) -> list[dict]:
    results = []
    for msg in TEST_MESSAGES:
        sys_prompt, user_prompt = summarize_memory(msg["content"])
        start = time.perf_counter()
        try:
            response = await router.generate(
                TaskType.SUMMARIZATION, user_prompt, system=sys_prompt, think=False,
            )
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.perf_counter() - start
        results.append({"response": response, "time": round(elapsed, 2)})
        await asyncio.sleep(0.1)
    return results


async def benchmark_ranking(router: LLMRouter) -> list[dict]:
    results = []
    for msg in TEST_MESSAGES:
        sys_prompt, user_prompt = rank_memory(msg["content"])
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.RANKING, user_prompt, system=sys_prompt, think=False,
            )
            rank = MemoryProcessor._parse_rank(raw)
        except Exception as e:
            raw = f"ERROR: {e}"
            rank = -1
        elapsed = time.perf_counter() - start
        results.append({"raw": raw, "parsed": rank, "time": round(elapsed, 2)})
        await asyncio.sleep(0.1)
    return results


async def benchmark_importance(router: LLMRouter) -> list[dict]:
    results = []
    for msg in TEST_MESSAGES:
        sys_prompt, user_prompt = ask_importance(msg["content"])
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.IMPORTANCE, user_prompt, system=sys_prompt, think=False,
            )
            score = MemoryProcessor._parse_float(raw, default=0.3)
        except Exception as e:
            raw = f"ERROR: {e}"
            score = -1.0
        elapsed = time.perf_counter() - start
        results.append({"raw": raw, "parsed": score, "time": round(elapsed, 2)})
        await asyncio.sleep(0.1)
    return results


async def benchmark_lessons(router: LLMRouter) -> list[dict]:
    results = []
    for conv in TEST_CONVERSATIONS:
        text = build_conversation_text(conv)
        sys_prompt, user_prompt = extract_lesson(text)
        start = time.perf_counter()
        try:
            response = await router.generate(
                TaskType.REASONING, user_prompt, system=sys_prompt, think=False,
            )
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.perf_counter() - start
        results.append({"response": response, "time": round(elapsed, 2)})
        await asyncio.sleep(0.1)
    return results


async def benchmark_entity_extraction(router: LLMRouter) -> list[dict]:
    extractor = EntityExtractor.__new__(EntityExtractor)  # only need _parse_triples
    results = []
    for summary in ENTITY_TEST_SUMMARIES:
        sys_prompt, user_prompt = extract_entities(summary)
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.REASONING, user_prompt, system=sys_prompt, think=False,
            )
            triples = extractor._parse_triples(raw)
            entities = sorted({t[0].lower() for t in triples} | {t[2].lower() for t in triples})
        except Exception as e:
            raw = f"ERROR: {e}"
            triples = []
            entities = []
        elapsed = time.perf_counter() - start
        results.append({
            "raw": raw,
            "triple_count": len(triples),
            "entities": entities,
            "time": round(elapsed, 2),
        })
        await asyncio.sleep(0.1)
    return results


async def benchmark_contradiction(router: LLMRouter) -> list[dict]:
    results = []
    for new_mem, existing_mem, expected_yes in CONTRADICTION_PAIRS:
        sys_prompt, user_prompt = detect_contradiction(new_mem, existing_mem)
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.REASONING, user_prompt, system=sys_prompt, think=False,
            )
            answer = raw.strip().upper()
            if answer.startswith("YES"):
                parsed = "YES"
            elif answer.startswith("NO"):
                parsed = "NO"
            else:
                parsed = "INVALID"
            expected = "YES" if expected_yes else "NO"
            correct = parsed == expected
        except Exception as e:
            raw = f"ERROR: {e}"
            parsed = "ERROR"
            expected = "YES" if expected_yes else "NO"
            correct = False
        elapsed = time.perf_counter() - start
        results.append({
            "raw": raw,
            "parsed": parsed,
            "expected": expected,
            "correct": correct,
            "time": round(elapsed, 2),
        })
        await asyncio.sleep(0.1)
    return results


async def benchmark_rank_and_importance(router: LLMRouter) -> list[dict]:
    results = []
    for msg in TEST_MESSAGES:
        sys_prompt, user_prompt = rank_and_importance(msg["content"])
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.RANKING, user_prompt, system=sys_prompt, think=False,
            )
            rank, importance = MemoryProcessor._parse_rank_and_importance(raw)
        except Exception as e:
            raw = f"ERROR: {e}"
            rank = -1
            importance = -1.0
        elapsed = time.perf_counter() - start
        results.append({
            "raw": raw,
            "rank": rank,
            "importance": importance,
            "time": round(elapsed, 2),
        })
        await asyncio.sleep(0.1)
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Summarization", show_lines=True, expand=True)
    table.add_column("Model", style="cyan", width=20, no_wrap=True)
    for label in MESSAGE_LABELS:
        table.add_column(label, ratio=1)

    for model in models:
        row = [model]
        for i in range(len(MESSAGE_LABELS)):
            r = all_results[model]["summarization"][i]
            cell = f"{escape(r['response'])}\n[dim]({r['time']}s)[/dim]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def print_ranking_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Ranking (1-5)", show_lines=True)
    table.add_column("Model", style="cyan", width=20, no_wrap=True)
    for label in MESSAGE_LABELS:
        table.add_column(label, justify="center", width=10)

    for model in models:
        row = [model]
        for i in range(len(MESSAGE_LABELS)):
            r = all_results[model]["ranking"][i]
            row.append(f"[bold]{r['parsed']}[/bold]\n[dim]{r['time']}s[/dim]")
        table.add_row(*row)

    console.print(table)


def print_importance_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Importance (0.0-1.0)", show_lines=True)
    table.add_column("Model", style="cyan", width=20, no_wrap=True)
    for label in MESSAGE_LABELS:
        table.add_column(label, justify="center", width=10)

    for model in models:
        row = [model]
        for i in range(len(MESSAGE_LABELS)):
            r = all_results[model]["importance"][i]
            row.append(f"[bold]{r['parsed']}[/bold]\n[dim]{r['time']}s[/dim]")
        table.add_row(*row)

    console.print(table)


def print_lessons_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Lesson Extraction", show_lines=True, expand=True)
    table.add_column("Model", style="cyan", width=20, no_wrap=True)
    for i in range(len(TEST_CONVERSATIONS)):
        table.add_column(f"Conv {i + 1}", ratio=1)

    for model in models:
        row = [model]
        for i in range(len(TEST_CONVERSATIONS)):
            r = all_results[model]["lessons"][i]
            cell = f"{escape(r['response'])}\n[dim]({r['time']}s)[/dim]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def print_entity_extraction_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Entity Extraction", show_lines=True, expand=True)
    table.add_column("Model", style="cyan", width=20, no_wrap=True)
    for label in ENTITY_LABELS:
        table.add_column(label, ratio=1)

    for model in models:
        if "entity_extraction" not in all_results[model]:
            continue
        row = [model]
        for i in range(len(ENTITY_LABELS)):
            r = all_results[model]["entity_extraction"][i]
            ents = ", ".join(r["entities"]) if r["entities"] else "(none)"
            cell = (
                f"[bold]{r['triple_count']} triples[/bold]\n"
                f"{escape(ents)}\n"
                f"[dim]({r['time']}s)[/dim]"
            )
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def print_contradiction_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Contradiction Detection", show_lines=True)
    table.add_column("Model", style="cyan", width=20, no_wrap=True)
    for label in CONTRADICTION_LABELS:
        table.add_column(label, justify="center", width=14)

    for model in models:
        if "contradiction" not in all_results[model]:
            continue
        row = [model]
        for i in range(len(CONTRADICTION_LABELS)):
            r = all_results[model]["contradiction"][i]
            if r["correct"]:
                mark = f"[green]{r['parsed']}[/green]"
            else:
                mark = f"[red]{r['parsed']}[/red]"
            cell = f"{mark} (exp: {r['expected']})\n[dim]{r['time']}s[/dim]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def print_rank_and_importance_table(all_results: dict):
    models = list(all_results.keys())
    table = Table(title="Combined Rank + Importance", show_lines=True)
    table.add_column("Model", style="cyan", width=20, no_wrap=True)
    for label in MESSAGE_LABELS:
        table.add_column(label, justify="center", width=12)

    for model in models:
        if "rank_and_importance" not in all_results[model]:
            continue
        row = [model]
        for i in range(len(MESSAGE_LABELS)):
            r = all_results[model]["rank_and_importance"][i]
            cell = (
                f"R=[bold]{r['rank']}[/bold] I=[bold]{r['importance']}[/bold]\n"
                f"[dim]{r['time']}s[/dim]"
            )
            row.append(cell)
        table.add_row(*row)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(models: list[str]):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "benchmark_results.json"

    # Load existing results to merge with
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        existing = [m for m in models if m in all_results]
        if existing:
            console.print(f"[yellow]Loaded existing results for: {', '.join(all_results.keys())}[/yellow]")

    # Only run models that were requested
    models_to_run = models
    console.print(f"[bold]Running benchmarks for: {', '.join(models_to_run)}[/bold]\n")

    for model in models_to_run:
        console.rule(f"[bold blue]Benchmarking: {model}")
        router = make_router(model)

        console.print(f"  [dim]Running summarization...[/dim]")
        summarization = await benchmark_summarization(router)

        console.print(f"  [dim]Running ranking...[/dim]")
        ranking = await benchmark_ranking(router)

        console.print(f"  [dim]Running importance...[/dim]")
        importance = await benchmark_importance(router)

        console.print(f"  [dim]Running lesson extraction...[/dim]")
        lessons = await benchmark_lessons(router)

        console.print(f"  [dim]Running entity extraction...[/dim]")
        entity_extraction = await benchmark_entity_extraction(router)

        console.print(f"  [dim]Running contradiction detection...[/dim]")
        contradiction = await benchmark_contradiction(router)

        console.print(f"  [dim]Running combined rank+importance...[/dim]")
        rank_imp = await benchmark_rank_and_importance(router)

        all_results[model] = {
            "summarization": summarization,
            "ranking": ranking,
            "importance": importance,
            "lessons": lessons,
            "entity_extraction": entity_extraction,
            "contradiction": contradiction,
            "rank_and_importance": rank_imp,
        }
        console.print(f"  [green]Done with {model}[/green]\n")

    # Save merged results to JSON first (so data isn't lost if table rendering fails)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[bold]Results saved to {output_path} ({len(all_results)} models)[/bold]")

    # Print comparison tables (all results, including previously loaded)
    console.rule("[bold green]Results")
    print_summary_table(all_results)
    console.print()
    print_ranking_table(all_results)
    console.print()
    print_importance_table(all_results)
    console.print()
    print_lessons_table(all_results)
    console.print()
    print_entity_extraction_table(all_results)
    console.print()
    print_contradiction_table(all_results)
    console.print()
    print_rank_and_importance_table(all_results)


def test_benchmark():
    """Entry point for pytest -s."""
    asyncio.run(run_benchmark(BENCHMARK_MODELS))


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else BENCHMARK_MODELS
    asyncio.run(run_benchmark(models))
