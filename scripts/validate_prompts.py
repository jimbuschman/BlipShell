"""Validate all LLM prompts against the configured production models.

Runs each prompt type with its assigned model and checks that the output
can be parsed correctly. Unlike the full benchmark suite (which compares
many models side by side), this is a quick pass/fail check for the
current config.

Usage:
    python scripts/validate_prompts.py
    python scripts/validate_prompts.py --db data/blipshell.db  # include real-data samples
"""

import argparse
import asyncio
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from blipshell.core.config import ConfigManager
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

console = Console()

# --- Test data (small, focused on format validation) ---

RANK_TESTS = [
    ("hey", 1, 2),           # expect rank 1-2
    ("ok thanks", 1, 2),
    ("I decided to switch from SQLite to PostgreSQL for the project", 3, 5),
    ("My daughter got a new laptop for Christmas", 4, 5),
]

IMPORTANCE_TESTS = [
    ("hey", 0.0, 0.3),
    ("I decided to switch from SQLite to PostgreSQL for the project", 0.4, 1.0),
]

SUMMARY_TESTS = [
    "I'm using a MAX98357 with the esp32 and the sound it produces is terrible.",
    "My cat's name is Luna and she's 3 years old.",
]

ENTITY_TESTS = [
    "User decided to use a two-module design for the desk robot with JST connectors.",
    "Assistant explained how to configure Ollama with GPU acceleration on Ubuntu.",
]

CONTRADICTION_TESTS = [
    ("User prefers dark mode", "User prefers light mode", True),
    ("User likes coffee", "User likes tea", False),
]

LESSON_CONVERSATION = [
    {"role": "user", "content": "man this desk robot is just being a pain. the connectors are too loose."},
    {"role": "assistant", "content": "For the ESP32 connections, JST-PH 2-pin connectors are much more reliable than dupont wires."},
]

RANK_AND_IMPORTANCE_TESTS = [
    ("ok thanks", 1, 2, 0.0, 0.3),
    ("I decided to switch from SQLite to PostgreSQL", 3, 5, 0.4, 1.0),
]


class ValidationResult:
    def __init__(self):
        self.results: list[dict] = []

    def add(self, prompt_type: str, model: str, task_type: str,
            passed: bool, detail: str, elapsed: float):
        self.results.append({
            "prompt_type": prompt_type,
            "model": model,
            "task_type": task_type,
            "passed": passed,
            "detail": detail,
            "time": round(elapsed, 2),
        })

    def print_report(self):
        table = Table(title="Prompt Validation Results", show_lines=True)
        table.add_column("Prompt Type", style="bold")
        table.add_column("Model")
        table.add_column("Task Type")
        table.add_column("Status")
        table.add_column("Detail")
        table.add_column("Time", justify="right")

        for r in self.results:
            status = "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]"
            table.add_row(
                r["prompt_type"], r["model"], r["task_type"],
                status, r["detail"], f"{r['time']}s",
            )

        console.print(table)

        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if not r["passed"])
        total = len(self.results)
        color = "green" if failed == 0 else "red"
        console.print(f"\n[{color}]{passed}/{total} passed, {failed} failed[/{color}]")

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)


async def validate_ranking(router: LLMRouter, result: ValidationResult):
    model = router.get_model(TaskType.RANKING)
    for text, min_rank, max_rank in RANK_TESTS:
        sys_prompt, user_prompt = rank_memory(text)
        start = time.perf_counter()
        try:
            raw = await router.generate(TaskType.RANKING, user_prompt, system=sys_prompt, think=False)
            rank = MemoryProcessor._parse_rank(raw)
            elapsed = time.perf_counter() - start
            if min_rank <= rank <= max_rank:
                result.add("rank_memory", model, "ranking", True,
                           f"rank={rank} (expected {min_rank}-{max_rank})", elapsed)
            else:
                result.add("rank_memory", model, "ranking", False,
                           f"rank={rank} (expected {min_rank}-{max_rank}), raw={raw[:50]}", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            result.add("rank_memory", model, "ranking", False, f"ERROR: {e}", elapsed)
        await asyncio.sleep(0.1)


async def validate_importance(router: LLMRouter, result: ValidationResult):
    model = router.get_model(TaskType.IMPORTANCE)
    for text, min_imp, max_imp in IMPORTANCE_TESTS:
        sys_prompt, user_prompt = ask_importance(text)
        start = time.perf_counter()
        try:
            raw = await router.generate(TaskType.IMPORTANCE, user_prompt, system=sys_prompt, think=False)
            score = MemoryProcessor._parse_float(raw, default=-1.0)
            elapsed = time.perf_counter() - start
            if 0.0 <= score <= 1.0 and min_imp <= score <= max_imp:
                result.add("ask_importance", model, "importance", True,
                           f"importance={score} (expected {min_imp}-{max_imp})", elapsed)
            elif score < 0:
                result.add("ask_importance", model, "importance", False,
                           f"Could not parse float, raw={raw[:50]}", elapsed)
            else:
                result.add("ask_importance", model, "importance", False,
                           f"importance={score} (expected {min_imp}-{max_imp})", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            result.add("ask_importance", model, "importance", False, f"ERROR: {e}", elapsed)
        await asyncio.sleep(0.1)


async def validate_summarization(router: LLMRouter, result: ValidationResult):
    model = router.get_model(TaskType.SUMMARIZATION)
    for text in SUMMARY_TESTS:
        sys_prompt, user_prompt = summarize_memory(text)
        start = time.perf_counter()
        try:
            raw = await router.generate(TaskType.SUMMARIZATION, user_prompt, system=sys_prompt, think=False)
            elapsed = time.perf_counter() - start
            # Validation: non-empty, reasonable length, not an error
            raw = raw.strip()
            if len(raw) < 5:
                result.add("summarize_memory", model, "summarization", False,
                           f"Summary too short ({len(raw)} chars)", elapsed)
            elif len(raw) > 500:
                result.add("summarize_memory", model, "summarization", False,
                           f"Summary too long ({len(raw)} chars)", elapsed)
            elif raw.upper().startswith("ERROR"):
                result.add("summarize_memory", model, "summarization", False,
                           f"Error response: {raw[:80]}", elapsed)
            else:
                result.add("summarize_memory", model, "summarization", True,
                           f"len={len(raw)}: {raw[:60]}...", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            result.add("summarize_memory", model, "summarization", False, f"ERROR: {e}", elapsed)
        await asyncio.sleep(0.1)


async def validate_entity_extraction(router: LLMRouter, result: ValidationResult):
    model = router.get_model(TaskType.REASONING)
    extractor = EntityExtractor.__new__(EntityExtractor)
    for summary in ENTITY_TESTS:
        sys_prompt, user_prompt = extract_entities(summary)
        start = time.perf_counter()
        try:
            raw = await router.generate(TaskType.REASONING, user_prompt, system=sys_prompt, think=False)
            triples = extractor._parse_triples(raw)
            elapsed = time.perf_counter() - start
            if len(triples) > 0:
                entities = sorted({t[0].lower() for t in triples} | {t[2].lower() for t in triples})
                result.add("extract_entities", model, "reasoning", True,
                           f"{len(triples)} triples, entities: {', '.join(entities[:5])}", elapsed)
            else:
                result.add("extract_entities", model, "reasoning", False,
                           f"No triples parsed, raw={raw[:80]}", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            result.add("extract_entities", model, "reasoning", False, f"ERROR: {e}", elapsed)
        await asyncio.sleep(0.1)


async def validate_contradiction(router: LLMRouter, result: ValidationResult):
    model = router.get_model(TaskType.REASONING)
    for new_mem, existing_mem, expected_yes in CONTRADICTION_TESTS:
        sys_prompt, user_prompt = detect_contradiction(new_mem, existing_mem)
        start = time.perf_counter()
        try:
            raw = await router.generate(TaskType.REASONING, user_prompt, system=sys_prompt, think=False)
            elapsed = time.perf_counter() - start
            answer = raw.strip().upper()
            if answer.startswith("YES"):
                parsed = "YES"
            elif answer.startswith("NO"):
                parsed = "NO"
            else:
                parsed = "INVALID"
            expected = "YES" if expected_yes else "NO"
            if parsed == expected:
                result.add("detect_contradiction", model, "reasoning", True,
                           f"parsed={parsed} expected={expected}", elapsed)
            elif parsed == "INVALID":
                result.add("detect_contradiction", model, "reasoning", False,
                           f"Could not parse YES/NO, raw={raw[:50]}", elapsed)
            else:
                result.add("detect_contradiction", model, "reasoning", False,
                           f"Wrong answer: parsed={parsed} expected={expected}", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            result.add("detect_contradiction", model, "reasoning", False, f"ERROR: {e}", elapsed)
        await asyncio.sleep(0.1)


async def validate_lesson_extraction(router: LLMRouter, result: ValidationResult):
    model = router.get_model(TaskType.REASONING)
    parts = []
    for m in LESSON_CONVERSATION:
        label = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{label}: {m['content']}")
    text = "\n".join(parts)

    sys_prompt, user_prompt = extract_lesson(text)
    start = time.perf_counter()
    try:
        raw = await router.generate(TaskType.REASONING, user_prompt, system=sys_prompt, think=False)
        elapsed = time.perf_counter() - start
        raw = raw.strip()
        if len(raw) < 10:
            result.add("extract_lesson", model, "reasoning", False,
                       f"Lesson too short ({len(raw)} chars)", elapsed)
        elif "NONE" in raw.upper() and len(raw) < 20:
            result.add("extract_lesson", model, "reasoning", True,
                       "No lesson extracted (NONE)", elapsed)
        elif raw.upper().startswith("ERROR"):
            result.add("extract_lesson", model, "reasoning", False,
                       f"Error response: {raw[:80]}", elapsed)
        else:
            result.add("extract_lesson", model, "reasoning", True,
                       f"len={len(raw)}: {raw[:60]}...", elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - start
        result.add("extract_lesson", model, "reasoning", False, f"ERROR: {e}", elapsed)


async def validate_rank_and_importance(router: LLMRouter, result: ValidationResult):
    model = router.get_model(TaskType.RANKING)
    for text, min_rank, max_rank, min_imp, max_imp in RANK_AND_IMPORTANCE_TESTS:
        sys_prompt, user_prompt = rank_and_importance(text)
        start = time.perf_counter()
        try:
            raw = await router.generate(TaskType.RANKING, user_prompt, system=sys_prompt, think=False)
            rank, importance = MemoryProcessor._parse_rank_and_importance(raw)
            elapsed = time.perf_counter() - start
            rank_ok = min_rank <= rank <= max_rank
            imp_ok = min_imp <= importance <= max_imp
            if rank_ok and imp_ok:
                result.add("rank_and_importance", model, "ranking", True,
                           f"rank={rank} imp={importance}", elapsed)
            else:
                parts = []
                if not rank_ok:
                    parts.append(f"rank={rank} (expected {min_rank}-{max_rank})")
                if not imp_ok:
                    parts.append(f"imp={importance} (expected {min_imp}-{max_imp})")
                result.add("rank_and_importance", model, "ranking", False,
                           "; ".join(parts), elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            result.add("rank_and_importance", model, "ranking", False, f"ERROR: {e}", elapsed)
        await asyncio.sleep(0.1)


async def main():
    parser = argparse.ArgumentParser(description="Validate LLM prompts against configured models")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config_mgr = ConfigManager(args.config)
    config = config_mgr.config

    # Create router from config
    endpoint_manager = EndpointManager(config.endpoints, config.llm)
    router = LLMRouter(config.models, endpoint_manager)

    console.print("[bold]Prompt Validation[/bold]")
    console.print(f"Config: {args.config}")
    console.print(f"Models: ranking={config.models.ranking}, importance={config.models.importance}, "
                  f"summarization={config.models.summarization}, reasoning={config.models.reasoning}")
    console.print()

    result = ValidationResult()

    console.print("[dim]Running ranking...[/dim]")
    await validate_ranking(router, result)

    console.print("[dim]Running importance...[/dim]")
    await validate_importance(router, result)

    console.print("[dim]Running summarization...[/dim]")
    await validate_summarization(router, result)

    console.print("[dim]Running entity extraction...[/dim]")
    await validate_entity_extraction(router, result)

    console.print("[dim]Running contradiction detection...[/dim]")
    await validate_contradiction(router, result)

    console.print("[dim]Running lesson extraction...[/dim]")
    await validate_lesson_extraction(router, result)

    console.print("[dim]Running rank+importance...[/dim]")
    await validate_rank_and_importance(router, result)

    console.print()
    result.print_report()

    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
