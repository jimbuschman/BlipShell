"""Background Task Model Benchmark — compare Ollama models on all pipeline tasks.

Curated test cases with known-correct answers. Scores each model on:
  - Ranking / Importance / Type classification (combined prompt)
  - Summarization (SKIP detection + key fact preservation)
  - Entity extraction (precision/recall against annotated entities)
  - Contradiction detection (YES/NO accuracy)
  - Dedup / Memory action (correct action selection)

Crash-safe: saves after each model. Resumes automatically (skips completed models).

Usage:
    python scripts/benchmark_background.py                          # test all available models
    python scripts/benchmark_background.py --models qwen3:14b qwen2.5:7b
    python scripts/benchmark_background.py --task ranking           # only test ranking
    python scripts/benchmark_background.py --resume                 # explicit resume
    python scripts/benchmark_background.py --timeout 120            # per-call timeout
"""

import argparse
import asyncio
import io
import json
import re
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
elif not sys.stdout.isatty():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from blipshell.llm.client import LLMClient
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import (
    decide_memory_action,
    detect_contradiction,
    extract_entities,
    rank_importance_and_classify,
    summarize_memory,
)
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.processor import MemoryProcessor
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig

console = Console()

OLLAMA_URL = "http://localhost:11434"
OUTPUT_PATH = Path("data") / "benchmark_background_results.json"

# ============================================================================
# TEST DATA — Curated cases with expected answers
# ============================================================================

# -- Ranking / Importance / Type --
# Each test: content, role, expected rank range, importance range, acceptable types
RANKING_TESTS = [
    # === Rank 1: Junk / noise ===
    {
        "content": "hey",
        "role": "user",
        "rank": [1, 1], "importance": [0.0, 0.2], "types": ["conversation"],
        "label": "greeting",
    },
    {
        "content": "ok",
        "role": "user",
        "rank": [1, 1], "importance": [0.0, 0.2], "types": ["conversation"],
        "label": "filler-ok",
    },
    {
        "content": "thanks",
        "role": "user",
        "rank": [1, 1], "importance": [0.0, 0.2], "types": ["conversation"],
        "label": "filler-thanks",
    },
    {
        "content": "sure thing",
        "role": "user",
        "rank": [1, 1], "importance": [0.0, 0.2], "types": ["conversation"],
        "label": "filler-sure",
    },
    {
        "content": "got it",
        "role": "user",
        "rank": [1, 1], "importance": [0.0, 0.2], "types": ["conversation"],
        "label": "filler-gotit",
    },
    {
        "content": "[System: <important_rules>\nYou are in agent mode.\nIf you need to use multiple tools...</important_rules>]\ntest",
        "role": "user",
        "rank": [1, 2], "importance": [0.0, 0.2], "types": ["conversation"],
        "label": "system-noise",
    },

    # === Rank 2: Low value ===
    {
        "content": "what time is it",
        "role": "user",
        "rank": [1, 2], "importance": [0.1, 0.3], "types": ["conversation"],
        "label": "time-question",
    },
    {
        "content": "yeah that makes sense",
        "role": "user",
        "rank": [1, 2], "importance": [0.1, 0.3], "types": ["conversation"],
        "label": "agreement",
    },
    {
        "content": "can you repeat that",
        "role": "user",
        "rank": [1, 2], "importance": [0.1, 0.3], "types": ["conversation"],
        "label": "repeat-request",
    },
    {
        "content": "hmm let me think about it",
        "role": "user",
        "rank": [1, 2], "importance": [0.1, 0.3], "types": ["conversation"],
        "label": "thinking",
    },
    {
        "content": "sanding paint when its still tacky?",
        "role": "user",
        "rank": [2, 3], "importance": [0.2, 0.4], "types": ["conversation", "skill"],
        "label": "casual-question",
    },

    # === Rank 3: Medium value ===
    {
        "content": "I'm using a MAX98357 with the esp32 and the sound it produces is terrible. Mostly sounds like just noise/garbage.",
        "role": "user",
        "rank": [3, 4], "importance": [0.4, 0.7], "types": ["event", "conversation", "fact"],
        "label": "esp32-audio",
    },
    {
        "content": "How do I connect an SSD1306 OLED to an ESP32?",
        "role": "user",
        "rank": [2, 3], "importance": [0.3, 0.5], "types": ["conversation"],
        "label": "oled-question",
    },
    {
        "content": "The build failed again, something about missing imports in the test file",
        "role": "user",
        "rank": [2, 3], "importance": [0.3, 0.5], "types": ["event", "conversation"],
        "label": "build-failure",
    },
    {
        "content": "Can you explain how asyncio.gather works with error handling?",
        "role": "user",
        "rank": [2, 3], "importance": [0.3, 0.5], "types": ["conversation"],
        "label": "asyncio-question",
    },
    {
        "content": "Now I can see the worker.py file! Let me provide a code review:\n\n## Code Review: worker.py\n\n### Issues Found:\n1. Missing stop() call on shutdown\n2. No retry logic for failed HTTP requests\n3. No model fallback mechanism",
        "role": "assistant",
        "rank": [3, 4], "importance": [0.4, 0.7], "types": ["skill", "event", "conversation"],
        "label": "code-review",
    },

    # === Rank 4: Important ===
    {
        "content": "I prefer dark mode in all my editors and IDEs",
        "role": "user",
        "rank": [4, 5], "importance": [0.6, 0.9], "types": ["preference"],
        "label": "dark-mode-pref",
    },
    {
        "content": "I'm working on a desk robot project using an ESP32 with a MAX98357 DAC for audio output",
        "role": "user",
        "rank": [3, 5], "importance": [0.5, 0.8], "types": ["fact", "event", "project"],
        "label": "desk-robot-project",
    },
    {
        "content": "for some reason my daughters laptop when we got in dec only wants to load 10 blocks in minecraft, thats pretty terrible",
        "role": "user",
        "rank": [3, 5], "importance": [0.5, 0.8], "types": ["fact", "event"],
        "label": "daughters-minecraft",
    },
    {
        "content": "yeah i think ill go with the two-module design for the desk robot. main board plus a sidecar for the sensors. JST connectors between them.",
        "role": "user",
        "rank": [4, 5], "importance": [0.6, 0.9], "types": ["fact", "preference", "event"],
        "label": "design-decision",
    },
    {
        "content": "I've been using Python for about 10 years, mostly backend and automation stuff. Recently got into embedded with MicroPython.",
        "role": "user",
        "rank": [4, 5], "importance": [0.6, 0.9], "types": ["fact", "skill"],
        "label": "python-experience",
    },
    {
        "content": "JST-PH 2-pin connectors are way more reliable than dupont wires for ESP32 projects. Learned that the hard way.",
        "role": "user",
        "rank": [3, 5], "importance": [0.5, 0.8], "types": ["skill", "fact", "preference"],
        "label": "jst-connectors",
    },

    # === Rank 5: Critical ===
    {
        "content": "My daughter's name is Emily and she's 8 years old",
        "role": "user",
        "rank": [4, 5], "importance": [0.7, 1.0], "types": ["fact"],
        "label": "daughter-emily",
    },
    {
        "content": "I decided to switch the whole project from MongoDB to PostgreSQL. The document model was causing more problems than it solved.",
        "role": "user",
        "rank": [4, 5], "importance": [0.7, 1.0], "types": ["fact", "event", "preference"],
        "label": "db-switch-decision",
    },
    {
        "content": "My cat's name is Luna, she's a 3 year old tabby",
        "role": "user",
        "rank": [4, 5], "importance": [0.7, 1.0], "types": ["fact"],
        "label": "cat-luna",
    },
    {
        "content": "I work from home as a software engineer. My home office is in the basement.",
        "role": "user",
        "rank": [4, 5], "importance": [0.7, 1.0], "types": ["fact"],
        "label": "work-from-home",
    },
]


# -- Summarization --
# Each test: content, role, should_skip, required_facts (keywords that must appear)
SUMMARY_TESTS = [
    # Should SKIP
    {
        "content": "hey",
        "role": "user",
        "should_skip": True,
        "required_facts": [],
        "label": "skip-greeting",
    },
    {
        "content": "ok thanks",
        "role": "user",
        "should_skip": True,
        "required_facts": [],
        "label": "skip-filler",
    },
    {
        "content": "sure",
        "role": "user",
        "should_skip": True,
        "required_facts": [],
        "label": "skip-sure",
    },
    {
        "content": "got it",
        "role": "user",
        "should_skip": True,
        "required_facts": [],
        "label": "skip-gotit",
    },
    {
        "content": "[System: <important_rules>\nYou are in agent mode. Follow all rules.\n</important_rules>]\ntest",
        "role": "user",
        "should_skip": True,
        "required_facts": [],
        "label": "skip-system-noise",
    },

    # Should summarize — check key facts preserved
    {
        "content": "I'm using a MAX98357 with the esp32 and the sound it produces is terrible. Mostly sounds like just noise/garbage.",
        "role": "user",
        "should_skip": False,
        "required_facts": ["max98357|dac|audio", "esp32", "noise|sound|audio"],
        "label": "summ-esp32-audio",
    },
    {
        "content": "for some reason my daughters laptop when we got in dec only wants to load 10 blocks in minecraft, thats pretty terrible",
        "role": "user",
        "should_skip": False,
        "required_facts": ["daughter", "laptop", "minecraft"],
        "label": "summ-daughters-minecraft",
    },
    {
        "content": "yeah i think ill go with the two-module design for the desk robot. main board plus a sidecar for the sensors. JST connectors between them.",
        "role": "user",
        "should_skip": False,
        "required_facts": ["two.module|sidecar|modular", "desk.robot|robot", "jst|connector"],
        "label": "summ-design-decision",
    },
    {
        "content": "My daughter's name is Emily and she's 8 years old",
        "role": "user",
        "should_skip": False,
        "required_facts": ["daughter|emily", "8|eight"],
        "label": "summ-daughter-emily",
    },
    {
        "content": "I decided to switch the whole project from MongoDB to PostgreSQL. The document model was causing more problems than it solved.",
        "role": "user",
        "should_skip": False,
        "required_facts": ["mongodb|mongo", "postgresql|postgres", "switch|migrat"],
        "label": "summ-db-switch",
    },
    {
        "content": "Now I can see the worker.py file! Let me provide a code review:\n\n## Code Review: worker.py\n\n### Issues Found:\n1. Missing stop() call on shutdown\n2. No retry logic for failed HTTP requests\n3. No model fallback mechanism",
        "role": "assistant",
        "should_skip": False,
        "required_facts": ["worker", "review|issue|bug", "shutdown|retry|fallback"],
        "label": "summ-code-review",
    },
    {
        "content": "I've been using Python for about 10 years, mostly backend and automation stuff. Recently got into embedded with MicroPython.",
        "role": "user",
        "should_skip": False,
        "required_facts": ["python", "10.year|decade", "backend|automation|embedded|micropython"],
        "label": "summ-python-experience",
    },
]


# -- Entity Extraction --
# Each test: a summary with expected entity triples (subject, predicate, object)
# We check: did the model extract at least the expected entities (by name)?
ENTITY_TESTS = [
    {
        "summary": "User asked about Python performance tuning for BlipShell.",
        "expected_entities": ["python", "blipshell"],
        "label": "entity-python-blipshell",
    },
    {
        "summary": "User's daughter Emily got a new laptop that struggles with Minecraft.",
        "expected_entities": ["emily", "minecraft", "laptop"],
        "label": "entity-emily-minecraft",
    },
    {
        "summary": "User decided to switch from MongoDB to PostgreSQL for the project.",
        "expected_entities": ["mongodb", "postgresql"],
        "label": "entity-db-switch",
    },
    {
        "summary": "User is building a desk robot using ESP32 with MAX98357 DAC for audio.",
        "expected_entities": ["esp32", "max98357"],
        "label": "entity-desk-robot",
    },
    {
        "summary": "User prefers dark mode in VS Code and all editors.",
        "expected_entities": ["vs code", "dark mode"],
        "label": "entity-dark-mode",
    },
    {
        "summary": "Assistant explained how to configure Ollama with GPU support on Linux.",
        "expected_entities": ["ollama", "gpu", "linux"],
        "label": "entity-ollama-gpu",
    },
    {
        "summary": "User has been using Python for 10 years, recently started MicroPython.",
        "expected_entities": ["python", "micropython"],
        "label": "entity-python-micropython",
    },
    {
        "summary": "User chose JST-PH connectors over dupont wires for ESP32 project.",
        "expected_entities": ["jst", "esp32"],
        "label": "entity-jst-esp32",
    },
    {
        "summary": "User works from home as a software engineer.",
        "expected_entities": ["software engineer"],
        "label": "entity-work-home",
    },
    {
        "summary": "User's cat Luna is a 3 year old tabby.",
        "expected_entities": ["luna"],
        "label": "entity-cat-luna",
    },
    {
        "summary": "User said hello and asked how the assistant is doing.",
        "expected_entities": [],
        "expect_none": True,
        "label": "entity-none-greeting",
    },
    {
        "summary": "User asked about something but was very vague.",
        "expected_entities": [],
        "expect_none": True,
        "label": "entity-none-vague",
    },
]


# -- Contradiction Detection --
# Clear yes/no pairs
CONTRADICTION_TESTS = [
    # YES — contradictions
    {
        "new": "User prefers dark mode in all editors",
        "existing": "User prefers light mode in all editors",
        "expected": "YES",
        "label": "contra-dark-vs-light",
    },
    {
        "new": "User uses Windows 11",
        "existing": "User uses Windows 10",
        "expected": "YES",
        "label": "contra-windows-version",
    },
    {
        "new": "User dislikes Python",
        "existing": "User's favorite programming language is Python",
        "expected": "YES",
        "label": "contra-python-opinion",
    },
    {
        "new": "User switched from MongoDB to PostgreSQL",
        "existing": "User chose MongoDB as the database for the project",
        "expected": "YES",
        "label": "contra-db-choice",
    },
    {
        "new": "User's daughter is 9 years old",
        "existing": "User's daughter is 6 years old",
        "expected": "YES",
        "label": "contra-daughter-age",
    },
    {
        "new": "User works at Microsoft",
        "existing": "User works at Google",
        "expected": "YES",
        "label": "contra-employer",
    },

    # NO — not contradictions
    {
        "new": "User likes coffee",
        "existing": "User likes tea",
        "expected": "NO",
        "label": "no-contra-coffee-tea",
    },
    {
        "new": "User knows Python",
        "existing": "User knows Rust",
        "expected": "NO",
        "label": "no-contra-languages",
    },
    {
        "new": "User has a cat named Luna",
        "existing": "User works as a software engineer",
        "expected": "NO",
        "label": "no-contra-unrelated",
    },
    {
        "new": "User built a desk robot with ESP32",
        "existing": "User uses Python for backend development",
        "expected": "NO",
        "label": "no-contra-different-topics",
    },
    {
        "new": "User prefers VS Code for Python",
        "existing": "User prefers vim for quick edits",
        "expected": "NO",
        "label": "no-contra-editors",
    },
    {
        "new": "User's daughter plays Minecraft",
        "existing": "User's daughter is named Emily",
        "expected": "NO",
        "label": "no-contra-daughter-facts",
    },
]


# -- Dedup / Memory Action --
# Pairs of (new_summary, existing_summaries, expected_action)
DEDUP_TESTS = [
    # NONE — redundant
    {
        "new_summary": "User prefers dark mode in editors.",
        "existing": ["User prefers dark mode in all editors and IDEs."],
        "expected_action": "NONE",
        "label": "dedup-redundant-dark-mode",
    },
    {
        "new_summary": "User's cat is named Luna.",
        "existing": ["User has a cat named Luna, a 3 year old tabby."],
        "expected_action": "NONE",
        "label": "dedup-redundant-cat",
    },

    # ADD — new information
    {
        "new_summary": "User is learning Rust for embedded programming.",
        "existing": ["User has been using Python for 10 years."],
        "expected_action": "ADD",
        "label": "dedup-add-rust",
    },
    {
        "new_summary": "User decided to use PostgreSQL instead of MongoDB.",
        "existing": ["User is building a desk robot with ESP32."],
        "expected_action": "ADD",
        "label": "dedup-add-unrelated",
    },
    {
        "new_summary": "User's daughter Emily started taking piano lessons.",
        "existing": ["User's daughter Emily is 8 years old."],
        "expected_action": "ADD",
        "label": "dedup-add-new-fact",
    },

    # UPDATE — refines existing
    {
        "new_summary": "User has been using Python for 10 years, specializing in backend, automation, and recently embedded via MicroPython.",
        "existing": ["User has been using Python for about 10 years."],
        "expected_action": "UPDATE",
        "label": "dedup-update-python",
    },
    {
        "new_summary": "User's daughter Emily is 8 years old and plays Minecraft on her laptop.",
        "existing": ["User's daughter Emily is 8 years old."],
        "expected_action": "UPDATE",
        "acceptable": ["ADD"],  # ADD is also reasonable here
        "label": "dedup-update-daughter",
    },

    # DELETE — contradicts existing
    {
        "new_summary": "User switched from MongoDB to PostgreSQL for the project.",
        "existing": ["User chose MongoDB as the primary database."],
        "expected_action": "DELETE",
        "acceptable": ["UPDATE"],  # UPDATE is also reasonable
        "label": "dedup-delete-db-switch",
    },
]


# ============================================================================
# ROUTER SETUP
# ============================================================================

def make_router(model_name: str, ollama_url: str = OLLAMA_URL) -> LLMRouter:
    """Create a router that sends all task types to a specific model."""
    models = ModelsConfig(
        reasoning=model_name,
        summarization=model_name,
        ranking=model_name,
        importance=model_name,
        ranking_importance=model_name,
    )
    ep = EndpointConfig(
        name="benchmark",
        url=ollama_url,
        roles=["reasoning", "summarization", "ranking", "importance", "ranking_importance"],
        priority=1,
        max_concurrent=1,
    )
    return LLMRouter(models, EndpointManager([ep], LLMConfig()), pii_enabled=False)


async def list_available_models(ollama_url: str = OLLAMA_URL) -> list[str]:
    """Get list of models available on Ollama."""
    client = LLMClient(host=ollama_url)
    try:
        return await client.list_models()
    except Exception as e:
        console.print(f"[red]Could not list models from {ollama_url}: {e}[/red]")
        return []


async def warmup_model(router: LLMRouter, model_name: str):
    """Send a throwaway request to load the model into memory."""
    try:
        await asyncio.wait_for(
            router.generate(TaskType.REASONING, "Say hello.", system="Be brief.", think=False),
            timeout=120,
        )
    except Exception:
        pass  # warmup failure is fine


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def score_ranking_test(rank: int, importance: float, mem_type: str, test: dict) -> dict:
    """Score a single ranking test result. Returns per-metric scores (0.0 or 1.0)."""
    r_min, r_max = test["rank"]
    i_min, i_max = test["importance"]
    valid_types = test["types"]

    # Rank: in range = 1.0, within 1 of range = 0.5, else 0.0
    if r_min <= rank <= r_max:
        rank_score = 1.0
    elif (r_min - 1) <= rank <= (r_max + 1):
        rank_score = 0.5
    else:
        rank_score = 0.0

    # Importance: in range = 1.0, within 0.15 of range = 0.5, else 0.0
    if i_min <= importance <= i_max:
        imp_score = 1.0
    elif (i_min - 0.15) <= importance <= (i_max + 0.15):
        imp_score = 0.5
    else:
        imp_score = 0.0

    # Type: match = 1.0, else 0.0
    type_score = 1.0 if mem_type in valid_types else 0.0

    return {"rank": rank_score, "importance": imp_score, "type": type_score}


def score_summary_test(raw: str, test: dict) -> dict:
    """Score a single summarization test result."""
    raw_stripped = raw.strip() if raw else ""
    is_skip = raw_stripped.upper().startswith("SKIP")

    if test["should_skip"]:
        # Should have returned SKIP
        return {"skip_correct": 1.0 if is_skip else 0.0, "fact_recall": None}

    if is_skip:
        # Returned SKIP but shouldn't have
        return {"skip_correct": 0.0, "fact_recall": 0.0}

    # Check fact preservation: each required_fact is a regex pattern (pipe = OR)
    facts_found = 0
    total_facts = len(test["required_facts"])
    if total_facts == 0:
        return {"skip_correct": 1.0, "fact_recall": 1.0}

    summary_lower = raw_stripped.lower()
    for fact_pattern in test["required_facts"]:
        if re.search(fact_pattern, summary_lower, re.IGNORECASE):
            facts_found += 1

    return {
        "skip_correct": 1.0,  # correctly didn't skip
        "fact_recall": facts_found / total_facts,
    }


def score_entity_test(raw: str, test: dict) -> dict:
    """Score a single entity extraction test result."""
    raw_stripped = raw.strip() if raw else ""
    is_none = raw_stripped.upper().startswith("NONE")

    if test.get("expect_none"):
        return {"correct": 1.0 if is_none else 0.0}

    if is_none:
        return {"correct": 0.0}

    # Check how many expected entities appear in the output
    raw_lower = raw_stripped.lower()
    found = 0
    for entity in test["expected_entities"]:
        # Flexible match — check if entity name appears anywhere in the output
        if entity.lower() in raw_lower:
            found += 1

    total = len(test["expected_entities"])
    recall = found / total if total > 0 else 1.0

    # Check format compliance (should have pipe-delimited lines)
    lines = [l.strip() for l in raw_stripped.split("\n") if l.strip() and "|" in l]
    format_ok = len(lines) > 0

    return {"recall": recall, "format_ok": 1.0 if format_ok else 0.0}


def score_contradiction_test(raw: str, test: dict) -> dict:
    """Score a single contradiction detection test result."""
    raw_upper = raw.strip().upper() if raw else ""
    # Extract YES or NO from response
    if "YES" in raw_upper and "NO" not in raw_upper:
        answer = "YES"
    elif "NO" in raw_upper and "YES" not in raw_upper:
        answer = "NO"
    elif raw_upper.startswith("YES"):
        answer = "YES"
    elif raw_upper.startswith("NO"):
        answer = "NO"
    else:
        answer = "UNKNOWN"

    return {"correct": 1.0 if answer == test["expected"] else 0.0}


def score_dedup_test(action: str, test: dict) -> dict:
    """Score a single dedup test result."""
    expected = test["expected_action"]
    acceptable = test.get("acceptable", [])
    all_ok = [expected] + acceptable

    return {"correct": 1.0 if action in all_ok else 0.0}


# ============================================================================
# BENCHMARK RUNNERS
# ============================================================================

async def bench_ranking(router: LLMRouter, timeout: float) -> dict:
    """Run ranking/importance/type tests. Returns per-test results + aggregates."""
    results = []
    for test in RANKING_TESTS:
        sys_prompt, user_prompt = rank_importance_and_classify(test["content"])
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                router.generate(TaskType.RANKING_IMPORTANCE, user_prompt, system=sys_prompt, think=False),
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            rank, importance, mem_type = MemoryProcessor._parse_rank_importance_type(raw)
            scores = score_ranking_test(rank, importance, mem_type, test)
            results.append({
                "label": test["label"],
                "raw": raw[:80],
                "parsed": {"rank": rank, "importance": importance, "type": mem_type},
                "expected": {"rank": test["rank"], "importance": test["importance"], "types": test["types"]},
                "scores": scores,
                "time": round(elapsed, 2),
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "label": test["label"],
                "error": str(e)[:120],
                "scores": {"rank": 0.0, "importance": 0.0, "type": 0.0},
                "time": round(elapsed, 2),
                "ok": False,
            })

    return _aggregate_results("ranking", results, ["rank", "importance", "type"])


async def bench_summarization(router: LLMRouter, timeout: float) -> dict:
    """Run summarization tests."""
    results = []
    for test in SUMMARY_TESTS:
        sys_prompt, user_prompt = summarize_memory(test["content"])
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                router.generate(TaskType.SUMMARIZATION, user_prompt, system=sys_prompt, think=False),
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            scores = score_summary_test(raw, test)
            results.append({
                "label": test["label"],
                "raw": raw[:150],
                "scores": scores,
                "time": round(elapsed, 2),
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "label": test["label"],
                "error": str(e)[:120],
                "scores": {"skip_correct": 0.0, "fact_recall": 0.0},
                "time": round(elapsed, 2),
                "ok": False,
            })

    return _aggregate_results("summarization", results, ["skip_correct", "fact_recall"])


async def bench_entities(router: LLMRouter, timeout: float) -> dict:
    """Run entity extraction tests."""
    results = []
    for test in ENTITY_TESTS:
        sys_prompt, user_prompt = extract_entities(test["summary"])
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                router.generate(TaskType.REASONING, user_prompt, system=sys_prompt, think=False),
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            scores = score_entity_test(raw, test)
            results.append({
                "label": test["label"],
                "raw": raw[:200],
                "scores": scores,
                "time": round(elapsed, 2),
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            default_scores = {"correct": 0.0} if test.get("expect_none") else {"recall": 0.0, "format_ok": 0.0}
            results.append({
                "label": test["label"],
                "error": str(e)[:120],
                "scores": default_scores,
                "time": round(elapsed, 2),
                "ok": False,
            })

    return _aggregate_results("entities", results, ["recall", "format_ok", "correct"])


async def bench_contradiction(router: LLMRouter, timeout: float) -> dict:
    """Run contradiction detection tests."""
    results = []
    for test in CONTRADICTION_TESTS:
        sys_prompt, user_prompt = detect_contradiction(test["new"], test["existing"])
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                router.generate(TaskType.REASONING, user_prompt, system=sys_prompt, think=False),
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            scores = score_contradiction_test(raw, test)
            results.append({
                "label": test["label"],
                "raw": raw[:80],
                "expected": test["expected"],
                "scores": scores,
                "time": round(elapsed, 2),
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "label": test["label"],
                "error": str(e)[:120],
                "expected": test["expected"],
                "scores": {"correct": 0.0},
                "time": round(elapsed, 2),
                "ok": False,
            })

    return _aggregate_results("contradiction", results, ["correct"])


async def bench_dedup(router: LLMRouter, timeout: float) -> dict:
    """Run dedup / memory action tests."""
    results = []
    for test in DEDUP_TESTS:
        sys_prompt, user_prompt = decide_memory_action(test["new_summary"], test["existing"])
        t0 = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                router.generate(TaskType.REASONING, user_prompt, system=sys_prompt, think=False),
                timeout=timeout,
            )
            elapsed = time.monotonic() - t0
            action, _ = MemoryProcessor._parse_memory_action(raw)
            scores = score_dedup_test(action, test)
            results.append({
                "label": test["label"],
                "raw": raw[:80],
                "parsed_action": action,
                "expected": test["expected_action"],
                "scores": scores,
                "time": round(elapsed, 2),
                "ok": True,
            })
        except Exception as e:
            elapsed = time.monotonic() - t0
            results.append({
                "label": test["label"],
                "error": str(e)[:120],
                "expected": test["expected_action"],
                "scores": {"correct": 0.0},
                "time": round(elapsed, 2),
                "ok": False,
            })

    return _aggregate_results("dedup", results, ["correct"])


def _aggregate_results(task_name: str, results: list[dict], metric_keys: list[str]) -> dict:
    """Aggregate per-test results into task-level stats."""
    ok_results = [r for r in results if r["ok"]]
    errors = len(results) - len(ok_results)
    times = [r["time"] for r in ok_results]

    # Compute average for each metric across all results (including errors with 0.0)
    metric_avgs = {}
    for key in metric_keys:
        values = []
        for r in results:
            v = r["scores"].get(key)
            if v is not None:
                values.append(v)
        metric_avgs[key] = round(sum(values) / len(values), 3) if values else 0.0

    return {
        "task": task_name,
        "results": results,
        "total": len(results),
        "errors": errors,
        "avg_time": round(sum(times) / len(times), 2) if times else 0.0,
        "total_time": round(sum(times), 1) if times else 0.0,
        "metrics": metric_avgs,
    }


# ============================================================================
# OUTPUT / DISPLAY
# ============================================================================

TASK_ORDER = ["ranking", "summarization", "entities", "contradiction", "dedup"]

# Mapping of task → metrics to show in the comparison table
TASK_METRICS = {
    "ranking": [("rank", "Rank"), ("importance", "Import"), ("type", "Type")],
    "summarization": [("skip_correct", "Skip"), ("fact_recall", "Facts")],
    "entities": [("recall", "Recall"), ("format_ok", "Format"), ("correct", "None-det")],
    "contradiction": [("correct", "Accuracy")],
    "dedup": [("correct", "Accuracy")],
}


def print_task_table(task_name: str, all_model_results: dict[str, dict]):
    """Print comparison table for one task across all models."""
    metrics_config = TASK_METRICS.get(task_name, [])
    if not metrics_config:
        return

    table = Table(title=f"{task_name.upper()}", show_lines=True, title_style="bold magenta")
    table.add_column("Model", style="cyan", min_width=25)
    for _, header in metrics_config:
        table.add_column(header, justify="right", min_width=8)
    table.add_column("Avg/s", justify="right", min_width=7)
    table.add_column("Err", justify="right", min_width=4)

    for model_name, model_data in all_model_results.items():
        task_data = model_data.get(task_name)
        if not task_data:
            continue

        row = [model_name]
        for metric_key, _ in metrics_config:
            val = task_data["metrics"].get(metric_key)
            if val is not None:
                pct = f"{val * 100:.0f}%"
                if val >= 0.8:
                    row.append(f"[green]{pct}[/green]")
                elif val >= 0.6:
                    row.append(f"[yellow]{pct}[/yellow]")
                else:
                    row.append(f"[red]{pct}[/red]")
            else:
                row.append("-")
        row.append(f"{task_data['avg_time']:.1f}s")
        row.append(str(task_data["errors"]))
        table.add_row(*row)

    console.print(table)
    console.print()


def print_scorecard(all_model_results: dict[str, dict]):
    """Print overall scorecard combining all tasks."""
    table = Table(title="OVERALL SCORECARD", show_lines=True, title_style="bold green")
    table.add_column("Model", style="cyan", min_width=25)
    for task in TASK_ORDER:
        table.add_column(task.capitalize()[:7], justify="right", min_width=8)
    table.add_column("OVERALL", justify="right", min_width=9, style="bold")
    table.add_column("Avg/s", justify="right", min_width=7)

    for model_name, model_data in all_model_results.items():
        row = [model_name]
        task_scores = []
        total_time = 0.0
        task_count = 0

        for task in TASK_ORDER:
            task_data = model_data.get(task)
            if not task_data:
                row.append("-")
                continue

            # Average all metrics for this task into one score
            metrics = task_data["metrics"]
            vals = [v for v in metrics.values() if v is not None]
            avg = sum(vals) / len(vals) if vals else 0.0
            task_scores.append(avg)
            total_time += task_data.get("avg_time", 0.0)
            task_count += 1

            pct = f"{avg * 100:.0f}%"
            if avg >= 0.8:
                row.append(f"[green]{pct}[/green]")
            elif avg >= 0.6:
                row.append(f"[yellow]{pct}[/yellow]")
            else:
                row.append(f"[red]{pct}[/red]")

        overall = sum(task_scores) / len(task_scores) if task_scores else 0.0
        overall_pct = f"{overall * 100:.0f}%"
        if overall >= 0.8:
            row.append(f"[green]{overall_pct}[/green]")
        elif overall >= 0.6:
            row.append(f"[yellow]{overall_pct}[/yellow]")
        else:
            row.append(f"[red]{overall_pct}[/red]")

        avg_time = total_time / task_count if task_count else 0.0
        row.append(f"{avg_time:.1f}s")
        table.add_row(*row)

    console.print(table)


def print_recommendations(all_model_results: dict[str, dict]):
    """Print per-task recommendations."""
    console.print("\n[bold]RECOMMENDATIONS[/bold]")
    console.print("-" * 60)

    for task in TASK_ORDER:
        best_model = None
        best_score = -1
        best_time = 999

        for model_name, model_data in all_model_results.items():
            task_data = model_data.get(task)
            if not task_data:
                continue
            metrics = task_data["metrics"]
            vals = [v for v in metrics.values() if v is not None]
            avg = sum(vals) / len(vals) if vals else 0.0

            # Prefer higher score, break ties by speed
            if avg > best_score or (avg == best_score and task_data["avg_time"] < best_time):
                best_score = avg
                best_model = model_name
                best_time = task_data["avg_time"]

        if best_model:
            console.print(
                f"  {task:15s}  [cyan]{best_model}[/cyan]  "
                f"({best_score * 100:.0f}%, {best_time:.1f}s avg)"
            )

    # Overall best
    best_model = None
    best_overall = -1
    for model_name, model_data in all_model_results.items():
        scores = []
        for task in TASK_ORDER:
            task_data = model_data.get(task)
            if not task_data:
                continue
            metrics = task_data["metrics"]
            vals = [v for v in metrics.values() if v is not None]
            scores.append(sum(vals) / len(vals) if vals else 0.0)
        overall = sum(scores) / len(scores) if scores else 0.0
        if overall > best_overall:
            best_overall = overall
            best_model = model_name

    if best_model:
        console.print(f"\n  [bold green]Best overall: {best_model} ({best_overall * 100:.0f}%)[/bold green]")


# ============================================================================
# PERSISTENCE
# ============================================================================

def save_results(all_results: dict, path: Path):
    """Save results to JSON (atomic write via temp file)."""
    path.parent.mkdir(exist_ok=True)
    # Strip per-test raw output to keep file manageable
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }
    for model_name, model_data in all_results.items():
        save_data["models"][model_name] = {}
        for task_name, task_data in model_data.items():
            # Keep metrics, timing, and per-test scores but truncate raw output
            save_data["models"][model_name][task_name] = {
                "metrics": task_data["metrics"],
                "total": task_data["total"],
                "errors": task_data["errors"],
                "avg_time": task_data["avg_time"],
                "total_time": task_data["total_time"],
                "results": task_data["results"],
            }

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)
    tmp.replace(path)


def load_results(path: Path) -> dict:
    """Load previous results. Returns {model_name: {task: data}}."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("models", {})
    except Exception as e:
        console.print(f"[yellow]Warning: could not load previous results: {e}[/yellow]")
        return {}


# ============================================================================
# MAIN
# ============================================================================

ALL_TASKS = ["ranking", "summarization", "entities", "contradiction", "dedup"]

BENCH_FUNCTIONS = {
    "ranking": bench_ranking,
    "summarization": bench_summarization,
    "entities": bench_entities,
    "contradiction": bench_contradiction,
    "dedup": bench_dedup,
}


async def run_model(model_name: str, router: LLMRouter, tasks: list[str], timeout: float) -> dict:
    """Run all requested benchmark tasks for one model. Returns {task: results}."""
    model_data = {}

    for task in tasks:
        bench_fn = BENCH_FUNCTIONS[task]
        console.print(f"    {task}...", end=" ")
        try:
            result = await bench_fn(router, timeout)
            # Compute quick score for display
            metrics = result["metrics"]
            vals = [v for v in metrics.values() if v is not None]
            avg = sum(vals) / len(vals) if vals else 0.0
            console.print(
                f"[{'green' if avg >= 0.8 else 'yellow' if avg >= 0.6 else 'red'}]"
                f"{avg * 100:.0f}%[/] ({result['avg_time']:.1f}s, {result['errors']} errors)"
            )
            model_data[task] = result
        except Exception as e:
            console.print(f"[red]FAILED: {e}[/red]")
            # Don't add to model_data — task is missing, not zeroed

    return model_data


async def main():
    parser = argparse.ArgumentParser(
        description="Background task model benchmark — compare Ollama models on all pipeline tasks"
    )
    parser.add_argument("--models", nargs="*", help="Models to test (default: all available)")
    parser.add_argument("--task", nargs="*", choices=ALL_TASKS, help="Tasks to run (default: all)")
    parser.add_argument("--timeout", type=float, default=120, help="Timeout per LLM call in seconds (default: 120)")
    parser.add_argument("--url", default=OLLAMA_URL, help=f"Ollama URL (default: {OLLAMA_URL})")
    parser.add_argument("--resume", action="store_true", help="Resume from previous results (skip completed models)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output JSON path")
    args = parser.parse_args()

    tasks = args.task or ALL_TASKS
    output_path = Path(args.output)

    console.rule("[bold]Background Task Model Benchmark[/bold]")
    console.print(f"Tasks: {', '.join(tasks)}")
    console.print(f"Timeout: {args.timeout}s per call")
    console.print(f"Output: {output_path}")

    # Load previous results for resume
    previous_results = load_results(output_path) if args.resume else {}
    if previous_results:
        console.print(f"[yellow]Resuming: {len(previous_results)} model(s) already completed[/yellow]")

    # Determine models to test
    if args.models:
        models_to_test = args.models
    else:
        console.print(f"\nQuerying available models on {args.url}...")
        available = await list_available_models(args.url)
        if not available:
            console.print("[red]No models found. Is Ollama running?[/red]")
            return
        console.print(f"Found {len(available)} models: {', '.join(available[:10])}")
        if len(available) > 10:
            console.print(f"  ... and {len(available) - 10} more")
        models_to_test = available

    # Filter out already-completed models
    if previous_results:
        skip = set()
        for model in models_to_test:
            prev = previous_results.get(model, {})
            # Skip if all requested tasks are already done
            if all(t in prev for t in tasks):
                skip.add(model)
        if skip:
            console.print(f"[yellow]Skipping {len(skip)} completed model(s): {', '.join(skip)}[/yellow]")
            models_to_test = [m for m in models_to_test if m not in skip]

    if not models_to_test:
        console.print("[green]All models already completed! Use without --resume to re-run.[/green]")
        # Still print results from previous data
        all_results = previous_results
    else:
        console.print(f"\nTesting {len(models_to_test)} model(s): {', '.join(models_to_test)}")

        # Test counts
        test_counts = {
            "ranking": len(RANKING_TESTS),
            "summarization": len(SUMMARY_TESTS),
            "entities": len(ENTITY_TESTS),
            "contradiction": len(CONTRADICTION_TESTS),
            "dedup": len(DEDUP_TESTS),
        }
        total_tests = sum(test_counts[t] for t in tasks)
        console.print(f"Tests per model: {total_tests} ({', '.join(f'{t}={test_counts[t]}' for t in tasks)})")
        console.print()

        all_results = dict(previous_results)  # start from previous if resuming

        for i, model_name in enumerate(models_to_test, 1):
            console.rule(f"[bold cyan]{model_name}[/bold cyan] ({i}/{len(models_to_test)})")

            router = make_router(model_name, args.url)

            # Warmup
            console.print("  Warming up...", end=" ")
            await warmup_model(router, model_name)
            console.print("[green]ready[/green]")

            # Run benchmarks
            model_data = await run_model(model_name, router, tasks, args.timeout)

            if model_data:
                all_results[model_name] = model_data
                save_results(all_results, output_path)
                console.print(f"  [dim]Saved to {output_path}[/dim]")
            else:
                console.print(f"  [red]No results for {model_name}[/red]")

            console.print()

    # Print comparison tables
    if all_results:
        console.rule("[bold]Results[/bold]")
        for task in tasks:
            print_task_table(task, all_results)

        print_scorecard(all_results)
        print_recommendations(all_results)

        console.print(f"\n[dim]Full results: {output_path}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
