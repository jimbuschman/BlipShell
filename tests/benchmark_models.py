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

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import (
    ask_importance,
    detect_contradiction,
    extract_entities,
    extract_lesson,
    rank_importance_and_classify,
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
# Each case carries its own ground truth: truth_rank (1=trivial .. 5=high-value)
# and truth_importance (0.0-1.0). Scoring (harness.py) compares the model's
# output against these per item via rank-correlation + calibration, so the set
# must span the full spectrum to discriminate. ~20 items for a meaningful signal.
TEST_MESSAGES = [
    {"role": "user", "content": "hey", "label": "greeting", "truth_rank": 1, "truth_importance": 0.05},
    {"role": "user", "content": "ok thanks", "label": "filler", "truth_rank": 1, "truth_importance": 0.05},
    {"role": "user", "content": "lol yeah", "label": "filler 2", "truth_rank": 1, "truth_importance": 0.05},
    {"role": "user", "content": "brb", "label": "brb", "truth_rank": 1, "truth_importance": 0.05},
    {"role": "user", "content": "what time is it", "label": "transient q", "truth_rank": 1, "truth_importance": 0.1},
    {
        "role": "user",
        "content": (
            "[System: <important_rules>\nYou are in agent mode.\n"
            "If you need to use multiple tools...</important_rules>]\ntest"
        ),
        "label": "system noise", "truth_rank": 1, "truth_importance": 0.05,
    },
    {"role": "user", "content": "thanks that worked", "label": "ack", "truth_rank": 1, "truth_importance": 0.1},
    {"role": "user", "content": "sanding paint when its still tacky?", "label": "casual q", "truth_rank": 2, "truth_importance": 0.25},
    {"role": "user", "content": "the meeting got moved to thursday at 3pm", "label": "transient fact", "truth_rank": 3, "truth_importance": 0.45},
    {
        "role": "user",
        "content": (
            "I'm using a MAX98357 with the esp32 and the sound it produces "
            "is terrible. Mostly sounds like just noise/garbage."
        ),
        "label": "ESP32 audio issue", "truth_rank": 3, "truth_importance": 0.55,
    },
    {
        "role": "user",
        "content": (
            "for some reason my daughters laptop when we got in dec only "
            "wants to load 10 blocks in minecraft, thats pretty terrbile"
        ),
        "label": "daughter minecraft", "truth_rank": 3, "truth_importance": 0.5,
    },
    {"role": "user", "content": "can you explain how RRF fusion works in hybrid search?", "label": "learning q", "truth_rank": 3, "truth_importance": 0.45},
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
        "label": "code review (asst)", "truth_rank": 3, "truth_importance": 0.55,
    },
    {"role": "user", "content": "I prefer dark mode in every app I use", "label": "preference", "truth_rank": 4, "truth_importance": 0.7},
    {"role": "user", "content": "the API key for the project lives in the .env file at the repo root", "label": "useful fact", "truth_rank": 4, "truth_importance": 0.7},
    {
        "role": "user",
        "content": "I swapped the speaker to a 4ohm on JST-PH connectors and the noise is gone — the 8ohm was the problem.",
        "label": "resolution", "truth_rank": 4, "truth_importance": 0.72,
    },
    {
        "role": "user",
        "content": (
            "yeah i think ill go with the two-module design for the desk robot. "
            "main board plus a sidecar for the sensors. JST connectors between them."
        ),
        "label": "desk robot decision", "truth_rank": 5, "truth_importance": 0.85,
    },
    {"role": "user", "content": "I decided to drop ChromaDB entirely and move the vector store to sqlite-vec.", "label": "arch decision", "truth_rank": 5, "truth_importance": 0.88},
    {"role": "user", "content": "my name is Jim and I work at Xanatek as a developer", "label": "identity fact", "truth_rank": 5, "truth_importance": 0.9},
    {"role": "user", "content": "my daughter's name is Lily and she's 8 years old", "label": "personal fact", "truth_rank": 5, "truth_importance": 0.85},
]

TEST_CONVERSATIONS = [
    # Each is a short session; lesson extraction is judge-graded (no ground truth).
    [
        {"role": "user", "content": "man this desk robot is just being a pain. the connectors are too loose."},
        {"role": "assistant", "content": "Yeah that's frustrating. For the ESP32 connections, JST-PH 2-pin connectors are much more reliable than dupont wires."},
        {"role": "user", "content": "yeah ill try those. also the speaker got really hot during testing"},
        {"role": "assistant", "content": "That's a critical warning sign - could be a short or wrong impedance speaker. Check the speaker ohms matches what the MAX98357 expects."},
    ],
    [
        {"role": "user", "content": "my python script keeps hanging on the ollama call and never times out"},
        {"role": "assistant", "content": "The ollama SDK defaults to timeout=None, so httpx waits forever. Pass an explicit httpx.Timeout to the client constructor."},
        {"role": "user", "content": "ok that fixed the hang. why didn't asyncio.wait_for catch it?"},
        {"role": "assistant", "content": "Because the call was running in a thread-pool executor (run_in_executor), which asyncio can't cancel. The timeout has to be enforced at the HTTP client, not the asyncio layer."},
    ],
    [
        {"role": "user", "content": "every assistant message is getting saved twice in the DB"},
        {"role": "assistant", "content": "Sounds like a race. _persist_message is fire-and-forget, so the memory ids aren't populated when the enqueue runs and a duplicate row gets inserted."},
        {"role": "user", "content": "how do we fix it cleanly"},
        {"role": "assistant", "content": "Track the persist tasks and await them before the enqueue reads the ids. Make the enqueue async and flush first."},
    ],
    [
        {"role": "user", "content": "search keeps returning almost nothing now that I have 31k memories"},
        {"role": "assistant", "content": "Your filters compound: similarity 0.5, then rank>=3, then a score floor. On a big corpus that removes ~85% of candidates. Lower the similarity threshold and replace the rank filter with an importance floor."},
        {"role": "user", "content": "that helped a lot. anything else?"},
        {"role": "assistant", "content": "Add a recency boost and give FTS-only hits a baseline similarity so keyword matches aren't dropped for having no vector score."},
    ],
    [
        {"role": "user", "content": "should I use a single model for all the background memory tasks?"},
        {"role": "assistant", "content": "Consolidating on one reasoning model (qwen3:14b) avoids constant model swaps, which dominate latency on local. Keep summarization on a separate fast model only if cloud is reliable."},
        {"role": "user", "content": "makes sense, swapping was killing throughput"},
        {"role": "assistant", "content": "Right — swap overhead is per-call, so batching same-model tasks together matters more than picking the theoretically-best model per task."},
    ],
    [
        {"role": "user", "content": "the entity graph has tons of near-duplicate entities like 'esp32' and 'ESP32'"},
        {"role": "assistant", "content": "Run a lexical-variant merge pass first (case/punctuation), then an embedding-similarity merge with a version guard so concurrent writes don't clobber. Archive, never hard-delete, so re-mentions can revive."},
        {"role": "user", "content": "why archive instead of delete?"},
        {"role": "assistant", "content": "Because a merged-away name can reappear in a later conversation; if you hard-deleted it the mention strands. Soft-archive lets you revive it on re-mention."},
    ],
    [
        {"role": "user", "content": "my cloud summarization keeps hitting 429s and falling over"},
        {"role": "assistant", "content": "The free tier has a 20-request burst limit. Add a per-endpoint rate limit and cascade to the next endpoint on RateLimitExhausted instead of failing the request."},
        {"role": "user", "content": "and if all cloud is down?"},
        {"role": "assistant", "content": "Fall back to a local model. Make sure the fallback path passes num_ctx from the endpoint config or you'll truncate context silently."},
    ],
    [
        {"role": "user", "content": "the executor keeps making 37 tool calls and never finishing"},
        {"role": "assistant", "content": "It's missing a completion signal. Add a task_complete tool instead of a magic string, inject a state block each turn, and a budget wind-down message at 80% so it wraps up."},
        {"role": "user", "content": "that's basically what claude code does right?"},
        {"role": "assistant", "content": "Yes — explicit completion tool, state awareness, and bounded tool output are the common pattern across the successful coding agents."},
    ],
    [
        {"role": "user", "content": "what's the cleanest way to test the chat loop without ollama?"},
        {"role": "assistant", "content": "Use a scripted LLM client that feeds canned tool-call/text turns into the real loop. That validates completion detection and tool wiring deterministically; reserve the Ollama PC for actual model-quality runs."},
        {"role": "user", "content": "so logic here, quality there"},
        {"role": "assistant", "content": "Exactly — separating wiring tests from model-behavior tests is the key mental model."},
    ],
    [
        {"role": "user", "content": "vision input — does minimax take video?"},
        {"role": "assistant", "content": "It generates video but does not accept video as input. Image input works on vision-capable models; declare both name forms in model_settings because the registry base-name match splits on ':' or '/'."},
        {"role": "user", "content": "got it, image only"},
        {"role": "assistant", "content": "Right. There's no video-input path; don't assume symmetry between generate and accept."},
    ],
]

# Short labels for table rows (derived so they always match the data length).
MESSAGE_LABELS = [m["label"] for m in TEST_MESSAGES]

# ---------------------------------------------------------------------------
# Test data — Entity Extraction. Each case carries the set of entities a good
# extraction should surface; scoring is F1 of extracted vs expected (lowercased).
# A trivial message with no real entities expects the empty set.
# ---------------------------------------------------------------------------
ENTITY_CASES = [
    {"summary": "User decided to use a two-module design for the desk robot with JST connectors.",
     "expected": {"user", "desk robot", "jst connectors"}},
    {"summary": "Assistant explained how to configure Ollama with GPU acceleration on an RTX 4090.",
     "expected": {"ollama", "gpu", "rtx 4090"}},
    {"summary": "User's daughter Lily has a Minecraft performance issue on her HP laptop.",
     "expected": {"lily", "minecraft", "hp laptop"}},
    {"summary": "User migrated the vector store from ChromaDB to sqlite-vec.",
     "expected": {"user", "chromadb", "sqlite-vec"}},
    {"summary": "The MAX98357 amplifier produced noise until the speaker was changed to 4 ohm.",
     "expected": {"max98357", "speaker"}},
    {"summary": "User works at Xanatek and prefers Python for backend work.",
     "expected": {"user", "xanatek", "python"}},
    {"summary": "Groq runs llama-3.3-70b for the ranking task at about 30 requests per minute.",
     "expected": {"groq", "llama-3.3-70b", "ranking"}},
    {"summary": "Assistant recommended JST-PH connectors over dupont wires for the ESP32.",
     "expected": {"jst-ph connectors", "dupont wires", "esp32"}},
    {"summary": "User scheduled a benchmark run on the Ollama PC for qwen3:14b.",
     "expected": {"user", "ollama pc", "qwen3:14b"}},
    {"summary": "Presidio with spaCy handles PII detection before cloud calls.",
     "expected": {"presidio", "spacy", "pii"}},
    {"summary": "User said hello.", "expected": set()},
    {"summary": "ok thanks, talk later", "expected": set()},
]

# Compact labels for the standalone table view (derived to match the data).
ENTITY_LABELS = [
    (c["summary"][:24] + "…") if len(c["summary"]) > 25 else c["summary"]
    for c in ENTITY_CASES
]

# ---------------------------------------------------------------------------
# Test data — Contradiction Detection. Balanced (6 true contradictions, 6 not).
# ---------------------------------------------------------------------------
CONTRADICTION_PAIRS = [
    ("User prefers dark mode", "User prefers light mode", True),
    ("User uses Windows 10", "User switched to Linux full-time", True),
    ("User likes Python", "User dislikes Python", True),
    ("The vector store is ChromaDB", "The vector store is sqlite-vec", True),
    ("User lives in Ohio", "User moved to Texas", True),
    ("The speaker is 8 ohm", "The speaker is 4 ohm", True),
    ("User likes coffee", "User likes tea", False),
    ("User has a cat named Luna", "User works at Xanatek", False),
    ("User knows Python", "User also knows Rust", False),
    ("The desk robot uses an ESP32", "The desk robot uses JST connectors", False),
    ("User prefers dark mode", "User prefers tabs over spaces", False),
    ("Ranking runs on Groq", "Summarization runs on a local model", False),
]

CONTRADICTION_LABELS = [
    "dark/light mode", "Win10/Linux", "likes/dislikes Python",
    "Chroma/sqlite-vec", "Ohio/Texas", "8ohm/4ohm",
    "coffee & tea", "cat & job", "Python & Rust",
    "esp32 & jst", "darkmode & tabs", "groq & local",
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
        results.append({"raw": raw, "parsed": rank, "truth_rank": msg["truth_rank"], "time": round(elapsed, 2)})
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
        results.append({"raw": raw, "parsed": score, "truth_importance": msg["truth_importance"], "time": round(elapsed, 2)})
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
    for case in ENTITY_CASES:
        sys_prompt, user_prompt = extract_entities(case["summary"])
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
            "extracted": entities,
            "expected": case["expected"],
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
    """Measure the LIVE memory pipeline's combined scoring call.

    Uses `rank_importance_and_classify` + RANKING_IMPORTANCE, matching
    processor.py:185 — the call that runs on every message. This previously used
    `rank_and_importance` + RANKING, which only the import/reprocess paths use
    (import_common.py:361), so `models.ranking_importance` — the busiest key in
    the config — had no valid number while appearing to have one.

    The extra `type` field the live prompt asks for is parsed and discarded here:
    scoring rank and importance keeps this comparable to what the column has
    always meant, and misclassification is a separate concern from miscalibration.
    """
    results = []
    for msg in TEST_MESSAGES:
        sys_prompt, user_prompt = rank_importance_and_classify(msg["content"])
        start = time.perf_counter()
        try:
            raw = await router.generate(
                TaskType.RANKING_IMPORTANCE, user_prompt, system=sys_prompt,
                think=False,
            )
            rank, importance, _mem_type = (
                MemoryProcessor._parse_rank_importance_type(raw)
            )
        except Exception as e:
            raw = f"ERROR: {e}"
            rank = -1
            importance = -1.0
        elapsed = time.perf_counter() - start
        results.append({
            "raw": raw,
            "rank": rank,
            "importance": importance,
            # score_rank_and_importance filters on these two keys, so without
            # them both of its sub-scores got an empty list and the job returned
            # None -- for every model, in every run ever recorded. The column was
            # invisible in the report (build_report skips categories nothing
            # scored), so the busiest key in the config looked simply unmeasured.
            "truth_rank": msg["truth_rank"],
            "truth_importance": msg["truth_importance"],
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
            ents = ", ".join(r["extracted"]) if r["extracted"] else "(none)"
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
