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
import re
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.prompts import reflect_on_response, structured_compaction_prompt
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import AgentConfig, EndpointConfig, LLMConfig, ModelsConfig

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
        "name": "context_compaction",
        "description": "Compact a conversation without losing decisions or open work",
        # Replaces a plan_generation case built on generate_plan(), which had
        # ZERO production callers (only this suite imported it) — the reasoning
        # key was partly graded on a prompt nothing runs. structured_compaction_prompt
        # IS live: chat_loop.py:520 calls it through TaskType.REASONING whenever a
        # conversation overflows, and losing a decision there corrupts every
        # later turn.
        "prompt_fn": "structured_compaction",
        "input": (
            "User: The nightly entity merge archived 7491 entities last run. I want to "
            "re-run it but I'm worried about the version-numbered ones.\n"
            "Assistant: projectecho_v1 and projectecho_v2 embed at 0.996 cosine, well "
            "above every threshold, so an embedding-only guard would merge them. "
            "entity_names.version_distinguished blocks any merge where names differ in a "
            "version or instance number.\n"
            "User: Is that guard on both merge paths?\n"
            "Assistant: No — until 2026-08-07 it lived on EntityMerger only, which is the "
            "path that ships disabled. Creation-time resolution, the enabled one, had no "
            "guard. Both use blipshell/memory/entity_names.py now.\n"
            "User: OK. Decision: re-run the merge, but keep the ARCHIVE-never-DELETE rule "
            "and take a backup first. Don't touch cleanup_entities.py, it hard-deletes.\n"
            "Assistant: Understood. I'll back up data/blipshell.db, run merge_entities in "
            "dry-run, then apply with --loop.\n"
            "User: Also the get_all_entity_names cache — anything writing entity rows "
            "outside the store's own methods has to call _invalidate_entity_name_cache."
        ),
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
    {
        "name": "tradeoff_analysis",
        "description": "Reason about a design tradeoff",
        "system": "You are a pragmatic software architect. Be concise and specific.",
        "prompt": "For a single-user local assistant with 30k memories, compare a dual-store setup (SQLite + a separate vector DB) vs a single store (sqlite-vec inside the SQLite file). What's the main risk of the dual-store approach and when is it nonetheless justified?",
    },
    {
        "name": "root_cause",
        "description": "Diagnose a described bug from symptoms",
        "system": "You are a debugging expert. Identify the most likely root cause and the fix.",
        "prompt": "A python asyncio call to a thread-pool executor never times out even though it's wrapped in asyncio.wait_for. Why doesn't the timeout fire, and what's the correct fix?",
    },
    {
        "name": "prioritization",
        "description": "Rank competing options with justification",
        "system": "Be concise. Give a ranked list with one-line justifications.",
        "prompt": "I have limited GPU. Rank these background jobs by how much they need a large local model vs a small one: memory ranking (1-5), summarization, entity extraction, importance scoring, contradiction detection. Justify each.",
    },
    {
        "name": "ambiguity_handling",
        "description": "Identify missing info before answering",
        "system": "Be concise and specific.",
        "prompt": "A user says 'the search is broken, fix it.' Before changing code, what are the 3 most important clarifying questions, and what would each answer change about your approach?",
    },
    {
        "name": "math_logic",
        "description": "Quantitative reasoning",
        "system": "Show the calculation briefly, then give the number.",
        "prompt": "A free API tier allows 20 requests per minute. You must process 5,000 memories, each needing 1 summarization call. Ignoring failures, what's the minimum wall-clock time in minutes, and what's one way to cut it roughly in half?",
    },
    {
        "name": "spec_compliance",
        "description": "Follow precise output constraints",
        "system": "Follow the format EXACTLY.",
        "prompt": "List exactly 4 reasons local models are attractive for a personal assistant. Output ONLY a numbered list 1-4, each reason 6 words or fewer, no preamble, no trailing commentary.",
    },
    {
        "name": "counterfactual",
        "description": "Reason about consequences of a change",
        "system": "Be concise and specific.",
        "prompt": "If a memory pipeline switched its embedding model but did NOT re-embed the existing corpus, what specifically would degrade, and why would search appear to 'mostly work' while quietly getting worse?",
    },
    {
        "name": "explain_to_level",
        "description": "Calibrate explanation to an audience",
        "system": "Be concise.",
        "prompt": "Explain Reciprocal Rank Fusion (RRF) for combining keyword and vector search results to a competent developer who has never used it. Include the formula and why the constant k matters, in under 120 words.",
    },
    {
        "name": "self_reflection_correct",
        "description": "Avoid false-positive correction on a CORRECT answer",
        "prompt_fn": "reflect_on_response",
        "user_message": "What's the default I2C address of most SSD1306 OLED displays?",
        "response": "The default I2C address for most SSD1306 OLED displays is 0x3C.",
        # This response is CORRECT — a good model should NOT invent an error.
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
    {
        "name": "fix_the_bug",
        "description": "Produce a corrected version of buggy code",
        "system": "Return only the corrected function. No markdown fences, no explanation.",
        "prompt": (
            "This function should return the number of retries actually used, but it's wrong. Fix it:\n\n"
            "def attempts_used(max_retries, succeeded_on):\n"
            "    # succeeded_on is the 1-based attempt that succeeded, or None if all failed\n"
            "    if succeeded_on:\n"
            "        return succeeded_on - 1\n"
            "    return max_retries\n"
            "# Bug: when it succeeds on attempt 1, 0 retries is right, but when succeeded_on is None\n"
            "# it returns max_retries when it should be max_retries (correct) — the real bug is the\n"
            "# off-by-one: a success on attempt N used N-1 retries, but the None branch and the truthiness\n"
            "# check on succeeded_on==0 are wrong. Fix the edge cases."
        ),
    },
    {
        "name": "generate_with_tests",
        "description": "Generate a function plus a test",
        "system": "Return Python code only: the function and one pytest-style test. No prose.",
        "prompt": "Write `clamp_unit(x: float) -> float | None` that clamps to [0,1], normalizes a value >1 and <=10 by /10, >10 and <=100 by /100, returns None for >100 or NaN or <0 after normalization. Include one pytest test covering the normalization and the None cases.",
    },
    {
        "name": "explain_code",
        "description": "Explain what a snippet does and spot the subtle issue",
        "system": "Be concise and specific.",
        "prompt": (
            "What does this do, and what's the subtle correctness problem?\n\n"
            "seen = set()\n"
            "def dedupe(items):\n"
            "    out = []\n"
            "    for it in items:\n"
            "        if it not in seen:\n"
            "            seen.add(it)\n"
            "            out.append(it)\n"
            "    return out\n"
        ),
    },
    {
        "name": "api_usage",
        "description": "Use an API correctly under a constraint",
        "system": "Return only the code.",
        "prompt": "Write a Python snippet that calls an httpx client with a hard 30-second total timeout that WILL actually fire even if the server hangs. Show the client construction and one GET call.",
    },
    {
        "name": "regex_task",
        "description": "Write a precise regex with explanation",
        "system": "Give the regex on its own line, then one sentence explaining it.",
        "prompt": "Write a Python regex that extracts a float score from text like `{\"score\": 0.82, ...}` OR `score = .9` OR `Score: 1`. It must capture the number into group 1 and tolerate missing leading zero.",
    },
    {
        "name": "refactor_suggestion",
        "description": "Propose a concrete refactor",
        "system": "Be concise; give the refactored shape.",
        "prompt": "This function does keyword filtering to decide which tools to pass to a model. Successful coding agents don't gate tools this way. Briefly explain why, and what to do instead.",
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
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file by replacing text",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old": {"type": "string"}, "new": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file with the given content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to create"},
                    "content": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path"}},
                "required": ["path"],
            },
        },
    },
]

# expected_args: each key must appear in the call's args (exact or loose key
# match) with the model's value containing the expected substring (case-insensitive).
TOOL_CALLING_TESTS = [
    {"name": "web_search", "message": "Search the web for an ESP32 MAX98357 I2S wiring diagram",
     "expected_tool": "web_search", "expected_args": {"query": "esp32"}},
    {"name": "read_file", "message": "Read the contents of worker.py",
     "expected_tool": "read_file", "expected_args": {"path": "worker.py"}},
    {"name": "memory_recall", "message": "What did we discuss last time about the desk robot connectors?",
     "expected_tool": "search_memories", "expected_args": {"query": "desk robot"}},
    {"name": "price_lookup", "message": "Look up the current price of a Raspberry Pi 5",
     "expected_tool": "web_search", "expected_args": {"query": "raspberry pi"}},
    {"name": "run_pytest", "message": "Run the test suite with pytest in this repo",
     "expected_tool": "run_command", "expected_args": {"command": "pytest"}},
    {"name": "edit_config", "message": "In config.yaml, change the server port to 9000",
     "expected_tool": ("edit_file", "read_file"), "expected_args": {"path": "config.yaml"}},
    {"name": "list_dir", "message": "List the files in the blipshell/memory folder",
     "expected_tool": "list_directory", "expected_args": {"path": "memory"}},
    {"name": "create_file", "message": "Create a new file notes.md containing a short todo list",
     "expected_tool": "write_file", "expected_args": {"path": "notes.md"}},
    {"name": "read_readme", "message": "Show me what's in README.md",
     "expected_tool": "read_file", "expected_args": {"path": "readme.md"}},
    {"name": "news_search", "message": "Find the latest news on the qwen3 model release",
     "expected_tool": "web_search", "expected_args": {"query": "qwen3"}},
    {"name": "recall_fact", "message": "Remind me what my daughter's name is",
     "expected_tool": "search_memories", "expected_args": {"query": "daughter"}},
    {"name": "git_status", "message": "Check the git status of the repository",
     "expected_tool": "run_command", "expected_args": {"command": "git status"}},
    {"name": "open_cli", "message": "Open blipshell/ui/cli.py so I can see it",
     "expected_tool": "read_file", "expected_args": {"path": "cli.py"}},
    {"name": "concept_search", "message": "Search online for how reciprocal rank fusion (RRF) works",
     "expected_tool": "web_search", "expected_args": {"query": "rank fusion"}},
    {"name": "edit_doc", "message": "Fix the typo in docs/SYSTEM_REVIEW.md",
     "expected_tool": ("edit_file", "read_file"), "expected_args": {"path": "system_review.md"}},
]

TOOL_LABELS = [t["name"] for t in TOOL_CALLING_TESTS]

# The production interactive path ALWAYS sends a system prompt: agent_chat.py
# builds one starting from AgentConfig.system_prompt before every ChatLoop turn.
# This suite used to send a bare user turn with tools attached and no system
# message at all, which measures "unprompted tool eagerness" — a condition
# BlipShell never runs. It penalised cautious models (minimax-m3 asked for a
# full path instead of calling read_file on "worker.py") and rewarded eager
# ones, and it inverted the ranking against qwen3:14b.
#
# We use the AgentConfig DEFAULT rather than the loaded config.agent.system_prompt
# on purpose: benchmark results are a committed corpus compared across two
# machines and across months, so the condition must be a constant in code, not
# whatever config.yaml happens to say on the box that ran it.
TOOL_CALLING_SYSTEM = AgentConfig().system_prompt

# How many assistant turns a tool-calling case gets before it is scored a miss.
# Single-turn scoring could not evaluate an agentic model: asked to read
# worker.py, minimax-m3 answered `run_command find . -name "worker.py"`, and
# asked for README.md it answered `list_directory .`. Both are correct first
# moves — locate, then read — and both scored as failures. Because a model
# picks stochastically among several reasonable openers and only one counted,
# that scoring WAS the measurement noise (sd ~0.09 across repeats), which is
# large enough to have driven a routing decision off a coin flip.
TOOL_CALLING_MAX_TURNS = 4

# A small fake workspace so orienting calls return something usable and the
# model can actually reach the goal. Contents are deliberately boring; the
# point is that list/find/read return consistent, plausible results.
_FAKE_FILES = {
    "readme.md": "# BlipShell\n\nLocal LLM assistant with persistent memory.\n",
    "worker.py": "class MemoryWorker:\n    def run(self):\n        pass\n",
    "config.yaml": "server:\n  port: 8080\n  host: 0.0.0.0\n",
    "notes.md": "",
    "docs/system_review.md": "# System Review\n\nThe retreival layer needs work.\n",
    "blipshell/ui/cli.py": "import click\n\n@click.group()\ndef main():\n    pass\n",
}
_FAKE_DIRS = {
    ".": ["README.md", "worker.py", "config.yaml", "docs/", "blipshell/", "tests/"],
    "docs": ["SYSTEM_REVIEW.md", "HISTORY.md", "V2_PLAN.md"],
    "blipshell": ["ui/", "memory/", "core/", "llm/"],
    "blipshell/memory": ["sqlite_store.py", "vector_store.py", "search.py", "processor.py"],
    "memory": ["sqlite_store.py", "vector_store.py", "search.py", "processor.py"],
    "blipshell/ui": ["cli.py", "views/", "commands/"],
}


def _fake_tool_result(name: str, args: dict) -> str:
    """Plausible, consistent result for a mock tool call."""
    args = args if isinstance(args, dict) else {}

    def val(*keys):
        return next((str(args[k]) for k in keys if k in args), "")

    if name == "list_directory":
        path = val("path", "directory", "dir").strip("./") or "."
        entries = _FAKE_DIRS.get(path) or _FAKE_DIRS.get(path.rstrip("/")) or _FAKE_DIRS["."]
        return "\n".join(entries)
    if name == "read_file":
        path = val("path", "file_path", "file").lstrip("./").lower()
        body = _FAKE_FILES.get(path)
        if body is None:
            body = _FAKE_FILES.get(path.split("/")[-1])
        return body if body is not None else "Error: %s: no such file" % path
    if name == "run_command":
        low = val("command", "cmd").lower()
        if "pytest" in low:
            return "===== 1546 passed, 3 skipped in 120.5s ====="
        if "git status" in low:
            return "On branch main\nnothing to commit, working tree clean"
        if low.startswith(("find", "ls", "dir")) or " find " in low:
            return "./README.md\n./worker.py\n./config.yaml\n./docs/SYSTEM_REVIEW.md"
        return "(command completed, no output)"
    if name == "web_search":
        q = val("query", "q")
        return "1. Result about %s\n2. Another result about %s\n3. Forum thread about %s" % (q, q, q)
    if name == "search_memories":
        q = val("query", "q")
        return "Memory 1: earlier discussion about %s\nMemory 2: a follow-up about %s" % (q, q)
    if name in ("edit_file", "write_file"):
        return "OK: file written"
    return "OK"

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUOTED = re.compile(r"""'[^']*'|"[^"]*\"""")


def _command_invokes(command: str, expected: str) -> bool:
    """True if `command` actually INVOKES `expected`, not merely mentions it.

    Substring matching produced false positives that inflated scores: asked to
    run the test suite, a model answered
    `find . -name "pytest.ini" -o -name "pyproject.toml"` and scored a pass,
    because "pytest" appears inside a quoted filename. We strip quoted regions
    first, then require the expected token to appear on a word boundary, so
    `pytest`, `python -m pytest` and `ls && pytest -q` pass while
    `find -name "pytest.ini"` does not.
    """
    stripped = _QUOTED.sub(" ", command)
    # Trailing "." must also block: an unquoted `grep -r pytest.ini .` mentions
    # pytest inside a filename just as surely as a quoted one does.
    pattern = r"(?<![\w./-])" + r"[\s]+".join(re.escape(p) for p in expected.split()) + r"(?![\w.-])"
    return re.search(pattern, stripped) is not None


def _tool_call_matches(called: dict, expected_tool, expected_args: dict | None) -> bool:
    """True if a tool call has the right name AND every required arg.

    `expected_tool` is a name or a tuple of ACCEPTABLE names. A tuple is used
    where more than one first action is correct under the production system
    prompt: its rule 5 is "Read a file before editing it", so answering an edit
    request with read_file on the right path is obedient, not wrong. Scoring is
    single-turn and cannot see the follow-up edit, so penalising it would
    measure the opposite of the behaviour we ask for. The expected_args path
    check still holds — the model must target the right file either way.

    For each expected arg, the call must include that key (exact or loose
    substring key match) and the value must contain the expected substring
    (case-insensitive) — lenient on phrasing, strict on whether the model
    actually supplied the right argument.
    """
    acceptable = (expected_tool,) if isinstance(expected_tool, str) else tuple(expected_tool)
    if called.get("name") not in acceptable:
        return False
    if not expected_args:
        return True
    args = called.get("args")
    if not isinstance(args, dict):
        return False
    lowered = {str(k).lower(): str(v).lower() for k, v in args.items()}
    for key, want in expected_args.items():
        actual = lowered.get(key.lower())
        if actual is None:  # loose key match (e.g. "file_path" contains "path")
            actual = next((v for k, v in lowered.items() if key.lower() in k), None)
        if actual is None:
            return False
        if key.lower() in ("command", "cmd"):
            if not _command_invokes(actual, str(want).lower()):
                return False
        elif str(want).lower() not in actual:
            return False
    return True


def parse_model_spec(spec: str) -> tuple[str, dict]:
    """Parse a model spec like 'gpt-oss:latest/low' into (model_name, extra_options).

    Supported suffixes:
      /low, /medium, /high  — reasoning_effort (for gpt-oss)
      /nothink              — disable thinking (think=False)
    Returns (model_name, options_dict).
    """
    if "/" in spec:
        parts = spec.rsplit("/", 1)
        model_name = parts[0]
        suffix = parts[1].lower()
        if suffix in ("low", "medium", "high"):
            return model_name, {"reasoning_effort": suffix}
        if suffix == "nothink":
            return model_name, {"nothink": True}
        # Not a known suffix — treat whole string as model name
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
    """Call router.generate(), passing reasoning_effort via the think parameter.

    GPT-OSS and similar models accept think="low"/"medium"/"high" to control
    thinking depth. Other models ignore the think parameter gracefully.
    """
    if not extra_options:
        return await router.generate(task_type, prompt, system=system)

    # /nothink — disable thinking
    if extra_options.get("nothink"):
        return await router.generate(task_type, prompt, system=system, think=False)

    # /low /medium /high — reasoning effort (gpt-oss)
    effort = extra_options.get("reasoning_effort")
    if effort:
        return await router.generate(task_type, prompt, system=system, think=effort)

    return await router.generate(task_type, prompt, system=system)


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


# Tool-call parsing is PRODUCTION'S — never a local copy. This suite used to
# carry its own two-tuple extractor that didn't parse OpenAI-style
# arguments-as-JSON-string, so every openai-provider candidate scored exactly
# 0.000 on tool_calling (deepseek-v4-flash and minimax-m3:free, 2026-09-02)
# while tool-calling fine through the same endpoint in production: the args
# stayed a string, and _tool_call_matches requires a dict. A benchmark that
# parses responses differently from ChatLoop measures the parser, not the
# model. (tests/benchmark_agent_buggy.py still has a copy; nothing loads it.)
from blipshell.core.chat_loop import extract_tool_call_info  # noqa: E402


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

async def benchmark_reasoning(router: LLMRouter, extra_options: dict | None = None) -> list[dict]:
    results = []
    for test in REASONING_TESTS:
        start = time.perf_counter()
        try:
            if test.get("prompt_fn") == "structured_compaction":
                system, prompt = structured_compaction_prompt(test["input"])
                response = await _generate_with_options(
                    router, TaskType.REASONING, prompt, system=system,
                    extra_options=extra_options,
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
    """Score whether the model REACHES the expected tool call within a few turns.

    Runs a real multi-turn loop against a fake workspace (see _fake_tool_result)
    with the production system prompt, mirroring how ChatLoop actually drives a
    model. A case passes as soon as any turn emits the expected call; orienting
    first (list/find, or read-before-edit) costs turns but is not a failure.
    `turns` is recorded per case so a model that goes straight to the goal is
    distinguishable from one that wanders there.
    """
    model, client = await router.get_model_and_client(TaskType.TOOL_CALLING)
    if not client:
        return [{"error": "No client available", "time": 0}] * len(TOOL_CALLING_TESTS)

    results = []
    for test in TOOL_CALLING_TESTS:
        messages = [
            {"role": "system", "content": TOOL_CALLING_SYSTEM},
            {"role": "user", "content": test["message"]},
        ]
        start = time.perf_counter()
        chat_kwargs = {}
        if extra_options:
            if extra_options.get("nothink"):
                chat_kwargs["think"] = False
            effort = extra_options.get("reasoning_effort")
            if effort:
                chat_kwargs["think"] = effort
        called_tools: list[dict] = []
        correct = False
        turns = 0
        content = ""
        try:
            for turn in range(TOOL_CALLING_MAX_TURNS):
                turns = turn + 1
                response = await client.chat(
                    messages=messages, model=model, tools=MOCK_TOOLS, **chat_kwargs)
                content, tool_calls = extract_response(response)

                turn_calls = []
                for tc in (tool_calls or []):
                    name, args, tc_id = extract_tool_call_info(tc)
                    turn_calls.append({"name": name, "args": args, "id": tc_id})
                called_tools.extend(turn_calls)

                if any(_tool_call_matches(t, test["expected_tool"], test.get("expected_args"))
                       for t in turn_calls):
                    correct = True
                    break
                if not turn_calls:
                    # Answered in prose with no tool call — it has stopped acting,
                    # so more turns would only re-ask the same question.
                    break

                # Feed results back and let it continue, exactly as ChatLoop does.
                messages.append({"role": "assistant", "content": content or "",
                                 "tool_calls": tool_calls})
                for t in turn_calls:
                    msg = {"role": "tool", "name": t["name"],
                           "content": _fake_tool_result(t["name"], t["args"])}
                    if t["id"]:  # OpenAI-compat endpoints require it; Ollama ignores it
                        msg["tool_call_id"] = t["id"]
                    messages.append(msg)

            result = {
                "content": content,
                "tool_calls": called_tools,
                "expected": test["expected_tool"],
                "expected_args": test.get("expected_args"),
                "correct": correct,
                "turns": turns,
                "time": round(time.perf_counter() - start, 2),
            }
        except Exception as e:
            result = {
                "content": f"ERROR: {e}",
                "tool_calls": called_tools,
                "expected": test["expected_tool"],
                "correct": False,
                "turns": turns,
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
