"""Mock router for benchmark testing without a live LLM.

Returns deterministic canned responses that exercise all scoring code paths.
Each role registers its mock responses via MOCK_RESPONSES.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Mock response registry: (task_type, prompt_pattern) → response
# Patterns are checked in order; first match wins.
# ---------------------------------------------------------------------------

# Each entry: (task_type_or_None, compiled_regex_on_user_prompt, response_str)
_MOCK_RULES: list[tuple[str | None, re.Pattern, str]] = []


def register_mock(task_type: str | None, prompt_pattern: str, response: str) -> None:
    """Register a mock response. task_type=None matches any."""
    _MOCK_RULES.append((task_type, re.compile(prompt_pattern, re.IGNORECASE | re.DOTALL), response))


def get_mock_response(task_type: str, user_prompt: str) -> str:
    """Look up a canned response for the given prompt."""
    for rule_type, pattern, response in _MOCK_RULES:
        if rule_type is not None and rule_type != task_type:
            continue
        if pattern.search(user_prompt):
            return response
    # Default fallback per task type
    return _DEFAULTS.get(task_type, "Mock response.")


_DEFAULTS: dict[str, str] = {
    "summarization": "User discussed a technical topic.",
    "ranking_importance": "3 0.5 conversation",
    "reasoning": "ADD",
}


class MockRouter:
    """Drop-in replacement for LLMRouter that returns canned responses."""

    async def generate(
        self,
        task_type: str,
        prompt: str,
        system: str | None = None,
        think: bool | str | None = None,
    ) -> str:
        return get_mock_response(task_type, prompt)


# ---------------------------------------------------------------------------
# Role 1: Summarization mock responses
# ---------------------------------------------------------------------------

# Filler/greetings → SKIP (no ^ anchor — content is inside the prompt template)
register_mock("summarization", r"\b(hey|hi there|hello)\b", "SKIP")
register_mock("summarization", r"ok thanks", "SKIP")
# System noise → SKIP
register_mock("summarization", r"System:.*important_rules", "SKIP")
# Short trivial → SKIP
register_mock("summarization", r"sanding paint", "SKIP")
# Technical content → 3rd person summary ≤30 words
register_mock("summarization", r"MAX98357.*esp32",
              "User reports garbage audio output from MAX98357 I2S amplifier connected to ESP32.")
register_mock("summarization", r"minecraft.*blocks",
              "User's daughter's laptop loads only 10 chunks in Minecraft, causing poor performance.")
register_mock("summarization", r"code review.*worker\.py",
              "Code review of worker.py identified missing stop call, no retry logic, and no model fallback.")
register_mock("summarization", r"two-module design.*desk robot",
              "User chose two-module design for desk robot with main board, sensor sidecar, and JST connectors.")

# ---------------------------------------------------------------------------
# Role 2: Scoring mock responses
# ---------------------------------------------------------------------------

register_mock("ranking_importance", r"\b(hey|hi there|hello)\b", "1 0.1 conversation")
register_mock("ranking_importance", r"ok thanks", "1 0.1 conversation")
register_mock("ranking_importance", r"System:.*important_rules", "1 0.1 conversation")
register_mock("ranking_importance", r"MAX98357.*esp32", "4 0.6 fact")
register_mock("ranking_importance", r"minecraft", "4 0.7 fact")
register_mock("ranking_importance", r"code review.*worker", "3 0.5 skill")
register_mock("ranking_importance", r"two-module design.*desk robot", "5 0.8 preference")
register_mock("ranking_importance", r"sanding paint", "2 0.3 conversation")

# ---------------------------------------------------------------------------
# Role 3: Dedup mock responses
# ---------------------------------------------------------------------------

# Unrelated → ADD
register_mock("reasoning", r"Raspberry Pi 4.*media server.*RTX 3070", "ADD")
register_mock("reasoning", r"dark roast coffee.*espresso machine", "ADD")
register_mock("reasoning", r"learning Rust.*Python", "ADD")
# Redundant → NONE
register_mock("reasoning", r"Python.*main programming.*primarily codes in Python", "NONE")
register_mock("reasoning", r"cat named Luna.*cat is called Luna", "NONE")
# Update → UPDATE N
register_mock("reasoning", r"RTX 3070.*32GB RAM.*Ryzen.*RTX 3070 for gaming", "UPDATE 1")
register_mock("reasoning", r"switched from VS Code to Neovim.*VS Code.*primary", "UPDATE 1")
register_mock("reasoning", r"completed the Rust CLI.*started learning Rust", "UPDATE 1")
# Delete → DELETE N
register_mock("reasoning", r"sold their RTX 3070.*MacBook.*desktop PC.*RTX 3070", "DELETE 1")
register_mock("reasoning", r"quit drinking coffee.*drinks coffee every morning", "DELETE 1")

# ---------------------------------------------------------------------------
# Role 4: Contradiction detection mock responses
# ---------------------------------------------------------------------------

register_mock("reasoning", r"(dark mode|light mode).*(light mode|dark mode)", "YES")
register_mock("reasoning", r"(Windows 10|Windows 11).*(Windows 11|Windows 10)", "YES")
register_mock("reasoning", r"(likes Python|dislikes Python).*(dislikes Python|likes Python)", "YES")
register_mock("reasoning", r"coffee.*tea", "NO")
register_mock("reasoning", r"(cat named|job at)", "NO")
register_mock("reasoning", r"knows Python.*knows Rust", "NO")

# ---------------------------------------------------------------------------
# Role 5: Lesson extraction mock responses
# ---------------------------------------------------------------------------

register_mock("reasoning", r"Extract lessons.*ESP32.*connector",
              "User prefers detailed troubleshooting steps over lengthy explanations.\n"
              "Always clarify which hardware project the user is referring to.")
register_mock("reasoning", r"Extract lessons.*Python code.*running really slowly",
              "User sometimes forgets standard library solutions — consider suggesting them proactively.")
register_mock("reasoning", r"Extract lessons.*React or Vue.*dashboard",
              "Use the user's existing knowledge when recommending tools rather than introducing new ones.")
register_mock("reasoning", r"Extract lessons.*(hey|ok thanks|hi there)",
              "SKIP")

# ---------------------------------------------------------------------------
# Role 6: Entity extraction mock responses
# ---------------------------------------------------------------------------

register_mock("reasoning", r"Extract entity triples.*Python.*performance",
              "user | uses | python | person | technology\n"
              "user | asked_about | data analysis | person | concept")
register_mock("reasoning", r"Extract entity triples.*desk robot",
              "user | builds | desk robot | person | project\n"
              "desk robot | uses | ESP32 | project | technology\n"
              "desk robot | has | JST connectors | project | technology")
register_mock("reasoning", r"Extract entity triples.*Ollama",
              "user | asked_about | ollama | person | technology\n"
              "ollama | uses | gpu | technology | technology")
register_mock("reasoning", r"Extract entity triples.*Minecraft",
              "user | plays | minecraft | person | technology")
register_mock("reasoning", r"Extract entity triples.*hello",
              "NONE")

# ---------------------------------------------------------------------------
# Role 7: Entity resolution mock responses
# ---------------------------------------------------------------------------

register_mock("reasoning", r"Entity A: PostgreSQL.*Entity B: postgres", "YES")
register_mock("reasoning", r"Entity A: JavaScript.*Entity B: JS", "YES")
register_mock("reasoning", r"Entity A: Jim Buschman.*Entity B: Jim B", "YES")
register_mock("reasoning", r"Entity A: VS Code.*Entity B: Visual Studio Code", "YES")
register_mock("reasoning", r"Entity A: Python.*Entity B: python", "YES")
register_mock("reasoning", r"Entity A: Mercury.*Entity B: mercury", "YES")
register_mock("reasoning", r"Entity A: Windows.*Entity B: Microsoft Windows", "YES")
register_mock("reasoning", r"Entity A: Rust.*Entity B: rust-lang", "YES")
register_mock("reasoning", r"Entity A: React\b.*Entity B: React Native", "NO")
register_mock("reasoning", r"Entity A: GPT-3.*Entity B: GPT-4", "NO")
register_mock("reasoning", r"Entity A: Docker.*Entity B: Kubernetes", "NO")
register_mock("reasoning", r"Entity A: macOS.*Entity B: iOS", "NO")

# ---------------------------------------------------------------------------
# Role 8: Tag discovery mock responses
# ---------------------------------------------------------------------------

register_mock("reasoning", r"Home Assistant.*Zigbee",
              "home-assistant: \\bhome.?assistant\\b\nsmart-home: \\bsmart.?home\\b|\\bzigbee\\b")
register_mock("reasoning", r"quadcopter.*Betaflight",
              "drone: \\bdrone\\b|\\bquadcopter\\b\nfpv: \\bfpv\\b|\\bfirst.person.view\\b")
register_mock("reasoning", r"Astro.*Tailwind",
              "astro: \\bastro\\b\nstatic-site: \\bstatic.site\\b")
register_mock("reasoning", r"Baldur.*Gate.*BG3",
              "baldurs-gate: \\bbaldur.?s?.?gate\\b|\\bbg3\\b")
register_mock("reasoning", r"LoRA.*Stable Diffusion",
              "stable-diffusion: \\bstable.?diffusion\\b|\\bsd.?xl\\b\ncomfyui: \\bcomfyui\\b")
# Expect NONE
register_mock("reasoning", r"Python script.*parse CSV.*python.*flask",
              "NONE")
register_mock("reasoning", r"what time.*good morning.*joke",
              "NONE")
register_mock("reasoning", r"Docker containers.*Docker Compose.*Dockerfile",
              "NONE")

# ---------------------------------------------------------------------------
# Role 9: Batch tag assignment mock responses
# (just returns tag names, one per line)
# ---------------------------------------------------------------------------

# Batch tag assignment — matches the prompt format from batch_assign_tags()
register_mock("reasoning", r"Available tags:.*Memories to tag:",
              "1: python, machine-learning\n"
              "2: home-automation, electronics\n"
              "3: gaming")

# ---------------------------------------------------------------------------
# Role 10: Session reflection mock responses
# ---------------------------------------------------------------------------

# Substantive session reflection
register_mock("reasoning", r"Review this session.*Session summary:.*(asyncio|memory worker|deployed)",
              "EFFECTIVENESS: effective\n\n"
              "WHAT_WORKED:\n"
              "- Breaking the problem into smaller steps (debugging iteratively)\n"
              "- Using concrete examples to illustrate the fix\n\n"
              "WHAT_DIDNT_WORK:\n"
              "- Initial approach of reading the full file was too slow\n\n"
              "TECHNICAL_INSIGHTS:\n"
              "- asyncio.gather() with return_exceptions=True isolates failures\n\n"
              "PROCESS_INSIGHTS:\n"
              "- Test schema changes on a copy before production")
# Substantive session reflection — SQLite migration
register_mock("reasoning", r"Review this session.*Session summary:.*(SQLite|ALTER TABLE|migration)",
              "EFFECTIVENESS: effective\n\n"
              "WHAT_WORKED:\n"
              "- Incremental ALTER TABLE preserved existing data\n"
              "- WHERE clause skipped already-processed rows\n\n"
              "WHAT_DIDNT_WORK:\n"
              "- Full rebuild was initially considered but too slow (90s)\n\n"
              "TECHNICAL_INSIGHTS:\n"
              "- VACUUM after schema changes reclaims space efficiently\n\n"
              "PROCESS_INSIGHTS:\n"
              "- Always benchmark migration time on production-size data")
# Substantive session reflection — Nginx proxy
register_mock("reasoning", r"Review this session.*Session summary:.*(Nginx|reverse proxy|SSL termination)",
              "EFFECTIVENESS: effective\n\n"
              "WHAT_WORKED:\n"
              "- Systematic debugging of proxy_pass configuration\n"
              "- Checking Host header resolved the 502 error\n\n"
              "WHAT_DIDNT_WORK:\n"
              "- Initially missed the proxy_set_header requirement\n\n"
              "TECHNICAL_INSIGHTS:\n"
              "- Nginx requires explicit Host header for upstream proxying\n\n"
              "PROCESS_INSIGHTS:\n"
              "- Check error logs first when debugging reverse proxy issues")
# Trivial session → SKIP
register_mock("reasoning", r"Review this session.*nothing, bye", "SKIP")
# Trivial session — factual question
register_mock("reasoning", r"Review this session.*Redis", "SKIP")

# ---------------------------------------------------------------------------
# Role 11: Session titling mock responses
# ---------------------------------------------------------------------------

register_mock("summarization", r"Generate.*concise title.*circular import",
              "Debugging Flask circular import")
register_mock("summarization", r"Generate.*concise title.*PostgreSQL",
              "PostgreSQL database setup")
register_mock("summarization", r"Generate.*concise title.*Nginx",
              "Nginx reverse proxy configuration")
register_mock("summarization", r"Generate.*concise title.*Git branch",
              "Git branching strategy discussion")
register_mock("summarization", r"Generate.*concise title.*Minecraft.*lag",
              "Minecraft server performance tuning")
register_mock("summarization", r"Generate.*concise title.*Docker.*ECS",
              "Docker deployment to AWS ECS")
register_mock("summarization", r"Generate.*concise title.*robot.*ESP32",
              "ESP32 desk robot wiring")
register_mock("summarization", r"Generate.*concise title.*(interview|LeetCode)",
              "Algorithm interview preparation")
register_mock("summarization", r"Generate.*concise title.*React.*hooks",
              "React class to hooks migration")
register_mock("summarization", r"Generate.*concise title.*(hiking|Rainier)",
              "Weekend hiking at Mt. Rainier")

# ---------------------------------------------------------------------------
# Role 12: Project digest mock responses
# ---------------------------------------------------------------------------

register_mock("reasoning", r"Project: weather-dashboard",
              "**Overview**\nweather-dashboard is a React/FastAPI web app that "
              "displays current weather, forecasts, and alerts using the OpenWeatherMap API.\n\n"
              "**Current Status**\nThe app is deployed on Vercel (frontend) and Railway (backend). "
              "Core features work: city search, 5-day forecast, hourly breakdown, user preferences, "
              "and weather alerts with push notifications.\n\n"
              "**Key Decisions**\n- SQLite for caching weather data (avoids API rate limits)\n"
              "- No user accounts (keeping project simple)\n"
              "- Browser Notification API for alerts\n"
              "- GeoDB API proxied through backend to avoid CORS\n\n"
              "**Recent Activity**\nAdded weather alerts via OneCall API with browser push notifications. "
              "Fixed stale hourly cache data with 30-minute TTL.\n\n"
              "**Open Issues**\n- Notification permission prompt fires on page load instead of after "
              "user interaction\n- No error handling when OpenWeatherMap API is down")

register_mock("reasoning", r"Project: inventory-tracker",
              "**Overview**\ninventory-tracker is a Django/HTMX server-rendered app for "
              "workshop inventory management with barcode scanning.\n\n"
              "**Current Status**\nCRUD views, barcode scanning, low-stock alerts, "
              "and transaction logging are implemented. Email alerts via SendGrid.\n\n"
              "**Key Decisions**\n- Django + HTMX over SPA (simpler for this use case)\n"
              "- PostgreSQL for data storage\n- Django admin for bulk import\n\n"
              "**Recent Activity**\nImplemented low-stock alerts and transaction audit log.\n\n"
              "**Open Issues**\n- Barcode scanner fails on blurry images (needs error handling)")

register_mock("reasoning", r"Project: cli-task-manager",
              "**Overview**\ncli-task-manager is a terminal-based Kanban task manager "
              "written in Rust using ratatui and SQLite.\n\n"
              "**Current Status**\nCore features complete: CRUD, Kanban columns, priorities, "
              "due dates, filtering, search, and export to JSON/Markdown.\n\n"
              "**Key Decisions**\n- Rust + ratatui for TUI (cross-platform)\n"
              "- SQLite via rusqlite for persistence\n"
              "- Custom List layout (Table widget didn't support variable row heights)\n\n"
              "**Recent Activity**\nAdded task filtering by tag/priority/due date and export.\n\n"
              "**Open Issues**\n- No undo for delete\n- Search doesn't highlight matches in UI")

# ---------------------------------------------------------------------------
# Role 13: Self-reflection mock responses
# ---------------------------------------------------------------------------

register_mock(None, r"list.*tuple.*mutable.*immutable", "NO_CHANGES")
register_mock(None, r"slice notation.*reversed.*my_string", "NO_CHANGES")
register_mock(None, r"try/except.*asyncio\.gather.*return_exceptions", "NO_CHANGES")
register_mock(None, r"B-tree.*CREATE INDEX.*O\(log n\)", "NO_CHANGES")
# Flawed responses → improved versions
register_mock(None, r"HTTPS uses port 80",
              "HTTPS uses port 443 by default. Port 80 is used by HTTP (unencrypted).")
register_mock(None, r"f = open\('data\.txt'",
              "Use a context manager to ensure the file is properly closed:\n"
              "```python\nwith open('data.txt', 'r') as f:\n    content = f.read()\n    print(content)\n```")
register_mock(None, r"dictionary is O\(n\)",
              "Searching a Python dictionary is O(1) on average because it uses a hash table. "
              "Only in the rare worst case of hash collisions does it degrade to O(n).")
register_mock(None, r"git commit`\n.*default editor",
              "To make a git commit:\n"
              "1. Stage your changes: `git add <file>` (or `git add .` for everything — "
              "but be careful, this stages all files including untracked ones)\n"
              "2. Commit with an inline message: `git commit -m \"Your message\"`\n"
              "Or just `git commit` to open your editor for a longer message.")

# ---------------------------------------------------------------------------
# Role 14: Plan generation mock responses
# ---------------------------------------------------------------------------

register_mock(None, r"Read the contents of config\.yaml",
              "1. Read config.yaml to examine model configuration (read_file)")
register_mock(None, r"Add a hello\(\) function to utils\.py",
              "1. Read utils.py to see current contents (read_file)\n"
              "2. Add hello() function to utils.py (edit_file)")
register_mock(None, r"Find all files that import.*requests",
              "1. Search for 'import requests' across the project (grep_files)")
register_mock(None, r"Rename the function process_data",
              "1. Find all references to process_data() (grep_files)\n"
              "2. Edit processor.py to rename the function definition (edit_file)\n"
              "3. Update each call site found in step 1 (edit_file)")
register_mock(None, r"new CLI command called.*stats",
              "1. Read cli.py to understand existing command structure (read_file)\n"
              "2. Read sqlite_store.py to find count/average query methods (read_file)\n"
              "3. Add stats command with database queries and table output (edit_file)\n"
              "4. Test the command with a sample database (run_command)")
register_mock(None, r"Refactor the error handling in router\.py",
              "1. Read router.py to catalog existing exception catches (read_file)\n"
              "2. Search for Exception catches across the codebase (grep_files)\n"
              "3. Create exceptions.py with custom exception classes (write_file)\n"
              "4. Refactor router.py to use the new exception classes (edit_file)")

# ---------------------------------------------------------------------------
# Role 15/16: Tool calling and coding — not mock-able in single-prompt mode
# These are multi-turn tests that need the full agent harness.
# Mock mode skips them and returns 0.0.
# ---------------------------------------------------------------------------
