"""
Curated test data for the unified benchmark suite.

57 new test cases across 7 roles that lacked coverage.
Each role's data matches the exact input format expected by its production prompt.

Roles covered:
  - Deduplication (10 cases)
  - Entity Resolution (12 cases)
  - Tag Discovery (8 cases)
  - Session Titling (10 cases)
  - Project Digest (3 cases)
  - Self-Reflection (8 cases)
  - Plan Generation (6 cases)
"""

# ---------------------------------------------------------------------------
# Role 3: Deduplication — decide_memory_action()
#
# Input:  new_memory (str), existing_memories (list[str])
# Output: ADD | NONE | UPDATE N | DELETE N
#
# 10 cases: 3 ADD (unrelated), 2 NONE (identical/redundant),
#           3 UPDATE (refines existing), 2 DELETE (contradicts existing)
# ---------------------------------------------------------------------------

DEDUP_CASES = [
    # --- ADD: new memory contains unique information ---
    {
        "id": "dedup_add_1",
        "new_memory": "User set up a Raspberry Pi 4 as a home media server running Jellyfin.",
        "existing_memories": [
            "User has a desktop PC with an RTX 3070 for gaming.",
        ],
        "expected_action": "ADD",
        "reason": "Completely different topic (media server vs gaming PC).",
    },
    {
        "id": "dedup_add_2",
        "new_memory": "User prefers dark roast coffee, especially Ethiopian single-origin beans.",
        "existing_memories": [
            "User drinks coffee every morning before work.",
            "User has a Breville espresso machine.",
        ],
        "expected_action": "ADD",
        "reason": "Adds new info (bean preference) not in existing memories.",
    },
    {
        "id": "dedup_add_3",
        "new_memory": "User started learning Rust for a side project building a CLI tool.",
        "existing_memories": [
            "User primarily codes in Python for work projects.",
        ],
        "expected_action": "ADD",
        "reason": "New language/project, not a duplicate of Python usage.",
    },

    # --- NONE: redundant, says the same thing ---
    {
        "id": "dedup_none_1",
        "new_memory": "User uses Python as their main programming language at work.",
        "existing_memories": [
            "User primarily codes in Python for work projects.",
        ],
        "expected_action": "NONE",
        "reason": "Semantically identical to existing memory.",
    },
    {
        "id": "dedup_none_2",
        "new_memory": "User has a cat named Luna.",
        "existing_memories": [
            "User's cat is called Luna.",
            "User adopted Luna from a shelter two years ago.",
        ],
        "expected_action": "NONE",
        "reason": "First existing memory already captures this fact.",
    },

    # --- UPDATE: new memory refines or adds detail to existing ---
    {
        "id": "dedup_update_1",
        "new_memory": "User's desktop has an RTX 3070, 32GB RAM, and a Ryzen 7 5800X CPU.",
        "existing_memories": [
            "User has a desktop PC with an RTX 3070 for gaming.",
        ],
        "expected_action": "UPDATE",
        "expected_target": 1,
        "reason": "Adds RAM and CPU details to existing hardware memory.",
    },
    {
        "id": "dedup_update_2",
        "new_memory": "User switched from VS Code to Neovim as their primary editor in January 2026.",
        "existing_memories": [
            "User uses VS Code as their primary code editor.",
            "User has tried Vim but found the learning curve steep.",
        ],
        "expected_action": "UPDATE",
        "expected_target": 1,
        "reason": "Updates the editor preference — VS Code is now outdated info.",
    },
    {
        "id": "dedup_update_3",
        "new_memory": "User completed the Rust CLI tool project and published it on crates.io.",
        "existing_memories": [
            "User started learning Rust for a side project building a CLI tool.",
        ],
        "expected_action": "UPDATE",
        "expected_target": 1,
        "reason": "Progresses the status of an existing project memory.",
    },

    # --- DELETE: existing memory is now factually wrong/stale ---
    {
        "id": "dedup_delete_1",
        "new_memory": "User sold their RTX 3070 desktop and now uses a MacBook Pro M3 exclusively.",
        "existing_memories": [
            "User has a desktop PC with an RTX 3070 for gaming.",
            "User prefers Windows for development.",
        ],
        "expected_action": "DELETE",
        "expected_target": 1,
        "reason": "Desktop no longer exists — existing memory is stale.",
    },
    {
        "id": "dedup_delete_2",
        "new_memory": "User quit drinking coffee entirely for health reasons.",
        "existing_memories": [
            "User drinks coffee every morning before work.",
            "User has a Breville espresso machine.",
        ],
        "expected_action": "DELETE",
        "expected_target": 1,
        "reason": "Coffee habit memory is now factually wrong.",
    },
]

# ---------------------------------------------------------------------------
# Role 7: Entity Resolution — resolve_entity_duplicate()
#
# Input:  new_entity (str), existing_entity (str)
# Output: YES (same entity) | NO (different entities)
#
# 12 cases: 4 same-entity, 4 different-entity, 4 tricky/ambiguous
# ---------------------------------------------------------------------------

ENTITY_RESOLUTION_CASES = [
    # --- Same entity (expect YES) ---
    {
        "id": "eres_same_1",
        "entity_a": "PostgreSQL",
        "entity_b": "postgres",
        "expected": "YES",
        "reason": "Common abbreviation of the same database.",
    },
    {
        "id": "eres_same_2",
        "entity_a": "JavaScript",
        "entity_b": "JS",
        "expected": "YES",
        "reason": "Standard acronym for the same language.",
    },
    {
        "id": "eres_same_3",
        "entity_a": "Jim Buschman",
        "entity_b": "Jim B.",
        "expected": "YES",
        "reason": "Name abbreviation for the same person.",
    },
    {
        "id": "eres_same_4",
        "entity_a": "VS Code",
        "entity_b": "Visual Studio Code",
        "expected": "YES",
        "reason": "Shortened name for the same editor.",
    },

    # --- Different entities (expect NO) ---
    {
        "id": "eres_diff_1",
        "entity_a": "React",
        "entity_b": "React Native",
        "expected": "NO",
        "reason": "Related but distinct frameworks with different APIs.",
    },
    {
        "id": "eres_diff_2",
        "entity_a": "GPT-3",
        "entity_b": "GPT-4",
        "expected": "NO",
        "reason": "Different model versions with different capabilities.",
    },
    {
        "id": "eres_diff_3",
        "entity_a": "Docker",
        "entity_b": "Kubernetes",
        "expected": "NO",
        "reason": "Related container tools but fundamentally different.",
    },
    {
        "id": "eres_diff_4",
        "entity_a": "macOS",
        "entity_b": "iOS",
        "expected": "NO",
        "reason": "Different operating systems despite same vendor.",
    },

    # --- Tricky/ambiguous (expect NO — different meanings) ---
    {
        "id": "eres_tricky_1",
        "entity_a": "Python",
        "entity_b": "python",
        "expected": "YES",
        "reason": "Same word, likely same entity (programming language) in a tech context.",
    },
    {
        "id": "eres_tricky_2",
        "entity_a": "Mercury",
        "entity_b": "mercury",
        "expected": "YES",
        "reason": "Without disambiguating context, case variants of the same name should merge.",
    },
    {
        "id": "eres_tricky_3",
        "entity_a": "Windows",
        "entity_b": "Microsoft Windows",
        "expected": "YES",
        "reason": "Expanded name for the same OS.",
    },
    {
        "id": "eres_tricky_4",
        "entity_a": "Rust",
        "entity_b": "rust-lang",
        "expected": "YES",
        "reason": "The language's community alias (rust-lang.org).",
    },
]

# ---------------------------------------------------------------------------
# Role 8: Tag Discovery — discover_tag_patterns()
#
# Input:  summaries (list[str]), existing_tags (list[str])
# Output: NONE or tag_name: regex_pattern lines
#
# 8 cases: 5 expect new patterns, 3 expect NONE
# ---------------------------------------------------------------------------

TAG_DISCOVERY_CASES = [
    # --- Expect new patterns (recurring topic not in existing tags) ---
    {
        "id": "tagd_new_1",
        "summaries": [
            "User set up Home Assistant on a Raspberry Pi for smart home automation.",
            "User configured Zigbee sensors in Home Assistant for temperature monitoring.",
            "User created an automation in Home Assistant to turn off lights at midnight.",
            "User installed HACS (Home Assistant Community Store) for custom integrations.",
            "User asked about Home Assistant energy monitoring dashboard setup.",
        ],
        "existing_tags": ["raspberry-pi", "linux", "python", "docker"],
        "expect_patterns": True,
        "expected_tag_keywords": ["home-assistant", "smart-home", "automation"],
        "reason": "Strong recurring topic (home-assistant) not covered by existing tags.",
    },
    {
        "id": "tagd_new_2",
        "summaries": [
            "User is building a quadcopter drone with an F405 flight controller.",
            "User configured Betaflight firmware for PID tuning on the drone.",
            "User ordered FPV goggles for first-person-view drone flying.",
            "User asked about drone battery LiPo cell balancing and storage voltage.",
            "User crashed the drone and needs to replace a motor arm.",
        ],
        "existing_tags": ["electronics", "3d-printing", "soldering"],
        "expect_patterns": True,
        "expected_tag_keywords": ["drone", "fpv", "quadcopter", "betaflight"],
        "reason": "Clear recurring drone/FPV topic not in existing tags.",
    },
    {
        "id": "tagd_new_3",
        "summaries": [
            "User migrated their blog from WordPress to Astro static site generator.",
            "User set up Astro with Tailwind CSS for the new blog design.",
            "User deployed the Astro site to Cloudflare Pages with automatic builds.",
            "User added MDX support to Astro for interactive blog components.",
            "User optimized Astro image handling with the built-in image component.",
        ],
        "existing_tags": ["javascript", "css", "web-dev", "html"],
        "expect_patterns": True,
        "expected_tag_keywords": ["astro", "static-site"],
        "reason": "Astro is a specific framework not covered by generic web-dev tag.",
    },
    {
        "id": "tagd_new_4",
        "summaries": [
            "User played through Baldur's Gate 3 Act 2 and discussed party composition.",
            "User asked about BG3 multiclassing options for a Paladin/Warlock build.",
            "User completed the Gauntlet of Shar in Baldur's Gate 3.",
            "User debated whether to side with the Emperor or Orpheus in BG3.",
            "User started a second BG3 playthrough as a dark urge character.",
        ],
        "existing_tags": ["gaming", "minecraft", "steam"],
        "expect_patterns": True,
        "expected_tag_keywords": ["baldurs-gate", "bg3"],
        "reason": "Specific game not covered by generic 'gaming' tag.",
    },
    {
        "id": "tagd_new_5",
        "summaries": [
            "User trained a LoRA model on Stable Diffusion for generating pixel art.",
            "User compared Stable Diffusion XL vs SD 1.5 for image quality at low steps.",
            "User set up ComfyUI as a node-based Stable Diffusion workflow tool.",
            "User generated character portraits using Stable Diffusion with ControlNet.",
            "User asked about VRAM requirements for running SDXL locally.",
        ],
        "existing_tags": ["machine-learning", "python", "gpu"],
        "expect_patterns": True,
        "expected_tag_keywords": ["stable-diffusion", "image-gen", "comfyui"],
        "reason": "Specific AI image generation tooling not covered by generic ML tag.",
    },

    # --- Expect NONE (content already well-tagged or no recurring pattern) ---
    {
        "id": "tagd_none_1",
        "summaries": [
            "User wrote a Python script to parse CSV files.",
            "User debugged a Python import error in their Flask app.",
            "User asked about Python virtual environments and pip.",
            "User refactored a Python class to use dataclasses.",
            "User profiled a Python function for performance bottlenecks.",
        ],
        "existing_tags": ["python", "flask", "data-processing", "debugging", "performance"],
        "expect_patterns": False,
        "reason": "All topics already covered by existing tags.",
    },
    {
        "id": "tagd_none_2",
        "summaries": [
            "User asked what time it is.",
            "User said good morning.",
            "User thanked the assistant for help yesterday.",
            "User asked the assistant to tell a joke.",
            "User said they were heading to lunch.",
        ],
        "existing_tags": ["python", "javascript", "docker"],
        "expect_patterns": False,
        "reason": "Filler/social content — no substantive recurring topic.",
    },
    {
        "id": "tagd_none_3",
        "summaries": [
            "User asked about setting up Docker containers for development.",
            "User configured Docker Compose for a multi-service app.",
            "User asked about Docker volume mounts for persistent data.",
            "User debugged a Docker networking issue between containers.",
            "User optimized a Dockerfile to reduce image size.",
        ],
        "existing_tags": ["docker", "containers", "devops", "linux", "networking"],
        "expect_patterns": False,
        "reason": "Docker topic already thoroughly covered by existing tags.",
    },
]

# ---------------------------------------------------------------------------
# Role 11: Session Titling — generate_session_title()
#
# Input:  session summary text (str)
# Output: concise title (≤10 words)
#
# 10 curated session summaries with expected keywords for relevance check.
# Since we can't access the DB from dev machine, these are synthetic but
# realistic based on BlipShell's actual usage patterns.
#
# SCORING NOTE: The scorer uses `expected_keywords` (≥1 keyword match) for
# the relevance metric, NOT string comparison against `ideal_title`.
# `ideal_title` is documentation-only — a human reference for what a good
# title looks like. This avoids brittle exact-match scoring.
# ---------------------------------------------------------------------------

SESSION_TITLING_CASES = [
    {
        "id": "title_1",
        "session_summary": (
            "User asked for help debugging a Python ImportError in their Flask "
            "application. The issue was a circular import between models.py and "
            "routes.py. Resolved by moving the import inside the function that "
            "needed it. User also asked about best practices for avoiding circular "
            "imports in Flask projects."
        ),
        "expected_keywords": ["import", "flask", "circular", "python", "debug"],
        "ideal_title": "Debugging Flask circular import",
    },
    {
        "id": "title_2",
        "session_summary": (
            "User set up a new PostgreSQL database for their project. Configured "
            "connection pooling with asyncpg. Created tables for users, sessions, "
            "and permissions. Added indexes on frequently queried columns. Discussed "
            "whether to use SQLAlchemy ORM or raw SQL queries."
        ),
        "expected_keywords": ["postgresql", "database", "setup", "asyncpg", "sql"],
        "ideal_title": "PostgreSQL database setup and configuration",
    },
    {
        "id": "title_3",
        "session_summary": (
            "Helped user configure Nginx as a reverse proxy for their Node.js "
            "application. Set up SSL certificates with Let's Encrypt. Added rate "
            "limiting and CORS headers. Troubleshot a 502 Bad Gateway error caused "
            "by the upstream app not listening on the correct port."
        ),
        "expected_keywords": ["nginx", "reverse", "proxy", "ssl", "node"],
        "ideal_title": "Nginx reverse proxy setup with SSL",
    },
    {
        "id": "title_4",
        "session_summary": (
            "User wanted to learn about Git branching strategies. Discussed "
            "trunk-based development vs GitFlow. User decided to adopt a simplified "
            "GitFlow with main, develop, and feature branches. Set up branch "
            "protection rules on GitHub."
        ),
        "expected_keywords": ["git", "branch", "gitflow", "strategy"],
        "ideal_title": "Git branching strategy discussion",
    },
    {
        "id": "title_5",
        "session_summary": (
            "User reported that their Minecraft server was lagging with 10+ players. "
            "Analyzed server logs and found the issue was chunk loading. Adjusted "
            "view distance from 16 to 10 chunks, allocated more RAM to the JVM, "
            "and installed the Lithium performance mod. Server TPS improved from "
            "12 to 19.8."
        ),
        "expected_keywords": ["minecraft", "server", "lag", "performance", "tps"],
        "ideal_title": "Minecraft server performance optimization",
    },
    {
        "id": "title_6",
        "session_summary": (
            "User asked about deploying a Docker containerized app to AWS ECS. "
            "Created a Dockerfile, built the image, pushed to ECR. Set up an ECS "
            "task definition and service. Configured an ALB for load balancing. "
            "Discussed Fargate vs EC2 launch types."
        ),
        "expected_keywords": ["docker", "aws", "ecs", "deploy", "container"],
        "ideal_title": "Deploying Docker containers to AWS ECS",
    },
    {
        "id": "title_7",
        "session_summary": (
            "Long conversation about the user's desk robot project. The robot uses "
            "an ESP32 microcontroller with servo motors. User needed help with the "
            "I2C communication protocol for an SSD1306 OLED display. Also discussed "
            "powering the servos without brownouts using a separate 5V supply."
        ),
        "expected_keywords": ["robot", "esp32", "servo", "i2c", "oled"],
        "ideal_title": "ESP32 desk robot I2C and servo wiring",
    },
    {
        "id": "title_8",
        "session_summary": (
            "User was preparing for a job interview at a FAANG company. Practiced "
            "solving LeetCode-style problems: binary search variations, two-pointer "
            "technique, and BFS/DFS graph traversal. Reviewed time complexity "
            "analysis for each solution."
        ),
        "expected_keywords": ["interview", "leetcode", "algorithm", "coding"],
        "ideal_title": "FAANG interview algorithm practice",
    },
    {
        "id": "title_9",
        "session_summary": (
            "User migrated a React class component to a functional component using "
            "hooks. Replaced componentDidMount with useEffect, this.state with "
            "useState, and extracted custom hooks for shared logic. Also added "
            "TypeScript types to the converted component."
        ),
        "expected_keywords": ["react", "hooks", "migration", "typescript", "component"],
        "ideal_title": "React class to hooks migration",
    },
    {
        "id": "title_10",
        "session_summary": (
            "Casual conversation about the user's weekend plans. User mentioned "
            "they were going hiking at Mt. Rainier. Brief discussion about weather "
            "conditions and trail recommendations. User also mentioned their dog "
            "would be joining them on the hike."
        ),
        "expected_keywords": ["hiking", "weekend", "rainier", "outdoor"],
        "ideal_title": "Weekend hiking plans at Mt. Rainier",
    },
]

# ---------------------------------------------------------------------------
# Role 12: Project Digest — generate_initial_digest()
#
# Input:  project_name (str), session_summaries (str)
# Output: 300-500 word digest with 5 sections:
#         Overview, Current Status, Key Decisions, Recent Activity, Open Issues
#
# 3 synthetic projects with realistic session histories.
# Scorer checks: section presence, word count range, project name mention.
#
# CAVEAT: All 3 cases are synthetic (DB unavailable on dev machine). Synthetic
# projects have predictable structure that models can pattern-match easily.
# Scores >0.95 on this role should be treated skeptically without real-DB
# validation. When running on Ollama PC, consider adding real projects from
# the database for a harder test.
# ---------------------------------------------------------------------------

PROJECT_DIGEST_CASES = [
    {
        "id": "digest_1",
        "project_name": "weather-dashboard",
        "session_summaries": (
            "Session 1: Bootstrapped the weather-dashboard project. Created a React "
            "frontend with Vite. Set up a FastAPI backend to proxy OpenWeatherMap "
            "API calls. Decided to use SQLite for caching weather data to avoid "
            "hitting API rate limits.\n\n"
            "Session 2: Built the main dashboard component showing current weather, "
            "5-day forecast, and hourly breakdown. Added a city search feature with "
            "autocomplete using the GeoDB API. Hit a CORS issue with the geocoding "
            "API and worked around it by proxying through the backend.\n\n"
            "Session 3: Added user preferences — favorite cities, temperature unit "
            "(Celsius/Fahrenheit), and dark mode toggle. Stored preferences in "
            "localStorage. Discussed whether to add user accounts but decided "
            "against it to keep the project simple.\n\n"
            "Session 4: Deployed to Vercel (frontend) and Railway (backend). Set up "
            "environment variables for the API key. Found a bug where the hourly "
            "forecast showed stale cached data. Fixed by adding cache TTL of 30 "
            "minutes. Still need to add error handling for when the API is down.\n\n"
            "Session 5: Added weather alerts using the OneCall API endpoint. "
            "Implemented push notifications via the browser Notification API. "
            "Backend sends alerts when severe weather is detected. Open issue: "
            "notification permission prompt timing is poor — shows on page load "
            "instead of after user interaction."
        ),
        "expected_sections": ["Overview", "Current Status", "Key Decisions",
                              "Recent Activity", "Open Issues"],
        "expected_details": ["weather-dashboard", "React", "FastAPI",
                             "OpenWeatherMap", "Vercel"],
    },
    {
        "id": "digest_2",
        "project_name": "inventory-tracker",
        "session_summaries": (
            "Session 1: Started the inventory-tracker project for a small workshop. "
            "Chose Django with HTMX for a server-rendered approach — no SPA needed "
            "for this use case. Set up PostgreSQL database with tables for items, "
            "categories, locations, and transactions.\n\n"
            "Session 2: Built CRUD views for inventory items. Added barcode scanning "
            "using the phone camera via the QuaggaJS library. Each item has: name, "
            "SKU, quantity, location, minimum stock level. Decided to use Django's "
            "built-in admin for bulk data entry rather than building a custom import.\n\n"
            "Session 3: Implemented low-stock alerts. When quantity drops below "
            "minimum, an alert appears on the dashboard and optionally sends an "
            "email via SendGrid. Added a transaction log that records every "
            "add/remove/transfer with timestamp and user. Bug found: barcode "
            "scanner fails on blurry images — needs error handling."
        ),
        "expected_sections": ["Overview", "Current Status", "Key Decisions",
                              "Recent Activity", "Open Issues"],
        "expected_details": ["inventory-tracker", "Django", "HTMX",
                             "PostgreSQL", "barcode"],
    },
    {
        "id": "digest_3",
        "project_name": "cli-task-manager",
        "session_summaries": (
            "Session 1: Created cli-task-manager, a terminal-based task manager "
            "written in Rust using the ratatui TUI framework. Chose SQLite via "
            "rusqlite for persistence. Implemented basic task CRUD: add, list, "
            "complete, delete. Decided on a Kanban-style column layout (Todo, "
            "In Progress, Done).\n\n"
            "Session 2: Added keyboard navigation between columns and tasks. "
            "Implemented task priorities (high/medium/low) with color coding. "
            "Added due dates with overdue highlighting. Found that ratatui's "
            "Table widget doesn't support variable row heights — switched to a "
            "custom List-based layout.\n\n"
            "Session 3: Implemented task filtering and search. Added tags and "
            "filtering by tag, priority, or due date. Exported tasks to JSON "
            "and Markdown formats. Open issues: no undo for delete, and the "
            "search doesn't highlight matches in the UI yet."
        ),
        "expected_sections": ["Overview", "Current Status", "Key Decisions",
                              "Recent Activity", "Open Issues"],
        "expected_details": ["cli-task-manager", "Rust", "ratatui",
                             "SQLite", "Kanban"],
    },
]

# ---------------------------------------------------------------------------
# Role 13: Self-Reflection — reflect_on_response()
#
# Input:  user_message (str), response (str)
# Output: "NO_CHANGES" if response is good, improved text if flawed
#
# 8 cases: 4 good responses (expect NO_CHANGES), 4 flawed (expect improvement)
#
# DESIGN NOTE: The 4 "good" responses must be genuinely well-crafted — correct,
# well-formatted, comprehensive, with examples where appropriate. A correct but
# terse or poorly-structured response would legitimately warrant improvement,
# making NO_CHANGES the wrong expected answer. Each "good" case is designed to
# leave no reasonable grounds for the reflection model to suggest changes.
# ---------------------------------------------------------------------------

SELF_REFLECTION_CASES = [
    # --- Good responses (expect NO_CHANGES) ---
    {
        "id": "refl_good_1",
        "user_message": "What's the difference between a list and a tuple in Python?",
        "response": (
            "Lists are mutable (you can add, remove, or change elements after "
            "creation), while tuples are immutable (once created, they can't be "
            "modified). Lists use square brackets `[1, 2, 3]`, tuples use "
            "parentheses `(1, 2, 3)`. Use tuples for fixed collections like "
            "coordinates or function return values, and lists when you need to "
            "modify the collection."
        ),
        "expected": "NO_CHANGES",
        "reason": "Accurate, concise, covers the key differences.",
    },
    {
        "id": "refl_good_2",
        "user_message": "How do I reverse a string in Python?",
        "response": (
            "The most common way is slice notation: `my_string[::-1]`. For example, "
            "`'hello'[::-1]` returns `'olleh'`. You can also use "
            "`''.join(reversed(my_string))`, which is more explicit but slightly "
            "slower. For most cases, the slice approach is preferred for its "
            "brevity and speed."
        ),
        "expected": "NO_CHANGES",
        "reason": "Covers both approaches, includes example, explains tradeoff.",
    },
    {
        "id": "refl_good_3",
        "user_message": "What's a good way to handle errors in async Python code?",
        "response": (
            "Use try/except blocks around your await calls, just like synchronous "
            "code. For gathering multiple tasks, use `asyncio.gather(..., "
            "return_exceptions=True)` to prevent one failure from canceling "
            "everything. You can also create custom exception handlers with "
            "`loop.set_exception_handler()` for unhandled exceptions in fire-and-"
            "forget tasks."
        ),
        "expected": "NO_CHANGES",
        "reason": "Practical, correct, covers common patterns.",
    },
    {
        "id": "refl_good_4",
        "user_message": "Explain what a database index is.",
        "response": (
            "A database index is a data structure (usually a B-tree) that speeds "
            "up row lookups at the cost of additional storage and slower writes. "
            "When you query a column that has an index, the database can find "
            "matching rows without scanning the entire table. For example, "
            "`CREATE INDEX idx_users_email ON users(email)` lets the database "
            "find a user by email in O(log n) instead of O(n). Create indexes on "
            "columns you frequently filter or join on, but avoid over-indexing "
            "since each index adds overhead to INSERT/UPDATE/DELETE operations."
        ),
        "expected": "NO_CHANGES",
        "reason": "Accurate, concrete SQL example, explains tradeoffs with complexity.",
    },

    # --- Flawed responses (expect improvement) ---
    {
        "id": "refl_flawed_1",
        "user_message": "What port does HTTPS use?",
        "response": "HTTPS uses port 80 by default.",
        "expected": "IMPROVED",
        "flaw": "Wrong port — HTTPS uses 443, not 80. Port 80 is HTTP.",
    },
    {
        "id": "refl_flawed_2",
        "user_message": "How do I read a file in Python?",
        "response": (
            "Use `open()` to read a file:\n"
            "```python\n"
            "f = open('data.txt', 'r')\n"
            "content = f.read()\n"
            "print(content)\n"
            "```"
        ),
        "expected": "IMPROVED",
        "flaw": "Missing f.close() or context manager (with statement). Resource leak.",
    },
    {
        "id": "refl_flawed_3",
        "user_message": "What's the time complexity of searching a Python dictionary?",
        "response": (
            "Searching a Python dictionary is O(n) because it needs to check "
            "each key-value pair sequentially until it finds a match."
        ),
        "expected": "IMPROVED",
        "flaw": "Wrong complexity — dict lookup is O(1) average case (hash table), not O(n).",
    },
    {
        "id": "refl_flawed_4",
        "user_message": "How do I make a git commit?",
        "response": (
            "To make a git commit:\n"
            "1. Stage your changes: `git add .`\n"
            "2. Commit: `git commit`\n"
            "This will open your default editor to write a commit message."
        ),
        "expected": "IMPROVED",
        "flaw": (
            "Response is mostly correct but doesn't mention `git commit -m` for "
            "inline messages, which is the more common workflow. Also doesn't warn "
            "that `git add .` stages everything including untracked files."
        ),
    },
]

# ---------------------------------------------------------------------------
# Role 14: Plan Generation — generate_plan()
#
# Input:  user_request (str), conversation_context (str, optional)
# Output: numbered steps with optional (tool_hint)
#
# 6 curated tasks from simple (1-2 steps) to complex (4-5 steps).
# Scorer checks: parse success, step count range, tool hints, relevance.
# ---------------------------------------------------------------------------

PLAN_GENERATION_CASES = [
    {
        "id": "plan_simple_1",
        "user_request": "Read the contents of config.yaml and tell me what models are configured.",
        "context": "",
        "expected_step_range": (1, 2),
        "expected_tool_hints": ["read_file"],
        "expected_keywords": ["config", "read"],
        "reason": "Single-file read task, should be 1 step.",
    },
    {
        "id": "plan_simple_2",
        "user_request": "Add a hello() function to utils.py that prints 'Hello, world!'.",
        "context": "",
        "expected_step_range": (1, 3),
        "expected_tool_hints": ["read_file", "edit_file"],
        "expected_keywords": ["utils", "hello", "function", "add"],
        "reason": "Simple edit: read file, then add function.",
    },
    {
        "id": "plan_medium_1",
        "user_request": "Find all files that import the 'requests' library and list them.",
        "context": "",
        "expected_step_range": (1, 2),
        "expected_tool_hints": ["grep_files"],
        "expected_keywords": ["import", "requests", "find", "grep"],
        "reason": "Single grep operation across the project.",
    },
    {
        "id": "plan_medium_2",
        "user_request": (
            "Rename the function process_data() to transform_data() in processor.py "
            "and update all call sites across the project."
        ),
        "context": "",
        "expected_step_range": (2, 4),
        "expected_tool_hints": ["grep_files", "edit_file"],
        "expected_keywords": ["rename", "process_data", "transform_data"],
        "reason": "Find references then edit multiple files.",
    },
    {
        "id": "plan_complex_1",
        "user_request": (
            "Add a new CLI command called 'stats' that queries the database for "
            "memory count, entity count, and average importance score, then "
            "prints them in a formatted table."
        ),
        "context": "The CLI is built with Click in cli.py. Database access is via sqlite_store.py.",
        "expected_step_range": (3, 5),
        "expected_tool_hints": ["read_file", "edit_file"],
        "expected_keywords": ["stats", "cli", "database", "command"],
        "reason": "Multi-step: read existing CLI, read DB layer, implement command.",
    },
    {
        "id": "plan_complex_2",
        "user_request": (
            "Refactor the error handling in router.py to use custom exception "
            "classes instead of generic Exception catches. Create an exceptions "
            "module if one doesn't exist."
        ),
        "context": "",
        "expected_step_range": (3, 5),
        "expected_tool_hints": ["read_file", "write_file", "edit_file", "grep_files"],
        "expected_keywords": ["error", "exception", "router", "refactor"],
        "reason": "Multi-step: explore current handling, create module, refactor.",
    },
]
