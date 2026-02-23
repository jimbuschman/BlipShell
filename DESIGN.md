# BlipShell Design Document

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Memory System](#3-memory-system)
4. [LLM Subsystem](#4-llm-subsystem)
5. [Agent Core](#5-agent-core)
6. [UI Layer](#6-ui-layer)
7. [Scripts & Operations](#7-scripts--operations)
8. [Testing](#8-testing)
9. [Configuration](#9-configuration)
10. [Data Models](#10-data-models)
11. [Code Review Findings](#11-code-review-findings)

---

## 1. Project Overview

BlipShell is a local LLM personal assistant with persistent memory. It uses Ollama
for local inference and supports multi-endpoint routing to cloud APIs (Groq, Gemini)
via OpenAI-compatible SDKs. Every conversation is automatically processed into
searchable, ranked, scored memory with entity graph relationships, enabling the
assistant to recall context across sessions.

### Key Capabilities
- **Persistent memory** with automatic summarization, ranking, and importance scoring
- **Hybrid search** combining semantic vectors (ChromaDB), full-text (FTS5), and entity graph expansion
- **Multi-endpoint LLM routing** with cascading fallback (cloud to local)
- **Native tool calling** with approval gates for dangerous operations
- **Project mode** for code-aware conversations with git integration
- **Background task offloading** to remote workers
- **Import pipelines** for ChatGPT, Claude, and DeepSeek conversation histories

### Two-PC Architecture
- **Development PC**: Runs the BlipShell application, web API, and CLI
- **Ollama PC**: Hosts the local LLM models, performs inference
- Code synced via git; Ollama accessed over the network

### Tech Stack
- **Python 3.11+** with async/await throughout
- **Ollama** for local inference (native async client)
- **OpenAI SDK** for cloud endpoints (Groq, Gemini)
- **SQLite** (WAL mode, aiosqlite) for structured data
- **ChromaDB** for vector embeddings
- **FastAPI** + WebSockets for the web API
- **Rich** + Click for the CLI
- **Pydantic** for configuration and data models

### Entry Points
```
blipshell                    # CLI: new session (pyproject.toml → blipshell.ui.cli:main)
blipshell --continue         # CLI: resume last session
blipshell --session <id>     # CLI: resume specific session
blipshell web                # Launch web API on port 8000
python -m blipshell.worker   # Remote worker daemon
```

---

## 2. Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │            UI Layer                  │
                          │   CLI (Rich/Click)  │  Web (FastAPI) │
                          │   Slash commands     │  REST + WS     │
                          │                     │  /v1 (OpenAI)   │
                          └─────────┬───────────┴────────┬───────┘
                                    │                    │
                          ┌─────────▼────────────────────▼───────┐
                          │           Agent Core                  │
                          │   Chat loop  │  Planner  │  Executor  │
                          │   Tool system │  Background tasks     │
                          │   Project activation │ Workflows      │
                          └──┬──────────┬──────────┬─────────────┘
                             │          │          │
              ┌──────────────▼┐   ┌─────▼────┐   ┌▼──────────────┐
              │  Memory System │   │  LLM     │   │  Session      │
              │  Store/Search  │   │  Router  │   │  Manager      │
              │  Processor     │   │  Clients │   │               │
              │  Entity Graph  │   │  Prompts │   │               │
              └──┬─────────┬──┘   └────┬─────┘   └───────────────┘
                 │         │           │
           ┌─────▼──┐  ┌──▼────┐  ┌───▼─────────────────────────┐
           │ SQLite  │  │Chroma │  │  LLM Endpoints              │
           │  (WAL)  │  │  DB   │  │  Ollama │ Groq │ Gemini     │
           └─────────┘  └───────┘  └─────────────────────────────┘
```

### Three-Tier Design

1. **UI Layer** — CLI slash commands, FastAPI REST/WebSocket, OpenAI-compatible `/v1`
2. **Core Layer** — Agent orchestration, memory processing, LLM routing, tool execution
3. **Storage Layer** — SQLite (structured data, FTS5), ChromaDB (vector embeddings), LLM endpoints

### Package Structure

```
blipshell/
├── core/                    # Agent orchestration
│   ├── agent.py             # Main chat loop (1673 lines)
│   ├── planner.py           # Complexity classification + task planning
│   ├── executor.py          # Step-by-step task execution
│   ├── background.py        # Async task manager
│   ├── config.py            # ConfigManager (YAML loading)
│   ├── repo_map.py          # AST-based code structure mapping
│   ├── router.py            # Internal message routing
│   ├── tool_rules.py        # Letta-style tool calling constraints
│   ├── workflows.py         # YAML workflow templates
│   └── tools/               # Tool implementations (10 files)
├── llm/                     # LLM routing & clients
│   ├── router.py            # Task-type-based model routing
│   ├── endpoints.py         # Multi-endpoint management
│   ├── client.py            # Ollama async client
│   ├── openai_client.py     # OpenAI-compatible client
│   ├── prompts.py           # All prompt templates
│   ├── exceptions.py        # Error classification
│   ├── model_settings.py    # Per-model behavioral config
│   └── job_queue.py         # Priority async queue
├── memory/                  # Memory subsystem
│   ├── manager.py           # Token budget pool manager
│   ├── processor.py         # Processing pipeline
│   ├── search.py            # Hybrid search engine
│   ├── sqlite_store.py      # SQLite storage (1757 lines)
│   ├── chroma_store.py      # ChromaDB wrapper
│   ├── entity_extractor.py  # Entity graph extraction
│   ├── consolidation.py     # Deduplication & merging
│   ├── tagger.py            # Regex-based topic/behavior tags
│   ├── tag_discovery.py     # LLM-discovered tag patterns
│   ├── noise.py             # Noise/filler filtering
│   └── query_profiles.py    # Dynamic pool budget allocation
├── models/                  # Pydantic data models
│   ├── config.py            # Configuration schema
│   ├── memory.py            # Memory, CoreMemory, Lesson
│   ├── session.py           # Session, SessionMessage
│   ├── task.py              # TaskPlan, BackgroundTask
│   └── tools.py             # ToolDefinition, ToolCall
├── session/                 # Session lifecycle
│   ├── manager.py           # Session CRUD + dump-to-memory
│   └── models.py            # Session data classes
├── ui/                      # User interfaces
│   ├── cli.py               # Rich CLI (2300+ lines)
│   └── web/
│       └── app.py           # FastAPI application (974 lines)
├── worker.py                # Remote worker daemon
├── export.py                # Data export
├── reprocess.py             # Memory reprocessing
├── import_common.py         # Shared import pipeline
├── import_chatgpt.py        # ChatGPT parser
├── import_claude.py         # Claude parser (official + scraped)
└── import_deepseek.py       # DeepSeek parser
```

---

## 3. Memory System

The memory system is BlipShell's core differentiator. It automatically processes every
conversation into searchable, scored, entity-linked memory without user intervention.

### 3.1 Storage Layer

#### SQLite Schema

The database uses WAL mode with foreign keys. Schema is defined in `memory/sqlite_store.py`
and migrated via try-except ALTER TABLE statements.

**Core Tables:**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `memories` | Conversation messages | content, summary, rank (1-5), importance (0.0-1.0), memory_type, access_count, is_archived |
| `sessions` | Conversation sessions | title, summary, project, message_count, is_archived |
| `core_memories` | Permanent user facts | content, category, importance, is_active |
| `lessons` | Extracted behavioral insights | content, summary, rank, importance, added_by |
| `tags` | Tag definitions | name, category (topic/behavior/background) |
| `memory_tags` | Memory-to-tag junction | memory_id, tag_id |
| `core_memory_tags` | Core memory-to-tag junction | core_memory_id, tag_id |
| `lesson_tags` | Lesson-to-tag junction | lesson_id, tag_id |

**Entity Graph Tables:**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `entities` | Named entities | name, entity_type, UNIQUE(name, entity_type) |
| `entity_relationships` | Subject-predicate-object triples | subject_id, predicate, object_id, valid_from, expired_at, expired_by |
| `entity_mentions` | Which memories mention which entities | entity_id, memory_id |
| `entity_aliases` | Merge audit trail | alias_name, canonical_entity_id, merge_method |

**System Tables:**

| Table | Purpose |
|-------|---------|
| `task_plans` / `task_steps` | Multi-step task execution state |
| `background_tasks` | Async work queue with progress |
| `turn_events` | Conversation flow observability |
| `projects` | Named project metadata |
| `discovered_tag_patterns` | LLM-discovered regex patterns |
| `app_metadata` | Key-value store for app state |
| `memories_fts` | FTS5 virtual table on memory summaries |

FTS5 is kept in sync via INSERT/UPDATE/DELETE triggers on the `memories` table.

#### ChromaDB Collections

ChromaDB stores vector embeddings using `nomic-embed-text` (384-dimensional):

| Collection | Contents | Metadata |
|------------|----------|----------|
| `memories` | Memory summaries (truncated to 2000 chars) | session_id, role |
| `core_memories` | Core memory content | — |
| `lessons` | Lesson content | — |
| `entities` | Entity names (for dedup similarity) | entity_type |

Similarity metric: cosine distance (0 = identical, 2 = opposite).

### 3.2 Processing Pipeline

Defined in `memory/processor.py`. Every user/assistant message goes through this pipeline
as a background task:

```
Raw Message
    │
    ▼
┌─────────────────────┐
│  1. Noise Filter     │──── skip if noise phrase, <3 words, or <80 chars without signal words
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. LLM Summarize    │──── condense to <30 words; returns "SKIP" for meta/filler
└─────────┬───────────┘     (TaskType.SUMMARIZATION → glm4:latest or cloud)
          ▼
┌─────────────────────┐
│  3. SQLite Insert    │──── persist Memory with rank=0, importance=0.0
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  4. ChromaDB Embed   │──── vector embedding of summary (nomic-embed-text)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  4b. Dedup Check     │──── find similar existing memories via ChromaDB (threshold 0.7)
│                      │     LLM decides: ADD / UPDATE / DELETE / NONE
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  5. Tag Message      │──── regex topic patterns (74 built-in) + behavior scoring
└─────────┬───────────┘     LLM-discovered patterns merged at startup
          ▼
┌─────────────────────┐
│  6. Rank+Importance  │──── single LLM call → "4 0.75 fact"
│     +Classify        │     rank (1-5), importance (0.0-1.0), memory_type
└─────────┬───────────┘     (TaskType.RANKING_IMPORTANCE → qwen2.5:14b)
          ▼
┌─────────────────────┐
│  7. Entity Extract   │──── async batch (separate pass, not per-message)
│     (batch)          │     LLM extracts triples: subject | predicate | object
└─────────────────────┘     (TaskType.REASONING → qwen3:14b)
```

**Noise Filter** (`noise.py`):
- 32 categories of known noise phrases (greetings, affirmatives, laughter, slang)
- Signal words ("you", "feel", "think", "remember", etc.) override short-length filter
- Configurable min_word_count (default 3)

**Dedup Actions**:
- **ADD**: New information, continue processing
- **UPDATE**: New supersedes old; transfer tags, archive old memory
- **DELETE**: Old contradicts new; archive old
- **NONE**: New is redundant; archive new, stop processing

**Importance Bonuses**:
- +0.1 for recency (`importance_recency_bonus`)
- +0.05 if >6 tags (`importance_tag_bonus`)
- Capped at 1.0

### 3.3 Search Pipeline

Defined in `memory/search.py`. Queries go through a multi-strategy search with
scoring, boosting, and re-ranking:

```
Query
    │
    ▼
┌───────────────────┐
│ 1. Noise Filter    │──── skip trivial queries
└────────┬──────────┘
         ▼
┌───────────────────┐     ┌───────────────────┐
│ 2a. ChromaDB      │     │ 2b. FTS5 Keyword  │
│     Semantic      │     │     Search        │
│     Search        │     │                   │
└────────┬──────────┘     └────────┬──────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌──────────────────┐
         │ 3. RRF Fusion     │──── Reciprocal Rank Fusion (k=60)
         │    score = Σ 1/(k+rank)  combines both rankings
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ 4. Filter         │──── similarity >= 0.5, rank >= 3, exclude current session
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ 5. Temporal Decay │──── per-type decay rates (fact: slow, event: fast)
         │    + Consolidation│     consolidation bonus: 1.0 + 0.1*tanh(access_count/5)
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ 6. Tag Boost      │──── +0.1 × (matching_tags / query_tags)
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ 7. Entity Expand  │──── word-boundary match entities in query (min 3 chars)
         │                   │     → follow relationships (capped: 10 matched, 20 connected)
         │                   │     → find memories (capped: 50 IDs)
         │                   │     +0.15 boost for entity-connected memories
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ 8. Project Boost  │──── +0.15 for active project session memories
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │ 9. Final Score    │──── boosted_score = similarity + importance_boost
         │    & Sort         │     + tag_boost + rrf_boost + project_boost + entity_boost
         └──────────────────┘     importance_boost = importance × weight × decay × consolidation
```

**Decay Rates** (half-life in days):

| Memory Type | Decay Rate | ~Half-Life |
|-------------|-----------|------------|
| fact | 0.0002 | 144 days |
| preference | 0.0003 | 96 days |
| skill | 0.0001 | 289 days |
| event | 0.002 | 14 days |
| conversation | 0.001 | 29 days |

**Access Reinforcement**: Returned memories have their `access_count` incremented, which
strengthens consolidation bonus and resists future decay.

### 3.4 Token Budget Pools

Defined in `memory/manager.py`. The context window is divided into five pools with
configurable percentages and hard caps:

| Pool | % | Hard Cap | Priority | Contents |
|------|---|----------|----------|----------|
| Core | 10% | 2048 | 5 (highest) | Core memories, system identity |
| ActiveSession | 35% | none | 3 | Current conversation messages |
| RecentHistory | 15% | none | 4 | Summarized overflow from ActiveSession |
| Recall | 30% | 8192 | 2 | Search results (long-term memory) |
| Buffer | 10% | none | 1 (lowest) | Scratch space |

**Pool Mechanics**:
- `add_memory(pool_name, PoolItem)` adds text with estimated tokens (len/4)
- Overflow triggers `_trim_pool()` which removes oldest items
- ActiveSession overflow triggers async summarization into RecentHistory
- `gather_memory(token_budget, pool_budgets)` collects items by priority order

**Dynamic Query Profiles** (`query_profiles.py`):
Query text is classified by regex heuristics into profiles that adjust pool budgets:

| Profile | Recall | ActiveSession | RecentHistory | Core | Buffer |
|---------|--------|---------------|---------------|------|--------|
| recall | 45% | 20% | 15% | 10% | 10% |
| session | 20% | 45% | 17% | 8% | 10% |
| coding | 35% | 30% | 10% | 15% | 10% |
| balanced | 30% | 35% | 15% | 10% | 10% |

### 3.5 Entity Graph

Implemented in `memory/entity_extractor.py`. Builds a knowledge graph of
subject-predicate-object triples from conversation summaries.

**Extraction**: LLM extracts 1-5 triples per memory summary:
```
user | discussed | python | person | technology
python | used_for | performance tuning | technology | concept
```

**3-Stage Entity Resolution**:
1. **Exact match**: `UNIQUE(name, entity_type)` constraint catches duplicates
2. **Embedding similarity**: ChromaDB search for similar entity names
   - >= 0.85 similarity → auto-merge
   - 0.70-0.85 → LLM arbitration ("are these the same entity?")
3. **Create new**: No match found, insert new entity

**Bi-temporal Edge Tracking**:
- Relationships have `valid_from` and `expired_at` timestamps
- `expired_by` tracks which new relationship caused the expiration
- Contradicting predicates auto-expire old edges:
  - `works_at` expires: left, quit, fired_from, resigned_from
  - `prefers` expires: dislikes, avoids, stopped_using
  - `lives_in` expires: moved_from, left

**Entity Aliases**: When entities are merged, the old name is recorded in `entity_aliases`
with the merge method (exact, embedding_auto, llm_resolved) for audit.

**Search Integration**: The search pipeline finds entities mentioned in the query
using word-boundary matching (minimum 3 characters to prevent "i", "a", "go" from
matching). Results are capped at every stage to prevent fan-out: max 10 matched
entities (preferring longer/more specific names), max 20 connected entities via
relationships, and max 50 memory IDs returned. Connected memories get a +0.15 boost.

### 3.6 Consolidation & Deduplication

**During Processing** (inline dedup, Step 4b):
- New memory compared against ChromaDB neighbors
- LLM decides action (ADD/UPDATE/DELETE/NONE)
- Prevents redundant information from being stored

**Batch Consolidation** (`memory/consolidation.py`):
- Processes unconsolidated memories (`consolidated_at IS NULL`)
- For each memory, queries ChromaDB for top-5 neighbors
- Merges near-duplicates above similarity threshold (0.85)
- Winner selection: importance > rank > timestamp (newer wins)
- Loser merge: transfer tags, sum access_count, keep longer summary
- Deletes loser from both SQLite and ChromaDB

> Note: `consolidation_batch_size` defaults to 0 (disabled) after an accidental
> deletion of 1,083 imported memories during early development.

### 3.7 Tag System

Defined in `memory/tagger.py` and `memory/tag_discovery.py`.

**Topic Tags** (74 built-in):
- Regex-based matching: `python`, `c#`, `sql`, `ollama`, `chromadb`, `memory-system`, etc.
- Category: `topic`

**Behavior Tags** (scored):
- Strong triggers (2 points): explicit behavioral signals
- Soft triggers (1 point): conversational hints
- Word boundary matching to prevent false positives
- Examples: identity, emotion, core-memory, reflection, goal, drift, autonomy, architecture

**Background Triggers** (for task queuing):
- research-idea, unfinished-thought, reflection-point, task-candidate, lesson-candidate
- Detected via phrase matching; used for background task routing

**LLM Tag Discovery** (`tag_discovery.py`):
- Runs periodically (default: every 7 days) on startup
- Samples poorly-tagged memories (max_tags: 1)
- LLM suggests new regex patterns
- Validates via `re.compile()`; skips invalid patterns
- Persists to SQLite `discovered_tag_patterns` table
- Merged into runtime TOPIC_PATTERNS at startup

**Tag Application**: Behavior tags first (prioritized), then topic tags, deduplicated,
capped at `max_tags` (7).

---

## 4. LLM Subsystem

### 4.1 Task Types and Model Assignments

The LLM router maps task types to specific models. Each task type has a primary
and fallback model:

| Task Type | Primary Model | Fallback | Usage |
|-----------|--------------|----------|-------|
| `tool_calling` | glm-5:cloud | gpt-oss:latest | Interactive chat with tools |
| `coding` | glm-5:cloud | gpt-oss:latest | /code command |
| `reasoning` | qwen3:14b | gpt-oss:latest | Entity extraction, lessons, contradiction |
| `summarization` | glm4:latest | glm4:latest | Memory/session summaries |
| `ranking` | qwen2.5:14b | qwen2.5:14b | Memory ranking 1-5 |
| `importance` | qwen3:14b | qwen3:14b | Memory importance 0.0-1.0 |
| `ranking_importance` | qwen2.5:14b | qwen2.5:14b | Combined rank+importance (batch) |
| `embedding` | nomic-embed-text | — | Vector search embeddings |

**Interactive** models (tool_calling, coding) run via cloud through Ollama.
**Background** models run locally on the Ollama PC.

### 4.2 Multi-Endpoint Routing

Defined in `llm/endpoints.py`. Four endpoints are configured:

| Endpoint | Provider | Priority | Roles | Rate Limits |
|----------|----------|----------|-------|-------------|
| groq | openai | 3 (highest) | summarization | 30 RPM, 1000 RPD |
| gemini | openai | 2 | summarization | 5 RPM, 1000 RPD |
| local | ollama | 1 | all tasks | unlimited |
| old-pc | ollama | 2 | ranking (disabled) | unlimited |

**Endpoint Selection** (`get_endpoint_for_role`):
1. Find candidates: supports role + enabled + can accept request
2. Sort by: highest priority first, then fewest active requests
3. Return the best candidate
4. If no candidates: fall back to any enabled endpoint

**Per-Endpoint Model Overrides**: Endpoints can remap model names. Groq maps
`summarization` to `openai/gpt-oss-120b`; Gemini maps it to `gemini-2.5-flash`.

**Health Loop**: Background task (60s interval) pings all endpoints, re-enables
recovered endpoints, disables downed ones, and clears the model failure cache.

### 4.3 Error Classification

Defined in `llm/exceptions.py`. Critical for correct fallback behavior:

**Model Errors** (don't penalize endpoint):
- `ollama.ResponseError` with "not found" or "model" keywords
- HTTP 502/503/504 on cloud proxy (model-level, not endpoint-level)
- `RateLimitExhaustedError` (temporary)
- `openai.NotFoundError`, `openai.RateLimitError`

**Endpoint Errors** (penalize endpoint, disable after 3 failures):
- Connection refused, timeouts, network errors
- Unknown/unclassifiable errors

### 4.4 Rate Limiting and Cascading Fallback

**Rate Limiting**:
- Tracked per-endpoint: RPM and RPD counters
- `can_accept_request()` checks both limits
- When exceeded: endpoint skipped during selection

**Cascading Fallback** (in `router.generate()`):

```
1. Get endpoint for task_type (e.g., Groq for summarization)
2. Resolve per-endpoint model override
3. Check if model is marked failed → skip to fallback
4. Call endpoint.client.generate()
5. On exception:
   ├── is_model_error() → mark model failed, don't penalize endpoint
   └── NOT model_error → endpoint.record_failure()
6. Try fallback model on fallback endpoint
   ├── Pass fallback endpoint's context_tokens (num_ctx)
   └── Disable thinking if fallback_think=false
7. Fallback fails → raise original error
```

**OpenAI Rate Limit Retry** (in `openai_client.py`):
- 2 special rate-limit retries with 5s × attempt delays (5s, 10s)
- After 2 retries: raise `RateLimitExhaustedError`
- Router catches this and cascades to next endpoint

### 4.5 Client Implementations

**OllamaClient** (`llm/client.py`):
- Uses `ollama.AsyncClient` for native async
- Methods: `chat()`, `chat_stream()`, `generate()`, `embed()`, `check_health()`
- LRU response cache (200 entries) for `generate()` only
- Retry with exponential backoff: 1s → 2s → 4s
- Per-call timeout: 120s default

**OpenAICompatClient** (`llm/openai_client.py`):
- Uses `openai.AsyncOpenAI` SDK
- Same interface as OllamaClient (chat, chat_stream, generate)
- Strips Ollama-specific kwargs (options, think, format)
- Special rate-limit retry logic (2 retries × 5s)
- Tool calls passed through in OpenAI format

### 4.6 Prompt Library

All prompts are centralized in `llm/prompts.py` (23 templates):

**Memory Processing:**
- `summarize_memory()` — Condense to <30 words, third person, returns SKIP for filler
- `rank_and_importance()` — Combined scoring with 8 few-shot examples to combat clustering
- `rank_importance_and_classify()` — Adds memory_type classification (fact/event/preference/skill/conversation)
- `decide_memory_action()` — Dedup: ADD/UPDATE/DELETE/NONE
- `detect_contradiction()` — YES/NO for core memory conflicts
- `extract_lesson()` — Behavioral insights (1-3 per conversation, or SKIP)

**Entity & Tags:**
- `extract_entities()` — Triples: `subject | predicate | object | subject_type | object_type`
- `discover_tag_patterns()` — New regex patterns for tagging
- `resolve_entity_duplicate()` — YES/NO for entity merge

**Agent & Planning:**
- `generate_plan()` — Decompose request into 3-7 numbered steps
- `execute_step()` — Run single step with accumulated context
- `reflect_on_response()` — Self-critique: return improved version or NO_CHANGES
- `summarize_plan_results()` — Final summary of all completed steps

**Session:**
- `summarize_session_chunk()` — 1-2 sentences per 20-message chunk
- `summarize_session_summaries()` — Meta-summary of chunks
- `generate_session_title()` — One-sentence title

**Utility:**
- `rephrase_as_memory_style()` — Convert question to declarative for search
- `classify_task_type()` — Route to reasoning/coding/summarization/tool_calling

---

## 5. Agent Core

### 5.1 Chat Loop

Defined in `core/agent.py` (1673 lines). The Agent class is the central orchestrator.

**Initialization Pipeline:**
1. Load SQLite store and ChromaDB
2. Initialize endpoint manager with fallback routing
3. Create memory manager, processor, and search systems
4. Register all tools (grouped by category)
5. Load discovered tag patterns into the tagger
6. Health check all endpoints
7. Auto-backup if >24h since last backup
8. Trigger auto tag discovery and entity extraction

**Two Execution Paths:**

#### Simple Chat (`_chat_simple`)
The default path for most interactions:

1. Build messages array with memory context (core memories, lessons, recall, history)
2. **Tool gating**: Detect which tool groups the message needs via regex patterns
   (`detect_tool_groups()`). Pure conversation → no tools passed → faster response.
   Project mode → all tools always available.
3. **Tool rules**: Apply structural constraints (max calls, cooldowns, terminal rules)
   to filter available tools on each iteration of the tool loop
4. If LLM returns tool calls → execute tools → feed results back → loop
5. Stream tokens via `on_token` callback
6. Auto-continue: if LLM explicitly asks "should I proceed/continue?" while mid-task
   (tool calls already made), auto-nudge with "Yes, go ahead."
7. On endpoint error: fall back to secondary model

Max tool iterations configurable per-model (default 5, up to 25 for some models).

#### Planned Chat (`_chat_planned`)
For multi-step tasks (triggered by `!plan` prefix or auto-detection):

1. ComplexityClassifier decides if planning is needed (heuristic, no LLM call)
2. TaskPlanner generates numbered steps via LLM
3. TaskExecutor runs each step sequentially with focused prompts
4. Results accumulated and summarized into final response

**Complexity Classification Heuristics:**
- Requires chaining words ("and then", "after that", "save as") + action verbs
- OR: 3+ distinct action verbs + message length > threshold
- Skip patterns: questions, conversational, "what is?", "can you?"

### 5.2 Message Building

`_build_messages()` constructs the context window for each LLM call:

1. **System prompt** with core memories and lessons
2. **Search results** from Recall pool (semantic + FTS + entity graph)
3. **Recent history** summaries
4. **Active session** messages
5. **Project context** (if active): git info, code map, README, CLAUDE.md

Pool ordering mitigates lost-in-the-middle: Core → Lessons → History → Recall.

Dynamic pool budgets adjust based on query profile (coding queries get more Recall;
conversational queries get more ActiveSession).

### 5.3 Tool System

Defined in `core/tools/base.py` and individual tool files (10 files).

**Tool Groups:**

| Group | Tools | Trigger Keywords |
|-------|-------|------------------|
| filesystem | read_file, write_file, edit_file, list_directory | file, read, write, edit, create, save |
| shell | run_command | run, execute, command, terminal |
| web | web_search, web_fetch | search, look up, find, web, url |
| memory | search_memories, save_core_memory, promote_to_core, list_sessions | remember, recall, memory, session |
| coding | grep_files, glob_files, git_status, git_diff, git_add, git_commit | grep, find files, git, code |
| tasks | start_background_task, check_task, list_tasks, run_workflow | background, task, workflow |

**Tool Gating** (`detect_tool_groups()` in `tools/base.py`):
Regex patterns match keywords in user messages to determine which tool groups are
relevant. When no project is active:
- Detected groups → only those tools (plus memory/general/tasks) are passed
- No groups detected → pure conversation, no tools passed (faster, no unwanted calls)
- Project mode → all tools always available

**Tool Rules Engine** (`core/tool_rules.py`, inspired by Letta/MemGPT):
Structural constraints that filter available tools dynamically during the tool loop:

| Rule Type | Purpose | Example |
|-----------|---------|---------|
| `MaxCallsRule` | Cap per-tool calls per turn | `list_directory` max 3 |
| `CooldownRule` | Prevent immediate re-calls | `grep_files` cooldown 1 |
| `SequenceRule` | Constrain valid next-tools | After `read_file`, only `edit_file`/`write_file` |
| `TerminalRule` | Stop after specific tool | `git_commit` ends the turn |

Two rule sets: `create_default_rules()` for conversation mode, `create_coding_rules()`
for project mode (more permissive caps). Rules switch automatically on project
activation/deactivation.

**File Re-read Prevention**: `ReadFileTool` holds a reference to the agent's
`_files_read` set. If a file was already read this session, the tool returns
`"ALREADY READ: ..."` instead of re-reading — enforced at execution time, not advisory.

**Approval Gates**: Dangerous tools (write_file, edit_file, git_commit, run_command)
require user approval. Options: allow once, allow for session, deny. File edits show
colored diffs before approval.

**Type Coercion**: The registry handles LLMs sending ints as strings, converting tool
arguments to expected types.

### 5.4 Project Activation

When a project is activated (`/project open <name>`):

1. Load project metadata from SQLite
2. Dump current session to memory if switching projects
3. Re-register file tools with `root_path` restriction
4. Register coding tools (grep, glob) scoped to project
5. Register git tools (status, diff, add, commit)
6. Switch to coding-mode tool rules (`create_coding_rules()`)
7. Initialize RepoMap for AST-based code structure
7. Build project context (cached for 1 hour):
   - Basic metadata (name, root, language, git URL)
   - Git info (branch, last 5 commits)
   - AST code map (classes, functions from Python files)
   - Directory tree (skips .git, node_modules, __pycache__)
   - Key files: README.md, pyproject.toml, CLAUDE.md (first 60 lines)
8. Inject project context into system prompt

### 5.5 Session Lifecycle

Defined in `session/manager.py`. Handles message tracking, text cleaning, and
memory persistence.

**During Session**: Messages are added to the in-memory list and the ActiveSession
pool. Every 5 messages, `dump_to_memory()` fires as a background task to persist
undumped messages through the full processing pipeline.

**Session End** (`end_session()`): Three phases, each with independent timeouts:

| Phase | Timeout | What It Does |
|-------|---------|--------------|
| Dump messages | 15s per message | Process each undumped message through the full pipeline (summarize + rank + embed). Per-message timeouts so a single slow LLM call doesn't block the batch. Failed messages are skipped, not marked as dumped. |
| Session summary | 60s | Chunk messages → summarize chunks → meta-summarize → generate title |
| Lesson extraction | 30s | LLM extracts behavioral insights from the full conversation (5+ messages only) |

**Text Cleaning**: Strips whitespace, normalizes line endings, collapses double spaces.

### 5.6 Background Tasks and Workflows

**Background Task Manager** (`core/background.py`):
- `submit_task(title, task_type, prompt, target_endpoint)` → task_id
- Tasks persist in SQLite (status: pending → running → completed/failed)
- Auto-routes to remote endpoint if configured
- Progress tracking with percentage and messages

**Workflow System** (`core/workflows.py`):
- YAML templates in `workflows/` directory
- Steps support `{placeholder}` substitution
- Conditional steps: `condition: parameter_name` (truthy check)
- Converted to TaskPlan and executed by TaskExecutor

**Remote Worker** (`worker.py`):
- Polls main instance for pending tasks
- Executes using local Ollama models
- Reports completion/failure back

---

## 6. UI Layer

### 6.1 CLI Slash Commands

The CLI (`ui/cli.py`, 2300+ lines) uses Rich for colored output and Click for
argument parsing.

**Session Commands:**
| Command | Description |
|---------|-------------|
| `/quit`, `/exit`, `/q` | End session |
| `/save` | Dump current conversation to persistent memory |

**Memory Commands:**
| Command | Description |
|---------|-------------|
| `/core` | List active core memories |
| `/core delete <id>` | Deactivate a core memory |
| `/memory` | Show memory pool usage breakdown |
| `/feedback <text>` | Save user feedback for quality tuning |

**Execution Control:**
| Command | Description |
|---------|-------------|
| `/think on\|off\|toggle` | Enable/disable extended thinking mode |
| `/reflect on\|off\|toggle` | Enable/disable self-critique second pass |
| `/approve all` | Auto-approve all dangerous tools for session |
| `/approve reset` | Reset tool approvals |
| `!plan <message>` | Force planned execution (prefix, not slash command) |

**Code & Projects:**
| Command | Description |
|---------|-------------|
| `/code [--model name] <path> [instruction]` | Dedicated code analysis |
| `/plan` | Show active plan |
| `/plans` | List all plans in current session |
| `/offload <task>` | Submit background task |
| `/project new <name> <path>` | Create project |
| `/project open <name>` | Activate project |
| `/project close` | Deactivate project |
| `/projects` | List all projects |

**Observability & Status:**
| Command | Description |
|---------|-------------|
| `/status` | Agent status (session, project, endpoints, tools) |
| `/changes` | Files modified in this session |
| `/health [quick]` | Run database audit (8 check categories) |
| `/flow [turn_number]` | Conversation flow events (last 5 turns or specific) |

**Background Tasks:**
| Command | Description |
|---------|-------------|
| `/tasks` | List running background tasks |
| `/task <id>` | Show task details |
| `/workflow list` | List available workflows |
| `/workflow run <name> <params>` | Execute workflow |

### 6.2 Web API Endpoints

FastAPI application (`ui/web/app.py`, 974 lines) with optional Bearer token auth.

**WebSocket:**
- `/ws/chat` — Streaming chat with token-by-token output
  - Send: `{type: "message", content: "..."}`
  - Receive: `{type: "token", content: "..."}` per token
  - Receive: `{type: "response_complete", content: "full response"}`

**REST API:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | GET | List sessions (paginated, filterable) |
| `/api/sessions/{id}` | GET | Session details |
| `/api/memories` | GET | Browse memories (sort: recent/rank/importance) |
| `/api/memories/search` | GET | Semantic memory search |
| `/api/memories/stats` | GET | SQLite + ChromaDB statistics |
| `/api/memories/{id}` | GET/PUT/DELETE | Memory CRUD |
| `/api/core-memories` | GET | List active core memories |
| `/api/core-memories/{id}` | PUT/DELETE | Core memory CRUD |
| `/api/lessons` | GET | List extracted lessons |
| `/api/lessons/{id}` | DELETE | Delete lesson |
| `/api/config` | GET/PUT | View/update configuration |
| `/api/status` | GET | Agent status snapshot |
| `/api/endpoints` | GET | Endpoint health + model availability |
| `/api/flow` | GET | Turn events (last 50 or specific turn) |
| `/api/export/{type}` | GET | Export sessions/memories/all as JSON |

**Worker API** (for remote task execution):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/worker/poll` | GET | Get pending tasks for endpoint |
| `/api/worker/claim/{id}` | POST | Claim a task |
| `/api/worker/complete/{id}` | POST | Report completion/failure |
| `/api/worker/progress/{id}` | POST | Update progress |

**OpenAI-Compatible API** (for VS Code Continue.dev integration):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List models (blipshell, blipshell-code) |
| `/v1/chat/completions` | POST | Chat completion (streaming SSE) |

### 6.3 Streaming Architecture

- **CLI**: `on_token` callback prints each token as received via Rich
- **WebSocket**: Server sends `{type: "token", content: token}` per token
- **REST /v1**: Server-Sent Events (SSE) in OpenAI chunk format
- All paths feed through the same Agent.chat() callback mechanism

---

## 7. Scripts & Operations

### 7.1 Database Health & Audit

**`scripts/audit_db.py`** (719 lines) — 8-category health check:

| Check | What It Validates |
|-------|-------------------|
| SQLite Integrity | PRAGMA integrity_check, WAL mode |
| Memory Pipeline | Coverage: summarized, ranked, tagged, embedded percentages |
| Entity Quality | LLM artifact detection (think tags, commentary, pronouns) |
| ChromaDB Sync | Count match between SQLite memories and ChromaDB embeddings |
| Sessions | Orphaned sessions, message count mismatches |
| Tags | Coverage distribution, orphaned tag associations |
| FTS5 | Row count match between memories and FTS5 index |
| Storage | Database and ChromaDB directory sizes |
| Endpoints | Ping each endpoint, verify model availability |

Accessible via:
- CLI: `/health` or `/health quick` (skips slow ChromaDB check)
- Script: `python scripts/audit_db.py --skip-chroma --skip-endpoints`
- Programmatic: `run_audit()` function

### 7.2 Backups

**`scripts/backup_db.py`** (244 lines):

- Uses SQLite online backup API (safe while DB is open)
- Copies ChromaDB directory
- Rotation: `--keep N` deletes oldest backups beyond N (default 5)
- Pre-operation helper: `backup_before_destructive(name)` — one-liner import
- Used by: cleanup_entities, rebuild_chroma, reprocess_scores_and_lessons
- Auto-backup on startup if >24h since last backup

### 7.3 Entity Cleanup

**`scripts/cleanup_entities.py`** (581 lines) — 9-pass cleanup:

1. Delete commentary in names (think tags, LLM reasoning artifacts)
2. Strip formatting prefixes (dashes, asterisks, numbered lists)
3. Delete single-char and numeric-only names
4. Merge "user" variants into canonical entity
5. Delete pronouns and vague entities
6. Fix invalid entity types (fuzzy-match to valid set)
7. Deduplicate by name
8. Delete long names (>60 chars)
9. Clean orphaned relationships and mentions

Idempotent, supports `--dry-run`.

### 7.4 Reprocessing

**`scripts/reprocess_scores_and_lessons.py`** (313 lines):
- Reprocess rank+importance for all memories
- Regenerate lessons (delete bad ones, extract new behavioral insights)
- Resume support via `data/reprocess_progress.json`
- Progress saved after each item

**`scripts/extract_entities.py`** (159 lines):
- Concurrent batch entity extraction from all memories
- Resume via `entities_extracted_at` column
- Configurable batch_size and concurrency

**`scripts/rebuild_chroma.py`** (177 lines):
- Wipe and re-embed all collections (memories, core_memories, lessons)
- Configurable model and batch size
- Pre-op backup

### 7.5 Embedding Benchmark

**`scripts/benchmark_embeds.py`** (269 lines):
- Tests: nomic-embed-text, mxbai-embed-large, snowflake-arctic-embed:335m, bge-m3
- Metrics: embed speed, query speed, similarity scores on 10 real-world queries
- Output: Rich tables comparing speed, quality, and overall score

### 7.6 Import Pipelines

Three format-specific parsers feed into a shared pipeline (`import_common.py`):

| Source | Parser | Format |
|--------|--------|--------|
| ChatGPT | `import_chatgpt.py` | Tree structure via mapping (parent/children UUIDs) |
| Claude | `import_claude.py` | Official: flat message array; Scraped: messages with date strings |
| DeepSeek | `import_deepseek.py` | Tree with fragments[] (REQUEST/RESPONSE/THINK/SEARCH) |

**Shared Pipeline** (`import_common.py`, 370+ lines):
1. Parse format → `list[ParsedConversation]`
2. Resume support: skip already-imported sessions
3. Noise filtering per message
4. Batch mode: all conversations processed in phases (summarize all → rank all → embed all)
5. Tagging and optional entity extraction
6. Returns ImportStats

---

## 8. Testing

### 8.1 Test Architecture

Tests use **pytest** with **pytest-asyncio** (`asyncio_mode = "auto"`).

The fixture system (`tests/conftest.py`) enables testing without a running Ollama instance
by providing mock LLM responses:

**Key Fixtures:**
- `sqlite_store` — Real SQLiteStore on temporary database file
- `mock_chroma` — MagicMock ChromaStore with preset search results
- `canned_router` — Smart mock that returns task-type-specific canned responses
- `memory_processor` — Real processor wired to real SQLite + mocked Chroma/LLM
- `entity_extractor` — Real extractor wired to real SQLite + mocked LLM
- `memory_config` — Test MemoryConfig with small budgets (4096 tokens)

**Canned Router**: The most important fixture. It inspects both `task_type` and
`system` prompt to determine the correct response:

```python
# Simplified logic
if task_type == "summarization": return "User discussed Python performance..."
if task_type == "ranking_importance": return "3 0.5"
if system contains "triple": return entity extraction triples
if system contains "tag": return tag discovery patterns
if system contains "contradict": return "NO"
```

### 8.2 Unit Tests

| Test File | Coverage |
|-----------|----------|
| `test_processor.py` | `_parse_rank_and_importance()`, `_parse_rank()`, `_parse_float()`, full pipeline |
| `test_entity_extractor.py` | `_parse_triples()` parsing, `extract_batch()` integration |
| `test_tag_discovery.py` | `_parse_response()` unit tests, `maybe_run()` integration |
| `test_noise_filter.py` | Noise phrase detection, signal words, length thresholds |
| `test_memory_manager.py` | Pool allocation, token estimation, overflow |
| `test_endpoint_manager.py` | Endpoint selection, rate limiting, fallback |
| `test_model_settings.py` | Per-model config resolution |
| `test_session_manager.py` | Session CRUD, auto-summary |
| `test_search.py` | Chroma + FTS + entity graph search |
| `test_entity_resolution.py` | Entity dedup/merge logic |
| `test_entity_edges.py` | Entity relationship edges |
| `test_bitemporal_edges.py` | Temporal edge tracking, expiration |
| `test_tool_registry.py` | Tool registration, execution, type coercion |
| `test_repo_map.py` | Repository structure mapping |
| `test_memory_quality.py` | Rank/importance quality metrics |
| `test_edit_fuzzy.py` | Fuzzy file edit matching |

### 8.3 Integration Tests

| Test File | What It Tests |
|-----------|---------------|
| `test_integration_pipeline.py` | Full round-trip: store → process → search → retrieve, entity expansion, turn events, search stats |

### 8.4 Benchmark Suite

Unified runner: `python tests/benchmark_all.py --suite <name> --models <model1> <model2>`

| Suite | Runner | Purpose |
|-------|--------|---------|
| pipeline | `benchmark_models.py` | Summarize, rank, importance, entity extraction, contradiction |
| reasoning | `benchmark_reasoning.py` | Reasoning, coding, tool-calling quality |
| realdata | `benchmark_realdata.py` | Stratified sample from real DB memories |
| coding | `benchmark_coding.py` | Code generation quality (/code command) |

Results stored in `data/benchmark_*_results.json`.

---

## 9. Configuration

### 9.1 Config File Structure

`config.yaml` is the single source of configuration, self-modifiable by the agent:

```yaml
models:                    # Task-type → model name mapping
  tool_calling: "glm-5:cloud"
  coding: "glm-5:cloud"
  reasoning: "qwen3:14b"
  summarization: "glm4:latest"
  ranking: "qwen2.5:14b"
  importance: "qwen3:14b"
  ranking_importance: "qwen2.5:14b"
  embedding: "nomic-embed-text"
  # Fallbacks for each task type
  tool_calling_fallback: "gpt-oss:latest"
  # ...
  fallback_think: false

endpoints:                 # LLM endpoint definitions
  - name: "groq"
    url: "https://api.groq.com/openai/v1"
    provider: "openai"     # "ollama" or "openai"
    api_key: "${GROQ_API_KEY}"  # env var expansion
    roles: ["summarization"]
    priority: 3
    rate_limit_rpm: 30
    rate_limit_rpd: 1000
    models:
      summarization: "openai/gpt-oss-120b"  # per-endpoint override

memory:                    # Memory subsystem config
  pools: { core, active_session, recent_history, recall, buffer }
  total_context_tokens: 32768
  similarity_threshold: 0.5
  importance_boost_weight: 0.2
  decay_rates: { fact, preference, skill, event, conversation }
  dedup: { enabled, similarity_threshold }
  entity_resolution: { enabled, thresholds }

session:                   # Session lifecycle
  max_messages_before_summary: 50
  summary_chunk_size: 20
  auto_save_interval: 300

agent:                     # Agent behavior
  max_tool_iterations: 5
  system_prompt: "..."
  stream: true

tools:                     # Tool-specific settings
  shell: { timeout, allowed_commands }
  filesystem: { max_file_size, blocked_paths }
  web: { max_fetch_size, timeout }

llm:                       # LLM client settings
  max_retries: 2
  retry_base_delay: 1.0
  timeout: 300

planner:                   # Task planning
  enabled: false
  auto_approve: true
  max_steps: 7

model_settings:            # Per-model behavioral overrides
  glm-5: { max_tool_calls: 15, use_repo_map: true }
  gpt-oss: { max_tool_calls: 10, think: false }
```

### 9.2 Pydantic Config Models

Defined in `models/config.py`:

- `BlipShellConfig` — Top-level config containing all sections
- `ModelsConfig` — Task-type to model name mapping + fallbacks
- `EndpointConfig` — Single endpoint definition
- `MemoryConfig` — Pool configs, thresholds, decay rates
- `MemoryPoolsConfig` — Pool percentages and hard caps
- `DecayRatesConfig` — Per-memory-type decay rates
- `EntityResolutionConfig` — Merge thresholds
- `LLMConfig` — Retry and timeout settings
- `AuthConfig` — Optional Bearer token auth

**Environment Variable Expansion**: `resolve_env_vars()` expands `${ENV_VAR}` syntax
in API keys and other string config values.

---

## 10. Data Models

### 10.1 Memory Models (`models/memory.py`)

```python
class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"
    SKILL = "skill"

class Memory:
    id: int
    session_id: int
    role: str                    # "user" or "assistant"
    content: str                 # Raw message text
    summary: str                 # LLM-generated summary
    timestamp: datetime
    rank: int                    # 1-5 (noise → critical)
    importance: float            # 0.0-1.0
    memory_type: MemoryType
    is_archived: bool
    access_count: int            # Reinforcement counter
    last_accessed: datetime
    consolidated_at: datetime    # When consolidated (or None)
    entities_extracted_at: datetime  # When entities extracted (or None)

class CoreMemory:
    id: int
    content: str
    category: str                # general, preference, fact, personality, project, system
    timestamp: datetime
    importance: float
    source_session_id: int
    is_active: bool              # Deactivated on contradiction

class Lesson:
    id: int
    content: str
    summary: str
    timestamp: datetime
    rank: int
    importance: float
    source_session_id: int
    added_by: str                # "system" or "user"
```

### 10.2 Session Models (`models/session.py`)

```python
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Session:
    id: int
    title: str
    summary: str
    project: str
    created_at: datetime
    last_active: datetime
    is_archived: bool
    message_count: int

class SessionMessage:
    role: MessageRole
    content: str
    timestamp: datetime
```

### 10.3 Task Models (`models/task.py`)

```python
class TaskPlan:
    id: int
    session_id: int
    user_request: str
    status: PlanStatus           # planning, approved, running, completed, failed
    result_summary: str
    steps: list[TaskStep]

class TaskStep:
    id: int
    plan_id: int
    step_number: int
    description: str
    status: StepStatus           # pending, running, completed, failed
    tool_hint: str
    output_result: str
    retry_count: int

class BackgroundTask:
    id: int
    session_id: int
    title: str
    task_type: str
    prompt: str
    status: BackgroundTaskStatus # pending, running, completed, failed
    progress_pct: float
    progress_message: str
    result: str
    target_endpoint: str
```

### 10.4 Tool Models (`models/tools.py`)

```python
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter]

class ToolParameter:
    name: str
    type: ToolParameterType      # string, integer, boolean, number
    description: str
    required: bool
    enum: list[str]              # Optional allowed values

class ToolCall:
    name: str
    arguments: dict

class ToolResult:
    tool_name: str
    result: str
    success: bool
```

---

## 11. Code Review Findings

### 11.1 Things Working Well

**Memory Pipeline**: The full processing pipeline (noise → summarize → embed → dedup →
tag → rank+importance → entities) is well-structured with clear separation of concerns.
Each step is independently testable.

**LLM Routing**: The multi-endpoint routing with error classification is sophisticated.
The distinction between model errors (don't penalize endpoint) and endpoint errors
(do penalize) prevents cascading false positives.

**Fixture System**: The canned_router fixture enables fast integration tests without
Ollama. It inspects both task_type and system prompt to return appropriate mock responses.

**Config-Driven Everything**: Model assignments, endpoint config, pool percentages,
decay rates, and tool settings are all in config.yaml. No hardcoded model names in
business logic.

**Search Pipeline**: The multi-strategy search (Chroma + FTS5 + entity graph + RRF
fusion + temporal decay + tag boost) is comprehensive and each layer adds measurable
value.

**Prompt Engineering**: The `rank_and_importance` prompt with 8 few-shot examples was
specifically designed to combat the clustering tendency of local models (everything
scored as rank 4). This is a real problem with small models and the mitigation is sound.

**Pre-Operation Backups**: Every destructive script calls `backup_before_destructive()`
first. Auto-backup on startup with 24h cooldown. Good defensive practice.

### 11.2 Potential Issues

**agent.py Size**: At ~1750 lines, `agent.py` is the largest file and handles many
responsibilities: chat loop, message building, tool execution, project activation,
auto-continue, file tracking, and session management. Some responsibilities have been
extracted (tool rules into `tool_rules.py`), but further extraction of `MessageBuilder`
and `ProjectManager` classes would improve maintainability.

**cli.py Size**: Similarly large at 2300+ lines. The slash command handling could be
extracted into a command registry pattern.

**Consolidation Disabled**: `consolidation_batch_size` defaults to 0 (disabled) after
accidental data loss. This means duplicate memories can accumulate over time. The
consolidation logic should be re-validated with safety guards before re-enabling.

**Single SQLite Connection**: `SQLiteStore` uses a single `aiosqlite.Connection`. Under
heavy concurrent load (web API + background processing + entity extraction), this could
become a bottleneck. WAL mode helps with concurrent reads, but all writes still
serialize through one connection.

**Hardcoded Topic Patterns**: The 74 topic tag patterns in `tagger.py` are hardcoded.
While LLM tag discovery supplements these, the base patterns should ideally live in
config or the database.

**Schema Migration Strategy**: Migrations use try-except around ALTER TABLE statements.
This works but doesn't track which migrations have been applied. A version counter in
`app_metadata` would make this more robust.

### 11.3 Recently Fixed Issues

**Entity Search Explosion** (fixed): The entity graph expansion used substring matching
against 31K+ entity names, causing queries like "time" to match every entity containing
"time" as a substring (15,757 hits). Fixed with word-boundary matching (`\b` regex),
minimum 3-character entity names, and caps at every stage (10 matched → 20 connected →
50 memory IDs).

**Tools Always Passed** (fixed): All registered tools were passed to the model on every
turn, causing the model to use tools on conversational messages. Fixed with
`detect_tool_groups()` — pure conversation gets no tools, tool groups are detected by
regex keyword matching. Project mode still passes all tools.

**File Re-read Advisory Only** (fixed): The `_files_read` tracking set was maintained
correctly, but only injected as an advisory system message. Models ignored it. Fixed by
passing the set into `ReadFileTool` and rejecting re-reads at execution time with an
`"ALREADY READ"` error message.

**Session End Batch Timeout** (fixed): A single 30-second timeout wrapped the entire
`dump_to_memory()` batch. With 2-3 LLM calls per message, sessions with many undumped
messages would timeout and lose data. Fixed with per-message 15-second timeouts — each
message gets its own timeout, failed messages are skipped but not marked as dumped.

**Auto-Continue Too Broad** (fixed): The `_should_auto_continue()` patterns matched
generic phrases like "what do you think", "let me know", "i need to" — triggering
unwanted auto-continuation on normal conversation. Tightened to only match explicit
"should I proceed/continue?" permission-asking and clear "let me [verb]" continuation
intent.

### 11.4 Dead Code / Stale Files

- `NUL` file in repo root (Windows artifact, should be gitignored)
- `old-pc` endpoint defined but disabled — could be removed if the second PC is no longer used
- `blipshell/export.py` and `blipshell/reprocess.py` — verify if these are still used or
  superseded by scripts

### 11.5 Improvement Opportunities (Prioritized)

1. **Re-enable consolidation with safety guards**: Add a similarity floor (e.g., 0.90
   instead of 0.85), require both memories to be in the same session or same topic,
   and add a dry-run mode that logs what would be merged without acting.

2. **Extract MessageBuilder from agent.py**: The `_build_messages()` method and its
   helper functions could be a standalone class, reducing agent.py by ~200 lines and
   making context construction independently testable.

3. **Add schema version tracking**: Store a `schema_version` in `app_metadata`.
   Run migrations only for versions above the current. This prevents re-running
   completed migrations on every startup.

4. **Connection pooling for SQLite**: Consider using a connection pool (e.g., via
   aiosqlite with a custom pool) for the web API path where concurrent requests
   are expected.

5. **Move topic patterns to config/DB**: Extract the 74 hardcoded TOPIC_PATTERNS
   into either config.yaml or the `discovered_tag_patterns` table, making them
   editable without code changes.

6. **Test coverage for web API**: The web API (`app.py`, 974 lines) has no dedicated
   test file. Adding tests for the REST endpoints and WebSocket streaming would
   improve confidence in the API layer.

7. **Streaming for background tasks**: Background task results are currently polled.
   WebSocket or SSE push notifications would improve the UX for long-running tasks.

---

*Last updated: 2026-02-22*
*Updated with tool rules engine, tool gating, entity search fix, session timeouts, file re-read enforcement*
*Generated from codebase review of BlipShell v0.1.0*
