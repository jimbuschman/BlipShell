# CLAUDE.md - Project Context for BlipShell

## What is BlipShell?
Local LLM personal assistant with persistent memory. Uses Ollama for local inference.
Multi-endpoint support: Ollama (local) + OpenAI-compatible cloud APIs (Groq, Gemini, etc.).
Config-driven model routing per task type with per-endpoint model overrides.

## Current Model Assignments (config.yaml)
**Interactive (cloud via Ollama):**
- tool_calling: glm-5:cloud — general chat + tool use
- coding: glm-5:cloud — /code command (benchmark winner: 13/13 checks)

**Background processing (cloud + local):**
- summarization: Groq gpt-oss-120b (priority 2) → local glm4 (Gemini disabled — 20-req burst limit too unreliable)
- ranking_importance: Groq llama-3.3-70b-versatile (0.4s avg) → local qwen2.5:7b fallback
- reasoning: qwen3:14b (local) — entity extraction, tag discovery, lessons, dedup, contradiction
- importance: qwen3:14b (local) — standalone importance scoring
- ranking: qwen2.5:14b (local) — standalone ranking (rarely used, ranking_importance preferred)
- embedding: nomic-embed-text (local) — vector search

**Fallbacks (local, when cloud is down):**
- tool_calling/coding/reasoning: gpt-oss:latest
- summarization: glm4:latest
- ranking_importance: qwen2.5:7b (benchmarked: 10/10 rank accuracy, 5.3s avg)

## Completed Work
- Memory consolidation (feature 4)
- Contradiction detection (feature 5)
- Dynamic pool allocation (feature 10)
- Graph memory layer (feature 9) — entity relationship triples extracted from all memories
- All imports complete (ChatGPT, Claude official, Claude scraped, DeepSeek)
- Ranking/importance scores reprocessed with validated local models
- Lessons regenerated
- Entity graph cleaned (31,469 entities, 56K relationships, 95K mentions)
- Entity type corrections applied
- Unified ChatLoop refactor — single loop in `blipshell/core/chat_loop.py`, ~250 lines dead code removed
- Real SSE streaming — tokens stream to UI as they arrive (via ChatLoop `stream_chat()`)
- Pipeline speed optimization — ranking moved to Groq llama-3.3-70b (10x faster), cloud load split
- Esc to cancel LLM call — `_poll_for_escape()` in cli.py races chat task vs escape key
- Background startup — entity extraction and unprocessed message sweep moved to MemoryWorker
- Fallback routing fix — `is_model_failed` now switches endpoint too, not just model name
- Gemini disabled — 20-request burst limit made free tier unreliable, summarization now Groq → local
- PII sanitization — Presidio NER + regex fallback, per-endpoint `pii_sanitize` flag, covers all cloud paths
- Type classification parser fix — last-word priority prevents misparse from explanatory text
- `/context` command — pool usage and context window breakdown (already existed)
- Agent modularization — 1687-line monolith → thin facade + 5 mixins (tools, background, project, session, chat)
- Executor architecture overhaul — task_complete tool, state injection, budget wind-down, edit linting
- Executor scaffolding — Claude Code-inspired state blocks, expanded prompts, context management
- Parallel tool calls — asyncio.gather with semaphore, sequential-first for approval tools
- Headless test harness — `blipshell test` with --canned, --stress, --simple-chat modes
- Plan mode — LLM self-restricts to read-only tools via enter/exit_plan_mode tools
- MCP client (Phase 1) — connect to external MCP servers, stdio transport, tool discovery and execution
- Message persistence — raw messages persisted immediately, startup sweep for unprocessed
- Centroid tagger full pass — 5,658 memories tagged, 16,634 tags assigned, 45s (zero LLM cost)

## Road to Usable — Priority Task List

### 1. LLM Routing Audit & Fallback Fixes — COMPLETE
- [x] Fix mid-request 429 fallback — `RateLimitExhaustedError` signals router to try next endpoint
- [x] Fix context size mismatch on fallback — fallback path now passes `num_ctx` from endpoint config
- [x] Don't penalize endpoints for model-level errors — `is_model_error()` in `exceptions.py`
- [x] Add fallback models for all task types — summarization/ranking/importance/ranking_importance fallbacks in config
- [x] Decide cloud vs local — Gemini primary for summarization, Groq for ranking_importance, local fallback
- [x] Unit tests validate: 429 cascade, model-not-found handling, context token passthrough (24 tests in `test_routing_fallback.py`)
- [x] Pipeline benchmark validated routing: Gemini→summarization, Groq→ranking_importance, local→reasoning

### 2. Conversation Flow Observability — COMPLETE
- [x] `turn_events` table in SQLite — logs event_type + JSON data per turn per session
- [x] Events logged: `turn_start`, `search_complete`, `context_built`, `llm_complete`
- [x] `MemorySearch.last_search_stats` tracks per-search hit counts
- [x] `/flow` CLI command — summary table of last 5 turns
- [x] `/flow <n>` CLI command — detailed event breakdown for a specific turn
- [x] `/api/flow` web endpoint — same data via REST
- [ ] Try `/flow` interactively to spot-check output quality

### 3. Full Integration Test Suite — COMPLETE
- [x] `tests/test_integration_pipeline.py` — end-to-end pipeline test
- [x] `tests/test_processor.py` — memory processor unit tests
- [x] `tests/test_entity_extractor.py` — entity extractor unit tests
- [x] `tests/test_tag_discovery.py` — tag discovery tests
- [x] `tests/conftest.py` — shared fixtures
- [x] Full suite passes: 278 passed, 4 skipped (~18 seconds)

### 4. Unified Prompt Test Suite — COMPLETE
- [x] `tests/benchmark_all.py` — unified runner, `--suite pipeline|reasoning|realdata`, `--models`, `--db`
- [x] Entity extraction benchmark (5 curated summaries + real-data on first 20 DB summaries)
- [x] Contradiction detection benchmark (6 YES/NO pairs + real core memory pairs from DB)
- [x] Combined `rank_and_importance` benchmark (production prompt, parsed with `_parse_rank_and_importance`)
- [x] `scripts/benchmark_pipeline_speed.py` — speed benchmark across 11 models (local + Groq + Gemini)
- [x] `scripts/diagnose_pipeline.py` — full pipeline diagnostic (routing, timing, concurrent DB, concurrent LLM+DB)
- [ ] Run `tests/benchmark_all.py` and `benchmark_realdata.py` on Ollama PC when time permits

### 5. Embedding Model Upgrade — COMPLETE (kept nomic-embed-text)
- [x] `scripts/benchmark_embeds.py` — tests nomic, mxbai-embed-large, snowflake-arctic-embed:335m, bge-m3
- [x] Benchmarked on Ollama PC — alternatives were too slow, nomic-embed-text retained

### 6. Automated Audit / Health Check — COMPLETE
- [x] `scripts/audit_db.py` — 8 check categories: SQLite integrity, memory pipeline, entity quality, ChromaDB sync, sessions, tags, FTS5, storage
- [x] Endpoint health check — pings each configured endpoint, verifies model availability on Ollama
- [x] `run_audit()` function for programmatic use (used by `/health` CLI command)
- [x] `/health` CLI command — runs audit inline with Rich-formatted output
- [x] `/health quick` — skips slow ChromaDB sync check
- [x] CLI flags: `--skip-chroma`, `--skip-endpoints`, `--config`

### 7. Auto DB Backups — COMPLETE
- [x] `scripts/backup_db.py` — SQLite online backup API + ChromaDB directory copy
- [x] `--keep N` rotation — deletes oldest backups beyond N
- [x] `backup_before_destructive(name)` — pre-operation backup helper, one-liner import
- [x] Pre-op backups in: `cleanup_entities.py`, `rebuild_chroma.py`, `reprocess_scores_and_lessons.py`
- [x] Auto-backup on startup if >24h since last backup (agent.py)
- [x] Auto-rotation keeps last 5 backups

### 8. Prompt & Model Refinement — IN PROGRESS
- [x] Pipeline speed benchmark (`scripts/benchmark_pipeline_speed.py`) — tested 11 models across local + Groq + Gemini
- [x] Ranking: llama-3.3-70b-versatile on Groq (0.85s, 9/10 accuracy) replaces local qwen2.5:14b (8.2s)
- [x] Dedup: kept local qwen2.5:14b (conservative, safe behavior)
- [x] Summarization: Groq gpt-oss-120b primary → local glm4 fallback (Gemini disabled)
- [x] Type classification parser fix — last-word priority instead of first-match scan (prevents "factual preference" misparse)
- [ ] Run type classification benchmark on Ollama PC to measure accuracy with prompt
- [ ] Run prompt benchmarks (`tests/benchmark_all.py`) on Ollama PC when time permits

## Benchmark Results (tests/benchmark_reasoning.py)
- Results stored in data/benchmark_reasoning_results.json on the Ollama PC
- qwen3:14b: Best speed/quality tradeoff for local reasoning/coding (~8.6s reasoning, ~4s coding)
- gpt-oss:latest: Best quality but slow (~78s reasoning, ~66s coding). Good tool calling (~3.7s)
- gpt-oss:latest/low: Good quality, ~2x faster than default gpt-oss
- glm4:latest: Fast but no tool support, failed self-reflection test
- qwen3:30b: Too slow for local use, not worth it
- Real-data benchmark validated: glm4 for summarization, qwen2.5:14b for ranking, qwen3:14b for importance

## Cloud Endpoints (OpenAI-compatible)
- Provider type `"openai"` in config.yaml — uses the `openai` Python SDK
- Per-endpoint `models:` map overrides global model names for each task type
- API keys via `${ENV_VAR}` syntax or plaintext in config
- Rate limiting: `rate_limit_rpm` / `rate_limit_rpd` — when hit, falls back to next endpoint automatically

### Current cloud routing (updated 2026-02-27)
- **Gemini**: DISABLED — 20-request burst limit made free tier unreliable (cascading 429 errors)
- **Groq** (priority 2): `llama-3.3-70b-versatile` for ranking_importance (0.4s avg, 30 RPM), `gpt-oss-120b` for summarization
- Fallback: Groq → local (glm4 for summarization, qwen2.5:7b for ranking_importance)
- Pipeline speed: ~7s per message (was ~20s with all-local ranking)

### Cloud benchmark results (scripts/benchmark_pipeline_speed.py)
- Groq llama-3.3-70b: **Best speed/quality for ranking** — 0.85s, 9/10 rank accuracy, 8/10 importance
- Groq gpt-oss-120b: Good summaries, bad ranking (7/10), slow for scoring (4.8s)
- Groq gpt-oss-20b: Slow (7.9s) and inaccurate (7/10 rank, 5/10 importance)
- Gemini 2.5 flash: 50% error rate on ranking (rate limits), fine for summarization
- Gemini 2.5 flash-lite: Fastest (0.79s) but unreliable (50% errors)
- Both free tiers useless for batch import, fine for interactive background tasks

## Feature Roadmap (Future)

### 9. Project Mode Memory Integration — COMPLETE
- [x] Memory IN: Search relevant memories before execution (`_chat_planned` → `search.search()`)
- [x] Memory IN: Pass last 10 chat messages as context so design discussions carry over
- [x] Memory IN: Inject both into executor via `memory_context` / `chat_history` params
- [x] Memory OUT: TASK_COMPLETE prompt updated to include design decisions, files changed
- [x] Memory OUT: Executor narrative builder cleans 170K transcript → ~3K useful narrative
- [x] Memory OUT: Narrative fed through `process_message()` (bypasses 5-message dump threshold)
- [x] Transcript saving: Full messages list saved to `data/project_transcripts/{project}__{timestamp}.json`
- [x] All execution paths feed memory: background tasks, plan results, workflows, dynamic execution
- [x] Unit tests validate memory integration (14 tests in `test_memory_integration.py`)
- [ ] Future: Executor checkpointing for resume on interrupt

### 10. Project Mode Conversational Prompt — COMPLETE
- [x] Replaced EXECUTION MODE prompt with conversational-by-default INTERACTION MODE
- [x] LLM discusses and clarifies before acting, only uses tools when user requests action
- [x] Tools always available but prompt controls when to use them

### 11. Interactive Executor — COMPLETE
- [x] `ask_user` tool: `blipshell/core/tools/interaction_tools.py` — LLM can ask clarifying questions mid-execution
- [x] CLI callback: `_ask_user_input()` in cli.py — prompts user with Rich formatting
- [x] Registered in `activate_project()`, unregistered in `deactivate_project()`
- [x] Benchmark-safe: returns "Make your best judgment" when no callback is set
- [x] Removed "Do NOT ask questions" from executor prompt
- [x] Prompt guides LLM to use ask_user for unclear requirements and after 2 failed attempts
- [ ] Verify ask_user works interactively (try giving an ambiguous task in project mode)
- [ ] Future: Approval gates for destructive operations
- [ ] Future: Mid-task steering and graceful interruption

### 12. Executor Prompt Refinement — COMPLETE
- [x] Plan-before-acting: "Before making changes, briefly state your plan"
- [x] Anti-over-engineering: "Make the minimum changes needed — don't refactor, add features, or 'improve' code beyond what was asked"
- [x] Smarter error recovery: "If something isn't working after 2 attempts, use ask_user"
- [x] Richer TASK_COMPLETE format: files modified, design decisions, things to test
- [ ] Parallel tool calls — future (OpenAI API supports this, free speedup)

### 13. Message Persistence & Recovery — COMPLETE
- [x] `session_messages` table: raw messages persisted immediately on `add_message()`
- [x] `is_processed` flag: set to True when message successfully goes through memory pipeline
- [x] Startup sweep: `_sweep_unprocessed_messages()` reprocesses failed messages on next launch
- [x] Executor narrative builder: `build_executor_narrative()` — extracts reasoning/actions from transcripts, drops tool result noise (99.7% reduction)
- [x] Project-scoped lessons: `project` column on lessons table, ChromaDB metadata, boost in `search_lessons()`
- [x] Session-close timeouts: 30s per message, 60s for lessons (model swap overhead)
- [x] Unit tests validate: persistence, recovery, project lessons (28 tests in `test_message_persistence.py`)
- [ ] Future: Scheduled overnight maintenance job (reprocess, health check, cleanup)

### 14. Executor Architecture Overhaul — COMPLETE
Research into Claude Code, Cline, SWE-agent, OpenHands, Aider, Codex CLI, and Cursor revealed
BlipShell's project mode executor has several anti-patterns that no successful coding agent uses.
Test transcript from 2026-02-24 confirmed: 37 tool calls, no TASK_COMPLETE emitted, 18 wasted
calls navigating cli.py, model hallucinated XML tool calls when tools weren't passed via API.

**7 Core Fixes (priority order):**

#### 14a. `task_complete` tool (replaces TASK_COMPLETE magic string)
- [x] Every successful agent uses either a completion tool (Cline's `attempt_completion`, OpenHands' `AgentFinishAction`) or natural "no tool calls = done" (Claude Code, Codex CLI)
- [x] Magic string in freeform text is an anti-pattern — models are trained to call tools, not emit protocol tokens
- [x] Create `task_complete` tool with params: `summary`, `files_modified`, `decisions_made`
- [x] Executor loop checks for this tool call instead of scanning text for "TASK_COMPLETE"
- [x] Keep implicit completion (>200 chars text, no tool calls) as fallback
- Files: `executor.py` (loop logic), new tool in `tools/`, `prompts.py` (remove TASK_COMPLETE instruction)

#### 14b. Remove tool gating (always pass all tools)
- [x] No successful agent does keyword-based tool filtering (Claude Code, Codex, Cline, OpenHands, SWE-agent all pass all tools always)
- [x] `detect_tool_groups()` in `tools/base.py` causes: "review" stripped of tools, model hallucinates XML
- [x] Remove `detect_tool_groups()` — always pass all tools in project mode (dead code fully deleted)
- [x] For non-project chat: keep tools available, let model decide (passing tools doesn't force their use)
- [x] This also fixes Problem 5 (XML hallucination) — model only does that when tools aren't available via API
- Files: `agent.py` (_chat_simple tool gating logic), `tools/base.py` (delete detect_tool_groups)

#### 14c. Prompt restructuring
- [x] Move behavioral rules from user message to system prompt (all successful agents do this)
- [x] User message = just the task, system prompt = who you are + all rules + tool descriptions
- [x] Fewer rules (3-5 critical ones, not 15+) — weaker models can't track many constraints
- [x] Add concrete examples (few-shot) of good tool sequences
- [x] Tool descriptions should include usage guidance ("only read files you haven't already read")
- Files: `prompts.py` (dynamic_execution_prompt), `executor.py` (system prompt assembly)

#### 14d. Bound and paginate tool outputs
- [x] `read_file`: add `start_line`/`max_lines` params, default 200 lines, show "File has N lines, use start_line for more"
- [x] `grep_files`: cap results at 50 hits (SWE-agent's approach)
- [x] `list_directory`: cap at reasonable depth
- [x] File cache should tell model "already read" instead of re-reading
- Files: `tools/filesystem_tools.py`, `tools/coding_tools.py`

#### 14e. Budget wind-down mechanism
- [x] Inject system message at 80% budget: "Running low on tool calls. Wrap up and use task_complete."
- [x] SWE-agent research: successful instances finish in ~12 steps, failures grind to ~21
- Files: `executor.py` (execute_dynamic loop)

#### 14f. Actionable error messages
- [x] Per Anthropic: "Error messages are prompts" — they guide the agent's next action
- [x] Instead of "text not found": "text not found in file.py (350 lines). Try read_file to see current contents."
- [x] Instead of "not a directory": "grep_files expects a directory path, not a file. Use grep_files on the parent directory or read_file for a specific file."
- Files: all tool execute() methods

#### 14g. Explicit `ask_user` guidance in prompt
- [x] Not enough to just have the tool available — model needs concrete examples of when to use it
- [x] Add to prompt: "Use ask_user when: requirements are ambiguous, something fails twice, choosing between approaches, destructive operations"
- Files: `prompts.py`

**Bonus fix (implement alongside):**
#### 14h. Edit linting
- [x] SWE-agent: linting every edit improved success rate by 3% absolute
- [x] After `edit_file` applies, run `ast.parse()` / `py_compile` on Python files
- [x] If syntax error, reject edit and return actionable error with line number
- Files: `tools/filesystem_tools.py` (edit_file execute method)

**Additional bugs found and fixed during implementation:**
- [x] Pagination cache corruption: executor cached paginated output, re-paginated on re-read — fixed
- [x] Disconnected file tracking: executor's `files_read` not wired to `ReadFileTool` (found by glm-5 code review) — fixed
- [x] TerminalRule for `task_complete` added to executor loop termination logic
- [x] Legacy `TASK_COMPLETE` string check removed from executor
- [x] CLI display fix: `task_complete` summary now shown in green panel

**Test results:**
- 34 tool calls, zero wasted re-reads, `task_complete` called, budget wind-down respected

**Remaining items:**
- [x] Windows shell compatibility — fixed: shell error messages redirect to correct tools, OS hints in tool descriptions
- [ ] Speed optimization — `num_ctx` tuning, model alternatives for faster execution

**Future considerations (after core fixes tested):**
- Architect/Editor split (Aider) — separate problem-solving from edit formatting, useful for weaker models
- LSP integration (Cursor) — deterministic symbol lookup instead of LLM grepping
- Mini-SWE-agent simplicity — if tool system still struggles, consider bash-only fallback

**Research sources:**
- [Claude Code: Behind the Scenes](https://blog.promptlayer.com/claude-code-behind-the-scenes-of-the-master-agent-loop/)
- [Cline System Prompt](https://cline.bot/blog/system-prompt)
- [SWE-agent ACI](https://swe-agent.com/1.0/background/aci/)
- [OpenHands Paper](https://arxiv.org/abs/2407.16741)
- [Aider Architect Mode](https://aider.chat/2024/09/26/architect.html)
- [Codex Agent Loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Anthropic: Writing Effective Tools](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

### 15. Executor Scaffolding Overhaul — COMPLETE
Replicated Claude Code's scaffolding approach to fix the fundamental gap:
the model lacked state awareness and the prompts were too sparse.

- [x] **State injection**: `_build_state_block()` injects `[STATE]` block before every LLM turn — tool call count/budget, files read/created/edited, recent actions
- [x] **Expanded system prompt**: From 5 rules to structured manual (~2300 chars) — How to Work, Tool Selection, Critical Rules, Completion, Error Recovery
- [x] **Richer tool descriptions**: read_file references [STATE] block and warns about re-reads, write_file warns to prefer edit_file, list_directory explains when to use vs grep/glob
- [x] **Executor context management**: `_estimate_messages_tokens()` + `_compact_messages()` compress older tool results at 70% context usage
- [x] **Forced completion at 95%**: Strips all tools except task_complete
- [x] **Same-args deduplication**: Blocks identical consecutive tool calls
- [x] **OS-aware shell guidance**: Shell tool errors redirect to correct tools (e.g. "Use grep_files tool instead")
- [x] **Interactive command prevention**: Shell tool warns about timeouts and launching apps
- [x] **Memory budget from endpoint**: `total_context_tokens` derived from endpoint `context_tokens` (single source of truth)
- [x] **Flow events for executor**: `/flow` now shows data for planned (executor) turns, not just simple chat

**Test results (2026-02-24):** 36+ tool calls, task_complete fired autonomously, model self-corrected IndentationError, no nudging needed.

### 16. Headless Test Harness — COMPLETE
- [x] `scripts/test_executor.py` — bootstraps full Agent, runs task via `agent.chat()`, outputs structured JSON report
- [x] `blipshell test` CLI command — delegates to test script
- [x] `--canned` mode: 3 quick tests (file creation, grep, read+edit)
- [x] `--stress` mode: 65 tests across 12 categories (tool coverage, multi-step, error recovery, scaffolding, simple chat, real-world, edge cases, observability, consistency, heavy, git, instruction)
- [x] `--simple-chat` mode: 14 tests for conversational + tool usage paths
- [x] JSON reports with: tool calls, errors, warnings, timing, flow events, files read/created/edited, completion method
- [x] `--quiet` mode for overnight runs (JSON only, no streaming)
- [x] A/B comparison mode for scaffolding experiments
- [x] Completion detection fix: `capture_inline_text` captures substantial text alongside tool calls as fallback response (predicted to fix 17/20 false negative failures in stress test)
- [x] First stress test run: 45/65 passed (69%), 17/20 failures were false negatives (model did work but didn't call `task_complete`)
- [x] `--canned` passed on Ollama PC
- [x] `--stress` run 3x overnight, analyzed results, harness bugs fixed (44/65 raw, ~91% adjusted)

### 17. Upgrade embedding model — COMPLETE (kept nomic-embed-text)
Benchmarked alternatives (mxbai-embed-large, snowflake-arctic-embed:335m, bge-m3) — all too slow. Keeping nomic-embed-text.

### 18. Tag Discovery Overhaul + Nightly Jobs — COMPLETE
Complete rewrite of tag discovery with three-layer approach plus nightly job infrastructure.

**Centroid Tagger** (`blipshell/memory/centroid_tagger.py`):
- [x] Computes centroid embedding vectors per tag from ChromaDB (zero LLM cost)
- [x] Assigns tags to poorly-tagged memories via cosine similarity (threshold 0.75)
- [x] Config: `centroid_tag_similarity`, `centroid_tag_min_members`, `centroid_tag_batch_size`
- [x] Full-pass results: 17,188 memories checked, 5,658 tagged, 16,634 tags assigned, 60 centroids built, 45s

**LLM Batch Tagger** (`blipshell/memory/batch_tagger.py`):
- [x] Sends batches of 10 poorly-tagged memory summaries to LLM for direct tag assignment
- [x] Validates assigned tags against known tag set (strict match)
- [x] Prompt in `prompts.py:batch_assign_tags()`
- [x] Config: `batch_tag_batch_size`, `batch_tag_max_batches`

**Nightly Job Runner** (`blipshell/core/nightly.py`):
- [x] Orchestrates 7 jobs: backup → cleanup → centroid_tag → batch_tag → prune → consolidate → tag_discovery
- [x] Each job isolated (one failure doesn't abort the run)
- [x] `NightlyRunner.create_from_config()` factory for headless execution
- [x] Stats persisted to `app_metadata["nightly_last_run"]`
- [x] `/nightly` interactive CLI command + `blipshell nightly` Click subcommand
- [x] `--quiet` flag for scheduled runs (JSON output only)
- [x] `--job` flag to run a single job

**Benchmark** (`scripts/benchmark_tagger.py`):
- [x] Tests model quality on tag assignment (precision/recall/F1 vs ground truth)
- [x] Rich table output + JSON results file
- [x] Tagger benchmark ran, model selected and configured

**Supporting changes**:
- [x] `sqlite_store.py`: 5 new methods (tag counts, poorly-tagged IDs, well-tagged sample, all tag names)
- [x] `chroma_store.py`: `get_embeddings_by_ids()` for raw vector retrieval
- [x] `config.py`: 5 new `MemoryConfig` fields
- [x] `/nightly` ran successfully (2026-02-28)
- [ ] Run `blipshell nightly --quiet` headless (verify JSON output)
- [ ] Run benchmark on Ollama PC to pick best model

**Future: In-process scheduler**:
- Add background asyncio task that triggers nightly run at ~3:12 AM if app is open and idle
- Check `app_metadata["nightly_last_run"]` staleness (>24h)
- Pause GPU-heavy jobs (batch_tag) if user activity resumes

**Future: Additional nightly jobs to add to runner**:
- `health_check` — run `audit_db.py` checks, log results (code exists, needs wrapper)
- `sync_check` — verify ChromaDB/SQLite aren't drifting (check exists in audit_db.py)
- `entity_cleanup` — prune orphaned entities, merge near-duplicates incrementally (needs code)
- `session_backfill` — generate summaries for sessions interrupted before summarization (needs code)
- `stale_lessons` — flag/archive lessons not relevant in 90+ days (needs code)
- `re_embed` — re-embed full corpus after embedding model swap (needs code, heavy job)

**Future: Health check feedback loop**:
- Store nightly health check reports (JSON in `data/` or `nightly_reports` table)
- On session startup, surface flagged issues as system notification
- LLM analyzes trends across reports ("sync drift increasing", "entity extraction spam")
- Generate actionable report for user review with suggested fixes

### 19. Esc to cancel current LLM call — COMPLETE
- [x] `_poll_for_escape()` in cli.py — Windows-native msvcrt key detection
- [x] Races chat task vs escape key poller using `asyncio.wait(FIRST_COMPLETED)`
- [x] Cancels pending chat task, saves partial response with `[cancelled]` suffix
- [x] `_drain_keyboard()` clears buffered keypresses after cancellation
- [x] Integrated in both main chat loop and `/code` command
- [x] Help text mentions "Press Esc during a response to cancel"

### 20. PII / Sensitive Data Sanitization — COMPLETE
PII detection in `blipshell/llm/pii.py`. Uses Microsoft Presidio (spaCy NER) when installed, regex fallback otherwise. Intercepts `router.generate()` before cloud endpoints.
- [x] Presidio engine: detects names, addresses, locations, plus all structured PII via NER + regex
- [x] Regex fallback: email, phone (US), SSN, credit card, GitHub tokens, AWS keys, generic API keys, IPs
- [x] API key patterns run on both engines (Presidio doesn't detect these)
- [x] IP exclusions: localhost, LAN (192.168.x.x, 10.x.x.x) preserved in both engines
- [x] Per-endpoint `pii_sanitize` flag: auto-detects for openai provider, explicit for cloud-via-Ollama (local-cloud)
- [x] Config: `pii.enabled` toggle in config.yaml, wired to router via `pii_enabled` flag
- [x] Local LLM calls NOT sanitized — raw text preserved for search quality
- [x] Tests: 34 tests — structured PII, regex fallback, Presidio-specific (skipped when not installed)
- [x] Dependencies: `presidio-analyzer`, `presidio-anonymizer`, `spacy` in pyproject.toml
- [x] Presidio + spaCy installed on Ollama PC (2026-02-28)

### 21. Modularize agent.py — COMPLETE
Refactored 1687-line monolith into thin facade + 5 mixin classes:
- [x] `agent_tools.py` — ToolsMixin: tool registration (105 lines)
- [x] `agent_background.py` — BackgroundMixin: memory worker callbacks, reflection (91 lines)
- [x] `agent_project.py` — ProjectMixin: project activation/deactivation, context scanning (251 lines)
- [x] `agent_session.py` — SessionMixin: session lifecycle, memory loading, startup tasks (237 lines)
- [x] `agent_chat.py` — ChatMixin: chat pipeline, message building, event logging (529 lines)
- [x] `agent.py` stays as facade (612 lines): __init__, initialize, end_session, status/properties
- [x] External callers (cli.py, web/app.py, test_executor.py) need zero changes
- [x] Mixin pattern avoids rewriting self.* references — all shared state stays on Agent

### 22. Entity Matching Performance — COMPLETE
- [x] Profiled: 31K entity names × regex per query = 1,160ms per query
- [x] Fix: substring pre-filter (`name in query_lower`) before regex — 340x speedup (1,160ms → 3.4ms)
- [x] Applied in `search.py:_expand_via_entities()` — 3-line change

### 23. `/context` command — COMPLETE
- [x] `_print_context()` in cli.py — shows pool usage breakdown, context window state, memory counts per pool
- Already existed, was just missing from roadmap tracking

### 24. Parallel tool calls in executor — COMPLETE
- [x] `chat_loop.py`: `enable_parallel` and `max_parallel` config fields on `LoopConfig`
- [x] 7-phase execution: parse → budget → dedup → display → partition → execute → results
- [x] Parallel tools via `asyncio.gather()` with `Semaphore(max_parallel=8)`
- [x] Sequential tools first (approval-required, ask_user), then parallel (reads, greps, etc.)
- [x] Result ordering preserved — appended to messages in original tool_call order
- [x] Batch dedup — catches duplicates within batch AND across batches
- [x] Error isolation — one failing tool doesn't crash the batch
- [x] Callbacks remain sequential (Phase 7) — no thread-safety changes needed
- [x] Parallel execution verified via stress tests (up to 22 tools in one batch)

### 25. Project Digest — CODE WRITTEN, NEEDS TESTING
Auto-maintained summary of project state across sessions. Solves the "full picture" problem —
when activating a project, the LLM now gets a structured digest of what's been done, key decisions,
current status, and open issues.

- [x] `ProjectDigestManager` in `blipshell/memory/project_digest.py` — bootstrap + incremental update
- [x] 3 prompt functions in `prompts.py` — `generate_initial_digest`, `update_digest_incremental`, `update_digest_with_sessions`
- [x] 2 new SQLite methods — `list_sessions_chronological()`, `get_lessons_by_project()`
- [x] Storage in `projects.metadata_json["digest"]` with timestamp and session ID tracking
- [x] Auto-update in `end_session()` — after lesson extraction, updates digest with new session summary
- [x] Injection in `activate_project()` → `_scan_project_context()` — digest loaded from metadata (no LLM call)
- [x] `/project digest` CLI command — show current digest with metadata
- [x] `/project digest rebuild` — regenerate from scratch
- [x] `rebuild_digests` nightly job — rebuilds all project digests
- [x] Bootstrap uses progressive batching (5 sessions at a time) + lesson folding
- [x] Uses REASONING model (qwen3:14b) for synthesis
- [ ] **NEEDS TESTING**: Run `/project digest rebuild` on a project with session history
- [ ] **NEEDS TESTING**: Verify digest updates automatically on session close
- [ ] **NEEDS TESTING**: Check digest injection via `/context` after project activation

### 26. MCP Client Support — CODE WRITTEN, NEEDS TESTING
Model Context Protocol client — BlipShell can connect to external MCP servers and use their tools.
Python SDK: `pip install mcp`. Phase 1 (stdio transport) implemented.

**What's implemented**:
- [x] `blipshell/mcp/manager.py` — `MCPManager`: starts subprocesses, manages sessions via `AsyncExitStack`, discovers tools, routes calls
- [x] `blipshell/mcp/tools.py` — `MCPTool(Tool)`: wraps MCP tools as BlipShell tools, handles timeouts and errors
- [x] `blipshell/mcp/schema.py` — Converts MCP JSON Schema to `ToolDefinition` (type mapping, parameter flattening)
- [x] `MCPServerConfig` in config.py — command, args, env (${ENV_VAR}), auto_approve, timeout
- [x] `mcp_servers` list in `BlipShellConfig` — config-driven server definitions
- [x] Agent init: connects enabled servers, registers tools in `ToolRegistry` with `mcp_{server}_` prefix
- [x] Agent cleanup: disconnects all servers on shutdown
- [x] `/mcp` CLI command — shows connected servers; `/mcp tools [server]` lists tools
- [x] Non-auto-approve tools require user confirmation
- [ ] **NEEDS TESTING**: Configure a test MCP server and verify tool discovery + execution
- [ ] **NEEDS TESTING**: Verify timeout handling (server hangs → clean error)
- [ ] **NEEDS TESTING**: Verify subprocess cleanup on disconnect

**Future phases**:
- Phase 2: HTTP/SSE transport, reconnection, Windows npx handling
- Phase 3: Dynamic server management, resource support, tool search for context budget

**MCP Server Research (2026-02-28)**:

Servers to integrate (high value, low overlap):
- **GitHub Official** (by GitHub, Go binary): 51 tools — issues, PRs, code search, notifications, security alerts. Use `--toolsets repos,issues,pull_requests` to reduce context (~30 tools). [github.com/github/github-mcp-server](https://github.com/github/github-mcp-server)
- **Context7** (library docs): 2 tools — `resolve-library-id` + `query-docs`. Fetches current, version-specific API docs. Free. Fixes #1 coding agent problem (hallucinated APIs). `npx -y @upstash/context7-mcp`. [github.com/upstash/context7](https://github.com/upstash/context7)
- **Telegram Communicator**: 4 tools — `ask_user`, `notify_user`, `send_file`, `zip_project`. LLM-to-user communication via Telegram bot. Great for background task notifications and remote ask_user. `npx -y mcp-communicator-telegram`. [github.com/qpd-v/mcp-communicator-telegram](https://github.com/qpd-v/mcp-communicator-telegram)

Servers to consider (medium value):
- **Sequential Thinking**: 1 tool — structured reasoning scratchpad with revision/branching. Better as native BlipShell tool (no MCP overhead). `npx -y @modelcontextprotocol/server-sequential-thinking`
- **Playwright** (Microsoft): 34 tools — full browser automation, accessibility tree snapshots, test assertions. Only option for JS-heavy sites. Heavy (requires Chromium). Phase 2. `npx -y @playwright/mcp`
- **Brave Search**: 6 tools — web/news/image/video search with freshness filtering and AI summaries. Better than ddgs but requires API key. `npx -y @smithery/cli install brave`

Servers to skip (redundant with BlipShell):
- Filesystem (13 tools) — BlipShell's native tools are better integrated (edit linting, file tracking)
- Fetch — BlipShell's web_fetch covers this; borrow pagination idea (start_index/max_length)
- Memory — JSONL file with substring search, BlipShell's memory is far superior
- Git (12 tools) — BlipShell has native git + run_command; borrow git_log date filtering idea
- Desktop Commander (18 tools) — fully redundant; borrow in-memory code execution idea

Ideas to steal for native tools (no MCP needed):
- `read_multiple_files` — batch reads in one call (from Filesystem)
- `web_fetch` pagination — `start_index`/`max_length` for long pages (from Fetch)
- `git_log` with date filtering — "what changed last week?" (from Git)
- Edit dry-run mode — preview changes before applying (from Filesystem)
- In-memory code execution — `run_python(code)` without file creation (from Desktop Commander)
- `directory_tree` — recursive JSON tree (from Filesystem)
- `sequential_thinking` — native reasoning scratchpad with revision/branching (from Sequential Thinking)

Context budget impact (131K local context):
- BlipShell native tools (~25): ~7.5K tokens
- + GitHub (~30 filtered): ~9K tokens
- + Context7 (2): ~600 tokens
- + Telegram (4): ~1.2K tokens
- Total: ~18.3K tokens of schema, leaves ~113K for conversation. Manageable.

### 27. Plan Mode (LLM-Controlled) — CODE WRITTEN, NEEDS TESTING
LLM can self-restrict to read-only tools before making complex changes.

**What's implemented**:
- [x] `read_only: bool` property on `Tool` base class — 13 tools tagged as read-only
- [x] `ToolRegistry.in_plan_mode` / `get_plan_mode_tools()` — filters to read-only tools + exit_plan_mode
- [x] `tool_provider` callback in `LoopConfig` — ChatLoop checks each iteration for current tool list
- [x] `enter_plan_mode` / `exit_plan_mode` tools in `blipshell/core/tools/plan_tools.py`
- [x] Wired into both `_chat_simple()` and `execute_dynamic()` via tool_provider
- [x] Safety reset: plan mode auto-clears if loop exits while still planning
- [x] Prompt guidance added to executor system prompt (rule 8)
- [ ] **NEEDS TESTING**: Verify plan mode filters tools correctly mid-loop
- [ ] **NEEDS TESTING**: Verify write tools blocked during plan mode
- [ ] **NEEDS TESTING**: Verify exit_plan_mode restores full tool set
- [ ] **NEEDS TESTING**: Verify canned tests still pass with plan tools registered

### Wish List
Items worth doing eventually but not prioritized:
- **Telegram notifications**: Via mcp-communicator-telegram — background tasks notify you on phone, executor can ask_user remotely. Set up @BotFather bot + chat ID.
- **Hooks system**: Shell commands that fire at lifecycle events (before commit, after file edit, on session start). User-configured, not automatic. Power-user feature for automation.
- **IDE integration**: VS Code extension for inline chat, file context, diagnostics. Big build (TypeScript extension API). Would dramatically expand UX but significant effort.

## Code Review Fixes (from independent review, 2026-03-02)

### 28. Add Missing Database Indexes — COMPLETE
- [x] `memories.is_archived`, `memories.memory_type`, composite `(session_id, is_archived)`
- Note: `entity_mentions.entity_id` already had an index (`idx_entity_mentions_entity`)

### 29. Path Traversal Fix in Filesystem Tools — COMPLETE
- [x] `_validate_within_root()` checks `resolved.relative_to(root_path)` after `.resolve()`
- [x] Applied to all 4 tools: `read_file`, `write_file`, `edit_file`, `list_directory`

### 30. Symlink Check Before Write — COMPLETE
- [x] `_check_symlink()` rejects symlink targets in `write_file` and `edit_file`

### 31. Initialize Race Condition — COMPLETE
- [x] `asyncio.Lock` on `Agent.initialize()` — extracted body to `_do_initialize()`

### 32. Exception Swallowing Sweep — COMPLETE
- [x] 13 bare `except Exception: pass` blocks replaced with `logger.warning()`/`logger.debug()`
- [x] Files: agent.py, agent_chat.py, agent_project.py, agent_session.py, executor.py, nightly.py, processor.py

### 33. SQLite/ChromaDB Sync Drift — NOT STARTED
Deletes/archives in SQLite and ChromaDB happen as separate calls with no transactional guarantee. Over time causes permanent drift between stores.
- [ ] Queue failed ChromaDB operations for retry
- [ ] Consider `sync_status` column or reconciliation job (audit_db.py already has a sync check)
- Files: `processor.py`, `consolidation.py`

### 34. FTS5 Backfill Runs Every Startup — COMPLETE
- [x] Tracks `fts5_backfill_done` in `app_metadata` — full-table scan only runs once

### 35. WebFetch SSRF Protection — COMPLETE
- [x] `_is_ssrf_target()` blocks private IPs, cloud metadata, localhost, non-HTTP schemes

### 36. Cache Key Collision Fix — COMPLETE
- [x] SHA-256 hash replaces string concatenation for cache keys

### 37. Rate Limiting Race Condition — COMPLETE
- [x] `start_request_atomic()` on EndpointManager runs under asyncio lock

### 38. Session State Validation — COMPLETE
- [x] `chat()` guards with `RuntimeError` if `session_manager` is None

## Architecture Notes
- Two-PC setup: Development on one PC, Ollama/benchmarks on another
- Code synced via git (github.com/jimbuschman/BlipShell)
- Ollama is up to date with thinking support
- qwen3 models degrade with think=False (hybrid-thinking architecture)
- context_tokens set at endpoint level (131072 for local), passed as num_ctx to Ollama
- Import pipeline uses: summarization, ranking, importance, embedding (not reasoning/coding/tool_calling)
