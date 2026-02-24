# CLAUDE.md - Project Context for BlipShell

## What is BlipShell?
Local LLM personal assistant with persistent memory. Uses Ollama for local inference.
Multi-endpoint support: Ollama (local) + OpenAI-compatible cloud APIs (Groq, Gemini, etc.).
Config-driven model routing per task type with per-endpoint model overrides.

## Current Model Assignments (config.yaml)
**Interactive (cloud via Ollama):**
- tool_calling: glm-5:cloud — general chat + tool use
- coding: glm-5:cloud — /code command (benchmark winner: 13/13 checks)

**Background processing (local):**
- reasoning: qwen3:14b — entity extraction, tag discovery, lessons, contradiction detection
- summarization: glm4:latest — memory/session summaries (Groq/Gemini tried first)
- ranking: qwen2.5:14b — memory ranking 1-5
- importance: qwen3:14b — memory importance 0.0-1.0
- ranking_importance: qwen2.5:14b — combined scoring for batch import
- embedding: nomic-embed-text — vector search

**Fallbacks (local, when cloud is down):**
- tool_calling/coding/reasoning: gpt-oss:latest
- summarization/ranking/importance: same as primary (local safety net)

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

## Road to Usable — Priority Task List

### 1. LLM Routing Audit & Fallback Fixes — CODE WRITTEN, NEEDS TESTING
- [x] Fix mid-request 429 fallback — `RateLimitExhaustedError` signals router to try next endpoint (was retrying same endpoint up to 6 times, now 2 retries then move on)
- [x] Fix context size mismatch on fallback — fallback path now passes `num_ctx` from endpoint config
- [x] Don't penalize endpoints for model-level errors — `is_model_error()` in `exceptions.py` distinguishes model not found / cloud proxy 502-504 from real endpoint failures
- [x] Add fallback models for all task types — summarization/ranking/importance/ranking_importance fallbacks in config
- [x] Decide cloud vs local — reasoning back to local qwen3:14b, Groq+Gemini re-enabled for summarization overflow (priority-ordered)
- [ ] **NEEDS TESTING**: Verify 429 fallback actually cascades (Groq → Gemini → local)
- [ ] **NEEDS TESTING**: Verify model-not-found doesn't disable the endpoint
- [ ] **NEEDS TESTING**: Verify context tokens pass through on fallback path
- [ ] Document the final routing decisions

### 2. Conversation Flow Observability — CODE WRITTEN, NEEDS TESTING
- [x] `turn_events` table in SQLite — logs event_type + JSON data per turn per session
- [x] Events logged: `turn_start` (route, query length), `search_complete` (chroma/FTS/entity hits, final count), `context_built` (pool budgets, pool usage, total items), `llm_complete` (endpoint, model, fallback, tools, response length)
- [x] `MemorySearch.last_search_stats` tracks per-search hit counts
- [x] `/flow` CLI command — summary table of last 5 turns
- [x] `/flow <n>` CLI command — detailed event breakdown for a specific turn
- [x] `/api/flow` web endpoint — same data via REST
- [ ] **NEEDS TESTING**: Run a few interactive turns and check `/flow` output makes sense
- [ ] **NEEDS TESTING**: Verify events log correctly across simple and planned paths

### 3. Full Integration Test Suite — CODE WRITTEN, NEEDS TESTING
- [x] `tests/test_integration_pipeline.py` — end-to-end pipeline test
- [x] `tests/test_processor.py` — memory processor unit tests
- [x] `tests/test_entity_extractor.py` — entity extractor unit tests
- [x] `tests/test_tag_discovery.py` — tag discovery tests
- [x] `tests/conftest.py` — shared fixtures
- [ ] **NEEDS TESTING**: Run full test suite (`pytest tests/`) and fix failures

### 4. Unified Prompt Test Suite — CODE WRITTEN, NEEDS TESTING
- [x] `tests/benchmark_all.py` — unified runner, `--suite pipeline|reasoning|realdata`, `--models`, `--db`
- [x] Entity extraction benchmark (5 curated summaries + real-data on first 20 DB summaries)
- [x] Contradiction detection benchmark (6 YES/NO pairs + real core memory pairs from DB)
- [x] Combined `rank_and_importance` benchmark (production prompt, parsed with `_parse_rank_and_importance`)
- [ ] **NEEDS TESTING**: `python tests/benchmark_all.py --suite pipeline --models qwen3:14b`
- [ ] **NEEDS TESTING**: `python tests/benchmark_realdata.py --models qwen3:14b --sample 10`
- [ ] Use results to tweak prompts or choose different models where needed

### 5. Embedding Model Upgrade — BENCHMARK SCRIPT EXISTS, NEEDS RUNNING
- [x] `scripts/benchmark_embeds.py` — tests nomic, mxbai-embed-large, snowflake-arctic-embed:335m, bge-m3
- [ ] **RUN ON OLLAMA PC**: `python scripts/benchmark_embeds.py --db data/blipshell.db --sample 500`
- [ ] Analyze results: if a model beats nomic-embed-text on quality AND speed, swap in config
- [ ] If swapping: run `scripts/rebuild_chroma.py` to re-embed full corpus

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

### 8. Prompt & Model Refinement — BLOCKED on items 1-4 test results
- Tweak prompts where tests show quality issues
- Swap models for specific task types if better options found
- Re-run extraction/scoring if prompts change significantly

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
- Groq: `openai/gpt-oss-120b` — summarization only (rank skew on scoring tasks, 200K TPD free limit)
- Gemini: `gemini-2.5-flash` — summarization only (5 RPM free tier)
- Both enabled as summarization overflow — keeps GPU free for interactive reasoning
- Ranking/importance stay on validated local models (cloud models skew to rank 4)

### Cloud endpoint testing results
- Groq gpt-oss-120b: Good summaries, bad ranking (84% rank 4), 200K TPD burns fast on batch
- Gemini 2.5 flash: 5 RPM free tier too slow for batch, untested on ranking quality
- Both free tiers useless for batch import, fine for interactive background tasks

## Feature Roadmap (Future)

### 9. Project Mode Memory Integration — CODE WRITTEN, NEEDS TESTING
- [x] Memory IN: Search relevant memories before execution (`_chat_planned` → `search.search()`)
- [x] Memory IN: Pass last 10 chat messages as context so design discussions carry over
- [x] Memory IN: Inject both into executor via `memory_context` / `chat_history` params
- [x] Memory OUT: TASK_COMPLETE prompt updated to include design decisions, files changed
- [x] Memory OUT: Executor narrative builder cleans 170K transcript → ~3K useful narrative
- [x] Memory OUT: Narrative fed through `process_message()` (bypasses 5-message dump threshold)
- [x] Transcript saving: Full messages list saved to `data/project_transcripts/{project}__{timestamp}.json`
- [x] All execution paths feed memory: background tasks, plan results, workflows, dynamic execution
- [ ] **NEEDS TESTING**: Verify memories appear in executor context
- [ ] **NEEDS TESTING**: Verify coding narrative gets processed into memory pipeline
- [ ] **NEEDS TESTING**: Verify transcripts save correctly
- [ ] Future: Executor checkpointing for resume on interrupt
- [ ] Future: Context compression for long conversations

### 10. Project Mode Conversational Prompt — COMPLETE
- [x] Replaced EXECUTION MODE prompt with conversational-by-default INTERACTION MODE
- [x] LLM discusses and clarifies before acting, only uses tools when user requests action
- [x] Tools always available but prompt controls when to use them

### 11. Interactive Executor — CODE WRITTEN, NEEDS TESTING
- [x] `ask_user` tool: `blipshell/core/tools/interaction_tools.py` — LLM can ask clarifying questions mid-execution
- [x] CLI callback: `_ask_user_input()` in cli.py — prompts user with Rich formatting
- [x] Registered in `activate_project()`, unregistered in `deactivate_project()`
- [x] Benchmark-safe: returns "Make your best judgment" when no callback is set
- [x] Removed "Do NOT ask questions" from executor prompt
- [x] Prompt guides LLM to use ask_user for unclear requirements and after 2 failed attempts
- [ ] **NEEDS TESTING**: Verify ask_user tool prompts user and returns answer to LLM
- [ ] **NEEDS TESTING**: Verify benchmark still passes without interactive callback
- [ ] Future: Approval gates for destructive operations
- [ ] Future: Mid-task steering and graceful interruption

### 12. Executor Prompt Refinement — COMPLETE
- [x] Plan-before-acting: "Before making changes, briefly state your plan"
- [x] Anti-over-engineering: "Make the minimum changes needed — don't refactor, add features, or 'improve' code beyond what was asked"
- [x] Smarter error recovery: "If something isn't working after 2 attempts, use ask_user"
- [x] Richer TASK_COMPLETE format: files modified, design decisions, things to test
- [ ] Parallel tool calls — future (OpenAI API supports this, free speedup)

### 13. Message Persistence & Recovery — CODE WRITTEN, NEEDS TESTING
- [x] `session_messages` table: raw messages persisted immediately on `add_message()`
- [x] `is_processed` flag: set to True when message successfully goes through memory pipeline
- [x] Startup sweep: `_sweep_unprocessed_messages()` reprocesses failed messages on next launch
- [x] Executor narrative builder: `build_executor_narrative()` — extracts reasoning/actions from transcripts, drops tool result noise (99.7% reduction)
- [x] Project-scoped lessons: `project` column on lessons table, ChromaDB metadata, boost in `search_lessons()`
- [x] Session-close timeouts: 30s per message, 60s for lessons (model swap overhead)
- [ ] **NEEDS TESTING**: Verify messages persist and survive session close
- [ ] **NEEDS TESTING**: Verify startup sweep catches and reprocesses timed-out messages
- [ ] **NEEDS TESTING**: Verify project lessons get boosted in search
- [ ] Future: Scheduled overnight maintenance job (reprocess, health check, cleanup)

### 14. Executor Architecture Overhaul — IN PROGRESS
Research into Claude Code, Cline, SWE-agent, OpenHands, Aider, Codex CLI, and Cursor revealed
BlipShell's project mode executor has several anti-patterns that no successful coding agent uses.
Test transcript from 2026-02-24 confirmed: 37 tool calls, no TASK_COMPLETE emitted, 18 wasted
calls navigating cli.py, model hallucinated XML tool calls when tools weren't passed via API.

**7 Core Fixes (priority order):**

#### 14a. `task_complete` tool (replaces TASK_COMPLETE magic string)
- Every successful agent uses either a completion tool (Cline's `attempt_completion`, OpenHands' `AgentFinishAction`) or natural "no tool calls = done" (Claude Code, Codex CLI)
- Magic string in freeform text is an anti-pattern — models are trained to call tools, not emit protocol tokens
- Create `task_complete` tool with params: `summary`, `files_modified`, `decisions_made`
- Executor loop checks for this tool call instead of scanning text for "TASK_COMPLETE"
- Keep implicit completion (>200 chars text, no tool calls) as fallback
- Files: `executor.py` (loop logic), new tool in `tools/`, `prompts.py` (remove TASK_COMPLETE instruction)

#### 14b. Remove tool gating (always pass all tools)
- No successful agent does keyword-based tool filtering (Claude Code, Codex, Cline, OpenHands, SWE-agent all pass all tools always)
- `detect_tool_groups()` in `tools/base.py` causes: "review" stripped of tools, model hallucinates XML
- Remove `detect_tool_groups()` — always pass all tools in project mode
- For non-project chat: keep tools available, let model decide (passing tools doesn't force their use)
- This also fixes Problem 5 (XML hallucination) — model only does that when tools aren't available via API
- Files: `agent.py` (_chat_simple tool gating logic), `tools/base.py` (delete detect_tool_groups)

#### 14c. Prompt restructuring
- Move behavioral rules from user message to system prompt (all successful agents do this)
- User message = just the task, system prompt = who you are + all rules + tool descriptions
- Fewer rules (3-5 critical ones, not 15+) — weaker models can't track many constraints
- Add concrete examples (few-shot) of good tool sequences
- Tool descriptions should include usage guidance ("only read files you haven't already read")
- Files: `prompts.py` (dynamic_execution_prompt), `executor.py` (system prompt assembly)

#### 14d. Bound and paginate tool outputs
- `read_file`: add `start_line`/`max_lines` params, default 200 lines, show "File has N lines, use start_line for more"
- `grep_files`: cap results at 50 hits (SWE-agent's approach)
- `list_directory`: cap at reasonable depth
- File cache should tell model "already read" instead of re-reading
- Files: `tools/filesystem_tools.py`, `tools/coding_tools.py`

#### 14e. Budget wind-down mechanism
- Inject system message at 80% budget: "Running low on tool calls. Wrap up and use task_complete."
- SWE-agent research: successful instances finish in ~12 steps, failures grind to ~21
- Files: `executor.py` (execute_dynamic loop)

#### 14f. Actionable error messages
- Per Anthropic: "Error messages are prompts" — they guide the agent's next action
- Instead of "text not found": "text not found in file.py (350 lines). Try read_file to see current contents."
- Instead of "not a directory": "grep_files expects a directory path, not a file. Use grep_files on the parent directory or read_file for a specific file."
- Files: all tool execute() methods

#### 14g. Explicit `ask_user` guidance in prompt
- Not enough to just have the tool available — model needs concrete examples of when to use it
- Add to prompt: "Use ask_user when: requirements are ambiguous, something fails twice, choosing between approaches, destructive operations"
- Files: `prompts.py`

**Bonus fix (implement alongside):**
#### 14h. Edit linting
- SWE-agent: linting every edit improved success rate by 3% absolute
- After `edit_file` applies, run `ast.parse()` / `py_compile` on Python files
- If syntax error, reject edit and return actionable error with line number
- Files: `tools/filesystem_tools.py` (edit_file execute method)

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

### 15. Upgrade embedding model
Benchmark new embedding models, then full re-embed. Benchmark first with real queries before committing.

### 16. LLM-powered tag discovery
LLM reviews recent memories periodically and suggests new regex patterns for the tagger. Route to cloud endpoint, run as scheduled background task.

### 17. Esc to cancel current LLM call
Allow user to press Escape during streaming response to cancel the in-flight request and return to the prompt.

## Architecture Notes
- Two-PC setup: Development on one PC, Ollama/benchmarks on another
- Code synced via git (github.com/jimbuschman/BlipShell)
- Ollama is up to date with thinking support
- qwen3 models degrade with think=False (hybrid-thinking architecture)
- context_tokens set at endpoint level (131072 for local), passed as num_ctx to Ollama
- Import pipeline uses: summarization, ranking, importance, embedding (not reasoning/coding/tool_calling)
