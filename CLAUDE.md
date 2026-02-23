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
- [x] Memory OUT: Explicit `process_message()` call after execution (bypasses 5-message dump threshold)
- [x] Transcript saving: Full messages list saved to `data/project_transcripts/{project}__{timestamp}.json`
- [ ] **NEEDS TESTING**: Verify memories appear in executor context
- [ ] **NEEDS TESTING**: Verify coding results get processed into memory pipeline
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

### 14. Upgrade embedding model
Benchmark new embedding models, then full re-embed. Benchmark first with real queries before committing.

### 15. LLM-powered tag discovery
LLM reviews recent memories periodically and suggests new regex patterns for the tagger. Route to cloud endpoint, run as scheduled background task.

### 16. Esc to cancel current LLM call
Allow user to press Escape during streaming response to cancel the in-flight request and return to the prompt.

## Architecture Notes
- Two-PC setup: Development on one PC, Ollama/benchmarks on another
- Code synced via git (github.com/jimbuschman/BlipShell)
- Ollama is up to date with thinking support
- qwen3 models degrade with think=False (hybrid-thinking architecture)
- context_tokens set at endpoint level (131072 for local), passed as num_ctx to Ollama
- Import pipeline uses: summarization, ranking, importance, embedding (not reasoning/coding/tool_calling)
