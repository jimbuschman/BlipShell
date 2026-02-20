# CLAUDE.md - Project Context for BlipShell

## What is BlipShell?
Local LLM personal assistant with persistent memory. Uses Ollama for local inference.
Multi-endpoint support: Ollama (local) + OpenAI-compatible cloud APIs (Groq, Gemini, etc.).
Config-driven model routing per task type with per-endpoint model overrides.

## Current Model Assignments (config.yaml)
- reasoning: qwen3:14b (local) — used for main chat
- tool_calling: glm-5:cloud
- coding: qwen3-coder:480b-cloud
- summarization: glm4:latest (local)
- ranking: qwen2.5:14b (local)
- importance: qwen3:14b (local)
- embedding: nomic-embed-text (local)
- Fallbacks: gpt-oss:latest (local) for reasoning/tool_calling/coding

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

### 1. LLM Routing Audit & Fallback Fixes
- Verify which model handles each task type in the running system
- Audit fallback logic: what happens when an endpoint goes down?
- Fix mid-request 429 fallback (currently retries same endpoint instead of falling to next)
- Fix context size mismatch on fallback (Groq/Gemini → local has different context limits)
- Decide cloud vs local for background tasks (summarization/ranking/importance)
- Document the final routing decisions

### 2. Conversation Flow Observability
- Log what searches return (ChromaDB hits, FTS hits, entity expansion hits)
- Log what context gets built and sent to the LLM
- Track which memory pools contribute to each response
- Provide a way to review/audit conversation flow after the fact
- Lightweight — event logging to a table, not heavy instrumentation

### 3. Full Integration Test Suite
- End-to-end test of the complete conversation flow on a DB copy
- Test memory storage → summarization → scoring → embedding → search → retrieval
- Test entity extraction → graph expansion → search enrichment
- Test lesson extraction and retrieval
- Test tag discovery and tag-based search
- Verify on real-world data, then clean up test artifacts
- Run against a copy of the DB so tests don't clutter production data

### 4. Unified Prompt Test Suite
- Test all LLM prompts with real-world data (existing benchmarks may cover some)
- Prompts to test: summarization, rank_and_importance, extract_entities, extract_lesson, contradiction_detection
- Unify existing benchmark scripts into a single "test everything" runner
- Verify each prompt produces expected output format and quality
- Use results to tweak prompts or choose different models where needed

### 5. Embedding Model Upgrade
- Benchmark newer embedding models for quality AND speed on local hardware
- Test with real queries against real memories (not synthetic data)
- Verify the system can handle re-embedding the full corpus
- Only commit to upgrade if benchmark shows clear improvement

### 6. Automated Audit / Health Check
- Script that checks DB health: entity quality, orphans, score distribution, missing summaries
- Check for LLM artifacts (reuse cleanup_entities patterns)
- Check endpoint health and model availability
- Output a summary report with severity levels
- Can run on-demand or as a quick startup check

### 7. Auto DB Backups
- Timestamped backup before destructive operations
- Periodic scheduled backups with rotation (keep last N)
- Could hook into startup flow

### 8. Prompt & Model Refinement (based on test results)
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
- **6. Upgrade embedding model** — Benchmark new embedding models, then full re-embed. Benchmark first with real queries before committing.
- **7. LLM-powered tag discovery** — LLM reviews recent memories periodically and suggests new regex patterns for the tagger. Route to cloud endpoint, run as scheduled background task.

## Architecture Notes
- Two-PC setup: Development on one PC, Ollama/benchmarks on another
- Code synced via git (github.com/jimbuschman/BlipShell)
- Ollama is up to date with thinking support
- qwen3 models degrade with think=False (hybrid-thinking architecture)
- context_tokens set at endpoint level (131072 for local), passed as num_ctx to Ollama
- Import pipeline uses: summarization, ranking, importance, embedding (not reasoning/coding/tool_calling)
