# CLAUDE.md - Project Context for BlipShell

## What is BlipShell?
Local LLM personal assistant with persistent memory. Uses Ollama for model inference.
Multi-endpoint support (local + cloud models via Ollama proxy). Config-driven model routing per task type.

## Current Model Assignments (config.yaml)
- reasoning: qwen3:14b (local) — used for main chat
- tool_calling: glm-5:cloud
- coding: qwen3-coder:480b-cloud
- summarization: glm4:latest (local)
- ranking: qwen2.5:14b (local)
- importance: qwen3:14b (local)
- embedding: nomic-embed-text (local)
- Fallbacks: gpt-oss:latest (local) for reasoning/tool_calling/coding

## Pending Tasks

### After ChatGPT Import Completes
1. **Switch reasoning to cloud by default** — Use a cloud model (e.g., gpt-oss) for interactive chat quality, with qwen3:14b as local fallback. Don't change during import since import doesn't use reasoning.
2. **Reprocess ranking/importance scores** — The first ~59% of imported conversations were scored by a cloud model that skewed heavily to rank 4 (~74%) and importance 0.8 (~52%). Need to reprocess all scores using the validated local models (qwen2.5:14b for ranking, qwen3:14b for importance) for a better distribution.
3. **Benchmark cloud vs local for background tasks** — Test whether cloud models produce better summaries/rankings/importance than local. Determine if the quality difference justifies the latency/cost for pipeline tasks (summarization, ranking, importance).
4. **Benchmark cloud models for task routing** — Test which specific cloud model is best for each task type (reasoning, tool_calling, coding). Current assignments are based on limited testing.

## Benchmark Results (tests/benchmark_reasoning.py)
- Results stored in data/benchmark_reasoning_results.json on the Ollama PC
- qwen3:14b: Best speed/quality tradeoff for local reasoning/coding (~8.6s reasoning, ~4s coding)
- gpt-oss:latest: Best quality but slow (~78s reasoning, ~66s coding). Good tool calling (~3.7s)
- gpt-oss:latest/low: Good quality, ~2x faster than default gpt-oss
- glm4:latest: Fast but no tool support, failed self-reflection test
- qwen3:30b: Too slow for local use, not worth it
- Real-data benchmark validated: glm4 for summarization, qwen2.5:14b for ranking, qwen3:14b for importance

## Architecture Notes
- Two-PC setup: Development on one PC, Ollama/benchmarks on another
- Code synced via git (github.com/jimbuschman/BlipShell)
- Ollama is up to date with thinking support
- qwen3 models degrade with think=False (hybrid-thinking architecture)
- context_tokens set at endpoint level (131072 for local), passed as num_ctx to Ollama
- Import pipeline uses: summarization, ranking, importance, embedding (not reasoning/coding/tool_calling)
