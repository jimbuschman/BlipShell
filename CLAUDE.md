# CLAUDE.md — Project Context for BlipShell

Current architecture, verified against code 2026-07-09. The old feature-by-feature
build log lives in `docs/HISTORY.md`; the grounded improvement roadmap is
`docs/SYSTEM_REVIEW.md` (2026-06-08). `config.yaml` is the source of truth for
models/endpoints/toggles — always check it before asserting an assignment.

## What is BlipShell?

Local LLM personal assistant with persistent memory. Ollama for local inference,
OpenAI-compatible cloud endpoints (OpenRouter, Groq) with config-driven routing
per task type and automatic fallback. Vision input works on vision-capable models.

**Direction (v2, decided 2026-06-15):** memory/continuity first. BlipShell has two
souls — a Claude-Code-style coding agent and a memory-centric continuous assistant.
We lean into the second; coding stays a good-enough tool (memory is the moat even
for coding: digests, lessons, decisions carried across sessions). Don't chase
coding-agent parity.

## Two-PC setup

- **Dev box** (this machine): Python 3.14, no Ollama. Validates *logic and wiring* —
  unit tests, loop-integration tests, simulation. Full pytest suite runs here.
- **Ollama PC**: runs models. Validates *model quality and behavior* — benchmarks,
  stress tests, live testing. Code synced via git (github.com/jimbuschman/BlipShell).
- Never conflate the two: a green suite here says nothing about model behavior.

## Package map

```
blipshell/
├── core/            # Agent facade + 5 mixins, ChatLoop, executor, guardrails,
│                    # nightly runner, self_reflection, tools/ (~25 tools, 13 files)
├── memory/          # sqlite_store, vector_store (sqlite-vec), search, processor,
│                    # entity_extractor/merger, consolidation, taggers, worker
├── llm/             # router, endpoints, clients (ollama + openai-compat),
│                    # ollama_gate, pii, prompts, model_settings, exceptions
├── session/         # SessionManager — messages, persistence, summaries
├── ui/              # cli.py (~4.4K lines, Rich/Click), web/ (FastAPI + WS + /v1)
├── mcp/             # MCP client — stdio transport, tool discovery
├── robotics/        # cube system + EmotionEngine (off by default, inert at rest)
├── simulate/        # multi-turn scenario runner (26 scenarios, 7 categories)
├── benchmark/       # model eval harness (`blipshell benchmark run <model>`)
└── models/          # pydantic config + data models
```

## Core loop

- `Agent` (core/agent.py) is a facade over 5 mixins: `agent_tools` (registration),
  `agent_background` (worker callbacks), `agent_project` (activation, repo map),
  `agent_session` (lifecycle, continuity loading), `agent_chat` (chat entry,
  message building, memory search).
- **One unified `ChatLoop`** (core/chat_loop.py) serves both paths: `_chat_simple`
  and the executor (`_chat_planned` → TaskExecutor → same loop via
  `chat_loop_runner`). Phases per iteration: compaction check → LLM call
  (OllamaGate-gated for local) → parse/budget/dedup tool calls → partition
  sequential (approval, ask_user) vs parallel (semaphore, max 8) → execute →
  completion check.
- Completion = `task_complete` tool (plus inline-text fallback). The complexity
  classifier was removed — `!plan` prefix forces the executor path.
- Executor extras: `[STATE]` block injection, file cache + stale-file detection,
  budget wind-down at 80%, forced completion at 95%, context compaction
  (mechanical first, LLM summarization when needed, recent messages preserved).
- **Guardrails** (core/guardrails.py) — 7 capabilities, most deterministic:
  requirement checklist (`confirm_plan`), trajectory monitor, completion audit
  (difficulty-gated: trivial tasks never call the judge LLM), correction detector
  (regex → anti-pattern lessons), context pinning, doom-loop detector,
  look-before-review gate (review intent → grounding guidance + read-first
  completion gate; validated live 2026-06-05).
- Plan mode: LLM self-restricts to read-only tools via enter/exit_plan_mode.

## Memory

- **Single database**: `data/blipshell.db` — SQLite (WAL) + FTS5 (indexes both
  summary AND raw content) + **sqlite-vec** vec0 tables (`vec_memories`,
  `vec_core_memories`, `vec_lessons`, `vec_entities`; 1024-dim,
  qwen3-embedding:0.6b). ChromaDB was removed 2026-04 — any doc mentioning it
  is historical.
- **Pipeline** (memory/processor.py, runs on the MemoryWorker thread): raw message
  persisted immediately (crash-safe, `is_processed=0`) → noise filter → summarize →
  embed → dedup (LLM decides ADD/UPDATE/DELETE/NONE) → tag → combined
  rank+importance+type call → mark processed. Startup sweep reprocesses failures.
- **Search** (memory/search.py): FTS5 + vec0 KNN fused with RRF (k=60), then
  boosts — importance, FadeMem recency (importance slows decay, each access resets
  effective age), tag overlap, active-project (+0.5), entity-graph expansion —
  optional reranker blend, Jaccard dedup of results. Retrieval declared
  good-enough for v2; do NOT tune preemptively.
- **Context assembly** (memory/manager.py): 5 token-budget pools — core 5%,
  lessons 5%, active_session 30%, recent_history 20%, recall 40% — with rollover.
- **Entity graph** (memory/entity_extractor.py): LLM triple extraction; 4-stage
  resolution — alias routing (merged names → canonical, follows chains) → exact
  match → embedding (≥0.85 auto-merge, 0.70–0.85 LLM arbitration) → create.
  Bi-temporal edges; `CONTRADICTING_PREDICATES` expires stale facts.
  **USER MANDATE: ARCHIVE, never DELETE** — merge/prune soft-archive only.
  Revive-on-re-mention un-archives pruned entities (merged husks excluded).
- Also: consolidation (near-dup merging), centroid + batch taggers, lessons with
  project scoping, project digests (stored in project metadata, auto-updated on
  session close).

## LLM routing

- `LLMRouter.generate()` (llm/router.py): TPM pre-flight check (skips to fallback
  *before* a 429), failed-model tracking, fallback cascade. Error classification
  matters: model errors mark the model failed (endpoint unharmed); bad requests
  penalize nothing; `RateLimitExhaustedError` → next endpoint.
- **OllamaGate** (llm/ollama_gate.py): serializes local Ollama calls, interactive
  preempts background. Async waiters are asyncio-cancellable and `acquire`/
  `async_gate` accept an optional `timeout` (raises `GateTimeout`). Cloud bypasses
  the gate.
- PII sanitization (Presidio → regex fallback) fires only on cloud paths;
  local calls keep raw text for search quality.
- Current assignments (see config.yaml `models:` + per-endpoint overrides):

| Task | Primary | Fallback (local) |
|---|---|---|
| tool_calling | minimax-m3:cloud (Ollama cloud) | gpt-oss:latest |
| coding | minimax/minimax-m3 (OpenRouter) | gpt-oss:latest |
| reasoning / ranking / importance / ranking_importance | qwen3:14b (local) | qwen3:14b / qwen3.5:9b |
| summarization | glm4:latest local; Groq gpt-oss-120b via endpoint priority | qwen3:14b |
| session_review | qwen3:14b (local, 32K) | qwen3:14b |
| embedding | qwen3-embedding:0.6b | — |

- Gemini endpoint exists but is disabled (free-tier burst limits). Groq serves
  ranking_importance (llama-3.3-70b) + summarization.
- **session_review is local-only since 2026-07-31.** Ollama retired
  `kimi-k2.5:cloud` with no notice; it 410'd mid-nightly and, because the 410
  wasn't classified as a model error, penalized the `local-cloud` endpoint that
  also serves interactive `tool_calling`. Consequence of going local: sessions
  over ~28K tokens no longer single-pass — they go through the chunk + merge
  path (`prepare_conversation_for_reflection` → per-chunk reflection →
  `merge_chunk_reflections`, N+1 LLM calls). Chunk reflections use a
  chunk-scoped prompt that forbids rating the session overall and forbids
  reporting anything as unresolved; without it a fragment gets judged as a
  whole session and invents "never addressed" findings that reach lessons.
  NOT yet benchmarked against the old single-pass reflections — do that on the
  Ollama PC before trusting reflection quality.

## Alive layer (the v2 soul)

- **Idle self-reflection** (core/self_reflection.py + agent.py reflection loop):
  after ~3h quiet, forms one "lingering thought" from its own prior thoughts.
  Surfaces two ways: one-shot pending injection on return, and per-turn semantic
  resurfacing (cosine prefilter → local LLM relevance judge, fail-closed, max 1).
- **Self-gravity step 1** (SHIPPED, enabled in config): per-thought weight —
  recurrence reinforces (+0.5 at ≥0.85 cosine echo), surfacing fatigues (×0.6),
  30-day half-life decay, floor 0.1; heavy thoughts render `[Thought · recurring]`.
  **The firewall**: two weight channels — "your weights" (boosted_score) drive
  retrieval untouched; "its weights" drive only the self-layer. Step 2
  (graph-relational gravity from currently-active entities) is gated on the
  step-1 live readout: does resurfacing feel like *caring vs indexing*?
- Robotics: cubes self-describe; LLM authors behavior profiles once; rules engine
  executes deterministically. Don't hardcode the "when" — cube describes how, LLM
  decides how/when. EmotionEngine is display-only (never affects responses).

## Testing

- **The validation split**: logic/wiring → HERE (pytest, seconds); model
  quality/behavior → Ollama PC only.
- `tests/` (~60 files, 1021 passing + 7 skipped): `tests/fakes.py`
  `ScriptedLLMClient` drives the REAL ChatLoop with canned turns —
  completion detection, guardrails gating, dedup validated deterministically
  (`tests/test_loop_integration.py`). `conftest.py` gives real in-memory SQLite +
  canned router.
- `blipshell simulate -t quick` — multi-turn scenarios against a real Agent
  (quick = no LLM; smoke/nightly tags need models).
- `blipshell benchmark run <model>` — ONE deep test across all 9 job types →
  `data/benchmark/report.md` (numbers only, no verdict). Ground-truth scorers are
  unit-testable here; real runs need the Ollama PC. Judge = OpenRouter
  (claude-opus-4.8), graceful-fail.
- `python scripts/test_executor.py --canned|--stress` — headless executor harness
  (Ollama PC).

## Nightly maintenance

`NightlyRunner` (core/nightly.py): ~21 ordered jobs (backup → cleanup →
backfills → entity extraction/merge/prune → taggers → consolidate → digests →
health check), each isolated, 300s/job timeout, Ollama-dependent jobs skipped
when it's down. `/nightly` command or `blipshell nightly --quiet`.
Entity merge/prune are config-gated with dry-run defaults (see the ARCHIVE
mandate above); `cleanup_entities.py --apply` and the old `entity_cleanup` job
hard-delete — do not use them.

## Conventions & hard-won rules

- **Test before commit — no exceptions.** Full suite runs here in ~1 min.
- **No quick fixes** — never propose "simplest for now" patches.
- **Trace the full call chain before fixing** (2026-03-04 lesson: 4 broken
  timeout fixes from patching one layer at a time).
- Test with real I/O, not mocks that patch asyncio.sleep.
- The ollama SDK defaults to timeout=None — LLMClient passes an explicit
  httpx.Timeout; verify whenever timeout behavior matters.
- Session-close operations deliberately have NO timeouts (session/manager.py) —
  OllamaGate serializes them behind background work; artificial timeouts lose
  work. Ctrl+C to bail. (Design decision, not an oversight.)
- qwen3 models degrade with think=False (hybrid-thinking architecture).
- `context_tokens` is set at endpoint level and passed as num_ctx.
- Windows console: keep script output ASCII-safe (cp1252 crashes).

## Known open items (as of 2026-07-09)

- Executor path lacks the vision gating/fallback that `agent_chat._run_chat_loop`
  has — project-mode vision gap; unify into ChatLoop.
- Guardrails consolidation: collapse the two LLM-judge features into the one
  difficulty-gated gate (mostly deletion; difficulty gate already shipped).
- Local fallback models are a generation behind — benchmark newer candidates on
  the Ollama PC before swapping (test-first, always).
- Consolidation throughput: ~20/night vs 17K corpus — effectively decorative.
- Memory reranker as L2 (different model on a working path) + entity-graph edge
  invalidation wiring — see docs/SYSTEM_REVIEW.md items C/E.
- Self-gravity step 2 (graph-relational) — gated on the step-1 live readout.
