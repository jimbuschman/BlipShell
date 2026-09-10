# CLAUDE.md — Project Context for BlipShell

Current architecture, verified against code 2026-08-05. The old feature-by-feature
build log lives in `docs/HISTORY.md`; the active plan is `docs/V3_PLAN.md`
(2026-09-08: evidence contract + accountable lessons, supersedes the sequencing
in `docs/V2_PLAN.md`, whose Phase 4/5 leftovers still live there).
`config.yaml` is the source of truth for models/endpoints/toggles — always check
it before asserting an assignment.

**Verify before you document.** This file has twice described features that had
no call site (`[STATE]` injection, budget wind-down) and counts that had drifted
by 40%. A claim here should be greppable.

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

- **Dev box** (this machine): Python 3.14, no local Ollama (GT 710 — no usable
  GPU). Validates *logic and wiring* — unit tests, loop-integration tests,
  simulation. Full pytest suite runs here.
- **Ollama PC** (HPBENDERTWO): runs models. Validates *model quality and
  behavior* — benchmarks, stress tests, live testing. Code synced via git
  (github.com/jimbuschman/BlipShell).
- **Since 2026-08-10 the dev box reaches the Ollama PC's models over Tailscale**
  (URL passed explicitly, e.g. `benchmark run --url`, when that PC is on; the
  repo is public — never commit the Tailscale URL. There is NO config
  override file or mechanism — a `config.local.yaml` was documented here for
  a month and never existed — and the dev box holds no cloud API keys, so
  anything that boots the full Agent with production routing, `simulate`
  included, runs on the Ollama PC): model-touching work — `benchmark run
  --url`, live prompt validation — can be DRIVEN from here, executing on the real GPU. The
  split still holds as a statement about *judgment* (a green suite here says
  nothing about model behavior; measure on real hardware), no longer as a
  statement about *access*. That GPU is shared with live BlipShell — ask
  before tying it up for a long run. Don't add the Tailscale endpoint to
  config.yaml (it's the Ollama PC's live production config).

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
├── ui/              # cli.py (~2.2K, Rich/Click) + views/commands/state/importers,
│                    # web/ (FastAPI + WS + /v1)
├── robotics/        # cube system + EmotionEngine (off by default, inert at rest)
├── simulate/        # multi-turn scenario runner (40 scenarios, 8 categories;
│                    # defaults to a throwaway temp DB — see --db / --real-db)
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
- **Endpoint recovery happens INSIDE the loop, at the model-call boundary**
  (2026-09-09, V3 A2). `LoopConfig.on_model_call_error` returns the next
  `(client, model, chat_kwargs)`; `ChatLoop._call_model` retries the SAME
  call there with `messages` untouched. `_run_chat_loop` is the endpoint
  selector that installs it. The old snapshot-and-rewind around each attempt
  kept the transcript clean but re-ran every tool of the failed attempt on
  the next endpoint (reproduced under budget=1) — a transcript repair is
  never a side-effect rollback. Every endpoint that failed this turn is
  excluded, not just the last one. `tests/test_endpoint_retry.py`.
- **Every announced tool call gets a result** (2026-09-09, V3 A3). Calls past
  the remaining budget are DENIED with `BUDGET_DENIED_RESULT` (never run,
  counted, dedup'd or passed to `on_tool_executed`); the completion tool is
  never denied. Slicing used to leave them announced and unanswered, which
  OpenAI-compatible endpoints 400 on. `repair_tool_pairing()` runs before
  every send as a backstop and logs at WARNING when it had to act; a raising
  tool becomes a failure result on both the sequential and parallel paths.
  `tests/test_tool_budget_pairing.py`.
- **Only `read_only` tools run in parallel** (2026-09-09, V3 A5). The
  partition used to sequence only approval-gated tools (when a callback
  existed) and `ask_user`, so in normal chat every mutating tool in a batch
  ran concurrently. `read_only` - the flag plan mode already trusts - is the
  effect class; writes run sequentially in announced order, unknown tools
  are sequential. `tests/test_parallel_partition.py`.
- Executor extras: file cache + stale-file detection, context compaction
  (mechanical first, LLM summarization when needed, recent messages preserved).
  (A `[STATE]` block and an 80%/95% budget wind-down were documented here for
  months; the block had no callers and the wind-down was never implemented.
  Both removed 2026-08-05 — don't re-document a feature without a call site.)
- **Guardrails** (core/guardrails.py) — all deterministic except the completion
  audit, which is difficulty-gated so trivial tasks never reach the judge LLM:
  requirement checklist (`confirm_plan`), trajectory monitor, completion audit,
  correction detector (regex → anti-pattern lessons), context pinning,
  doom-loop detector, look-before-review gate (review intent → grounding
  guidance + read-first completion gate; validated live 2026-06-05).
  The three LLM "critique provider" features were deleted 2026-08-05 (all
  defaulted off, never enabled; the field moved away from stacked judges).
- **Guardrails attach to BOTH paths** as of 2026-08-05. Chat gets the
  deterministic set with `trajectory_monitor` overridden off (its synthetic
  checkpoint injection is for long unsupervised runs; in chat the user's next
  message is the correction). Completion audit + review gate need a completion
  tool, so they remain executor-only.
- Plan mode: LLM self-restricts to read-only tools via enter/exit_plan_mode.
- **Tool failure is a TYPE, not a wording** (2026-09-06): tools return
  `ToolFailure("...")` (a `str` subclass, `tools/base.py`); the chokepoint
  `execute_tool_call` sets `success=False` on the type. The two-prefix string
  match (`Error:` / `Error executing`) is only a backstop — "Search error:",
  "Fetch error:", "Workflow 'x' failed:" and "saved in memory but failed to
  persist" all counted as SUCCESS under it. Not an exception on purpose:
  ~100 direct `tool.execute()` callers read the string. Wrap at the point of
  return; an f-string over a ToolFailure yields a plain str.
- **Guardrail internals fail OPEN, and say so at WARNING** (2026-09-06). The
  doom-loop, review-gate, pause and trajectory checks used to log their own
  exceptions at DEBUG, so a guardrail that threw on every turn was invisible.
  The fail-open decision stands (a broken check must not block the turn);
  only the log level changed. `test_broken_guardrail_is_fail_open_but_logged_at_warning`.

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
  **The dedup verdict is parsed by a strict grammar** (`memory/dedup_decision.py`,
  2026-09-08, V3 A1): the reply must BE a verdict (first/last line, last
  sentence segment), UPDATE/DELETE need an explicit in-range 1-based index,
  and anything else is RETRY → re-ask once → keep the new memory, archive
  nothing, stamp `dedup_undecided` on it. The old parser matched action
  SUBSTRINGS and defaulted a missing index to item 0, so "Do not DELETE
  anything; ADD this as distinct." archived candidate #1 — the model being
  careful was the trigger. Every archive stamps the archived row's metadata
  `dedup = {action, by, candidates, reply, at}`; `blipshell repair
  --unarchive-memory ID` explains and reverses it (re-embeds, keeps history).
  `memory.dedup.structured_output` (default OFF) asks for a schema-constrained
  JSON verdict instead; it is an EXPERIMENT until benchmark job
  `dedup_structured` shows `valid_rate` ≥ ~0.98 on the model that actually
  serves TaskType.REASONING — per config.yaml that is the `local` endpoint
  only, qwen3:14b (fallback gpt-oss:latest), NOT the chat model — called the
  way production calls it: `think=False` (processor.py `_ask_dedup_verdict`;
  the benchmark job matches). Two cautions: qwen3 degrades with think=False
  (see Conventions), so a low valid_rate may be the think flag, not the
  schema — measure a think=True variant before concluding; and local models
  misbehave under schema constraints in thinking modes, which is why this is
  measured and not assumed. **Measured 2026-09-09 on qwen3:14b (3 live
  repeats, 144 calls): valid_rate 1.000 on BOTH paths; accuracy 0.750 on
  both at think=False with identical, safe-direction (ADD) misses; JSON costs
  ~5x latency. Structured output stays OFF: it clears the gate and buys
  nothing.** Numbers and reading in V3_PLAN A1.
  `router.generate(response_format=...)` forwards
  the schema as Ollama `format`; the OpenAI-compat client drops it, so
  validation is the contract, not the constraint.
- **Search** (memory/search.py): FTS5 + vec0 KNN fused with RRF (k=60), then
  boosts — importance, FadeMem recency (importance slows decay, each access resets
  effective age), tag overlap, active-project (+0.5), entity-graph expansion —
  optional reranker blend, Jaccard dedup of results. Retrieval declared
  good-enough for v2; do NOT tune preemptively.
  **Time-aware (2026-09-02)**: queries naming a time range ("yesterday",
  "last week", "in July") rank in-range memories first — deterministic regex
  parse (memory/timeparse.py, no LLM on the per-turn path), partition-prefer
  not hard-filter, so a wrong or empty range degrades to the old ranking.
  `memory.time_aware_search` toggles it. Evidence: temporal is every memory
  system's worst benchmark category and time-range filtering its best-attested
  fix (FIELD_SURVEY_2026_09.md 3.1).
- **Context assembly** (memory/manager.py + agent_chat `_build_messages`):
  4 item pools — core 5%, lessons 5%, recent_history 20%, recall 40% — plus
  the conversation window (the active_session 30% share, spent on ROLE
  MESSAGES, never pool items). **Since 2026-09-09 (V3 B1-B3)** the
  conversation appears exactly once (it used to be mirrored into the system
  message AND appended as role messages); turns that fall out of the window
  are summarised into RecentHistory once (`history_summarized_upto`); the
  request is budgeted as a whole (measured prefix + tool schemas + reply
  reserve, then pools); Recall packs first and RecentHistory skips memory
  ids Recall already sent; packing SKIPS an oversized item instead of
  stopping, recording the reason; recalled memories render a QUERY-RELEVANT
  excerpt (`memory/excerpt.py`) with their speaker instead of the first
  1,200 chars; and the trace has retrieved / sent / omitted stages, which
  `/why` prints. The continuity set is the gate for all of it.
  **Provenance on derived layers (V3 B4):** `core_memories` and `lessons`
  carry `source_type` (user_statement | assistant_inference |
  tool_observation | reflection | import | unknown) and
  `verification_state` (stated | inferred | verified | contradicted |
  unknown); vocabulary + defaults in `models/memory.py`. Every creation site
  stamps them; `promote_to_core_memory` follows the source memory's role.
  Inferred items render with `[inferred] ` in Core, Lessons and Recall; pre-B4
  rows are `unknown` and get no invented label. Pass `source_type` when you
  add a creation site - the default on `process_core_memory` is
  `assistant_inference` because a model-initiated save is the model's call
  even when it quotes the user.
- **Supersession is a record, not an archive** (2026-09-09, V3 E1,
  `memory/supersession.py`, table `supersessions`): when the dedup verdict
  says DELETE/UPDATE or the core contradiction check says YES, the OLD row
  stays (un-archived, vector intact) and a row records old -> new with
  scope, relation, detector, evidence and the new record's provenance.
  Search drops superseded memories for current-state questions and shows
  them labelled `[superseded <date> by memory N]` for historical ones
  (`is_historical_question`, deterministic regex); RecentHistory never
  shows them. **Scope**: dedup drops candidates from a different project
  before asking the verdict, so a correction in one project cannot
  supersede a look-alike fact in another. `undo` reverses, never deletes.
  **Decisions** (`memory/decisions.py`, tools record/revise/reopen/list)
  are memory rows of type `decision` with `DECISION/BECAUSE/REVISIT WHEN`
  content; revising writes a `revises` supersession, reopening undoes it.
  `memory/noise.py` used to drop sub-80-char messages without a signal word,
  so a short correction never reached memory; `CORRECTION_PATTERN` (narrow,
  word-bounded) now counts as signal. Keep it narrow - it is tested for
  negatives too.
- **Correction attribution is RECORD ONLY** (2026-09-09, V3 D2a phase 1,
  `memory/attribution.py`): `lesson_uses` logs which lessons reached each
  request; an accepted correction becomes a `corrections` row with the
  lessons present on the corrected turn, and a background LOCAL judge stores
  an attribution (lesson_wrong | lesson_ignored | lesson_misapplied |
  unrelated | unattributed, strict parse, confidence floor 0.7). Two
  toggles (`config.yaml` `attribution:`): `enabled` (default true) is the
  logging - no model call, no behaviour change; `judge_enabled` (default
  FALSE) adds the background local judge call per correction. Ship with
  the defaults; enable the judge only once a labelled set exists to
  evaluate it. **No code
  reads these to change a lesson**, and none may until the hand-labelled
  evaluation (`python -m scripts.attribution_readout`, >= 10-15 genuine
  lesson_wrong positives, agreement >= 0.8, FP <= 0.1) passes and the user
  approves phase 2. Do not add that authority as a side effect of anything.
  The evaluation itself is `python -m scripts.attribution_eval` (build /
  label / freeze / run, `memory/attribution_eval.py`): the set is frozen
  against the CURRENT lesson-selection behaviour and tagged by generation
  (`pre-D1`); runs record a judge hash and the first run is the baseline;
  a changed judge is a new version, a post-D1 set is a new generation, and
  the two are never merged. **D1 (per-turn lesson selection) waits for the
  pre-D1 baseline.** Set texts live in `data/attribution_eval/` (gitignored).
- **Entity graph** (memory/entity_extractor.py): LLM triple extraction; 4-stage
  resolution — alias routing (merged names → canonical, follows chains) → exact
  match, typed on `(name, entity_type)` → embedding (≥0.85 auto-merge,
  0.70–0.85 LLM arbitration) → create.
  Bi-temporal edges; `CONTRADICTING_PREDICATES` expires stale facts.
  **USER MANDATE: ARCHIVE, never DELETE** — merge/prune soft-archive only.
  Revive-on-re-mention un-archives pruned entities (merged husks excluded).
  Name-comparison rules live in `memory/entity_names.py` and are used by BOTH
  merge paths — `version_distinguished` blocks any merge of names differing in
  a version/instance number (`projectecho_v1` vs `_v2` embeds at 0.996, clear
  of every threshold). They were methods on `EntityMerger` alone until
  2026-08-07, i.e. on the path that ships disabled, while creation-time
  resolution — the enabled one — had no guard. If you add a third merge site,
  it uses this module.
  **Archived entities are two populations** (2026-09-06, `entity_names.husk_sql`):
  a HUSK (merged away; name in `entity_aliases`) is dead and must never take a
  mention; a DORMANT entity (pruned; no alias) revives on re-mention and must
  stay a resolution candidate WITH its vector. The June 2026 merge left 7,557
  husk vectors in `vec_entities`; Stage 2 matched them by MEANING (Stage 0
  alias routing only covers same-NAME) and merged 46 new mentions into dead
  entities. Now: `search_similar_entities` filters husks, Stage 2 routes a
  husk candidate to its canonical (`resolve_husk`), the orphan sweep and
  entity backfill agree on the predicate, and `blipshell repair
  --repoint-husks` drains stranded references. A blanket `is_archived = 0`
  filter would have killed revive-by-meaning for 15,218 dormant entities —
  use `husk_sql`, not the flag. Plan + numbers: `docs/HYGIENE_2026_09.md`.
  `get_all_entity_names()` is cached; anything writing entity rows outside the
  store's own methods must call `_invalidate_entity_name_cache()`.
  Failed extractions are left unmarked and counted as `retryable` — never mark
  a failure done, the triples are lost with nothing to find them by.
- Also: consolidation (near-dup merging), centroid + batch taggers, lessons with
  project scoping, project digests (stored in project metadata, auto-updated on
  session close).
- **Project dossier is assembled from records; the digest is one labelled
  section of it** (2026-09-09, V3 E2, `memory/dossier.py` +
  `memory/project_events.py`, table `project_events`). Decisions in force
  with reason and revisit condition, recently superseded decisions, open
  follow-ups oldest first, last completed work ("claimed by assistant, not
  verified" until a `verification` event exists - there is no writer for
  that kind yet, so every completion says claimed), last session, sources.
  The prose digest renders as `[inferred by the assistant from session
  summaries]`. "Next useful action" is the oldest open follow-up or "Ask
  before assuming one" - never invented. Updates are EVENT-DRIVEN: every
  writer (decision tools, follow-up tools, `task_complete`, session close)
  appends an event and marks the render stale. The nightly `rebuild_digests`
  reconciles ACTIVE projects only (event in the last 14 days). Activation
  injects the dossier as its own block, NOT inside the hour-cached repo
  scan; its decision memories are skipped by every pool
  (`MemoryManager.rendered_elsewhere`) and its follow-ups by the OPEN
  FOLLOW-UPS block, so nothing renders twice. Behavioural gate measured
  2026-09-09 on the gpt-oss FALLBACK (5 runs): the records are used (ids,
  reasons, next action, no re-proposal) but every resume reply stated the
  "claimed, not verified" completion as fact - the label does not survive
  into the reply. Production (minimax-m3, 2026-09-09, 5 runs): same, 10/10.
  After the record-layer fix (dossier carries the reporting rule, 2026-09-10,
  frozen batch, 5 runs): 9/10 resume replies report it as unverified, 1
  states it as fact; frozen batch verdict still FAIL on other clauses.
  A deterministic backstop now exists (`core/claim_check.py`, wired in
  `_chat_simple`): an unhedged statement of a dossier claim gets an appended
  `[Unverified: ...]` note; narrow to the dossier's claims, never rewrites.
  Project scope is enforced at SELECTION (2026-09-10): while a project is
  active, RecentHistory items and Recall results belonging to a DIFFERENT
  project are omitted ("other project"); global items stay. Scorer v4;
  continuity scenarios run on a fresh DB each (`SimScenario.fresh_db`).
  V3_PLAN Stage E gate + completion batch + v3 correctness closures.
  **Authorization rule (2026-09-10, `core/turn_kind.py`)**: a turn is a
  question (discuss, change nothing), an instruction (act, disclose the
  overridden decision) or declarative (a stated requirement: update records
  and propose, no file/command mutations unless the executor path's standing
  mandate covers the task). Instrumented - rule appended to the tail on
  declarative turns, `mutation_without_mandate` event when violated - never
  blocked. Scorer v5: "at least one decision in force with its reason".

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
  local calls keep raw text for search quality. **Presidio + spaCy are an
  optional extra since 2026-09-06** (`pip install -e .[pii]`, then
  `python -m spacy download en_core_web_lg`; no spaCy wheel for Python 3.14,
  which is why the dev box runs regex-only and its 3 NER tests SKIP, never
  fail). Regex-only redacts credentials/keys/IPs, NOT names or places. The
  engine is reported at startup (WARNING when regex-only and an endpoint
  relays offsite) and in `/status`. `pii.require_ner: true` keeps
  `router.generate()` traffic — the FULL-sanitize background path — off
  sanitizing endpoints while Presidio is unavailable (falls back to local or
  raises); chat bypasses `generate()` and is governed by `/local`, not this.
  Check what production actually loads on the Ollama PC:
  `python -c "from blipshell.llm.pii import engine_description; print(engine_description())"`.
- Current assignments (see config.yaml `models:` + per-endpoint overrides):

| Task | Primary | Fallback (local) |
|---|---|---|
| tool_calling | deepseek/deepseek-v4-flash (OpenRouter; 0.973 tool_calling post-parser-fix, agent eval 28/30) | gemma4:31b-cloud (Ollama cloud), then gpt-oss:latest |
| coding | minimax/minimax-m3 (OpenRouter) | gpt-oss:latest |
| reasoning / ranking / importance / ranking_importance | qwen3:14b (local) | qwen3:14b / qwen3.5:9b |
| summarization | glm4:latest local; Groq gpt-oss-120b via endpoint priority | qwen3:14b |
| session_review | deepseek/deepseek-v4-flash (OpenRouter, 204K; measured 0.887 review / 0.537 lessons) | qwen3:14b (local, 32K) |
| reflection | gemma4:31b-cloud (Ollama cloud) | qwen3:14b |
| embedding | qwen3-embedding:0.6b | — |

- Gemini endpoint exists but is disabled (free-tier burst limits). Groq serves
  ranking_importance (llama-3.3-70b) + summarization.
- **session_review → `minimax-m3:cloud`, decided on benchmark evidence 2026-08-04.**
  It beats local qwen3:14b on BOTH jobs the key controls: session review 0.944 vs
  0.844, lessons 0.585 vs 0.345. Lessons decided it — qwen3 scored 0.420 then
  0.345 on a re-run, a real weakness, and lessons feed a permanent context pool
  plus the anti-pattern store. `kimi-k2.7-code` was rejected at 0.395 lessons,
  worst of every model measured, despite leading session review at 0.925 — the
  canonical example of why a key must be judged on *all* the jobs it controls.
  Two costs, both live: (1) `pii_sanitize` scrubs names/dates from the transcript
  before the model sees it, and the benchmark feeds UNSANITIZED synthetic
  sessions, so **0.585 is an upper bound** on real lesson quality; (2) same free
  tier that retired kimi-k2.5 with no notice — now bounded, since a 410 is
  classified as a model error and falls back to local without penalising the
  endpoint that also serves interactive chat.
- Chunking still matters, on the **fallback** path: cloud gives session_review a
  128K window so most sessions single-pass, but local qwen3:14b at 32K sends
  anything over ~28K through `prepare_conversation_for_reflection` → per-chunk
  reflection → `merge_chunk_reflections` (N+1 calls). Chunk reflections use a
  chunk-scoped prompt that forbids rating the session overall and forbids
  reporting anything unresolved; without it a fragment gets judged as a whole
  session and invents "never addressed" findings that reach lessons.

## Alive layer (the v2 soul)

- **Idle self-reflection** (core/self_reflection.py + agent.py reflection loop):
  after ~3h quiet, forms one "lingering thought" from its own prior thoughts.
  Surfaces two ways: one-shot pending injection on return, and per-turn semantic
  resurfacing (cosine prefilter → local LLM relevance judge, fail-closed, max 1).
  Routed through `TaskType.REFLECTION` (own config key, 2026-09-01): the task is
  unusually model-sensitive — Wisp measured, varying ONLY the model on one corpus
  and prompt, 3 distinct themes (phi4:14b) vs 14 (gemma4:31b-cloud) across 20
  reflections. A **nightly `self_reflection` job** (core/nightly.py) forms one
  thought per night regardless of app usage — idle + on-return alone measured
  ~1 thought/month, which left the step-2 gate ("10 NEW thoughts") a year away —
  skipping when a thought already formed that day (`reflection.nightly_min_gap_hours`).
- **Theme-diversity metric** (memory/themes.py, ported from Wisp, hand-validated
  there): deterministic content-word-Jaccard clustering over the thought corpus.
  Quote `distinct_themes` and `domination` (largest NO-CHAIN family share — the
  single-link figure inflates ~7x on a paraphrase chain and is reported only for
  comparison). The nightly job emits it every run; `python -m scripts.theme_readout`
  prints it on demand. This is the number that replaces "does resurfacing feel
  like caring vs indexing?" as the step-2 readout.
- **Retrieval provenance** (`python -m scripts.retrieval_provenance`, Ollama PC):
  measures whether the real search pipeline over-selects assistant-authored
  memories ("exhaust") beyond their corpus share — Wisp's most portable finding
  (its percept holding an answer ranked 47th behind 27 of its own replies).
  This is the MEASUREMENT that must precede any retrieval tuning; the retrieval
  good-enough mandate stands until it shows a pathology.
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
- `tests/` (~69 files, 2005 passing + 3 skipped as of 2026-09-08): `tests/fakes.py`
  `ScriptedLLMClient` drives the REAL ChatLoop with canned turns —
  completion detection, guardrails gating, dedup validated deterministically
  (`tests/test_loop_integration.py`). `conftest.py` gives real in-memory SQLite +
  canned router.
- **Conversation continuity across sessions is the OUTSTANDING requirement
  and the next active task** (2026-09-10; V3_PLAN "Outstanding requirement").
  Diagnosis from the corpus: what carried forward was retrospective and
  importance/opening-selected, abnormally ended sessions left nothing, and a
  continuity question went to Recall (which returns old "did you forget"
  memories). Shipped: mid-session refresh of the handoff note
  (`handoff.refresh_every_turns`), a verbatim stop block of the previous
  session's last exchanges at boot (`handoff.stop_block_pairs`), tier 2 takes
  the END of a session, and a `continuity` query profile. The live A/B probe
  in `core/handoff.py` is still the test that counts; the model's own "feels
  better" does not.
- **Continuity set** (2026-09-09, V3 Stage C): `python -m blipshell.benchmark.continuity`
  boots a REAL agent per case with no network (`Agent._build_subsystems`, the
  DB-only half of initialize, + the deterministic embedder in `tests/fakes.py`
  + a recording chat client), plants memories, asks a question and scores
  the REQUEST the model would have been sent: survival (answer text present)
  and exclusion (superseded/speculative/other-project text absent or
  labelled on its line). No model is called; it is the Stage B gate.
  Cases: `tests/benchmark_continuity.py`; instrument tests:
  `tests/test_continuity_set.py -s` prints the table. Baseline 2026-09-09:
  survival 0.833, exclusion 0.429, and every recalled memory rendered twice
  (Recall + RecentHistory); after Stages B + E1: 1.0 / 1.0 / 0 over 16
  cases; after E2: 1.0 / 1.0 / 0 over 17 (cases may set `active_project`
  to compose the real project context, dossier included). `Seed.via="pipeline"`
  drives the REAL write path with only the
  dedup verdict scripted, so supersession records are created by production
  code, never fixture metadata. Result files are `kind: context_delivery` —
  they say what reached the request, not what a model did with it; keep
  behavioural (real-model) results separate. See V3_PLAN Stage C / E1.
- `blipshell simulate` — multi-turn scenarios against a real Agent. Scopes are
  `-s <scenario>` / `-c <category>`; there is NO `-t` tag flag. Runs against a
  throwaway temp DB by default (`--db PATH` to pick one, `--real-db` to use the
  live corpus — it writes real sessions, lessons and digests). Needs a model:
  the Ollama PC, or the dev box over Tailscale (`openai` is installed here).
  **`-c continuity` is the V3 Stage E behavioural gate** (2026-09-09,
  `simulate/scenarios/continuity.py`): `SimScenario.setup` plants a fixed
  14-day-old world (digest, decision in force, superseded decision, open
  follow-up, an unverified `task_completed` claim, another project's facts)
  BEFORE the session starts; each scenario is one chat turn scored by a
  deterministic `response_validator` (soft -> WARN with the named miss:
  "unverified completion presented as fact", "superseded decision presented
  as current", ...). `tests/test_simulate_continuity.py` proves the
  instrument here; the gate is a real-model measurement over several runs,
  never one reply. Run 2026-09-09 x5 from the dev box over Tailscale
  (fallback model; results `benchmark_results/simulate_continuity__*`).
  Results persist BEFORE the console report (`e5c74e7`). Every run that
  day hung at exit: simulate's cleanup called `end_session` but never
  `force_cleanup`, so SQLite stayed open and aiosqlite's NON-daemon
  connection thread kept the interpreter alive. Fixed in the runner; the
  CLI always did both. `tests/test_simulate_cleanup.py` keeps the negative
  control. When you bootstrap an Agent anywhere else, close it the same way.
  **Scorer v2 (2026-09-10)**: `SCORER_VERSION` in `scenarios/continuity.py`
  is stamped into every run; change a rule -> bump it -> publish
  `python -m scripts.rescore_continuity` (originals never rewritten). Steps
  carry an explicit `outcome` (scored | timeout | error | blocked);
  `simulate --require-model X` refuses to start without an endpoint + key
  that can serve X and marks any step served by another model `blocked`.
  `python -m scripts.run_gate_batch` is the predefined production batch;
  its pass/blocked/fail criteria are in its docstring - do not extend or
  rerun it to chase a pass. **Scorer v3 (2026-09-10)**: an explicit
  instruction to change a decision is authorization - the gate scores
  DISCLOSURE of the overridden decision and its reason, not abstention.
- **External review 2026-09-10 reconciled**: six findings, all reproduced
  at HEAD and fixed in `0de62fe` (typed `(record_kind, id)` pool identity;
  `superseded(..., for_project=)` scope-aware reads; event-id cursor for
  the dossier reconcile; reopen = discussion vs `restore=True`; kind-aware
  `undo` for core memories; hard whole-request bound with trim-then-refuse
  via `ContextOverflowError`). Table with evidence in V3_PLAN. The two
  incomplete contracts were then integrated: `blipshell repair
  --supersessions KIND:ID` / `--undo-supersession ID` make the kind-aware
  undo reachable (with the vector store), and the executor's first request
  goes through `core/request_bound.py` - same trim order as chat, same
  refusal. Historical core-memory retrieval is still undefined (repair
  listing only).
- `blipshell benchmark run <model>` — ONE deep test across all 9 job types →
  `data/benchmark/report.md` (numbers only, no verdict). Ground-truth scorers are
  unit-testable here; real runs need the Ollama PC. Judge = OpenRouter
  (claude-opus-4.8), graceful-fail.
- **Results are COMMITTED files in `benchmark_results/`** (one JSON per run),
  not the gitignored `benchmark.db` — that DB now holds only the refetchable
  discovery catalog. Rationale: a gitignored store can't sync across the two-PC
  split, so the comparison corpus never accumulated and reports silently omitted
  models the other machine had measured (the 2026-06-24 report compared four
  cloud models and left out `qwen3:14b`, which serves half the jobs). One file
  per run means two machines never conflict — merging results is `git pull`.
  **Commit the result file after a run**, or the other box never sees it.
  Each run records `git_sha` + `host`; the report's Provenance table shows them,
  because scores from different commits are not strictly comparable.
  `python -m scripts.migrate_benchmark_results` moves pre-2026-08-03 rows out of
  an old DB (run once, on the machine holding the history).
- **Read `report.md`'s first section, "Which model to use where"** — one block per
  `config.yaml` key, listing every job that key controls, with a verdict
  (KEEP / CONSIDER / UNKNOWN) and the exact command to resolve an UNKNOWN. Also
  echoed to the console. `benchmark/advice.py` owns it; `JOB_OWNERS` there maps
  key → jobs and is derived from real `TaskType` call sites, NOT from
  `config.yaml`'s comments (which have drifted — the `reasoning:` comment claims
  it handles lessons; lessons go through SESSION_REVIEW at `processor.py:275`).
  **Under-specifying a key produces confidently wrong advice**: with only its
  namesake job attached, `tool_calling` recommended lfm2.5 (0.933 tool calling,
  0.450 reasoning) for interactive chat. When routing changes, re-verify JOB_OWNERS.
- Job/prompt correspondence (audited 2026-08-03) — `summarization`, `lessons`,
  `contradiction`, `entity`, `rank_importance` measure the prompts production
  actually runs. `ranking` and `importance` measure **import-path-only** prompts
  and are labelled as such. `coding` in runs before 2026-08-03 conflated judged
  code-gen with agentic pass rate under one `task_type`; it renders as
  "Coding (legacy - ambiguous)" and is excluded from the composite. New runs
  report `code_gen` and `coding_agentic` separately.
- ~~Known benchmark gap: `run_session_review` never exercises chunk+merge~~ —
  CLOSED 2026-08-18. `SESSION_REVIEW_CHUNKED_CASES` runs the real production
  sequence (chunk-scoped `reflect_on_session(part=(i,n))` per chunk ->
  `merge_chunk_reflections`) and scores the specific failure that path exists
  to prevent: a problem raised in part 1 and FIXED in part 3 must not be
  reported as "never addressed", because that invented finding flows into
  lessons, a permanent context pool. New job `session_review_chunked`; owned by
  `models.session_review` in JOB_OWNERS. The single-chunk cases stay: they are
  111-190 tokens and always took the single-chunk branch, which is why this
  went unmeasured for months.
  Scored as mean(coverage, fidelity), both in `raw`. **Measured 2026-08-18 —
  do not overread it**: coverage discriminates (minimax-m3 1.00 at 181 words
  vs gemma4 0.80 at 89), but it correlates with verbosity; fidelity does NOT
  discriminate and is unvalidated live — re-running with the chunk-scoped
  prompt REMOVED changed nothing (0.900/0.800/1.000 either way, no invented
  "never addressed" finding in any per-chunk reflection). Fidelity is a cheap
  regression guard with unit-level teeth only: a drop is a real signal, a 1.0
  means nothing. Hardening it needs a longer transcript or a chunk boundary
  that splits mid-argument.
- Benchmark faithfulness (audited 2026-08-18, suite -> production call site).
  **Faithful**: `rank_importance` (processor.py:188), `contradiction`
  (processor.py:362), `entity` (entity_extractor.py:148), `summarization`
  (processor.py:101), `lessons` (processor.py:279). **Import-path-only and
  labelled**: `ranking` (import_common.py:598), `importance` (:648).
  `JOB_OWNERS` re-derived from `TaskType.*` and confirmed correct.
  Two defects found and FIXED:
  - `tool_calling` sent a bare user turn with NO system prompt, while
    production always sends one. It also scored a single turn, so an agentic
    opener (`find . -name worker.py`, `list_directory .`, read-before-edit)
    counted as a miss. **That mis-scoring WAS the metric's noise**: repeats
    gave sd ~0.09 / spread 0.13-0.20, so minimax-m3's recorded 0.667 vs
    qwen3:14b's 0.933 was substantially a coin flip. Now a real multi-turn
    loop (`TOOL_CALLING_MAX_TURNS`) against a fake workspace with the
    production system prompt: measured 0.9333 with spread 0.000 over 3
    repeats. Command args match on INVOCATION, not substring — `find . -name
    "pytest.ini"` used to count as "ran pytest".
  - The `reasoning` suite graded `generate_plan`, which has ZERO production
    callers (only the benchmark imported it; DESIGN.md:588 still documents it
    as live). Replaced with a `structured_compaction_prompt` case — live at
    chat_loop.py:520 through TaskType.REASONING.
  Advice is now noise-aware: `MEANINGFUL_DELTA` (0.03) is only the fallback for
  unreplicated runs; with repeats, a gain must clear the MEASURED spread
  (`advice._noise_floor`). `--repeats` defaults to 5 — a single run cannot
  separate two close models and will invert rankings.
- `python scripts/run_executor.py --canned|--stress` — headless executor harness
  (Ollama PC).

## Nightly maintenance

`NightlyRunner` (core/nightly.py): 25 ordered jobs (backup → cleanup →
backfills → session/self reflection → entity extraction/merge/prune → taggers →
consolidate → digests → user model → memory mirror → health check), each
isolated, 300s/job timeout, Ollama-dependent jobs skipped when it's down.
`export_mirror` writes the believed-state layers (user model, core memories,
lessons, self-thoughts) to `data/mirror/*.md` — EXPORT-ONLY transparency
(hand-edits are not read back; regenerated wholesale), gitignored because it
is the distilled personal layer. `/nightly` command or `blipshell nightly --quiet`.
**`batch_tag` progress is monotonic since 2026-09-08**: every memory a batch
examines leaves the pool, either with >1 tags or with the `_skip` marker
(`sqlite_store.BATCH_TAG_SKIP_MARKER`, excluded from the pool by NAME — counted
as a tag it never lifted a memory out of a "<=1" pool). Before that the
pool was re-read newest-first with no cursor and the marker was gated on
`allow_new_tags` (never on in the nightly), so the Sep 2 run re-sent the same
ten memories until the 270s budget ran out: 11 touched, 17,080 in the pool.
"Stopped early" now carries `remaining_pool` + `est_hours_to_drain`; the job
returns `checked`, which is what `blipshell nightly --job batch_tag --loop`
keys on, so that command is the drain (ask before tying up the GPU).
Junk vocabulary (`nnone`, from the model writing NONE) is purged each run.
**Commit evidence for the user model is a durable queue** (2026-09-09, V3 A4,
`memory/commit_ingest.py`, table `commit_evidence` unique on repo+sha):
acquisition cursor (newest acquired epoch, no +1) is separate from
consumption; rows are drained oldest-first, 10 per project per night, and
marked consumed only after `revise_from_reflections` persisted the doc (or
honestly concluded nothing). Before this, `--max-count=10` newest-first plus a
watermark past the newest collected commit dropped anything older forever,
and the watermark was stamped before the model ran. `update_user_model`
stats show `commits_pending`; a growing number means revisions keep failing.
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
- **Relative paths in config.yaml resolve against the CONFIG FILE, never the
  cwd** — use `resolve_config_relative()` (`core/config.py`). `blipshell` is an
  installed console script, so the cwd is wherever the user is standing while
  the config is always found relative to the install; two anchors for one
  setting means the config loads reliably and then names a file that moves.
  Cost of learning this: 2026-08-11..08-20 the live instance was launched one
  directory deep, `data/blipshell.db` resolved under `<repo>/blipshell/`,
  SQLite CREATED a 16MB database and 9 days of sessions went there while the
  real 491MB corpus sat untouched. Nothing raised — **an absent SQLite file is
  a creation, not a failure** — so the only symptom was an assistant that had
  quietly lost its history. `database.path` is anchored at the `ConfigManager`
  chokepoint (1b2286e); absolute paths pass through, which is what keeps
  `simulate --db` working. A `dir /s /b` sweep found SEVEN live-shaped copies,
  and `data/data/` + `blipshell/blipshell/data/` can only come from a wrong-cwd
  launch — this had recurred for months. `benchmark/` had already solved it
  locally and that duplicate is exactly why the main path stayed broken: when
  you fix a path bug, fix it at the chokepoint and delete the copies.
  Since 2026-09-01 `database.require_existing: true` (production config)
  additionally REFUSES to load when no file exists at the resolved path —
  anchoring fixed the known cause, the guard catches the class ("an absent
  SQLite file is a creation, not a failure"). Fresh installs set it false;
  post-load overrides (`simulate --db`, benchmark temp DBs) are untouched.
- **Every unguarded third-party import must be a declared dependency**
  (2026-09-09, V3 A6): `tests/test_declared_dependencies.py` walks the
  package with `ast` and maps imports to distributions, because the dev box
  never sees a clean install (numpy was undeclared for months; the external
  review's fresh venv failed collection in five files). Optional extras go
  inside `try/except` - that is what the walker treats as guarded. The
  scratchpad read is anchored through `resolve_config_relative` like the
  database (`tests/test_scratchpad_anchor.py`). Known and exempted:
  `nightly.py` imports repo-root `scripts.*` modules, which only resolve
  under an editable install.
- **A test that compares the config VALUE instead of the resolved TARGET is
  vacuous.** `test_same_path_from_any_working_directory` passed with anchoring
  disabled, because unanchored the string stays `"data/blipshell.db"` from
  either directory — matching strings, different files. Mutation testing was
  the only reason this was caught, and one mutation was not enough: the
  `save()` guard needed its own, because the first tripped a shared
  precondition instead of the behaviour.

## Known open items (as of 2026-08-05)

See `docs/V2_PLAN.md` for the full phased plan; Phases 0 and 1 are done, Phase 2
is in progress. Remaining:

- `cli.py`'s terminal plumbing (~440 lines: Windows VT/msvcrt, Esc-cancel,
  approval prompt) is the last thing left in that file that isn't Click
  wiring. Deliberately not extracted — the suite can't validate it, so a
  regression would only surface interactively.
- Shared context assembly: `!plan` silently drops scratchpad, session notes,
  follow-ups and the 5-pool budgeting that `_chat_simple` gets.
- ~~MemoryWorker tests~~ — DONE, twice, by accident. 17 dispatch/threading
  tests with a faked processor landed 2026-08-05 (`ecf8401`) while this line
  still said "ZERO tests"; a second session trusted the line and rebuilt them
  on 2026-08-07 before checking `tests/` (recovered from git, merged as
  `test_memory_worker_pipeline.py` — real pipeline, canned router). The two
  files are deliberate layers now. The meta-lesson is this bullet itself:
  **stale docs cause duplicate work — grep `tests/` before building tests.**
- Local fallback models are a generation behind — benchmark newer candidates on
  the Ollama PC before swapping (test-first, always).
- ~~Consolidation throughput~~ — REWRITTEN 2026-08-06/07. Neighbours come from
  stored vectors (`VectorStore.find_neighbors`, zero embedding calls), the
  batch is time-budgeted rather than fixed at 20, the scan gets only
  `SCAN_BUDGET_SHARE` of that budget so it can't starve the merge phase,
  losers are ARCHIVED, and `--loop` advances a persisted cursor in dry-run.
  Each pass self-verifies against `get_integrity_counts()` and prints
  `integrity_ok`. Five separate bugs surfaced only in live runs — treat this
  module as the one where the dev-box suite is least predictive.
  **Full sweep done 2026-08-07**: all 31,977 active memories checked, 59
  archived (0.18%), no orphaned edges/mentions. That yield is expected, not
  broken — write-time dedup (`processor.py`, LLM-arbitrated at 0.7) already
  removes the real duplicates, so consolidation is a narrow net behind a wider
  one. **Do not lower `consolidation_similarity_threshold` from 0.92**:
  `scripts/consolidation_calibrate.py` measured 0% of the corpus above 0.96,
  and the 0.88–0.92 band is ~60% same-topic-different-fact (a `llama3.2`
  payload vs the identical `gemma3` one scores 0.9198). Re-run that script
  before ever revisiting the number; don't argue it from a merge count.
- Memory reranker as L2 — NOTE: enabling it as-written would *degrade* ranking
  (normalization is applied only to the top-N, and `logprobs` is never
  requested). Fix before any future enable.
- Self-gravity step 2 (graph-relational) — gated on a clean step-1 readout,
  which needs ~10 NEW thoughts now that the fatigue/eviction bugs are fixed.
- Identity privacy on cloud chat: credentials are stripped, but the recall pool
  still ships matched personal memories to cloud endpoints every turn. The fix
  is routing (a local mode), not more redaction. See V2_PLAN D1.
