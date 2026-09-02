# CLAUDE.md — Project Context for BlipShell

Current architecture, verified against code 2026-08-05. The old feature-by-feature
build log lives in `docs/HISTORY.md`; the active plan is `docs/V2_PLAN.md`
(2026-08-04, supersedes the sequencing in `docs/SYSTEM_REVIEW.md`).
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
  (`http://[tailscale-ip]:11434`, when that PC is on): model-touching work —
  `benchmark run --url`, simulate (`openai` installed here), live prompt
  validation — can now be DRIVEN from here, executing on the real GPU. The
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
├── simulate/        # multi-turn scenario runner (37 scenarios, 7 categories;
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
  **Time-aware (2026-09-02)**: queries naming a time range ("yesterday",
  "last week", "in July") rank in-range memories first — deterministic regex
  parse (memory/timeparse.py, no LLM on the per-turn path), partition-prefer
  not hard-filter, so a wrong or empty range degrades to the old ranking.
  `memory.time_aware_search` toggles it. Evidence: temporal is every memory
  system's worst benchmark category and time-range filtering its best-attested
  fix (FIELD_SURVEY_2026_09.md 3.1).
- **Context assembly** (memory/manager.py): 5 token-budget pools — core 5%,
  lessons 5%, active_session 30%, recent_history 20%, recall 40% — with rollover.
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
  `get_all_entity_names()` is cached; anything writing entity rows outside the
  store's own methods must call `_invalidate_entity_name_cache()`.
  Failed extractions are left unmarked and counted as `retryable` — never mark
  a failure done, the triples are lost with nothing to find them by.
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
- `tests/` (~69 files, 1620 passing + 3 skipped): `tests/fakes.py`
  `ScriptedLLMClient` drives the REAL ChatLoop with canned turns —
  completion detection, guardrails gating, dedup validated deterministically
  (`tests/test_loop_integration.py`). `conftest.py` gives real in-memory SQLite +
  canned router.
- `blipshell simulate` — multi-turn scenarios against a real Agent. Scopes are
  `-s <scenario>` / `-c <category>`; there is NO `-t` tag flag. Runs against a
  throwaway temp DB by default (`--db PATH` to pick one, `--real-db` to use the
  live corpus — it writes real sessions, lessons and digests). Needs the Ollama
  PC: agent bootstrap requires the `openai` package.
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
  chokepoint (d1e6ebf); absolute paths pass through, which is what keeps
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
  tests with a faked processor landed 2026-08-05 (`eb6a2e4`) while this line
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
