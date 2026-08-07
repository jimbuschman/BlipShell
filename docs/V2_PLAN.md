# BlipShell v2 Plan — Trust First, Then Alive

_Date: 2026-08-04. Synthesis of a five-subsystem deep-dive audit (core loop, memory,
LLM layer, UI/session/alive layer, testing/nightly), the rewrite question, the coding
direction question, and an external borrow-ideas research pass. Supersedes the
sequencing in `docs/SYSTEM_REVIEW.md` (2026-06-08) — that doc's findings are folded in
here; its research citations remain valid._

**File:line references were verified against main on 2026-08-04.** Re-check before
fixing — code moves.

---

## The verdict that shapes everything

- **No rewrite.** The architecture is sound (unified ChatLoop, duck-typed clients,
  single-DB memory, the weight firewall). The bugs are wiring, not walls, and the
  codebase's real asset is encoded scar tissue a rewrite would discard.
- **The systemic disease is "one good core, many bypasses":** interactive chat
  bypasses `router.generate()` (PII, accounting, policy); `_chat_simple` bypasses
  guardrails; `execute_plan` bypasses `chat_loop_runner`; simulate duplicates the CLI
  dispatcher by hand; the web UI bypasses a command layer that doesn't exist yet.
  The cure is chokepoints, not rebuilds.
- **The second disease is silent failure:** tools report errors as strings the
  registry counts as success; the nightly report says "all clean" through timeouts,
  skips, and failed backups; `end_session` swallows every step failure; nothing in
  `cli.py` ever sets an exit code.
- **Coding direction:** keep agency (tools/loop) unconditionally; keep coding
  capability (project mode, `!plan`, git) but **frozen**; abandon coding-agent
  *ambition* (parity features, code-quality judges — most already dead code).
  The differentiated move is BlipShell as the **memory layer over your coding life**,
  not a competitor to the tools that write the code.
- **v2 = a direction, not a codebase.** The phases below are strangler-style: each
  lands independently, the suite stays green throughout.

## Standing decisions honored throughout

- ARCHIVE, never DELETE (entities — and now enforced at the *edge* level too).
- Retrieval is declared good-enough; nothing here tunes ranking. Correctness fixes
  (archived leaking into results) are bugs, not tuning.
- Test before commit, full suite, no exceptions. Logic/wiring validates here;
  model behavior validates on the Ollama PC only.
- No quick fixes; trace the full call chain first.

## Open decisions (user calls, each blocks one item only)

| # | Decision | Options | Blocks |
|---|---|---|---|
| D1 | ~~PII on interactive chat~~ **RESOLVED 2026-08-05** | Split by category, not by all-or-nothing: interactive chat strips **credentials only**; background calls keep full sanitization. Identity protection moves to the routing layer. | — |
| D2 | ~~MCP: commit or archive~~ **RESOLVED 2026-08-05** | Archived (removed). Rationale: anything BlipShell actually needs can be coded directly; MCP is integration breadth, which isn't the moat. Reversible — `git checkout 063d06f -- blipshell/mcp` brings it back. | — |
| D3 | Coding freeze confirmation | watch 2 weeks of real usage; if project mode earns its keep weekly, keep layer 2 "warm" instead of frozen | Phase 2 deletions of anything user-facing |

### D1 resolution + what it leaves open

Sanitizing everything on the interactive path was rejected on evidence: spaCy
NER tags product names as entities (Groq, Ollama, Presidio, Devstral become
`[PERSON]`/`[LOCATION]`), so a technical conversation would lose its technical
nouns; it leaks anyway (a username sits in every `C:\Users\...` path, plus
nicknames, handles, names in identifiers); and full NER over the whole
assembled context every turn is too slow for the user-facing path. That is the
worst quadrant — full comprehension cost for partial protection.

Instead the two categories are separated, because their cost profiles are
opposite. **Credentials** (keys, tokens, JWTs, private keys) cost the model
nothing to redact and are the one category with immediate concrete harm →
always stripped, on every cloud path including chat. **Identity** (names,
dates, locations, URLs) stays intact on the interactive path.

**Still open — the real privacy control is routing, not redaction.** Usage is a
genuine mix of personal and technical, and the leak is structural: the recall
pool is 40% of budget and is filled by semantic search over all 17K memories on
*every* turn, so a purely technical question still ships whatever personal
memories matched. Per-message classification cannot fix that. Two candidates,
neither built:

1. **A deliberate local mode** (`/local` … `/cloud`, or local-by-default for
   plain chat with explicit escalation). Cheap, comprehensible, and the honest
   version of "this never left the machine". Favourable asymmetry: personal
   continuity chat is model-light, while the model-hungry work (agentic coding)
   is the less sensitive category — so the split costs little.
2. **Pool-level gating** — when the active endpoint is cloud, restrict which
   memories are injected (project-scoped only, or filtered by tag/type). Most
   aligned with the memory-first architecture, but needs a personal/technical
   signal on memories, so it's a build rather than a tweak.

Worth checking before deciding how much this matters: the current
data-retention / training-use terms of the OpenRouter and Ollama-cloud free
tiers.

## Progress

- **Phase 0 — DONE 2026-08-05** (`79c7384`, `bb10e81`). Suite 1021 → 1200.
  Bonus finds: two test files were silently skipped on Python 3.14 by stale
  ChromaDB guards (re-enabled); the FTS `fts_match` flag only landed on
  FTS-only hits, so keyword protection vanished when vector KNN also returned
  the id with low similarity (fixed); `dump_to_memory` now flushes pending
  persists first (a Phase 1 item, pulled forward by a re-enabled test).
- **Phase 1 — DONE 2026-08-05** (`c9cd279`, `71685a1`, `66f4f2e`, `5f24532`,
  `9d7c8f9`, `46525cc`, `e6bcc07`). Suite 1200 → 1270. All eight items landed
  except the D1-gated half of 1.4 (interactive-chat sanitization), which is
  waiting on the decision above; every PII fix that is correct regardless of
  D1 shipped. Tracing the `end_session` swallow found it repeated at two more
  layers (`_extract_lessons`, `processor.process_lesson`), and the router's
  rate-limit test turned out to be asserting the bug.

---

## Phase 0 — Memory correctness (the moat has leaks)

_All dev-box testable. Mostly one-liners. Each fix gets a pinning regression test._

1. **Dedup-archive zombie loop**: set `is_processed=1` when dedup archives
   (`processor.py:159-167`). Kills a permanent, growing LLM workload (startup sweep +
   nightly re-process the same memories forever).
2. **Archived memories still searchable**: add `is_archived = 0` filtering to
   `search_fts` (join `memories`), the vector enrichment SELECT
   (`vector_store.py:434-438`), and a guard in the scoring loop (`search.py:~281`).
   FTS triggers don't fire on archive-flag updates, so this never self-heals.
3. **`lastrowid` after `INSERT OR IGNORE`** returns a bogus id on duplicate triples
   (`sqlite_store.py:2330-2340`, `:2596-2607`) → corrupted `expired_by`, wrong
   self-exclusion in edge invalidation. Fix: `rid if cursor.rowcount == 1 else None`.
4. **Bi-temporal re-assertion blocked**: `UNIQUE(subject_id, predicate, object_id)`
   makes an expired fact unrepeatable ("Jim rejoined Acme" is impossible). Replace
   with a partial unique index `WHERE expired_at IS NULL`.
5. **Session-reflection embeddings unretrievable**: written to `vec_lessons` at
   `rowid = reflection_id + 100000` (`processor.py:608`) which matches no lessons row
   at enrichment. Give reflections their own vec table (or fix enrichment); also
   removes the future id-collision at lessons > 100000.
6. **Self-thought eviction kills the heaviest thought**: `_save` keeps `items[-50:]`
   by recency (`self_reflection.py:171`) while recurrence updates in place — the
   most-recurred (earliest) thought is evicted first. Evict by lowest *effective
   weight* instead.
7. **Monoculture anchor**: `diverse_recent` seeds with `items[-1]`
   (`self_reflection.py:347`), guaranteeing the dominant theme seeds every
   reflection. Seed with the highest-weight thought *outside* the dominant cluster,
   or randomly.

**Exit criteria:** suite green + a one-off count on the Ollama PC DB: unprocessed
backlog stops regrowing; archived rows absent from search results.

---

## Phase 1 — Honest failure reporting + safety alignment

1. **Tool error convention**: `execute_tool_call` treats an `Error:`-prefixed return
   as `success=False` (`tools/base.py:213-219`) — one chokepoint fix covering 44
   return sites. Stops failed writes poisoning the executor file cache and
   satisfying completion checks. (Longer term: `ToolError` exception.)
2. **Nightly tells the truth**:
   - Add `tag_discovery` + `consolidate` to `_OLLAMA_JOBS` (`nightly.py:54-61`).
   - `_build_and_store_report` collects `timeout`, non-config `skipped`,
     `stopped_early`, `timed_out`, and `warning` — not just `status == "error"`
     (`nightly.py:1116-1150`). Add a `job_statuses` counts block.
   - Distinguish intentional skips (config-gated) from unhealthy skips (Ollama down).
   - Staleness warning in `_check_nightly_report` when the last run is > ~36h old.
   - `blipshell nightly` exits nonzero on any error/timeout; same for `simulate`
     on FAIL (there is currently **no** `sys.exit` anywhere in `cli.py`).
3. **`end_session` honesty**: the `_status` callback carries outcome per step, not
   just intent; fix the `dump_to_memory` docstring that falsely claims per-message
   timeouts (`manager.py:196-198`); make `end_session` idempotent (web/telegram call
   it twice today). Fix the `dump_to_memory`-without-flush duplicate window
   (`agent_project.py:49`).
4. **PII (per D1)** — regardless of the decision:
   - Fix the Presidio offset bug (API-key regex mutates length before Presidio
     offsets apply — `pii.py:76-104`).
   - session_review quality recovery: numbered session-stable pseudonyms
     (`[PERSON_1]`…), `DATE_TIME → [DATE]`, stop scrubbing URLs for this role (or a
     per-role sanitize override). The scrubbing plausibly consumes most of the
     measured 0.585-vs-0.345 lesson margin — re-judge the cloud routing after this.
   - Expand key patterns (gh* variants, `sk-ant-`, `sk-or-v1-`, JWTs, Bearer).
5. **Guardrails on the path with traffic**: attach a `GuardrailsEngine` to
   `_chat_simple`'s LoopConfig (`agent_chat.py:608-621`) — activates doom-loop,
   trajectory monitor, and the look-before-review gate for normal chat (~6 lines).
   Completion audit stays executor-only (needs `completion_tool`).
6. **Simulate isolation**: `--db`/temp-DB switch in `SimRunner` — a full run
   currently writes ~38 real sessions, real lessons, and real digest mutations into
   the production corpus.
7. **Router accounting**: register fallback requests on the fallback endpoint
   (`router.py:275` — currently invisible to rate limits and concurrency caps);
   propagate `min_context_tokens` to fallback selection (a 100K session_review can
   silently truncate at 32K); stop blacklisting the *model* globally on rate-limit
   errors (`exceptions.py:71`, `router.py:241-246`).
8. **Alive-layer metering fixes**: charge fatigue at context-render, not pool-insert
   (`agent_chat.py:938-949` — budget-evicted thoughts currently pay ×0.6 for
   surfacings that never happened); persist `echo_count` per thought and show it in
   `/thoughts` (the recurrence *rate* is the actual gravity signal and is currently
   invisible). Consider a reserved slot for the surfaced thought instead of letting
   it consume Recall-pool budget (the firewall's one token-level leak).

**Exit criteria:** a deliberately-broken nightly (Ollama stopped) reports itself as
broken at next startup; a failed `write_file` shows red in the loop and does not
enter the file cache; simulate leaves the production DB untouched.

---

## Phase 2 — Chokepoints and deletions (the coding-freeze dividend)

1. **Guardrails consolidation** (SYSTEM_REVIEW item B, now verified): delete
   `critique_edit`, `critique_trajectory`, `critique_completion` and their prompts,
   loop blocks, config fields, and tests — **~460 lines removed, ~5 added, zero
   production behavior change** (all three default off; `config.yaml` has no
   guardrails block). The difficulty-gated audit already shipped and is the only
   judge on the live path. Watch: the composite "full" A/B scenario in
   `tests/benchmark_coding.py`, and the numbered section markers in `guardrails.py`.
2. **Dead code sweep**: `core/tool_rules.py`, `TaskPlanner` (`core/planner.py` —
   note `/plan`//`/plans` in the sim dispatcher reference attributes that don't
   exist), `llm/job_queue.py` (background task that does nothing, started every
   boot), `_build_state_block` + wind-down flags, the stale `[STATE]` instruction in
   `read_file`'s description (`filesystem.py:63-65` — shipped to the model every
   turn for a block that no longer exists), 4 dead prompts
   (`rephrase_as_memory_style`, `generate_memory_name`, `summarize_file`,
   `classify_task_type`), dead config toggles (`pii.cloud_only`, score-floor keys,
   `min_rank_threshold`, unused `decay_rates` — either wire FadeMem or remove the
   config that implies it), `chroma_retry_queue` table.
3. ~~**Executor/ChatLoop unification**~~ — **DONE 2026-08-05** (`78eee56`,
   `885d1c9`). `_execute_step` (the `/workflow` path) now goes through the shared
   runner, so it finally has endpoint fallback, vision gating and PII scrubbing;
   `images` ride on the executor's task message instead of truncatable history;
   and each endpoint attempt snapshots `messages` and rewinds on failure, killing
   the dirty-retry replay (tool budget multiplying by endpoint count, and
   orphaned `tool_calls` 400ing on OpenAI-compat).

   Two deviations from the plan as written, both worth remembering:
   - **Deleting the legacy direct-ChatLoop branch was wrong as specified.**
     "Tests can inject a runner" doesn't cover it — `tests/benchmark_coding.py`
     builds a bare `TaskExecutor` with no Agent, so that branch is a live
     consumer on the Ollama PC. It became a single shared `_run_loop` helper
     instead of being deleted.
   - The `EndpointRunner` extraction was **not** needed. Routing both executor
     paths through the existing `chat_loop_runner` got the same result without
     moving the vision/fallback logic out of `ChatMixin`. Revisit only if a
     third non-Agent caller appears.
4. ~~**`cli.py` split**~~ — **MOSTLY DONE 2026-08-05/06** (`3526969`,
   `aca0e24`, `340c529`). 4,518 → 2,184 lines:
   - renderers → `ui/views.py` (33 functions, moved verbatim)
   - slash dispatch → `ui/commands.py` registry + `ui/command_handlers.py`;
     `/help` is now DERIVED from it, and `simulate/slash_dispatcher.py` went
     from a 564-line hand-maintained copy to a 108-line driver over the same
     registry (a test asserts the two command sets are equal)
   - five copy-paste importers → one `ui/importers.py`, which surfaced that
     four of the five never took `import_lock` and so could collide with
     nightly
   - UI globals → `ui/state.py`, so simulation gets isolated state

   **Terminal plumbing (~440 lines) deliberately NOT moved.** It's the one
   part the dev-box suite can't validate — Esc-cancel and the approval prompt
   were verified interactively on the Ollama PC — so a subtle break there
   would pass CI and only show up under your hands. Two functions in views.py
   and one in command_handlers.py import from it lazily to avoid a cycle.
5. **Shared context assembly** — **LIGHT VERSION DONE 2026-08-06** (`b785754`).
   The executor now gets the scratchpad, session notes, follow-ups and time
   anchor via one shared `_build_continuity_block()`, so `!plan` no longer
   carries less continuity than ordinary chat.

   **Still deferred (the full version):** the executor uses a flat 15-result
   memory search instead of the 5-pool token-budget system. Unifying that is
   the real refactor — the two paths genuinely want different shapes, so it
   needs a design decision, not a move. Revisit if the executor's recall feels
   thin in use.

6. ~~**MCP (per D2)**~~ — **DONE 2026-08-05.** Removed: the package, the
   `_connect_mcp_servers` registration path, `MCPServerConfig`, the `/mcp`
   command and renderer, the simulate dispatcher entry and two scenario steps,
   and the `mcp>=1.0.0` dependency (~400 lines). It had never run — no
   `mcp_servers:` key ever existed in `config.yaml` and there were zero tests
   — so there was no behavior to preserve. Restore from `063d06f`.
7. **Tests for the riskiest untested code**: `MemoryWorker` (zero tests today; the
   only second-event-loop-in-a-thread in the codebase); ChatLoop's six untested
   terminal states (budget, nudge, stopped, empty, text + compaction-in-loop,
   parallel execution, failing tool — add a `RaisingTool` to fakes); a 3-fake-job
   `NightlyRunner` test (ok/raises/hangs); `Agent.end_session` teardown ordering.
8. **`scripts/` hygiene**: `git mv` the ~14 completed one-offs to `scripts/archive/`;
   rename `test_*.py` scripts so bare pytest can't collect them (one hits live
   Ollama); consider `blipshell/maintenance/` for the four modules nightly imports
   from `scripts/`.
9. **Docs**: refresh CLAUDE.md ([STATE]/wind-down claims, vision-gap overstatement,
   21→23 nightly jobs, 26→37 scenarios) and prune `config.yaml` comment drift
   (`reasoning:` still claims lessons; Groq/Gemini priority comments).

---

## Phase 3 — Throughput + benchmark realism (Ollama PC involved)

1. ~~**Consolidation rewrite**~~ — **DONE 2026-08-06** (`e65714d`). Queries with
   each memory's stored vector (no Ollama round trip per check), batch default
   20 → 2000 bounded by a resumable time budget, losers ARCHIVED not deleted
   (deleting cascaded their entity edges away), plus a dry-run mode, a partial
   index on `consolidated_at`, and `access_count` in `update_memory`'s allowed
   set. Also fixed a space mismatch: it used to compare a summary-embedding
   against content-embeddings, so 0.85 was never really calibrated.

   **BEFORE THE FIRST REAL RUN:** set `consolidation_dry_run: true`, run
   `blipshell nightly --job consolidate`, and read the merge log. At 2000/night
   a miscalibrated threshold does far more damage than it could at 20, and
   merges are one-way. Back up first.

   **SWEEP COMPLETE 2026-08-07.** All 31,977 active memories checked; 59
   archived as near-duplicates (0.18%); `integrity_ok`, and an independent
   check found no edge or mention orphaned across 68K edges and 113K mentions,
   so archive-never-delete held.

   **DO NOT lower the 0.92 threshold on the strength of that low yield** — the
   question was settled with data, see `scripts/consolidation_calibrate.py`
   (2000-memory sample):

   | band | share |
   |---|---|
   | 0.96+ | 0.00% |
   | 0.92 – 0.96 | 0.30% |
   | 0.88 – 0.92 | 9.40% |
   | 0.85 – 0.88 | 9.90% |

   Nothing above 0.96 survives anywhere in the corpus, because the write-time
   LLM dedup (`processor.py`, threshold 0.7) already removes it. Consolidation
   is a narrow mechanical net behind a wider LLM-arbitrated one, so a near-zero
   yield is the *expected* result, not a symptom.

   Reading the 0.88–0.92 band by hand, ~4 pairs in 10 are genuine duplicates.
   The rest are same-topic-different-fact — an Ollama payload using
   `llama3.2:latest` vs one using `gemma3:latest` scores 0.9198, the same
   failure mode as `_v1`/`_v2` entity names: the embedding scores the shape and
   misses the distinguishing token. Dropping to 0.88 would merge ~188 pairs per
   2000 and get well over half of them wrong, one-way.

2. **Benchmark closes its known gap**: add 2-3 `SESSION_REVIEW_CASES` above the
   ~28.6K chunk threshold, route `run_session_review` through
   `MemoryProcessor.process_reflection`, score `chunk_merge` as its own job. Retire
   `scripts/audit_reflection_quality.py` into it. (This makes `JOB_OWNERS`'
   session_review claim honest — chunk+merge is the highest-variance prompt in the
   system and has never been scored.)
3. ~~**Entity pipeline hardening**~~ — **DONE 2026-08-07** (`871d800`). All four:
   `get_all_entity_names()` cached (invalidated on create/archive/revive/merge,
   and explicitly by `_job_entity_cleanup`, which writes through the raw
   connection); `get_entity_id_by_name` matches `(name, entity_type)` with a
   stable lowest-id fallback; failed extractions stay unmarked and report
   `retryable` rather than being recorded as done; the version rules moved to
   `memory/entity_names.py` so the creation-time resolver uses the same guard
   as the merger.

   Two notes for whoever reads this next. The guard `continue`s past a blocked
   candidate instead of returning — candidates are similarity-sorted, so a
   version variant routinely outranks the real match, and the fall-through is
   also the only path that upserts the new entity's embedding. And the cache
   deliberately does *not* invalidate on `get_or_create_entity`'s read path or
   on a no-op `revive_entities`; both fire per-triple during extraction, so
   invalidating there would make the cache pointless. 23 regression tests,
   each mutation-verified.
4. **Local fallback model refresh** (SYSTEM_REVIEW D): benchmark newer local
   candidates on the Ollama PC, test-first as always.

**PARKED 2026-08-07 — items 2 and 4, deliberately.** The models in production
are working; there is no pull to change them, so the leverage isn't in scoring
more of them. Revisit after Phases 4–5. Note that three of the four local
models in `config.yaml` (`gpt-oss:latest`, `glm4:latest`, `qwen3.5:9b`) have
never been benchmarked at all — `gpt-oss` is the fallback for tool_calling,
coding AND reasoning, i.e. the entire interactive path when cloud is down. That
is a known unmeasured risk, accepted on purpose, not an oversight.

### When it is picked back up: monitoring, not procurement

The harness answers "which model is best" (cross-sectional, judge-scored,
expensive, rarely run). The actual question is "has my model stopped being good
enough" — a time series. Evidence from the committed corpus for why the current
shape can't answer it:

- `contradiction` scores **exactly 1.000** for every model measured
  (minimax-m3, glm-5.2, kimi-k2.7, qwen3:14b). A saturated metric carries no
  information.
- `ranking` and `importance` sit in 0.914–0.973 across everything from local
  qwen3 to cloud minimax — and both are import-path-only prompts that don't
  measure production at all.
- **Within-model variance is ~40% of between-model signal**: qwen3:14b scored
  entity 0.825 then 0.853; the whole between-model spread is 0.790–0.868.
  Single runs are reported as precise point scores with no confidence interval.

Three tiers, cheapest first:

1. **Production failure counters** — free, no LLM calls. Unambiguous failures
   already happening and being discarded: entity extraction yielding no
   parseable triples (`EntityExtractor`'s `retryable` counter already counts
   this), empty/over-length summaries, unparseable tool calls, session reviews
   producing zero lessons, endpoint 410/4xx (the kimi-k2.5-retired class).
   Count per model per day.
2. **Canary suite** — ~15 deterministic checkable cases, no judge, nightly,
   against live assignments only. Detects *change*, not quality: a drifting
   local model, or a cloud provider swapping a checkpoint underneath you.
3. **The existing deep benchmark becomes the diagnostic**, run in response to
   1 or 2 firing (or to evaluate a specific candidate) rather than as a chore.

Build tier 1 first — it needs no Ollama, no new cases, and accumulates
unattended, so tier 2's thresholds can be set from real data instead of guessed.

---

## Design notes for remaining phases (2026-08-07)

Written at the model switch: design decisions resolved up front so the
implementation session inherits them instead of making them ad hoc. The
escalation rule stands — one-way migrations, security-relevant code, and new
design decisions come back for review BEFORE running against the live DB.

### Phase 4.1 — thought store migration — DONE 2026-08-07 (`dbc4fb5`)

Current state (verified 2026-08-07): all thoughts live in ONE app_metadata
row (`self_thoughts`) as a JSON list; each item carries its 1024-float
embedding INLINE, so every load parses ~50 × 1024 floats; weights are keyed
by thought TEXT, and echo-folding rewrites text ("the evolved phrasing
wins"), so a fold can orphan its own weight and duplicate texts collide.

Decisions:
- Table `self_thoughts`: id PK, text, created_at, weight, surface_count,
  echo_count, pending (the one-shot greeting flag), is_archived. Embeddings
  go to `vec_self_thoughts` (vec0, dim 1024, rowid = thought id) like the
  other four vec tables — never inline.
- Everything keys by id from then on. Folding updates the loser row
  (is_archived=1, fold provenance in a metadata column) — ARCHIVE, never
  delete, same mandate as everywhere else.
- Migration: startup, guarded by an app_metadata marker. Do NOT delete the
  JSON blob — rename its key to `self_thoughts_pre_migration` so rollback is
  one rename. Thoughts without embeddings migrate as rows and stay
  backfillable (the existing `_backfill_embeddings` semantics).
- ThoughtStore's public API stays identical; only storage changes. The
  two-channel weight firewall is untouched — table weights are the
  self-layer channel only, retrieval never reads them.

**Shipped.** One deviation from the spec below: embeddings are a float32
BLOB on the row, NOT `vec_self_thoughts`. Only the VectorStore connection
loads sqlite-vec, so vec0 would put a cross-connection write on the per-turn
path — the same aiosqlite-vs-sync contention that already bit the nightly
orphan sweep — and at max_keep=50 the KNN advantage is nil. Eviction and
folding ARCHIVE (with `folded_into` provenance). The old JSON blob is kept
at `self_thoughts_pre_migration`; **rollback is renaming that key back to
`self_thoughts` and clearing the table**.

### D1 — local-mode routing (decision still with the user)

Recommendation: an explicit per-session `/local` toggle (+ config default
`privacy.local_mode_default: false`), implemented as a ROUTER-level endpoint
filter — local mode excludes every endpoint that has `pii_sanitize: true`
(i.e., everything that leaves the machine), rather than touching call sites.
Show state in the status line. Auto-detection of "personal" content was
considered and rejected: it costs an LLM call per turn and fails exactly on
the borderline content it exists to protect. Known edges to fold in when
built (from the 2026-08-07 review): multimodal content parts and tool-call
arguments are not secret-scrubbed — acceptable now, in scope for D1.

### MemoryWorker tests (the riskiest untested code)

Design: real thread + real event loop + real tmp-DB SQLite — no mocked
sleeps (house rule). The enabler is making the timing constants
(`_IDLE_EXTRACT_INTERVAL` etc.) constructor-injectable; the 2026-08-06
mutation-testing round proved a wall-clock-bound test here is vacuous.
Must cover: crash-safety (rows persisted `is_processed=0` are reprocessed by
the startup sweep), queue draining on shutdown (no lost messages), the idle
entity-extraction trigger, and double-start/double-stop idempotency.

### Phase 5.1 — user model (shape only; rest is written where listed)

One document, LLM-REVISED (never appended) by a nightly job, hard size cap
~1.5K tokens, lives in the core pool. Reasoned conclusions with a
confidence marker per line — not facts (the entity graph owns facts).
Revision prompt must be able to RETRACT: a conclusion the recent sessions
contradict gets weakened or dropped, not argued with.

## Phase 4 — Alive layer: step 2, properly gated

_Gate: re-take the `/thoughts` readout only after Phase 0.6/0.7 and Phase 1.8 have
been live long enough to produce ≥10 NEW thoughts (not N days). The current readout
is distorted by the eviction and fatigue bugs._

1. **Thought store → real table + `vec_self_thoughts`** (matching the other four vec
   tables). Prereq for step 2; also removes per-turn JSON-blob parsing and gives
   thoughts stable ids (weights are currently keyed by text).
2. **Entity names as reflection stimulus**: inject currently-active entity names
   into the reflection prompt as *stimulus, not content*. Breaks the closed-loop
   monoculture without violating the no-transcript rule, and IS step 2's foundation
   — `_expand_via_entities` already computes the active-entity set and throws it
   away.
3. **Graph-relational gravity**: cosine-gated spreading activation / Personalized
   PageRank over the existing edges table, seeded from active entities
   (HippoRAG-style; matches the saved ACT-R blueprint). Self-layer ONLY — the
   firewall stands; retrieval ranking is untouched.
4. **Restart the reflection task on re-`start_session`** (it currently dies for the
   process lifetime after any `end_session` in web/telegram flows), and move
   `_backfill_embeddings` off the per-turn path.
   **DEMOTED 2026-08-07 — user: "web ui is basically dead at this point."** In
   CLI use, `end_session` means the process is exiting anyway, so the dead
   reflection loop is unreachable. Not worth a dedicated pass; fix it in
   passing if session lifecycle is ever touched, or as a prereq if Telegram
   (5.5) is ever picked up.

---

## Phase 5 — New capabilities (pick by appetite; ordered by leverage)

1. **User model (Honcho-inspired)**: one LLM-maintained document of reasoned
   conclusions about the user (preferences, values, working style) — not facts, the
   layer above facts. Nightly job revises it from recent sessions; lives in the core
   pool. Small build, compounds with everything.
2. **Morning briefing**: first interaction of the day assembles nightly status
   (including failures — this is also the trust ledger), consolidation counts,
   digest changes, and the current lingering thought. The continuity experience and
   the observability fix are the same feature. (Letta sleep-time compute is the
   precedent: idle work should improve the *next conversation*, not just plumbing.)
3. **Digests-as-committed-files**: render project digests + anti-pattern lessons as
   markdown committed into the target repo, where Claude Code / any tool reads them.
   This is the concrete mechanism for "memory layer over your coding life" — ~a
   rendering job on data already maintained.
4. **`/why` provenance**: persist `last_search_stats` into the already-designed
   `turn_events` table; answer "why did you bring that up?" with the actual
   retrieval trace. Anti-confabulation Phase 2, and the observability needed before
   any future retrieval change.
5. **Telegram presence**: the alive layer matters more when "you return" can mean
   your phone. Prereqs: the web session race fix (global Agent + shared
   SessionManager; two connections wipe each other), per-connection sessions,
   idempotent `end_session`.
   (Note 2026-08-07: the web UI is effectively unused — "can't remember the
   last time I started it." Its session-race prereqs only matter if this item
   is ever picked up; don't fix them speculatively.)
6. **Idle anticipation (ProAct-lite)**: during idle, pre-run memory searches for the
   top unresolved follow-up/friction thread; stage warm context for the return.
   Free at 3 AM on a local GPU.
7. **Claude Code session ingestion as a routine source**: the `claude_code` importer
   exists — schedule it, so the heaviest daily LLM activity feeds the assistant's
   memory instead of bypassing it.

---

## Explicitly NOT doing

- Rewrite. (Settled above.)
- Retrieval tuning, reranker enablement, FadeMem enablement — good-enough mandate
  stands. (Note for whenever that changes: the reranker as-written would *degrade*
  ranking — normalization scope bug + `logprobs` never requested — fix before any
  future enable.)
- Web UI feature parity. (Command registry first; web becomes a thin driver.)
- mem0/Zep/Letta adoption. (Reconfirmed 2026-08: vendor-benchmark-driven; we own an
  equivalent stack.)
- Coding-agent parity features. (Frozen per direction decision.)
- Chasing `qwen3` etc. model swaps without Ollama-PC benchmarks. (Standing rule.)

## Suggested cadence

Phase 0 is 1-2 sessions and pure wins — start there. Phases 1-2 interleave fine
(each item is independent). Phase 3.1-3.2 need Ollama-PC time — batch them with a
benchmark day. Phase 4 waits for its gate by design. Phase 5 items are morale food —
schedule one every few sessions so the project stays fun while the plumbing gets
honest.
