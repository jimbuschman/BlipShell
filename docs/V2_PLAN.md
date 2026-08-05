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
| D1 | PII on interactive chat | (a) wire sanitize into the chat path; (b) consciously accept raw cloud chat and delete the misleading `pii_sanitize` flags | the remaining half of Phase 1.4 |
| D2 | MCP: commit or archive | (a) invest (resources + supervision + tests); (b) archive the package | Phase 2.6 |
| D3 | Coding freeze confirmation | watch 2 weeks of real usage; if project mode earns its keep weekly, keep layer 2 "warm" instead of frozen | Phase 2 deletions of anything user-facing |

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
3. **Executor/ChatLoop unification** (SYSTEM_REVIEW E2, scope now known — ~1 day):
   extract vision gating + endpoint fallback from `_run_chat_loop` into an
   `EndpointRunner`; point `_execute_step` (workflow path — currently has *no*
   fallback, vision, guardrails, or compaction) and the legacy branch at it; thread
   `images` onto the executor's task message (today the image rides on truncatable
   history); snapshot/restore `messages` per endpoint attempt (the dirty-retry bug:
   failed attempts replay half-finished tool exchanges and 400 on OpenAI-compat).
4. **`cli.py` split (4,518 lines → ~400)**: slash dispatch → `ui/commands/` registry
   (**deletes `simulate/slash_dispatcher.py`**, a hand-maintained copy already
   drifted to 24-of-33 commands — the sim harness cannot currently exercise
   `/thoughts` at all); `_print_*` renderers (~1,900 lines) → `ui/views/`; five
   copy-paste importers → one parameterized command; terminal plumbing →
   `ui/terminal.py`. The registry is the frontend-agnostic layer the web UI and
   Telegram drive later — web parity is NOT pursued feature-by-feature.
5. **Shared context assembly**: extract `_build_messages`' assembly into a
   `ContextBuilder` both paths use. Today `!plan` silently drops scratchpad, session
   notes, follow-ups, and the 5-pool budget system — a memory-continuity regression
   on exactly the path used for hard tasks. This is the "memory is the moat even for
   coding" item.
6. **MCP (per D2)**: if commit — resources first (external context into pools),
   then connection supervision, plus a config entry and a first-ever test; if
   archive — remove the package (it has never run: no `mcp_servers:` key has ever
   existed, zero tests).
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

1. **Consolidation rewrite** (~20/night → ~2,000/night): reuse *stored* vectors
   (self-join / `get_embeddings_by_ids` — the current code re-embeds via Ollama per
   check, comparing summary-embeddings against content-embeddings); resumable time
   budget like `merge_entities`; **archive-not-delete** (current `delete_memory`
   cascades away entity edges — violates the mandate at the edge level); dry-run
   mode before raising throughput; index on `memories(consolidated_at)`.
2. **Benchmark closes its known gap**: add 2-3 `SESSION_REVIEW_CASES` above the
   ~28.6K chunk threshold, route `run_session_review` through
   `MemoryProcessor.process_reflection`, score `chunk_merge` as its own job. Retire
   `scripts/audit_reflection_quality.py` into it. (This makes `JOB_OWNERS`'
   session_review claim honest — chunk+merge is the highest-variance prompt in the
   system and has never been scored.)
3. **Entity pipeline hardening**: cache `get_all_entity_names()` (a full ~31K-row
   read + Python substring loop on *every* search); `get_entity_id_by_name` must
   match on `(name, entity_type)`; extraction failures get a retry/error column
   instead of being marked done; version-guard (`_v1`/`_v2`) the creation-time
   resolver like the merger already does — the *unguarded* path is the one running.
4. **Local fallback model refresh** (SYSTEM_REVIEW D): benchmark newer local
   candidates on the Ollama PC, test-first as always.

---

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
