# BlipShell v3 Plan - Evidence You Can Trust, Then Learning That Changes Outcomes

_Date: 2026-09-08. Synthesis of three inputs: an external system review of commit
`efcbe02` (`BlipShell-system-review.pdf`, 12 pp., eight reproduced findings), a
conversation about what memory and lessons are leaving on the table (`blipshell
review from gpt6.txt`), and two rounds of review of this plan's first draft
(`output.txt`). All eight code findings were re-verified against main on
2026-09-08 by reading the call sites listed below. Supersedes the sequencing in
`docs/V2_PLAN.md` - v2 Phases 0-2 are done; its Phase 4 (alive layer step 2) and
Phase 5 items carry forward unchanged and are not restated here._

**File:line references were verified against main on 2026-09-08.** Re-check before
fixing - code moves.

---

## The verdict that shapes everything

- **No rewrite, again.** Both external reviewers reached the same conclusion
  independently: the parts are worth keeping; the contracts between them are weak.
  The failures found are silent-correctness bugs and lost evidence, not
  architecture.
- **The v3 thesis: memory is only worth what survives assembly and what changes
  behaviour.** v2 made failure honest. v3 makes evidence traceable (a retrieved
  fact must reach the model, and `/why` must tell the truth about it) and makes
  learning accountable (a lesson must be judged on outcomes, not on a model
  agreeing with another model's reflection).
- **The pivot is parked.** The gpt6 conversation proposed a Voyager-style
  executable-skill library with a self-running learning runner and transfer
  tests. That is a research programme, not an assistant feature, and it is
  coding-agent territory the v2 direction declined. If it is built, it is built
  in Wisp or a separate repo, after this plan's correctness work, because a
  learning experiment measured through a loop that double-executes tools and
  truncates answer-bearing text would be measuring noise. The one thread from
  that conversation kept here is outcome-backed lessons (Stage D).
- **Two gates, and a stop rule.** Stage B has a deterministic gate (continuity set
  survival + false-recall rates). Stage E has a behavioural gate (resume an
  abandoned project correctly). If a stage's gate does not move, stop and
  reassess rather than refining infrastructure because the metrics look cleaner.

## Standing decisions honored throughout

- ARCHIVE, never DELETE. Every "remove" below is a soft archive.
- Retrieval ranking is good-enough and stays untuned. Everything here fixes
  what happens *after* ranking (packing, rendering, truncation, provenance) or
  *before* it (what gets stored, and as what). The Sep 2 provenance readout
  (59.8% assistant-authored corpus vs 37.6% of top-10) stands: no source penalty.
- Test before commit, full suite, no exceptions. Logic/wiring validates here;
  model behaviour validates on the Ollama PC (or over Tailscale, ask first).
- No quick fixes; trace the full call chain first. Fix at the chokepoint,
  delete the copies.
- Nothing in this plan changes model routing. No new model is justified by a
  code review.

## Progress

| Stage | Status |
|---|---|
| A - Correctness | A1 BUILT 2026-09-08 (`v3/a1-dedup-parser`, merged). A2 + A3 BUILT 2026-09-09 (`v3/a2-a3-loop-recovery`, two commits). Structured-dedup measurement: see A1 "Measured". A4-A6 not started |
| B - Context contract | not started |
| C - Continuity set | not started |
| D - Accountable lessons | not started |
| E - Project dossier + decisions | not started |
| F - Bounded initiative | deferred until E shows reuse |

---

## Stage A - Correctness (silent failures with side effects)

Eight findings from the external review, all reproduced there with scripted
models and temp state, all re-verified here at the source. Ordered by damage.
Each item ships with its own test in `tests/`, driven by `ScriptedLLMClient`
where a loop is involved; none needs a model.

### A1. Dedup parser can archive the wrong memory (HIGH)

`memory/processor.py:447-465` `_parse_memory_action`: substring scan over the
upper-cased reply in the order NONE, UPDATE, DELETE, ADD; UPDATE/DELETE with no
number default to **index 0**. So `"Do not DELETE anything; ADD this as
distinct."` parses as `DELETE 0`, and a bare `"UPDATE"` targets item 0. The
caller then archives the selected old memory. Archive-not-delete means the row
survives, but the fact leaves ordinary retrieval - the exact failure the
memory layer exists to prevent, triggered by the model being *careful*.

Fix, two paths behind one config toggle (`memory.dedup_structured_output`,
default **off**):

- **Strict parser (default).** Whole-response grammar: the reply must BE an
  action, optionally followed by an index, and nothing else (first/last
  non-empty line, reduced to its last sentence segment; two differing
  verdicts is a conflict). UPDATE/DELETE require an explicit in-range
  1-based index. Anything else returns `("RETRY", None)`. Missing or
  out-of-range index is a parse failure, never a default.
  **As built (2026-09-08), RETRY re-asks ONCE with a sharper suffix, then
  applies ADD**: the new memory stays, nothing is archived, and the new row's
  metadata gets `dedup_undecided = {candidates, reply, at}` so the cases can
  be found. The draft said "leave `is_processed=0` for the sweep" - rejected:
  that re-runs summarize+embed on every sweep and, for a consistently
  ambiguous model, is exactly the dedup zombie loop the 2026-08 deep-dive
  found. `tests/test_dedup_decision.py::TestPipelineUndecided` pins the row
  ending processed.
- **Structured output (experimental).** Ask for a JSON object
  `{action, target_index, reason}` via the `format` kwarg that
  `llm/openai_client.py:144,183,283` already passes through; validate against a
  schema; invalid -> same RETRY path. **Treat as an experiment until measured
  on the model that actually serves this call, over Tailscale.** Which model
  that is comes from config.yaml's endpoint role maps, not the `models:`
  block and not CLAUDE.md's table: the verdict is `TaskType.REASONING`, only
  the `local` endpoint lists `reasoning` in its roles, and it has no override,
  so today it is qwen3:14b with gpt-oss:latest as fallback - NOT the chat
  model (deepseek-v4-flash on OpenRouter). Production calls it with
  `think=False` (`processor._ask_dedup_verdict`) and the benchmark job does
  the same. Two cautions when reading the number: qwen3 degrades with
  think=False (CLAUDE.md Conventions), so a low `valid_rate` may be the think
  flag rather than the schema - run a think=True variant before concluding;
  and local models behave oddly under schema constraints in thinking modes,
  which is why this is measured, not assumed. Benchmark job `dedup_structured`
  (owned by `models.reasoning` in `JOB_OWNERS`) scores schema-validity rate;
  enable only if it clears ~98%.
  **Measured 2026-09-09** (`benchmark_results/20260909T025558__qwen3_14b.json`,
  3 repeats, 144 live calls, cache bypassed - the first run that morning was
  discarded because every repeat was a 0.0s cache hit; see commit 9539e8d):

  | path | think | valid_rate | accuracy |
  |---|---|---|---|
  | text (production) | False | 1.000 | 0.750 |
  | structured JSON | False | 1.000 | 0.750 |
  | text | True | 1.000 | 0.750 |
  | structured JSON | True | 1.000 | 0.806 +/- 0.083 |

  Both grammars accept 100% of what qwen3:14b actually emits, so the schema
  gate clears - but structured output buys nothing at think=False (identical
  misses: two NONE-gold cases answered ADD, one UPDATE-gold answered ADD) and
  costs ~5x latency (median 0.24s text vs 1.30s JSON, from the transcripts).
  **Decision: `structured_output` stays OFF.** Every
  miss is in the SAFE direction (ADD keeps both records; nothing wrong was
  archived), and two of the three are arguable gold - the prompt's own rule
  "if it adds ANY new information choose ADD" is what the model followed.
  Do not tune the prompt off this; if redundancy matters, that is
  consolidation's job. The one thing worth knowing: think=True nudged the
  JSON path up, not the text path, and cost ~25x latency (median 6-8s, max
  63s) - not worth it for a background call that already defaults to
  think=False.
- Either way: log the candidate set, the raw reply, and the decision at INFO,
  and stamp the ARCHIVED row's metadata with
  `dedup = {action, by, candidates, reply, structured, at}`. `blipshell repair
  --unarchive-memory ID` (repeatable, honours `--dry-run`, not part of
  `--all`) prints that record, un-flags the row, re-embeds it from raw content
  and keeps the record with an `unarchived_at` - history is never erased.
- Benchmark: TWO pipeline rows, `dedup` (text path; unparseable counts as
  WRONG, since in production it costs a re-ask and a default) and
  `dedup_structured` (JSON path; accuracy over valid replies, `valid_rate`
  separate and deciding). Both owned by `models.reasoning` in `JOB_OWNERS`;
  `dedup_structured` is displayed but excluded from the composite because
  production does not run it. Twelve cases, three per verdict, UPDATE/DELETE
  targets deliberately not always item 1.

Tests (`tests/test_dedup_decision.py`, 40 cases + `test_benchmark_harness.py`):
the two quoted replies produce RETRY; noisy-but-unambiguous replies still
parse; RETRY twice leaves every candidate unarchived, the new row processed,
and `dedup_undecided` recorded; a valid second reply is applied; out-of-range
is undecided; the archive stamp survives existing metadata; the structured
path forwards the schema and rejects text verdicts; `router.generate`
forwards `response_format` as `format`; unarchive restores, re-embeds, keeps
history, and is a no-op in dry-run.

### A2. Endpoint retry re-executes completed tool calls (HIGH)

`core/agent_chat.py:396` snapshots `messages` before `loop.run`;
`:426` rewinds to it on any exception, then the next endpoint replays the
turn. `ChatLoop.run` starts `tool_call_count` at zero per call. Reproduced by
the reviewer: a state-changing tool ran twice under `budget=1`. The rewind
was added deliberately (see the comment above `:396`) to stop OpenAI-compatible
endpoints 400-ing on an assistant `tool_calls` message with no results -
removing it naively brings that back.

Fix: recovery moves **inside the loop, at the model-call boundary**.
**As built (2026-09-09):** `LoopConfig.on_model_call_error(exc) ->
(client, model, chat_kwargs) | None`. `ChatLoop._call_model` wraps every
model call (main loop and budget nudge); on failure it asks the callback for
the next endpoint and retries the SAME call with `messages` untouched.
`_run_chat_loop` is now an endpoint SELECTOR: it picks the first endpoint,
installs the callback (which books the failure - `mark_model_failed` or
`record_failure` - releases the request slot, and selects the next), and
runs the loop once. The snapshot/rewind is gone. Every endpoint that failed
this turn is excluded (the old code excluded only the last one, so A-B-A-B
cycled a dead endpoint); the fallback-model path clears the set so each
endpoint gets one shot with the new model. Tool count is turn-wide because
there is only one `run()`.
The "failure between assistant append and results" window no longer exists:
the sequential tool path now catches a RAISING tool and records a failure
result (the parallel path already did), and the A3 pairing repair runs
before every send. So the rewind's one legitimate case is structurally
prevented rather than handled. The durable-id/started-completed ledger the
review proposed was NOT built: with no cross-endpoint replay there is
nothing to reconcile; revisit only if a real double-execution shows up.
**A transcript repair is never a side-effect rollback.**

Tests (`tests/test_endpoint_retry.py`, rewritten to the new contract): the
tool runs once and endpoint B sees the completed exchange; budget=1 is
turn-wide across endpoints; failure before any mutation hands B the
untouched conversation; the caller's list holds one consistent exchange;
every endpoint dead -> None without re-execution; fallback model continues
the turn; request accounting balances; the loop hook alone swaps clients,
re-raises on None, and a raising tool becomes a failure result, not an
orphan.

### A3. Budget trimming orphans announced tool calls (HIGH)

`core/chat_loop.py:815-818` appends the assistant message with the **full**
`tool_calls`; `:828` slices `parsed_calls` to the remaining budget. Announced
calls past the budget get no result. OpenAI-compatible endpoints reject the
next request. Fix: every unexecuted call gets an explicit tool message
(`ToolFailure("budget exhausted: not executed")`), and a pairing check runs
**before every outbound request**, not only after batches. Executed-call
accounting stays separate from declared-call accounting.
**As built (2026-09-09):** calls past the budget are DENIED
(`BUDGET_DENIED_RESULT`, success=False), never executed/counted/dedup'd and
never passed to `on_tool_executed`; the completion tool is never denied
(slicing used to drop `task_complete` silently and end the turn on
"budget"). `repair_tool_pairing()` inserts synthetic "no result recorded"
results for orphaned ids and runs before every send, logging at WARNING
when it had to. `tests/test_tool_budget_pairing.py`.

### A4. Commit ingest drops backlog forever (MEDIUM)

`memory/commit_ingest.py:27` `MAX_SUBJECTS_PER_PROJECT = 10`, newest first;
`:95` advances the watermark to the newest collected epoch + 1. Twelve commits
-> ten ingested, two never seen again. The watermark is also stamped before the
user-model revision persists, so a failed revision consumes its evidence.
Fix: a durable evidence queue keyed by `(repo identity, commit hash)`; drain in
bounded batches oldest-first; acknowledge only after the derived update is
stored; acquisition cursor separate from revision cursor. `tests/test_commit_ingest.py`
covers the watermark but not the >10 case - add it with a real temp repo.
**As built (2026-09-09):** table `commit_evidence` (unique on
`(repo_root, sha)`, status pending/consumed). `acquire_commits` runs
`git log --since=@cursor` with NO count cap into the queue; the cursor is
the newest acquired epoch with no +1, because the hash is the identity and
a same-second re-read is an ignored insert. A never-seen project takes its
newest `MAX_INITIAL_COMMITS` (50) only - older history predates tracking.
`collect_commit_evidence` acquires then drains oldest-first, 10 per project,
returning `CommitEvidence(lines, ids, acquired, pending_after)`;
`acknowledge_commit_evidence(ids)` is called by `UserModel.revise_from_reflections`
only after the doc is persisted or the model honestly concluded nothing. A
raised revision leaves the rows pending. The legacy watermark key seeds the
cursor once so an upgraded install does not re-queue judged commits. The
nightly `update_user_model` stats now carry `commits` and `commits_pending`.
Test-writing lesson: git's `--since` matches NOTHING for tiny epochs
(`@1000`), so pinned test dates must sit in a realistic range.

### A5. Parallel partition ignores read/write (MEDIUM)

`core/chat_loop.py:698-725` `_partition_for_parallel` sequences only
approval-gated tools **when an approval callback exists** plus `ask_user`. In
normal chat there is no callback, so every mutating tool runs concurrently.
Fix: tools declare an `effect` at registration (`read` | `write`, v1); only
reads parallelize; writes run sequentially in declared order. Extend to
`pure_read | external_read | idempotent_write | mutating_write | exclusive`
only when a real conflict shows up - the two-class split is the right first
version.

### A6. Small ones, same change set

- **numpy undeclared**: `memory/centroid_tagger.py:12` imports it at module
  scope; `nightly.py` imports the tagger; a clean install fails test collection
  in five files. Add to `pyproject.toml` dependencies. Add a clean-venv import
  smoke test for the CLI and nightly runner (CI item from the 2026-09-04 audit).
- **Scratchpad is cwd-relative**: `core/agent_chat.py:1402,1412` use
  `os.path.join("data", ...)`. This is the split-database class of bug.
  Route through `resolve_config_relative()`.
- **Test count drift**: the review counted 2,001 passing on Python 3.12;
  CLAUDE.md says 1,620. Recount and fix the doc.

**Stage A gate:** failure injection preserves completed actions; every
announced tool id pairs with a result before every outbound request; the two
quoted dedup replies archive nothing; twelve commits ingest as twelve; a
clean venv imports `blipshell.core.nightly`. All deterministic, all here.

---

## Stage B - Context contract (evidence that reaches the model)

Three reproduced findings plus a design rule. This stage is bigger than it
looks because of one coupling, called out in B1.

### B1. One representation of the current conversation

`session/manager.py:154` adds each turn to the `ActiveSession` pool (rendered
inside the system message) **and** `core/agent_chat.py:1330` appends the last
20 session messages as role messages. The current user turn appears twice.
Pool usage stats omit the appended history, tools and instructions, so the
budget is against a fraction of the real request; the mechanical compactor
preserves system messages, so it preserves the duplicate.

Fix: the conversation is role messages, once, chronological. `ActiveSession`
stops carrying transcript text. **Coupling:** `memory/manager.py:232`
summarises `ActiveSession` overflow into `RecentHistory` - that trigger has to
move to the role-message history (by token count of the appended window), or
`RecentHistory` silently stops filling. The whole assembled request (system
prefix, tools, history, recall, response reserve) is budgeted against the
selected endpoint's `context_tokens`; on fallback to a smaller window, rebuild
from the same evidence records instead of only changing `context_limit`.

### B2. Pool packing that skips instead of stopping

`memory/manager.py:93-100` `Pool.get_top_entries` breaks at the first
non-fitting item; a 101-token top item under a 100-token cap selects nothing.
Fix: per-pool rule. Recall and Lessons **skip** an oversized item and keep
packing (record why it was omitted). Core may reserve slots. Conversation is
chronological, never independently packed. Every rejected item leaves a
reason in the trace.

### B3. Query-relevant passages, and a `/why` that tells the truth

`core/agent_chat.py:893` cuts every recalled memory to its first 1,200 chars;
`core/tools/memory_tools.py:55` does the same for `search_memories`. A fact
found by FTS/vector at character 1,400 is retrieved and then thrown away.
`core/agent_chat.py:994,997` records the trace as `"injected"` **before**
`gather_memory` decides what fits, and `ui/command_handlers.py:177` `/why`
prints that trace as if it were what the model saw.

Fix: replace prefix truncation with a deterministic sentence window around the
lexical/vector match (expandable via memory id). The pool item carries
`memory_id`, `role`, `session_id`, `source_type`, `verification_state` (see B4)
and passage offsets. The trace records four stages separately: **retrieved,
selected, serialized, sent** (per request attempt, so a fallback shows which
endpoint saw what). `/why` reports *sent*, labelled honestly: transmission is
not proof the model relied on it. Recall rendering shows the speaker.

### B4. Minimal provenance, now (not the full graph)

The raw `memories` table already has `role` (`sqlite_store.py:54`), so "Jim
said" vs "the model said" exists at the source and is lost at rendering. The
derived layers - `core_memories`, `lessons`, user-model conclusions - have
`source_session_id` at best and no source type or verification state at all.
Add to each derived record and to the pool item:

```
source_type        user_statement | assistant_inference | tool_observation | reflection | import
source_id          memory/lesson/session id the record derives from
created_by         which job or path wrote it
verification_state stated | inferred | verified | contradicted
```

Rule this enforces: **an assistant inference never becomes equivalent to a
user assertion.** "Jim probably prefers X" (assistant_inference) and "I prefer
X" (user_statement) render differently and are weighted differently in D. This
is cheap now and unpleasant after another ten thousand memories. The linkage
graph (`supported_by`, `derived_from`, `supersedes`) is explicitly **not** built
here; B3's `source_id` is the hook if it is ever needed.

**Stage B gate (deterministic, runs here):** on the Stage C continuity set,
answer-bearing passage survival into *sent* and false-recall exclusion rate,
measured before and after B. Every source `/why` reports maps to the actual
request. Local fallback fits its window. If neither rate moves, stop.

---

## Stage C - Continuity set (the instrument for B and D)

A compact, frozen evaluation set that exercises the real ingest-to-answer path
against a throwaway DB, driven by `ScriptedLLMClient` for the deterministic
half. **This does not reopen retrieval tuning**: it measures assembly and
exclusion, not ranking. Extend `blipshell/benchmark/` (new job family
`continuity`) rather than building a second framework.

~25 seeded cases in two halves, each with a known answer-bearing sentence and
known distractors:

**Survival (right fact reaches the model)**
- fact buried past character 1,200 of a long message
- fact split across two sessions
- fact stated once, paraphrased three times by the assistant (paraphrases must
  not count as confirmation)
- exact recall of a number/date/path
- unknown answer requiring abstention
- fallback after a completed tool action (ties to A2)

**False recall (wrong fact stays out) - weighted at least as heavily**
- old preference explicitly corrected later; only the correction may appear
- two conflicting project states; newer wins, older is labelled superseded
- assistant speculation that must not surface as fact
- near-identical memories from two different projects; only the active one
- fact belonging to another session/person that must not leak
- stale imported external content that must show its age

Scoring: source recall, passage survival into *sent*, superseded/contradicted
exclusion, unsupported-claim count, duplicate side effects, tokens per correct
outcome. **Evaluation must not mutate production state**: search records
access counts (FadeMem reset) and revote touches lessons, so runs use a
disposable snapshot or an explicit non-mutating mode.

The model-quality half (same evidence packets, two models; then same model, two
assembly variants) runs over Tailscale. It separates "assembly lost it" from
"the model could not use it". Both can be true.

---

## Stage D - Accountable lessons

Lessons are a permanent context pool: the top 30 by importance ride in **every**
prompt (`core/agent_session.py:163` `_load_lessons`), and since 2026-09 their
importance is revoted nightly by a local model judging a session reflection
(`memory/lesson_revote.py`). Two problems with that as evidence: it is a model
judging a model's description of a model, and the lessons table
(`sqlite_store.py:78-88`) has **no usage or outcome columns at all**. Nobody
can answer "was this lesson present, was it applied, did it help" - three
separate questions.

### D1. Per-turn lesson selection

Because the top 30 are always present, "present" carries zero information.
Switch to Recall-style selection per turn from `vec_lessons` (it already
exists: `memory/vector_store.py:46`) plus a small always-on set (explicit
correction lessons, see D3). Now "selected" means "judged relevant to this
turn" and becomes a usable signal. Budget stays in the Lessons pool.

### D2. Outcome evidence, negative-only at first

New table `lesson_uses(lesson_id, session_id, turn_index, selected_by,
correction_followed, correction_similarity)`. Deterministic signals only:

- **Selected** this turn (D1).
- **Correction followed**: the guardrails correction detector
  (`core/guardrails.py:146` `detect_correction`) fired on the next user turn.
- **Attribution**: embed the correction text and weight each selected
  lesson's penalty by cosine similarity to it (one vector query, no LLM).
  Unrelated lessons take a near-zero hit; always-on lessons take none.

Absence of correction is **near-neutral**, not positive: a lesson can be
irrelevant, ignored, or accidentally followed. Positive evidence is not
manufactured here. Feed `correction_followed` into the existing revote as
CONTRADICTS evidence weighted above reflection judgments. Real causal
attribution ("which lesson drove the behaviour that got corrected") is a later
problem; do not block on it.

### D3. Two creation paths, two promotion rules

Lessons are created from two places with different trust:
`core/agent_chat.py:246` (correction detector -> anti-pattern lesson, i.e.
**Jim explicitly said so**) and `memory/processor.py:310` (session review ->
reflection lesson, i.e. **the model concluded so**). `added_by` already exists
(`sqlite_store.py:86`). Correction-derived lessons activate immediately and
join the always-on set. Reflection-derived lessons start as **candidates**: not
selectable until N sessions pass without a CONTRADICTS vote (N from config,
default 3). Demotion is to candidate, archive is to `importance` floor - never
delete; the revision history stays.

### D4. Lesson as scoped procedure (shape only)

Where a lesson is really a procedure ("when diagnosing a config path, compare
resolved file identities from two cwds; success = same inode"), store trigger,
scope, action, expected result, counterexample. This is the "remember what
worked, with enough detail to reuse it" thread from gpt6. Shape it in the
schema now; populate from correction lessons first (they have the most
concrete context). No promotion without a held-out replay (Stage C harness).

**Stage D experiment (the one worth running):** does a lesson's importance
trajectory under outcome evidence diverge from its trajectory under
reflection-only evidence? Run both revote channels in parallel for a month on
the live corpus, dry-run, and compare. If they never diverge, the outcome
signal is too weak and we learned that for the cost of a table.

---

## Stage E - Project dossier + decisions with conditions

### E1. Decisions as a memory type, with an exit

`memory_type='decision'`, structured fields in `metadata_json`: `decision`,
`constraint` (why), `revisit_when` (condition, not a date), `status`
(active | superseded | reopened), `project`, `source_type` (B4). **One module
owns every read and write** (`memory/decisions.py`) so moving to a first-class
table later is a one-file change - decisions have state transitions and
supersession that ordinary memories do not, and the generic abstraction should
not hold them forever. Searchable like any memory; folded into the dossier.
Created by a tool (`record_decision`) and by session review when it detects
"we decided / we rejected".

Experiment: simulate scenarios that bait the assistant into re-proposing a
rejected approach (`-c continuity`). Score re-proposals before/after, and
correct reopening when the scenario changes the constraint.

### E2. Digest -> dossier

`memory/project_digest.py` produces prose. Extend to sections: current
objective, accepted constraints, **last verified state** (tool observation,
not claim), rejected approaches (from E1), open questions (from `follow_ups`,
`sqlite_store.py:332`), next useful action, source ids. Distinguish planned /
attempted / completed / verified.

**Update is event-driven, nightly reconciles.** The digest already updates on
session close (`session/manager.py:355`). Add triggers: decision recorded,
`task_complete`, follow-up resolved, verification observed. The nightly job
compacts and reconciles active projects only (skip dormant); compare
preparation cost with later reuse before extending it.

### E3. Runbook memory (shape only)

Environment-specific procedures that actually worked: command, required state,
failure it recovered from. Same shape as D4; same promotion rule. Populate
opportunistically from `task_complete` turns that followed a `ToolFailure`.

**Stage E gate - the behavioural milestone (GPT's, adopted):**
> BlipShell resumes a project abandoned for two weeks and correctly states the
> goal, current state, last decision, blocker and next action - claiming
> nothing unverified.

Tested as simulate return-after-gap scenarios against a seeded DB; scored on
correct unfinished step, respected constraints, zero unverified completion
claims, and irrelevant interruptions. This gate sits **here**, not after
Stage B: A and B fix silent correctness, and a dossier milestone placed after
B would fail for the wrong reason and stop the plan early.

---

## Stage F - Bounded initiative (deferred)

A question queue for unknowns that block an active goal (source question,
project, expected decision benefit, evidence source, expiry, compute budget),
investigated one at a time in an authorized idle window, findings surfaced on
return. It is the dossier's open-questions section with an external
dependency added. **Not before E shows measured reuse.** Imported content is
evidence, never instruction; fetched time and source version are stored so old
knowledge looks old.

---

## Explicitly NOT doing

- Enabling the reranker (would degrade ranking as written - v2 note stands).
- A second graph or a graph database. The entity graph is bi-temporal already.
- Source penalties on assistant-authored memories (refuted by the Sep 2 readout).
  B4's *typing* of assistant inference is the fix; the numeric penalty is not.
- Model self-escalation ("do I need a bigger brain"). Observable triggers only,
  and not in v3.
- More reflection frequency, or turning the self-layer into task planning. The
  lingering-thought mechanism stays separate and its recurrence weights never
  touch factual retrieval (the firewall).
- The full provenance linkage graph up front. B4 is the minimal contract.
- The skill library / learning runner / transfer benchmark (the pivot). Wisp or
  a separate repo, after A-C, if at all.
- Coding-agent parity, MCP, autonomous prompt rewriting.

## How each stage is validated, and by whom

Everything below is automated. The only human steps are approving GPU time and
reading a readout. Nothing requires babysitting a run.

| Stage | What proves it | Runs where | Driven by |
|---|---|---|---|
| A | pytest: `ScriptedLLMClient` loop tests, real temp git repo, clean-venv import | dev box, seconds | Claude Code, every commit |
| A1 structured output | benchmark job `dedup_structured`, schema-validity rate | Ollama PC via Tailscale, minutes | Claude Code, one GPU run, ask first |
| B | pytest + the Stage C deterministic half (survival / false-recall rates) | dev box | Claude Code |
| C deterministic half | benchmark `continuity` job, scripted model, throwaway DB | dev box | Claude Code |
| C model half | same job with real models, two evidence variants | Ollama PC via Tailscale | Claude Code, ask first |
| D experiment | nightly dry-run of both revote channels for ~a month; `scripts/lesson_readout.py` prints divergence | Ollama PC nightly, unattended | nightly runner; anyone reads the script output |
| E gate | simulate `-c continuity` return-after-gap scenarios, seeded DB, scored | Ollama PC via Tailscale | Claude Code or any LLM with the repo |

Rule inherited from the harness work: **read transcripts before believing
scores** (the 2026-08 agent eval found four harness bugs that all penalised the
stronger models). Every new scorer ships with a unit test on a hand-written
transcript before it scores a real one.

## Branching

One short-lived branch per Stage A finding (`v3/a1-dedup-parser`, ...),
merged to `main` when the suite is green - the Ollama PC runs live off `main`,
and these are correctness fixes it should get promptly. Stage B is one branch
(`v3/b-context-contract`) because B1-B4 change the same code path and are not
independently shippable. C, D, E are one branch each. No long-lived `v3`
integration branch: the two-PC sync is `git pull`, and a branch the live box
never sees is a branch whose bugs surface a month late.

## Suggested cadence

Stage A is about a week of careful, test-first work and lands as one PR per
finding. Stage B is the largest single change (B1's `RecentHistory` coupling)
and should be preceded by the Stage C survival cases so its gate exists before
the change does. D and E each fit in a few days once B is in; D's experiment
then runs for a month in dry-run alongside the live nightly. The Tailscale
measurements (A1 structured output, C model-quality half) are short GPU runs -
ask first.

## Sources

- `BlipShell-system-review.pdf` - external review of `efcbe02`, 2026-09-08.
  Findings F1-F8 map to A2, A3, A1, B1, B2, B3, A4, A6 respectively.
- `blipshell review from gpt6.txt` - the lessons/learning conversation; the
  skill-library pivot is parked, outcome-backed lessons and decisions with
  conditions are adopted (D, E1).
- `output.txt` - review of this plan's first draft; adopted: negative-only
  outcome evidence with attribution, event-driven dossier, decisions with a
  migration exit, minimal provenance now, assistant-inference typing,
  correction-vs-reflection promotion rules, false-recall cases, the Stage E
  behavioural gate, structured dedup as an experiment with both paths kept.
- Research cited by the review and worth reading before the matching stage:
  LazyMem (B3, selective evidence construction), ExpeL (D, success/failure
  contrast), LongMemEval / LongMemEval-V2 (C categories; E3 workflow
  knowledge), Sleep-time Compute (E2 nightly refresh, measure before extending).
