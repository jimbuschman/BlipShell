# Isolated system acceptance — 2026-09-16

Status: **acceptance checks executed; functional recovery verified, load issues remain**.

## Isolation and reproducibility

`scripts/acceptance_blipshell.py` creates an online SQLite snapshot of the configured
database, then separate full copies for nightly and Agent lifecycle tests. This
run uses `data/acceptance_20260916_003029/`. Private contents and detailed results
stay in that ignored directory; `events.jsonl` records each check. The manifest
contains the source path, source SHA-256, and snapshot row counts.

The snapshot contains 42,564 memories, 1,841 sessions, 813 lessons, 12 core memories,
and 36,734 entities. Production writes are not part of this acceptance pass.

Model calls use the configured localhost Ollama service and installed local
models. Cloud endpoints are removed from the test configs. Backup, benchmark,
temporary, tokenizer, model-cache, and copied project export paths are redirected
into the run folder.
A Python audit hook blocks outside writes, outside SQLite connections, subprocess
execution, and non-loopback connections in the test child. It is an additional
test guard, not an operating-system sandbox.

Agent tests disable Telegram, robotics, idle reflection, and competing startup
entity/tag-discovery work. Nightly jobs use their configured eligibility and
feature flags. A disabled job or a zero-work result is not evidence that its
processing path was exercised. Cloud-provider operation and device integrations
are outside this local acceptance pass.

## Results

- Actual backup and restore of the full snapshot: row counts preserved, SQLite
  integrity and foreign keys passed, restored store reopened successfully.
- Full FTS token/content integrity check on the restored copy passed. All 42,564
  memories also have matching FTS document IDs.
- Final full regression rerun: **2,778 passed, 3 skipped** (381 seconds), including
  backup, FTS membership, batch-deadline, and registered/unregistered project
  close cases. All three skips explicitly report Presidio not installed.
- Real Agent startup, real web authentication routes, and approval denial passed.
  Web checks used in-process ASGI transport, not a browser or deployed server.
- Three real-model chat turns passed; persisted memories were processed and
  retrieved through long-term search. Compaction preserved surviving database IDs.
- A fresh Agent resumed the interrupted session and recalled the release code.
- After the project-label fix, session close completed in **303.39 seconds**.
  Bookkeeping, message saving, summary, and lessons reported success; the
  unregistered-project digest was explicitly skipped. All eight raw messages
  remained present and processed, with clean integrity and foreign-key checks.
- All 27 nightly job entry points were invoked on the full copy. Actual work
  included lesson generation/scoring, entity extraction/cleanup, centroid tagging,
  model tagging, consolidation checks, tag discovery, user-model update, reflection,
  and exports. Cleanup/backfill paths with no eligible work did not exercise new
  processing. Lesson revoting, entity merging/pruning, and memory pruning were
  disabled by the real configuration. Digest rebuild found eight current digests;
  their exports were checked in redirected project folders.
- The corrected full-corpus tagging run completed in **263.2 seconds**: 20 checked,
  20 tagged, 44 tags assigned, zero failures, one skip marker. It stopped at its
  budget and reported **16,379 pending** in the test copy; that is not a claim
  that every checked memory acquired multiple useful tags.
- A real connection-refused outage preserved retryable work. A forced 0.3-second
  live batch deadline cancelled the batch without skip-marking it; reconnecting
  to the real local model then tagged the same memory successfully.
- Forced termination of the actual test Python worker preserved its committed
  input. Reopening passed SQLite integrity and foreign-key checks. A fresh
  process then tagged that exact pending memory successfully using the real model.
- The production database fingerprint remained unchanged through the final
  verification. Its WAL was empty. All acceptance Python test processes finished;
  no test runner was left operating on the copies.

The first nightly trial was invalidated by harness compatibility defects (Windows
null device handling and byte-string localhost addresses). The retry starts from
a pristine copy, `nightly_retry.db`. Those initial failures are not application
failures. A public tokenizer vocabulary was pre-cached before installing the
network guard; no corpus content was included in that download. The first
conversation trial also rejected a correctly recalled identifier solely because
the model used a Unicode non-breaking hyphen. The corrected check accepts
typographic hyphens while requiring the same name and number; the lifecycle was
repeated on a fresh snapshot.

## Changes discovered during acceptance

1. Added optional `database.backup_dir`, anchored relative to the config file,
   so startup and nightly backups can be isolated with the selected database.
   Leaving it unset preserves the existing default location.
2. Nightly backup now requests quiet output. The previous Unicode console status
   message could raise an encoding error under redirected Windows output.
3. A tagging pass that finishes its last batch exactly at the budget boundary
   now reports drained instead of simultaneously reporting an early stop.
4. The FTS audit now compares actual indexed document IDs with all memory IDs.
   The old external-content table count could both report a false mismatch for
   unsummarized memories and hide a real missing index entry. Membership checks
   are explicitly distinguished from token-level integrity.
5. The full-corpus tagging pass reproduced an outer 300-second timeout after
   earlier tags had already been committed. Checking the budget only between
   batches did not constrain a suddenly slow model call. Each in-flight batch
   now has the remaining deadline; cancellation returns completed-batch metrics,
   an interrupted-batch count, and a fresh remaining-pool measurement. Pending
   work remains retryable. The full-corpus run and a forced live deadline both
   passed revalidation.
6. Session close exceeded the harness's 15-minute limit while bootstrapping 40
   broadly matching memories for an unregistered project label. The digest
   writer cannot save a digest without a project row. Session close now skips
   that unsavable work before calling the models. Registered-project behavior
   remains intact. Deliberate skipped close steps no longer count as failures.
   Both branches have regression coverage; live recovery and close passed.

## Remaining live findings

The conversation run reproduced embedding timeouts while background entity
extraction was active. Memory search fell back to keyword search; core-memory and
lesson retrieval logged failures for that turn. The conversation still recalled
the requested fact, so a successful reply alone would have hidden this degraded
retrieval. Background idle extraction also delayed the newly queued conversation
messages. The initial 30-second worker-shutdown wait was exceeded on the resume
run, although the queue subsequently drained and close completed without losing
messages. This performance/resource-scheduling issue is **not cleared** by
passing unit tests or the tagging deadline fix. Its precise resource cause has
not been isolated.

The final health check had no integrity errors but four warnings: 8,174 orphaned
entities, 16,379 pending tags, 10,679 memories without non-placeholder tags, and
247 skip-marked memories with low tag coverage. These counts describe the test
copy. They require an explicit cleanup/coverage policy and sustained throughput;
finishing one nightly run does not clear them. At this run's observed rate, the
tagger estimated roughly 60 hours for its remaining pool; that is a rough runtime
estimate, not a completion promise or a measurement of production inflow.

This runtime reports regex-only privacy filtering (`presidio_analyzer` absent).
Private-copy calls are restricted to localhost. Cloud NER sanitization is not
certified by these runs.

## Follow-up 2026-09-16: scheduling under load

Root cause of the retrieval degradation above, traced in code and then shown
by the gate event log of a live run. (1) `VectorStore.search_memories` embedded
the query WITHOUT the model gate (the method was documented "NOT gated"), so
the turn's embeddings ran on the GPU alongside the worker's entity-extraction
generations and timed out. (2) The worker judged the system idle from its own
queue alone, so idle extraction started while a chat turn was generating.
(3) `loop.run_in_executor` does not copy contextvars, so an embedding requested
by a background caller reached the gate as interactive.

Change: `_embed`/`_embed_batch` take the gate; `Agent.chat` holds
`OllamaGate.interactive_turn()` for the whole turn, during which new background
acquisitions park (a running call is never interrupted) and are woken when the
turn ends; priority is a contextvar set by `background_model_work` /
`interactive_model_work`, and every embedding thread hop is
`asyncio.to_thread`; the worker's idle branch checks `interactive_active`.
Files: `llm/ollama_gate.py`, `memory/vector_store.py`, `memory/worker.py`,
`memory/search.py`, `core/agent_chat.py`, `core/nightly.py`, `core/agent.py`.

Tests: `tests/test_gate_scheduling.py` (20) and
`tests/test_memory_worker.py::TestIdleExtractionDefersToChat` (1); real
threads and event loops, only the embedding client faked. Full suite after
the change: **2,799 passed, 3 skipped**.

Live check (`scripts/validate_live_scheduling.py`, driven from the dev box
against the Ollama PC, on a COPY of the dev-box database: 64 memories, all
awaiting extraction; not the acceptance snapshot). Local models only:
`gpt-oss:latest` chat, `qwen3:14b` extraction, `qwen3-embedding:0.6b`.

| Observation | Turn 1 | Turn 2 |
|---|---|---|
| Background calls granted after the turn began | 0 | 0 |
| Background request parked during the turn | 53.2 s, granted at turn end | 38.3 s, granted at turn end |
| The turn's own gate waits (max) | 1.5 s (three concurrent embeddings serialising) | 24.1 s (in-flight background call from the zero-length gap between turns) |
| Semantic hits / keyword fallback | 30 / none | 15 / none |
| Planted fact | acknowledged | recalled correctly |
| Turn latency | 53 s | 59 s |

Log counts for "Semantic search unavailable", "Core memory search failed",
"Lesson search failed" and timeouts: all zero. Background extraction resumed
within 3 s of the last turn ending, with zero gate wait.

Not cleared by this. The 24 s first-call wait in turn 2 is the no-preemption
design: a background generation that starts between turns runs to completion.
Worker shutdown still waited past its timeout with a call in flight (the
separate "improve shutdown" item). The gate is process-local, so a separate
`blipshell nightly` process is not scheduled by it. The acceptance `lifecycle`
stage itself has not been rerun on the Ollama PC since this change; the
scheduling script above is the isolated-copy check that was run.

## Follow-up 2026-09-16: bounded shutdown

The "initial 30-second worker-shutdown wait was exceeded" above had two
causes in `MemoryWorker.shutdown`: the SHUTDOWN item was appended AFTER the
queue, so close drained every queued message (several model calls each, 15 s
per item budgeted, no upper bound), and an in-flight item (an entity
extraction batch of up to ten memories) could not be stopped at all. A worker
stuck in a model call also kept the interpreter alive, because its aiosqlite
connection thread is not a daemon.

Change (`memory/worker.py`, `core/agent.py`): shutdown takes no further queue
items; the remainder is deferred to the existing startup sweep, which is
sound because every queued item is durable already - a `PROCESS_MESSAGE`
names a raw row with `is_processed=0`, unextracted memories are re-found - and
the one exception (a message whose raw persist never landed) is persisted raw
at deferral. The in-flight item gets the grace period (`WORKER_CLOSE_GRACE`,
30 s at session close; 5 s in `force_cleanup`), then is cancelled via its own
loop and left retryable (an interrupted message stays `is_processed=0`, an
interrupted extraction stays unmarked). `shutdown()` returns a report
(`deferred` by kind, `interrupted`, `exited`) and close shows it to the user.

Tests: `tests/test_memory_worker.py::TestBoundedShutdown` (6, real worker
thread): deferral instead of draining, raw persistence of the non-durable
case, grace period honoured, cancellation of a worker parked behind an open
chat turn, cancellation of an in-flight idle extraction, and the report text.
The pipeline test that asserted a full drain now asserts the invariant that
survives: every enqueued message has a row, processed or raw.

Live (second `validate_live_scheduling` run, same setup as above): cleanup
was called with an entity-extraction batch in flight and four messages queued.
The worker stopped in 5.0 s: the extraction was cancelled and left retryable,
the four messages were deferred to the startup sweep, and the earlier "Memory
worker did not exit within 5s" warning did not recur. Scheduling verdict on
that run: PASS again (0 background grants after either turn began).

Review closures (same day). (1) The in-flight item was the one place a
row-less message could still be lost: a raw-persist failure followed by a
shutdown that cancelled the pipeline before it created a row. The worker now
persists such a message raw before it enters the cancellable task and
processes it as an update to that row. (2) Cancelling the awaiting task does
not stop a blocking call already running on a pool thread (an embedding HTTP
call, a vector write under the gate). The worker's loop now waits for its
executor threads before exiting, so the thread stays alive exactly while such
a call runs; the shutdown report says so (`executor_busy`), and neither close
path closes the vector store over a live worker. Tests:
`TestShutdownReviewFindings` (3), one of them driving a real
`asyncio.to_thread` vector write that blocks until released.

Second review (same day). Durability was not yet a hard boundary: when the
worker-side raw persist itself failed, the item still entered the cancellable
pipeline row-less. Now it is requeued with a backoff and never processed
without a row; a message still row-less at shutdown is logged in full at
ERROR as lost rather than counted as deferred. And a shutdown that began while
the durability write was in progress issued a cancel into an empty slot, so
processing could start after shutdown had begun; the loop re-checks the flag
after the write and defers the now-durable item instead. Tests:
`TestDurabilityBoundary` (2), one with a failing worker-side persist, one with
the shutdown timed inside the write.

Third review (same day). A message whose final persist failed was still
counted as "deferred", and so was noise, although neither is recoverable by
the sweep (one is lost, the other dropped by design). The report now carries
`deferred`, `lost` and `discarded` separately and the close output shows
them. The reviewer also asked whether a hang in that final persist (not
cancellable) is worth handling: the probe test shows shutdown() returning
`exited=False` with the report naming the wait (`finalizing`) and confirming
no outcome until the write returns, after which the worker exits and the
item is deferred normally. `Agent.end_session` then proceeds into summary and
lessons with the worker thread alive. Left as is: a single INSERT hanging
past SQLite's 60 s busy timeout means the database is unavailable to those
close steps as well, and the report says what is happening.

Not changed: the session's own close steps (summary, lessons, digest) still
run to completion with no timeout, per the standing design decision. Whether
those should also be deferred to the nightly is a separate decision.

## Real-corpus runs 2026-09-16 (Ollama PC, after the scheduling and shutdown fixes)

Both stages ran on the Ollama PC against fresh copies of the production
database (`data/acceptance_20260916_192346`, `data/live_scheduling_20260916_194004`).

**Lifecycle: PASS on every check.** Startup, web auth routes, tool approval
denial, three chat turns (65 s, 13 s, 55 s), persist-and-recall, compaction
identity, session close and restart-and-resume. Session close took **94 s**
(was 303 s in the first acceptance run). One `isolation_violation: BLOCKED`
event during agent start is the guard refusing a write outside the run folder;
the run continued and passed.

**Scheduling: the gate passed; the validator reported a false FAIL.** The two
"background call(s) granted after the turn began" were:

| | requested | turn open at request | granted | parked |
|---|---|---|---|---|
| turn 1 | 20.781 s | yes | 79.531 s (= turn 1 end) | 58.75 s |
| turn 2 | 93.922 s | yes | 140.781 s (= turn 2 end) | 46.86 s |

Both background requests were parked for the whole turn and woken as the turn
closed. Semantic search stayed alive (59 vector hits per turn, no keyword
fallback, no search failures, no timeouts), the planted fact was recalled,
background work resumed, and the gate ended with zero waiters and zero
cancels. Turn 1's own gate waits peaked at 0.84 s; turn 2 waited 18 s behind
the call released at turn 1's end (the no-preemption cost, as designed; the
turns were back-to-back).

The validator judged "granted after the turn began" by `t_grant <= t_end`,
but `t_end` is stamped after `agent.chat()` returns and the
`interactive_model_work` wrapper closes the turn (waking the parked waiter)
just before that return, so the grant is recorded a few ms early.
`scripts/validate_live_scheduling.py` now records the gate's own
`interactive_active` at grant time and fails only when a BACKGROUND
acquisition is granted while a turn is open; released-at-turn-end calls are
reported separately with their parked time, and a `--turn-gap` (default 3 s)
separates the turns like a user would. `tests/test_validate_live_scheduling_verdict.py`
replays the recorded events: pass under the corrected rule, and shows a
genuine mid-turn grant is still a miss. The scheduler was not changed. A
re-run on the Ollama PC with the corrected validator is the remaining step.

## Commands

Use the development environment with dependencies installed:

```powershell
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py prepare
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py restore --root <run-folder>
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py nightly --root <run-folder>
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py lifecycle --root <run-folder>
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py outage --root <run-folder>
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py crash --root <run-folder>
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py recover_crash --root <run-folder>
.venv-review\Scripts\python.exe scripts/acceptance_blipshell.py verify_source --root <run-folder>
```

Run model stages sequentially. The crash stage owns and forcibly stops only its
synthetic-fixture child process. Inspect event statuses: a nightly process exit
code of zero means the orchestration finished, not that every job passed.
If lifecycle close is interrupted, the `resume` stage independently reopens its
saved session, checks recall, and retries close while recording each step's result.
