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
