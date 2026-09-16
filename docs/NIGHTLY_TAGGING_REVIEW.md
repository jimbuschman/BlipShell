# Nightly tagging investigation — 2026-09-15

The database was inspected read-only. No production repairs, tag assignments,
or model calls were run. Counts describe the local snapshot, not an assumed
live ingestion rate.

## Evidence

- 32,117 active memories; 17,012 meet the existing batch-tag pool predicate.
- 11,042 active memories have no tag links. All have summaries and
  `is_processed=1`; their message timestamps are before 2026. These timestamps
  do not establish when they were imported. The current schema cannot establish
  whether tags were never assigned or later removed. Most have no provenance
  metadata; 140 carry `imported_agent_log`.
- Only nine active memories have exactly the `neutral` tag. Neutral-only
  accumulation does not explain the current pool by itself.
- 681 active discovered patterns across 679 tag names supplement the static
  regex table. Pattern existence does not establish coverage or quality.
- The saved nightly report records centroid tagging checking 500 memories,
  tagging one, and adding four tags. Batch tagging checked 30, tagged 28,
  marked seven skip, and reported 17,001 remaining, with a 74.9-second average
  batch. These outcomes overlap: a row may receive one tag and also be skipped.
  Tag discovery produced six patterns during that run.

The evidence supports a large historical-message backlog and low observed
maintenance throughput. It does not prove an ongoing write-time arrival rate.
The current tag count grouped by message date cannot reconstruct that rate.

## Changes

- Keep the latest report for existing consumers, and retain the last 120 run
  records under `nightly_run_history` in app metadata. Each job checkpoints
  the run before and after execution. A terminated process leaves its last
  active job visible as `running`; this is not proof that the process is still
  alive. Skips due to an import lock are also recorded.
- Measure active, untagged, non-placeholder coverage, neutral-only, skip-marked,
  skipped low-coverage, and pending counts before and after each run. Record
  snapshots around centroid and batch tagging separately. Changes during a
  run can include concurrent writes and other maintenance; they are not
  labeled as a causal write-time arrival rate.
- Warn on backlog/coverage thresholds: at least 20 records and either 10% of
  the active corpus or 100 records. A declining queue does not erase a
  warning about skip-marked records with low coverage.
- Compare the last recorded snapshot on each UTC day. Warn on growth across
  four consecutive observed days; repeated loop passes are not four nights.
  Trend evidence begins with new runs; no historical trend is invented from
  the old single-report snapshot. Detailed history is bounded by runs, so
  intensive looping may shorten the available date range.
- Correct the audit severity mismatch (`warn`/`error` versus uppercase checks)
  so health findings reach the saved report and startup notification.
- Keep a tag-write failure retryable rather than assigning `_skip`. Count
  failed records and records with no valid model assignment separately from
  assigned tags and skip markers. No valid assignment is not assumed to mean
  an explicit model refusal; it can include an unusable reply.
- Prevent centroid tagging from propagating `_skip`, `neutral`, or known junk
  labels. Choose the strongest similarity matches before the five-tag cap,
  rather than the first matching labels in vocabulary order.
- Derive the batch tagger's inner time allowance from its effective outer job
  timeout. No timeout or batch-size increase is included in this change.

## Inspection

`blipshell nightly --history` reads history without initializing a runner or
calling a model. Add `--quiet` for JSON containing per-job counts and stage
snapshots. Before the first new nightly run, history is empty.

The normal database audit now reports both the queue and coverage metrics.
The latest nightly report retains them too. Use several observed runs to
compare pending counts, useful coverage, skip outcomes, and stage throughput
before deciding whether to change tagging policy or allocate more GPU time.

Tag counts are a coverage proxy, not proof that the assigned topics are
correct. Semantic quality still requires reviewing a representative sample.

## Validation

The full local suite passed: 2,741 passed, three skipped (optional Presidio
tests). Regression coverage includes retained/checkpointed history, independent
UTC-day trends, placeholder-versus-queue counts, lowercase audit severities,
retryable tag-write failures, centroid marker exclusion and similarity ordering,
budget propagation, and a read-only history CLI that cannot initialize models.
Targeted tests were rerun after final review adjustments. The test environment
uses an isolated `.venv-review` directory; no production database was modified.
