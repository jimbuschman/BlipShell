# Live validation — September 15, 2026

## Actual model checks

Ollama 0.33.3 was reachable at the configured local address and idle before the checks. No BlipShell process was running. The tests used real Ollama model calls through BlipShell's router, nightly runner and chat loop; these were not mock responses.

The configured corpus was opened read-only. Twenty pending summaries and the current 159-tag vocabulary were copied to `data/live_validation_20260915_185711/sample.db`. All tagging, history and report writes went to that scratch database. No private sample text was sent to a cloud model. No production memory rows or configuration were changed.

| Check | Observed result |
|---|---|
| Initial local request | `qwen3:14b` returned the expected response in 16.094 seconds, including initial loading |
| Interrupted nightly | Cancelled during a real model request after three seconds; history retained `running` / `batch_tag`, with all 20 rows still pending |
| Reopen and retry | Reopened the scratch database and reran nightly: 20 examined, 20 tagged, 59 tag assignments, zero failures, zero skip markers |
| Backlog movement | Scratch pending count fell from 20 to zero; all ten initially untagged rows acquired tags |
| Nightly duration | 237.3 seconds for two batches, including competing foreground requests |
| Same-process foreground request during tagging | Succeeded, but took 94.610 seconds including gate wait |
| Separate-process BlipShell ChatLoop during tagging | Streamed the expected response in 75.531 seconds using `qwen3:14b` |
| Repeat drained job | Zero examined rows and zero model calls |
| Embeddings | `qwen3-embedding:0.6b` returned a 1,024-dimensional embedding in 7.25 seconds |
| Scratch database integrity | `PRAGMA quick_check` returned `ok` |
| Saved history | One interrupted checkpoint followed by two completed runs |

The current production snapshot had 32,121 active memories and 16,475 pending tagging rows. These were read-only counts. The sample is the next twenty pending records, not a random or representative quality benchmark. A spot check found plausible broad topic tags, but some labels were coarse; tag presence does not prove semantic accuracy.

## Additional routing defect found and corrected

With the real configuration and local mode enabled, endpoint selection chose `local`, but chat's default model was still `gemma4:31b-cloud`. Coding similarly retained the cloud-configured `minimax/minimax-m3` name. A local Ollama URL alone therefore did not establish local inference. This was verified through actual configuration resolution without sending a private prompt to those models.

The fix checks model names as well as endpoint policy. Known models configured on sanitizing endpoints and explicit `:cloud` / `-cloud` names cannot be used in local mode. The router, Agent chat selection and standalone executor now select a configured local fallback or fail closed. The router rechecks the model before generation as well.

After the fix, the real configuration resolved chat and coding to `gpt-oss:latest` on `local`, while ranking remained `qwen3:14b`. A real BlipShell ChatLoop request through that corrected chat selection returned the expected response in **28.891 seconds**. No configuration edit was necessary.

## What the checks do and do not establish

The tagging pipeline can make progress, persist an interrupted checkpoint, resume, finish a finite sample and avoid reprocessing it. Local embeddings and the chat loop also work. This is stronger evidence than the earlier unit tests alone.

Responsiveness under concurrent tagging remains poor: both foreground probes waited over a minute. The gate is process-local and does not preempt a model request already running. Keep bulk catch-up separate from interactive use when responsiveness matters. These timings include competing work and are not an unloaded throughput forecast for the full corpus.

The cancellation test was graceful coroutine cancellation followed by reopening the database, not a forced process kill or power-loss drill. It interrupted the first batch before tag writes completed; it does not prove every possible mid-write recovery case. This run did not initialize the full Agent UI, execute live mutation tools, call cloud providers, test physical integrations, or perform a backup restore.

The job reported `stopped_early: true` when its time-budget check ran immediately after draining the sample. The same result correctly reported `remaining_pool: 0`; the next run confirmed no work remained. This is a reporting inconsistency, not lost work, and should not be interpreted as an incomplete corpus repair on its own.

The live helper explicitly disabled retries/caching and used a 180-second model timeout to keep individual attempts observable and bounded. The normal configuration uses 300 seconds and two retries; it was not changed. Raw metrics remain in the ignored data directory; this report intentionally excludes private summary text. A final read-only check found the production memory totals and all reported tagging counts unchanged.

## Reproduction and validation

Run `scripts/validate_live_nightly.py` with the development Python environment to create another isolated twenty-row sample and repeat the bounded local checks. Each run creates its own scratch directory. It uses real local inference, so it should run only when that load is appropriate.

The broad run completed with **2,759 passed, 3 skipped and 9 failures** in 5 minutes 14 seconds. All nine failures came from two older test fixtures: one unconstrained mock had evaluated as local mode enabled, and one handwritten endpoint manager lacked the field. Both fixtures were corrected to explicitly set `local_only=False`, preserving their intended cloud-enabled scenarios.

After those corrections, **111 focused tests passed**, including all nine previously failing cases, plus routing, privacy, endpoint retry, executor behavior, vision and the audit regressions. There are no known remaining test failures. The broad suite was not repeated after the fixture-only corrections; the passing focused run is the final verification for those changes.
