# BlipShell system audit — September 15, 2026

## Implementation update

Subsequent bounded checks against real local models are recorded in
[the live validation report](LIVE_VALIDATION_2026_09_15.md). They confirmed
tagging progress, graceful interruption/retry, embeddings and chat responses;
they also exposed substantial chat latency under tagging load and an additional
cloud-model selection gap, which was corrected. The original deferred-live-test
statements below describe the earlier source-audit phase.

After the user stopped the nightly loop, fixes were implemented for the eleven findings below. The numbered findings describe the original defects, not the current implementation. Regression coverage is in `tests/test_system_audit_fixes.py`; `scripts/audit_system_probes.py` now runs those tests rather than asserting that the defects exist.

- Compaction preserves surviving memory IDs, processed flags and history offsets; it rejects blank summaries or changes made during summarization, and defers if the excluded messages have not entered memory processing. Resuming a session also clears previous session state.
- The foreground and memory worker share a thread-safe local-mode flag. Explicit endpoint targets use the router, including sanitization, NER restrictions and endpoint-specific models. The router rechecks policy immediately before generation, including after waiting for the local model gate.
- Nonzero shell exits report failure, including truncated output and background completion. Background stdout/stderr drain continuously with a bounded 1 MiB tail per stream.
- Required tool approvals are installed at Agent construction. Interfaces without an approval callback deny those tools unless automatic approval was explicitly configured.
- Empty authentication keys fail closed when authentication is enabled. The web server grants an exclusive conversation lease; a second WebSocket or nonstreaming API conversation receives a busy response rather than changing the active session. API requests replay their supplied history in fresh session state. This is intentionally single-conversation operation, not concurrent multi-user hosting.
- Streaming API failures produce an error rather than a successful stop, and closing the stream cancels and collects its chat task.
- Vector search failures allow keyword retrieval to continue.
- Local and remote background runners use atomic claims. Progress/completion requires the claim token and an active claimed/running state, so foreign and late results are rejected.
- Nightly history uses one atomic SQLite JSON update, preserving separate run records across connections without a process-local lock.

The full suite passed with **2,762 passed and 3 skipped** in 4 minutes 20 seconds. Additional worker-claim changes were made while that suite was running; the final targeted run passed **135 tests**, covering the audit fixes, nightly history, session management, routing/privacy, memory worker and tool failure/approval handling. The warnings were FastAPI's existing startup/shutdown API deprecation. Production memories were not changed; tests used temporary databases and fake model clients.

### Restart and compatibility

Restart BlipShell from this checkout to load the changes. The existing `blipshell nightly --job batch_tag --local --loop` command remains valid. The first normal database initialization adds a nullable `background_tasks.claim_token` column through the existing migration mechanism; no memory rows are rewritten. If using a separate remote worker daemon, update/restart it together with the main server because completion/progress now requires the returned claim token. No commit or push was performed.

Real model throughput, server contention, physical integrations and a historical impact assessment remain outside this validation. The process-local Ollama gate still does not coordinate separate CLI processes. This change does not prove that every previous nightly incident had the same cause, or that historical compaction affected this corpus.

## Scope and evidence

Reviewed HEAD `9e358be` plus the existing uncommitted nightly/tagging changes. This is a broad source and failure-path audit, not a claim that every feature or deployment has been certified. The user was running `blipshell nightly --job batch_tag --local --loop` from this checkout. This audit did not change runtime source, open the live database for writes, start a model, launch a shell subprocess through BlipShell, or run GPU/load tests.

Eleven diagnostic scenarios reproduced their respective defects with synthetic objects and mocked clients. Run them from the repository using:

```powershell
.venv-review\Scripts\python.exe scripts/audit_system_probes.py
```

The script prints evidence; `reproduced: true` means a defect is present. These are diagnostic probes, not acceptance tests. The web probes invoke actual route functions without starting the application. The local-toggle probe demonstrates the command's state mutation; the separate worker routing context was verified by source inspection. No probe establishes that these issues already damaged the live corpus or sent real data to a cloud endpoint.

Earlier validation of the nightly/tagging work completed with 2,741 tests passing and 3 skipped, followed by 39 targeted tests after the final changes. That is baseline evidence from the earlier work, not a full-suite run performed again for this audit. The eleven new probes demonstrate gaps despite that passing suite.

## Findings and required fixes

### 1. P1 — Manual compaction leaves memory IDs attached to the wrong messages

Locations: `blipshell/core/agent.py:1290`, `blipshell/core/agent_background.py:47`, `blipshell/session/manager.py:225`, `blipshell/memory/processor.py:439`.

Compaction replaces the message array with a summary and the last four messages. It resets dumped indices to `{0}` but does not remap `_memory_db_ids`. The background queue subsequently looks up IDs using the new positions. The processor trusts that ID and updates its summary/vector, without checking that the raw content belongs to the supplied text.

Reproduction: six already-processed messages had IDs 100–105. After compaction, `message2` was queued against ID 101 rather than 102, and `message5` against 104 rather than 105. All four were unnecessarily requeued. This can corrupt the relationship between raw text, summary, and embedding; noise/skip handling can also archive the wrong row.

Fix: put compaction behind a SessionManager operation. Settle pending persistence first, preserve stable message identity, remap every index-based structure and retain processed state for surviving messages. Reject blank summaries before replacing history. Test pending persistence, processed and unprocessed survivors, subsequent appends, and session-close processing. Do not repair historical records solely by guessing which sessions were compacted.

### 2. P1 — `/local on` does not update the memory worker's routing

Locations: `blipshell/ui/command_handlers.py:256`, `blipshell/memory/worker.py:125`.

The command changes the foreground endpoint manager and promises that both chat and background calls stay local. The worker constructs and retains a separate manager/router from startup configuration. Changing the foreground manager does not change that worker. This matters when startup permits cloud routing and the user later enables local mode.

Reproduction: invoking the actual command changed the foreground flag to true while the separate worker flag stayed false. Worker ownership of a separate manager was verified in `_run`.

Fix: provide a thread-safe runtime privacy policy shared by all routing contexts, or an explicit worker policy-update protocol. Define what happens to queued and already-started requests; a toggle cannot recall data already sent. Test switching after worker startup and queued work. This finding is about the interactive toggle, not evidence that the current nightly `--local` invocation bypasses its startup policy.

### 3. P1 — Targeted background tasks bypass router privacy checks

Location: `blipshell/core/background.py:137`.

When a target endpoint is specified, `_run_task` selects an enabled endpoint directly and invokes its client. That bypasses local-only filtering, sanitization and the NER gate in the router. A target can also come from automatic worker configuration, not just an explicit user request.

Reproduction: a synthetic cloud client's `generate` was called once with the original prompt while the endpoint manager had `local_only=True`.

Fix: route targeted tasks through a router API that constrains endpoint selection while retaining all policy checks. Reject incompatible targets. Test local-only, PII sanitization, unavailable NER, disabled targets and endpoint-specific model selection. Do not duplicate privacy logic inside the background manager.

### 4. P1 — Web connections share mutable conversation state

Locations: `blipshell/ui/web/app.py:132`, `blipshell/ui/web/app.py:155`, `blipshell/ui/web/app.py:464`, `blipshell/session/manager.py:77`.

Each WebSocket records a separate session ID, but every connection calls the same global Agent. Starting another session changes that Agent's session manager. The stored per-socket ID is not used to select state when handling chat. The compatible API also uses one global session ID and shares the same Agent.

Reproduction: connection one received session 1; after connection two started session 2, connection one's message was handled under session 2. This can mix histories and persist messages to the wrong session. With multiple users, it can expose conversation context across users.

Fix: isolate per-conversation Agent/session state and serialize turns within each conversation. Define API conversation identity explicitly. A lock around `chat` alone is insufficient because another connection has already replaced the active state. Test two interleaved WebSockets and mixed WebSocket/API traffic.

### 5. P1 — Enabled authentication silently allows access when the key is empty

Locations: `blipshell/ui/web/app.py:65`, `blipshell/ui/web/app.py:116`.

HTTP auth explicitly returns success for an empty key; WebSocket auth also skips validation. Enabling auth without configuring the key therefore leaves the service open. This is configuration-dependent; it does not establish that the user's service is currently exposed.

Reproduction: `verify_auth(None)` accepted a request with `AuthConfig(enabled=True, api_key='')`.

Fix: reject this configuration at startup and fail closed in both request paths. Preserve intentionally disabled auth as a separate supported setting. Cover missing, empty, incorrect and correct credentials.

### 6. P1 — Web/Telegram do not install the configured tool approval policy

Locations: `blipshell/core/tools/base.py:213`, `blipshell/core/agent_tools.py:71`, `blipshell/ui/web/app.py:81`; compare `blipshell/ui/cli.py:578`.

The registry checks approval only when a callback exists. The CLI installs that callback based on configuration; the web and Telegram entry points do not. Agent initialization still registers tools such as file writing and shell execution. Thus `auto_approve_tools=False` does not enforce the expected confirmation behavior in those interfaces.

Reproduction: a synthetic tool listed as requiring approval executed successfully without a callback. Entry-point wiring was verified by source inspection; no real command or file operation was executed.

Fix: install policy centrally, distinguish intentional auto-approval from an unavailable approval mechanism, and deny protected operations when confirmation is required but unavailable. Implement a confirmation UI or return an actionable denial in each headless interface. Test actual entry-point policy wiring as well as registry behavior.

### 7. P1 — Failed shell commands can be reported as successful tools

Location: `blipshell/core/tools/shell.py:177`.

Nonzero exits append an exit-code line but return a plain string. The registry's failure detection does not classify that suffix as failure. A failing test/build with ordinary stdout can consequently produce `ToolResult.success=True`. This undermines automated verification and error recovery.

Reproduction: a mocked command exited 7 and printed `tests failed`; the registry reported success. No subprocess was launched.

Fix: preserve the exit outcome using `ToolFailure` for every nonzero return code, including after truncation. Apply the same semantics to completed background processes. Keep the existing protection for a successful command whose output happens to start with `Error:`. Test zero/nonzero codes, empty output, stderr and truncation.

### 8. P2 — Embedding failures prevent available keyword search

Location: `blipshell/memory/search.py:224–260`.

Semantic search runs before FTS and its exceptions propagate out of `search`. If embeddings are unavailable, keyword search is never attempted even though the local database may be healthy.

Reproduction: a synthetic embedding exception escaped and `search_fts` was called zero times.

Fix: treat retrieval sources as independently fallible. Continue with FTS when vector retrieval fails, retain available candidates when only one vector pass fails, and expose degraded-search diagnostics. Preserve archive, supersession and project filtering. Test vector failure with useful keyword matches, partial project-pass failure and failure of all sources.

### 9. P2 — Worker task claims are not atomic

Location: `blipshell/ui/web/app.py:412`.

Claiming performs a read of pending state followed by an unconditional update. Two workers can both observe pending and both receive a successful claim, allowing duplicate work.

Reproduction: synchronized synthetic reads returned pending to two concurrent requests; both returned `status: claimed`.

Fix: use an atomic conditional update and check the affected-row count. Include ownership/claim tokens in progress and completion checks so stale workers cannot overwrite a newer claim. Test concurrent claims with separate SQLite connections, plus cancellation/reclaim behavior.

### 10. P2 — Streaming API converts chat exceptions into normal completion

Location: `blipshell/ui/web/app.py:532–571`.

The stream creates a chat task but never awaits its result. The task sets `done` in `finally`, so exceptions still produce `finish_reason: stop` and `[DONE]`. Clients see an apparently successful answer, possibly empty, instead of the failure.

Reproduction: a synthetic chat exception was unhandled while the response emitted normal completion.

Fix: collect the task outcome, use an explicit stream error path, and always cancel/await the task when the response generator closes. Add tests for exceptions before/after tokens and disconnects during a turn. Source inspection also found missing cancellation cleanup; disconnect behavior was not exercised against a real server in this audit.

### 11. P2 — New nightly history can lose records during overlapping runs

Location: `blipshell/core/nightly_history.py:22` — introduced in the current uncommitted work, not Claude's HEAD.

History is one JSON value updated through read/modify/write. Two processes can read the same old value and overwrite one another's entries. Checkpoints from a long-running invocation can also race another run. This affects monitoring history, not the underlying memory records.

Reproduction: two concurrent synthetic `save_run` calls with different start times left only one record.

Fix: preferably store runs as separate rows with unique run IDs and atomic upserts, plus a retention operation. Alternatively use a properly serialized database transaction for the entire read/modify/write. A process-local Python lock is insufficient for separate CLI processes. Test independent SQLite connections updating distinct runs and the same run. Apply this before treating the new history as reliable under overlapping invocations.

## Additional source-level concern

Background shell processes use stdout/stderr pipes, but `CheckProcessTool` reads them only after the process exits (`blipshell/core/tools/shell.py:150`, `:305`). A sufficiently chatty child can block on a full pipe before exiting. Add continuous bounded draining or file-backed output, and test it in an isolated subprocess environment. This was not counted among the eleven reproduced scenarios because no subprocess/load experiment was performed during the tagging run.

The Ollama gate uses process-local threading locks (`blipshell/llm/ollama_gate.py:81`). It coordinates threads in one BlipShell process, not a separate nightly CLI and interactive CLI. Do not assume that the catch-up loop automatically yields to a separately running chat process. Whether the model server supplies adequate cross-process scheduling remains a deferred operational check, not a measured conclusion here.

## Execution order and acceptance

1. After the running process is stopped or finished, fix compaction identity preservation and shell result classification. Add regression tests before considering these paths trustworthy.
2. Unify runtime privacy enforcement, including targeted tasks and the worker. Validate with fake cloud clients that forbidden requests never reach transport.
3. Fix web session isolation, authentication validation and headless approval enforcement before relying on multi-client/headless operation.
4. Add retrieval degradation, atomic task claims, stream failure/cancellation handling and transactional nightly history.
5. Run targeted regression tests and the full suite after fixes. Then perform a bounded live check of tagging throughput, interrupted-run recovery, and interactive latency while maintenance is active. Measure this on the actual server/configuration without rewriting corpus data for testing.

Use small commits with explicit acceptance tests. The routing-policy fixes are related and should share one implementation. Avoid unrelated runtime changes while the user's current loop is active. The audit itself added only this document and the diagnostic script.

## Coverage and remaining limits

| Area | Evidence in this review | Still needed |
|---|---|---|
| Nightly/tagging | Earlier database snapshot and isolated tests; current history race probe | Live throughput, interruption recovery, overlap and server contention |
| Session/memory lifecycle | Compaction, persistence mapping and queue paths; synthetic reproduction | Historical impact assessment only if evidence identifies affected sessions |
| Routing/privacy | Factory, worker, runtime toggle and direct-target paths | Deployment-specific transport checks without sending private test data |
| Retrieval | Search failure paths; existing filtering/ranking code and baseline tests | Live recall quality and latency evaluation |
| Tool execution | Registry, shell results, approval wiring and filesystem/memory-file guard paths | Background-output and cancellation subprocess tests |
| Web/API/background workers | Auth, shared sessions, claims, streaming; fake-route reproductions | Real multi-client/disconnect integration tests |
| Backup, imports, handoff, integrations | Selected source paths and prior review/test evidence | End-to-end restore drill and external/physical integration validation |

No live endpoint availability, cloud privacy compliance, physical device behavior, restore success, or whole-corpus correctness is certified by these checks. The findings provide specific reproducible fixes; they do not establish a single historical cause for every nightly incident.
