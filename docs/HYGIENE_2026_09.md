# Hygiene 2026-09-06 — derived-layer cleanup, failure visibility, PII engine

Origin: a read-only readout of the 2026-09-02 corpus copy (42,397 memories,
36,548 entities, 515 MB) taken while answering "should memories be cleaned
up over time?", plus two external review claims (dependency weight of the
privacy layer; 344 blind excepts). The memory layer needs no cleanup — the
noise filter, write-time dedup and consolidation already do it, and usage is
flat across memory age (27-31% of memories from each of 2024/2025/2026
surfaced in the seven months access has been tracked). The defects are in
the DERIVED layers and in how failures are reported.

**Status: Parts A-E built and tested on the dev box (suite green). Nothing
committed yet. Ollama PC steps at the end have NOT been run.**

Verified numbers:

| Fact | Value | Where measured |
|---|---|---|
| archived entities still in `vec_entities` | 22,748 | live DB, 2026-09-06 |
| ... of which merged HUSKS (name in `entity_aliases`) | 7,557 | Sep 2 copy |
| ... of which DORMANT (pruned, no alias) | 15,218 | Sep 2 copy |
| husks still referenced (mentions/relationships/alias rows) | 672 | Sep 2 copy, `repair_husk_references(dry_run)` |
| ... stranded mentions / relationships / alias rows | 72 / 106 / 900 | same |
| ... dead-end alias chains (unrepairable automatically) | 7 | same |
| post-merge mentions resolved INTO husks | 46, all husks | Sep 2 copy |
| `except Exception` sites, whole package | 342 (tools 19, memory 93, agent 51, nightly 26, ui 27) | grep |
| PII tests on a box without Presidio | 76 pass, 3 SKIP — never fail | dev box |

## Part A — husk vectors poison creation-time resolution  (code) — DONE

**Defect.** The June 2026 merge archived 22,775 entities via
`archive_entities` (soft, per the ARCHIVE mandate). The per-ID vector
delete on that path swallows exceptions, the nightly orphan sweep only
covered `vec_memories`, and `search_similar_entities` enriched KNN hits with
no archive check. Resolution Stage 2 therefore auto-merged new mentions
into dead entities (`emotionengine` -> archived `emotionengine`, sim ~1.0,
2026-09-02). `revive_entities`' docstring assumed "new mentions never land
on a husk" — true for same-NAME mentions (Stage 0 alias routing), false for
same-MEANING ones. `scripts/rebuild_vectors.py` embedded ALL entities, the
entity backfill had no filter, and `scripts/audit_db.py` expected
`vec_entities` == count(entities): a sweep would have been undone the same
night and reported as drift.

**Design change during the build.** The first cut filtered `is_archived = 0`
everywhere. Measuring the Sep 2 copy showed 15,218 of the archived entities
are PRUNED, not merged: each still holds the single mention it was pruned
with, and `create_entity_mention` revives them on re-mention — including
re-mention by MEANING through Stage 2. A blanket archived filter would have
ended that. So the split is husk vs dormant, defined once in
`entity_names.husk_sql()`:

- HUSK = archived AND name in `entity_aliases` -> dead. Filtered from KNN
  results, vectors swept, never backfilled, Stage 2 routes to canonical.
- DORMANT = archived, no alias -> asleep. Stays a candidate, keeps its
  vector, is backfilled, revives on mention. Unchanged.

**Changes.**
1. `VectorStore.search_similar_entities`: over-fetch x3, enrich with
   `NOT husk_sql`, trim to `n_results`.
2. `VectorStore.cleanup_orphan_vectors`: also sweeps `vec_entities` rows
   for husks and for missing entities. Result keeps `archived`/`missing`
   (memories) and adds `entities_husks`, `entities_missing`.
   `count_orphan_vectors()` is the read-only twin; the repair CLI's dry-run
   uses it instead of its own copy of the SQL.
3. `_SOURCE_TABLES["entities"].active_filter = NOT husk_sql("s")` so the
   nightly backfill cannot re-embed swept husks.
4. `SQLiteStore.resolve_husk(id)` -> `(canonical_id, name) | None`;
   `EntityExtractor._resolve_entity` Stage 2 substitutes the canonical for a
   husk candidate (name too, so the version guard judges the real target).
5. `scripts/rebuild_vectors.py`, `scripts/audit_db.py`: husk-excluded.

**Tests.** `tests/test_entity_husk_vectors.py` (real VectorStore, fake
embedder): husk never returned, dormant still returned, k budget survives
the filter, sweep/count/idempotence, backfill does not resurrect husks but
does re-embed dormant + active, `resolve_husk` incl. chain following.
`tests/test_entity_resolution.py::TestHuskRouting`: auto-merge and LLM
merge into a husk land on the canonical; dormant candidate used as-is and
revives on mention; version guard applies to the canonical name.

## Part B — drain references already stranded on husks  (data) — DONE

**Change.** `SQLiteStore.repair_husk_references(dry_run)`: for every husk
(`find_husks_with_references`) that still owns mentions, relationships, or
is the `canonical_entity_id` of an alias row, resolve the terminal
canonical via the alias chain; `merge_entity(husk, canonical)` moves
mentions + relationships; alias rows naming the husk are repointed. A husk
whose chain dead-ends is counted `unresolved` and left alone. Idempotent.
Exposed as `blipshell repair --repoint-husks` (in `--all`, honours
`--dry-run`). The plan's original "revive dormant with stray mention" step
was dropped: a dormant entity's mention is normal, not stranded.

Dry run against the Sep 2 copy: 672 husks referenced, 665 repairable, 7
dead-end; 72 mentions, 106 relationships, 900 alias rows would move; 0.9s.
The 900 alias rows are mostly pre-merge chains (x -> a, then a -> b) that
`resolve_alias` already followed; flattening them is harmless.

**Tests.** `tests/test_husk_repair.py`: move, dormant untouched, dry-run
writes nothing, idempotent, chain (intermediate husk repaired too),
dead-end counted not guessed (FK off to reproduce the old hard-delete).

## Part C — tools signal failure with a type  (code) — DONE

**Defect.** The chokepoint (`tools/base.py::execute_tool_call`) guessed
failure from two prefixes. `Search error:`, `Fetch error:`,
`Workflow 'x' failed:` / `not found`, `Invalid params JSON`,
`Cannot edit a directory`, `Invalid source_type` and
`saved in memory but failed to persist` never matched, so those failures
counted as success: the completion audit accepted a turn whose only action
had failed and the red-x display never fired — the phantom-write class the
prefix check was added to stop (2026-08-04).

**Design change during the build.** The plan said `raise ToolError`. A
blast-radius check found ~100 direct `tool.execute()` calls in tests and
harnesses that read the returned string (some assert `startswith("Error:")`).
Raising would have rewritten all of them for no behavioural gain. Shipped
instead: `class ToolFailure(str)`. Tools `return ToolFailure(<same text>)`;
`result_reports_failure` checks the type first, prefix second. Every direct
caller sees the identical string; the chokepoint sees the type. Sharp edge,
documented and tested: rebuilding the string (f-string, `+`) yields a plain
str — wrap at the point of return.

**Changes.** 70 return sites across 12 tool modules wrapped (65 by pattern:
`Error...`, multi-line `return (` blocks; 3 by hand: params JSON,
`Cannot ...` in memory_fs, `Invalid source_type`). Prefix backstop kept for
third-party or missed sites. Both loop paths (sequential and parallel) go
through the chokepoint.

**Tests.** `tests/test_tool_failure_type.py`: type semantics, chokepoint
(ToolFailure without prefix -> failure; same wording as plain str -> still
success, documenting the backstop's limit), and the real tools that were
mis-scored: web_fetch SSRF block, web_fetch unreachable host (real DNS on
a reserved `.invalid` name), workflow not-found/failed/bad-params, note
persist failure (in-memory save still happens), filesystem helper failures
keep the type through the caller, memory_fs directory edit.
`tests/test_relayed_content_not_failure.py` still passes (run_command
relaying stderr stays a success).

## Part D — guardrail failures are visible  (code) — DONE

Decision, per guardrail: all stay FAIL-OPEN (a broken check must not block
the turn; a completion-audit LLM outage must not make every task
uncompletable). Only visibility changed: doom-loop, review-gate, pause and
trajectory internal errors log at WARNING (were DEBUG), each line naming
what was skipped; the correction judge's fail-closed path too. Completion
audit was already WARNING/ERROR. Test:
`test_broken_guardrail_is_fail_open_but_logged_at_warning` (ScriptedLLMClient
loop, doom-loop check stubbed to raise; turn completes; WARNING captured).

## Part E — PII engine is optional, visible, enforceable  (code + packaging) — DONE

**Defect.** Presidio + spaCy were hard dependencies yet absent on the dev
box (no Python 3.14 wheel), and the suite was green: the 3 NER tests skip.
Any exception loading Presidio logged one INFO line and silently downgraded
to regex — credentials/keys/IPs redacted, names and places NOT. Chat to
cloud is credentials-only by design and governed by `/local`; the exposure
is `router.generate()`, the FULL-sanitize path used by every background job
(session review, lessons, summaries — whole transcripts). Confirmed:
`agent_chat.py:375` — the interactive path bypasses `generate()`.

**Changes.**
1. `pyproject.toml`: `[project.optional-dependencies] pii = [...]`.
   Install: `pip install -e .[pii]`, then
   `python -m spacy download en_core_web_lg`.
2. `pii.engine_status()` (no load; for `/status`) and
   `pii.engine_description()` (loads; for startup). `Agent._report_pii_engine`
   runs off-thread at startup: WARNING when regex-only AND an endpoint
   relays offsite, INFO otherwise. `/status` shows a "PII engine" row.
3. `PIIConfig.require_ner` (default false; documented in `config.yaml`).
   `LLMRouter(require_ner=...)`: `_ner_blocked_endpoints()` excludes every
   `should_sanitize_pii` endpoint from the primary pick AND all three
   fallback hops while Presidio is unavailable; if nothing remains,
   `generate()` raises naming the flag rather than leaking. Logged once per
   process. `EndpointManager.get_endpoint_for_role(exclude=...)` now accepts
   a collection as well as a name; `EndpointManager.endpoints` is public.
   Wired in `Agent` and `MemoryWorker` (which builds its own router).

**Tests.** `tests/test_pii_engine_gate.py` (14): status does not trigger the
load; descriptions; agent WARNING/INFO branches; gate routes to local with
UNSANITIZED text; logs once; opt-in (regex path unchanged without the flag,
API key still redacted); no-op when Presidio loads; refuses when only cloud
exists; holds on the error-fallback hop; `pii.enabled=false` disables it;
`exclude` accepts str and set.

## Ollama PC follow-up (NOT done — needs that machine)

1. `git pull` (after commit).
2. `python -c "from blipshell.llm.pii import engine_description; print(engine_description())"`
   — if regex-only, production background jobs have been shipping names to
   cloud; decide whether to install `[pii]` or set `pii.require_ner: true`.
3. `blipshell repair --sweep-orphans --repoint-husks --dry-run`, then without
   `--dry-run`. Expected: ~7,557 husk vectors + ~60 missing removed, ~665
   husks repointed (72 mentions, 106 relationships, 900 alias rows), 7
   unresolved. Dormant vectors (15,218) untouched.
4. One nightly; confirm the backfill does not re-add husk vectors
   (`count_orphan_vectors()` stays at zero husks).

## Not in scope, deliberately

- Pruning or decaying memories. No measured win anywhere
  (FIELD_SURVEY_2026_09.md sec. 2) and the corpus shows no dead-weight tail.
- A 342-site except sweep. The sites that matter were Parts C and D; the
  rest are background/maintenance logging already named "silent-failure
  disease" in the 2026-08-04 deep dive.
- Retuning the importance scorer. It clumps (48% at 0.8) and does not
  predict use (0.4-scored memories surface more than 0.8-scored), but its
  boost weight is small and the survey's atomic-fact direction (sec. 3.2)
  supersedes it. Recorded so it is not rediscovered.
- The 7 dead-end alias chains. They point at entities hard-deleted by the
  retired `cleanup_entities` path; repairing them means choosing a canonical
  by hand.
