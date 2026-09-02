# Sapphire comparison + borrow list

Written 2026-08-20. Source: `github.com/ddxfish/sapphire` (282 stars, 73 forks,
AGPL-3.0, created 2025-12-13, last push 2026-08-06, Python/FastAPI, ~16MB).

Status: NOTES ONLY. Nothing below has been implemented. Every BlipShell claim
was verified against code at `39f4fa6` and carries a file:line. Every Sapphire
claim came from their README/docs and is marked accordingly.

---

## 1. What Sapphire is

Self-hosted AI companion platform. Tagline "She's the AI agent you come home to."

- Voice-first: openWakeWord -> Faster Whisper -> Kokoro TTS
- Browser dashboard on :8073, 3D animated avatars, plugin store
- 11 built-in personas (prompt + voice + tools + scopes bundle)
- 65+ tools; Discord, Telegram, email, Home Assistant, SSH, Google Calendar,
  ComfyUI, MCP, Bitcoin wallet
- Tool Maker: the model writes and installs its own tools at runtime
- Supervised multi-process (`main.py` respawns `sapphire.py` on exit code 42)
- ContextVar scope isolation; settings merge with hot / file-watched / restart
  tiers; ed25519 plugin signing; machine-bound Fernet credential encryption

### The core difference

Sapphire is a **breadth** product: surface area, integrations, polish, install
experience. Their memory (per `docs/MEMORY.md`) is SQLite + FTS5 + a vector
index, retrieved as a **cascade** (full-text first, then vector), with memories
capped at ~450 chars as "quick reference snippets." No decay, consolidation,
entity graph, dedup arbitration, or token-pool budgeting documented.

BlipShell is a **depth** product: RRF fusion at k=60, FadeMem decay, importance
weighting, tag overlap, entity-graph expansion, bi-temporal edges with
CONTRADICTING_PREDICATES expiry, 4-stage entity resolution, LLM-arbitrated
dedup, consolidation, 5-pool budgeting.

Summary: Sapphire wins a demo. BlipShell wins a six-month relationship. Their
memory is a searchable notepad; ours is the actual thesis.

### Licensing caution

Sapphire is **AGPL-3.0**. Read for ideas freely; copying code into BlipShell
drags AGPL obligations along. Everything below is a pattern to reimplement,
NOT a file to lift.

---

## 2. Borrow #1 - the ghost-message rail  [HIGHEST VALUE]

### Their idea

Volatile per-turn content (persona "spice" snippets) is injected as a labeled
message immediately BEFORE the user's turn - visible to the model, hidden from
the user. Their stated reason is exact: injecting into the system prompt
"invalidated the cache" for prompt caching. Late placement also exploits
recency bias.

### What is actually true in BlipShell  [VERIFIED]

`_build_messages` (`blipshell/core/agent_chat.py:1139`) concatenates everything
into ONE system message. `_chat_simple` then appends five more blocks to
`messages[0]["content"]`:

| Block | Line |
|---|---|
| mood awareness | agent_chat.py:548 |
| morning briefing | agent_chat.py:562 |
| quiet thought (self-reflection) | agent_chat.py:577 |
| research mode | agent_chat.py:608 |
| review grounding | agent_chat.py:628 |

Pool population sites prove which pools are stable and which are not:

| Pool | Populated at | Volatility |
|---|---|---|
| Core | agent_session.py:88,107 | STABLE per session |
| Lessons | agent_session.py:126 | STABLE per session |
| RecentHistory | agent_session.py:289,297,319 | STABLE per session |
| Recall | agent_chat.py:897,918,939,962,1028 | CHANGES EVERY TURN |
| ActiveSession | session/manager.py:154 | grows every turn |

`pool_order` at `agent_chat.py:1192` is:

    ["Core", "Recall", "Lessons", "RecentHistory", "ActiveSession"]

The single volatile pool renders **second of five**, ahead of three
session-stable pools. `_files_read` (`agent_chat.py:1284`, grows per turn) then
sits ahead of the capability block and time anchor.

### The irony

`agent_chat.py:1297` already documents the intent:

> "Absolute time anchor. Placed at the END so the cacheable prefix of the
> system prompt stays stable across turns - a changing timestamp near the top
> would bust prompt caching on cloud endpoints every turn."

The insight is already ours. The assembly order defeats it: everything after
the Recall block is uncacheable anyway, so that optimization is currently
**inert**. Sapphire's contribution is the discipline (one rail for volatile
content), not the idea.

### Proposed shape (NOT implemented)

1. Reorder so all session-stable content forms an unbroken prefix:
   Core -> Lessons -> RecentHistory -> capability block -> [stable boundary]
2. Move Recall + ActiveSession + `_files_read` + the five `_chat_simple`
   appendices onto a late-injected rail before the user's turn.
3. Precedent already exists in-repo: external file-change notices use exactly
   this pattern at `agent_chat.py:603` (`messages.insert(-1, ...)`).

Moving Recall late is consistent with recency-bias reasoning AND fixes
caching, so it does not contradict the existing "attention is strongest at the
top" comment at `agent_chat.py:1181`.

### Caveats  [IMPORTANT - do not skip]

- There is **ZERO** caching support in `blipshell/llm/`. Grep for
  `cache_control|prompt_cach|cached_tokens|ephemeral` returns nothing. Any
  cloud win depends on providers' AUTOMATIC prefix caching.
- **UNVERIFIED**: whether minimax-m3 via OpenRouter, or Groq, actually do
  automatic prefix caching. Do not assume. Measure.
- The provider-independent win is local Ollama KV-cache reuse on a stable
  prefix - affects the fallback path and every background job. That one is
  real regardless.
- **Measure before building.** The payoff must be a number, not a theory.

### Test surface

`tests/test_build_messages.py` already exists - ordering is testable here on
the dev box, no Ollama PC needed.

---

## 3. Borrow #2 - privacy / local mode  [ALREADY BUILT - CLAUDE.md IS STALE]

The initial ranking was wrong. This is shipped.

- `/local` and `/cloud` commands: `ui/command_handlers.py:235-255`
- `endpoint_manager.local_only`, set from `config.pii.local_mode_default`
  (`core/agent.py:214`), default False (`models/config.py:306-313`)
- Endpoint filter (`llm/endpoints.py:302-304`):

      def _allowed(self, ep: Endpoint) -> bool:
          """False for endpoints local mode hides (they relay off the machine)."""
          return not (self.local_only and ep.should_sanitize_pii)

- `nightly.py:123,131,147` accepts `local_only` too
- The CLI even warns about the session-close review boundary

This is AHEAD of Sapphire, which has one global switch; ours is
per-conversation with a documented edge case.

### ACTION ITEM (not yet done)

CLAUDE.md's final open item still reads: "Identity privacy on cloud chat ...
The fix is routing (a local mode), not more redaction. See V2_PLAN D1."
That describes work that EXISTS. This is exactly the failure mode CLAUDE.md's
own preamble warns about, and the same class of bug as the MemoryWorker-tests
entry that caused duplicate work twice.

The genuine remaining gap is narrower: nothing AUTOMATICALLY engages local mode
when the recall pool surfaces identity-bearing memories. That is a refinement,
not a missing feature. Rewrite the bullet to say so.

---

## 4. Borrow #3 - goals with append-only journals  [GENUINE GAP]

### Their design

Hierarchical goals (subtasks roll under parents), title / description /
priority (high|medium|low) / status (active|completed|abandoned). A
**timestamped append-only progress journal** - updates never edit history, they
append, giving an auditable timeline. A **permanent flag** the AI cannot
complete or delete; only the user can. Tools: `create_goal`, `update_goal`,
`list_goals`. Scoped like the rest of their Mind section.

### BlipShell state  [VERIFIED]

No goals table among the ~25 in `memory/sqlite_store.py`. The only `goal` hits
in the codebase are unrelated (`core/import_lock.py`, `memory/tagger.py`,
`robotics/rules.py`).

Adjacent machinery to build on, not duplicate:
- `task_plans` / `task_steps` (sqlite_store.py:137,148) - executor-scoped,
  per-task, not durable user goals
- `follow_ups` (:328) - cross-session, closest existing analogue
- `projects` (:124) - has metadata + digests

### Why it fits

The permanent flag maps cleanly onto the standing USER MANDATE: ARCHIVE, never
DELETE. Append-only journals match the existing bi-temporal instinct in the
entity graph. This is the borrow most aligned with the v2 memory/continuity
thesis.

---

## 5. Borrow #4 - unified scheduler with a probability roll  [PARTIAL]

### Their design

Cron-scheduled autonomous tasks with an **optional chance %**, foreground or
background execution, an optional persistent chat name so history accumulates
across runs, and a memory slot for state between runs. A Triggers sidebar shows
a timeline of upcoming runs with their probabilities and a manual play button.

### BlipShell state  [VERIFIED]

Six hand-rolled background loops already exist in `core/agent.py`:

| Loop | Line |
|---|---|
| reflection loop | agent.py:326 |
| mood loop | agent.py:356 |
| failed-model clear (60s) | agent.py:381 |
| periodic memory flush | agent.py:389 |
| idle friction probe | agent.py:394 |
| auto-nightly (15min check / 5min idle / configured hour) | agent.py:517-556 |

Missing: any USER-AUTHORED schedule, and any stochastic trigger.

### Assessment

Larger than it first looks. A unified scheduler that the existing six loops
converge onto is the right design, but that is a refactor of live background
machinery, not an addition. The `chance %` is the cheap part and the part that
makes a system read as alive rather than as cron.

---

## 6. Also noted, lower priority

- **`search_help_docs()` as a tool** - Sapphire's assistant can query its own
  documentation (29 docs in `docs/`). Given CLAUDE.md opens with a warning that
  it has twice documented features with no call site, a grep-able self-doc tool
  is thematically on the nose.
- **Tiered code validation** - strict (~60 allowlisted imports) / moderate /
  "system killer" (syntax only). Good pattern even without a Tool Maker.
  Shipping "system killer" as an option is a footgun; would not copy that.
- **Mind events** - fail-open pub/sub so UI stays in sync after tool writes.
  Relevant if the web UI ever grows a memory browser.

## 7. Explicitly NOT borrowing

Voice / avatar / 3D, Discord / Telegram, Bitcoin wallet, personas-and-spice
roleplay, runtime Tool Maker. That is companion-product surface area. Per the
v2 direction note, chasing breadth is how the memory moat stops getting built.
Runtime self-authored tools also sit badly against the standing
"no quick fixes / test before commit" rules.

---

## 8. Ranking + sequence

1. **Ghost-message rail** - highest value, smallest diff, existing test file,
   makes an already-documented intent actually work
2. **Goals + journal + permanent flag** - real gap, clean fit with the thesis
3. **Unified scheduler with chance roll** - good, but scope-creeps into
   consolidating six existing loops
4. ~~Privacy mode~~ - BUILT; reduces to a CLAUDE.md correction plus an optional
   auto-engage refinement

### Before any code

- [ ] Fix the stale CLAUDE.md identity-privacy bullet (section 3 above)
- [ ] MEASURE current cache behavior so the rail's payoff is a number
- [ ] Trace the full call chain for #1 including whether the executor path
      shares `_build_messages` - per the 2026-03-04 lesson, do not patch one
      layer at a time
- [ ] Take #1 into plan mode; it touches both chat paths
