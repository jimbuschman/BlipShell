# Model & Endpoint Decisions — dated log

## 2026-09-02 (night) — chat → deepseek/deepseek-v4-flash (OpenRouter)

After the tool_calling parser fix (commit 73b01f3), the re-run put
deepseek/deepseek-v4-flash at **0.973 tool_calling — the best score in the
36-model corpus, on the exact stack routed to** (its prior 0.000 was entirely
the parser). The advice engine said KEEP gemma on the every-job rule
(deepseek reasoning 0.768 vs 0.810, code_gen 0.852 vs 0.889) — **overridden,
reasons on the record**: those deltas sit at the judged-metric noise floor
(measured sd ~0.05-0.09) while +0.640 lands on the key's core interactive
job; the agent eval — the purpose-built chat instrument — preferred deepseek
28/30 over gemma's 27/30, both with clean reliability; gemma's 0.333 remains
unexplained-and-suspect (Ollama path, so NOT the parser bug — transcripts
still unread); and chat moves off the free tier that ran dry mid-benchmark
this week, onto dedicated serving at ~$0.086/M (~$2/month at assistant
volume). Wiring: openrouter endpoint priority 1 → 3 (now primary for chat,
coding, session_review); gemma4:31b-cloud via local-cloud is the chat
FALLBACK; reflection stays gemma (its own measurement). Revert is one line:
set openrouter's tool_calling override back to minimax/minimax-m3 and
priority back to 1.

**Watch item:** deepseek's one agent-eval weakness is destructive_caution
(1/3 — most willing to reach for deletion on "clean up the junk"); the
approval flow on destructive tools is the working mitigation. Also still
open: gemma's 0.333 transcript read, and deepseek's session_review_chunked
job (UNKNOWN — fill with `blipshell benchmark run deepseek/deepseek-v4-flash
--provider openai --url https://openrouter.ai/api/v1 --api-key-env
OPENROUTER_API_KEY --jobs session_review`).


## 2026-09-02 (evening) — session_review → deepseek/deepseek-v4-flash (OpenRouter, paid)

The same-day OpenRouter benchmark round (`--jobs session_review,pipeline,
reasoning --repeats 5`, judged) produced a clear winner for this slot:
**deepseek/deepseek-v4-flash ties gemma4:31b-cloud on session review (0.887)
and beats it on LESSONS 0.537 vs 0.386** (+0.151, ~2x the historical lessons
run-to-run noise), approaching minimax's old 0.585 — at $0.086/M in, i.e.
pennies/month at nightly volume. Routed via the openrouter endpoint (full pii
sanitization on background calls, 204K window); `local` endpoint priority
lowered 1 → 0 so openrouter wins the session_review tie (local is sole
provider of its other roles, so nothing else shifts — verified per-role).
Fallback unchanged: local qwen3:14b when OpenRouter is down or /local hides it.

Three companion findings from the same round, all load-bearing:

1. **Free OpenRouter variants are measurably not the same model.**
   minimax/minimax-m3:free scored 0.307 on lessons vs 0.585 for the same
   weights on Ollama cloud; z-ai/glm-5.2:free was unmeasurable outright
   (upstream shared pool 429-saturated at benchmark pace, twice). The ":free"
   route is closed for anything scheduled or quality-sensitive — this is the
   cross-stack rule with numbers on it.
2. **The benchmark's tool_calling suite cannot score openai-provider
   candidates.** deepseek-v4-flash and minimax:free both scored exactly
   0.000 while their reasoning/code_gen ran normally — and production pushes
   tool calls through the identical OpenRouter endpoint daily (coding). The
   table shows a stack gradient (local ~0.9, Ollama-cloud 0.3-0.7,
   OpenRouter 0.0), not a capability gradient. OPEN ITEM: read the
   transcripts / fix the suite's response parsing for the openai path before
   using tool_calling numbers in any cross-stack comparison. gemma4's 0.333
   is under the same suspicion.
3. **Chat stays on gemma4:31b-cloud for now.** deepseek's 28/30 agent eval
   (clean: 0 silent/0 turn-limit, 13.3s/ep; one flaw, destructive_caution
   1/3) was measured on Ollama's stack, and finding 1 above is proof that
   stack changes can move quality a lot. Revisit chat after the tool_calling
   artifact is resolved or after a live trial; at $0.086/M the cost argument
   for switching is already made if capability holds.


Why each `config.yaml` routing choice is what it is. One dated entry per
decision, newest first. This file exists because the rationale used to live in
config.yaml comments, which (a) had drifted before (the `reasoning:` comment
claimed it handled lessons; lessons go through SESSION_REVIEW), and (b) were
one `ConfigManager.save()` away from deletion — programmatic saves rewrite the
file without comments. Config comments now stay terse and point here.

**The rule, unchanged:** model choices are decided on measured evidence —
`blipshell benchmark run <model> --repeats 5` for the job keys, the agent-eval
harness (`scripts/agent_eval/`) for the main model — and a key is judged on
EVERY job it controls.

---

## 2026-09-02 — minimax-m3 delisted from Ollama free tier

**tool_calling → `gemma4:31b-cloud`.** Evidence: the 2026-08-19 agent eval
(24 models, 720 episodes, docs/AGENT_EVAL_2026_08.md). Among the free-tier
candidates then available (gemma4:31b, gpt-oss:120b/20b, nemotron-3
nano/super/ultra): gemma4:31b scored 27/30 and was the ONLY one with a clean
reliability record — 0 silent episodes, 0 turn-limit burns, 0 API errors —
at 10.5s/episode. The nemotrons go silent or doom-loop (ultra: all three
destructive_caution episodes burned 20 turns saying nothing) at 24–40s/ep;
nano invents answers under false premises (nonexistent_file 0/3 —
disqualifying for a memory assistant); gpt-oss:120b returns nothing when it
can't resolve something; gpt-oss:20b scored 18/30 with 9 silent episodes.
**Known, consistent flaw:** domain misreading on config questions (multi_fact
0/3 — asked whether memory dedup is enabled, it reaches for Linux KSM/esxcli).
Watch for sysadmin-flavored answers to BlipShell questions.

**session_review → `gemma4:31b-cloud` — RESOLVED 2026-09-03, measured.**
The scoped benchmark run (`--jobs session_review,pipeline`, after the first
attempt exhausted the free tier): session_review **0.887**, lessons **0.386**.
Verdict: KEEP. On lessons — the deciding number for this slot — it is well
below minimax's old 0.585, but it beats local qwen3:14b on BOTH jobs the key
controls (0.887 vs 0.844; 0.386 vs 0.345) and keeps the 128K single-pass
window. Best available at the current tier; the standing pii upper-bound
caveat applies. If the paid tier ever returns: glm-5.2 measured 0.658 lessons.

**Open flag from the same run: gemma4:31b-cloud scored 0.333 on the
`tool_calling` benchmark suite** (qwen3:14b: 0.933) — in tension with its
27/30 agent eval, which is the stronger instrument for the main-model choice
(720 hand-verified episodes vs 3-case invocation matching) and which this
decision continues to rest on. Per the eval doc's own rule, a low score is a
claim about the harness until the transcript is read: check this run's
tool_calling transcripts on the Ollama PC before treating 0.333 as real. If
live chat shows missed/garbled tool calls, this number is the likely why and
the chat decision should be revisited.

**OpenRouter's minimax routes are untouched** (coding, chat fallback):
separate billing, unaffected by the Ollama delisting.

**Endpoints `gemini` and `old-pc` removed from config** in the same-day
config prune — both were `enabled: false`. Gemini: free-tier 20-request burst
limit made it unreliable (had priority 3 for summarization). old-pc
(192.168.0.225, ranking role): unused. Definitions recoverable from git
history (`config.yaml` before 2026-09-02).

## 2026-09-01 — reflection gets its own key

**reflection → `gemma4:31b-cloud`** (via local-cloud endpoint override; the
global key stays local `qwen3:14b` for when routing lands on the local
endpoint or /local mode is active). Evidence: Wisp's brain-swap measurement —
same corpus, same lingering-thought prompt, only the model varied: phi4:14b
produced 3 distinct themes across 20 reflections (domination 0.750);
gemma4:31b-cloud produced 14 (domination 0.150), 3x faster. BlipShell's own
23-of-24-thoughts-on-one-subject incident (2026-07-09) is the same pathology.
That measurement is Wisp's corpus, not ours: the nightly `self_reflection`
job reports theme_diversity every run — **if distinct_themes doesn't move
within ~2 weeks of nightly cadence, revisit.** Privacy: prior thoughts leave
the machine pii-sanitized; /local hides the endpoint.
**reflection_fallback → `qwen3:14b`**: reflection fires after hours idle or
at the 2am nightly — same already-resident logic as session_review_fallback.

## 2026-08-19 — agent eval (main-model instrument)

24 models x 10 scenarios x 3 episodes against BlipShell's real tool schemas
and system prompt. Full writeup: docs/AGENT_EVAL_2026_08.md. Verdict then:
keep minimax-m3:cloud (26/30, scattered singleton failures, fastest at
7.3s/ep) over higher-scoring gemma4 variants because of gemma4's domain
blind spot. That calculus flipped 2026-09-02 when minimax left the tier and
every remaining alternative had a worse failure MODE than gemma4's.

## 2026-08-04 — session_review → minimax-m3:cloud (now superseded)

Beat local qwen3:14b on BOTH jobs the key controls: session review 0.944 vs
0.844, lessons 0.585 vs 0.345 (qwen3 scored 0.420 then 0.345 on re-run — a
real weakness, and lessons feed a permanent context pool). kimi-k2.7-code
rejected at 0.395 lessons — worst of every model measured — despite leading
session review at 0.925: the canonical example of judging a key on ALL its
jobs. Standing caveat inherited by any cloud successor: pii_sanitize scrubs
the transcript before the model sees it and the benchmark feeds unsanitized
sessions, so benchmark lesson scores are an UPPER BOUND on live quality.

## 2026-07-31 — free-tier churn lesson + cold-load fallbacks

Ollama retired kimi-k2.5 mid-nightly with no notice (it held the
session_review slot). Mitigations that remain live: a 410/404 is classified
as a MODEL error, so it falls back without penalizing the endpoint that also
serves interactive chat. **session_review_fallback → qwen3:14b** (was
gpt-oss:latest — that forced a large COLD model load at 2am and llama-server
died on CUDA init); the fallback for any idle/nightly task should be a model
that's already resident.

## Earlier (from the 2026-06..07 benchmark corpus)

- **tool_calling → minimax-m3:cloud (2026-06-01, superseded)**: 100% tool
  pass; tied m2.5, beat m2.7 (which skipped web searches).
- **coding → minimax/minimax-m3 (OpenRouter)**: matched m2.5 quality on
  bench, newer + larger context (204.8K cap to match the chat endpoint and
  bound request cost).
- **ranking / ranking_importance → qwen3:14b** (was qwen2.5:14b): matches the
  reasoning model, avoids a model swap on the shared GPU.
- **ranking_importance_fallback → qwen3.5:9b** (was qwen2.5:7b): 98/98% vs
  92/90%.
- **summarization → glm4:latest**, with Groq (gpt-oss-120b) tried first via
  endpoint priority; Gemini endpoint retired (see 2026-09-02).
  **summarization_fallback → qwen3:14b** (was glm4): 100/95% vs 92/100%, and
  already loaded.
- **embedding → qwen3-embedding:0.6b**: ~10% better similarity than nomic on
  our corpus; understands domain terms. Local always.
- **Benchmark judge → anthropic/claude-opus-4.8 via OpenRouter**: strong,
  neutral (never a candidate), graceful-fail when unreachable.

## Standing config facts worth knowing (not decisions)

- `memory.auto_prune_days` — RESOLVED 2026-09-02: removed from config.yaml so
  the code default 0 (disabled) rules. History: commit 66e01ea (2026-02-20)
  disabled the default after 90-day pruning archived 1083 imported memories —
  but only changed config.py; config.yaml kept `90` and silently overrode the
  fix for six months of nightlies. The user did not recall choosing 90; the
  recorded decision was 0. Rows the prune archived since February remain
  archived (reversible in principle, but 66e01ea's blanket
  `UPDATE memories SET is_archived = 0` restore is NO LONGER SAFE — the
  consolidate/curation jobs also archive rows now, so a restore would need to
  target the prune's profile: old + importance <= 0.3 + rank <= 2. They were
  low-value by construction and invisible to search either way; leaving them
  archived is the default position.)
- `llm.timeout: 300` (default 120): slower local models need it; pinned by
  `tests/test_benchmark_timeout.py`.
- `planner.enabled: false`: auto-detection off, `!plan` triggers manually.
- `reflection.gravity_enabled: true` (default false): graduated opt-in taken.
- `database.require_existing: true`: this instance HAS a corpus — a missing
  file at the resolved path is always a wrong-path launch (see the nine-lost-
  days incident, CLAUDE.md). Fresh installs set false.
