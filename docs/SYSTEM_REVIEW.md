# BlipShell System Review & Research Synthesis

_Date: 2026-06-08. A grounded internal review (four subsystems) plus targeted external
research (mid-2026 state of the art) aimed at the review's findings._

**How to read this:** Findings are flagged by confidence. Only a few are independently
verified; most are grounded leads to **verify before acting**. Model/benchmark numbers are
largely vendor-sourced — directional, re-validate on our own data (the standing "test before
swap" rule). The research agents did adversarial verification and flagged vendor marketing
(e.g. ">85% memory benchmarks") as unreliable; that skepticism is carried through here.

---

## Executive summary

The research **validates the review**: BlipShell is largely aligned with mid-2026 best
practice. Almost everything below is an **incremental upgrade, not a rewrite.** Several
findings converge across the internal review, external research, and BlipShell's own
memory/lessons — that triangulation is the high-confidence core.

The single strongest theme — **self-knowledge drift** — is both the root cause of the
"confidently describes things that don't exist" problem we've chased for weeks AND has a
clear, industry-consensus fix.

---

## A. Self-model drift — HIGH value, cheap, well-sourced

**Problem (verified):** BlipShell's self-description — agent system prompt, config comments,
project docs — is hand-maintained with no sync to code, and demonstrably drifts:
- The model denied it could see images right after using vision (system prompt said text-only).
- CLAUDE.md still describes ChromaDB, removed ~April 2026 (sqlite-vec migration, see project memory).
- Config comments described a "reranker gate" replaced months ago by an LLM judge.

**Research consensus (well-sourced):** Don't hand-write capabilities — *generate* them from
live runtime state. Battle-tested across Claude Code, Cline, OpenHands:
- Dynamically assembled system prompts from the live tool registry; tools self-remove when
  unavailable (Cline `contextRequirements`). [Cline DeepWiki; dbreunig "How Claude Code Builds
  a System Prompt", 2026-04-04]
- Per-turn environment/state block (cwd, OS, git, date, active MCP). [Claude Code Agent SDK docs, vendor-official]
- "Generate, don't hand-maintain" — registry as single source of truth. [REGAL, arXiv:2603.03018, 2026-03]

**Recommendations:**
1. Generate the capability section of the system prompt from the live `ToolRegistry` (kills the
   vision-denial class by construction). _Highest leverage; we already have the registry._
2. Add a capability/environment state block (extend the existing `_build_state_block()`).
3. Refresh CLAUDE.md (stale ChromaDB, stale model assignments).
4. Startup capability self-audit — _our_ idea; sound but NOT a sourced pattern. Build as cheap
   insurance, not a guarantee.

---

## B. Guardrails — HIGH value, well-sourced, ties to the confabulation theme

**Review finding:** 7-feature guardrails engine; reviewer called it sprawl with unbounded LLM
cost (up to 2 LLM-judge calls per completion attempt).

**Research consensus (well-sourced, strong convergence):** The field moved *away* from stacking
LLM-judge layers.
- Snorkel "self-critique paradox" (2026): forced self-critique dropped a model 98%→57% on
  tasks it already did well — _"the critic, primed to find errors, invented them."_ (Same disease
  as our confabulation problem.) Recommendation: difficulty-gated, never flat-always.
- Anthropic ("Building Effective Agents", "Effective harnesses for long-running agents") and
  Cognition ("Don't Build Multi-Agents"): simplicity + grounding (tests, compile, read the real
  code) over feature-stacking. Deterministic checks ubiquitous; LLM-judge sparingly.
- Tiered eval consensus: deterministic (Tier 1, ~zero cost, ~60% of failures) → heuristics
  (Tier 2) → LLM-judge only for the residual.

**Per-feature verdict:**

| Feature | Mechanism | Verdict |
|---|---|---|
| Context pinning | deterministic | ✅ Keep — strongest alignment |
| Requirement checklist (`confirm_plan`) | one-time gated | ✅ Keep |
| Doom-loop detector | deterministic | ✅ Keep |
| Correction detector | regex | ✅ Keep (watch false positives) |
| Trajectory monitor | deterministic inject | 🟡 Keep, consider merging into context pinning |
| Completion audit | LLM-judge | 🟠 Realign — deterministic checks first, LLM only on hard/ambiguous |
| Critique provider | LLM-judge | 🔴 Strongest over-engineering candidate; overlaps completion audit |
| look-before-review | grounding constraint | ✅ Keep — the field's primary confabulation fix |

**Recommendation:** Collapse the two LLM-judge features into **one grounded, difficulty-gated
gate** (deterministic first: tests pass? files touched? then LLM-judge only when hard/ambiguous,
≤1 call, never on easy tasks). Keep the four deterministic guardrails. Net: 7 → ~5 features,
LLM-judge calls from "up to 2 always" to "≤1, only when warranted."

_Sources: snorkel.ai self-critique-paradox; anthropic.com building-effective-agents &
effective-harnesses-for-long-running-agents; cognition.ai dont-build-multi-agents; CRITIC
arXiv:2305.11738._

---

## C. Memory & retrieval — MEDIUM, incremental (already aligned)

**Review:** sqlite-vec migration eliminated the dual-store drift (verified). Residual risks:
vector write-lock contention under concurrent embeds, consolidation throughput (~20/night vs
17K memories), binary dedup (possible info loss), orphan-vector cleanup gaps.

**Research:** BlipShell independently arrived at the 2026 consensus (hybrid FTS+vector+RRF,
bi-temporal graph, consolidation, decay). Genuine gaps vs frontier:
1. **Cross-encoder reranker as L2** (e.g. `bge-reranker-v2-m3`) — highest-ROI retrieval upgrade.
   Note: `reranker.py` exists but was disabled because the *Qwen3* reranker didn't work via
   Ollama — answer is a *different* reranker on a working path, not the abandoned one.
2. **Edge invalidation in the entity graph** — we have bi-temporal edges + contradiction
   detection (~80% there); wiring them (expire old fact, set `t_invalid`) is what gives
   Zep/Graphiti their temporal-reasoning edge. [Zep arXiv:2501.13956 — the only independently
   published memory benchmark; ~64–71% LongMemEval vs ~55–60% full-context]
3. Embedding bump (qwen3-embedding:4b / EmbeddingGemma-300M) — LOW priority; re-embed cost.

**Do NOT** rewrite around mem0/Letta/Cognee — their ">85% memory benchmarks" are vendor
marketing (agent-verified as non-comparable methodology). Staleness is an unsolved problem
industry-wide, which validates our decay/consolidation investment.

_Sources: Zep arXiv:2501.13956; A-MEM arXiv:2502.12110; Graphiti/Neo4j; EmbeddingGemma
arXiv:2509.20354. Vendor LongMemEval scores >75% flagged unreliable._

---

## D. Model routing — MEDIUM, test-first

| Role | Current | Finding | Action |
|---|---|---|---|
| Interactive tool/coding | minimax-m3 | **Current** — launched 2026-06-01. Vendor benchmarks contested. | Keep; optionally A/B **GLM-5.1** (comparable, MIT, stronger long-horizon) |
| Local fallback coding/chat | qwen3:14b / qwen2.5:14b | **A generation behind** | Test **Qwen 3.6-27B** (24GB GPU) / **3.6-35B-A3B** (16GB) — 73–77% SWE-bench Verified |
| Background summarize/rank | Groq llama-3.3-70b | Overkill | Test **Groq Llama 3.1 8B Instant** (~12× cheaper, 2× faster); re-run ranking benchmark |
| Embeddings | qwen3-embedding:0.6b | Still well-rated | Optional bump qwen3-embedding:4b; LOW priority (re-embed cost) |
| Local vision (none today) | — | No local vision fallback | Optional: **Gemma 4** (vision + tool calling locally) |

**All model numbers are vendor/aggregator-sourced — directional. Re-validate on our own
harness before any swap** (standing rule). Biggest clear win: the local fallback models.

_Sources: minimax.io/blog/minimax-m3 (+ TechTimes skeptic); InsiderLLM best-local-coding-2026;
klymentiev.com groq-pricing; Qwen blog. Independent side-by-side leaderboard: couldn't verify._

---

## E. Reliability bugs (from review — verify before fixing)

1. **OllamaGate blocks indefinitely (no timeout)** — corroborated by our own 2026-03-04
   timeout-disaster memory (names the non-cancellable gate). HIGH. Add timeout/fallback.
2. **Dual execution paths diverged** — `executor.py` lacks the vision gating/degradation and
   fallback that `agent_chat._run_chat_loop` has → vision feature has a project-mode gap. Unify.
3. **Possible local-model-name-to-cloud fallback** (`router.py:209-212`) — verify; would 404
   instead of degrading gracefully.

---

## Suggested sequencing

1. **Self-model fix (A)** — cheap, well-sourced, closes the confabulation theme. _Started 2026-06-08._
2. **Guardrails realignment (B)** — pairs naturally with A; well-sourced.
3. **Reliability bugs (E)** — verify the OllamaGate + dual-path items (own-memory corroborated).
4. **Memory reranker + edge invalidation (C)** — incremental, measurable; build a small
   LongMemEval-style local eval first.
5. **Model routing (D)** — test-first on the Ollama PC; local fallback upgrade is the clear win.
