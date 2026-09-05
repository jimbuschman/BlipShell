# Field Survey 2026-09-01 — What the Rest of the World Is Doing

Four primary-source research passes (papers, official docs, and source code read
at file-path level — not blog roundups), synthesized against BlipShell's
architecture as of commit 051becb. Companion to `SAPPHIRE_COMPARISON.md`: that
doc compared against one rival; this one compares against the field.

Evidence tags used throughout, strongest first:
**[MEASURED]** = published numbers from the mechanism's own ablation or a
neutral benchmark; **[VERIFIED]** = mechanism confirmed by reading the source
code, no effectiveness numbers; **[SHIPPED]** = product behavior documented
first-party, unmeasured; **[CLAIMED]** = vendor-run numbers only.

---

## 0. The two headline facts about the field itself

1. **The two most-cited memory architectures were abandoned by their own
   authors in 2025-26.** Letta archived the MemGPT block/archival server and
   rebuilt memory as a git-backed Markdown filesystem ("MemFS",
   docs.letta.com/concepts/memfs). Mem0's v2.0.0 OSS rewrite deleted its famous
   LLM-arbitrated ADD/UPDATE/DELETE reconciliation loop and removed graph
   memory from OSS entirely (additive-only now; verified in
   `mem0/memory/main.py` at v1.0.0 vs main). Any article describing either
   system's "signature" mechanism is describing an archive.

2. **Every vendor benchmark number in this space is self-run, and they openly
   war over each other's harnesses.** Mem0 vs Zep on LoCoMo: each side
   demonstrably misconfigured the other (blog.getzep.com "Lies, Damn Lies &
   Statistics"; getzep/zep-papers#5); Zep's DMR "win" over MemGPT rests on a
   cross-paper citation it admits it could not reproduce; Letta's counter-post
   showed plain file tools + GPT-4o-mini at 74.0 beating Mem0's best 68.5.
   The one result that reproduces in *everyone's* tables, including Mem0's own
   paper: **full-context stuffing beats the memory systems on LoCoMo
   accuracy** — the honest claim industry-wide is cost/latency, not accuracy.
   This is a live case study in why BlipShell's benchmark discipline
   (pre-registration, measured noise floors, committed per-run results with
   git_sha + host) is the right religion.

---

## 1. What the field validates in BlipShell as-built (do not rebuild)

- **Plain hybrid retrieval is the winning core.** MemBench (ACL Findings 2025,
  arXiv:2506.21605) [MEASURED]: plain RetrievalMemory 0.692 beats MemGPT 0.455,
  GenerativeAgent 0.478, MemoryBank 0.442 on factual recall; sophisticated
  forgetting/summarization mechanisms *lose* on facts. HippoRAG 2's own paper
  (arXiv:2502.14802) [MEASURED]: "all previous structure-augmented methods
  underperform against the strongest embedding-based RAG" — GraphRAG,
  RAPTOR, LightRAG all lose to a good embedder on factual tasks.
  BlipShell's RRF hybrid core is the right shape; the "good-enough, do NOT
  tune preemptively" mandate survives this survey intact.
- **Keep the FTS5/BM25 leg forever.** Source-bias literature (KDD'24,
  arXiv:2310.20501) [MEASURED]: neural retrievers rank LLM-generated text >30%
  (relative) higher than human text with the same meaning; BM25 is far less
  biased. The lexical leg is a structural defense, not a legacy fallback.
- **Write-time LLM reconciliation (dedup arbitration) is rarer than it looks.**
  Mem0's version of BlipShell's pipeline was its flagship feature — and was
  dropped for cost. BlipShell pays that cost on a background worker thread,
  which is exactly how it stays affordable.
- **Bi-temporal edges + contradiction expiry** — Zep/Graphiti's headline
  architecture [VERIFIED in graphiti_core/edges.py] is what BlipShell's entity
  graph already does. The stealable *increment* is below (§3.5), not the
  mechanism itself.
- **Graded session reflections appear ahead of the published field.** No found
  system grades its own reflections for effectiveness; the journaling
  literature is humans-journaling-with-LLM-help (DiaryMate, CHI'24), not
  agents keeping measured diaries.

---

## 2. Negative results worth respecting (money NOT to spend)

- **Do not enable the reranker as-written — and probably not at all.**
  Independently of the known normalization/logprobs bugs (V2_PLAN:574),
  arXiv:2606.04194 [MEASURED] found off-the-shelf web-search cross-encoder
  rerankers *degrade* conversational-memory Hit@1 by 6.9 pts. Generic
  rerankers do not transfer to personal memory.
- **Do not build GraphRAG-style community summaries.** Indexing cost measured
  at 757M tokens vs 62M for naive RAG on HotpotQA (arXiv:2503.04338); wins
  are on LLM-judged "sensemaking preference," not QA accuracy; naive RAG wins
  "directness" consistently (arXiv:2404.16130's own tables).
- **Do not extend FadeMem/decay on vibes.** Ebbinghaus-style decay
  (MemoryBank, arXiv:2305.10250) has no measured win anywhere; MemBench's
  authors explicitly blame forgetting mechanisms for factual losses. The only
  decay with any ablation support is Generative Agents' recency term, and its
  metric was believability of sims, not answer accuracy.
- **Memory can make answers worse.** MemTrapBench (arXiv:2608.20202)
  [MEASURED]: on Gemini-3-Flash every tested memory strategy scored 54.7-71.2%
  vs 85.2% with NO memory — retrieved priors fixate reasoning. LoCoMo:
  abstention collapses as context grows (2.1% at 16K vs 70.2% at 4K).
  Implication: injected-memory volume is a risk dial, not just a benefit dial,
  and abstention deserves its own measurement (Wisp's exam battery already
  scores abstention; BlipShell's benchmark does not).
- **Identity drift grows with model size** (arXiv:2412.00804) [MEASURED]:
  larger models drift more, and persona prompts don't reliably prevent it.
  Directly relevant to any move toward larger cloud brains.

---

## 3. The borrow list, ranked

Ordered by (evidence strength x fit x cost). Items 1-5 are small; 6-10 are
projects.

### 3.1 Time-aware query expansion + date-filtered retrieval — [MEASURED, replicated]
Temporal reasoning is every system's worst category (LoCoMo: 20.3% vs human
92.6%). LongMemEval's own ablations: LLM time-range extraction + range
filtering = +6.8-11.3% temporal retrieval; Zep credits its +38.4% temporal
delta largely to timestamp handling. BlipShell has timestamps on every memory
and no time-aware querying at all ("what did I say last week" is served purely
by embedding luck). *Cost: low — one extraction call (or regex fast-path) +
a date filter on the existing index.*

### 3.2 Fact/observation extraction as retrieval keys — [MEASURED, two independent benchmarks]
Index extracted atomic assertions alongside raw turns: LongMemEval +9.4%
recall@k, +5.4% end accuracy; LoCoMo: "observations" beat raw turns on
temporal QA 41.8 vs 26.2 F1. BlipShell indexes summaries + raw content —
close, but summaries are per-message prose, not atomic facts. The write
pipeline already runs an LLM per message, so the marginal cost is prompt
surface, not a new call. *Cost: medium (schema + backfill).*

### 3.3 Source-bias countermeasure for assistant-authored memories — [MEASURED, published confirmation of "exhaust outranks perception"]
The Wisp finding is now corroborated at KDD/ACL level: retrievers prefer LLM
text >30% (arXiv:2310.20501); in iterative loops LLM text takes 73-81% of
dense top-5 after ONE iteration and human content falls below 10% of top-50 by
iteration ten (Spiral of Silence, ACL'24, arXiv:2404.10496). Zep's worst
LongMemEval category (-17.7%) is retrieving its own assistant outputs.
BlipShell stores both roles and the corpus share of assistant text only grows.
**Run `scripts/retrieval_provenance.py` first** (shipped 051becb); if it shows
over-representation, the measured fixes are a provenance-based score penalty
(echo weighting) + overfetch-then-rank — Wisp measured that neither works
alone. The `role` column already exists; this is config + a scoring term.
*Cost: low, gated on the diagnosis. This is the one retrieval change that can
overrule the no-preemptive-tuning mandate, because it would no longer be
preemptive.*

**RESOLVED 2026-09-02 — NO PATHOLOGY, item CLOSED.** The diagnosis ran on the
live corpus (25 probes, top-10 scored): corpus 59.8% assistant-authored,
retrieval top-10 only 37.6% assistant — **0.63x, i.e. retrieval
UNDER-represents exhaust**; median rank of the first user-authored result was
1, and no probe lacked a user result. BlipShell's summarize-then-embed
pipeline, importance scoring, and FTS leg apparently defend against the bias
Wisp measured on raw-text storage. No echo-weighting, no overfetch change —
the retrieval good-enough mandate stands. (Caveats for a future re-run: 25
probes, recent-user-message probe set; re-run after any change that stores
raw assistant text or after sleep-time rewriting ships, since that grows the
self-authored corpus — §3.6's interaction warning still applies.)

### 3.4 Importance-triggered reflection (accumulator, not clock) — [MEASURED ablation]
Generative Agents (arXiv:2304.03442): reflection fires when summed importance
of recent events crosses a threshold (150, ~2-3x/day), not on a timer; the
ablation shows reflection worth ~3 TrueSkill points of believability. Two
increments for BlipShell's reflection layer: (a) an importance accumulator as
an ADDITIONAL trigger next to idle/return/nightly — an eventful afternoon
earns a thought, a dead week doesn't force one; (b) the reflection-tree move:
periodically ask "what are the 3 most salient high-level questions" over
recent thoughts/memories, answer them as insights that CITE their evidence
rows, and store those back — thoughts that synthesize upward instead of only
accumulating sideways. Detail from their code worth copying: insights carry
pointers to the memories they came from (`reflect.py`), which makes every
higher-level belief auditable. *Cost: medium. Note their paper/code
discrepancy before copying constants: paper says retrieval weights all 1.0,
shipped code uses [0.5, 3, 2] (recency, relevance, importance).*

### 3.5 Event-time extraction ("valid_at") + deterministic newest-fact supersession — [MEASURED]
BlipShell's bi-temporal edges use ingestion-side time. Zep extracts EVENT time
from content ("last summer we...") into valid_at, distinct from created_at
[VERIFIED, graphiti_core]. And two 2026 preprints converge on: deterministic
"newest valid fact wins" retrieval policies beat LLM freshness-reasoning —
MemStrata (arXiv:2606.26511) 0.95+ vs 0.20-0.47 for RAG on evolving facts;
arXiv:2606.01435 +28 pts over HippoRAG-2 on FactConsolidation. Extraction
quality is the known bottleneck (97% on clean facts, 44% on messy NL).
*Cost: medium — extraction prompt + a retrieval-time policy; entity graph
already has the schema shape.*

### 3.6 Sleep-time compute: point nightly cycles at REWRITING, not only adding — [MEASURED]
Letta's paper (arXiv:2504.13171): background rewriting of standing context
into "learned context" = ~5x less test-time compute at equal accuracy,
+13-18% when scaled; gains concentrate where future queries are predictable —
and a personal assistant's queries are highly predictable. BlipShell already
has the two required assets (a nightly slot, an LLM-authored user-model doc);
the steal is a nightly job that CONSOLIDATES AND REWRITES the user model,
digests, and the thought corpus — resolving contradictions, merging themes,
pre-drawing inferences — instead of only appending new material. LangMem's
debounce pattern [VERIFIED: `ReflectionExecutor.submit(after_seconds=N)`,
cancel-and-reschedule] is the cheap scheduling trick: consolidate when
activity settles, not per-message. **Interaction warning:** sleep-time output
is assistant-authored text entering the corpus — it raises the §3.3 stakes,
so ship after the provenance diagnosis, and sanitize inputs BEFORE
summarization (state-contamination result, arXiv:2605.16746: cleaning the
finished summary leaves laundered influence intact). *Cost: medium.*

### 3.7 Telegram as the second surface — [SHIPPED, ecosystem-convergent]
The entire local-first ecosystem answered "how do I reach my assistant from my
phone" the same way: a messaging channel into the one memory hub
(OpenClaw — 388K stars — WhatsApp/Telegram/Signal/iMessage gateways; Hermes
Agent 239K stars, same; Khoj serves WhatsApp). Nobody syncs memory between
devices; the interfaces come to the hub. BlipShell has a DISABLED Telegram
integration sitting in config.yaml — re-enabling it is the cheapest
multi-device win in this document, and voice rides along free (voice memos →
Whisper transcription is the ecosystem's pragmatic 2026 voice answer; a live
duplex pipeline is not). Pair with the proactive ladder (§3.9). *Cost: low —
the integration exists.*

### 3.8 "No hidden state": a human-readable memory mirror + per-turn transparency — [SHIPPED pattern]
Three tiers seen in the wild: markdown-files-as-memory (OpenClaw's
USER.md/MEMORY.md/daily notes, "the model only remembers what gets saved to
disk"; gptme goes further — memory as a git repo, every change diffable),
memory panels (Open WebUI, Letta ADE), and per-turn injection transparency
(Elroy's "Relevant Context" panel showing exactly which memories reached the
prompt this turn). BlipShell has `/why` traces and DIGEST.md export — the
increments: (a) a nightly-exported USER_MODEL.md / MEMORY.md mirror the user
can read and correct (Claude's memory product treats the editable summary as
the contract [SHIPPED]; Open WebUI's `replace_memory_content` tool is the
"correct, don't duplicate" verb worth copying), (b) surface which memories
were injected per turn in the CLI. Also worth copying cheaply: OpenClaw's
"dreaming" gate — daily notes only PROMOTE to long-term memory after passing
recall-frequency and query-diversity gates; facts must earn residency
(MemoryOS's heat-based promotion, EMNLP'25 oral, is the academic twin).
*Cost: low for the mirror/export; medium for promotion gating.*

### 3.9 A proactive channel with a judged gate — [MEASURED guardrail]
BlipShell already computes a morning briefing and runs nightly research-shaped
jobs; nothing DELIVERS unprompted. The field: Khoj automations mail scheduled
query results; Hermes crons deliver to any channel; ChatGPT Pulse does nightly
research into morning cards. The measured caution: ProactiveBench
(arXiv:2410.12361) — even a model TRAINED on human accept/reject reaches only
F1 66% at proposing help people wanted; unfiltered proactivity annoys a third
of the time. So: deliveries pass a "would the user want this now?" LLM judge
(BlipShell already has the relevance-judge pattern in resurfacing), and start
at the bottom of LangChain's notify < question < review ladder — notify only.
Concretely: morning briefing + nightly report + starved lingering thought →
one Telegram message, judged, rate-limited. *Cost: low once §3.7 exists.*

### 3.10 ExpeL-style lesson lifecycle: upvote/downvote/edit — [MEASURED on agent tasks]
ExpeL (AAAI-24, arXiv:2308.10144): abstracted insights maintained via
ADD/EDIT/UPVOTE/DOWNVOTE, +11-19 pts on agent benchmarks; insights matter most
for reasoning tasks. BlipShell's lessons have scores but no lifecycle — a
lesson contradicted by later experience just sits there. The steal: a nightly
pass that revotes lessons against recent session reflections (upvote
confirmed, downvote contradicted, edit refined), turning the lessons pool
from append-only into self-correcting. Fits the same nightly slot as §3.6.
*Cost: medium.*

### Deliberately not recommended
- **HippoRAG-2 / PPR graph retrieval**: real but small wins (+4 F1 multi-hop)
  at full-KG infrastructure cost; same paper shows plain dense retrieval
  beating every other graph method. BlipShell's entity-expansion boost already
  captures the cheap part.
- **KV-cache memory tiers** (MemOS): unreplicated, impractical on Ollama.
- **Turn-level late-interaction + z-score fusion** (arXiv:2606.04194: +13-24
  pts Hit@1, score-fusion beats RRF by 3.4): promising but single-paper, and
  it touches the retrieval core the mandate protects. Park it until the §3.3
  diagnosis forces retrieval work anyway; then consider both in one measured
  change.
- **Prompt self-rewriting as procedural memory** (LangMem optimizers):
  unmeasured, high blast radius. Research-organism material (Wisp), not
  assistant material.

---

## 4. Cross-cutting observations for the roadmap

- **Memory portability is becoming a real feature.** Letta's `.af` agent-file
  format serializes blocks + full history with per-message in-context flags;
  Hermes ships `claw migrate` to import a rival's memory files; Claude memory
  offers import/export. A BlipShell export (user model + core memories +
  lessons as markdown) is cheap insurance and doubles as the §3.8 mirror.
- **The ecosystem converged on markdown-file memory partly as UI.** OpenClaw
  (388K stars) and Hermes (239K) both won mindshare with "your memory is a
  folder you can read." BlipShell's SQLite core is strictly more capable —
  but the mirror layer is why users trust those systems.
- **Import remains everyone's weakness** (email especially). BlipShell's chat
  import pipelines are ahead of the field here; screenpipe's event-driven
  ambient capture (21K stars, local SQLite+FTS5, MCP-queryable) is the
  interesting perception-organ complement if ambient capture is ever wanted.
- **Benchmarks to borrow probes from** rather than trust wholesale:
  LongMemEval (temporal reasoning, knowledge updates, abstention as first-
  class abilities — abstention is missing from BlipShell's benchmark),
  MemoryAgentBench's FactConsolidation, MemTrapBench's memory-hurts scenarios.

---

## 5. Suggested sequencing

1. **Now / diagnosis-gated:** run `retrieval_provenance` on the live corpus
   (§3.3); if pathological, echo-weight + overfetch in one measured change.
   — DONE 2026-09-02: no pathology (0.63x, under-represented); no tuning.
2. **Small, high-certainty:** time-aware query filtering (§3.1); abstention
   cases in the benchmark (§2); memory mirror export (§3.8a).
3. **The surface bet:** re-enable Telegram (§3.7), then judged notify-only
   proactive delivery (§3.9) — this is the largest user-visible change per
   unit work in the document.
4. **The nightly upgrade:** sleep-time rewrite of user model + lesson
   revoting + promotion gating (§3.6, §3.10, §3.8c) — one coherent "the
   nights get smarter" project, shipped with before/after theme + provenance
   readouts.
5. **The reflection upgrade:** importance accumulator + evidence-citing
   insight synthesis (§3.4) — after the nightly cadence has produced enough
   thoughts to synthesize.
6. **Structural, later:** atomic-fact retrieval keys (§3.2) and event-time
   supersession (§3.5) in one schema-touching milestone.

---

## Sources

Frameworks: MemGPT arXiv:2310.08560 · Letta MemFS docs.letta.com/concepts/memfs
· sleep-time arXiv:2504.13171 + letta.com/blog/sleep-time-compute · Mem0
arXiv:2504.19413 + mem0 source v1.0.0/main · Zep arXiv:2501.13956 +
graphiti_core source · LangMem source (reflection.py, extraction.py, tools.py)
· Cognee source (DataPoint.py) · MemoryOS arXiv:2506.06326 · A-MEM
arXiv:2502.12110 · MemOS arXiv:2507.03724 · MIRIX arXiv:2507.07957 · Honcho ·
Memobase · agent-file (github.com/letta-ai/agent-file).
Continuity: Generative Agents arXiv:2304.03442 + joonspk-research repo
(reflect.py, retrieve.py) · Reflexion arXiv:2303.11366 · ExpeL
arXiv:2308.10144 · ChatGPT memory help.openai.com/8590148 + /11146739 · Claude
memory claude.com/blog/memory · ChatGPT tasks help.openai.com/10291617 · Pulse
openai.com/index/introducing-chatgpt-pulse · ambient agents
langchain.com/blog/introducing-ambient-agents · ProactiveBench
arXiv:2410.12361 · SCM arXiv:2604.20943 · identity drift arXiv:2412.00804 ·
MicroVerse arXiv:2608.15844.
Local-first: OpenClaw docs.openclaw.ai/concepts/memory · Hermes
github.com/NousResearch/hermes-agent · Khoj docs.khoj.dev · Open WebUI
docs.openwebui.com/features/chat-conversations/memory · LibreChat
librechat.ai/docs/features/memory · Elroy github.com/elroy-bot/elroy · gptme
gptme.org/docs/agents.html · ha-ai-memory · screenpipe
github.com/screenpipe/screenpipe. Star counts via api.github.com 2026-09-01.
Techniques: LongMemEval arXiv:2410.10813 · LoCoMo arXiv:2402.17753 · MemBench
arXiv:2506.21605 · MemoryAgentBench arXiv:2507.05257 · MemTrapBench
arXiv:2608.20202 · HippoRAG 2 arXiv:2502.14802 · GraphRAG arXiv:2404.16130 +
cost analyses arXiv:2503.04338, arXiv:2506.05690 · RAPTOR arXiv:2401.18059 ·
lexical-dense fusion arXiv:2606.04194 · MemStrata arXiv:2606.26511 ·
deterministic freshness arXiv:2606.01435 · source bias arXiv:2310.20501 ·
Spiral of Silence arXiv:2404.10496 · ConsistencyGate arXiv:2607.22962 · state
contamination arXiv:2605.16746 · HyDE arXiv:2212.10496 · MemoryBank
arXiv:2305.10250.
Disputes: blog.getzep.com/lies-damn-lies-statistics · getzep/zep-papers#5 ·
letta.com/blog/benchmarking-ai-agent-memory.
