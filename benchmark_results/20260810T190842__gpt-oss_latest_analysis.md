# gpt-oss:latest — pilot analysis (2026-08-10)

First run of the reworked benchmark (transcripts + repeats), and the first
measurement ever of the model that backs `tool_calling_fallback`,
`coding_fallback` and `reasoning_fallback` — i.e. the entire interactive
path when cloud is down, and the model `/local` deliberately routes to.
Run from the dev box over Tailscale against HPBENDERTWO (quiet GPU),
num_ctx=32768, timeout 420s, 2 repeats, **no judge** (no OpenRouter key on
the dev box — quality metrics that need the judge are absent; verdicts
below rest on deterministic scores plus reading the transcripts).

## Verdict, per job this model actually serves

**Reasoning fallback — KEEP, comfortably.** Deterministic pipeline scores
sit at or above qwen3:14b's own numbers: ranking 0.896, importance 0.904,
rank_importance 0.949, entity 0.864 (qwen3: 0.914 / 0.934 / 0.964 / 0.853).
Read qualitatively, the long-form reasoning outputs are genuinely good —
structured, technically correct, well-organized tables. No refusals, no
format breakage, zero errors in 286 calls.

**Tool-calling fallback — ADEQUATE, with a visible gap.** tool_pass_rate
0.77 (0.73 / 0.80 across repeats — the one metric that genuinely varied).
Compare minimax-m3 at 1.00 and even lfm2.5 at 0.93. One in four tool calls
failing means a degraded-but-usable chat when cloud is down: expect
occasional "why didn't it just read the file" moments in fallback mode.
Not disqualifying for an emergency fallback; would be disqualifying as a
primary.

**Coding fallback — FAILS. coding_agentic = 0.19.** Both repeats, 15
sandbox tasks each. This is the finding that matters: if OpenRouter is
down, `!plan`/project-mode work drops to a model that completes about one
agentic task in five. The number should be treated as "does not work"
rather than "works poorly." (Caveat below: the agentic path bypassed
transcript capture, so this is the score without the ability to read *why*
— fixing that is follow-up #1.)

**Speed — usable for fallback chat, not snappy.** Repeat-1 (cold-honest)
medians: pipeline calls 6.1s, reasoning-class calls 22.9s, session-review
21.3s, worst case 61s. gpt-oss spends heavily on thinking tokens (invisible
in the response text, so the est_tok/s column is meaningless for it — see
harness notes). A fallback chat turn lands in the ~10–25s range.

**One behavioral flag: over-SKIPping + extreme terseness on
summarization.** 4 of 20 summarize cases answered `SKIP` (the escape hatch
for meta/self-referential content) and the rest averaged ~10 words. It is
not the summarization fallback (qwen3 is), so this costs nothing today —
but if it's ever considered for that key, judge this first.

## Bottom line

The fallback config is *two-thirds fine*: reasoning and (with a grimace)
tool_calling degrade acceptably when cloud is out. `coding_fallback:
gpt-oss:latest` is a fiction — a cloud outage effectively disables project
mode, and it would be more honest for the executor to say so than to grind
at a 19% task success rate. Options, cheapest first: accept and document;
have `!plan` warn when running on the fallback; or benchmark a newer local
coder (qwen3-coder-class) for that one key. No urgency — this fires only
when OpenRouter/Ollama-cloud are down — but it is no longer an unknown.

## What the pilot proved about the reworked benchmark itself

- **Transcripts pay for themselves immediately.** The SKIP behavior, the
  thinking-token speed explanation, and the reasoning-quality read all came
  from reading outputs, not scores.
- **Repeats against a warm Ollama are NOT independent samples.** Repeat 2
  ran 10–25× faster with byte-identical outputs — server-side caching plus
  deterministic sampling. Every spread=0.0 in the result file is an
  artifact of that, not evidence of stability. tool_calling (0.73→0.80)
  varied because tool-call parsing runs a different path. Follow-up:
  vary seed per repeat or bust the cache, else --repeats measures the
  cache, not the model.
- **Coverage gap: the agentic coding suite and tool-calling chat bypass
  `router.generate()`** (they use `client.chat` with tools), so the
  RecordingRouter captured none of their traffic — exactly the suite whose
  failures most need reading. Follow-up: record at the client layer too.
- **est_tok/s is unusable for thinking models** (response-chars/4 ÷ wall
  time, and thinking tokens never appear in the response). Trust latency;
  drop or fix the rate column.
- No judge on the dev box means no longitudinal quality scores from runs
  launched here. If dev-box runs become the norm, set OPENROUTER_API_KEY
  here.

Provenance: run 20260810T190842, host XTEK-COMPUTER5 driving HPBENDERTWO,
jobs pipeline+reasoning+session_review+coding, transcripts in the sibling
`__transcripts.json` (286 calls).
