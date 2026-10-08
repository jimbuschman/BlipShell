# Model benchmarking

BlipShell's benchmark is deliberately judge-free. A run spends tokens only on
the candidate being measured. Objective checks are scored locally; open-ended
answers are exported for optional review in ChatGPT, Claude, or by a human.

## Run tiers

```powershell
# Fast plumbing/capability check: dedup + reasoning, one repeat
blipshell benchmark run qwen3:14b

# Routing-job comparison: pipeline + reasoning + session review, three repeats
blipshell benchmark run qwen3:14b --tier compare

# Production decision: every suite, coding and embedding, five repeats
blipshell benchmark run qwen3:14b --tier decision
```

`--jobs` and `--repeats` override a tier. Use identical model endpoint,
context window, timeout, tier, repeats, benchmark version, dataset version,
and Git commit for every model in a comparison cohort.

The estimated duration depends heavily on the model and hardware. `decision`
is intentionally several times more expensive than a single old-style full
run because it performs five complete repeats and all 15 agentic coding tasks.

## Offline external review

Every run writes a review source containing the open-ended cases. Export a
blinded packet after running all candidates in the cohort:

```powershell
blipshell benchmark review export `
  --models qwen3:14b,deepseek/deepseek-v4-flash `
  --name september-routing-cohort
```

Give only the generated Markdown or packet JSON to the reviewer. Do not give it
the companion key file: that key maps blinded labels back to model identifiers.
The packet instructs the reviewer to return JSON containing a 0.0-1.0 score and
short rationale for each item.

Import the returned file:

```powershell
blipshell benchmark review import C:\path\to\returned-review.json
```

The report is regenerated automatically. A second independent review of the
same packet can be imported the same way; review scores are averaged and every
reviewer remains named in the stored evidence. Candidate inference is not run
again.

## Artifacts

New artifacts are separated under `benchmark_results/`:

```text
benchmark_results/
  model-runs/          versioned metric rows and exact run settings
  transcripts/         candidate call transcripts
  review-sources/      open-ended cases before blinding
  external-reviews/
    pending/           Markdown and JSON packets
    keys/              private alias-to-model mappings
    imported/          returned reviews and normalized score rows
  reports/             portable Markdown and JSON reports
  continuity/          deterministic and behavioral continuity evidence
```

Pre-v2 flat result files remain readable as historical evidence. New model-run
discovery ignores transcripts and continuity artifacts instead of logging them
as malformed runs.

## What the report enforces

- Benchmark, dataset, and scorer versions are recorded in every new run.
- Required coverage is a fixed 15-category contract, not the widest model in
  the current corpus.
- Completion rate is separate from ability. Effective scores multiply quality
  by completion so timeouts cannot improve a model's composite.
- Open-ended jobs are blank until an external review is imported.
- Models are comparable only in a same-configuration cohort.
- The unified report includes adjacent multi-turn agent and behavioral
  continuity evidence without silently merging similar model identifiers.
- Live-corpus prompt and response text is never written to committed
  transcripts.

Regenerate the report without rerunning a model:

```powershell
blipshell benchmark report
```

The portable files are `benchmark_results/reports/report.md` and
`benchmark_results/reports/report.json`.
