"""Shareable benchmark report — the single deliverable of `blipshell benchmark`.

One deep run per model produces metric rows in the store; this turns ALL stored
models into one self-contained file (Markdown + JSON) you can hand to a stronger
LLM to decide per-job model assignments (local vs cloud) and to judge whether a
new model is worth upgrading to.

Deliberately makes NO switch recommendation: BlipShell routes a mix of local and
cloud models per job with fallbacks, and that's the user's call. The report lays
out quality + speed (+ cost/context for cloud) per category and explains exactly
what each number means so the reading LLM can reason about tradeoffs.

`build_report` is pure (rows in, dict out); `write_report` renders md + json.
"""

import json
from pathlib import Path
from typing import Optional

from blipshell.benchmark.manifest import (
    BENCHMARK_VERSION, DATASET_VERSION, OPEN_ENDED_CATEGORIES, REQUIRED_CATEGORIES,
)

# Scoring metrics (higher = better). Everything else (latency, agreement) is
# informational and shown separately.
SCORING_METRICS = {"accuracy", "quality", "tool_pass_rate"}

# The job categories, in display order, with what each measures + how it's scored
# (this text goes into the report so the reading LLM knows what the numbers mean).
CATEGORIES = [
    ("rank_importance", "Rank+Importance (live pipeline)",
     "The combined rank+importance+type call every message goes through "
     "(rank_importance_and_classify, processor.py:185). This is models.ranking_importance.",
     "Mean of rank-correlation and importance calibration vs gold."),
    ("ranking", "Ranking (import path)",
     "Standalone rank prompt used only by the bulk import path "
     "(rank_memory, import_common.py:598) — NOT the live pipeline.",
     "Rank-correlation of predicted vs gold rank (order matters); flat output scores ~0."),
    ("importance", "Importance (import path)",
     "Standalone importance prompt used only by the bulk import path "
     "(ask_importance, import_common.py:648) — NOT the live pipeline.",
     "Average of correlation and calibration (1-MAE) vs gold importance."),
    ("contradiction", "Contradiction", "Decides whether two memories contradict.",
     "Exact YES/NO accuracy over a balanced set."),
    ("dedup", "Dedup verdict (live pipeline)",
     "Given a new memory and up to three similar ones, picks ADD / NONE / "
     "UPDATE n / DELETE n as a one-line verdict (processor._decide_and_apply_action). "
     "Parsed by the strict grammar production runs; an unparseable reply counts "
     "as WRONG here because in production it costs a re-ask and then defaults "
     "to keeping the memory.",
     "Accuracy: action AND target must match gold. valid_rate (informational) = "
     "share of replies the strict grammar accepted."),
    ("dedup_structured", "Dedup verdict (structured, experimental)",
     "Same cases, asked as a schema-constrained JSON object "
     "(memory.dedup.structured_output). NOT the production default: this row "
     "exists to decide whether it should be. Displayed, not in the composite.",
     "Accuracy over schema-VALID replies; valid_rate = share of replies that "
     "validated. Enable the toggle only when valid_rate clears ~0.98 on the "
     "model that serves models.reasoning."),
    ("entity", "Entity extraction", "Extracts entities/relationships from text.",
     "F1 of extracted entities vs the expected entity set per item."),
    ("summarization", "Summarization", "Condenses a message into a memory note.",
     "Imported external review, 0-1 (faithful / concise / 3rd-person voice)."),
    ("lessons", "Lessons", "Extracts a reusable insight from a conversation.",
     "Imported external review, 0-1 (grounded / reusable / concise)."),
    ("reasoning", "Reasoning", "Plans, analysis, diagnosis, calibrated explanation.",
     "Imported external review, 0-1 (correct / complete / actionable)."),
    ("code_gen", "Code generation", "Writes code for a stated task (no tools, no sandbox).",
     "Imported external review, 0-1 (correct / complete / idiomatic)."),
    ("coding_agentic", "Coding (agentic)", "Real multi-step coding tasks in a sandbox.",
     "Fraction of verification checks passed (executes code, runs pytest)."),
    ("coding", "Coding (legacy - ambiguous)",
     "Runs before 2026-08-03 reported code generation AND agentic coding under one "
     "task_type, so this column is whichever one happened to be written last.",
     "NOT comparable to the two rows above, or between models. Re-run to replace it."),
    ("tool_calling", "Tool calling", "Picks the right tool with the right arguments.",
     "Exact tool name + required-argument match."),
    ("session_review", "Session review", "Produces a structured session reflection.",
     "Imported external review, 0-1; structure completeness is reported separately."),
    ("session_review_chunked", "Session review (chunked, multi-call)",
     "The path a session too big for the context window takes: chunk-scoped "
     "reflection per part, then merge_chunk_reflections (processor.py:553-577). "
     "Single-chunk cases never reach it.",
     "1.0 when the merged reflection does NOT report work that a later chunk "
     "resolved as 'never addressed' — the invented finding that flows to lessons."),
    ("embedding", "Embedding retrieval", "Embeds queries/memories for semantic search.",
     "Mean of Precision@5 / Recall@10 / MRR on a labeled retrieval set."),
]

# Latency is recorded per SUITE (a group of jobs), not per individual job.
LATENCY_SUITES = ["pipeline", "reasoning_suite", "session_review", "coding", "realdata_suite"]

# Categories that are displayed but must NOT feed the composite or coverage.
# "coding" is the pre-2026-08-03 row where code generation and agentic coding
# shared one task_type: the value is whichever was written last, so averaging it
# in would silently mix two different metrics across models.
NON_COMPARABLE = {"coding"}
# "dedup_structured" measures a path production does not run by default
# (memory.dedup.structured_output); it is displayed so the toggle decision has a
# number, but a model's composite must reflect what it actually does in service.
NON_COMPARABLE = NON_COMPARABLE | {"dedup_structured"}

# Judged jobs whose output length is worth showing as a verbosity cross-check.
JUDGED_JOBS = {"summarization", "lessons", "reasoning", "code_gen", "coding",
               "session_review"}


def _scoring_map(rows: list[dict]) -> dict[str, float]:
    """task_type -> scoring value (accuracy/quality/tool_pass_rate)."""
    out = {}
    for r in rows:
        if r.get("metric") in SCORING_METRICS and r.get("value") is not None:
            out[r["task_type"]] = float(r["value"])
    return out


def _spread_map(rows: list[dict]) -> dict[str, float]:
    """task_type -> spread (max-min across repeats) for scoring metrics.

    merge_repeat_rows computes this and nothing consumed it, so a mean over 5
    repeats rendered identically to a single run. Measured run-to-run spread on
    tool_calling is 0.13-0.20, far above the 0.03 delta that used to count as a
    "gain" — without this, advice recommends noise.
    """
    out: dict[str, float] = {}
    for r in rows:
        if r.get("metric") not in SCORING_METRICS:
            continue
        sp = r.get("spread")
        if isinstance(sp, (int, float)):
            out[r["task_type"]] = float(sp)
    return out


def _partial_map(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """task_type -> (scored, cases) for scores that graded fewer cases than were run.

    A judged score whose calls timed out covers only what completed, and those
    are the easier cases — so the number is biased upward. Surfacing the ratio
    is the difference between "0.62" and "0.62, from 2 of 9 cases".
    """
    out: dict[str, tuple[int, int]] = {}
    for r in rows:
        if r.get("metric") not in SCORING_METRICS:
            continue
        raw = r.get("raw")
        if not isinstance(raw, dict):
            continue
        scored, cases = raw.get("scored"), raw.get("cases")
        if isinstance(scored, int) and isinstance(cases, int) and 0 <= scored < cases:
            out[r["task_type"]] = (scored, cases)
    return out


def _latency_map(rows: list[dict]) -> dict[str, float]:
    """suite (task_type of the latency row) -> mean seconds/call."""
    return {
        r["task_type"]: float(r["value"])
        for r in rows
        if r.get("metric") == "latency_s" and r.get("value") is not None
    }


def _length_map(rows: list[dict]) -> dict[str, float]:
    """task_type -> mean output length (words) for judged jobs (verbosity check)."""
    return {
        r["task_type"]: float(r["value"])
        for r in rows
        if r.get("metric") == "length_words" and r.get("value") is not None
    }


def _completion_map(rows: list[dict]) -> dict[str, float]:
    return {
        r["task_type"]: float(r["value"])
        for r in rows
        if r.get("metric") == "completion_rate" and r.get("value") is not None
    }


def _cohort_signatures(rows: list[dict]) -> list[str]:
    """Exact experiment signatures contributing objective candidate scores."""
    signatures = set()
    for row in rows:
        if row.get("metric") not in SCORING_METRICS or row.get("external_review"):
            continue
        manifest = row.get("benchmark_manifest") or {}
        settings = row.get("run_settings") or {}
        experiment = {
            "benchmark": manifest.get("benchmark_version") or "legacy",
            "dataset": manifest.get("dataset_version") or "legacy",
            "scorer": manifest.get("scorer_version") or "legacy",
            "source_fingerprint": manifest.get("source_fingerprint") or "legacy",
            "git": row.get("git_sha") or "legacy",
            "provider": settings.get("provider") or "legacy",
            "endpoints": settings.get("endpoint_names") or ["legacy"],
            "tier": settings.get("tier") or "legacy",
            "jobs": settings.get("jobs") or ["legacy"],
            "context_tokens": settings.get("context_tokens") or "legacy",
            "timeout_s": settings.get("timeout_s") or "legacy",
            "repeats": settings.get("repeats") or "legacy",
        }
        signatures.add(json.dumps(experiment, sort_keys=True, separators=(",", ":")))
    return sorted(signatures)


def _weighted_composite(scores: dict[str, float], weights: dict[str, float]) -> Optional[float]:
    """Weighted mean over the comparable scoring categories a model measured."""
    num = den = 0.0
    comparable = {c[0] for c in CATEGORIES} - NON_COMPARABLE
    for key, val in scores.items():
        if key not in comparable:
            continue  # informational task_types + non-comparable legacy rows
        w = weights.get(key, 1.0)
        num += w * val
        den += w
    return round(num / den, 4) if den else None


def build_report(
    model_rows: dict[str, list[dict]],
    *,
    catalog: Optional[dict[str, dict]] = None,
    judge_model: Optional[str] = None,
    generated_ts: str = "",
    task_weights: Optional[dict[str, float]] = None,
    provenance: Optional[dict[str, dict]] = None,
    required_categories: Optional[set[str]] = None,
    system_evidence: Optional[dict] = None,
) -> dict:
    """Turn stored metric rows for every model into one structured report. Pure.

    ``provenance`` is model -> {run_ts, git_sha, host, tier, judge_model}. It is
    rendered as its own table because results now persist indefinitely across
    machines: two numbers in the same column can come from different weeks and
    different code, and a comparison that hides that is worse than no
    comparison.
    """
    catalog = catalog or {}
    task_weights = task_weights or {}
    provenance = provenance or {}
    models = sorted(model_rows)

    scoring = {m: _scoring_map(rows) for m, rows in model_rows.items()}
    latency = {m: _latency_map(rows) for m, rows in model_rows.items()}
    length = {m: _length_map(rows) for m, rows in model_rows.items()}
    completion = {m: _completion_map(rows) for m, rows in model_rows.items()}
    cohort_signatures = {m: _cohort_signatures(rows) for m, rows in model_rows.items()}
    partial = {m: _partial_map(rows) for m, rows in model_rows.items()}
    spreads = {m: _spread_map(rows) for m, rows in model_rows.items()}

    categories = []
    for key, label, measures, method in CATEGORIES:
        scores = {m: scoring[m][key] for m in models if key in scoring[m]}
        if not scores:
            continue  # nothing measured this category yet
        # Models whose score for this job covers fewer cases than were run.
        # They cannot win the row: their average is over the subset that didn't
        # time out, which is the easier subset.
        incomplete = {m: partial[m][key] for m in models if key in partial.get(m, {})}
        effective = {m: v * completion[m].get(key, 1.0) for m, v in scores.items()}
        eligible = {m: v for m, v in effective.items() if m not in incomplete}
        categories.append({
            "key": key, "label": label, "measures": measures, "scoring": method,
            "scores": {m: round(v, 4) for m, v in scores.items()},
            "effective_scores": {m: round(v, 4) for m, v in effective.items()},
            # Per-model spread for THIS job, so the verdict can require a gain
            # to clear measured noise instead of a fixed 0.03 constant.
            "spread": {m: spreads[m][key] for m in models
                       if key in spreads.get(m, {})},
            "incomplete": incomplete,
            "best_model": (max(eligible, key=lambda m: eligible[m]) if eligible
                           else max(scores, key=lambda m: scores[m])),
        })

    # Coverage = how many scoring categories this model actually measured.
    # A composite over 1 of 10 jobs is not comparable to one over 10, and left
    # unmarked the partial model can top the table (kimi-k2.7-code briefly
    # showed the best composite, 0.925, from a single session_review run).
    # Production callers pass the versioned manifest's fixed set.  Keeping a
    # dynamic fallback makes this pure helper useful for small ad-hoc reports
    # and keeps pre-v2 consumers readable without pretending they are v2 runs.
    cat_keys = (set(required_categories) if required_categories is not None
                else ({c[0] for c in CATEGORIES} - NON_COMPARABLE))
    coverage = {m: len(set(scoring[m]) & cat_keys) for m in models}
    max_coverage = (len(cat_keys) if required_categories is not None
                    else (max(coverage.values()) if coverage else 0))

    composite = {}
    for m in models:
        effective_scores = {
            key: value * completion[m].get(key, 1.0)
            for key, value in scoring[m].items()
        }
        c = _weighted_composite(effective_scores, task_weights)
        if c is not None:
            composite[m] = c

    cat_info = {}
    for m in models:
        info = catalog.get(m) or {}
        cat_info[m] = {
            "price_in": info.get("price_in"),
            "price_out": info.get("price_out"),
            "tok_per_s": info.get("tok_per_s"),
            "context_length": info.get("context_length"),
        }

    return {
        "generated": generated_ts,
        "judge_model": judge_model,
        "models": models,
        "categories": categories,
        "composite": composite,
        "latency": latency,           # model -> {suite: seconds}
        "length": length,             # model -> {task_type: mean words} (judged jobs)
        "completion": completion,
        "catalog": cat_info,          # model -> {price/context/speed}
        "provenance": {m: provenance.get(m, {}) for m in models},
        "coverage": coverage,         # model -> categories measured
        "max_coverage": max_coverage,
        "required_categories": sorted(cat_keys),
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_version": DATASET_VERSION,
        "pending_external_review": ({
            m: sorted(OPEN_ENDED_CATEGORIES - set(scoring[m])) for m in models
        } if required_categories is not None else {}),
        "system_evidence": system_evidence or {},
        "cohort_signatures": cohort_signatures,
        "comparison_contract_enforced": required_categories is not None,
    }


# --------------------------------------------------------------------------- md

def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(report: dict, advice: str = "") -> str:
    """Render the report. `advice` is the per-config-key section from
    advice.render_advice(); it goes FIRST because it's the part you act on —
    the per-job matrix below it is supporting detail."""
    models = report["models"]
    if not report["categories"]:
        return ("# BlipShell model benchmark\n\nNo benchmarked models yet. "
                "Run `blipshell benchmark run <model>` to populate this report.\n")

    parts: list[str] = []

    parts.append("# BlipShell model benchmark")
    if report.get("generated"):
        parts.append(f"_Generated {report['generated']}_")
    parts.append("")
    if advice:
        parts.append(advice)
        parts.append("---")
        parts.append("")
    parts.append(
        "## How to read this\n"
        "BlipShell routes a **mix of local and cloud models per job**, each with a fallback, "
        "chosen by endpoint priority and availability. Cloud is generally strongest but we do "
        "NOT need cloud for every job — the point of this benchmark is to find, per job, the "
        "cheapest model that's good enough. This report makes **no switch recommendation**: it "
        "lays out quality and speed (and cost/context for cloud models) per job so you can decide. "
        "Higher quality = better; lower latency = faster. Objective jobs are scored locally; "
        "open-ended jobs remain blank until a blinded external review is imported. COMPOSITE "
        "uses effective score (quality multiplied by completion rate)."
    )
    parts.append(
        f"\nBenchmark contract: **v{report.get('benchmark_version', '?')}**, dataset "
        f"**{report.get('dataset_version', '?')}**. Required coverage is fixed at "
        f"{report.get('max_coverage', 0)} production categories."
    )
    if report.get("judge_model"):
        parts.append(
            f"\nLegacy imported scores in this report name **{report['judge_model']}** as "
            "their automatic judge. New v2 runs never call a judge API."
        )
    parts.append("")

    # Quality table
    parts.append("## Quality by job (higher is better)")
    header = ["Job"] + models
    qrows = []
    any_incomplete = False
    for c in report["categories"]:
        row = [c["label"]]
        for m in models:
            v = c["scores"].get(m)
            inc = (c.get("incomplete") or {}).get(m)
            # Show the repeat spread next to the mean. A bare 0.933 reads as
            # precise; "0.933 +/-0.13" says a rerun could land at 0.80, which
            # is the difference between a decision and a coin flip.
            sp = (c.get("spread") or {}).get(m)
            suffix = f" +/-{sp:.2f}" if isinstance(sp, (int, float)) and sp > 0 else ""
            rate = report.get("completion", {}).get(m, {}).get(c["key"])
            if isinstance(rate, (int, float)) and rate < 1.0:
                suffix += f" @ {rate:.0%} complete"
            if v is None:
                row.append("—")
            elif inc:
                any_incomplete = True
                row.append(f"{v:.3f} ({inc[0]}/{inc[1]} cases)")
            elif m == c["best_model"]:
                row.append(f"**{v:.3f}**{suffix}")
            else:
                row.append(f"{v:.3f}{suffix}")
        qrows.append(row)
    comp = report["composite"]
    cov = report.get("coverage", {})
    max_cov = report.get("max_coverage", 0)
    partial_models = [m for m in models if cov.get(m, 0) < max_cov]
    if comp:
        # Only models with full coverage compete for "best composite" — a
        # partial model's average is over a different (easier) set of jobs.
        full = {m: v for m, v in comp.items() if cov.get(m, 0) >= max_cov}
        best_comp = max(full, key=lambda m: full[m]) if full else None
        crow = ["**COMPOSITE**"]
        for m in models:
            v = comp.get(m)
            if v is None:
                crow.append("—")
            elif m in partial_models:
                crow.append(f"{v:.3f} (partial)")   # measured fewer jobs
            elif m == best_comp:
                crow.append(f"**{v:.3f}**")
            else:
                crow.append(f"{v:.3f}")
        qrows.append(crow)
    parts.append(_md_table(header, qrows))
    if any_incomplete:
        parts.append(
            "\n**(n/m cases)** means some calls for that job failed — almost always "
            "a timeout — so the score covers only the cases that finished. Those are "
            "the shorter, easier ones, so the number is biased UPWARD and cannot win "
            "its row. Raise `llm.timeout`, or reduce the candidate's `num_ctx` if "
            "generation is slow, then re-run."
        )
    if partial_models:
        detail = ", ".join(f"{m} ({cov.get(m, 0)}/{max_cov} jobs)" for m in partial_models)
        parts.append(
            f"\n**(partial) Incomplete coverage:** {detail}. A composite averaged over "
            "fewer jobs is NOT comparable to a full one and cannot win the row — "
            "a model that only ran its strongest job would otherwise top the "
            "table. Blank cells mean 'not measured', never 'scored zero'. "
            "Re-run those models across all jobs before comparing composites."
        )
    parts.append("")

    signatures = report.get("cohort_signatures") or {}
    comparable_groups: dict[str, list[str]] = {}
    for model, sigs in signatures.items():
        if len(sigs) == 1 and '"benchmark":"legacy"' not in sigs[0]:
            comparable_groups.setdefault(sigs[0], []).append(model)
    cohorts = [group for group in comparable_groups.values() if len(group) > 1]
    if cohorts:
        parts.append("## Same-configuration comparison cohorts")
        parts.append("")
        parts.append(_md_table(
            ["Cohort", "Models"],
            [[str(i), ", ".join(group)] for i, group in enumerate(cohorts, 1)],
        ))
        parts.append("")
    elif any(signatures.values()):
        parts.append(
            "**No same-configuration comparison cohort exists yet.** Run the candidate set "
            "from the same Git commit with the same tier, context window, repeat count, "
            "benchmark version, and dataset version before making a routing decision."
        )
        parts.append("")

    evidence = report.get("system_evidence") or {}
    agent_rows = evidence.get("agent_eval") or []
    continuity_rows = evidence.get("continuity") or []
    if agent_rows or continuity_rows:
        parts.append("## Adjacent system evidence")
        parts.append(
            "These suites are kept separate because they measure a different boundary, but are "
            "included here so a routing decision does not ignore agent behavior or continuity. "
            "Model identifiers are shown exactly as recorded; similarly named serving stacks are "
            "not silently merged."
        )
        parts.append("")
        if agent_rows:
            parts.append("### Multi-turn agent evaluation")
            parts.append("")
            parts.append(_md_table(
                ["Model", "Score", "Silent", "Turn limit", "API errors", "s/episode", "Run"],
                [[r["model"], f"{r.get('score', '—')}/{r.get('max', '—')}",
                  str(r.get("silent", "—")), str(r.get("turn_limit", "—")),
                  str(r.get("api_errors", "—")), str(r.get("seconds_per_episode", "—")),
                  str(r.get("run_date", "—"))] for r in agent_rows],
            ))
            parts.append("")
        if continuity_rows:
            parts.append("### Behavioral continuity")
            parts.append("")
            parts.append(_md_table(
                ["Model", "Passed", "Failed", "Scored"],
                [[r["model"], str(r["passed"]), str(r["failed"]), str(r["scored"])]
                 for r in continuity_rows],
            ))
            parts.append("")

    pending = report.get("pending_external_review") or {}
    waiting = {m: jobs for m, jobs in pending.items() if jobs}
    if waiting:
        parts.append("## Awaiting external review")
        parts.append(
            "These open-ended jobs have no imported review score. A new v2 run creates "
            "the review sources needed by `blipshell benchmark review export`; review the "
            "packet in ChatGPT, Claude, or manually, then use "
            "`blipshell benchmark review import RESPONSE.json`."
        )
        parts.append("")
        parts.append(_md_table(
            ["Model", "Unscored open-ended jobs"],
            [[m, ", ".join(jobs)] for m, jobs in waiting.items()],
        ))
        parts.append("")

    # Latency table
    lat = report["latency"]
    suites = [s for s in LATENCY_SUITES if any(s in lat.get(m, {}) for m in models)]
    if suites:
        parts.append("## Latency by suite — mean seconds/call (lower is faster)")
        lrows = []
        for s in suites:
            vals = {m: lat.get(m, {}).get(s) for m in models}
            present = {m: v for m, v in vals.items() if v is not None}
            fastest = min(present, key=lambda m: present[m]) if present else None
            row = [s]
            for m in models:
                v = vals[m]
                row.append("—" if v is None else (f"**{v:.2f}**" if m == fastest else f"{v:.2f}"))
            lrows.append(row)
        parts.append(_md_table(["Suite"] + models, lrows))
        parts.append("\n_Latency is measured per suite, so jobs in the same suite share a number "
                     "(pipeline = ranking/importance/contradiction/entity/summarization/lessons; "
                     "reasoning_suite = reasoning/coding-gen/tool_calling)._")
        parts.append("")

    # Cost / context (cloud)
    cat = report["catalog"]
    if any(any(v is not None for v in cat[m].values()) for m in models):
        parts.append("## Cost & context (from discovery catalog, where known)")
        crows = []
        for m in models:
            info = cat[m]
            pin = info.get("price_in")
            pout = info.get("price_out")
            price = f"${pin:.2f}/${pout:.2f}" if (pin is not None or pout is not None) else "— (local)"
            tps = f"{info['tok_per_s']:.0f}" if info.get("tok_per_s") else "—"
            ctx = f"{info['context_length']:,}" if info.get("context_length") else "—"
            crows.append([m, price, tps, ctx])
        parts.append(_md_table(["Model", "$/1M in/out", "tok/s", "context"], crows))
        parts.append("")

    # Output length (verbosity check) — judged jobs only
    length = report.get("length", {})
    judged = [c[0] for c in CATEGORIES if c[0] in JUDGED_JOBS]
    len_jobs = [j for j in judged if any(j in length.get(m, {}) for m in models)]
    if len_jobs:
        parts.append("## Output length (mean words, externally reviewed jobs) — longer is NOT better")
        lrows = []
        for j in len_jobs:
            label = next((c[1] for c in CATEGORIES if c[0] == j), j)
            row = [label]
            for m in models:
                v = length.get(m, {}).get(j)
                row.append(f"{v:.0f}" if v is not None else "—")
            lrows.append(row)
        parts.append(_md_table(["Job"] + models, lrows))
        parts.append("\n_External-review rubrics do not reward length. If a model wins while writing "
                     "far more, inspect the packet and rationale before acting._")
        parts.append("")

    # Provenance — when/where/what-code produced each column
    prov = report.get("provenance", {})
    if any(prov.get(m) for m in models):
        parts.append("## Provenance — when each column was measured")
        prows = []
        for m in models:
            p = prov.get(m) or {}
            ts = (p.get("run_ts") or "—")[:16].replace("T", " ")
            # Scope is derived from measured coverage, NOT from the run's --jobs
            # field. Rows merge per metric, so a model can reach full coverage
            # from several scoped runs; the recorded intent of any single run
            # would then misdescribe the column. Coverage cannot drift from
            # what the numbers actually are.
            n = cov.get(m, 0)
            scope = "all jobs" if n >= max_cov and max_cov else f"{n}/{max_cov} jobs"
            # Distinguish "migrated from the old DB, commit unknowable" from
            # "git was unavailable" — both would otherwise render as a dash,
            # and the reader needs to know which kind of unknown it is.
            if p.get("git_sha"):
                code = p["git_sha"]
            elif p.get("migrated_from_db"):
                code = "pre-migration"
            else:
                code = "—"
            # Rows are merged per metric, so one model's numbers can span runs.
            # Show the span rather than pretending it's a single measurement.
            oldest = (p.get("oldest_ts") or "")[:10]
            newest = (p.get("run_ts") or "")[:10]
            span = newest if (not oldest or oldest == newest) else f"{oldest}..{newest}"
            if p.get("mixed_code"):
                code += " (mixed)"
            version = p.get("benchmark_version") or "legacy"
            dataset = p.get("dataset_version") or "legacy"
            if p.get("mixed_benchmark") or p.get("mixed_dataset"):
                version += " (mixed)"
            scorer = p.get("scorer_version") or "legacy"
            fingerprint = p.get("source_fingerprint") or "legacy"
            if p.get("mixed_scorer") or p.get("mixed_source"):
                scorer += " (mixed)"
            prows.append([m, ts, span, str(p.get("run_count") or 1), code,
                          version, dataset, scorer, fingerprint, scope])
        parts.append(_md_table(
            ["Model", "Newest run (UTC)", "Data spans", "Runs", "Code (git)",
             "Benchmark", "Dataset", "Scorer", "Source", "Scope"], prows))
        parts.append("\n_Results persist across machines and months. Columns measured at "
                     "different dates or different commits are NOT strictly comparable — a "
                     "scorer or prompt change between them moves scores on its own. When two "
                     "rows disagree and their commits differ, re-run the older one before "
                     "concluding anything._")
        if partial_models:
            parts.append("\n_A model showing fewer than all jobs did not measure every "
                         "category — usually a `--jobs`-scoped run. Its blank cells mean "
                         "'not run', not 'scored zero', and its COMPOSITE averages over "
                         "fewer jobs than a full run's, so do not compare composites across "
                         "differing scopes._")
        parts.append("")

    # Methodology
    parts.append("## Methodology — what each job measures and how it's scored")
    mrows = [[c["label"], c["measures"], c["scoring"]] for c in report["categories"]]
    parts.append(_md_table(["Job", "Measures", "Scoring"], mrows))
    parts.append("")

    # Caveats
    parts.append(
        "## Caveats\n"
        "- Latency is a per-suite mean, not per individual job; cloud latency includes network.\n"
        "- Open-ended quality is absent until an external review is imported. Review files name "
        "the reviewer and remain independently replaceable; candidate inference is never rerun.\n"
        "- Compare models from the same benchmark version, dataset version, Git commit, tier, "
        "context window, and repeat count. Mixed or legacy columns are historical context only.\n"
        "- Agentic coding executes real tasks; a model with weak tool-calling will score low there "
        "even if its raw code quality is fine — that's intended (it reflects real executor use).\n"
        "- Embedding needs a labeled retrieval set in the DB; it's blank if unavailable."
    )
    return "\n".join(parts) + "\n"


def write_report(report: dict, out_dir: str, advice: str = "") -> dict:
    """Write report.md + report.json into out_dir. Returns {md, json} paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    json_path = out / "report.json"
    md_path.write_text(render_markdown(report, advice=advice), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"md": str(md_path), "json": str(json_path)}
