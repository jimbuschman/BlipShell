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

# Scoring metrics (higher = better). Everything else (latency, agreement) is
# informational and shown separately.
SCORING_METRICS = {"accuracy", "quality", "tool_pass_rate"}

# The job categories, in display order, with what each measures + how it's scored
# (this text goes into the report so the reading LLM knows what the numbers mean).
CATEGORIES = [
    ("ranking", "Ranking (1-5)", "Assigns a memory-importance rank to messages.",
     "Rank-correlation of predicted vs gold rank (order matters); flat output scores ~0."),
    ("importance", "Importance (0-1)", "Scores how important a memory is to retain.",
     "Average of correlation and calibration (1-MAE) vs gold importance."),
    ("rank_importance", "Rank+Importance", "Combined rank and importance in one call.",
     "Mean of the ranking and importance scores above."),
    ("contradiction", "Contradiction", "Decides whether two memories contradict.",
     "Exact YES/NO accuracy over a balanced set."),
    ("entity", "Entity extraction", "Extracts entities/relationships from text.",
     "F1 of extracted entities vs the expected entity set per item."),
    ("summarization", "Summarization", "Condenses a message into a memory note.",
     "Neutral judge 0-1 (faithful / concise / 3rd-person voice)."),
    ("lessons", "Lessons", "Extracts a reusable insight from a conversation.",
     "Neutral judge 0-1 (grounded / reusable / concise)."),
    ("reasoning", "Reasoning", "Plans, analysis, diagnosis, calibrated explanation.",
     "Neutral judge 0-1 (correct / complete / actionable)."),
    ("coding", "Coding (agentic)", "Real multi-step coding tasks in a sandbox.",
     "Fraction of verification checks passed (executes code, runs pytest)."),
    ("tool_calling", "Tool calling", "Picks the right tool with the right arguments.",
     "Exact tool name + required-argument match."),
    ("session_review", "Session review", "Produces a structured session reflection.",
     "Neutral judge 0-1 (or section-completeness if no judge)."),
    ("embedding", "Embedding retrieval", "Embeds queries/memories for semantic search.",
     "Mean of Precision@5 / Recall@10 / MRR on a labeled retrieval set."),
]

# Latency is recorded per SUITE (a group of jobs), not per individual job.
LATENCY_SUITES = ["pipeline", "reasoning_suite", "session_review", "coding", "realdata_suite"]


def _scoring_map(rows: list[dict]) -> dict[str, float]:
    """task_type -> scoring value (accuracy/quality/tool_pass_rate)."""
    out = {}
    for r in rows:
        if r.get("metric") in SCORING_METRICS and r.get("value") is not None:
            out[r["task_type"]] = float(r["value"])
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


def _weighted_composite(scores: dict[str, float], weights: dict[str, float]) -> Optional[float]:
    """Weighted mean over the scoring categories a model measured."""
    num = den = 0.0
    for key, val in scores.items():
        if key not in {c[0] for c in CATEGORIES}:
            continue  # exclude informational task_types (e.g. realdata agreement)
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

    categories = []
    for key, label, measures, method in CATEGORIES:
        scores = {m: scoring[m][key] for m in models if key in scoring[m]}
        if not scores:
            continue  # nothing measured this category yet
        categories.append({
            "key": key, "label": label, "measures": measures, "scoring": method,
            "scores": {m: round(v, 4) for m, v in scores.items()},
            "best_model": max(scores, key=lambda m: scores[m]),
        })

    composite = {}
    for m in models:
        c = _weighted_composite(scoring[m], task_weights)
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
        "catalog": cat_info,          # model -> {price/context/speed}
        "provenance": {m: provenance.get(m, {}) for m in models},
    }


# --------------------------------------------------------------------------- md

def _md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(report: dict) -> str:
    models = report["models"]
    if not report["categories"]:
        return ("# BlipShell model benchmark\n\nNo benchmarked models yet. "
                "Run `blipshell benchmark run <model>` to populate this report.\n")

    parts: list[str] = []

    parts.append("# BlipShell model benchmark")
    if report.get("generated"):
        parts.append(f"_Generated {report['generated']}_")
    parts.append("")
    parts.append(
        "## How to read this\n"
        "BlipShell routes a **mix of local and cloud models per job**, each with a fallback, "
        "chosen by endpoint priority and availability. Cloud is generally strongest but we do "
        "NOT need cloud for every job — the point of this benchmark is to find, per job, the "
        "cheapest model that's good enough. This report makes **no switch recommendation**: it "
        "lays out quality and speed (and cost/context for cloud models) per job so you can decide. "
        "Higher quality = better; lower latency = faster. Quality scores are deterministic or "
        "neutral-judge graded (see Methodology) and are designed to discriminate capable models, "
        "not saturate."
    )
    if report.get("judge_model"):
        parts.append(f"\nOpen-ended tasks were graded by a neutral judge: **{report['judge_model']}** "
                     "(not one of the candidates).")
    parts.append("")

    # Quality table
    parts.append("## Quality by job (higher is better)")
    header = ["Job"] + models
    qrows = []
    for c in report["categories"]:
        row = [c["label"]]
        for m in models:
            v = c["scores"].get(m)
            if v is None:
                row.append("—")
            elif m == c["best_model"]:
                row.append(f"**{v:.3f}**")
            else:
                row.append(f"{v:.3f}")
        qrows.append(row)
    comp = report["composite"]
    if comp:
        best_comp = max(comp, key=lambda m: comp[m])
        crow = ["**COMPOSITE**"]
        for m in models:
            v = comp.get(m)
            crow.append("—" if v is None else (f"**{v:.3f}**" if m == best_comp else f"{v:.3f}"))
        qrows.append(crow)
    parts.append(_md_table(header, qrows))
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
    judged = [c[0] for c in CATEGORIES if c[0] in
              {"summarization", "lessons", "reasoning", "coding", "session_review"}]
    len_jobs = [j for j in judged if any(j in length.get(m, {}) for m in models)]
    if len_jobs:
        parts.append("## Output length (mean words, judged jobs) — longer is NOT better")
        lrows = []
        for j in len_jobs:
            label = next((c[1] for c in CATEGORIES if c[0] == j), j)
            row = [label]
            for m in models:
                v = length.get(m, {}).get(j)
                row.append(f"{v:.0f}" if v is not None else "—")
            lrows.append(row)
        parts.append(_md_table(["Job"] + models, lrows))
        parts.append("\n_The judge is instructed not to reward length, but check this: if a model "
                     "wins a judged job while writing far more, treat the margin with suspicion._")
        parts.append("")

    # Provenance — when/where/what-code produced each column
    prov = report.get("provenance", {})
    if any(prov.get(m) for m in models):
        parts.append("## Provenance — when each column was measured")
        prows = []
        partial = False
        for m in models:
            p = prov.get(m) or {}
            ts = (p.get("run_ts") or "—")[:16].replace("T", " ")
            jobs = p.get("jobs")
            if jobs:
                scope = ", ".join(jobs)
                partial = True
            else:
                scope = "all jobs"
            # Distinguish "migrated from the old DB, commit unknowable" from
            # "git was unavailable" — both would otherwise render as a dash,
            # and the reader needs to know which kind of unknown it is.
            if p.get("git_sha"):
                code = p["git_sha"]
            elif p.get("migrated_from_db"):
                code = "pre-migration"
            else:
                code = "—"
            prows.append([
                m, ts, code, p.get("host") or "—",
                p.get("judge_model") or "—", scope,
            ])
        parts.append(_md_table(
            ["Model", "Run date (UTC)", "Code (git)", "Host", "Judge", "Scope"], prows))
        parts.append("\n_Results persist across machines and months. Columns measured at "
                     "different dates or different commits are NOT strictly comparable — a "
                     "scorer or prompt change between them moves scores on its own. When two "
                     "rows disagree and their commits differ, re-run the older one before "
                     "concluding anything._")
        if partial:
            parts.append("\n_At least one model was benchmarked with `--jobs` and so did not "
                         "measure every category. Its blank cells above mean 'not run', not "
                         "'scored zero', and its COMPOSITE is an average over fewer jobs than "
                         "a full run's — do not compare composites across differing scopes._")
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
        "- Quality for open-ended jobs depends on the judge model noted above. The judge is told "
        "not to reward length, but cross-check the output-length table — a much wordier winner is "
        "a verbosity-bias smell.\n"
        "- Agentic coding executes real tasks; a model with weak tool-calling will score low there "
        "even if its raw code quality is fine — that's intended (it reflects real executor use).\n"
        "- Embedding needs a labeled retrieval set in the DB; it's blank if unavailable."
    )
    return "\n".join(parts) + "\n"


def write_report(report: dict, out_dir: str) -> dict:
    """Write report.md + report.json into out_dir. Returns {md, json} paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    json_path = out / "report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"md": str(md_path), "json": str(json_path)}
