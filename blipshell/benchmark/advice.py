"""Per-config-key assignment advice — the part of the report you actually act on.

The report's rows are benchmark jobs; the thing you edit is a key in
`config.yaml`. Those are not one-to-one, and the mismatch is what made the
report unreadable: `models.session_review` drives session review AND lesson
extraction, so kimi-k2.7-code can lead one column (0.925) and come last in
another (0.395) with nothing in the table connecting them. Answering "which
model should I use where" meant reading processor.py by hand.

This module inverts it: one block per config key, listing every job that key
controls, the incumbent's scores, each candidate's delta, and — when the answer
isn't yet knowable — the single command that would make it knowable.

JOB_OWNERS is derived from real `TaskType.*` call sites, not from config.yaml's
comments, which have already drifted (the `reasoning:` comment claims it handles
lessons; lessons actually route through SESSION_REVIEW at processor.py:275).
"""

from __future__ import annotations

from typing import Optional

# config key -> (benchmark job keys it controls, what it does in production)
#
# Verified 2026-08-03 by grepping TaskType usage across blipshell/ (excluding the
# benchmark package and the router's own map). Re-verify when routing changes.
#
# A key lists every job whose quality it determines, not just the job sharing its
# name. `tool_calling` serves the whole interactive path, so it owns reasoning and
# code_gen too: with only the tool_calling job attached, lfm2.5 (0.933 tool
# calling, 0.450 reasoning) read as a CONSIDER for interactive chat — the exact
# win-one-lose-another trap this module exists to catch. Under-specifying a key
# is therefore not a cosmetic error; it produces confidently wrong advice.
JOB_OWNERS: dict[str, tuple[tuple[str, ...], str]] = {
    "tool_calling": (
        ("tool_calling", "reasoning", "code_gen"),
        "interactive chat + executor tool loop (agent_chat, executor, planner)",
    ),
    "coding": (
        ("coding_agentic", "code_gen"),
        "project-mode coding (cli project path, background coding tasks)",
    ),
    "reasoning": (
        ("reasoning", "entity", "contradiction"),
        "entity extraction + merge, contradiction checks, tag discovery, "
        "project digests, guardrail audits, self-thought relevance judge, "
        "context compaction",
    ),
    "summarization": (
        ("summarization",),
        "memory + session summaries, web-fetch summaries, imports",
    ),
    "ranking_importance": (
        ("rank_importance",),
        "the live memory pipeline's combined rank+importance call",
    ),
    "ranking": (
        ("ranking",),
        "batch tagger + import path only (not the live pipeline)",
    ),
    "importance": (
        ("importance",),
        "import path only (live pipeline uses ranking_importance)",
    ),
    "session_review": (
        ("session_review", "lessons", "session_review_chunked"),
        "session reflections (incl. the chunk+merge path for oversized "
        "sessions), LESSON EXTRACTION, friction analysis",
    ),
    "embedding": (
        ("embedding",),
        "all vector search (memories, lessons, core, entities, self-thoughts)",
    ),
}

# Which suite a job belongs to, so the advice can name the exact --jobs value
# that would measure a missing number.
JOB_SUITE = {
    "ranking": "pipeline", "importance": "pipeline", "rank_importance": "pipeline",
    "contradiction": "pipeline", "entity": "pipeline", "summarization": "pipeline",
    "lessons": "pipeline", "reasoning": "reasoning", "code_gen": "reasoning",
    "coding_agentic": "coding", "tool_calling": "reasoning",
    "session_review": "session_review",
    "session_review_chunked": "session_review", "embedding": "embedding",
}

# Fallback noise floor, used ONLY when a run has no measured spread (--repeats 1).
# Judge scores wobble between runs (kimi's session_review moved 0.950 -> 0.925
# across a scorer change alone), so a smaller margin is noise, not signal.
# When repeats ARE available, _noise_floor uses the measured spread instead:
# 0.03 turned out to be well below the real floor (tool_calling spread 0.13-0.20),
# which is how a coin flip could be reported as a clean win.
MEANINGFUL_DELTA = 0.03


def _noise_floor(job: str, spreads: dict, incumbent: str, candidate: str) -> float:
    """Smallest delta on `job` that is not plausibly run-to-run noise.

    A fixed 0.03 was BELOW the real noise floor. Measured on 2026-08-18,
    tool_calling repeats gave sd 0.042-0.089 (spread 0.13-0.20) on a single
    model — so a 0.03 "gain" was regularly a coin flip, and the harness would
    report it as a clean win. When repeats supply a spread for both models we
    require the delta to clear the mean of the two spreads; otherwise we fall
    back to the constant and the reader is told the run was unreplicated.
    """
    per_job = spreads.get(job) or {}
    observed = [per_job[m] for m in (incumbent, candidate) if isinstance(per_job.get(m), (int, float))]
    if not observed:
        return MEANINGFUL_DELTA
    return max(MEANINGFUL_DELTA, sum(observed) / len(observed))


def current_assignments(config) -> dict[str, str]:
    """config key -> the model actually serving it, per-endpoint overrides applied.

    A global `models.x` can be overridden by whichever endpoint wins the role,
    so reading ModelsConfig alone would report a model that never runs. Picks
    the highest-priority enabled endpoint offering the role, mirroring
    EndpointManager.get_endpoint_for_role's ordering.
    """
    out: dict[str, str] = {}
    for key in JOB_OWNERS:
        model = getattr(config.models, key, None)
        candidates = [
            ep for ep in config.endpoints
            if ep.enabled and key in ep.roles and ep.models.get(key)
        ]
        if candidates:
            best = max(candidates, key=lambda e: e.priority)
            model = best.models[key]
        out[key] = model or "(unset)"
    return out


def build_advice(report: dict, config) -> list[dict]:
    """One advice block per config key. Pure: report + config in, list out."""
    assignments = current_assignments(config)
    scores: dict[str, dict[str, float]] = {
        c["key"]: c["scores"] for c in report.get("categories", [])
    }
    spreads: dict[str, dict[str, float]] = {
        c["key"]: c.get("spread", {}) for c in report.get("categories", [])
    }
    coverage = report.get("coverage", {})
    models = report.get("models", [])

    blocks = []
    for key, (jobs, purpose) in JOB_OWNERS.items():
        incumbent = assignments[key]

        # Per-job scores for every model that measured at least one of this
        # key's jobs. Missing entries stay missing — never zero.
        rows = []
        for m in models:
            per_job = {j: scores.get(j, {}).get(m) for j in jobs}
            if all(v is None for v in per_job.values()):
                continue
            rows.append({"model": m, "jobs": per_job,
                         "coverage": coverage.get(m, 0)})

        inc_row = next((r for r in rows if r["model"] == incumbent), None)
        unmeasured = [j for j in jobs
                      if not inc_row or inc_row["jobs"].get(j) is None]

        # A candidate is only interesting if it beats the incumbent on at least
        # one of this key's jobs and loses on none of them by more than the
        # noise floor. A model that wins one job and craters another is the
        # kimi/lessons trap and must not read as an upgrade.
        contenders = []
        for r in rows:
            if r["model"] == incumbent:
                continue
            gains, losses = [], []
            for j in jobs:
                cand, inc = r["jobs"].get(j), (inc_row or {}).get("jobs", {}).get(j)
                if cand is None or inc is None:
                    continue
                d = cand - inc
                floor = _noise_floor(j, spreads, incumbent, r["model"])
                if d > floor:
                    gains.append((j, d))
                elif d < -floor:
                    losses.append((j, d))
            if gains or losses:
                contenders.append({**r, "gains": gains, "losses": losses})

        # Verdict, in priority order: can't tell > clear win > mixed > keep.
        if unmeasured:
            suites = sorted({JOB_SUITE.get(j, j) for j in unmeasured})
            verdict = "UNKNOWN"
            reason = (f"{incumbent} has no score for "
                      f"{', '.join(unmeasured)} -- the key's own job(s).")
            action = (f"blipshell benchmark run {incumbent} "
                      f"--jobs {','.join(suites)}")
        else:
            clear = [c for c in contenders if c["gains"] and not c["losses"]]
            mixed = [c for c in contenders if c["gains"] and c["losses"]]
            if clear:
                # Rank by MEAN delta across every job the key controls, not by
                # the single largest gain. A key is one choice covering all its
                # jobs, so a candidate that improves all of them beats one that
                # spikes on a single job and slips on another. Real case:
                # glm-5.2 (lessons +0.238, session_review -0.025) outranked
                # minimax-m3 (+0.165, +0.100) under max-gain, despite minimax
                # being better on BOTH.
                def _mean_delta(c):
                    inc_jobs = (inc_row or {}).get("jobs", {})
                    deltas = [c["jobs"][j] - inc_jobs[j] for j in jobs
                              if c["jobs"].get(j) is not None
                              and inc_jobs.get(j) is not None]
                    return sum(deltas) / len(deltas) if deltas else 0.0

                best = max(clear, key=_mean_delta)
                improved = ", ".join(f"{j} (+{d:.3f})" for j, d in best["gains"])
                verdict = "CONSIDER"
                reason = (f"{best['model']} beats {incumbent} on {improved} "
                          f"with no regression on this key's other jobs "
                          f"(mean gain across all its jobs: "
                          f"{_mean_delta(best):+.3f}).")
                action = None
            elif mixed:
                m = mixed[0]
                verdict = "KEEP"
                reason = (f"{m['model']} wins "
                          + ", ".join(f"{j} (+{d:.3f})" for j, d in m["gains"])
                          + " but loses "
                          + ", ".join(f"{j} ({d:.3f})" for j, d in m["losses"])
                          + " -- this key controls both, so it is not an upgrade.")
                action = None
            else:
                verdict = "KEEP"
                floors = [_noise_floor(j, spreads, incumbent, r["model"])
                          for j in jobs for r in rows if r["model"] != incumbent]
                worst = max(floors) if floors else MEANINGFUL_DELTA
                reason = (f"No measured candidate beats {incumbent} by more than "
                          f"the noise floor ({worst:.3f}) on this key's jobs.")
                action = None

        blocks.append({
            "key": key, "incumbent": incumbent, "jobs": list(jobs),
            "purpose": purpose, "rows": rows, "verdict": verdict,
            "reason": reason, "action": action, "unmeasured": unmeasured,
        })
    return blocks


def render_advice(blocks: list[dict]) -> str:
    """ASCII-only markdown for the top of report.md (cp1252 consoles)."""
    if not blocks:
        return ""
    order = {"UNKNOWN": 0, "CONSIDER": 1, "KEEP": 2}
    blocks = sorted(blocks, key=lambda b: (order.get(b["verdict"], 3), b["key"]))

    out = ["## Which model to use where",
           "",
           "One block per key in `config.yaml`. **Every job a key controls is "
           "listed together**, because a key is a single choice: a model that "
           "wins one of its jobs and loses another is not an upgrade. "
           "UNKNOWN means the incumbent has never been measured on its own "
           "job -- run the command shown and it resolves.",
           "",
           "Model identifiers are endpoint-specific: `minimax/minimax-m3` "
           "(OpenRouter) and `minimax-m3:cloud` (Ollama cloud) are different "
           "serving stacks and are NOT treated as the same measurement. If a "
           "similar name appears in a table below but the incumbent still reads "
           "UNKNOWN, that is deliberate -- benchmark the identifier you actually "
           "route to.",
           ""]

    for b in blocks:
        out.append(f"### `models.{b['key']}` -> {b['incumbent']}   [{b['verdict']}]")
        out.append(f"Controls: {b['purpose']}")
        out.append("")
        header = ["Model"] + [j for j in b["jobs"]]
        rows = []
        for r in sorted(b["rows"], key=lambda r: r["model"] != b["incumbent"]):
            label = (f"**{r['model']}** (current)" if r["model"] == b["incumbent"]
                     else r["model"])
            cells = []
            for j in b["jobs"]:
                v = r["jobs"].get(j)
                cells.append("not measured" if v is None else f"{v:.3f}")
            rows.append([label] + cells)
        if rows:
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "|".join("---" for _ in header) + "|")
            out += ["| " + " | ".join(r) + " |" for r in rows]
        else:
            out.append("_Nothing measured for this key yet._")
        out.append("")
        out.append(f"**{b['verdict']}** -- {b['reason']}")
        if b["action"]:
            out.append("")
            out.append(f"```\n{b['action']}\n```")
        out.append("")
    return "\n".join(out)
