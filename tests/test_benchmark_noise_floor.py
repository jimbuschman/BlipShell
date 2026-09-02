"""The verdict must not recommend a model on run-to-run noise.

MEANINGFUL_DELTA was a fixed 0.03. Measured on 2026-08-18, tool_calling
repeats on ONE model gave spread 0.13-0.20 (sd 0.042-0.089) — so a 0.03 gain
was routinely a coin flip the harness reported as a clean win. When repeats
supply a spread, a gain must clear it.

These drive the real rows -> build_report -> build_advice path, so they also
guard the plumbing that carries `spread` from merge_repeat_rows to the verdict.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.benchmark.advice import (
    JOB_OWNERS,
    MEANINGFUL_DELTA,
    _noise_floor,
    build_advice,
)
from blipshell.benchmark.report import _spread_map, build_report
from blipshell.benchmark.runner import merge_repeat_rows


def _rows(model, scores, spreads=None):
    out = []
    for job, v in scores.items():
        row = {"suite": "s", "task_type": job, "metric": "quality",
               "value": v, "unit": "ratio", "raw": None, "model": model}
        sp = (spreads or {}).get(job)
        if sp is not None:
            row["spread"] = sp
        out.append(row)
    return out


def _report(model_scores, model_spreads=None):
    return build_report({m: _rows(m, s, (model_spreads or {}).get(m))
                         for m, s in model_scores.items()})


class _Models:
    def __init__(self, **kw):
        for k in JOB_OWNERS:
            setattr(self, k, kw.get(k, f"default-{k}"))


class _Cfg:
    def __init__(self, models=None):
        self.models = models or _Models()
        self.endpoints = []


def _block(blocks, key):
    return next(b for b in blocks if b["key"] == key)


def _cfg(**kw):
    return _Cfg(_Models(**kw))


# ---------------------------------------------------------------------------
# _noise_floor
# ---------------------------------------------------------------------------

def test_noise_floor_defaults_to_constant_without_spread():
    assert _noise_floor("tool_calling", {}, "a", "b") == MEANINGFUL_DELTA


def test_noise_floor_rises_to_mean_measured_spread():
    spreads = {"tool_calling": {"a": 0.20, "b": 0.10}}
    assert _noise_floor("tool_calling", spreads, "a", "b") == pytest.approx(0.15)


def test_noise_floor_never_falls_below_the_constant():
    spreads = {"tool_calling": {"a": 0.001, "b": 0.001}}
    assert _noise_floor("tool_calling", spreads, "a", "b") == MEANINGFUL_DELTA


def test_noise_floor_uses_whichever_model_has_a_spread():
    spreads = {"tool_calling": {"a": 0.20}}
    assert _noise_floor("tool_calling", spreads, "a", "b") == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

# Every job models.session_review owns. Missing one makes the incumbent read as
# unmeasured, and every verdict becomes UNKNOWN regardless of the noise logic.
SR_JOBS = ("session_review", "lessons", "session_review_chunked")


def test_gain_inside_measured_noise_is_not_an_upgrade():
    """+0.04 on both jobs, but both models swing 0.20 across repeats."""
    rep = _report(
        {"mine": {j: 0.80 for j in SR_JOBS}, "rival": {j: 0.84 for j in SR_JOBS}},
        {"mine": {j: 0.20 for j in SR_JOBS}, "rival": {j: 0.20 for j in SR_JOBS}},
    )
    assert _block(build_advice(rep, _cfg(session_review="mine")),
                  "session_review")["verdict"] == "KEEP"


def test_same_gain_counts_when_repeats_are_tight():
    """Identical +0.04, but replication says it is stable."""
    rep = _report(
        {"mine": {j: 0.80 for j in SR_JOBS}, "rival": {j: 0.84 for j in SR_JOBS}},
        {"mine": {j: 0.01 for j in SR_JOBS}, "rival": {j: 0.01 for j in SR_JOBS}},
    )
    assert _block(build_advice(rep, _cfg(session_review="mine")),
                  "session_review")["verdict"] == "CONSIDER"


def test_large_gain_survives_a_noisy_measurement():
    rep = _report(
        {"mine": {j: 0.50 for j in SR_JOBS}, "rival": {j: 0.90 for j in SR_JOBS}},
        {"mine": {j: 0.20 for j in SR_JOBS}, "rival": {j: 0.20 for j in SR_JOBS}},
    )
    assert _block(build_advice(rep, _cfg(session_review="mine")),
                  "session_review")["verdict"] == "CONSIDER"


def test_noise_sized_loss_does_not_block_a_real_win():
    """A -0.04 'regression' inside a 0.20 band is not a regression."""
    rep = _report(
        {"mine": {"session_review": 0.50, "lessons": 0.84, "session_review_chunked": 0.50},
         "rival": {"session_review": 0.90, "lessons": 0.80, "session_review_chunked": 0.90}},
        {"mine": {"session_review": 0.02, "lessons": 0.20, "session_review_chunked": 0.02},
         "rival": {"session_review": 0.02, "lessons": 0.20, "session_review_chunked": 0.02}},
    )
    b = _block(build_advice(rep, _cfg(session_review="mine")), "session_review")
    assert b["verdict"] == "CONSIDER"
    assert not b.get("losses")


def test_real_loss_outside_the_noise_band_still_blocks():
    """The kimi/lessons trap must survive the noise-aware rewrite."""
    rep = _report(
        {"mine": {"session_review": 0.844, "lessons": 0.600, "session_review_chunked": 0.90},
         "kimi": {"session_review": 0.925, "lessons": 0.395, "session_review_chunked": 0.90}},
        {"mine": {"session_review": 0.02, "lessons": 0.02, "session_review_chunked": 0.02},
         "kimi": {"session_review": 0.02, "lessons": 0.02, "session_review_chunked": 0.02}},
    )
    b = _block(build_advice(rep, _cfg(session_review="mine")), "session_review")
    assert b["verdict"] == "KEEP"
    assert "lessons" in b["reason"]


def test_unreplicated_runs_still_use_the_constant():
    """No spread recorded -> behaves exactly as before this change."""
    rep = _report({"mine": {j: 0.80 for j in SR_JOBS},
                   "rival": {j: 0.90 for j in SR_JOBS}})
    assert _block(build_advice(rep, _cfg(session_review="mine")),
                  "session_review")["verdict"] == "CONSIDER"


# ---------------------------------------------------------------------------
# Plumbing: merge_repeat_rows -> _spread_map -> categories
# ---------------------------------------------------------------------------

def test_spread_map_reads_merged_rows():
    rows = [{"suite": "reasoning", "task_type": "tool_calling",
             "metric": "tool_pass_rate", "value": 0.86, "spread": 0.13}]
    assert _spread_map(rows) == {"tool_calling": 0.13}


def test_spread_map_ignores_unreplicated_rows():
    rows = [{"suite": "reasoning", "task_type": "tool_calling",
             "metric": "tool_pass_rate", "value": 0.86}]
    assert _spread_map(rows) == {}


def test_merge_repeat_rows_produces_a_spread_the_report_can_read():
    """End-to-end: repeats -> merged row -> spread map."""
    sets = [[{"suite": "reasoning", "task_type": "tool_calling",
              "metric": "tool_pass_rate", "value": v}] for v in (0.73, 0.93, 0.80)]
    merged = merge_repeat_rows(sets)
    assert _spread_map(merged)["tool_calling"] == pytest.approx(0.20)


def test_categories_carry_spread_for_the_verdict():
    rep = _report({"mine": {"lessons": 0.8}}, {"mine": {"lessons": 0.13}})
    cat = next(c for c in rep["categories"] if c["key"] == "lessons")
    assert cat["spread"]["mine"] == pytest.approx(0.13)
