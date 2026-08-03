"""Per-config-key assignment advice.

The report's rows are benchmark jobs; the thing you edit is a key in
config.yaml, and they are not one-to-one. `models.session_review` drives session
review AND lesson extraction, so kimi-k2.7-code leads one (0.925) and comes last
in the other (0.395) with nothing in the per-job table connecting them.

The rules these tests pin exist because each was violated by a real draft:
- a key must list EVERY job it determines (an under-specified `tool_calling`
  recommended lfm2.5 -- 0.933 tool calling, 0.450 reasoning -- for interactive chat)
- win-one-lose-another is KEEP, never CONSIDER
- per-endpoint overrides decide what actually runs, not the `models:` block
- output stays ASCII (cp1252 console)
"""

import pytest

from blipshell.benchmark.advice import (
    JOB_OWNERS,
    JOB_SUITE,
    MEANINGFUL_DELTA,
    build_advice,
    current_assignments,
    render_advice,
)
from blipshell.benchmark.report import build_report


def _rows(model, scores):
    """scores: {job: value} -> harness-shaped rows."""
    return [{"suite": "s", "task_type": job, "metric": "quality",
             "value": v, "unit": "ratio", "raw": None, "model": model}
            for job, v in scores.items()]


def _report(model_scores):
    return build_report({m: _rows(m, s) for m, s in model_scores.items()})


class _Ep:
    def __init__(self, name, roles, models, priority=1, enabled=True):
        self.name = name
        self.roles = roles
        self.models = models
        self.priority = priority
        self.enabled = enabled
        self.url = "http://x"
        self.context_tokens = 4096


class _Models:
    def __init__(self, **kw):
        for k in JOB_OWNERS:
            setattr(self, k, kw.get(k, f"default-{k}"))


class _Cfg:
    def __init__(self, models=None, endpoints=None):
        self.models = models or _Models()
        self.endpoints = endpoints or []


def _block(blocks, key):
    return next(b for b in blocks if b["key"] == key)


class TestMappingIntegrity:
    def test_every_key_exists_on_the_real_models_config(self):
        """A key that isn't a real config field is advice about nothing."""
        from blipshell.models.config import ModelsConfig
        real = ModelsConfig()
        for key in JOB_OWNERS:
            assert hasattr(real, key), f"models.{key} does not exist"

    def test_every_owned_job_has_a_suite_so_the_fix_command_is_nameable(self):
        for key, (jobs, _) in JOB_OWNERS.items():
            for j in jobs:
                assert j in JOB_SUITE, f"{key} owns '{j}' with no suite mapping"

    def test_every_owned_job_is_a_real_report_category(self):
        from blipshell.benchmark.report import CATEGORIES
        known = {c[0] for c in CATEGORIES}
        for key, (jobs, _) in JOB_OWNERS.items():
            for j in jobs:
                assert j in known, f"{key} owns '{j}' which the report can't render"

    def test_session_review_owns_lessons(self):
        """processor.py:275 routes lesson extraction through SESSION_REVIEW.
        Missing this is what made the kimi decision require reading source."""
        assert "lessons" in JOB_OWNERS["session_review"][0]

    def test_tool_calling_owns_more_than_the_tool_calling_job(self):
        """It serves the whole interactive path. With only its namesake job
        attached, a model strong at tool calls and weak at reasoning reads as an
        upgrade for chat."""
        jobs = JOB_OWNERS["tool_calling"][0]
        assert "tool_calling" in jobs
        assert "reasoning" in jobs


class TestCurrentAssignments:
    def test_reads_the_global_models_block(self):
        cfg = _Cfg(_Models(reasoning="qwen3:14b"))
        assert current_assignments(cfg)["reasoning"] == "qwen3:14b"

    def test_per_endpoint_override_wins(self):
        """What actually runs is the winning endpoint's model, not models.x --
        e.g. ranking_importance really resolves to Groq's llama-3.3-70b."""
        cfg = _Cfg(
            _Models(ranking_importance="qwen3:14b"),
            [_Ep("groq", ["ranking_importance"],
                 {"ranking_importance": "llama-3.3-70b-versatile"}, priority=2)],
        )
        assert current_assignments(cfg)["ranking_importance"] == "llama-3.3-70b-versatile"

    def test_highest_priority_endpoint_wins(self):
        cfg = _Cfg(
            _Models(tool_calling="global"),
            [_Ep("lo", ["tool_calling"], {"tool_calling": "low"}, priority=1),
             _Ep("hi", ["tool_calling"], {"tool_calling": "high"}, priority=5)],
        )
        assert current_assignments(cfg)["tool_calling"] == "high"

    def test_disabled_endpoint_is_ignored(self):
        cfg = _Cfg(
            _Models(summarization="glm4"),
            [_Ep("off", ["summarization"], {"summarization": "gemini"},
                 priority=9, enabled=False)],
        )
        assert current_assignments(cfg)["summarization"] == "glm4"


class TestVerdicts:
    def _cfg(self, **kw):
        return _Cfg(_Models(**kw))

    def test_unmeasured_incumbent_job_is_unknown_with_a_command(self):
        rep = _report({"other": {"session_review": 0.9, "lessons": 0.5}})
        b = _block(build_advice(rep, self._cfg(session_review="mine")), "session_review")
        assert b["verdict"] == "UNKNOWN"
        assert "session_review" in b["action"] or "pipeline" in b["action"]
        assert "mine" in b["action"]

    def test_win_one_lose_another_is_keep_not_consider(self):
        """The kimi/lessons trap: leads session_review, last on lessons, and the
        one key controls both."""
        rep = _report({
            "mine": {"session_review": 0.844, "lessons": 0.600},
            "kimi": {"session_review": 0.925, "lessons": 0.395},
        })
        b = _block(build_advice(rep, self._cfg(session_review="mine")), "session_review")
        assert b["verdict"] == "KEEP"
        assert "lessons" in b["reason"]

    def test_clear_win_on_all_owned_jobs_is_consider(self):
        rep = _report({
            "mine": {"session_review": 0.600, "lessons": 0.500},
            "better": {"session_review": 0.900, "lessons": 0.800},
        })
        b = _block(build_advice(rep, self._cfg(session_review="mine")), "session_review")
        assert b["verdict"] == "CONSIDER"
        assert "better" in b["reason"]

    def test_noise_sized_gain_is_keep(self):
        """kimi's session_review moved 0.950 -> 0.925 across a scorer change
        alone, so a margin that small is not evidence."""
        rep = _report({
            "mine": {"session_review": 0.900, "lessons": 0.700},
            "other": {"session_review": 0.900 + MEANINGFUL_DELTA / 2,
                      "lessons": 0.700},
        })
        b = _block(build_advice(rep, self._cfg(session_review="mine")), "session_review")
        assert b["verdict"] == "KEEP"

    def test_strong_at_one_job_weak_at_another_never_wins_interactive_chat(self):
        """The lfm2.5 regression, pinned: 0.933 tool calling but 0.450
        reasoning must not read as an upgrade for models.tool_calling."""
        rep = _report({
            "minimax-m3:cloud": {"tool_calling": 0.733, "reasoning": 0.912,
                                 "code_gen": 0.800},
            "lfm2.5": {"tool_calling": 0.933, "reasoning": 0.450,
                       "code_gen": 0.300},
        })
        b = _block(build_advice(rep, self._cfg(tool_calling="minimax-m3:cloud")),
                   "tool_calling")
        assert b["verdict"] == "KEEP"
        assert "lfm2.5" in b["reason"]

    def test_no_candidates_at_all_is_keep(self):
        rep = _report({"mine": {"session_review": 0.8, "lessons": 0.7}})
        b = _block(build_advice(rep, self._cfg(session_review="mine")), "session_review")
        assert b["verdict"] == "KEEP"
        assert b["action"] is None


class TestRendering:
    def _advice(self):
        rep = _report({
            "mine": {"session_review": 0.844, "lessons": 0.600},
            "kimi": {"session_review": 0.925, "lessons": 0.395},
        })
        return build_advice(rep, _Cfg(_Models(session_review="mine")))

    def test_marks_the_incumbent(self):
        md = render_advice(self._advice())
        assert "(current)" in md

    def test_lists_all_owned_jobs_as_columns(self):
        md = render_advice(self._advice())
        assert "session_review" in md and "lessons" in md

    def test_unknowns_are_ordered_first(self):
        """They're the actionable ones; a wall of KEEP above them buries the work."""
        md = render_advice(self._advice())
        first = md.index("[UNKNOWN]") if "[UNKNOWN]" in md else 10**9
        keep = md.index("[KEEP]") if "[KEEP]" in md else 10**9
        assert first < keep

    def test_output_is_ascii_only(self):
        md = render_advice(self._advice())
        assert all(ord(c) < 128 for c in md), \
            sorted({c for c in md if ord(c) > 127})

    def test_empty_input_renders_nothing(self):
        assert render_advice([]) == ""

    def test_never_reports_a_missing_score_as_zero(self):
        rep = _report({"mine": {"session_review": 0.8}})   # no lessons
        md = render_advice(build_advice(rep, _Cfg(_Models(session_review="mine"))))
        assert "not measured" in md
        assert "0.000" not in md
