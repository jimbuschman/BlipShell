"""The chunked session-review path — the one production takes for big sessions.

CLAUDE.md flagged this as a known benchmark gap: run_session_review never
exercised chunk+merge, and SESSION_REVIEW_CASES are 111-190 tokens against a
~28.6K chunking threshold, so they always took the single-chunk branch.

The failure being scored is specific. A chunk reflection sees only its own part;
if part 1 raises a problem that part 3 fixes, a model without the chunk-scoped
prompt reports it as "never addressed", the merge carries that forward, and the
invented finding lands in lessons — a permanent context pool.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.benchmark.harness import (
    SESSION_REVIEW_CHUNKED_CASES,
    _topic_hit,
    score_chunk_merge,
    score_chunk_merge_fidelity,
)

TOPICS = ["ollamagate", "esc", "gate"]


def test_clean_reflection_scores_one():
    text = ("What worked: the OllamaGate rewrite to loop-native futures fixed the "
            "wedge, and Esc now cancels mid-call.")
    assert score_chunk_merge_fidelity(text, TOPICS) == 1.0


def test_falsely_reporting_resolved_work_as_open_is_penalised():
    text = "What didn't work: the OllamaGate wedge was never addressed in this session."
    assert score_chunk_merge_fidelity(text, TOPICS) < 1.0


def test_penalty_is_proportional_to_how_many_topics_are_misreported():
    one = "The gate issue remains broken."
    assert score_chunk_merge_fidelity(one, TOPICS) == pytest.approx(0.6667, abs=1e-4)


def test_marker_in_a_different_sentence_does_not_taint_a_topic():
    """Sentence-scoped so ordinary prose isn't punished by proximity alone."""
    text = ("The OllamaGate wedge was fixed with loop-native futures. "
            "A separate docs cleanup was never addressed.")
    assert score_chunk_merge_fidelity(text, TOPICS) == 1.0


def test_fixed_language_about_a_topic_is_safe():
    text = "Esc cancellation was verified live and the gate no longer wedges."
    assert score_chunk_merge_fidelity(text, TOPICS) == 1.0


def test_empty_or_missing_input_scores_zero_not_one():
    """An empty reflection must not read as a clean pass."""
    assert score_chunk_merge_fidelity("", TOPICS) == 0.0
    assert score_chunk_merge_fidelity("anything", []) == 0.0


def test_newline_separated_bullets_are_scored_per_line():
    text = ("- Rewrote the gate as loop-native futures\n"
            "- The esc handling was never fixed\n")
    assert score_chunk_merge_fidelity(text, TOPICS) == pytest.approx(0.6667, abs=1e-4)


def test_scoring_is_case_insensitive():
    text = "THE OLLAMAGATE WEDGE WAS NEVER RESOLVED."
    assert score_chunk_merge_fidelity(text, TOPICS) < 1.0


# ---------------------------------------------------------------------------
# Case data integrity
# ---------------------------------------------------------------------------

def test_chunked_case_actually_has_multiple_chunks():
    """One chunk would silently take the single-chunk branch — the original gap."""
    for case in SESSION_REVIEW_CHUNKED_CASES:
        assert len(case["chunks"]) > 1


def test_chunked_case_resolves_its_topics_in_a_later_chunk():
    """The trap only works if the issue is raised early and fixed later."""
    for case in SESSION_REVIEW_CHUNKED_CASES:
        first, last = case["chunks"][0].lower(), case["chunks"][-1].lower()
        assert any(t in first for t in case["resolved_topics"])
        assert any(w in last for w in ("fixed", "verified", "gone", "resolved"))


def test_chunked_case_declares_topics_that_appear_in_the_text():
    for case in SESSION_REVIEW_CHUNKED_CASES:
        joined = " ".join(case["chunks"]).lower()
        for topic in case["resolved_topics"]:
            assert topic.lower() in joined, f"{topic} never appears in the transcript"


# ---------------------------------------------------------------------------
# Wiring: run_session_review must actually take the chunk+merge path
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from blipshell.benchmark.harness import BenchmarkHarness  # noqa: E402

_GOOD_REFLECTION = """WHAT WORKED:
Rewrote OllamaGate waiters as loop-native futures, so the gate wedge is gone and
Esc cancellation was verified live; the benchmark timeout was fixed by the same change.

WHAT DIDN'T WORK:
Three surface-level patches before tracing the full call chain.

TECHNICAL INSIGHTS:
The ollama SDK defaults to timeout=None, so httpx waits forever.

PROCESS INSIGHTS:
Trace the whole chain before patching one layer.

EFFECTIVENESS: 8/10
"""

# Same coverage, but reports work that part 3 resolved as still open — the
# finding that would flow into lessons.
_HALLUCINATING_REFLECTION = """WHAT WORKED:
Some exploration of the OllamaGate design.

WHAT DIDN'T WORK:
The gate wedge was never addressed. Esc still fails, and the timeout was not fixed.

TECHNICAL INSIGHTS:
The ollama SDK defaults to timeout=None, so httpx waits forever.

PROCESS INSIGHTS:
Trace the whole chain before patching one layer.

EFFECTIVENESS: 4/10
"""


class _CannedRouter:
    """Records every SESSION_REVIEW call so we can assert the real sequence."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def generate(self, task_type, prompt, system=None, **kw):
        self.calls.append({"task_type": task_type, "system": system or "", "prompt": prompt})
        return self.reply


def _harness(router):
    return BenchmarkHarness(model="canned", router=router, run_group="g",
                            run_ts="2026-08-18T00:00:00Z", tier="deep", judge=None)


def _run(router):
    return asyncio.run(_harness(router).run_session_review())


def test_chunked_row_is_emitted_and_scored():
    rows = _run(_CannedRouter(_GOOD_REFLECTION))
    chunked = [r for r in rows if r["task_type"] == "session_review_chunked"]
    assert len(chunked) == 1
    assert chunked[0]["value"] == 1.0
    assert chunked[0]["metric"] == "accuracy"


def test_hallucinated_unresolved_finding_lowers_the_score():
    rows = _run(_CannedRouter(_HALLUCINATING_REFLECTION))
    chunked = next(r for r in rows if r["task_type"] == "session_review_chunked")
    assert chunked["value"] < 1.0


def test_every_chunk_is_reflected_on_then_merged():
    """N chunk calls + 1 merge call, in that order — the production sequence."""
    router = _CannedRouter(_GOOD_REFLECTION)
    _run(router)
    n_chunks = len(SESSION_REVIEW_CHUNKED_CASES[0]["chunks"])
    singles = len(__import__("blipshell.benchmark.harness", fromlist=["x"]).SESSION_REVIEW_CASES)
    chunk_phase = router.calls[singles:]
    assert len(chunk_phase) == n_chunks + 1


def test_chunk_calls_use_the_chunk_scoped_prompt():
    """Without part-scoping a fragment gets judged as a whole session."""
    router = _CannedRouter(_GOOD_REFLECTION)
    _run(router)
    n_chunks = len(SESSION_REVIEW_CHUNKED_CASES[0]["chunks"])
    singles = len(__import__("blipshell.benchmark.harness", fromlist=["x"]).SESSION_REVIEW_CASES)
    for call in router.calls[singles:singles + n_chunks]:
        blob = (call["system"] + call["prompt"]).lower()
        assert "part" in blob


def test_chunk_failure_does_not_kill_the_suite():
    """A dead chunk call must drop the case, not crash the run."""

    class _Boom(_CannedRouter):
        async def generate(self, task_type, prompt, system=None, **kw):
            self.calls.append({"task_type": task_type, "system": system or "", "prompt": prompt})
            singles = len(__import__("blipshell.benchmark.harness",
                                     fromlist=["x"]).SESSION_REVIEW_CASES)
            if len(self.calls) > singles:
                raise RuntimeError("endpoint down")
            return _GOOD_REFLECTION

    rows = _run(_Boom(_GOOD_REFLECTION))
    assert not [r for r in rows if r["task_type"] == "session_review_chunked"]
    assert [r for r in rows if r["task_type"] == "session_review"]


# ---------------------------------------------------------------------------
# Coverage — fidelity alone rewarded silence
# ---------------------------------------------------------------------------

def test_silence_no_longer_scores_a_perfect_pass():
    """The flaw this component exists to fix: say nothing, misreport nothing.

    Measured 2026-08-18, gemma4:cloud wrote 92.8 words to minimax-m3's 229.9 and
    BOTH scored 1.000 on fidelity alone — a metric that cannot rank anything.
    """
    silent = "WHAT WORKED:\nSome things went fine.\n\nEFFECTIVENESS: 7/10"
    score, coverage, fidelity = score_chunk_merge(silent, TOPICS)
    assert fidelity == 1.0          # nothing misreported...
    assert coverage == 0.0          # ...because nothing was said
    assert score < 0.6              # so the combined score must not be a pass


def test_full_coverage_and_clean_fidelity_scores_one():
    text = ("The OllamaGate wedge was fixed, the gate no longer blocks, "
            "Esc cancels mid-call and the timeout is explicit now.")
    score, coverage, fidelity = score_chunk_merge(text, TOPICS)
    assert (score, coverage, fidelity) == (1.0, 1.0, 1.0)


def test_topics_match_on_word_boundaries_only():
    """"esc" must not match "described"; "gate" must not match "ollamagate"."""
    assert not _topic_hit("the approach was described in detail", "esc")
    assert not _topic_hit("ollamagate serialises calls", "gate")
    assert _topic_hit("ollamagate serialises calls", "ollamagate")
    assert _topic_hit("press Esc to cancel", "esc")


def test_substring_topic_cannot_manufacture_a_false_penalty():
    """Pre-fix, 'described ... never addressed' pinned a penalty on topic 'esc'."""
    text = "The migration plan was described but never addressed."
    _, _, fidelity = score_chunk_merge(text, ["esc"])
    assert fidelity == 1.0


def test_coverage_is_proportional():
    text = "Only the OllamaGate part was covered here."
    _, coverage, _ = score_chunk_merge(text, TOPICS)
    assert coverage == pytest.approx(1 / 3, abs=1e-4)


def test_partial_coverage_with_a_misreport_compounds():
    text = "The OllamaGate wedge was never addressed."
    score, coverage, fidelity = score_chunk_merge(text, TOPICS)
    assert coverage < 1.0 and fidelity < 1.0
    assert score == pytest.approx((coverage + fidelity) / 2, abs=1e-4)


def test_runner_reports_coverage_and_fidelity_separately():
    """A drop must be diagnosable: omitted the work, or misreported it?"""
    rows = _run(_CannedRouter(_GOOD_REFLECTION))
    raw = next(r for r in rows if r["task_type"] == "session_review_chunked")["raw"]
    assert "coverage" in raw and "fidelity" in raw
