"""Lesson revoting: nightly evidence review for the lessons pool (pure half).

The lessons pool is standing instruction — the top 30 by importance ride in
every prompt (agent_session._load_lessons) — but until now a lesson's
importance was scored once, at extraction, and never revisited: a lesson
contradicted by months of later evidence kept its seat. The 2026-09 audit
found the store accumulating stale and false guidance with no lifecycle.

ExpeL-style fix (docs/FIELD_SURVEY_2026_09.md 3.10, measured +11-19 pts on
agent benchmarks): each night, pair fresh session reflections with the
lessons most similar to them and ask the LOCAL model whether the evidence
CONFIRMS or CONTRADICTS the lesson. Confirmed lessons drift up; contradicted
ones sink until they fall out of the top 30 — demotion, never deletion, so a
lesson can recover if later evidence supports it.

This module is the pure, unit-tested half (prompts, verdict parsing, the
importance arithmetic); the nightly job in core/nightly.py does the IO.
Routed through TaskType.REASONING — the local model, deliberately: lessons
and reflections are the distilled personal layer.
"""

from __future__ import annotations

CONFIRMS = "CONFIRMS"
CONTRADICTS = "CONTRADICTS"
NEUTRAL = "NEUTRAL"

# Importance floor/ceiling. The floor keeps a contradicted lesson recoverable
# (and visible in exports) instead of deleting it; the pool's top-30 cut is
# what actually removes it from prompts.
MIN_IMPORTANCE = 0.1
MAX_IMPORTANCE = 1.0

JUDGE_SYSTEM = (
    "You review one standing LESSON about how to assist a user against one "
    "piece of new EVIDENCE (a reflection on a recent session).\n"
    f"Reply with exactly one word:\n"
    f"- {CONFIRMS}: the evidence shows the lesson's guidance working or its "
    "claim holding.\n"
    f"- {CONTRADICTS}: the evidence shows the lesson's guidance failing, or "
    "its claim about the user being wrong.\n"
    f"- {NEUTRAL}: the evidence is about something else, or is ambiguous.\n"
    f"When unsure, reply {NEUTRAL} — an unearned vote in either direction is "
    "worse than no vote."
)


def revote_prompt(lesson_content: str, evidence: str) -> str:
    return (
        f"LESSON:\n{lesson_content}\n\n"
        f"EVIDENCE (session reflection):\n{evidence}\n\n"
        f"One word — {CONFIRMS}, {CONTRADICTS}, or {NEUTRAL}:"
    )


def parse_verdict(reply: str | None) -> str | None:
    """Strict parse; anything unclear counts as no vote (None).

    Deliberately NOT mapped to NEUTRAL: a garbled reply is a judge failure,
    and the stats should show it rather than launder it into 'considered and
    neutral'.
    """
    if not reply:
        return None
    word = reply.strip().upper().split()[0].strip(".,:;!")
    if word in (CONFIRMS, CONTRADICTS, NEUTRAL):
        return word
    return None


def adjusted_importance(current: float, verdict: str,
                        up: float, down: float) -> float:
    """New importance after one vote, clamped to [MIN, MAX].

    Down is deliberately larger than up (defaults 0.15 vs 0.05): a lesson
    needs sustained confirmation to climb, but real counter-evidence should
    move it quickly — the cost asymmetry of stale standing instruction.
    """
    if verdict == CONFIRMS:
        current = current + up
    elif verdict == CONTRADICTS:
        current = current - down
    return max(MIN_IMPORTANCE, min(MAX_IMPORTANCE, round(current, 3)))
