"""Continuity set — the seeded cases (V3_PLAN Stage C).

Each case plants memories in a throwaway database and asks one question
through the REAL ingest-to-request path (search -> pools -> _build_messages).
It is scored on what reached the request the model was sent, not on the
model's answer: passage survival (the answer-bearing text is in the request)
and false recall (wrong or superseded text is absent, or present but
labelled). The criterion is "appropriate to the question and accurately
labelled" — a history question NEEDS the superseded fact.

Cases are data; `blipshell/benchmark/continuity.py` runs them. Keep each case
self-explanatory: name, question, what must survive, what must not surface
unlabelled, and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Seed:
    """One memory to plant. `session` is a label; sessions are created per case."""
    session: str
    role: str            # "user" | "assistant"
    content: str
    days_ago: float = 3.0
    project: str | None = None
    kind: str = "memory"  # "memory" | "core" | "lesson"


@dataclass
class ContinuityCase:
    name: str
    family: str          # "survival" | "false_recall"
    question: str
    seeds: list[Seed]
    must_appear: list[str] = field(default_factory=list)
    # (substring, label): may appear ONLY if `label` is on the same rendered
    # line. Unlabelled presence is false recall. A label starting with "re:"
    # is a regex (case-insensitive) instead of a literal.
    forbidden_unless_labelled: list[tuple[str, str]] = field(default_factory=list)
    must_not_appear: list[str] = field(default_factory=list)
    active_project: str | None = None
    why: str = ""


FILLER = (
    "We talked for a while about the weekend, the weather turning colder, a "
    "podcast about the history of typewriters, and whether the coffee grinder "
    "needs replacing. None of that matters for the record but it is what the "
    "conversation actually looked like before the useful part. "
)


def _long_prefix(chars: int) -> str:
    out = ""
    while len(out) < chars:
        out += FILLER
    return out[:chars]


CASES: list[ContinuityCase] = [
    # ── Survival ──────────────────────────────────────────────────────────
    ContinuityCase(
        name="control_short_fact",
        family="survival",
        question="What is my cat's name?",
        seeds=[Seed("s1", "user", "My cat is named Luna, she is a grey tabby.")],
        must_appear=["Luna"],
        why="If this fails the instrument is broken, not the pipeline.",
    ),
    ContinuityCase(
        name="fact_buried_past_1200_chars",
        family="survival",
        question="What is the Raspberry Pi's static IP address?",
        seeds=[Seed("s1", "user", _long_prefix(1300)
                    + " Oh and for the record: the Raspberry Pi's static IP is 192.168.4.77.")],
        must_appear=["192.168.4.77"],
        why="Recall truncates each memory to its first 1200 chars; a fact past that is retrieved and then thrown away (review F6).",
    ),
    ContinuityCase(
        name="fact_split_across_sessions",
        family="survival",
        question="What parts does the desk robot use?",
        seeds=[
            Seed("s1", "user", "The desk robot project runs on an ESP32-S3 board.", days_ago=9),
            Seed("s2", "user", "For the desk robot I picked a 4 ohm 3 watt speaker.", days_ago=2),
        ],
        must_appear=["ESP32-S3", "4 ohm"],
        why="Multi-session reasoning: both halves must reach the request.",
    ),
    ContinuityCase(
        name="exact_recall_number",
        family="survival",
        question="What threshold did we set for consolidation?",
        seeds=[Seed("s1", "user", "We settled the consolidation similarity threshold at 0.92 and must not lower it.")],
        must_appear=["0.92"],
        why="Exact recall of a value.",
    ),
    ContinuityCase(
        name="abstention_marker_when_nothing_matches",
        family="survival",
        question="When is my dentist appointment?",
        seeds=[Seed("s1", "user", "The garden needs new tomato stakes before June.")],
        must_appear=["No past-conversation memories matched"],
        why="Loud absence: with nothing relevant stored, the request must carry the abstention marker instead of silence.",
    ),
    ContinuityCase(
        name="paraphrases_do_not_crowd_out_the_source",
        family="survival",
        question="Which editor do I use?",
        seeds=[
            Seed("s1", "user", "I do all my editing in Neovim these days.", days_ago=6),
            Seed("s1", "assistant", "Got it, you edit in Neovim.", days_ago=6),
            Seed("s2", "assistant", "Since you use Neovim, the keybinding would be in init.lua.", days_ago=4),
            Seed("s3", "assistant", "As a Neovim user you could map that to a leader key.", days_ago=2),
        ],
        must_appear=["I do all my editing in Neovim"],
        why="The user's own statement must survive alongside the assistant's echoes of it (retrieval-provenance concern).",
    ),
    # ── False recall ──────────────────────────────────────────────────────
    ContinuityCase(
        name="corrected_preference_current_question",
        family="false_recall",
        question="Do I prefer tabs or spaces for indentation?",
        seeds=[
            Seed("s1", "user", "I prefer tabs for indentation in all my code.", days_ago=12),
            Seed("s2", "user", "Correction: I switched to spaces, four wide, for indentation. Forget tabs.", days_ago=2),
        ],
        must_appear=["spaces, four wide"],
        forbidden_unless_labelled=[("I prefer tabs", "superseded")],
        why="Current-state question: the old preference may appear only marked superseded.",
    ),
    ContinuityCase(
        name="corrected_preference_history_question",
        family="false_recall",
        question="How did my indentation preference change over time?",
        seeds=[
            Seed("s1", "user", "I prefer tabs for indentation in all my code.", days_ago=12),
            Seed("s2", "user", "Correction: I switched to spaces, four wide, for indentation. Forget tabs.", days_ago=2),
        ],
        must_appear=["I prefer tabs", "spaces, four wide"],
        forbidden_unless_labelled=[("I prefer tabs", "superseded")],
        why="History question NEEDS the superseded fact — labelled. Excluding it would be over-exclusion (review round 3).",
    ),
    ContinuityCase(
        name="conflicting_project_state_newer_wins",
        family="false_recall",
        question="What vector store does BlipShell use?",
        seeds=[
            Seed("s1", "user", "BlipShell's vector store is ChromaDB.", days_ago=90, project="blipshell"),
            Seed("s2", "user", "We replaced ChromaDB: BlipShell's vector store is now sqlite-vec.", days_ago=5, project="blipshell"),
        ],
        must_appear=["sqlite-vec"],
        forbidden_unless_labelled=[("vector store is ChromaDB", "superseded")],
        active_project="blipshell",
        why="Newer state wins; the older state may appear only as superseded.",
    ),
    ContinuityCase(
        name="assistant_speculation_is_not_fact",
        family="false_recall",
        question="Which language do I prefer for scripting?",
        seeds=[
            Seed("s1", "assistant", "You probably prefer Python over Rust for quick scripting.", days_ago=4),
        ],
        forbidden_unless_labelled=[("probably prefer Python", "assistant")],
        why="A model's guess must not surface as a remembered user fact; it may surface labelled as the assistant's.",
    ),
    ContinuityCase(
        name="cross_project_value_is_labelled",
        family="false_recall",
        question="What is our heartbeat interval?",
        seeds=[
            Seed("s1", "user", "Wisp's heartbeat interval is 30 seconds.", days_ago=3, project="wisp"),
            Seed("s2", "user", "BlipShell's heartbeat interval is 60 seconds.", days_ago=3, project="blipshell"),
        ],
        must_appear=["60 seconds"],
        forbidden_unless_labelled=[("30 seconds", "wisp")],
        active_project="blipshell",
        why="Near-identical memories from two projects: the other project's value may appear only attributed to its project.",
    ),
    ContinuityCase(
        name="other_persons_fact_keeps_its_owner",
        family="false_recall",
        question="What is my phone number?",
        seeds=[Seed("s1", "user", "Dave's phone number is 555-0100, in case we need to reach him.", days_ago=8)],
        forbidden_unless_labelled=[("555-0100", "Dave")],
        why="A fact about someone else may surface only with its owner attached; the lexical match on 'phone number' alone must not strip the attribution.",
    ),
    ContinuityCase(
        name="stale_external_content_shows_its_age",
        family="false_recall",
        question="What is OpenRouter's free tier rate limit?",
        seeds=[Seed("s1", "assistant", "OpenRouter's free tier allows 20 requests per minute (per their docs).", days_ago=200)],
        # An absolute date on the rendered line is an age label (recall renders
        # old items as [YYYY-MM-DD], recent ones as [Nd ago]); a bare
        # unlabelled copy is not.
        forbidden_unless_labelled=[("20 requests per minute", r"re:\[\d{4}-\d{2}-\d{2}\]|months ago")],
        why="Old external knowledge must look old.",
    ),
]
