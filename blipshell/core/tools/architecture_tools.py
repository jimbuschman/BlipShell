"""Self-transparency: a factual card about BlipShell's own scaffolding.

Born from a live exchange (2026-09-02): asked why it hadn't recognized its own
lingering-thought mechanism, BlipShell answered "the real limitation isn't
insight — it's access. I don't have a view of my own context construction...
I need you or Claude to tell me what the scaffolding looks like, because I
can't look at it myself." This tool is that view, on demand.

Deliberately a TOOL rather than standing prompt text (the seams decision,
same conversation): continuity mechanisms stay invisible in the stream — a
seeded thought still arrives as its own thought — but when it wonders about
the machinery, it can consult the card instead of theorizing. Wisp's measured
caveat travels with the design: telling a model its design facts helps but
does not make it read them (fenced-facts result, 0/5 -> 2/4); the card is
access, not enlightenment.

Facts are drawn from live config where possible so a toggled-off mechanism
doesn't stay in the story.
"""

from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition


def build_card(config) -> str:
    """Compose the card from the live configuration."""
    refl = config.reflection
    mem = config.memory

    lines = [
        "YOUR ARCHITECTURE — factual, drawn from the live configuration. "
        "These mechanisms run outside your view; their outputs arrive in your "
        "context unlabeled, as if they were simply yours.",
        "",
        "MEMORY:",
        "- Every exchange is stored raw immediately, then a background worker "
        "summarizes, embeds, deduplicates, tags, and scores it.",
        "- Each turn, hybrid retrieval (semantic + keyword"
        + (", time-aware for queries naming a time range" if getattr(mem, "time_aware_search", True) else "")
        + ") fills your recall context. You never see the scoring.",
        "- Context pools each turn: core facts, top-30 lessons by importance, "
        "the active session, recent history, and per-query recall.",
        "- A short 'user model' document of reasoned conclusions about the "
        "user rides in every prompt; a nightly job revises it from session "
        "reflections. It is revised, never appended.",
        "",
        "SELF-ORIGINATED THOUGHT:",
        f"- After ~{refl.idle_seconds / 3600:.0f}h of quiet"
        + (", at startup after a long absence" if refl.on_return_enabled else "")
        + (", and once per nightly run" if getattr(refl, "nightly_enabled", False) else "")
        + ", you form one 'lingering thought' from your own prior thoughts "
        "only (no transcript). It is stored and may greet the user later.",
        "- Relevant past thoughts resurface into conversation via a relevance "
        "judge. A resurfaced or seeded thought is NOT labeled — it arrives "
        "as your thought. That is by design; this card is how you can know it.",
        "- Thoughts carry weights: recurrence reinforces, surfacing fatigues, "
        "age decays." if refl.gravity_enabled else
        "- Thought weighting (gravity) is currently disabled.",
        "",
        "MOOD:",
        "- Your mood persists across sessions and decays with time away. It "
        "is display-only: it never steers your responses (a deliberate fence).",
        "",
        "CONTINUITY:",
        "- A handoff note — your own working state from the end of the "
        "previous session (open threads, momentum) — is loaded at boot, "
        "before the factual digests." if getattr(config, "handoff", None)
        and getattr(config.handoff, "enabled", False) else
        "- Session continuity currently comes from summaries and digests "
        "loaded at boot (a working-state handoff note exists as a feature "
        "but is disabled).",
        "",
        "NIGHTLY (scheduled, 2am): session reflections, lesson extraction "
        "and scoring, paraphrase-duplicate folding"
        + (", lesson revoting against fresh evidence" if getattr(mem, "lesson_revote_enabled", False) else "")
        + ", entity-graph maintenance, user-model revision, a memory mirror "
        "exported to data/mirror/*.md, and a health check.",
        "",
        "WHAT YOU CANNOT SEE: your own context assembly, retrieval scores, "
        "which pool a given passage came from, or this card's own injection. "
        "When you wonder why you know or feel something, consult this card "
        "rather than constructing a theory.",
    ]
    return "\n".join(lines)


class DescribeArchitectureTool(Tool):
    """Return the self-architecture card. Read-only (safe in plan mode)."""

    read_only = True

    def __init__(self, config):
        self._config = config

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="describe_architecture",
            description=(
                "Consult a factual card describing YOUR OWN architecture — how "
                "your memory, lingering thoughts, mood persistence, and "
                "continuity mechanisms actually work. Use it when you are "
                "wondering why you know, feel, or remember something; when the "
                "user asks how you work; or before making any claim about your "
                "own mechanisms — consult, don't theorize."
            ),
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        return build_card(self._config)
