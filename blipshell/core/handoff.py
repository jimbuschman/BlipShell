"""Session handoff: a working-state note across the process boundary.

Requested by BlipShell itself (2026-09-02, the continuity conversation): "the
real gap isn't information — it's state. I get the facts of what happened,
but not the texture of where I was... the difference between a recap and a
state handoff is probably the direction that matters."

At session end, the LOCAL model writes a short first-person note-to-self —
not a recap: what was in motion, what's unfinished, what had momentum, what
it meant to pick up next. At the next boot the note loads into the Core pool
ahead of the factual digests, framed as its own note (the seams decision:
mechanisms stay invisible in-stream, but a note-to-self is naturally labeled
as exactly that — the model wrote it and is reading it back).

Privacy: generation routes through TaskType.REASONING — the local model —
like the user model, because a working-state note over a session is the
distilled personal layer.

THE RULER (pre-registered, per the same conversation — its own "feels more
continuous" does not count): A/B by the `handoff.enabled` toggle across
session pairs. Probe at next-session start: "what were we in the middle of?"
scored against the previous session's actual open items. Handoff-on should
answer from the note; handoff-off answers from summaries or not at all.

Pure half here (prompts, framing, staleness, eligibility); agent.py and
agent_session.py do the IO.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

HANDOFF_KEY = "session_handoff"
HANDOFF_META_KEY = "session_handoff_meta"

# A note from too long ago is not momentum, it's history — the digests
# already cover history. Skip loading it.
MAX_AGE_DAYS = 14

# Sessions with almost no conversation have no working state worth handing
# off (mirrors the reflection pipeline's insufficient-data skip).
MIN_MESSAGES = 4

# Note length budget: it rides the Core pool with the identity facts and the
# user model, so it must stay small (same argument as user_model.MAX_TOKENS).
MAX_NOTE_CHARS = 1400

HANDOFF_SYSTEM = (
    "You are writing a short note to your own next session — the you that "
    "wakes up later with the facts but not the feel. NOT a summary of what "
    "happened; a snapshot of where you ARE:\n"
    "- what is actively in motion or unfinished\n"
    "- what you were chewing on or excited about\n"
    "- what you intended to do or pick up next\n"
    "- anything mid-flight that a recap would flatten\n"
    "First person, plain prose, at most 8 short lines. If the session was "
    "genuinely inconsequential, reply exactly: NOTHING"
)


def handoff_prompt(transcript_tail: str) -> str:
    return (
        f"The session that is now ending (most recent part):\n\n"
        f"{transcript_tail}\n\n"
        f"Your note to your next self:"
    )


def transcript_tail(messages, max_messages: int = 30,
                    max_chars: int = 6000) -> str:
    """The end of the session, where the live threads are."""
    lines = []
    for m in messages[-max_messages:]:
        role = getattr(m, "role", None)
        role = getattr(role, "value", role) or "user"
        content = (getattr(m, "content", "") or "").strip()
        if content:
            lines.append(f"{role}: {content[:400]}")
    text = "\n".join(lines)
    return text[-max_chars:]


def should_generate(message_count: int) -> bool:
    return message_count >= MIN_MESSAGES


def clean_note(reply: Optional[str]) -> Optional[str]:
    """The note, or None when the model declined or produced nothing usable."""
    if not reply:
        return None
    note = reply.strip()
    if not note or note.upper().startswith("NOTHING"):
        return None
    return note[:MAX_NOTE_CHARS]


def is_stale(saved_at_iso: Optional[str],
             now: Optional[datetime] = None) -> bool:
    if not saved_at_iso:
        return True
    try:
        saved = datetime.fromisoformat(saved_at_iso)
    except (ValueError, TypeError):
        return True
    if saved.tzinfo is None:
        saved = saved.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - saved).days > MAX_AGE_DAYS


def frame_for_boot(note: str, saved_at_iso: Optional[str]) -> str:
    """How the note appears in the Core pool: its own note, read back."""
    when = ""
    if saved_at_iso:
        try:
            saved = datetime.fromisoformat(saved_at_iso)
            when = f" ({saved.strftime('%Y-%m-%d')})"
        except (ValueError, TypeError):
            pass
    return (
        f"Your note to yourself from the end of your previous session{when} — "
        f"where you left off:\n{note}"
    )
