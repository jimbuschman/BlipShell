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


# Per message, before it has to be excerpted. The note is about what was in
# MOTION, so the transcript must carry enough of each turn to show a thread,
# not just its opening.
PER_MESSAGE_CHARS = 800

# Below this an excerpt says nothing; the message is dropped and counted
# instead of rendered as a stub.
MIN_EXCERPT_CHARS = 120

# Room held back for the omission notice so a dropped message never evicts a
# kept one. Comfortably above the longest notice max_messages can produce.
NOTICE_RESERVE = 48

# Of an excerpted message, the share given to its END. A long final turn puts
# the decision, the conclusion and the "next I want to..." last; a head-only
# truncation drops exactly the part the handoff exists to carry.
TAIL_SHARE = 0.6


def _excerpt(text: str, budget: int) -> str:
    """`text` in at most `budget` chars, keeping its END.

    Head AND tail, with the omission marked: the head says what the turn was
    about, the tail says where it landed. An unmarked excerpt is worse than a
    short one — the model would read a cut-off message as a complete thought.
    """
    if len(text) <= budget:
        return text

    def build(omitted: int):
        marker = f" [...{omitted} chars omitted...] "
        room = budget - len(marker)
        if room < 2:
            return None
        tail_len = max(1, int(room * TAIL_SHARE))
        head_len = room - tail_len
        excerpt = text[:head_len] + marker + text[len(text) - tail_len:]
        return excerpt, len(text) - head_len - tail_len

    # The marker states the count, and the count depends on the marker's own
    # length. A couple of passes settle it; if it will not settle, keep the
    # end and say so.
    guess = len(text) - budget
    for _ in range(3):
        built = build(guess)
        if built is None:
            break
        excerpt, actual = built
        if actual == guess:
            return excerpt
        guess = actual

    marker = "[...earlier text omitted...] "
    if budget > len(marker) + 1:
        return marker + text[-(budget - len(marker)):]
    return text[-budget:]


def transcript_tail(messages, max_messages: int = 30,
                    max_chars: int = 6000) -> str:
    """The end of the session, where the live threads are.

    Built BACKWARDS from the newest message, because the budget belongs to
    the most recent turns, and rendered forwards so the model reads it in
    order. Every line keeps its role label.

    It used to take each message's first 400 characters and then slice the
    joined text with `text[-max_chars:]`. Two losses: a long final turn was
    cut at character 400, taking the decision and the next action with it —
    the note exists to carry exactly that — and the closing slice could land
    mid-line, handing the model a fragment with no role on it.
    """
    # Room set aside for the omission notice, so a dropped message never
    # costs a kept one. Evicting content to make room for the notice was
    # worse than the problem: at max_chars=300 a 300-character excerpt of the
    # newest turn was thrown away to fit 36 characters of bookkeeping. The
    # longest possible notice is ~37 chars (max_messages bounds the count),
    # and at small budgets the reserve is skipped - nothing else fits there
    # either.
    reserve = NOTICE_RESERVE if max_chars >= 4 * NOTICE_RESERVE else 0
    working = max_chars - reserve

    lines: list[str] = []
    used = 0
    dropped = 0

    for m in reversed(list(messages[-max_messages:])):
        role = getattr(m, "role", None)
        role = getattr(role, "value", role) or "user"
        content = (getattr(m, "content", "") or "").strip()
        if not content:
            continue

        overhead = len(role) + 2 + (1 if lines else 0)   # "role: " and a newline
        available = working - used - overhead
        if available <= 0:
            dropped += 1
            continue

        if len(content) <= min(PER_MESSAGE_CHARS, available):
            # It fits WHOLE. MIN_EXCERPT_CHARS is a floor on how short an
            # excerpt may usefully be; applying it here dropped complete
            # short messages and replaced them with an omission notice
            # longer than the message itself.
            body = content
        elif available >= MIN_EXCERPT_CHARS:
            body = _excerpt(content, min(PER_MESSAGE_CHARS, available))
        else:
            dropped += 1
            continue

        line = f"{role}: {body}"
        used += len(line) + (1 if lines else 0)
        lines.append(line)

    lines.reverse()

    if dropped:
        notice = f"[...{dropped} earlier message(s) omitted...]"
        # The notice is part of the budget, not an addition to it — including
        # when it is the ONLY thing left. A budget too small even for the
        # notice yields nothing rather than an over-budget string.
        if used + len(notice) + (1 if lines else 0) <= max_chars:
            lines.insert(0, notice)

    return "\n".join(lines)


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
