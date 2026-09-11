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

# The note is refreshed DURING the session every N assistant turns, not only
# at close (2026-09-10, the continuity investigation): the sessions where
# BlipShell described the disconnect (1919, 1920 on 2026-08-11) ended with
# message_count 0 - no summary, and a close-only note would have been lost
# too. A live note is where the momentum actually is; the close pass refines it.
REFRESH_EVERY_TURNS_DEFAULT = 6
# How many of the last exchanges are carried at the next boot, word for word
# where they fit and excerpted (opening + ending) where they do not. The
# thread the conversation stopped on is the state; summaries and
# importance-ranked lines are not (session 1926: the previous substantive
# session's top-by-importance lines and a third-person summary were loaded,
# the half-formed idea it stopped on was not).
STOP_BLOCK_PAIRS_DEFAULT = 2
# Per-message char budget the packers aim for; the real allocation is
# water-filled across the messages actually present (see `pack_tail`), so a
# short turn hands its surplus to a long one instead of wasting it.
PER_MESSAGE_CHARS = 400
# Below this a line carries no usable state, so the OLDEST lines are dropped
# instead of every line being shaved to a stub.
MIN_MESSAGE_CHARS = 160

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


def _allocate(lengths: list[int], total: int) -> list[int]:
    """Water-fill `total` across `lengths`: every message gets an equal share,
    and whatever a short message does not need is redistributed to the ones
    still over budget. Deterministic; any remainder goes to the LAST (most
    recent) message."""
    alloc = [0] * len(lengths)
    remaining, live = total, list(range(len(lengths)))
    while live and remaining > 0:
        share = remaining // len(live)
        if share <= 0:
            break
        still = []
        for i in live:
            take = min(lengths[i] - alloc[i], share)
            alloc[i] += take
            remaining -= take
            if alloc[i] < lengths[i]:
                still.append(i)
        if not still:
            break
        live = still
    if remaining > 0 and alloc:
        alloc[-1] += remaining
    return alloc


def excerpt_label(ident) -> str:
    """The marker a rendered excerpt carries: it says the line is a window,
    and names the memory whose full text `search_memories(memory_ids=...)`
    returns."""
    return f" [excerpt of memory {ident}]" if ident else " [excerpt]"


def pack_tail(rows, total_chars: int, flatten: bool = False,
              label: Optional[object] = None):
    """Pack the most recent exchanges into `total_chars`, from the end.

    `rows` are (role, content, ident) oldest-first; returns
    (role, text, ident, excerpted). The role-prefix overhead of the rendered
    lines is taken out of the budget first, so the assembled block stays
    inside it without a blind final cut. A turn over its share is excerpted
    head+END (`memory.excerpt.head_tail_excerpt`), never cut at its opening:
    the unfinished next step at the END of a long turn is the state a handoff
    exists to carry, and the prefix cut dropped exactly that (review
    2026-09-11, finding 1).

    `label` is the caller's `ident -> str` marker for an excerpted line; its
    length is charged to that line's own share, so the rendered block stays
    inside `total_chars` whether or not anything had to be cut.
    """
    from blipshell.memory.excerpt import head_tail_excerpt

    prepared = []
    for role, content, ident in rows:
        text = (content or "").strip()
        if flatten:
            text = " ".join(text.split())
        if text:
            prepared.append((str(role), text, ident))
    if not prepared:
        return []
    while len(prepared) > 1:
        overhead = sum(len(r[0]) + 2 for r in prepared) + (len(prepared) - 1)
        if (total_chars - overhead) // len(prepared) >= MIN_MESSAGE_CHARS:
            break
        prepared = prepared[1:]  # drop the OLDEST; the end carries the thread
    overhead = sum(len(r[0]) + 2 for r in prepared) + (len(prepared) - 1)
    budgets = _allocate([len(r[1]) for r in prepared],
                        max(0, total_chars - overhead))
    out = []
    for (role, text, ident), budget in zip(prepared, budgets):
        if len(text) <= budget:
            out.append((role, text, ident, False))
        else:
            marked = budget - (len(label(ident)) if label else 0)
            out.append((role, head_tail_excerpt(text, max(0, marked)), ident, True))
    return out


def transcript_tail(messages, max_messages: int = 30,
                    max_chars: int = 6000) -> str:
    """The end of the session, where the live threads are."""
    rows = []
    for m in messages[-max_messages:]:
        role = getattr(m, "role", None)
        role = getattr(role, "value", role) or "user"
        rows.append((role, getattr(m, "content", "") or "", None))
    return "\n".join(f"{role}: {text}"
                     for role, text, _ident, _excerpted
                     in pack_tail(rows, max_chars))


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


def render_stop_block(memories, saved_when: Optional[str] = None,
                      max_pairs: int = STOP_BLOCK_PAIRS_DEFAULT,
                      total_chars: Optional[int] = None):
    """The last exchanges of a session, in order - where it stopped.

    Returns `(text, carried_ids)`. `carried_ids` are the memories the block
    actually rendered (whole or excerpted), which is what the tier loaders
    skip so nothing renders twice; a turn the budget dropped is NOT in it.

    `memories` are chronological rows with `.role`, `.content` and (when they
    are stored memories) `.id`. Returns None when there is less than one
    exchange.

    Short turns are carried word for word. A turn over its share of the
    budget is EXCERPTED - its opening and its ending, with the elided middle
    marked and the memory id named so the full turn can be fetched
    (`search_memories(memory_ids=...)`). The heading says so: the block used
    to keep each turn's first 400 characters and call itself verbatim, which
    dropped the unfinished next step at the end of a long turn - the one
    thing it exists to carry (review 2026-09-11, finding 1).
    """
    rows = [m for m in memories if (getattr(m, "content", "") or "").strip()
            and getattr(m, "role", "") in ("user", "assistant")]
    tail = rows[-(2 * max_pairs):]
    if len(tail) < 2:
        return None, set()
    budget = total_chars if total_chars is not None else 2 * max_pairs * PER_MESSAGE_CHARS
    packed = pack_tail([(m.role, m.content, getattr(m, "id", None)) for m in tail],
                       budget, flatten=True, label=excerpt_label)
    if not packed:
        return None, set()
    when = f" ({saved_when})" if saved_when else ""
    body, excerpted = [], False
    for role, text, ident, was_cut in packed:
        where = ""
        if was_cut:
            excerpted = True
            where = excerpt_label(ident)
        body.append(f"{role}{where}: {text}")
    dropped = len(packed) < len(tail)
    head = [f"Where the last session stopped{when}, in order"
            + (", oldest turns omitted" if dropped else "")
            + (" (long turns excerpted: opening and ending kept, the middle "
               "marked elided; line breaks flattened)" if excerpted
               else ", word for word (line breaks flattened)") + ":"]
    return "\n".join(head + body), {ident for _r, _t, ident, _c in packed if ident}


def stop_block(memories, saved_when: Optional[str] = None,
               max_pairs: int = STOP_BLOCK_PAIRS_DEFAULT,
               total_chars: Optional[int] = None) -> Optional[str]:
    """`render_stop_block`'s text half."""
    return render_stop_block(memories, saved_when, max_pairs, total_chars)[0]


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
