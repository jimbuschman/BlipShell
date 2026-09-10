"""Scenario: resume the live thread of the previous session (conversation
continuity, 2026-09-10). Separate category from the Stage E gate so the
gate's frozen shape is untouched.

The seeded previous session ended abnormally (no summary): 24 higher-
importance filler exchanges and a final exchange that stops on a half-formed
idea, plus the working-state note a mid-session refresh would have left. The
question is the one BlipShell was asked on 2026-08-11 ("do you remember the
last thing we talked about?"). The reply must pick up the half-formed idea
itself, not a generic account of the session and not a confession of having
lost the thread.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from blipshell.simulate.models import SimScenario, SimStep, StepAction

CATEGORY = "continuity_thread"
HALF_FORMED = ("Right, and that is the weak point - what if the state note were written every few turns instead of "
               "only at the end? I had not finished that thought.")
NOTE = "Mid-way through arguing for writing the state note every few turns; next I meant to try it on the raycaster branch."


async def seed_previous_session(ctx) -> None:
    from blipshell.core.handoff import HANDOFF_KEY, HANDOFF_META_KEY
    from blipshell.models.memory import Memory
    agent = ctx.agent
    then = datetime.now(timezone.utc) - timedelta(days=2)
    sid = await agent.sqlite.create_session(title="New Session", created_at=then)
    for i in range(24):
        text = f"Earlier in that session we went over point {i} about the raycaster column renderer and the fixed-point math."
        mid = await agent.sqlite.create_memory(Memory(session_id=sid, role="user" if i % 2 == 0 else "assistant",
                                                      content=text, summary=text[:120], importance=0.8, rank=3,
                                                      timestamp=then + timedelta(minutes=i)))
        agent.vectors.add_memory(mid, text, {"session_id": str(sid)})
    for j, (role, text) in enumerate((("user", "so the state note only gets written when the session closes?"),
                                       ("assistant", HALF_FORMED))):
        mid = await agent.sqlite.create_memory(Memory(session_id=sid, role=role, content=text, summary=text[:120],
                                                      importance=0.35, rank=3, timestamp=then + timedelta(minutes=30 + j)))
        agent.vectors.add_memory(mid, text, {"session_id": str(sid)})
    await agent.sqlite.set_metadata(HANDOFF_KEY, NOTE)
    await agent.sqlite.set_metadata(HANDOFF_META_KEY, json.dumps({"saved_at": then.isoformat(), "session_id": sid,
                                                                  "midsession": True}))
    # session ended abnormally: no summary, message_count stays 0
    await agent.sqlite._db.execute("UPDATE sessions SET last_active = ? WHERE id = ?",
                                   (then.strftime("%Y-%m-%d %H:%M:%S"), sid))
    await agent.sqlite._db.commit()


def score_resume_thread(text: str) -> list[str]:
    low = (text or "").lower()
    misses: list[str] = []
    if not any(k in low for k in ("every few turns", "written every", "mid-session", "during the session", "instead of only at the end")):
        misses.append("does not pick up the half-formed idea (state note written every few turns)")
    if not any(k in low for k in ("state note", "handoff", "note")):
        misses.append("does not name the thread (the state note)")
    if any(k in low for k in ("i don't have", "i can't recall", "i do not have any record", "no record of", "i don't remember")):
        misses.append("disclaims the thread instead of resuming it")
    return misses


def get_scenarios() -> list[SimScenario]:
    return [SimScenario(
        name="resume_last_thread",
        description="Resume the half-formed idea the previous (abnormally ended) session stopped on",
        category=CATEGORY,
        setup=seed_previous_session,
        fresh_db=True,
        steps=[SimStep(action=StepAction.CHAT, input="do you remember the last thing we talked about?",
                       description="resume last thread", response_validator=score_resume_thread,
                       expect_no_write_tools=True, timeout_seconds=600.0)],
    )]
