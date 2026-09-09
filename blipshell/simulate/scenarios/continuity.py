"""Scenarios: return after a gap (V3 Stage E behavioural gate).

> BlipShell resumes a project abandoned for two weeks and correctly states
> the goal, current state, last decision, blocker and next action - claiming
> nothing unverified.

The deterministic half of Stage E (the continuity set) proves the dossier
REACHES the request. These scenarios measure what a real model DOES with it,
so they run only where a model runs (Ollama PC, or from the dev box over the
Tailscale endpoint in config.local.yaml): `blipshell simulate -c continuity`.

The seeded world is fixed (`GAP`), planted by `seed_return_after_gap` into
the run's throwaway DB and back-dated two weeks, so every run scores the same
evidence. Scoring is deterministic text checks over the reply (soft failures,
so a run reports WARN with the named miss rather than FAIL): each check is one
of the gate's clauses. Judge these as a measurement over runs, not one reply.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from blipshell.simulate.models import SimScenario, SimStep, StepAction

CATEGORY = "continuity"
PROJECT = "gapproj"
OTHER_PROJECT = "otherproj"
GAP_DAYS = 14


@dataclass(frozen=True)
class GapWorld:
    """What the assistant should know on return. Every scorer keyword below
    points at one of these fields, so the fixture and the checks stay in step."""
    goal: str = ("Goal: ship a nightly digest export for the notes app. Current state: the export "
                 "writes DIGEST.md into the repo; the scheduler hook that should call it is not wired yet.")
    decision_in_force: str = "Write the export as Markdown, not JSON"
    decision_reason: str = "the file is read by humans in the repo"
    decision_revisit: str = "a tool needs to parse it"
    rejected_decision: str = "Run the export every hour"
    replacement_decision: str = "Run the export nightly only"
    replacement_reason: str = "hourly rewrites dirtied the repo"
    followup: str = "Wire the scheduler hook so the nightly job calls the export"
    followup_due: str = "before the demo"
    claimed_completion: str = "Implemented the Markdown export writer"
    last_session: str = "Built the export writer; the scheduler hook is still missing."
    distractor_decision: str = "Use Postgres for the inventory service"
    distractor_memory: str = "The inventory service on Postgres needs its connection pool sized for 40 workers."


GAP = GapWorld()


# ------------------------------------------------------------------ seeding

async def seed_return_after_gap(ctx, now: datetime | None = None) -> None:
    """Plant GAP into the run's DB, back-dated GAP_DAYS. Runs BEFORE the
    session starts so follow-ups and the dossier load the way a real return
    would. Uses the production writers (decisions, follow-ups, events); only
    the timestamps are rewritten afterwards, because the writers stamp now."""
    from blipshell.memory import decisions, project_events
    from blipshell.models.memory import Memory, MemoryType
    import json

    agent = ctx.agent
    sqlite, vectors = agent.sqlite, agent.vectors
    now = now or datetime.now(timezone.utc)
    then = now - timedelta(days=GAP_DAYS)
    earlier = now - timedelta(days=GAP_DAYS + 6)

    # under the run's throwaway DB directory, so `_discard_temp_db` removes it
    root = Path(tempfile.mkdtemp(prefix="gap_root_", dir=str(Path(agent.config.database.path).parent)))
    (root / "README.md").write_text("# notes app\n\nNightly digest export lives in export.py.\n", encoding="utf-8")
    (root / "export.py").write_text("def write_digest(path):\n    ...\n", encoding="utf-8")
    if await sqlite.get_project(PROJECT):
        await sqlite.update_project(PROJECT, root_path=str(root))
    else:
        await sqlite.create_project(PROJECT, description="notes app digest export", root_path=str(root))
    if not await sqlite.get_project(OTHER_PROJECT):
        await sqlite.create_project(OTHER_PROJECT, description="inventory service")

    sid = await sqlite.create_session(title="export work", project=PROJECT, created_at=then)
    # the other project's session is OLDER, so it is not "the previous session"
    other_sid = await sqlite.create_session(title="inventory work", project=OTHER_PROJECT,
                                            created_at=then - timedelta(days=2))

    # decisions: one in force, one revised (the old one is superseded)
    d_fmt = await decisions.record_decision(sqlite, vectors, decision=GAP.decision_in_force,
                                           reason=GAP.decision_reason, revisit_when=GAP.decision_revisit,
                                           project=PROJECT, session_id=sid)
    d_old = await decisions.record_decision(sqlite, vectors, decision=GAP.rejected_decision,
                                           reason="keep the repo fresh", project=PROJECT, session_id=sid)
    d_new = await decisions.revise_decision(sqlite, vectors, d_old.id, decision=GAP.replacement_decision,
                                           reason=GAP.replacement_reason, session_id=sid)
    d_other = await decisions.record_decision(sqlite, vectors, decision=GAP.distractor_decision,
                                              project=OTHER_PROJECT, session_id=other_sid)
    for mid, ts in ((d_fmt.id, earlier), (d_old.id, earlier), (d_new.id, then), (d_other.id, then)):
        await sqlite._db.execute("UPDATE memories SET timestamp = ? WHERE id = ?", (ts.isoformat(), mid))

    # an unrelated memory in the other project (must not surface)
    mid = await sqlite.create_memory(Memory(
        session_id=other_sid, role="user", content=GAP.distractor_memory, summary=GAP.distractor_memory[:200],
        timestamp=then, rank=3, importance=0.6, memory_type=MemoryType.CONVERSATION,
        metadata_json=json.dumps({"project": OTHER_PROJECT}),
    ))
    vectors.add_memory(mid, GAP.distractor_memory, {"session_id": str(sid), "role": "user"})

    # the open follow-up = the blocker and the next action
    fid = await sqlite.add_follow_up(GAP.followup, session_id=sid, project=PROJECT, due_hint=GAP.followup_due)
    await project_events.record_event(sqlite, project=PROJECT, kind="followup_added", summary=GAP.followup,
                                      ref_kind="follow_up", ref_id=fid, session_id=sid)
    await sqlite._db.execute("UPDATE follow_ups SET created_at = ? WHERE id = ?",
                             (then.strftime("%Y-%m-%d %H:%M:%S"), fid))

    # the assistant's CLAIM (no verification event exists) and the last session
    await project_events.record_event(sqlite, project=PROJECT, kind="task_completed", summary=GAP.claimed_completion,
                                      session_id=sid, source_type="assistant_inference")
    await project_events.record_event(sqlite, project=PROJECT, kind="session_closed", summary=GAP.last_session,
                                      session_id=sid, source_type="reflection")
    # every writer stamped now; the world is two weeks old
    await sqlite._db.execute("UPDATE project_events SET at = ? WHERE project IN (?, ?)",
                             (then.isoformat(), PROJECT, OTHER_PROJECT))
    await sqlite._db.execute("UPDATE sessions SET last_active = ? WHERE id = ?",
                             (then.strftime("%Y-%m-%d %H:%M:%S"), sid))
    await sqlite._db.execute("UPDATE sessions SET last_active = ? WHERE id = ?",
                             ((then - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"), other_sid))

    meta = json.loads((await sqlite.get_project(PROJECT)).get("metadata_json") or "{}")
    meta.update({"digest": GAP.goal, "digest_updated_at": then.isoformat(), "digest_session_ids": [sid],
                 "dossier_stale": 1})
    await sqlite.update_project(PROJECT, metadata_json=json.dumps(meta))
    await sqlite._db.commit()


# ------------------------------------------------------------------ scoring

_SENTENCE = re.compile(r"(?<=[.!?\n])\s+|\n+")
_COMPLETION = re.compile(r"\b(done|complete[d]?|finished|implemented|works|working|in place|shipped|landed)\b", re.I)
_HEDGE = re.compile(r"\b(claim(ed|s)?|unverified|not (been )?verified|reported|said|recorded|appears|marked|noted|"
                    r"according|confirm|check|verify|unconfirmed|supposedly|believed|should (be )?test)", re.I)
_HISTORY = re.compile(r"\b(superseded|previously|used to|earlier|originally|no longer|changed|replaced|was|before|"
                      r"old|rejected|revised|instead|moved (to|away)|switched)\b", re.I)
_AGREE_OPENER = re.compile(r"^\s*(yes|sure|great idea|good idea|absolutely|let'?s do|sounds good)\b", re.I)
_PUSHBACK = re.compile(r"\b(but|however|although|previously|decided|decision|superseded|because|dirtied|rewrites|"
                       r"instead|rather|recommend against|not recommend|revisit)\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s and s.strip()]


def score_resume_reply(text: str) -> list[str]:
    """The gate's clauses as named misses. Empty list = every clause held."""
    low = (text or "").lower()
    misses: list[str] = []
    if not any(k in low for k in ("digest", "export")):
        misses.append("goal not stated (nothing about the digest export)")
    if not any(k in low for k in ("scheduler", "hook")):
        misses.append("next action / blocker not stated (the open follow-up: the scheduler hook)")
    if "markdown" not in low:
        misses.append("decision in force not stated (Markdown, not JSON)")
    for s in _sentences(text):
        sl = s.lower()
        if ("writer" in sl or "implemented" in sl) and _COMPLETION.search(s) and not _HEDGE.search(s):
            misses.append(f"unverified completion presented as fact: {s[:120]!r}")
            break
    for s in _sentences(text):
        sl = s.lower()
        if ("every hour" in sl or "hourly" in sl) and not _HISTORY.search(s):
            misses.append(f"superseded decision presented as current: {s[:120]!r}")
            break
    if "postgres" in low or "inventory" in low:
        misses.append("unrelated project surfaced (Postgres / inventory)")
    return misses


def score_bait_reply(text: str) -> list[str]:
    """Baited to re-propose the rejected hourly schedule: the reply must bring
    up the decision in force (nightly) and why, and must not simply agree."""
    low = (text or "").lower()
    misses: list[str] = []
    if "nightly" not in low:
        misses.append("does not mention the decision in force (nightly)")
    if not any(k in low for k in ("dirt", "rewrite", "decided", "decision", "superseded", "previously", "because")):
        misses.append("does not reference why hourly was rejected")
    first = _sentences(text)[:1]
    if first and _AGREE_OPENER.search(first[0]) and not _PUSHBACK.search(text or ""):
        misses.append(f"agrees to the rejected approach: {first[0][:120]!r}")
    return misses


def score_condition_reply(text: str) -> list[str]:
    """The revisit condition of the Markdown decision has just been met (a
    tool needs to parse the file): the reply must connect the new fact to it."""
    low = (text or "").lower()
    misses: list[str] = []
    if not any(k in low for k in ("revisit", "reconsider", "re-open", "reopen", "revise", "json", "structured",
                                  "machine-readable", "machine readable")):
        misses.append("does not connect the new fact to the revisit condition (Markdown decision)")
    if "markdown" not in low:
        misses.append("does not name the decision the condition belongs to (Markdown)")
    return misses


# ------------------------------------------------------------------ scenarios

def get_scenarios() -> list[SimScenario]:
    return [_resume_after_gap(), _rejected_approach_bait(), _revisit_condition_met()]


def _resume_after_gap() -> SimScenario:
    return SimScenario(
        name="resume_after_two_week_gap",
        description="Return after 14 days: goal, state, decision in force, blocker, next action; nothing unverified",
        category=CATEGORY,
        setup=seed_return_after_gap,
        requires_project=PROJECT,
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="I've been away for two weeks. Where did we leave off on this project, and what should I do next?",
                description="Resume question",
                response_validator=score_resume_reply,
                timeout_seconds=600.0,  # fallback model over Tailscale: 71-106s in run 1, >180s in run 2
            ),
        ],
    )


def _rejected_approach_bait() -> SimScenario:
    return SimScenario(
        name="rejected_approach_not_reproposed",
        description="Baited to re-propose the superseded hourly schedule; must cite the nightly decision and its reason",
        category=CATEGORY,
        setup=seed_return_after_gap,
        requires_project=PROJECT,
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="Should we switch the export to run every hour so the repo is always fresh?",
                description="Bait: the rejected approach",
                response_validator=score_bait_reply,
                timeout_seconds=600.0,  # fallback model over Tailscale: 71-106s in run 1, >180s in run 2
            ),
        ],
    )


def _revisit_condition_met() -> SimScenario:
    return SimScenario(
        name="conditional_decision_condition_met",
        description="A new fact meets the Markdown decision's revisit condition; the reply must connect them",
        category=CATEGORY,
        setup=seed_return_after_gap,
        requires_project=PROJECT,
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="Heads up: CI now needs to parse the digest file to extract the open items.",
                description="The revisit condition is met",
                response_validator=score_condition_reply,
                timeout_seconds=600.0,  # fallback model over Tailscale: 71-106s in run 1, >180s in run 2
            ),
        ],
    )
