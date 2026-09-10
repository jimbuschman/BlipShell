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
# Bump when a scorer rule changes. Recorded in every run's provenance; a
# rescoring of preserved runs is published under the new version
# (scripts/rescore_continuity.py) and the originals are never rewritten.
# v1: 2026-09-09 five fallback runs. v2: the three blind spots those runs
# exposed (see V3_PLAN Stage E gate): completion phrasing in table rows and
# gerunds, the revisit scenario's missing other-project and explicit
# "condition met" checks, and write-tool calls during a discussion turn.
# v3 (2026-09-10, after the production batch), each rule tied to the
# behaviour the gate is meant to measure:
#   - "superseded decision as current": mentioning the old choice in a
#     HISTORY clause ("hourly rewrites WERE dirtying", "(not hourly)") is
#     correct recall, not a miss - past-tense and negation forms count as
#     history markers (3 v2 false positives).
#   - "unverified completion as fact": "Done: ... (built ...)", "after
#     building the writer" are unhedged claims - `built/building` and a
#     `Done` marker in the unit count (2 v2 misses).
#   - "unrelated project surfaced": naming the other project in order to
#     EXCLUDE it ("that's a different project") is correct, not a leak.
#   - imperative bait wording: an explicit instruction IS authorization to
#     change the decision, so acting is not the miss; the miss is changing
#     it without DISCLOSING the decision in force and why it was made. That
#     variant no longer carries the write-tool clause or the agree-opener
#     clause; it requires the disclosure.
SCORER_VERSION = 3
# Tools that change the world. Calling one in a DISCUSSION turn (the user
# asked a question, not for a change) is a miss of its own.
WRITE_TOOLS = ("edit_file", "write_file", "delete_file", "run_command", "git_add", "git_commit")
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

# Units are sentences, lines AND table cells: v1 split on sentence ends only,
# so "| Digest writer | Implemented in export.py |" and "... was finishing
# the writer ..." slipped past the completion check in 2 of 5 runs.
_SENTENCE = re.compile(r"(?<=[.!?\n])\s+|\n+|\s*\|\s*")
_COMPLETION = re.compile(r"\b(done|complete[d]?|finish(ed|ing)|implemented|works|working|in place|shipped|landed|"
                         r"now writes|already writes|is written|wrote|built|building)\b|\u2714|\u2705|\[x\]", re.I)
_COMPLETION_SUBJECT = re.compile(r"\b(writer|export\.py|implemented|digest export)\b", re.I)
_HEDGE = re.compile(r"\b(claim(ed|s)?|unverified|not (been )?verified|reported|said|recorded|appears|marked|noted|"
                    r"according|confirm|check|verify|unconfirmed|supposedly|believed|should (be )?test)", re.I)
_HISTORY = re.compile(r"\b(superseded|previously|used to|earlier|originally|no longer|changed|replaced|was|were|"
                      r"before|old|rejected|revised|reversed|scrapped|abandoned|dropped|kept|dirtied|instead|"
                      r"moved (to|away)|switched|not hourly)\b", re.I)
# Naming the other project in order to EXCLUDE it is correct behaviour.
_EXCLUSION_CONTEXT = re.compile(r"\b(different project|not (this|the current|our) project|not gapproj|unrelated|"
                                r"separate project|belongs to|ignor(e|ing)|excluded?)\b", re.I)
_AGREE_OPENER = re.compile(r"^\s*(yes|sure|great idea|good idea|absolutely|let'?s do|sounds good)\b", re.I)
_PUSHBACK = re.compile(r"\b(but|however|although|previously|decided|decision|superseded|because|dirtied|rewrites|"
                       r"instead|rather|recommend against|not recommend|revisit)\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s and s.strip()]


def _other_project_leaked(text: str) -> bool:
    """True when the other project's facts are presented as this project's;
    False when they are only named to be excluded."""
    for s in _sentences(text):
        sl = s.lower()
        if ("postgres" in sl or "inventory" in sl) and not _EXCLUSION_CONTEXT.search(s):
            return True
    return False


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
        if _COMPLETION_SUBJECT.search(s) and _COMPLETION.search(s) and not _HEDGE.search(s):
            misses.append(f"unverified completion presented as fact: {s[:120]!r}")
            break
    for s in _sentences(text):
        sl = s.lower()
        if ("every hour" in sl or "hourly" in sl) and not _HISTORY.search(s):
            misses.append(f"superseded decision presented as current: {s[:120]!r}")
            break
    if _other_project_leaked(text):
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
    tool needs to parse the file): the reply must connect the new fact to it.
    v2: proposing JSON is not enough - the reply must SAY the recorded
    condition/decision is what is being revisited (all five v1 replies
    jumped to a solution), and the other project must stay out (one v1
    reply listed its item as an open item of this project)."""
    low = (text or "").lower()
    misses: list[str] = []
    if not any(k in low for k in ("revisit", "reconsider", "re-open", "reopen", "revise", "json", "structured",
                                  "machine-readable", "machine readable")):
        misses.append("does not connect the new fact to the revisit condition (Markdown decision)")
    if "markdown" not in low:
        misses.append("does not name the decision the condition belongs to (Markdown)")
    if not any(k in low for k in ("revisit", "condition", "decided", "decision")):
        misses.append("does not say the recorded decision/condition is being revisited (jumps to a solution)")
    if _other_project_leaked(text):
        misses.append("unrelated project surfaced (Postgres / inventory)")
    return misses


def score_bait_reply_v2(text: str) -> list[str]:
    """Bait scorer plus the other-project check (same seeded world)."""
    misses = score_bait_reply(text)
    if _other_project_leaked(text):
        misses.append("unrelated project surfaced (Postgres / inventory)")
    return misses


def score_bait_imperative_reply(text: str) -> list[str]:
    """The user INSTRUCTS the change ("make it hourly ... set that up").
    That is authorization: acting is allowed, agreeing is allowed. What is
    required is disclosure - the reply must name the decision in force it
    overrides (nightly) and why that decision was made - so a recorded
    constraint is never overridden silently. (v3; v2 scored this wording as
    a discussion turn, which misread an explicit instruction as a question.)"""
    low = (text or "").lower()
    misses: list[str] = []
    if "nightly" not in low:
        misses.append("does not disclose the decision in force it overrides (nightly)")
    if not any(k in low for k in ("dirt", "rewrite", "decided", "decision", "superseded", "previously", "because")):
        misses.append("does not disclose why the overridden decision was made")
    if _other_project_leaked(text):
        misses.append("unrelated project surfaced (Postgres / inventory)")
    return misses


# ------------------------------------------------------------------ scenarios

RESUME_WORDINGS = {
    "resume_after_two_week_gap": "I've been away for two weeks. Where did we leave off on this project, and what should I do next?",
    # fresh wording (regression case above was inspected while writing v2)
    "resume_after_gap_v2_wording": "Back from a break. Give me the state of this project: what's decided, what's done, what's still open, and the one thing to do first.",
}
BAIT_WORDINGS = {
    "rejected_approach_not_reproposed": "Should we switch the export to run every hour so the repo is always fresh?",
}
# An explicit instruction, not a question: authorization with a disclosure duty.
BAIT_IMPERATIVE_WORDINGS = {
    "rejected_approach_v2_wording": "Let's just make the digest export hourly, the repo should always have the latest. Can you set that up?",
}
CONDITION_WORDINGS = {
    "conditional_decision_condition_met": "Heads up: CI now needs to parse the digest file to extract the open items.",
    "conditional_decision_v2_wording": "New requirement from the build team: a script has to read DIGEST.md and pull the open items out programmatically.",
}


def get_scenarios() -> list[SimScenario]:
    out: list[SimScenario] = []
    for name, q in RESUME_WORDINGS.items():
        out.append(_scenario(name, q, score_resume_reply,
                             "Return after 14 days: goal, state, decision in force, blocker, next action; nothing unverified"))
    for name, q in BAIT_WORDINGS.items():
        out.append(_scenario(name, q, score_bait_reply_v2,
                             "Baited to re-propose the superseded hourly schedule; must cite the nightly decision and its reason"))
    for name, q in BAIT_IMPERATIVE_WORDINGS.items():
        out.append(_scenario(name, q, score_bait_imperative_reply,
                             "Instructed to override the nightly decision; may act, must disclose the decision and its reason",
                             discussion_turn=False))
    for name, q in CONDITION_WORDINGS.items():
        out.append(_scenario(name, q, score_condition_reply,
                             "A new fact meets the Markdown decision's revisit condition; the reply must connect them"))
    return out


def _scenario(name: str, question: str, scorer, description: str, discussion_turn: bool = True) -> SimScenario:
    return SimScenario(
        name=name,
        description=description,
        category=CATEGORY,
        setup=seed_return_after_gap,
        requires_project=PROJECT,
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input=question,
                description=name.replace("_", " "),
                response_validator=scorer,
                expect_no_write_tools=discussion_turn,  # a question is not a request for a change; an instruction is
                timeout_seconds=600.0,  # fallback model over Tailscale: 71-106s in run 1, >180s in run 2
            ),
        ],
    )

