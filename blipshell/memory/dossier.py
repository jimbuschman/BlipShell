"""Project dossier: what a return-after-a-gap needs, assembled from records (V3 E2).

The digest (memory/project_digest.py) is prose the model wrote about the
project; it stays, labelled as inferred. Around it the dossier assembles the
things that are RECORDS and need no model to be right:

- decisions in force, each with its reason and revisit condition (E1)
- decisions recently superseded, labelled, so a rejected idea is not
  re-proposed and the reason it was rejected is one line away
- open follow-ups (the questions still to answer), oldest first
- last completed work: task_complete summaries, marked as the assistant's
  CLAIM unless a verification event exists
- the last closed session
- the next useful action: the oldest open follow-up - never invented
- sources: session and event ids

Updates are EVENT-DRIVEN: every writer appends a project event
(memory/project_events.py) and marks the dossier stale; the next read
re-renders. The nightly `reconcile` folds the new events into the prose
digest (one LLM call, active projects only) and re-renders. Rendering is a
pure function; the render is cached in the project's metadata_json
(`dossier_md`) and exported to the repo's .blipshell/DIGEST.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from blipshell.memory import decisions as dec
from blipshell.memory import project_events as pe
from blipshell.memory import supersession

logger = logging.getLogger(__name__)

RECENT_SUPERSEDED = 5
RECENT_COMPLETED = 5
OPEN_FOLLOWUPS = 10


@dataclass
class Dossier:
    project: str
    digest: Optional[str]
    digest_updated_at: Optional[str]
    decisions_active: list[dec.Decision] = field(default_factory=list)
    decisions_reopened: list[dec.Decision] = field(default_factory=list)  # pending discussion, NOT in force
    decisions_superseded: list[tuple[dec.Decision, str]] = field(default_factory=list)  # (decision, date)
    open_followups: list[dict] = field(default_factory=list)   # oldest first
    completed: list[dict] = field(default_factory=list)        # task_completed / verification events, newest first
    last_session: Optional[dict] = None
    sources: dict = field(default_factory=dict)

    @property
    def next_action(self) -> Optional[dict]:
        return self.open_followups[0] if self.open_followups else None


async def build(sqlite, project: str) -> Dossier:
    row = await sqlite.get_project(project)
    meta = json.loads((row or {}).get("metadata_json") or "{}")
    d = Dossier(project=project, digest=meta.get("digest"), digest_updated_at=meta.get("digest_updated_at"))

    all_decisions = await dec.list_decisions(sqlite, project=project, limit=200)
    d.decisions_active = [x for x in all_decisions if x.status == "active"]
    d.decisions_reopened = [x for x in all_decisions if x.status == "reopened"]
    sup_rows = await supersession.superseded(sqlite, "memory", [x.id for x in all_decisions if x.status == "superseded"],
                                             for_project=project)
    d.decisions_superseded = sorted(
        [(x, (sup_rows[x.id].at if x.id in sup_rows else x.at)[:10]) for x in all_decisions if x.status == "superseded"],
        key=lambda t: t[1], reverse=True,
    )[:RECENT_SUPERSEDED]

    pending = await sqlite.get_pending_follow_ups(project=project, limit=200)
    pending = [f for f in pending if f.get("project") == project]  # the store also returns project-less ones
    d.open_followups = sorted(pending, key=lambda f: f.get("created_at") or "")[:OPEN_FOLLOWUPS]

    d.completed = await pe.events(sqlite, project, kinds=("task_completed", "verification"), limit=RECENT_COMPLETED)
    closed = await pe.events(sqlite, project, kinds=("session_closed",), limit=1)
    d.last_session = closed[0] if closed else None

    d.sources = {
        "digest_session_ids": meta.get("digest_session_ids", []),
        "event_ids": [e["id"] for e in d.completed] + ([d.last_session["id"]] if d.last_session else []),
        "decision_ids": [x.id for x in d.decisions_active] + [x.id for x in d.decisions_reopened],
        "followup_ids": [f["id"] for f in d.open_followups],
    }
    return d


def render(d: Dossier, now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    L = [f"# {d.project} - dossier", f"_Rendered {now.strftime('%Y-%m-%d')}_", ""]

    L.append("## Objective and current state")
    if d.digest:
        stamp = f" (updated {d.digest_updated_at[:10]})" if d.digest_updated_at else ""
        L.append(f"[inferred by the assistant from session summaries{stamp}]")
        L.append(d.digest.strip())
    else:
        L.append("_No digest yet._")
    L.append("")

    L.append("## Decisions in force")
    if d.decisions_active:
        for x in d.decisions_active:
            tag = ""
            who = "" if x.decided_by == "user" else " [proposed by assistant]"
            line = f"- #{x.id}{tag}{who} {x.decision}"
            if x.reason:
                line += f" - because {x.reason}"
            if x.revisit_when:
                line += f" - revisit when: {x.revisit_when}"
            L.append(line)
    else:
        L.append("_None recorded._")
    L.append("")

    if d.decisions_reopened:
        L.append("## Reopened for discussion (NOT in force)")
        for x in d.decisions_reopened:
            line = f"- #{x.id} {x.decision}"
            if x.reopened_reason:
                line += f" - reopened because {x.reopened_reason}"
            if x.superseded_by:
                line += f" - currently replaced by #{x.superseded_by}"
            L.append(line)
        L.append("")

    if d.decisions_superseded:
        L.append("## Recently superseded decisions")
        for x, day in d.decisions_superseded:
            line = f"- #{x.id} [superseded {day}] {x.decision}"
            if x.reason:
                line += f" - was because {x.reason}"
            if x.superseded_by:
                line += f" - replaced by #{x.superseded_by}"
            L.append(line)
        L.append("")

    L.append("## Open questions and follow-ups")
    if d.open_followups:
        for f in d.open_followups:
            due = f" (due: {f['due_hint']})" if f.get("due_hint") else ""
            L.append(f"- #{f['id']} {f['content']}{due}")
    else:
        L.append("_None open._")
    L.append("")

    L.append("## Last completed work")
    if d.completed:
        for e in d.completed:
            state = "verified" if e["kind"] == "verification" else "claimed by assistant, not verified"
            L.append(f"- [{(e.get('at') or '')[:10]}, {state}] {e['summary']}")
    else:
        L.append("_No task completions recorded._")
    L.append("")

    if d.last_session:
        L.append("## Last session")
        L.append(f"[{(d.last_session.get('at') or '')[:10]}] {d.last_session['summary']}")
        L.append("")

    L.append("## Next useful action")
    if d.next_action:
        f = d.next_action
        # Reference, not a repeat: the text is one section up.
        L.append(f"Follow-up #{f['id']} (the oldest open question above)"
                 + (f", due {f['due_hint']}" if f.get("due_hint") else ""))
    else:
        L.append("_None recorded - no open follow-up. Ask before assuming one._")
    L.append("")

    L.append("## Sources")
    L.append(f"digest sessions {d.sources.get('digest_session_ids', [])}; decisions {d.sources.get('decision_ids', [])}; "
             f"follow-ups {d.sources.get('followup_ids', [])}; events {d.sources.get('event_ids', [])}")
    return "\n".join(L)


def listed_ids(d: Dossier) -> tuple[set[int], set[int]]:
    """(decision memory ids, follow-up ids) the render carries. While the
    project is active the pools and the follow-ups block skip these: the
    dossier is their canonical, structured place, a second copy is waste."""
    decision_ids = ({x.id for x in d.decisions_active} | {x.id for x in d.decisions_reopened}
                    | {x.id for x, _ in d.decisions_superseded})
    return decision_ids, {f["id"] for f in d.open_followups}


async def refresh(sqlite, project: str) -> Optional[str]:
    """Build + render + cache. Returns the markdown, or None if no such project."""
    row = await sqlite.get_project(project)
    if not row:
        return None
    d = await build(sqlite, project)
    md = render(d)
    meta = json.loads(row.get("metadata_json") or "{}")
    meta["dossier_md"] = md
    meta["dossier_updated_at"] = datetime.now(timezone.utc).isoformat()
    meta["dossier_stale"] = 0
    await sqlite.update_project(project, metadata_json=json.dumps(meta))
    return md


async def get_dossier_md(sqlite, project: str) -> Optional[str]:
    """Cached render, re-rendered when an event marked it stale or none exists."""
    row = await sqlite.get_project(project)
    if not row:
        return None
    meta = json.loads(row.get("metadata_json") or "{}")
    if meta.get("dossier_md") and not meta.get("dossier_stale"):
        return meta["dossier_md"]
    return await refresh(sqlite, project)


async def get_dossier(sqlite, project: str) -> tuple[Optional[str], set[int], set[int]]:
    """(markdown, decision ids carried, follow-up ids carried) for activation.
    The ids come from a fresh build so they match the records even when the
    cached render is served."""
    md = await get_dossier_md(sqlite, project)
    if md is None:
        return None, set(), set()
    decision_ids, followup_ids = listed_ids(await build(sqlite, project))
    return md, decision_ids, followup_ids


RECONCILE_BATCH = 200


async def reconcile(sqlite, router, project: str) -> dict:
    """Nightly: fold the OLDEST unfolded events (up to RECONCILE_BATCH) into
    the prose digest with one LLM call, then re-render.

    Acknowledgement is a queue, not a timestamp (external review 2026-09-10,
    finding 3 - the same class the commit-evidence queue fixed): the cursor
    `dossier_reconciled_event_id` is the id of the last event actually
    incorporated, advanced ONLY after the updated digest is persisted. A
    failed or empty model reply leaves every event pending; events beyond
    the batch, or arriving during the call, stay pending for the next run.
    Without a digest nothing is folded and nothing is acknowledged: the
    events are already rendered by the dossier itself, and the first session
    close writes the digest they will be folded into.

    Returns: folded (incorporated this run), pending (still unfolded after
    this run), digest_updated, and reason/error when nothing was folded."""
    from blipshell.llm.prompts import update_digest_with_sessions
    from blipshell.llm.router import TaskType

    row = await sqlite.get_project(project)
    if not row:
        return {"project": project, "folded": 0, "pending": 0, "digest_updated": False, "reason": "no such project"}
    meta = json.loads(row.get("metadata_json") or "{}")
    cursor = int(meta.get("dossier_reconciled_event_id") or 0)
    batch = await pe.events_after(sqlite, project, cursor, limit=RECONCILE_BATCH)
    stats = {"project": project, "folded": 0, "pending": len(batch), "digest_updated": False}
    if not batch:
        stats["pending"] = 0
        await refresh(sqlite, project)
        return stats
    if not meta.get("digest"):
        stats["pending"] = await pe.pending_count(sqlite, project, cursor)
        stats["reason"] = "no digest yet - events stay pending until a session close writes one"
        await refresh(sqlite, project)
        return stats

    lines = "\n".join(f"- [{e['kind']}] {e['summary']}" for e in batch)
    system, user = update_digest_with_sessions(meta["digest"], f"[Project events since last reconcile]\n{lines}")
    try:
        updated = await router.generate(TaskType.REASONING, user, system=system)
    except Exception as e:
        logger.warning("Dossier reconcile for '%s' could not fold %d events (left pending): %s",
                       project, len(batch), e)
        stats["error"] = str(e)
        stats["pending"] = await pe.pending_count(sqlite, project, cursor)
        await refresh(sqlite, project)
        return stats
    if not updated or not updated.strip():
        stats["reason"] = "empty model reply - events left pending"
        stats["pending"] = await pe.pending_count(sqlite, project, cursor)
        await refresh(sqlite, project)
        return stats

    # Persist digest + cursor together, on a FRESH read of the metadata so a
    # concurrent writer (a session close during the call) is not overwritten
    # from this stale snapshot.
    fresh_row = await sqlite.get_project(project)
    fresh = json.loads((fresh_row or {}).get("metadata_json") or "{}")
    fresh["digest"] = updated.strip()
    fresh["digest_updated_at"] = datetime.now(timezone.utc).isoformat()
    fresh["dossier_reconciled_event_id"] = int(batch[-1]["id"])
    fresh["dossier_reconciled_at"] = fresh["digest_updated_at"]
    await sqlite.update_project(project, metadata_json=json.dumps(fresh))
    stats["folded"] = len(batch)
    stats["digest_updated"] = True
    stats["pending"] = await pe.pending_count(sqlite, project, int(batch[-1]["id"]))
    await refresh(sqlite, project)
    return stats
