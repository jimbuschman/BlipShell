"""Decisions with conditions (V3 Stage E1).

"We rejected X because Y; reconsider if Y changes." The reason is worth more
than the decision, and the revisit condition is what stops a rejected idea
from being re-proposed forever OR being treated as a permanent prohibition.

A decision is a memory row (`memory_type = 'decision'`) whose content is a
fixed rendering the search index and Recall already understand, with the
structured fields in `metadata_json`. This module is the ONLY reader and
writer of that shape, so moving decisions to a first-class table later is a
one-file change. Revising a decision writes a supersession record
(memory/supersession.py) from the old row to the new one - the old row stays,
un-archived, and surfaces labelled for historical questions. Reopening undoes
that supersession and marks the row reopened, with the reason.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from blipshell.memory import project_events, supersession
from blipshell.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)

STATUSES = ("active", "superseded", "reopened")


@dataclass
class Decision:
    id: int
    decision: str
    reason: str
    revisit_when: str
    status: str
    project: Optional[str]
    decided_by: str          # user | assistant
    at: str
    superseded_by: Optional[int] = None
    reopened_reason: Optional[str] = None

    def render(self) -> str:
        return render_content(self.decision, self.reason, self.revisit_when)


def render_content(decision: str, reason: str, revisit_when: str) -> str:
    """The memory `content` for a decision - what search matches and Recall shows."""
    parts = [f"DECISION: {decision.strip()}"]
    if reason and reason.strip():
        parts.append(f"BECAUSE: {reason.strip()}")
    if revisit_when and revisit_when.strip():
        parts.append(f"REVISIT WHEN: {revisit_when.strip()}")
    return " ".join(p.rstrip(".") + "." for p in parts)


def _meta(decision: str, reason: str, revisit_when: str, status: str, project: Optional[str],
          decided_by: str, **extra) -> str:
    d = {"decision": {"decision": decision, "reason": reason, "revisit_when": revisit_when,
                      "status": status, "project": project, "decided_by": decided_by,
                      "at": datetime.now(timezone.utc).isoformat(), **extra}}
    return json.dumps(d)


def _from_memory(mem: Memory) -> Optional[Decision]:
    try:
        meta = json.loads(mem.metadata_json or "{}").get("decision")
    except (ValueError, TypeError):
        meta = None
    if not isinstance(meta, dict):
        return None
    return Decision(
        id=mem.id, decision=meta.get("decision", ""), reason=meta.get("reason", ""),
        revisit_when=meta.get("revisit_when", ""), status=meta.get("status", "active"),
        project=meta.get("project"), decided_by=meta.get("decided_by", "user"),
        at=meta.get("at", ""), superseded_by=meta.get("superseded_by"),
        reopened_reason=meta.get("reopened_reason"),
    )


async def record_decision(sqlite, vectors, *, decision: str, reason: str = "",
                          revisit_when: str = "", project: Optional[str] = None,
                          session_id: Optional[int] = None, decided_by: str = "user") -> Decision:
    """Store a decision. `decided_by` is the provenance: the user's decision is
    a user_statement; the assistant's own proposal is an inference."""
    decided_by = "assistant" if decided_by == "assistant" else "user"
    content = render_content(decision, reason, revisit_when)
    mem = Memory(
        session_id=session_id, role=decided_by, content=content, summary=content[:200],
        rank=4, importance=0.8, memory_type=MemoryType.DECISION,
        metadata_json=_meta(decision, reason, revisit_when, "active", project, decided_by),
    )
    mid = await sqlite.create_memory(mem)
    if vectors is not None:
        try:
            vectors.add_memory(mid, content, {"session_id": str(session_id), "role": decided_by,
                                              "memory_type": "decision"})
        except Exception as e:
            logger.warning("Decision %d embed failed (FTS still finds it): %s", mid, e)
    logger.info("Decision %d recorded (%s): %s", mid, decided_by, decision[:80])
    await project_events.record_event(
        sqlite, project=project, kind="decision_recorded", summary=decision, ref_kind="memory",
        ref_id=mid, session_id=session_id,
        source_type="user_statement" if decided_by == "user" else "assistant_inference",
    )
    return (await get_decision(sqlite, mid))


async def get_decision(sqlite, decision_id: int) -> Optional[Decision]:
    mem = await sqlite.get_memory(decision_id)
    if mem is None or mem.memory_type != MemoryType.DECISION:
        return None
    return _from_memory(mem)


async def _update_meta(sqlite, decision_id: int, **fields) -> None:
    mem = await sqlite.get_memory(decision_id)
    try:
        meta = json.loads(mem.metadata_json or "{}")
    except (ValueError, TypeError):
        meta = {}
    d = meta.get("decision") or {}
    d.update(fields)
    meta["decision"] = d
    await sqlite.update_memory(decision_id, metadata_json=json.dumps(meta))


async def revise_decision(sqlite, vectors, old_id: int, *, decision: str, reason: str = "",
                          revisit_when: str = "", session_id: Optional[int] = None,
                          decided_by: str = "user") -> Optional[Decision]:
    """A new decision replaces an old one: new row + supersession record.
    Returns the new decision, or None if `old_id` is not a decision."""
    old = await get_decision(sqlite, old_id)
    if old is None:
        return None
    new = await record_decision(sqlite, vectors, decision=decision, reason=reason,
                                revisit_when=revisit_when, project=old.project,
                                session_id=session_id, decided_by=decided_by)
    await _update_meta(sqlite, old_id, status="superseded", superseded_by=new.id)
    await supersession.record(
        sqlite, old_kind="memory", old_id=old_id, new_kind="memory", new_id=new.id,
        scope=old.project, relation="revises", detected_by="decision_tool",
        evidence=(reason or decision)[:300],
        source_type="user_statement" if decided_by != "assistant" else "assistant_inference",
    )
    await project_events.record_event(
        sqlite, project=old.project, kind="decision_revised",
        summary=f"#{old_id} -> #{new.id}: {decision}", ref_kind="memory", ref_id=new.id,
        session_id=session_id,
        source_type="user_statement" if decided_by != "assistant" else "assistant_inference",
    )
    return new


async def reopen_decision(sqlite, decision_id: int, *, reason: str = "",
                          restore: bool = False) -> Optional[Decision]:
    """Two distinct transitions (external review 2026-09-10, finding 4):

    `restore=False` - REOPEN FOR DISCUSSION. The decision's status becomes
    `reopened`: pending, NOT in force. Whatever replaced it stays in force
    and its supersession stays active; `superseded_by` is kept as history.
    The dossier lists it under "reopened for discussion", never among the
    decisions in force, so two contradictory decisions are never both
    governing.

    `restore=True` - RESTORE AS THE GOVERNING DECISION. Its active
    replacement (if any) is retired: marked superseded by this decision,
    with a `revises` supersession record from replacement -> this one, and
    the original supersession that retired this one is undone. This one
    returns to `active` with `superseded_by` cleared. History is preserved
    in both directions.

    Both record a `decision_reopened` event. Returns the decision, or None
    if `decision_id` is not one."""
    dec = await get_decision(sqlite, decision_id)
    if dec is None:
        return None
    now = datetime.now(timezone.utc).isoformat()
    if not restore:
        await _update_meta(sqlite, decision_id, status="reopened", reopened_reason=reason, reopened_at=now)
        await project_events.record_event(
            sqlite, project=dec.project, kind="decision_reopened",
            summary=f"#{decision_id} reopened for discussion: {reason or dec.decision}", ref_kind="memory",
            ref_id=decision_id, source_type="user_statement",
        )
        return await get_decision(sqlite, decision_id)

    # restore: retire the replacement(s) that are still governing
    for rec in await supersession.history_of(sqlite, "memory", decision_id):
        if rec.old_id == decision_id and rec.undone_at is None:
            await supersession.undo(sqlite, rec.id)
            repl = await get_decision(sqlite, rec.new_id)
            if repl is not None and repl.status in ("active", "reopened"):
                await _update_meta(sqlite, repl.id, status="superseded", superseded_by=decision_id)
                await supersession.record(
                    sqlite, old_kind="memory", old_id=repl.id, new_kind="memory", new_id=decision_id,
                    scope=dec.project, relation="revises", detected_by="decision_tool",
                    evidence=(reason or f"restored #{decision_id}")[:300], source_type="user_statement",
                )
    await _update_meta(sqlite, decision_id, status="active", superseded_by=None,
                       reopened_reason=reason, reopened_at=now)
    await project_events.record_event(
        sqlite, project=dec.project, kind="decision_reopened",
        summary=f"#{decision_id} restored as the governing decision: {reason or dec.decision}",
        ref_kind="memory", ref_id=decision_id, source_type="user_statement",
    )
    return await get_decision(sqlite, decision_id)


async def list_decisions(sqlite, *, project: Optional[str] = None,
                         status: Optional[str] = None, limit: int = 50) -> list[Decision]:
    cur = await sqlite._db.execute(
        "SELECT * FROM memories WHERE memory_type = 'decision' AND is_archived = 0 "
        "ORDER BY timestamp DESC LIMIT ?", (int(limit) * 4,),
    )
    rows = await cur.fetchall()
    out = []
    for r in rows:
        mem = sqlite._row_to_memory(r)
        d = _from_memory(mem)
        if d is None:
            continue
        if project is not None and d.project != project:
            continue
        if status is not None and d.status != status:
            continue
        out.append(d)
        if len(out) >= limit:
            break
    return out
