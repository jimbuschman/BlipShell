"""Project events: the deterministic record the dossier is built from (V3 E2).

Every material change to a project's state is appended here by the code
that makes it - a decision recorded/revised/reopened, a task_complete call,
a follow-up added/resolved/dismissed, a session closed. The dossier
(memory/dossier.py) reads these; the nightly reconcile folds them into the
prose digest. Events carry the provenance of their source (B4 vocabulary):
a task_complete summary is the assistant's CLAIM until something verifies it.

Kept separate from dossier.py so the writers (decisions, tools, session
close) can import it without a cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

EVENT_KINDS = (
    "decision_recorded", "decision_revised", "decision_reopened",
    "task_completed", "verification",
    "followup_added", "followup_resolved", "followup_dismissed",
    "session_closed",
)


async def record_event(sqlite, *, project: Optional[str], kind: str, summary: str,
                       ref_kind: Optional[str] = None, ref_id: Optional[int] = None,
                       session_id: Optional[int] = None,
                       source_type: str = "assistant_inference") -> Optional[int]:
    """Append one event. No project -> nothing to attach it to -> None.
    Never raises into the caller: the dossier is derived state."""
    if not project:
        return None
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown event kind {kind!r}")
    try:
        cur = await sqlite._db.execute(
            "INSERT INTO project_events (project, kind, ref_kind, ref_id, summary, source_type, session_id, at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (project, kind, ref_kind, ref_id, (summary or "")[:500], source_type, session_id,
             datetime.now(timezone.utc).isoformat()),
        )
        await sqlite._db.execute(
            "UPDATE projects SET metadata_json = json_set(COALESCE(metadata_json, '{}'), '$.dossier_stale', 1) "
            "WHERE name = ?", (project,),
        )
        await sqlite._db.commit()
        return int(cur.lastrowid)
    except Exception as e:
        logger.warning("project event not recorded (%s/%s): %s", project, kind, e)
        return None


async def events(sqlite, project: str, *, since: Optional[str] = None,
                 kinds: Optional[Iterable[str]] = None, limit: int = 100) -> list[dict]:
    sql = "SELECT * FROM project_events WHERE project = ?"
    args: list = [project]
    if since:
        sql += " AND at > ?"
        args.append(since)
    if kinds:
        ks = list(kinds)
        sql += f" AND kind IN ({','.join('?' for _ in ks)})"
        args.extend(ks)
    sql += " ORDER BY at DESC, id DESC LIMIT ?"
    args.append(int(limit))
    cur = await sqlite._db.execute(sql, args)
    return [dict(r) for r in await cur.fetchall()]


async def events_after(sqlite, project: str, after_id: int, *, limit: int = 200) -> list[dict]:
    """Events with id > after_id, OLDEST first - the consumption order for a
    cursor. Ids are the stable high-water mark; timestamps are not (two
    events can share one, and a reconcile that stamps 'now' skips whatever
    arrived during the model call)."""
    cur = await sqlite._db.execute(
        "SELECT * FROM project_events WHERE project = ? AND id > ? ORDER BY id ASC LIMIT ?",
        (project, int(after_id), int(limit)),
    )
    return [dict(r) for r in await cur.fetchall()]


async def pending_count(sqlite, project: str, after_id: int) -> int:
    cur = await sqlite._db.execute(
        "SELECT COUNT(*) FROM project_events WHERE project = ? AND id > ?", (project, int(after_id)))
    return int((await cur.fetchone())[0])


async def active_projects(sqlite, *, days: int = 14) -> list[str]:
    """Projects with an event or a session in the window - the ones worth
    spending nightly compute on."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cur = await sqlite._db.execute(
        "SELECT DISTINCT project FROM project_events WHERE at > ? "
        "UNION SELECT DISTINCT project FROM sessions WHERE project IS NOT NULL AND last_active > ?",
        (cutoff, cutoff),
    )
    return sorted({r[0] for r in await cur.fetchall() if r[0]})


async def follow_up_project(sqlite, follow_up_id: int) -> Optional[str]:
    cur = await sqlite._db.execute("SELECT project FROM follow_ups WHERE id = ?", (int(follow_up_id),))
    r = await cur.fetchone()
    return r[0] if r else None
