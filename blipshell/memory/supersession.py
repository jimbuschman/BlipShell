"""Explicit, scoped supersession with provenance (V3 Stage E1).

When a newer fact replaces an older one - the dedup verdict says DELETE or
UPDATE, the core-memory contradiction check says YES, a decision is revised -
the old record used to be ARCHIVED: gone from retrieval, its vector deleted,
the relationship living only in a log line. That loses the history ("how did
my preference change?") and hides the reason.

Now a supersession is a RECORD in its own table:

    old (kind, id)  --relation-->  new (kind, id)
    scope        : the project the supersession holds in, or 'global'
    relation     : contradicts | refines | revises
    detected_by  : dedup_verdict | core_contradiction | decision_tool | user
    evidence     : the verdict text / judge answer / user's words (short)
    source_type  : provenance of the NEW record (B4 vocabulary)
    at / undone_at

The old record stays where it was, un-archived, vector intact. Search hides
superseded records for CURRENT-state questions and includes them, labelled,
for HISTORICAL ones (`is_historical_question`). Reversible: `undo` clears
`undone_at`-style, never deletes. Scope protects unrelated projects: a
verdict reached in project A may only supersede candidates in A or in no
project - see `same_scope`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

RELATIONS = ("contradicts", "refines", "revises")
DETECTORS = ("dedup_verdict", "core_contradiction", "decision_tool", "user")
GLOBAL_SCOPE = "global"

# A question about how something USED to be, or how it changed, wants the
# superseded record too. Deterministic, over-inclusive on purpose: a false
# positive shows a labelled old fact; a false negative hides history.
_HISTORICAL = re.compile(
    r"(?:"
    r"\b(?:how|when|why) did .{0,60}\b(?:change|switch|evolve|move|go from|become|start|stop)\b"
    r"|\bover time\b|\bhistory of\b|\bused to\b|\bat first\b|\boriginally\b|\bpreviously\b"
    r"|\b(?:what|which) did (?:i|we) (?:use|prefer|have|decide|say|think)\b.{0,40}\b(?:before|earlier|then|originally|previously)\b"
    r"|\bbefore (?:i|we) (?:switched|changed|moved|decided)\b"
    r"|\b(?:changed|switched|moved) (?:my|our|the) \w+ (?:from|preference)\b"
    r"|\bwhat (?:was|were) (?:my|our|the) .{0,40}\b(?:before|originally|previously|back then)\b"
    r"|\btimeline\b|\bchronolog"
    r")",
    re.IGNORECASE,
)


def is_historical_question(query: str) -> bool:
    return bool(query) and _HISTORICAL.search(query) is not None


def scope_of(project: Optional[str]) -> str:
    return project if project else GLOBAL_SCOPE


def same_scope(new_project: Optional[str], candidate_project: Optional[str]) -> bool:
    """May a record from `new_project` supersede one from `candidate_project`?

    Yes when either side has no project (global facts are shared) or both
    name the same project. Two DIFFERENT projects never supersede each other:
    "the vector store is now sqlite-vec" in blipshell says nothing about
    Wisp's vector store, however similar the sentences look.
    """
    if not new_project or not candidate_project:
        return True
    return new_project == candidate_project


@dataclass
class Supersession:
    id: int
    old_kind: str
    old_id: int
    new_kind: str
    new_id: int
    scope: str
    relation: str
    detected_by: str
    evidence: str
    source_type: str
    at: str
    undone_at: Optional[str] = None

    def label(self) -> str:
        """Rendered prefix for the OLD record when it is shown at all."""
        day = (self.at or "")[:10]
        return f"[superseded {day} by {self.new_kind} {self.new_id}] "


async def record(sqlite, *, old_kind: str, old_id: int, new_kind: str, new_id: int,
                 scope: Optional[str], relation: str, detected_by: str,
                 evidence: str = "", source_type: str = "unknown") -> int:
    """Write one supersession. Idempotent on (old, new, relation): a repeated
    detection returns the existing active row instead of a duplicate."""
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}, got {relation!r}")
    if detected_by not in DETECTORS:
        raise ValueError(f"detected_by must be one of {DETECTORS}, got {detected_by!r}")
    scope = scope_of(scope)
    cur = await sqlite._db.execute(
        "SELECT id FROM supersessions WHERE old_kind=? AND old_id=? AND new_kind=? AND new_id=? "
        "AND relation=? AND undone_at IS NULL",
        (old_kind, int(old_id), new_kind, int(new_id), relation),
    )
    row = await cur.fetchone()
    if row:
        return int(row[0])
    cur = await sqlite._db.execute(
        "INSERT INTO supersessions (old_kind, old_id, new_kind, new_id, scope, relation, detected_by, "
        "evidence, source_type, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (old_kind, int(old_id), new_kind, int(new_id), scope, relation, detected_by,
         (evidence or "")[:300], source_type or "unknown", datetime.now(timezone.utc).isoformat()),
    )
    await sqlite._db.commit()
    logger.info("Supersession: %s %d -%s-> %s %d [%s] by %s", old_kind, old_id, relation,
                new_kind, new_id, scope, detected_by)
    return int(cur.lastrowid)


def _row(r) -> Supersession:
    return Supersession(
        id=int(r["id"]), old_kind=r["old_kind"], old_id=int(r["old_id"]),
        new_kind=r["new_kind"], new_id=int(r["new_id"]), scope=r["scope"],
        relation=r["relation"], detected_by=r["detected_by"], evidence=r["evidence"] or "",
        source_type=r["source_type"] or "unknown", at=r["at"] or "", undone_at=r["undone_at"],
    )


async def superseded(sqlite, kind: str, ids: Iterable[int]) -> dict[int, Supersession]:
    """old_id -> its active supersession (the most recent, if several)."""
    ids = [int(i) for i in ids if i is not None]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    cur = await sqlite._db.execute(
        f"SELECT * FROM supersessions WHERE old_kind = ? AND old_id IN ({placeholders}) "
        f"AND undone_at IS NULL ORDER BY at ASC, id ASC",
        [kind, *ids],
    )
    out: dict[int, Supersession] = {}
    for r in await cur.fetchall():
        out[int(r["old_id"])] = _row(r)  # later rows overwrite: newest wins
    return out


async def history_of(sqlite, kind: str, record_id: int) -> list[Supersession]:
    """Every supersession touching a record, as old OR new, oldest first, undone included."""
    cur = await sqlite._db.execute(
        "SELECT * FROM supersessions WHERE (old_kind=? AND old_id=?) OR (new_kind=? AND new_id=?) "
        "ORDER BY at ASC, id ASC",
        (kind, int(record_id), kind, int(record_id)),
    )
    return [_row(r) for r in await cur.fetchall()]


async def undo(sqlite, supersession_id: int) -> bool:
    """Mark a supersession undone (the old record is current again). Never deletes."""
    cur = await sqlite._db.execute(
        "UPDATE supersessions SET undone_at = ? WHERE id = ? AND undone_at IS NULL",
        (datetime.now(timezone.utc).isoformat(), int(supersession_id)),
    )
    await sqlite._db.commit()
    return bool(cur.rowcount)


async def memory_projects(sqlite, memory_ids: Iterable[int]) -> dict[int, Optional[str]]:
    """memory_id -> project of its session (None when the session has none)."""
    ids = [int(i) for i in memory_ids if i is not None]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    cur = await sqlite._db.execute(
        f"SELECT m.id, s.project FROM memories m LEFT JOIN sessions s ON s.id = m.session_id "
        f"WHERE m.id IN ({placeholders})",
        ids,
    )
    return {int(r["id"]): r["project"] for r in await cur.fetchall()}
