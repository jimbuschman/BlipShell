"""Correction attribution - Phase 1, RECORD ONLY (V3 D2a, approved 2026-09-09).

Two deterministic records and one judgment, and the judgment changes nothing.

- `lesson_uses`: which lessons were in the request on which turn (from the
  always-on Lessons pool or a per-query Recall hit). Written every turn.
- `corrections`: one row per correction the existing two-stage detector
  accepts (regex candidate -> local YES/NO judge, core/agent_chat.py), with
  the user's text, the previous assistant excerpt, and the lesson ids that
  were present on the turn being corrected. `attribution` starts as
  `unattributed`.
- The attribution judge: one LOCAL call per correction, in the background,
  asked which of the present lessons (if any) explains the corrected
  behaviour and how - `lesson_wrong` | `lesson_ignored` |
  `lesson_misapplied` | `unrelated`. Its answer is parsed strictly; anything
  ambiguous, or below CONFIDENCE_FLOOR, is stored as `unattributed`. The raw
  reply is kept for the evaluation.

**Authority: none.** Nothing here reads or writes lesson importance, status
or selection, and nothing else reads these tables to do so. Phase 2 (one
`lesson_wrong` -> one CONTRADICTS vote in the revote) is gated on a
hand-labelled evaluation set (>= 10-15 genuine lesson_wrong positives,
agreement >= 0.8, false-positive rate <= 0.1, repeat spread < 0.1) and the
user's explicit approval. `scripts/attribution_readout.py` shows the rows,
records human labels, and computes the agreement.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

ATTRIBUTIONS = ("lesson_wrong", "lesson_ignored", "lesson_misapplied", "unrelated", "unattributed")
CONFIDENCE_FLOOR = 0.7
MAX_LESSONS_JUDGED = 5

JUDGE_SYSTEM = (
    "You attribute ONE user correction to at most one of the standing LESSONS "
    "that were in the assistant's context when it produced the reply being "
    "corrected. Decide which case applies:\n"
    "- lesson_wrong: a listed lesson's guidance PRODUCED the behaviour the user corrected.\n"
    "- lesson_ignored: a listed lesson was appropriate and the assistant did NOT follow it.\n"
    "- lesson_misapplied: the assistant followed a listed lesson in a situation it did not fit.\n"
    "- unrelated: the correction concerns something no listed lesson addresses.\n"
    "Reply with ONLY a JSON object: {\"lesson_id\": <id or null>, "
    "\"attribution\": \"lesson_wrong|lesson_ignored|lesson_misapplied|unrelated\", "
    "\"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"}. "
    "lesson_id must be one of the listed ids, or null for unrelated. When unsure, say unrelated with low confidence."
)

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def judge_prompt(correction_text: str, prev_assistant: str, lessons: list[tuple[int, str]]) -> str:
    parts = []
    if prev_assistant:
        parts.append(f"Assistant's reply being corrected (excerpt):\n{prev_assistant}")
    parts.append(f"User's correction:\n{correction_text}")
    if lessons:
        parts.append("Lessons present in the assistant's context:\n" + "\n".join(
            f"  [{lid}] {text}" for lid, text in lessons[:MAX_LESSONS_JUDGED]))
    else:
        parts.append("Lessons present in the assistant's context: none")
    parts.append("Attribute the correction (JSON only):")
    return "\n\n".join(parts)


@dataclass
class Verdict:
    attribution: str
    lesson_id: Optional[int]
    confidence: float
    reason: str
    raw: str


def parse_verdict(text: Optional[str], allowed_lesson_ids: Iterable[int]) -> Verdict:
    """Strict parse. Anything ambiguous -> unattributed (raw kept)."""
    raw = text or ""
    allowed = {int(i) for i in allowed_lesson_ids}
    unatt = Verdict("unattributed", None, 0.0, "", raw)
    if not raw.strip():
        return unatt
    try:
        obj = json.loads(_FENCE.sub("", raw.strip()).strip())
    except (ValueError, TypeError):
        return unatt
    if not isinstance(obj, dict):
        return unatt
    attribution = obj.get("attribution")
    if not isinstance(attribution, str) or attribution not in ATTRIBUTIONS or attribution == "unattributed":
        return unatt
    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        return unatt
    lid = obj.get("lesson_id")
    if attribution == "unrelated":
        lid = None
    else:
        if isinstance(lid, bool) or not isinstance(lid, int) or lid not in allowed:
            return unatt  # attributed to a lesson that was not present
    reason = obj.get("reason") if isinstance(obj.get("reason"), str) else ""
    v = Verdict(attribution, lid, float(conf), reason[:300], raw)
    if v.confidence < CONFIDENCE_FLOOR:
        return Verdict("unattributed", None, v.confidence, v.reason, raw)
    return v


# ---------------------------------------------------------------- records

async def record_lesson_uses(sqlite, *, session_id: Optional[int], turn_index: int,
                             uses: Iterable[tuple[int, str]]) -> int:
    """uses: (lesson_id, selected_by) with selected_by in {'pool', 'recall'}."""
    rows = [(int(lid), session_id, int(turn_index), by, datetime.now(timezone.utc).isoformat())
            for lid, by in uses if lid]
    if not rows:
        return 0
    await sqlite._db.executemany(
        "INSERT INTO lesson_uses (lesson_id, session_id, turn_index, selected_by, at) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    await sqlite._db.commit()
    return len(rows)


async def record_correction(sqlite, *, session_id: Optional[int], turn_index: int, text: str,
                            prev_assistant: str, lessons_present: Iterable[int]) -> int:
    cur = await sqlite._db.execute(
        "INSERT INTO corrections (session_id, turn_index, text, prev_assistant_excerpt, lessons_present, "
        "attribution, at) VALUES (?, ?, ?, ?, ?, 'unattributed', ?)",
        (session_id, int(turn_index), text[:2000], (prev_assistant or "")[:500],
         json.dumps([int(i) for i in lessons_present]), datetime.now(timezone.utc).isoformat()),
    )
    await sqlite._db.commit()
    return int(cur.lastrowid)


async def get_correction(sqlite, correction_id: int) -> Optional[dict]:
    cur = await sqlite._db.execute("SELECT * FROM corrections WHERE id = ?", (int(correction_id),))
    r = await cur.fetchone()
    return dict(r) if r else None


async def attribute_correction(sqlite, router, correction_id: int) -> Verdict:
    """Run the judge on one correction and STORE the result. Record only:
    no lesson is read for writing, none is written. Fails to `unattributed`."""
    from blipshell.llm.router import TaskType

    row = await get_correction(sqlite, correction_id)
    if row is None:
        return Verdict("unattributed", None, 0.0, "no such correction", "")
    try:
        present = [int(i) for i in json.loads(row.get("lessons_present") or "[]")]
    except (ValueError, TypeError):
        present = []
    lessons: list[tuple[int, str]] = []
    for lid in present[:MAX_LESSONS_JUDGED]:
        try:
            lesson = await sqlite.get_lesson(lid)
        except Exception:
            lesson = None
        if lesson is not None:
            lessons.append((lid, (lesson.content or "")[:300]))

    if not lessons:
        verdict = Verdict("unrelated", None, 1.0, "no lessons were present", "")
    else:
        try:
            raw = await router.generate(
                TaskType.REASONING, judge_prompt(row["text"], row.get("prev_assistant_excerpt") or "", lessons),
                system=JUDGE_SYSTEM, think=False,
            )
        except Exception as e:
            logger.warning("Attribution judge failed for correction %d (recorded unattributed): %s",
                           correction_id, e)
            raw = ""
        verdict = parse_verdict(raw, [lid for lid, _ in lessons])

    await sqlite._db.execute(
        "UPDATE corrections SET attribution = ?, lesson_id = ?, confidence = ?, judged_by = ?, "
        "judged_at = ?, judge_raw = ? WHERE id = ?",
        (verdict.attribution, verdict.lesson_id, verdict.confidence, "local_judge",
         datetime.now(timezone.utc).isoformat(), (verdict.raw or "")[:1000], int(correction_id)),
    )
    await sqlite._db.commit()
    logger.info("Correction %d attributed: %s (lesson %s, conf %.2f) - record only",
                correction_id, verdict.attribution, verdict.lesson_id, verdict.confidence)
    return verdict
