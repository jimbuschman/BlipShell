"""Dedup decision grammar + archive provenance (V3_PLAN Stage A1).

The write-time dedup step asks a model what to do with a new memory given up
to three similar existing ones: ADD / UPDATE n / DELETE n / NONE. UPDATE and
DELETE ARCHIVE an existing memory and drop its vector, so the parse of that
one-line reply decides whether a fact stays findable.

Until 2026-09-08 the parser scanned the reply for action SUBSTRINGS (in the
order NONE, UPDATE, DELETE, ADD) and defaulted a missing index to item 0.
"Do not DELETE anything; ADD this as distinct." therefore parsed as DELETE 0
and archived candidate #1 - the model being careful was the trigger. The
external review reproduced it; this module replaces it.

Rules:

- The reply must BE a verdict, not contain one. We look at the first and last
  non-empty line (models put the verdict at either end), reduce each to its
  last sentence segment, and accept only a bare `ADD`, `NONE`, `UPDATE <n>`
  or `DELETE <n>`. Two different valid verdicts in one reply is a conflict.
- UPDATE/DELETE require an explicit 1-based index. There is no default
  target. Range is checked by the caller, which knows the candidate count.
- Anything else is RETRY. The caller re-asks once with a sharper reminder,
  then keeps the new memory and archives nothing. Ambiguity never destroys a
  record.
- The structured path (`memory.dedup.structured_output`) asks for a JSON
  object constrained by MEMORY_ACTION_SCHEMA and validates it here. It is an
  experiment until its schema-validity rate is measured on the local model
  with thinking on (benchmark job `dedup_structured`); the text path stays
  the default.
- Every archive stamps the archived row's metadata with who/why, and
  `unarchive_memory` reverses one while keeping that history. Both live here
  so the CLI and the processor agree on the record shape.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

RETRY = "RETRY"
ACTIONS = ("ADD", "NONE", "UPDATE", "DELETE")
INDEXED_ACTIONS = ("UPDATE", "DELETE")

# The verdict grammar. A label prefix ("Action:", "Decision:") is tolerated;
# the index may carry a '#'. Nothing else may appear in the segment.
_VERDICT_RE = re.compile(
    r"^(?:(?:action|decision|verdict|answer)\s*[:=-]\s*)?"
    r"(?P<action>ADD|NONE|UPDATE|DELETE)"
    r"(?:\s+#?(?P<index>-?\d+))?$",
    re.IGNORECASE,
)
_MARKUP_RE = re.compile(r"[*`_\"'\[\]()]")
_TRAILING_PUNCT_RE = re.compile(r"[.!,;:]+$")
_SEGMENT_SPLIT_RE = re.compile(r"[.;:!]\s+|[.;:!]$")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

# JSON schema for the structured path. Passed to Ollama as `format` (the
# OpenAI-compat client drops it; the prompt still asks for the object and
# validation below catches a free-text reply).
MEMORY_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "target_index": {"type": ["integer", "null"],
                         "description": "1-based number of the existing memory; required for UPDATE/DELETE"},
        "reason": {"type": "string"},
    },
    "required": ["action", "target_index"],
    "additionalProperties": False,
}


def _reduce_line(line: str) -> str:
    """Strip markup and trailing punctuation from one candidate line."""
    line = _MARKUP_RE.sub("", line).strip()
    return _TRAILING_PUNCT_RE.sub("", line).strip()


def _match(segment: str) -> Optional[tuple[str, Optional[int]]]:
    m = _VERDICT_RE.match(segment.strip())
    if not m:
        return None
    action = m.group("action").upper()
    idx = m.group("index")
    if action in INDEXED_ACTIONS:
        if idx is None:
            return None  # no default target, ever
        n = int(idx)
        if n < 1:
            return None  # 1-based
        return action, n - 1
    if idx is not None:
        return None  # "ADD 2" is not a thing
    return action, None


def _verdict_from_line(line: str) -> Optional[tuple[str, Optional[int]]]:
    reduced = _reduce_line(line)
    if not reduced:
        return None
    hit = _match(reduced)
    if hit:
        return hit
    # "Explanation first. UPDATE 2" - the verdict is the last sentence segment.
    segments = [s for s in _SEGMENT_SPLIT_RE.split(reduced) if s and s.strip()]
    if len(segments) > 1:
        return _match(_reduce_line(segments[-1]))
    return None


def parse_action_text(text: Optional[str]) -> tuple[str, Optional[int]]:
    """Parse a free-text dedup reply into (action, 0-based index | None).

    Returns (RETRY, None) for anything that is not unambiguously one verdict.
    """
    if not text or not text.strip():
        return RETRY, None
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return RETRY, None
    candidates = [lines[0]] if len(lines) == 1 else [lines[0], lines[-1]]
    verdicts = {v for v in (_verdict_from_line(ln) for ln in candidates) if v is not None}
    if len(verdicts) != 1:
        return RETRY, None  # none found, or first and last line disagree
    return verdicts.pop()


def parse_action_json(text: Optional[str]) -> tuple[str, Optional[int]]:
    """Parse a structured dedup reply; (RETRY, None) unless it validates."""
    if not text or not text.strip():
        return RETRY, None
    raw = _FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return RETRY, None
    if not isinstance(obj, dict):
        return RETRY, None
    action = obj.get("action")
    if not isinstance(action, str) or action.upper() not in ACTIONS:
        return RETRY, None
    action = action.upper()
    idx = obj.get("target_index")
    if action in INDEXED_ACTIONS:
        # bool is an int subclass; `true` must not become index 1
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 1:
            return RETRY, None
        return action, idx - 1
    return action, None


def parse_action(text: Optional[str], *, structured: bool) -> tuple[str, Optional[int]]:
    return parse_action_json(text) if structured else parse_action_text(text)


def in_range(action: str, idx: Optional[int], n_candidates: int) -> bool:
    """A parsed verdict is applicable only if its target exists."""
    if action in INDEXED_ACTIONS:
        return idx is not None and 0 <= idx < n_candidates
    return action in ACTIONS


# ---------------------------------------------------------------------------
# Provenance on the archived row
# ---------------------------------------------------------------------------

def _load_meta(metadata_json: Optional[str]) -> dict:
    if not metadata_json:
        return {}
    try:
        obj = json.loads(metadata_json)
    except (ValueError, TypeError):
        return {"_unparsed_metadata": metadata_json}
    return obj if isinstance(obj, dict) else {"_prior_metadata": obj}


async def merge_metadata(sqlite, memory_id: int, patch: dict) -> dict:
    """Shallow-merge `patch` into a memory's metadata_json and persist it."""
    mem = await sqlite.get_memory(memory_id)
    meta = _load_meta(mem.metadata_json if mem else None)
    meta.update(patch)
    await sqlite.update_memory(memory_id, metadata_json=json.dumps(meta))
    return meta


def archive_record(*, action: str, by_memory_id: int, candidates: list[int],
                   reply: str, structured: bool) -> dict:
    return {
        "action": action,
        "by": by_memory_id,
        "candidates": list(candidates),
        "reply": (reply or "")[:300],
        "structured": structured,
        "at": datetime.now(timezone.utc).isoformat(),
    }


def undecided_record(*, candidates: list[int], reply: str, structured: bool) -> dict:
    return {
        "candidates": list(candidates),
        "reply": (reply or "")[:300],
        "structured": structured,
        "at": datetime.now(timezone.utc).isoformat(),
    }


async def unarchive_memory(sqlite, vectors, memory_id: int, *, dry_run: bool = False) -> dict:
    """Reverse an archive: un-flag the row, re-embed it, keep the history.

    Returns a report dict; `restored` is True only when the row was changed.
    Works for any archived memory, but its `dedup` field explains the ones
    this module's caller archived.
    """
    mem = await sqlite.get_memory(memory_id)
    if mem is None:
        return {"id": memory_id, "restored": False, "reason": "not found"}
    meta = _load_meta(mem.metadata_json)
    dedup = meta.get("dedup")
    if not mem.is_archived:
        return {"id": memory_id, "restored": False, "reason": "not archived", "dedup": dedup}
    report = {
        "id": memory_id,
        "restored": False,
        "dedup": dedup,
        "summary": (mem.summary or mem.content or "")[:120],
    }
    if dry_run:
        report["reason"] = "dry-run"
        return report

    await sqlite.update_memory(memory_id, is_archived=False)
    if isinstance(dedup, dict):
        dedup["unarchived_at"] = datetime.now(timezone.utc).isoformat()
        meta["dedup"] = dedup
        await sqlite.update_memory(memory_id, metadata_json=json.dumps(meta))
    # Same embed choice as the pipeline: raw content, summary as fallback.
    embed_text = mem.content or mem.summary or ""
    embed_meta = {"session_id": str(mem.session_id), "role": mem.role}
    try:
        vectors.add_memory(memory_id, embed_text, embed_meta)
        report["reembedded"] = True
    except Exception as e:  # the row is restored either way; say so
        logger.warning("Unarchived memory %d but re-embed failed: %s", memory_id, e)
        report["reembedded"] = False
        report["embed_error"] = str(e)
    report["restored"] = True
    return report
