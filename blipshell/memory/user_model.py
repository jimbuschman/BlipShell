"""The user model: one document of reasoned conclusions about the user.

Not facts — the entity graph owns facts ("Jim works at Xanatek"). This is
the layer above: preferences, values, working style, what frustrates them,
what they trust ("prefers thorough automated testing over quick wins;
frustrated by fix-then-still-broken cycles"). Honcho-inspired (V2_PLAN 5.1).

Three rules, all enforced or prompted here:

* REVISED, never appended. A nightly job rewrites the whole document from
  the current version plus recent session reflections. Append-only user
  models grow into contradiction soup; revision forces reconciliation.
* RETRACTABLE. The prompt explicitly instructs weakening or dropping any
  conclusion the new evidence contradicts — a user model that argues with
  its own evidence is worse than none.
* SIZE-CAPPED (~1.5K tokens, hard-enforced after generation). It lives in
  the core pool on every turn; an unbounded document would eat the budget
  that core memories share.

Privacy: revision routes through TaskType.REASONING — the LOCAL model —
deliberately. This document is the distilled personal layer, the exact
content /local exists to protect; synthesizing it through a cloud endpoint
would ship the most sensitive artifact in the system every night.
"""

import logging
from typing import Optional

from blipshell.llm.router import TaskType
from blipshell.memory.manager import estimate_tokens

logger = logging.getLogger(__name__)

DOC_KEY = "user_model"
UPDATED_KEY = "user_model_updated_at"

MAX_TOKENS = 1500
MAX_LINES = 25
# How many recent reflections one revision may digest. More than this and
# the prompt drowns the current model in evidence; the rest waits for the
# next night.
MAX_EVIDENCE = 12

# The model may honestly conclude there is nothing yet worth concluding.
EMPTY = "NOTHING"


def revision_prompt(current_doc: str, evidence: list[str]) -> tuple[str, str]:
    """(system, user) prompts for revising the user model."""
    system = (
        "You maintain a short working model of the user you assist: reasoned "
        "conclusions about their preferences, values, working style and "
        "frustrations — the layer ABOVE facts. Rules:\n"
        "- Output ONLY the revised document: one conclusion per line, "
        "formatted `- (high|medium|low) conclusion`, where the marker is "
        "your confidence.\n"
        "- REVISE the current document in light of the new evidence. Merge, "
        "reword, strengthen or weaken existing lines; do not simply append.\n"
        "- RETRACT: if evidence contradicts a conclusion, weaken its "
        "confidence or drop the line entirely. Never argue with evidence.\n"
        "- No biographical facts (names, employers, dates) — conclusions "
        "only. 'Prefers X', 'Trusts Y', 'Gets frustrated when Z'.\n"
        f"- At most {MAX_LINES} lines. Fewer, well-supported lines beat "
        "many speculative ones.\n"
        f"- If there is genuinely nothing to conclude yet, reply exactly: {EMPTY}"
    )
    parts = []
    if current_doc:
        parts.append(f"CURRENT MODEL:\n{current_doc}")
    else:
        parts.append("CURRENT MODEL: (none yet — this is the first revision)")
    parts.append(
        "NEW EVIDENCE (recent session reflections):\n"
        + "\n".join(f"- {e}" for e in evidence)
    )
    parts.append("Revised model:")
    return system, "\n\n".join(parts)


def enforce_cap(doc: str, max_tokens: int = MAX_TOKENS,
                max_lines: int = MAX_LINES) -> str:
    """Hard cap AFTER generation — the prompt asks, this enforces.

    Truncates at line boundaries so a conclusion is dropped whole, never
    cut mid-sentence into something the next revision would misread.
    """
    lines = [l for l in doc.strip().splitlines() if l.strip()]
    lines = lines[:max_lines]
    kept: list[str] = []
    total = 0
    for line in lines:
        t = estimate_tokens(line)
        if total + t > max_tokens:
            break
        kept.append(line)
        total += t
    return "\n".join(kept)


class UserModel:
    """Storage + revision for the user-model document."""

    def __init__(self, sqlite, router):
        self._sqlite = sqlite
        self._router = router

    async def get(self) -> Optional[str]:
        return await self._sqlite.get_metadata(DOC_KEY)

    async def updated_at(self) -> Optional[str]:
        return await self._sqlite.get_metadata(UPDATED_KEY)

    async def revise_from_reflections(self) -> dict:
        """One nightly revision step. Returns stats for the job report.

        The watermark advances to the created_at of the last reflection
        actually READ — never to "now", which would silently skip everything
        beyond MAX_EVIDENCE whenever a backlog exceeds one batch.
        """
        since = await self.updated_at()
        rows = await self._sqlite.get_reflection_texts_since(
            since, limit=MAX_EVIDENCE,
        )
        if not rows:
            return {"revised": False, "reason": "no new reflections"}
        evidence = [text for text, _ in rows]
        watermark = rows[-1][1]

        current = await self.get() or ""
        system, prompt = revision_prompt(current, evidence)
        response = await self._router.generate(
            TaskType.REASONING, prompt, system=system,
        )
        response = (response or "").strip()

        if not response or response.upper() == EMPTY:
            # An honest "nothing to conclude" still advances the watermark,
            # or the same evidence would be re-judged every night forever.
            await self._sqlite.set_metadata(UPDATED_KEY, watermark)
            return {"revised": False, "reason": "model concluded nothing",
                    "evidence": len(evidence)}

        doc = enforce_cap(response)
        if not doc:
            return {"revised": False, "reason": "empty after cap"}

        await self._sqlite.set_metadata(DOC_KEY, doc)
        await self._sqlite.set_metadata(UPDATED_KEY, watermark)
        logger.info(
            "User model revised: %d lines from %d reflections",
            len(doc.splitlines()), len(evidence),
        )
        return {"revised": True, "lines": len(doc.splitlines()),
                "evidence": len(evidence), "tokens": estimate_tokens(doc)}
