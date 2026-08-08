"""Morning briefing: the first turn of the day opens with what happened
overnight (V2_PLAN 5.2).

The continuity experience and the observability fix are the same feature:
nightly does real work — and real failures — while nobody watches, and a
system that only mentions the failures when asked has a trust problem.
So the first interaction of each day hands BlipShell a digest of the last
nightly run (including everything that went wrong), the user-model
revision, and headline counts, phrased as context it can weave into its
greeting naturally.

Deterministic assembly — no LLM call. The MODEL decides how (and how much)
to say; this module only decides what it knows.

One-shot per calendar day, gated by an app_metadata date stamp. The date is
LOCAL time: "morning" is the user's morning, and a UTC boundary would flip
the briefing at 6-7pm for a US user.
"""

import json
import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

LAST_SHOWN_KEY = "morning_briefing_last_date"

# Job stats worth a headline number, and how to phrase them.
_HEADLINES = [
    ("consolidate", "merged", "{} near-duplicate memories merged"),
    ("session_reflections", "reflected", "{} sessions reflected on"),
    ("entity_extraction", "extracted", "{} memories mined for entities"),
    ("prune", "pruned", "{} old low-value memories archived"),
    ("rebuild_digests", "rebuilt", "{} project digests rebuilt"),
]


async def build_briefing(sqlite) -> Optional[str]:
    """The day's briefing text, or None if today's was already shown.

    Marks today as shown WHEN a briefing is returned — callers inject it
    into the very next prompt, so returning is showing.
    """
    today = date.today().isoformat()
    try:
        if await sqlite.get_metadata(LAST_SHOWN_KEY) == today:
            return None
    except Exception as e:
        logger.debug("Briefing gate check failed: %s", e)
        return None

    sections = []

    # --- nightly run: the trust ledger ---------------------------------
    try:
        raw = await sqlite.get_metadata("nightly_report")
        report = json.loads(raw) if raw else None
    except (json.JSONDecodeError, TypeError):
        report = None

    if report:
        ts = report.get("timestamp")
        when = (
            datetime.fromtimestamp(ts).strftime("%A %I:%M %p").replace(" 0", " ")
            if ts else "recently"
        )
        errors = report.get("errors") or []
        warnings = report.get("warnings") or []
        if errors or warnings:
            problems = errors[:4] + warnings[:3]
            sections.append(
                f"Last nightly maintenance ({when}) had problems — the user "
                "should hear these, not have to ask:\n"
                + "\n".join(f"  - {p}" for p in problems)
                + (f"\n  - ...and {len(errors) + len(warnings) - len(problems)} more"
                   if len(errors) + len(warnings) > len(problems) else "")
            )
        else:
            sections.append(f"Nightly maintenance ran clean ({when}).")

        counts = []
        summary = report.get("summary") or {}
        for job, key, phrase in _HEADLINES:
            n = (summary.get(job) or {}).get(key)
            if n:
                counts.append(phrase.format(n))
        if counts:
            sections.append("Overnight: " + "; ".join(counts) + ".")

        um = (summary.get("update_user_model") or {})
        if um.get("revised"):
            sections.append(
                f"Your working model of the user was revised overnight "
                f"({um.get('lines', '?')} conclusions, "
                f"{um.get('evidence', '?')} new reflections digested)."
            )
    else:
        sections.append(
            "No nightly report found — maintenance may never have run. "
            "Worth mentioning /nightly if the user has a moment."
        )

    if not sections:
        return None

    try:
        await sqlite.set_metadata(LAST_SHOWN_KEY, today)
    except Exception as e:
        logger.warning("Could not stamp briefing date (may repeat): %s", e)

    body = "\n".join(sections)
    return (
        "[Morning briefing — first conversation of the day]\n"
        f"{body}\n"
        "Weave whatever matters into your greeting naturally and briefly — "
        "a sentence or two, not a report. Failures are worth surfacing; "
        "clean runs deserve at most a clause. Then get on with what the "
        "user actually asked."
    )
