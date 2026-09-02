"""Deterministic time-range extraction from search queries.

Temporal reasoning is every memory system's worst category (LoCoMo: best LLM
20.3% vs human 92.6%), and the best-attested fix is embarrassingly direct:
extract the time range the query names and prefer memories inside it —
LongMemEval's own ablation credits time-aware query handling with +6.8-11.3%
temporal retrieval, and Zep attributes its +38% temporal delta largely to
timestamp handling (docs/FIELD_SURVEY_2026_09.md #3.1). BlipShell timestamps
every memory and, until this module, used none of them at query time — "what
did I say last week" was served purely by embedding luck.

Deterministic regex rather than an LLM call, deliberately: this runs inside
`MemorySearch.search()` on the per-turn recall path, where an extra local LLM
round-trip is real latency; and a deterministic parser scores the same
transcript the same way forever, which keeps any future ablation re-scorable
(the same argument as memory/themes.py). The cost of that choice is recall —
exotic phrasings ("the week my brother visited") are not parsed — and that is
the right trade: a missed range degrades to exactly today's behavior.

Day boundaries are computed in a LOCAL timezone ("yesterday" means the user's
yesterday, not UTC's) and returned as aware-UTC bounds, because memory
timestamps are stored UTC.

ASCII-safe (cp1252 console rule).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
    "saturday": 5, "sunday": 6,
}
_UNIT_SECONDS = {
    "minute": 60.0, "hour": 3600.0, "day": 86400.0, "week": 604800.0,
}
# "recently" / "lately" — vague on purpose, so the window is too: two weeks.
RECENT_DAYS = 14


@dataclass(frozen=True)
class TimeRange:
    """[start, end) in aware UTC, plus the phrase that produced it."""

    start: datetime
    end: datetime
    phrase: str


def in_time_range(ts: Optional[datetime], tr: TimeRange) -> bool:
    """Is a memory timestamp inside the range? Naive timestamps are UTC
    (SQLite CURRENT_TIMESTAMP); a missing timestamp is never in range."""
    if ts is None:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return tr.start <= ts < tr.end


def _day_bounds(local_dt: datetime) -> tuple[datetime, datetime]:
    start = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _week_bounds(local_dt: datetime) -> tuple[datetime, datetime]:
    start = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    start -= timedelta(days=start.weekday())
    return start, start + timedelta(days=7)


def _month_bounds(year: int, month: int, tz: tzinfo) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=tz)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=tz)
    else:
        end = datetime(year, month + 1, 1, tzinfo=tz)
    return start, end


def _count(word: str) -> int:
    if word in ("a", "an", "one"):
        return 1
    words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "couple": 2, "few": 3}
    if word in words:
        return words[word]
    return int(word)

_N = r"(\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)"


def parse_time_range(
    query: str, now: Optional[datetime] = None, tz: Optional[tzinfo] = None,
) -> Optional[TimeRange]:
    """Extract the first recognized time expression, or None.

    `now` must be timezone-aware when given (default: real now, UTC).
    `tz` is the timezone whose calendar days the user means (default: the
    machine's local zone). Pass both explicitly in tests.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if tz is None:
        tz = now.astimezone().tzinfo
    local = now.astimezone(tz)
    q = query.lower()

    def rng(start: datetime, end: datetime, phrase: str) -> TimeRange:
        return TimeRange(
            start=start.astimezone(timezone.utc),
            end=end.astimezone(timezone.utc),
            phrase=phrase,
        )

    # Ordered: more specific phrasings first, so "day before yesterday" is
    # not swallowed by "yesterday", "last night" not by "last week"'s family.
    m = re.search(r"\bday before yesterday\b", q)
    if m:
        s, e = _day_bounds(local - timedelta(days=2))
        return rng(s, e, m.group(0))

    m = re.search(r"\blast night\b", q)
    if m:
        y_start, _ = _day_bounds(local - timedelta(days=1))
        t_start, _ = _day_bounds(local)
        return rng(y_start + timedelta(hours=18), t_start + timedelta(hours=6),
                   m.group(0))

    m = re.search(r"\byesterday\b", q)
    if m:
        s, e = _day_bounds(local - timedelta(days=1))
        return rng(s, e, m.group(0))

    m = re.search(r"\b(this morning|this afternoon|this evening|tonight|today|earlier today)\b", q)
    if m:
        s, e = _day_bounds(local)
        return rng(s, e, m.group(0))

    # "in/over/during the last/past N units" and bare "last/past N units"
    m = re.search(
        rf"\b(?:in |over |during |within )?the (?:last|past) {_N} "
        rf"(minute|hour|day|week|month)s?\b", q,
    ) or re.search(rf"\b(?:last|past) {_N} (minute|hour|day|week|month)s?\b", q)
    if m:
        n, unit = _count(m.group(1)), m.group(2)
        if unit == "month":
            delta = timedelta(days=30 * n)
        else:
            delta = timedelta(seconds=_UNIT_SECONDS[unit] * n)
        return rng(local - delta, local, m.group(0))

    # "N units ago" — a window, not an instant: the containing day for
    # days/weeks/months (weeks widen to the week, months to the month),
    # +/- one unit for hours/minutes.
    m = re.search(rf"\b{_N} (minute|hour|day|week|month)s? ago\b", q)
    if m:
        n, unit = _count(m.group(1)), m.group(2)
        if unit in ("minute", "hour"):
            width = timedelta(seconds=_UNIT_SECONDS[unit])  # +/- one unit
            point = local - timedelta(seconds=_UNIT_SECONDS[unit] * n)
            return rng(point - width, min(point + width, local), m.group(0))
        if unit == "day":
            s, e = _day_bounds(local - timedelta(days=n))
        elif unit == "week":
            s, e = _week_bounds(local - timedelta(weeks=n))
        else:  # month
            month_index = (local.year * 12 + (local.month - 1)) - n
            s, e = _month_bounds(month_index // 12, month_index % 12 + 1, tz)
        return rng(s, e, m.group(0))

    m = re.search(r"\b(last|this) (week|month|year)\b", q)
    if m:
        which, unit = m.group(1), m.group(2)
        if unit == "week":
            s, e = _week_bounds(local)
            if which == "last":
                s, e = s - timedelta(days=7), s
        elif unit == "month":
            if which == "last":
                month_index = local.year * 12 + (local.month - 1) - 1
                s, e = _month_bounds(month_index // 12, month_index % 12 + 1, tz)
            else:
                s, e = _month_bounds(local.year, local.month, tz)
        else:  # year
            year = local.year - 1 if which == "last" else local.year
            s = datetime(year, 1, 1, tzinfo=tz)
            e = datetime(year + 1, 1, 1, tzinfo=tz)
        return rng(s, e, m.group(0))

    # "on monday" / "last tuesday" — the most recent such day before today.
    m = re.search(
        r"\b(?:on|last) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q,
    )
    if m:
        target = WEEKDAYS[m.group(1)]
        back = (local.weekday() - target - 1) % 7 + 1
        s, e = _day_bounds(local - timedelta(days=back))
        return rng(s, e, m.group(0))

    # "in january" / "in january 2025" — most recent occurrence not in the future.
    m = re.search(
        r"\b(?:in|during|back in) (january|february|march|april|may|june|july"
        r"|august|september|october|november|december)(?: (\d{4}))?\b", q,
    )
    if m:
        month = MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else (
            local.year if month <= local.month else local.year - 1
        )
        s, e = _month_bounds(year, month, tz)
        return rng(s, e, m.group(0))

    m = re.search(r"\b(?:in|during|back in) ((?:19|20)\d{2})\b", q)
    if m:
        year = int(m.group(1))
        s = datetime(year, 1, 1, tzinfo=tz)
        e = datetime(year + 1, 1, 1, tzinfo=tz)
        return rng(s, e, m.group(0))

    m = re.search(r"\b(recently|lately)\b", q)
    if m:
        return rng(local - timedelta(days=RECENT_DAYS), local, m.group(0))

    return None
