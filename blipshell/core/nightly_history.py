"""Bounded run history, including checkpoints for interrupted nightly runs."""

import json
from datetime import UTC, datetime
from itertools import pairwise

HISTORY_KEY = 'nightly_run_history'
HISTORY_LIMIT = 120


def decode_history(raw) -> list[dict]:
    if not raw:
        return []
    value = json.loads(raw)
    if not isinstance(value, list):
        raise TypeError('Nightly history must be a list')
    return value


async def save_run(store, record: dict) -> None:
    await store.save_nightly_run(HISTORY_KEY, record, HISTORY_LIMIT)


def growing_pool_warning(history: list[dict]) -> str | None:
    """Compare the last observation on each UTC day, not repeated --loop passes.

    Require consecutive observed days: missing nights cannot prove a trend.
    These are stock measurements, never claimed to be write-time arrival rates.
    """
    days = {}
    for run in sorted(history, key=lambda r: r.get('completed_at') or 0):
        if run.get('status') != 'completed':
            continue
        snapshot = run.get('tagging', {}).get('after')
        if not snapshot:
            continue
        day = datetime.fromtimestamp(run['completed_at'], UTC).date()
        days[day] = snapshot['pending']
    recent = sorted(days)[-4:]
    if len(recent) != 4:
        return None
    if all((b - a).days == 1 and days[b] > days[a]
           for a, b in pairwise(recent)):
        return (f"Tagging pool grew on four consecutive observed UTC days: "
                f"{days[recent[0]]} -> {days[recent[-1]]}. "
                "Compare tagging throughput with new work before changing budgets.")
    return None
