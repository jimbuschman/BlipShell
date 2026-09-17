"""When does `blipshell nightly --job X --loop` stop, and what does it say?

The loop used to stop the moment a pass reported no work counter above zero
and announce "Nothing left to process - done." That is true when the pool is
drained. It is FALSE when the pass made no progress for another reason: the
first batch overran the per-pass time budget (`checked` stays 0 - see
`test_first_batch_is_also_bounded`), the job timed out or raised (its stats
carry only `status` and `error`), or every batch errored (`checked` grows,
nothing is tagged, the pool does not shrink). The night of 2026-09-16 the
loop said "done" with 14,515 memories still pending.

The rule here is a pure function so it can be tested; cli.py only applies it.
A pass is judged by three questions, in order:

1. Did the job fail (status timeout/error)?           -> retry, not done
2. Does the job report its pool (`remaining_pool`)?
     pool == 0                                        -> done (drained)
     pool shrank since the last pass / first pass
       mutated something                              -> continue
     pool did not shrink                              -> retry, not done
3. No pool reported: any work counter > 0?            -> continue, else done

"retry" is bounded: after MAX_STALLED_PASSES consecutive retries the caller
stops, says how much remains and why, and exits nonzero. The one thing this
must never do is call an undrained pool "done".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Consecutive no-progress passes before the loop gives up (nonzero exit).
MAX_STALLED_PASSES = 3
# Pause before retrying a no-progress pass, so a model outage does not spin.
RETRY_BACKOFF_SECONDS = 60.0

# Any of these above zero means the pass did something, for jobs that do
# not report a pool. `checked` counts examined-and-marked items: consolidation
# examines a batch and usually merges nothing, and its pool shrinks anyway.
WORK_KEYS = ("resummarized", "scored", "processed", "merged", "deleted_junk",
             "deleted_dupes", "pruned", "rebuilt", "checked")
# For a job that reports a pool, only these prove the pool actually moved on
# the first pass (there is no previous count to compare against yet).
# `checked` is deliberately absent: batch_tag counts a batch as checked even
# when its model call failed and nothing was tagged or marked.
POOL_MUTATION_KEYS = ("memories_tagged", "memories_marked_skip", "merged",
                      "pruned", "processed")


@dataclass(frozen=True)
class Decision:
    action: str          # "continue" | "done" | "retry"
    reason: str
    remaining: Optional[int] = None   # total reported pool after this pass, if any

    @property
    def is_done(self) -> bool:
        return self.action == "done"


def decide(job_stats: dict, previous_remaining: Optional[dict] = None) -> Decision:
    """Judge one --loop pass.

    `job_stats` is `result["jobs"]` from NightlyRunner.run; `previous_remaining`
    maps job name -> `remaining_pool` from the previous pass (None on the first).
    """
    for name, stats in job_stats.items():
        status = stats.get("status")
        if status in ("timeout", "error"):
            return Decision("retry", f"{name} {status}: {stats.get('error') or 'no detail'}")

    pools = {name: stats["remaining_pool"] for name, stats in job_stats.items()
             if isinstance(stats.get("remaining_pool"), int)}
    if pools:
        total = sum(pools.values())
        if total == 0:
            return Decision("done", "pool drained", remaining=0)
        if previous_remaining is None:
            mutated = any(stats.get(k, 0) > 0 for stats in job_stats.values()
                          for k in POOL_MUTATION_KEYS)
            if mutated:
                return Decision("continue", f"{total} remain", remaining=total)
            reasons = [stats.get("stop_reason") or stats.get("error") or "no progress"
                       for stats in job_stats.values() if stats.get("remaining_pool")]
            return Decision("retry", f"{total} remain and this pass changed nothing "
                            f"({'; '.join(reasons)})", remaining=total)
        shrank = any(pools[name] < previous_remaining.get(name, pools[name] + 1)
                     for name in pools)
        if shrank:
            return Decision("continue", f"{total} remain", remaining=total)
        reasons = [stats.get("stop_reason") or stats.get("error") or "no progress"
                   for stats in job_stats.values() if stats.get("remaining_pool")]
        return Decision("retry", f"{total} remain and the pool did not shrink "
                        f"({'; '.join(reasons)})", remaining=total)

    worked = any(stats.get(k, 0) > 0 for stats in job_stats.values() for k in WORK_KEYS)
    if worked:
        return Decision("continue", "work done this pass")
    return Decision("done", "nothing left to process")


def remaining_by_job(job_stats: dict) -> dict:
    """`remaining_pool` per job, for the next pass's comparison."""
    return {name: stats["remaining_pool"] for name, stats in job_stats.items()
            if isinstance(stats.get("remaining_pool"), int)}
