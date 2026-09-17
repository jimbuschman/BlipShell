"""`nightly --loop` must never call an undrained pool "done".

Night of 2026-09-16: `blipshell nightly --job batch_tag --local --loop` ran
overnight and announced "Nothing left to process - done." with 14,515
memories still in the pool. The loop keyed on `checked > 0`; a pass whose
first batch overran the budget (checked 0), a job that timed out or raised
(no counters at all), or a pass whose every batch errored (checked grows,
pool unchanged) all looked like "drained".
"""

from blipshell.core.nightly_loop import (
    MAX_STALLED_PASSES,
    Decision,
    decide,
    remaining_by_job,
)


def _tag_pass(**over):
    base = {"status": "ok", "elapsed_s": 100.0, "batches": 1, "checked": 10,
            "memories_tagged": 8, "tags_assigned": 20, "memories_marked_skip": 2,
            "failed": 0, "errors": 0, "stopped_early": True,
            "stop_reason": "time budget reached", "interrupted_batches": 0,
            "remaining_pool": 14515}
    base.update(over)
    return {"batch_tag": base}


class TestNeverDoneWithPoolRemaining:
    def test_first_batch_interrupted_is_a_retry_not_done(self):
        """The overnight shape: budget hit inside the first batch, checked 0,
        14,515 remain. The old rule said done."""
        d = decide(_tag_pass(checked=0, memories_tagged=0, memories_marked_skip=0,
                             interrupted_batches=1,
                             stop_reason="time budget reached during a batch; unfinished work remains retryable"))
        assert d.action == "retry"
        assert d.remaining == 14515
        assert "14515 remain" in d.reason and "time budget reached during a batch" in d.reason

    def test_job_timeout_is_a_retry_not_done(self):
        d = decide({"batch_tag": {"status": "timeout", "error": "Timed out after 300s", "elapsed_s": 300.0}})
        assert d.action == "retry" and "timeout" in d.reason

    def test_job_error_is_a_retry_not_done(self):
        d = decide({"batch_tag": {"status": "error", "error": "connection refused", "elapsed_s": 1.2}})
        assert d.action == "retry" and "connection refused" in d.reason

    def test_every_batch_erroring_does_not_count_checked_as_progress(self):
        """checked grows on model failures (the memories were fetched); nothing
        was tagged or marked, so the pool is exactly where it was."""
        first = decide(_tag_pass(checked=30, memories_tagged=0, memories_marked_skip=0, errors=3))
        assert first.action == "retry"
        second = decide(_tag_pass(checked=30, memories_tagged=0, memories_marked_skip=0, errors=3),
                        previous_remaining={"batch_tag": 14515})
        assert second.action == "retry" and "did not shrink" in second.reason


class TestProgressAndDrain:
    def test_first_pass_that_tagged_something_continues(self):
        d = decide(_tag_pass())
        assert d.action == "continue" and d.remaining == 14515

    def test_shrinking_pool_continues(self):
        d = decide(_tag_pass(remaining_pool=14400), previous_remaining={"batch_tag": 14515})
        assert d.action == "continue"

    def test_drained_pool_is_done(self):
        d = decide(_tag_pass(remaining_pool=0, stopped_early=False, stop_reason=None))
        assert d.is_done and d.remaining == 0

    def test_remaining_by_job_feeds_the_next_pass(self):
        assert remaining_by_job(_tag_pass()) == {"batch_tag": 14515}
        assert remaining_by_job({"consolidate": {"status": "ok", "checked": 5}}) == {}


class TestJobsWithoutAPool:
    """Consolidation and friends keep the old behaviour: work counters decide."""

    def test_checked_counts_as_work(self):
        assert decide({"consolidate": {"status": "ok", "checked": 40, "merged": 0}}).action == "continue"

    def test_no_work_is_done(self):
        assert decide({"consolidate": {"status": "ok", "checked": 0, "merged": 0}}).is_done


class TestBounds:
    def test_stall_limit_is_small_and_positive(self):
        assert 1 <= MAX_STALLED_PASSES <= 5

    def test_decision_is_frozen(self):
        d = Decision("done", "x")
        try:
            d.action = "retry"
        except AttributeError:
            return
        raise AssertionError("Decision must be immutable")
