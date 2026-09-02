"""Stage-2 correction judge (guardrails.confirm_correction).

Regex is the prefilter; a LOCAL LLM yes/no renders the verdict before an
anti-pattern lesson is minted. Reading the live store showed why: strong
patterns fire inside recounted dialogue, jokes, and the user's own habits —
no pattern set separates those. FAIL-CLOSED: any judge error means no lesson.
"""

from unittest.mock import AsyncMock, MagicMock

from blipshell.core.guardrails import (
    CORRECTION_JUDGE_SYSTEM,
    confirm_correction,
    correction_judge_prompt,
)
from blipshell.llm.router import TaskType


def _router(reply: str | None = None, error: Exception | None = None):
    router = MagicMock()
    if error:
        router.generate = AsyncMock(side_effect=error)
    else:
        router.generate = AsyncMock(return_value=reply)
    return router


async def test_yes_confirms():
    router = _router("YES")
    assert await confirm_correction(router, "you didn't run the tests") is True


async def test_no_rejects():
    router = _router("NO")
    assert await confirm_correction(
        router, 'i asked "are you ok?" she said "i\'m not great"') is False


async def test_verbose_yes_still_counts():
    assert await confirm_correction(_router("Yes — the user is correcting."),
                                    "you keep doing that") is True


async def test_garbage_reply_rejects():
    """An unparseable verdict is not a YES — fail toward minting nothing."""
    assert await confirm_correction(_router("The user seems frustrated..."),
                                    "you didn't") is False


async def test_judge_error_fails_closed():
    assert await confirm_correction(_router(error=RuntimeError("ollama down")),
                                    "you didn't") is False


async def test_routes_through_reasoning_local():
    """The candidates are personal messages — they must route to the LOCAL
    reasoning key, never a cloud task type."""
    router = _router("NO")
    await confirm_correction(router, "message", "prev")
    task_type = router.generate.await_args.args[0]
    assert task_type == TaskType.REASONING


def test_prompt_carries_both_sides():
    prompt = correction_judge_prompt("the user text", "the assistant text")
    assert "the user text" in prompt
    assert "the assistant text" in prompt
    assert "YES or NO" in prompt
    # System prompt teaches the observed false classes.
    for cue in ("recounting", "quoting", "joking", "own habits"):
        assert cue in CORRECTION_JUDGE_SYSTEM
