"""Tests for look-before-review: the review-request classifier and the
deterministic completion gate that refuses task_complete when a review was
'completed' without the model reading or searching any code.

These are pure/deterministic — no LLM, runnable on the dev box. The behavioral
effect (does the model actually read before reviewing) is validated separately
on the Ollama PC.
"""

import pytest

from blipshell.core.intent_detection import detect_review_intent


# ---------------------------------------------------------------------------
# Classifier — positives
# ---------------------------------------------------------------------------

REVIEW_POSITIVES = [
    "what could be improved in cli.py?",
    "what do you think could be improved?",
    # The actual incident phrasing that started this whole thread:
    "tell me what you thought could be improved about the code",
    "review this code",
    "can you review the executor module for me",
    "any bugs in cli.py?",
    "what's wrong with this function",
    "whats wrong with my approach here",
    "critique my approach here please",
    "what would you improve about this module?",
    "look for bugs in the auth flow",
    "any code smells in here?",
    "are there any issues with the search pipeline?",
    "give me feedback on this design",
    "what are the refactor opportunities in the memory layer?",
    "is there technical debt I should address?",
    "audit the code in agent_chat.py",
    # build as a NOUN must not be vetoed by the 'build' action suppressor:
    "what could be improved in the build system?",
]


@pytest.mark.parametrize("msg", REVIEW_POSITIVES)
def test_review_positives(msg):
    assert detect_review_intent(msg) is True, msg


def test_weak_signals_need_length_and_question():
    # Two weak hits on a long message → review.
    assert detect_review_intent(
        "I'd love your thoughts on this design — is this approach correct?"
    ) is True


# ---------------------------------------------------------------------------
# Classifier — negatives
# ---------------------------------------------------------------------------

REVIEW_NEGATIVES = [
    "fix the bug in cli.py",
    "add a new function to parse the config",
    "refactor this module",          # bare action verb = do it, not review
    "implement caching for the API layer",
    "what is the status of the build",
    "show me the config",
    "run the tests",
    "list the projects",
    "how does the parser work?",     # explain/research, not review
    "create a new endpoint for uploads",
    "hi",                            # too short
    "thanks!",                       # too short
    "can you fix the failing test",
]


@pytest.mark.parametrize("msg", REVIEW_NEGATIVES)
def test_review_negatives(msg):
    assert detect_review_intent(msg) is False, msg


def test_empty_and_none():
    assert detect_review_intent("") is False
    assert detect_review_intent("   ") is False


# ---------------------------------------------------------------------------
# Deterministic completion gate (GuardrailsEngine.check_review_grounding)
# ---------------------------------------------------------------------------

from blipshell.core.guardrails import GuardrailsEngine
from blipshell.models.config import GuardrailsConfig


def _engine(original_request: str, review_grounding: bool = True) -> GuardrailsEngine:
    cfg = GuardrailsConfig(review_grounding=review_grounding)
    eng = GuardrailsEngine(cfg, router=None)  # router unused by the gate
    eng.original_request = original_request
    return eng


def test_gate_blocks_ungrounded_review():
    eng = _engine("what could be improved in cli.py?")
    feedback = eng.check_review_grounding(tool_call_names=[])
    assert feedback is not None
    assert "REVIEW NOT GROUNDED" in feedback


def test_gate_allows_when_read_file_used():
    eng = _engine("what could be improved in cli.py?")
    assert eng.check_review_grounding(["read_file"]) is None


def test_gate_allows_when_grep_used():
    eng = _engine("any bugs in the search pipeline?")
    assert eng.check_review_grounding(["grep_files"]) is None


def test_gate_ignores_non_review_request():
    eng = _engine("fix the bug in cli.py")
    assert eng.check_review_grounding([]) is None


def test_gate_fires_at_most_once():
    eng = _engine("review this code")
    assert eng.check_review_grounding([]) is not None   # first: blocks
    assert eng.check_review_grounding([]) is None        # second: lets it through


def test_gate_respects_disabled_flag():
    eng = _engine("review this code", review_grounding=False)
    assert eng.check_review_grounding([]) is None
