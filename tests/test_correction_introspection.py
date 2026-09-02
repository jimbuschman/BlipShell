"""Correction detector: introspective questions are not corrections.

Regression cover for the 2026-09-02 live incident: "so, why do you think you
didn't see the mechanism?" matched the weak pattern `you didn't`, minted a
permanent ANTI-PATTERN lesson claiming the user corrected the assistant, and
nudged mood to chastened. A store that accumulates these teaches future
sessions to treat the user's curiosity as rebuke.

The rule: WEAK signals (behavior-referencing words that also occur in innocent
questions) are suppressed inside introspective framings; STRONG signals
(explicit correction language) always fire.
"""

import pytest

from blipshell.core.guardrails import (
    detect_correction,
    is_introspective_question,
)
from scripts.sweep_correction_lessons import judge


# --- the live incident and its family: must NOT detect ----------------------

@pytest.mark.parametrize("message", [
    "so, why do you think you didn't see the mechanism?",
    "why do you think you forgot about that?",
    "what do you think you missed in the last session?",
    "out of curiosity, why weren't you doing that already? you skipped it",
    "I'm curious why you ignored the config there — what's your theory?",
    "how would you say you keep ending up in that groove?",
    "in your view, what did you think you didn't understand?",
])
def test_introspective_questions_are_not_corrections(message):
    assert detect_correction(message) is None, message


# --- real corrections: must STILL detect -------------------------------------

@pytest.mark.parametrize("message", [
    "you didn't run the tests",                       # weak, no frame
    "you missed the config file again",               # weak, no frame
    "you keep adding docstrings I didn't ask for",    # weak, no frame
    "stop doing that",                                # weak, no frame
    "I already told you not to touch that file",      # strong
    "that's not what I meant",                        # strong
    "that's wrong, read what I said",                 # strong
    # Strong markers override the introspective frame:
    "no, I meant the other file — why do you think you did that?",
    "that's not right. do you think you misread the plan?",
])
def test_real_corrections_still_detect(message):
    assert detect_correction(message) is not None, message


def test_frame_detection():
    assert is_introspective_question("why do you think you didn't see it?")
    assert is_introspective_question("out of curiosity, what happened?")
    assert not is_introspective_question("you didn't run the tests")


# --- the sweep's re-judgment logic -------------------------------------------

def _lesson(user_said: str) -> str:
    return (
        'ANTI-PATTERN: User corrected the assistant. '
        'Signal: "why do you think you didn\'t see". '
        'Previous response (excerpt): "..." . '
        f'User said: "{user_said}"'
    )


def test_sweep_flags_the_live_incident():
    verdict = judge(_lesson("so, why do you think you didn't see the mechanism?"))
    assert verdict is not None
    assert verdict[0] == "false_positive"


def test_sweep_keeps_real_corrections():
    assert judge(_lesson("you didn't run the tests before committing")) is None


def test_sweep_falls_back_to_signal_when_format_is_old():
    old_format = ('ANTI-PATTERN: User corrected the assistant. '
                  'Signal: "why do you think you didn\'t see".')
    verdict = judge(old_format)
    assert verdict is not None
    assert verdict[0] == "false_positive_lowconf"


def test_sweep_leaves_unrecognized_content_alone():
    assert judge("ANTI-PATTERN: User corrected the assistant. malformed") is None
