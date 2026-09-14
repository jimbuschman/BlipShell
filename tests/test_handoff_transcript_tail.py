"""The handoff transcript keeps the END of recent messages.

`transcript_tail` took each message's FIRST 400 characters and then sliced
the joined text with `text[-max_chars:]`. A long final turn was therefore cut
at character 400 — taking the decision and the next action with it, which is
precisely what the note exists to carry ("what you intended to do or pick up
next"). The closing slice could also land mid-line, handing the model a
fragment with no role label on it.
"""

from types import SimpleNamespace

import pytest

from blipshell.core.handoff import (
    MIN_EXCERPT_CHARS,
    PER_MESSAGE_CHARS,
    _excerpt,
    transcript_tail,
)


def msg(role, content):
    return SimpleNamespace(role=role, content=content)


DECISION = "DECISION: revert the cursor change. NEXT: re-run the gate batch."


# --- the end of the final message survives ----------------------------------------


def test_a_decision_past_character_400_survives():
    """THE regression: the old head-truncation cut it off."""
    long_turn = "context. " * 60 + DECISION          # decision at ~char 540
    assert len(long_turn) > 400
    out = transcript_tail([msg("user", "where are we?"), msg("assistant", long_turn)])

    assert DECISION in out
    assert long_turn[:50] in out, "the opening is kept too — it says what the turn is about"


def test_the_old_head_truncation_is_what_this_replaces():
    """Pin the premise."""
    long_turn = "context. " * 60 + DECISION
    assert DECISION not in long_turn[:400]


def test_a_decision_at_the_very_end_of_a_very_long_turn_survives():
    long_turn = "x" * 5000 + " " + DECISION
    out = transcript_tail([msg("assistant", long_turn)])
    assert DECISION in out


def test_an_excerpted_message_is_marked_as_excerpted():
    """An unmarked excerpt reads as a complete thought."""
    out = transcript_tail([msg("assistant", "y" * 4000)])
    assert "omitted" in out


def test_a_message_that_fits_is_not_marked():
    out = transcript_tail([msg("assistant", "a short complete turn")])
    assert out == "assistant: a short complete turn"
    assert "omitted" not in out


# --- several long messages --------------------------------------------------------


def test_multiple_long_messages_each_keep_their_ending():
    endings = [f"ENDING-{i}" for i in range(4)]
    messages = [msg("assistant", f"opening {i} " + "z" * 2000 + " " + endings[i])
                for i in range(4)]

    out = transcript_tail(messages, max_chars=6000)

    for ending in endings:
        assert ending in out, f"{ending} was cut"


def test_chronological_order_is_preserved_though_the_budget_is_spent_backwards():
    messages = [msg("user", f"turn {i} " + "q" * 900) for i in range(5)]
    out = transcript_tail(messages, max_chars=6000)
    positions = [out.index(f"turn {i}") for i in range(5)]
    assert positions == sorted(positions)


def test_every_line_carries_its_role_label():
    messages = [msg("user", "u" * 1500), msg("assistant", "a" * 1500),
                msg("user", "u2" * 800)]
    out = transcript_tail(messages, max_chars=6000)
    for line in out.split("\n"):
        assert line.startswith(("user: ", "assistant: ", "[...")), line


def test_roles_are_read_from_an_enum_value_too():
    role = SimpleNamespace(value="assistant")
    out = transcript_tail([SimpleNamespace(role=role, content="hello")])
    assert out == "assistant: hello"


# --- the total budget -------------------------------------------------------------


@pytest.mark.parametrize("max_chars", [300, 500, 1000, 2000, 6000])
def test_the_total_character_budget_is_never_exceeded(max_chars):
    messages = [msg("assistant", f"turn {i} " + "w" * 3000) for i in range(30)]
    out = transcript_tail(messages, max_chars=max_chars)
    assert len(out) <= max_chars


def test_budget_truncation_drops_whole_messages_and_says_how_many():
    messages = [msg("assistant", f"turn {i} " + "w" * 2000) for i in range(20)]
    out = transcript_tail(messages, max_chars=2000)

    assert "earlier message(s) omitted" in out
    assert out.startswith("[...")
    # The newest turn is the one that survives.
    assert "turn 19" in out
    assert "turn 0" not in out


def test_no_line_is_a_fragment_without_a_label():
    """The old `text[-max_chars:]` could slice mid-line."""
    messages = [msg("assistant", "m" * 3000) for _ in range(10)]
    out = transcript_tail(messages, max_chars=1500)
    first = out.split("\n")[0]
    assert first.startswith("[...") or first.startswith(("user: ", "assistant: "))


def test_max_messages_still_bounds_the_window():
    messages = [msg("user", f"turn {i}") for i in range(100)]
    out = transcript_tail(messages, max_messages=5)
    assert "turn 99" in out
    assert "turn 94" not in out


def test_blank_messages_are_skipped_not_rendered_as_empty_lines():
    messages = [msg("user", "real"), msg("assistant", "   "), msg("user", "also real")]
    out = transcript_tail(messages)
    assert out == "user: real\nuser: also real"


def test_an_empty_transcript_is_an_empty_string():
    assert transcript_tail([]) == ""
    assert transcript_tail([msg("user", "")]) == ""


# --- the excerpt primitive --------------------------------------------------------


def test_excerpt_returns_text_unchanged_when_it_fits():
    assert _excerpt("short", 100) == "short"
    assert _excerpt("exactly ten", 11) == "exactly ten"


@pytest.mark.parametrize("budget", [MIN_EXCERPT_CHARS, 200, 400, PER_MESSAGE_CHARS, 3000])
def test_excerpt_respects_its_budget_and_keeps_the_ending(budget):
    text = "HEAD " + "." * 10000 + " TAIL"
    out = _excerpt(text, budget)
    assert len(out) <= budget
    assert out.endswith("TAIL")
    assert out.startswith("HEAD")


@pytest.mark.parametrize("budget", [5, 10, 20, 40, 60])
def test_excerpt_still_keeps_the_end_at_an_absurdly_small_budget(budget):
    text = "a" * 500 + "END"
    out = _excerpt(text, budget)
    assert len(out) <= budget
    assert out.endswith("END")


def test_excerpt_states_a_count_that_matches_what_it_dropped():
    import re
    text = "z" * 5000
    out = _excerpt(text, 500)
    stated = int(re.search(r"\[\.\.\.(\d+) chars omitted\.\.\.\]", out).group(1))
    kept = len(out) - len(f" [...{stated} chars omitted...] ")
    assert stated + kept == len(text)
