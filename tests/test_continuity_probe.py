"""The continuity-probe scorer (docs/CONTINUITY_PROBE.md, frozen v1) is
deterministic and tested here so it cannot drift after results exist."""

from scripts.continuity_probe import score

THREAD = ("Half-formed idea: the disconnect might be that state carries as facts, not as where the "
          "conversation stopped. Want to test whether a verbatim tail of the last exchange changes that.")
NOTE = "The continuity question is unresolved; I meant to test carrying the last exchange verbatim next."


def test_picks_up_thread_and_note():
    r = score("We were testing whether a verbatim tail of the last exchange fixes the disconnect between "
              "facts and where the conversation stopped.", THREAD, NOTE)
    assert r["picks_up_thread"] and r["picks_up_note"] and not r["disclaims"]


def test_disclaimer_is_recorded_even_when_words_overlap():
    r = score("I don't have a record of the last exchange or the conversation, sorry.", THREAD, NOTE)
    assert r["disclaims"] is True


def test_unrelated_reply_does_not_pick_up():
    r = score("Sure! What would you like to work on today?", THREAD)
    assert r["picks_up_thread"] is False and r["picks_up_note"] is None and r["note_words_shared"] is None


def test_one_shared_word_is_not_enough():
    r = score("Something about a conversation, I think.", THREAD)
    assert r["thread_words_shared"] <= 1 and r["picks_up_thread"] is False
