"""Tests for noise filtering (memory/noise.py)."""

import pytest

from blipshell.memory.noise import (
    _is_noise,
    _normalize,
    contains_signal_words,
    should_skip_memory,
)


class TestNormalize:
    def test_lowercase(self):
        assert _normalize("HELLO") == "hello"

    def test_strip_whitespace(self):
        assert _normalize("  hi  ") == "hi"

    def test_remove_punctuation(self):
        assert _normalize("hello!") == "hello"

    def test_collapse_whitespace(self):
        assert _normalize("hello    world") == "hello world"


class TestContainsSignalWords:
    def test_has_signal(self):
        assert contains_signal_words("what do you think about this?")

    def test_no_signal(self):
        assert not contains_signal_words("lol")

    def test_empty(self):
        assert not contains_signal_words("")

    def test_none_like(self):
        assert not contains_signal_words("   ")


class TestIsNoise:
    def test_empty_is_noise(self):
        assert _is_noise("")
        assert _is_noise("   ")
        assert _is_noise(None)

    def test_greetings_are_noise(self):
        assert _is_noise("hi")
        assert _is_noise("Hello")
        assert _is_noise("hey there")
        assert _is_noise("bye")

    def test_affirmatives_are_noise(self):
        assert _is_noise("ok")
        assert _is_noise("yeah")
        assert _is_noise("sure")
        assert _is_noise("nope")

    def test_reactions_are_noise(self):
        assert _is_noise("wow")
        assert _is_noise("cool")
        assert _is_noise("nice")
        assert _is_noise("interesting")

    def test_filler_words_are_noise(self):
        assert _is_noise("um")
        assert _is_noise("uh")

    def test_very_short_is_noise(self):
        assert _is_noise("ab")  # 2 chars, 1 word

    def test_laughter_is_noise(self):
        assert _is_noise("hahaha")
        assert _is_noise("lololol")

    def test_meaningful_content_not_noise(self):
        assert not _is_noise("Can you help me write a Python script for data processing?")
        assert not _is_noise("I want to learn about machine learning algorithms")


class TestShouldSkipMemory:
    def test_noise_skipped(self):
        assert should_skip_memory("hi")
        assert should_skip_memory("ok")

    def test_short_without_signal_skipped(self):
        assert should_skip_memory("it was")

    def test_very_short_skipped(self):
        assert should_skip_memory("ab")

    def test_meaningful_not_skipped(self):
        assert not should_skip_memory(
            "I've been working on a machine learning project for sentiment analysis"
        )

    def test_short_with_signal_not_skipped(self):
        # "remember" is a signal word; 3+ words
        assert not should_skip_memory(
            "do you remember what we discussed about the architecture?"
        )




class TestCorrectionSignals:
    """V3 E1 finding: short corrections were dropped as noise. The added
    markers are narrow and word-bounded; nothing else about the filter moved."""

    @pytest.mark.parametrize("text", [
        "Correction: I switched to spaces, four wide, for indentation. Forget tabs.",
        "Correction: my cat is Luna, not Lunar.",
        "I no longer use Neovim.",
        "Not anymore - I moved off Chroma.",
        "Actually, the Pi is on 192.168.4.78 now.",
        "Scratch that, it is 4 ohm.",
        "I prefer spaces now.",
        "Switched to sqlite-vec last week.",
        "Use Groq instead of Gemini.",
        "We changed the threshold to 0.92.",
        "Revised: nightly runs at 3am.",
        "Rather than tabs, spaces.",
        "Forget what I said about Rust.",
        "I now use Neovim for everything.",
    ])
    def test_short_corrections_are_kept(self, text):
        assert len(text) < 80, "these must be SHORT to test the filter"
        assert contains_signal_words(text)
        assert not should_skip_memory(text)

    @pytest.mark.parametrize("text", [
        "cool, sounds good",
        "haha nice one",
        "that was a fun weekend",
        "the weather is cold today",
        "great job on that",
        "actually",              # opener with no clause is still filler
        "actually lol",
        "no",
        "the switch is in the hall",   # 'switch' as a noun, not 'switched to'
        "a correctional facility",     # not the word 'correction'
        "instead",                     # bare, no 'of'
        "preferences page loads slowly",  # 'prefer' as a substring only
    ])
    def test_short_non_corrections_are_still_dropped(self, text):
        assert len(text) < 80
        assert should_skip_memory(text)

    def test_long_messages_are_unaffected(self):
        long = "x " * 60
        assert not should_skip_memory(long + "nothing correction-like here at all")
        assert not should_skip_memory(long)

    def test_existing_signal_words_still_work(self):
        assert contains_signal_words("what do you think")
        assert not contains_signal_words("cold today")
