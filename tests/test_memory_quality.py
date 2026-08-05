"""Tests for memory quality features: classification parser, action parser, decay math."""

from math import exp

import pytest

from blipshell.memory.processor import MemoryProcessor


# --- Classification parser (Feature 1) ---


class TestParseRankImportanceType:
    """Tests for _parse_rank_importance_type() which parses '4 0.7 fact' format."""

    def test_standard_fact(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("4 0.7 fact")
        assert rank == 4
        assert imp == 0.7
        assert mtype == "fact"

    def test_standard_event(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("3 0.5 event")
        assert rank == 3
        assert imp == 0.5
        assert mtype == "event"

    def test_standard_preference(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("4 0.7 preference")
        assert rank == 4
        assert imp == 0.7
        assert mtype == "preference"

    def test_standard_skill(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("4 0.7 skill")
        assert rank == 4
        assert imp == 0.7
        assert mtype == "skill"

    def test_standard_conversation(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("2 0.3 conversation")
        assert rank == 2
        assert imp == 0.3
        assert mtype == "conversation"

    def test_fallback_to_conversation_when_missing_type(self):
        """When LLM returns only rank+importance (old format), type defaults to conversation."""
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("3 0.5")
        assert rank == 3
        assert imp == 0.5
        assert mtype == "conversation"

    def test_noisy_response(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type(
            "I'd rate this a 4 with importance 0.7, it's a fact"
        )
        assert rank == 4
        assert imp == 0.7
        assert mtype == "fact"

    def test_defaults_on_garbage(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("garbage text")
        assert rank == 3
        assert imp == 0.3
        assert mtype == "conversation"

    def test_unknown_type_falls_back(self):
        """Unknown type words should fall back to conversation."""
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("4 0.7 unknown_type")
        assert rank == 4
        assert imp == 0.7
        assert mtype == "conversation"

    def test_type_with_punctuation(self):
        """Type word with trailing punctuation should still be recognized."""
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("4 0.7 fact.")
        assert rank == 4
        assert imp == 0.7
        assert mtype == "fact"

    def test_whitespace_variations(self):
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type("  5   0.9  event  ")
        assert rank == 5
        assert imp == 0.9
        assert mtype == "event"

    def test_multiline_response(self):
        """Some models might wrap their response in extra text."""
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type(
            "Based on analysis:\n4 0.7 skill\n"
        )
        assert rank == 4
        assert imp == 0.7
        assert mtype == "skill"

    def test_last_word_priority(self):
        """Type should be parsed from last word, not first type-like word found."""
        # LLM says "this is a factual preference" — last word wins
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type(
            "4 0.7 this is a factual preference"
        )
        assert mtype == "preference"

    def test_type_word_in_explanation_ignored(self):
        """If LLM explains with type words, the actual type (last) should win."""
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type(
            "This fact about events is 4 0.8 skill"
        )
        assert mtype == "skill"

    def test_last_word_with_trailing_junk(self):
        """Last word with punctuation should still be recognized."""
        rank, imp, mtype = MemoryProcessor._parse_rank_importance_type(
            "4 0.7 event."
        )
        assert mtype == "event"


# --- Action parser (Feature 3: Memory Dedup) ---


class TestParseMemoryAction:
    """Tests for _parse_memory_action() which parses ADD/UPDATE/DELETE/NONE."""

    def test_add(self):
        action, idx = MemoryProcessor._parse_memory_action("ADD")
        assert action == "ADD"
        assert idx is None

    def test_none(self):
        action, idx = MemoryProcessor._parse_memory_action("NONE")
        assert action == "NONE"
        assert idx is None

    def test_update_with_number(self):
        action, idx = MemoryProcessor._parse_memory_action("UPDATE 1")
        assert action == "UPDATE"
        assert idx == 0  # 1-based to 0-based

    def test_update_number_2(self):
        action, idx = MemoryProcessor._parse_memory_action("UPDATE 2")
        assert action == "UPDATE"
        assert idx == 1

    def test_delete_with_number(self):
        action, idx = MemoryProcessor._parse_memory_action("DELETE 1")
        assert action == "DELETE"
        assert idx == 0

    def test_noisy_add(self):
        action, idx = MemoryProcessor._parse_memory_action(
            "The new memory adds unique information. ADD"
        )
        assert action == "ADD"

    def test_noisy_none(self):
        action, idx = MemoryProcessor._parse_memory_action(
            "This is redundant. NONE"
        )
        assert action == "NONE"

    def test_noisy_update(self):
        action, idx = MemoryProcessor._parse_memory_action(
            "This refines existing memory 2. UPDATE 2"
        )
        assert action == "UPDATE"
        assert idx == 1

    def test_default_on_garbage(self):
        action, idx = MemoryProcessor._parse_memory_action("garbage response")
        assert action == "ADD"  # default to ADD

    def test_case_insensitive(self):
        action, _ = MemoryProcessor._parse_memory_action("none")
        assert action == "NONE"

    def test_update_without_number_defaults_to_first(self):
        action, idx = MemoryProcessor._parse_memory_action("UPDATE")
        assert action == "UPDATE"
        assert idx == 0
