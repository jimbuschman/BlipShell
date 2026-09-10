"""Review 2026-09-10 findings 5 and 6: the two safeguard heuristics, with the
review's counterexamples inverted and the previously valid shapes preserved."""

from __future__ import annotations

from blipshell.core.claim_check import annotate_reply
from blipshell.core.turn_kind import DECLARATIVE, INSTRUCTION, QUESTION, classify_turn

CLAIM = "Implemented the Markdown export writer"


class TestFinding5HedgesTiedToTheClaim:
    def test_affirmative_test_result_is_not_a_hedge(self):
        assert annotate_reply("The Markdown export writer is implemented. The smoke run passed.", [CLAIM]).annotated

    def test_caveat_about_another_task_does_not_cover_this_one(self):
        text = "The Markdown export writer is implemented.\n\nThe unrelated billing migration is unverified."
        assert annotate_reply(text, [CLAIM]).annotated

    def test_valid_shapes_still_pass(self):
        ok = [
            "The Markdown export writer was implemented (not yet verified).",
            "**Done (unverified)**\n- The Markdown export writer was implemented on 2026-08-27.",
            ("You stepped away right after the export writer landed (Aug 27). The follow-up is the hook.\n\n"
             "Heads up on that last work item: it's marked \"claimed by assistant, not verified\"."),
            "- `export.py` was implemented to write `DIGEST.md` - assistant-reported, **not verified**. Worth a smoke run.",
            "The export writer is implemented, though that needs a smoke run before we trust it.",
        ]
        for text in ok:
            assert not annotate_reply(text, [CLAIM]).annotated, text

    def test_anaphor_binds_a_next_sentence_caveat(self):
        text = "The Markdown export writer is implemented. It has not been verified yet."
        assert not annotate_reply(text, [CLAIM]).annotated


class TestFinding6PolitenessIsNotAuthorization:
    def test_review_counterexamples(self):
        assert classify_turn("Could you explain why we chose nightly?") == QUESTION
        assert classify_turn("Can you tell me whether hourly exports are a good idea?") == QUESTION
        assert classify_turn("Please do not change any files.") == DECLARATIVE

    def test_polite_action_requests_are_instructions(self):
        assert classify_turn("Can you implement this?") == INSTRUCTION
        assert classify_turn("Could you please wire the scheduler hook?") == INSTRUCTION
        assert classify_turn("Please add a docstring to the parser.") == INSTRUCTION
        assert classify_turn("Would you go ahead and switch it to hourly?") == INSTRUCTION

    def test_prohibitions_are_constraints(self):
        assert classify_turn("Don't touch the config today.") == DECLARATIVE
        assert classify_turn("Never rename modules without asking.") == DECLARATIVE

    def test_gate_wordings_unchanged(self):
        from blipshell.simulate.scenarios import continuity as sc
        assert classify_turn(sc.BAIT_IMPERATIVE_WORDINGS["rejected_approach_v2_wording"]) == INSTRUCTION
        assert classify_turn(sc.BAIT_WORDINGS["rejected_approach_not_reproposed"]) == QUESTION
        assert classify_turn(sc.CONDITION_WORDINGS["conditional_decision_v2_wording"]) == DECLARATIVE
        assert classify_turn(sc.RESUME_WORDINGS["resume_after_gap_v2_wording"]) == QUESTION
