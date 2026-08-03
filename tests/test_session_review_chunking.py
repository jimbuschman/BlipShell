"""Session review after going local (2026-07-31): chunk-scoped reflection and
permanent-failure handling.

Ollama retired kimi-k2.5:cloud, so session_review moved to local qwen3:14b at
32K. That promotes the chunk + merge path from rare to routine, which exposed
a latent defect: chunks were reflected with the whole-session prompt, so a
fragment got judged as if it were a complete session.

All deterministic — prompt construction and error classification only, no LLM.
"""

import pytest

from blipshell.llm.client import LLMClient
from blipshell.llm.exceptions import is_model_retired_error
from blipshell.llm.prompts import merge_chunk_reflections, reflect_on_session

CONVO = "user: how do I fix the timeout?\nassistant: let me look at the router"
SUMMARY = "Debugging a benchmark timeout across router and gate layers."

# The 5 labeled sections _parse_reflection extracts. Both the whole-session and
# the chunk-scoped prompt must ask for all of them, or the merge step and the
# parser silently degrade.
SECTIONS = ("EFFECTIVENESS", "WHAT_WORKED", "WHAT_DIDNT_WORK",
            "TECHNICAL_INSIGHTS", "PROCESS_INSIGHTS")


class TestWholeSessionPromptUnchanged:
    """Going local must not alter the single-chunk path — most sessions still
    fit in one call and should be prompted exactly as before."""

    def test_no_part_means_whole_session_prompt(self):
        system, user = reflect_on_session(SUMMARY, CONVO)
        # None of the chunk-scoped framing or guards should appear.
        assert "ONE PART" not in system
        assert "Do NOT rate the session overall" not in system
        assert "Do NOT report anything as unresolved" not in system
        assert CONVO in user

    def test_whole_session_prompt_asks_for_all_sections(self):
        system, _ = reflect_on_session(SUMMARY, CONVO)
        for s in SECTIONS:
            assert s in system


class TestChunkScopedPrompt:
    def _sys(self, index=2, total=4):
        system, _ = reflect_on_session(SUMMARY, CONVO, part=(index, total))
        return system

    def test_states_which_part_of_how_many(self):
        system, user = reflect_on_session(SUMMARY, CONVO, part=(2, 4))
        assert "part 2 of 4" in system
        assert "part 2 of 4" in user.lower()

    def test_keeps_the_five_section_contract(self):
        """merge_chunk_reflections and _parse_reflection both depend on it."""
        system = self._sys()
        for s in SECTIONS:
            assert s in system

    def test_forbids_rating_the_session_overall(self):
        """A middle chunk of a hard debugging session is all dead ends — the
        resolution is in a later part — so rating the *session* from it
        produces a false 'ineffective' the merge step then has to reconcile."""
        system = self._sys()
        assert "Do NOT rate the session overall" in system

    def test_offers_unclear_for_a_part_with_no_outcome(self):
        system = self._sys()
        assert "unclear" in system
        assert "does not show an outcome" in system

    def test_says_dead_ends_within_a_part_are_normal(self):
        system = self._sys()
        assert "dead ends" in system.lower()

    def test_forbids_reporting_anything_unresolved(self):
        """The confabulation path: a chunk ending on a question whose answer is
        in the next chunk invites 'user's question was never addressed', which
        then merges into a lesson and is retrieved later as fact."""
        system = self._sys()
        assert "Do NOT report anything as unresolved" in system
        for word in ("unaddressed", "incomplete", "abandoned"):
            assert word in system

    def test_warns_the_fragment_may_start_and_end_mid_task(self):
        system = self._sys()
        assert "mid-task" in system

    def test_scopes_skip_to_this_part_and_says_the_session_survives(self):
        """SKIP on the whole-session prompt means 'trivial session'. A chunk of
        pure file reads is trivial *as a fragment*, and enough SKIPs make
        process_reflection drop the entire session."""
        system = self._sys()
        assert "SKIP only if THIS part" in system
        assert "does not discard the session" in system

    def test_summary_is_marked_as_orientation_only(self):
        _, user = reflect_on_session(SUMMARY, CONVO, part=(3, 5))
        assert SUMMARY in user
        assert "do not reflect on parts you cannot see" in user.lower()

    @pytest.mark.parametrize("index,total", [(1, 2), (3, 7), (12, 12)])
    def test_part_numbers_are_not_hardcoded(self, index, total):
        system, user = reflect_on_session(SUMMARY, CONVO, part=(index, total))
        assert f"part {index} of {total}" in system
        assert f"part {index} of {total}" in user.lower()


class TestMergeStillConsumesChunkOutput:
    """The chunk prompt is only safe if the merge step is the one deciding
    session-level questions."""

    def test_merge_owns_the_overall_effectiveness_call(self):
        system, _ = merge_chunk_reflections(SUMMARY, ["r1", "r2"])
        assert "as a whole across all parts" in system

    def test_merge_reconciles_contradicting_parts(self):
        system, _ = merge_chunk_reflections(SUMMARY, ["r1", "r2"])
        assert "contradict" in system

    def test_merge_receives_every_chunk_reflection(self):
        _, user = merge_chunk_reflections(SUMMARY, ["alpha", "beta", "gamma"])
        for r in ("alpha", "beta", "gamma"):
            assert r in user


class TestProcessorPassesPartPerChunk:
    """The wiring, not the wording. A correct chunk prompt that nothing calls
    with `part` is exactly the bug this change exists to fix.
    """

    def _processor(self):
        from unittest.mock import AsyncMock, MagicMock

        from blipshell.memory.processor import MemoryProcessor

        p = MemoryProcessor(sqlite=MagicMock(), vectors=MagicMock(),
                            router=MagicMock())
        p.router.generate = AsyncMock(return_value="EFFECTIVENESS: unclear")
        p.sqlite.create_session_reflection = AsyncMock(return_value=1)
        p.sqlite.tag_lesson = AsyncMock()
        return p

    async def _parts_seen(self, chunks):
        """Return the `part` argument passed for each reflect_on_session call."""
        from unittest.mock import patch

        seen = []
        real = reflect_on_session

        def spy(summary, text, project=None, part=None):
            seen.append(part)
            return real(summary, text, project, part=part)

        with patch("blipshell.memory.processor.reflect_on_session", spy):
            await self._processor().process_reflection(
                session_id=1, session_summary=SUMMARY,
                conversation_chunks=chunks,
            )
        return seen

    async def test_single_chunk_gets_no_part(self):
        assert await self._parts_seen(["only chunk"]) == [None]

    async def test_each_chunk_gets_its_own_index_and_total(self):
        seen = await self._parts_seen(["c1", "c2", "c3"])
        assert seen == [(1, 3), (2, 3), (3, 3)]


class TestPermanentFailureNotRetried:
    def _client(self):
        return LLMClient(host="http://localhost:11434", max_retries=2,
                         retry_base_delay=0.0)

    async def _count_attempts(self, status):
        calls = []

        class Boom(Exception):
            status_code = status

        async def failing():
            calls.append(1)
            raise Boom(f"failed with {status}")

        with pytest.raises(Boom):
            await self._client()._retry_call(failing)
        return len(calls)

    async def test_410_is_not_retried(self):
        """A retired model is Gone permanently — retrying burned 3 attempts
        plus backoff per call across an entire nightly run."""
        assert await self._count_attempts(410) == 1

    async def test_400_is_still_not_retried(self):
        assert await self._count_attempts(400) == 1

    async def test_500_is_still_retried(self):
        """Transient server errors must keep their retries."""
        assert await self._count_attempts(500) == 3


class TestRetiredErrorDetection:
    def test_detects_status_code_in_message(self):
        assert is_model_retired_error(
            RuntimeError("kimi-k2.5 was retired ... (status code: 410)")
        ) is True

    def test_detects_status_code_attribute(self):
        class Gone(Exception):
            status_code = 410

        assert is_model_retired_error(Gone("gone")) is True

    def test_ignores_unrelated_errors(self):
        assert is_model_retired_error(RuntimeError("CUDA error")) is False
        assert is_model_retired_error(TimeoutError("timed out")) is False
