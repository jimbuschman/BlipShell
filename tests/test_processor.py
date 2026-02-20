"""Tests for MemoryProcessor — parsing utilities and full pipeline."""

import pytest

from blipshell.memory.processor import MemoryProcessor


# --- Unit tests for static parsing methods ---


class TestParseRankAndImportance:
    def test_standard(self):
        assert MemoryProcessor._parse_rank_and_importance("3 0.5") == (3, 0.5)

    def test_noisy_response(self):
        rank, imp = MemoryProcessor._parse_rank_and_importance(
            "I'd rate this a 4 with importance 0.7"
        )
        assert rank == 4
        assert imp == 0.7

    def test_defaults_on_garbage(self):
        assert MemoryProcessor._parse_rank_and_importance("garbage") == (3, 0.3)

    def test_rank_out_of_range_high(self):
        rank, _ = MemoryProcessor._parse_rank_and_importance("9 0.5")
        assert rank == 3  # out of 1-5 range, defaults

    def test_rank_out_of_range_zero(self):
        rank, _ = MemoryProcessor._parse_rank_and_importance("0 0.5")
        assert rank == 3

    def test_importance_clamped(self):
        _, imp = MemoryProcessor._parse_rank_and_importance("3 5.0")
        assert imp == 1.0

    def test_whitespace_variations(self):
        assert MemoryProcessor._parse_rank_and_importance("  4   0.8  ") == (4, 0.8)

    def test_single_number(self):
        rank, imp = MemoryProcessor._parse_rank_and_importance("4")
        assert rank == 4
        assert imp == 0.3  # default


class TestParseRank:
    def test_single_digit(self):
        assert MemoryProcessor._parse_rank("3") == 3

    def test_in_text(self):
        assert MemoryProcessor._parse_rank("I would rate this a 4") == 4

    def test_out_of_range(self):
        assert MemoryProcessor._parse_rank("9") == 3

    def test_default_on_no_digit(self):
        assert MemoryProcessor._parse_rank("no numbers here") == 3


class TestParseFloat:
    def test_standard(self):
        assert MemoryProcessor._parse_float("0.75") == 0.75

    def test_clamped_high(self):
        assert MemoryProcessor._parse_float("5.0") == 1.0

    def test_integer(self):
        assert MemoryProcessor._parse_float("1") == 1.0

    def test_default_on_garbage(self):
        assert MemoryProcessor._parse_float("no number", default=0.5) == 0.5

    def test_in_sentence(self):
        assert MemoryProcessor._parse_float("importance is 0.6") == 0.6


# --- Integration tests for full pipeline ---


class TestProcessMessagePipeline:
    async def test_full_pipeline(self, memory_processor, sqlite_store, mock_chroma):
        """Full pipeline: summarize → SQLite → Chroma → tag → rank."""
        mem_id = await memory_processor.process_message(
            text="We discussed Python performance tuning and how to use cProfile for profiling.",
            role="user",
            session_id=1,
        )

        assert mem_id is not None

        # Verify memory stored in SQLite
        memory = await sqlite_store.get_memory(mem_id)
        assert memory is not None
        assert memory.session_id == 1
        assert memory.role == "user"
        assert memory.summary is not None
        assert len(memory.summary) > 0

        # Verify rank and importance were set
        assert memory.rank >= 1
        assert memory.importance > 0.0

        # Verify Chroma embed was called
        mock_chroma.add_memory.assert_called_once()

        # Verify tags were assigned (python should match)
        tags = await sqlite_store.get_tags_for_memory(mem_id)
        assert isinstance(tags, list)
        assert "python" in tags

    async def test_noise_skipped(self, memory_processor):
        """Short noise messages should be filtered out."""
        result = await memory_processor.process_message(
            text="ok",
            role="user",
            session_id=1,
        )
        assert result is None

    async def test_skip_response(self, memory_processor, canned_router):
        """LLM returning SKIP should filter the message."""
        # Override canned router to return SKIP for summarization
        original_side_effect = canned_router.generate.side_effect

        def skip_summarize(task_type, prompt="", system=None, think=None):
            if task_type == "summarization":
                return "SKIP"
            return original_side_effect(task_type, prompt, system, think)

        canned_router.generate.side_effect = skip_summarize

        result = await memory_processor.process_message(
            text="I am an AI assistant and I process information.",
            role="assistant",
            session_id=1,
        )
        assert result is None


class TestProcessLesson:
    async def test_creates_lesson(self, memory_processor, sqlite_store, mock_chroma):
        """process_lesson should create a lesson in SQLite and embed in Chroma."""
        # First create a session for the foreign key
        session_id = await sqlite_store.create_session("Test Session")

        lesson_id = await memory_processor.process_lesson(
            conversation_text="User asked about Python performance. We discussed cProfile.",
            session_id=session_id,
        )

        assert lesson_id is not None

        # Verify lesson in SQLite
        lessons = await sqlite_store.get_all_lessons()
        assert len(lessons) >= 1
        found = [l for l in lessons if l.id == lesson_id]
        assert len(found) == 1
        assert len(found[0].content) > 0

        # Verify Chroma embed was called
        mock_chroma.add_lesson.assert_called_once()


class TestProcessCoreMemory:
    async def test_creates_core_memory(self, memory_processor, sqlite_store, mock_chroma):
        """process_core_memory should store in SQLite and embed in Chroma."""
        mem_id = await memory_processor.process_core_memory(
            text="User prefers Python for data analysis.",
            session_id=None,
        )

        assert mem_id is not None

        # Verify in SQLite
        core_memories = await sqlite_store.get_active_core_memories()
        assert len(core_memories) >= 1
        found = [cm for cm in core_memories if cm.id == mem_id]
        assert len(found) == 1

        # Verify Chroma embed
        mock_chroma.add_core_memory.assert_called_once()
