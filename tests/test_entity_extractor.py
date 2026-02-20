"""Tests for EntityExtractor — triple parsing and batch extraction."""

import pytest

from blipshell.memory.entity_extractor import EntityExtractor
from blipshell.models.memory import Memory, MemoryType


# --- Unit tests for _parse_triples ---


class TestParseTriples:
    def setup_method(self):
        self.extractor = EntityExtractor.__new__(EntityExtractor)

    def test_standard_triple(self):
        result = self.extractor._parse_triples(
            "user | uses | python | person | technology"
        )
        assert len(result) == 1
        subj, pred, obj, s_type, o_type = result[0]
        assert subj == "user"
        assert pred == "uses"
        assert obj == "python"
        assert s_type == "person"
        assert o_type == "technology"

    def test_multiple_triples(self):
        result = self.extractor._parse_triples(
            "user | discussed | python | person | technology\n"
            "python | is_a | programming_language | technology | concept"
        )
        assert len(result) == 2

    def test_none_response(self):
        assert self.extractor._parse_triples("NONE") == []

    def test_empty_response(self):
        assert self.extractor._parse_triples("") == []

    def test_malformed_line_too_few_parts(self):
        result = self.extractor._parse_triples("incomplete | data")
        assert len(result) == 0

    def test_vague_subject_skipped(self):
        result = self.extractor._parse_triples(
            "it | uses | python | thing | technology"
        )
        assert len(result) == 0

    def test_vague_object_skipped(self):
        result = self.extractor._parse_triples(
            "user | likes | something | person | thing"
        )
        assert len(result) == 0

    def test_missing_types_default_to_concept(self):
        result = self.extractor._parse_triples("user | uses | python")
        assert len(result) == 1
        _, _, _, s_type, o_type = result[0]
        assert s_type == "concept"
        assert o_type == "concept"

    def test_mixed_valid_and_invalid(self):
        result = self.extractor._parse_triples(
            "user | uses | python | person | technology\n"
            "incomplete\n"
            "NONE\n"
            "it | does | something\n"
            "jim | likes | rust | person | technology"
        )
        assert len(result) == 2  # only user|uses|python and jim|likes|rust

    def test_whitespace_handling(self):
        result = self.extractor._parse_triples(
            "  user  |  uses  |  python  |  person  |  technology  "
        )
        assert len(result) == 1
        assert result[0][0] == "user"


# --- Integration tests for extract_batch ---


class TestExtractBatch:
    async def test_creates_entities(self, entity_extractor, sqlite_store):
        """extract_batch should create entities, relationships, and mentions."""
        # Insert a memory with a summary (simulating a processed message)
        session_id = await sqlite_store.create_session("Test")
        memory = Memory(
            session_id=session_id,
            role="user",
            content="I use Python for data analysis",
            summary="User uses Python for data analysis",
            memory_type=MemoryType.CONVERSATION,
        )
        mem_id = await sqlite_store.create_memory(memory)

        # Run extraction
        stats = await entity_extractor.extract_batch()

        assert stats["extracted"] == 1
        assert stats["triples"] >= 1
        assert stats["errors"] == 0

        # Verify entities were created
        entity_names = await sqlite_store.get_all_entity_names()
        assert len(entity_names) >= 2  # at least "user" and "python"

        # Verify mentions link back to the memory
        entity_ids = await sqlite_store.get_entity_ids_by_names(["user"])
        if entity_ids:
            mem_ids = await sqlite_store.get_memory_ids_for_entities(entity_ids)
            assert mem_id in mem_ids

    async def test_marks_extracted(self, entity_extractor, sqlite_store):
        """Memories should be marked as extracted after processing."""
        session_id = await sqlite_store.create_session("Test")
        memory = Memory(
            session_id=session_id,
            role="user",
            content="Test content",
            summary="Test summary",
            memory_type=MemoryType.CONVERSATION,
        )
        await sqlite_store.create_memory(memory)

        # Before extraction
        unextracted = await sqlite_store.get_unextracted_memory_ids(limit=100)
        assert len(unextracted) == 1

        await entity_extractor.extract_batch()

        # After extraction
        unextracted = await sqlite_store.get_unextracted_memory_ids(limit=100)
        assert len(unextracted) == 0

    async def test_empty_batch(self, entity_extractor):
        """No unextracted memories should return zero stats."""
        stats = await entity_extractor.extract_batch()
        assert stats == {"extracted": 0, "triples": 0, "errors": 0}
