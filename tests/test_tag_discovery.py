"""Tests for TagDiscovery — response parsing and discovery pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from blipshell.memory.tag_discovery import TagDiscovery
from blipshell.memory.tagger import TOPIC_PATTERNS
from blipshell.models.memory import Memory, MemoryType


# --- Unit tests for _parse_response ---


class TestParseResponse:
    def setup_method(self):
        self.discovery = TagDiscovery.__new__(TagDiscovery)

    def test_valid_pattern(self):
        result = self.discovery._parse_response("docker: \\bdocker\\b")
        assert "docker" in result
        assert len(result["docker"]) == 1

    def test_multiple_patterns(self):
        result = self.discovery._parse_response(
            "docker: \\bdocker\\b\n"
            "kubernetes: \\bk8s\\b|\\bkubernetes\\b"
        )
        assert "docker" in result
        assert "kubernetes" in result

    def test_none_response(self):
        assert self.discovery._parse_response("NONE") == {}

    def test_empty_response(self):
        assert self.discovery._parse_response("") == {}

    def test_invalid_regex_skipped(self):
        result = self.discovery._parse_response("bad: [invalid")
        assert result == {}

    def test_existing_tag_skipped(self):
        # Pick a tag that exists in TOPIC_PATTERNS
        existing_tag = next(iter(TOPIC_PATTERNS))
        result = self.discovery._parse_response(
            f"{existing_tag}: \\bsome_pattern\\b"
        )
        assert existing_tag not in result

    def test_bad_tag_name_skipped(self):
        result = self.discovery._parse_response("BAD NAME!: \\bpattern\\b")
        assert result == {}

    def test_no_colon_skipped(self):
        result = self.discovery._parse_response("just some text without colons")
        assert result == {}

    def test_empty_tag_name_skipped(self):
        result = self.discovery._parse_response(": \\bpattern\\b")
        assert result == {}

    def test_empty_regex_skipped(self):
        result = self.discovery._parse_response("docker: ")
        assert result == {}


# --- Integration test for maybe_run ---


class TestMaybeRun:
    async def test_discovers_patterns(self, sqlite_store, canned_router):
        """maybe_run should discover and persist new tag patterns."""
        # Create a session and some poorly-tagged memories
        session_id = await sqlite_store.create_session("Test")
        for i in range(3):
            memory = Memory(
                session_id=session_id,
                role="user",
                content=f"Docker container orchestration discussion {i}",
                summary=f"Discussion about docker containers and orchestration {i}",
                memory_type=MemoryType.CONVERSATION,
                rank=3,
                importance=0.5,
            )
            await sqlite_store.create_memory(memory)

        # Override canned router to return tag discovery patterns
        original_side_effect = canned_router.generate.side_effect

        def tag_discovery_response(task_type, prompt="", system=None, think=None):
            if task_type == "reasoning" and system and "tag" in system.lower():
                return "docker: \\bdocker\\b|\\bcontainer\\b"
            return original_side_effect(task_type, prompt, system, think)

        canned_router.generate.side_effect = tag_discovery_response

        discovery = TagDiscovery(
            sqlite=sqlite_store,
            router=canned_router,
            interval_days=0,  # force run
            sample_size=10,
        )

        stats = await discovery.maybe_run()

        # May discover patterns depending on whether the tag already exists
        # in TOPIC_PATTERNS. If "docker" isn't hardcoded, it should discover it.
        if "docker" not in TOPIC_PATTERNS:
            assert stats["discovered"] >= 1
            # Verify persisted
            patterns = await sqlite_store.get_discovered_tag_patterns()
            assert len(patterns) >= 1

    async def test_skips_when_interval_not_elapsed(self, sqlite_store, canned_router):
        """maybe_run should skip if interval hasn't elapsed."""
        from datetime import datetime, timezone
        # Set last run to now
        await sqlite_store.set_metadata(
            "last_tag_discovery",
            datetime.now(timezone.utc).isoformat(),
        )

        discovery = TagDiscovery(
            sqlite=sqlite_store,
            router=canned_router,
            interval_days=7,
            sample_size=10,
        )

        stats = await discovery.maybe_run()
        assert stats["skipped_reason"] != ""
        assert stats["discovered"] == 0
