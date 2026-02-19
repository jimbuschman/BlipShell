"""LLM-powered tag pattern discovery.

Periodically reviews poorly-tagged memories and asks an LLM to suggest
new topic tag regex patterns. Discovered patterns are persisted to SQLite
and merged into the tagger at startup.
"""

import logging
import re
from datetime import datetime, timezone

from blipshell.llm.prompts import discover_tag_patterns
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tagger import TOPIC_PATTERNS

logger = logging.getLogger(__name__)

METADATA_KEY = "last_tag_discovery"


class TagDiscovery:
    """Discovers new tag patterns by analyzing poorly-tagged memories with an LLM."""

    def __init__(
        self,
        sqlite: SQLiteStore,
        router: LLMRouter,
        interval_days: int = 7,
        sample_size: int = 20,
    ):
        self.sqlite = sqlite
        self.router = router
        self.interval_days = interval_days
        self.sample_size = sample_size

    async def maybe_run(self) -> dict:
        """Run discovery if enough time has elapsed since last run.

        Returns stats dict with keys: discovered, skipped, skipped_reason.
        """
        stats = {"discovered": 0, "skipped": 0, "skipped_reason": ""}

        # Check if enough time has passed
        last_run = await self.sqlite.get_metadata(METADATA_KEY)
        if last_run:
            try:
                last_dt = datetime.fromisoformat(last_run)
                elapsed = datetime.now(timezone.utc) - last_dt.replace(
                    tzinfo=timezone.utc if last_dt.tzinfo is None else last_dt.tzinfo,
                )
                if elapsed.days < self.interval_days:
                    stats["skipped_reason"] = (
                        f"last run {elapsed.days}d ago (interval={self.interval_days}d)"
                    )
                    return stats
            except (ValueError, TypeError):
                pass  # Invalid timestamp, proceed with discovery

        # Get poorly-tagged memory summaries
        summaries = await self.sqlite.get_poorly_tagged_memory_summaries(
            max_tags=1, limit=self.sample_size,
        )
        if not summaries:
            stats["skipped_reason"] = "no poorly-tagged memories found"
            await self._update_timestamp()
            return stats

        # Build full list of existing tag names
        existing_tags = sorted(TOPIC_PATTERNS.keys())

        # Ask LLM for new patterns
        system, prompt = discover_tag_patterns(summaries, existing_tags)
        try:
            response = await self.router.generate(
                TaskType.REASONING, prompt, system=system, think=False,
            )
        except Exception as e:
            logger.error("Tag discovery LLM call failed: %s", e)
            stats["skipped_reason"] = f"LLM error: {e}"
            return stats

        # Parse and validate response
        patterns = self._parse_response(response)
        if not patterns:
            logger.info("Tag discovery found no new patterns")
            await self._update_timestamp()
            return stats

        # Persist to SQLite
        await self.sqlite.save_discovered_tag_patterns(patterns)
        total = sum(len(v) for v in patterns.values())
        stats["discovered"] = total
        logger.info(
            "Tag discovery found %d new patterns across %d tags",
            total, len(patterns),
        )

        await self._update_timestamp()
        return stats

    def _parse_response(self, response: str) -> dict[str, list[str]]:
        """Parse LLM response into {tag_name: [regex_pattern, ...]}."""
        if not response or response.strip().upper() == "NONE":
            return {}

        patterns: dict[str, list[str]] = {}
        for line in response.strip().splitlines():
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue

            # Expected format: "tag_name: regex_pattern"
            if ":" not in line:
                continue

            tag_name, _, regex_str = line.partition(":")
            tag_name = tag_name.strip().lower()
            regex_str = regex_str.strip()

            if not tag_name or not regex_str:
                continue

            # Sanitize tag name: only lowercase letters, digits, hyphens
            if not re.match(r"^[a-z0-9][a-z0-9-]*$", tag_name):
                continue

            # Validate regex
            try:
                re.compile(regex_str, re.IGNORECASE)
            except re.error:
                logger.debug("Invalid regex from LLM for '%s': %s", tag_name, regex_str)
                continue

            # Skip if this tag already exists in hardcoded patterns
            if tag_name in TOPIC_PATTERNS:
                continue

            patterns.setdefault(tag_name, []).append(regex_str)

        return patterns

    async def _update_timestamp(self):
        """Update the last run timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        await self.sqlite.set_metadata(METADATA_KEY, now)
