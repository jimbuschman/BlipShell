"""User model (V2_PLAN 5.1): one revised-never-appended document of
reasoned conclusions about the user.

The three enforced properties: the size cap is HARD (post-generation, line
-boundary truncation), the revision watermark advances to the last
reflection actually read (never to "now", which loses backlog), and the
document routes through the LOCAL model — it is the distilled personal
layer, the exact content /local exists to protect.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from blipshell.memory.user_model import (
    EMPTY, MAX_EVIDENCE, UserModel, enforce_cap, revision_prompt,
)


class TestCap:
    def test_line_count_cap(self):
        doc = "\n".join(f"- (high) conclusion {i}" for i in range(60))
        assert len(enforce_cap(doc).splitlines()) <= 25

    def test_token_cap_truncates_at_line_boundary(self):
        long_line = "- (low) " + "word " * 200      # ~200 tokens per line
        doc = "\n".join([long_line] * 20)
        capped = enforce_cap(doc)
        for line in capped.splitlines():
            assert line == long_line, "a conclusion was cut mid-sentence"
        from blipshell.memory.manager import estimate_tokens
        assert estimate_tokens(capped) <= 700

    def test_cap_fits_the_local_core_pool(self):
        """The collision this cap prevents: Core is 5% of context, which on
        the 32K local path is ~1600 tokens. The document must leave room for
        the curated facts it shares that pool with — at 1500 tokens it
        evicted all of them on /local (review 2026-08-10)."""
        from blipshell.memory.user_model import MAX_TOKENS
        local_core_pool = int(32768 * 0.05)
        assert MAX_TOKENS <= local_core_pool * 0.5, (
            "the user model can consume more than half the local Core pool"
        )

    def test_short_doc_passes_through(self):
        doc = "- (high) prefers thorough tests\n- (medium) dislikes rework"
        assert enforce_cap(doc) == doc


class TestPrompt:
    def test_prompt_demands_revision_and_retraction(self):
        system, user = revision_prompt("- (high) old view", ["new evidence"])
        assert "RETRACT" in system
        assert "append" in system.lower()
        assert "- (high) old view" in user
        assert "new evidence" in user

    def test_first_revision_has_no_current_model(self):
        _, user = revision_prompt("", ["evidence"])
        assert "first revision" in user


@pytest.fixture
async def seeded(sqlite_store):
    """Three reflections with distinct, ordered created_at stamps."""
    for i in range(3):
        sid = await sqlite_store.create_session(f"s{i}")
        await sqlite_store._db.execute(
            "INSERT INTO session_reflections "
            "(session_id, reflection_text, created_at) VALUES (?, ?, ?)",
            (sid, f"reflection {i}", f"2026-08-0{i + 1}T00:00:00"),
        )
    await sqlite_store._db.commit()
    return sqlite_store


class TestRevision:
    def _router(self, response):
        r = MagicMock()
        r.generate = AsyncMock(return_value=response)
        return r

    async def test_revision_stores_doc_and_advances_watermark(self, seeded):
        router = self._router("- (high) prefers thorough automated testing")
        um = UserModel(seeded, router)

        stats = await um.revise_from_reflections()

        assert stats["revised"] is True
        assert "thorough automated testing" in await um.get()
        # Watermark = created_at of the LAST reflection read, not "now".
        assert await um.updated_at() == "2026-08-03T00:00:00"
        # Local model only: the one generate call went to REASONING.
        task_type = router.generate.await_args.args[0]
        assert str(task_type).lower().find("reasoning") != -1 or \
            getattr(task_type, "value", "") == "reasoning"

    async def test_second_run_sees_only_new_evidence(self, seeded):
        router = self._router("- (high) x")
        um = UserModel(seeded, router)
        await um.revise_from_reflections()

        stats = await um.revise_from_reflections()

        assert stats == {"revised": False, "reason": "no new reflections"}
        assert router.generate.await_count == 1

    async def test_backlog_beyond_limit_is_not_lost(self, sqlite_store):
        """The watermark bug this module was almost born with: stamping "now"
        after reading a capped batch silently skips the rest of a backlog."""
        for i in range(MAX_EVIDENCE + 3):
            sid = await sqlite_store.create_session(f"s{i}")
            await sqlite_store._db.execute(
                "INSERT INTO session_reflections "
                "(session_id, reflection_text, created_at) VALUES (?, ?, ?)",
                (sid, f"reflection {i:02d}", f"2026-07-01T00:00:{i:02d}"),
            )
        await sqlite_store._db.commit()
        router = self._router("- (medium) y")
        um = UserModel(sqlite_store, router)

        first = await um.revise_from_reflections()
        second = await um.revise_from_reflections()

        assert first["evidence"] == MAX_EVIDENCE
        assert second["evidence"] == 3, (
            "the backlog beyond one batch was silently skipped"
        )

    async def test_nothing_to_conclude_still_advances(self, seeded):
        router = self._router(EMPTY)
        um = UserModel(seeded, router)

        stats = await um.revise_from_reflections()

        assert stats["revised"] is False
        assert await um.get() is None
        # Watermark advanced — the same evidence is not re-judged nightly.
        assert (await um.revise_from_reflections())["reason"] == "no new reflections"

    async def test_oversized_response_is_capped_before_storage(self, seeded):
        router = self._router(
            "\n".join(f"- (low) speculative conclusion {i}" for i in range(80))
        )
        um = UserModel(seeded, router)

        await um.revise_from_reflections()

        assert len((await um.get()).splitlines()) <= 25
