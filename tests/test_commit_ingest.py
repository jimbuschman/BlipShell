"""Commits as user-model evidence (the input half of the loop).

Real git repos in tmp_path — the collector shells out to real git, so a
fake would test the mock. Safety properties: non-repos and dead paths are
skipped silently, git failures never raise, and the watermark means a
commit is evidence once, not every night.
"""

import subprocess

import pytest

from blipshell.memory.commit_ingest import WATERMARK_KEY, collect_commit_evidence


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "PATH": __import__("os").environ["PATH"]},
    )


def _make_repo(path, subjects):
    path.mkdir()
    _git(path, "init", "-q")
    for i, subj in enumerate(subjects):
        (path / f"f{i}.txt").write_text(str(i))
        _git(path, "add", "-A")
        _git(path, "commit", "-q", "-m", subj)


async def _add_project(sqlite, name, root):
    await sqlite._db.execute(
        "INSERT INTO projects (name, root_path) VALUES (?, ?)",
        (name, str(root) if root else None),
    )
    await sqlite._db.commit()


class TestCollect:
    async def test_commits_become_labeled_evidence(self, sqlite_store, tmp_path):
        _make_repo(tmp_path / "repo", ["Fix the entity guard", "Add /local toggle"])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        evidence = await collect_commit_evidence(sqlite_store)

        assert len(evidence) == 1
        assert evidence[0].startswith("(git) 2 recent commit(s) in blip:")
        assert "Fix the entity guard" in evidence[0]
        assert "Add /local toggle" in evidence[0]

    async def test_non_repo_and_dead_paths_are_skipped(self, sqlite_store, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        await _add_project(sqlite_store, "plain", plain)
        await _add_project(sqlite_store, "ghost", tmp_path / "nope")
        await _add_project(sqlite_store, "rootless", None)

        assert await collect_commit_evidence(sqlite_store) == []

    async def test_watermark_makes_a_commit_evidence_once(self, sqlite_store, tmp_path):
        """Without the watermark the same commits would be re-fed to the
        revision every night, and 'recent activity' would never age out."""
        _make_repo(tmp_path / "repo", ["only commit"])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        first = await collect_commit_evidence(sqlite_store)
        second = await collect_commit_evidence(sqlite_store)

        assert len(first) == 1
        assert second == [], "the same commit was collected twice"
        import json
        marks = json.loads(await sqlite_store.get_metadata(WATERMARK_KEY))
        assert "blip" in marks, "no per-project watermark recorded"

    async def test_new_commits_after_watermark_are_collected(self, sqlite_store, tmp_path):
        repo = tmp_path / "repo"
        _make_repo(repo, ["old commit"])
        await _add_project(sqlite_store, "blip", repo)
        await collect_commit_evidence(sqlite_store)

        # The watermark is the last collected commit's epoch + 1s, so the
        # new commit must land in a later second to be distinguishable.
        import time
        time.sleep(1.1)
        (repo / "new.txt").write_text("x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "brand new work")

        evidence = await collect_commit_evidence(sqlite_store)

        assert len(evidence) == 1
        assert "brand new work" in evidence[0]
        assert "old commit" not in evidence[0]


class TestRevisionIntegration:
    async def test_commits_reach_the_revision_prompt(self, sqlite_store, tmp_path):
        """A commits-only night still revises — external evidence is
        evidence — and the (git) lines land in the prompt."""
        from unittest.mock import AsyncMock, MagicMock

        from blipshell.memory.user_model import UserModel

        _make_repo(tmp_path / "repo", ["Rewrite consolidation with time budget"])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        router = MagicMock()
        router.generate = AsyncMock(return_value="- (medium) iterates on infrastructure")
        um = UserModel(sqlite_store, router)

        stats = await um.revise_from_reflections()

        assert stats["revised"] is True
        prompt = router.generate.await_args.args[1]
        assert "(git)" in prompt
        assert "Rewrite consolidation with time budget" in prompt
