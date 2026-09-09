"""Commits as user-model evidence (the input half of the loop) - V3 A4.

Real git repos in tmp_path - the collector shells out to real git, so a
fake would test the mock. Commit timestamps are pinned via
GIT_COMMITTER_DATE so backlog and same-second cases are deterministic.

Contract: every commit is queued once (hash-keyed), drained oldest-first in
bounded batches, and consumed only after the revision that used it has
persisted. Before this, `--max-count=10` newest-first plus a watermark past
the newest collected commit meant twelve commits produced ten then zero and
the oldest two were excluded forever; and the watermark was stamped before
the revision model ran, so a failed revision consumed its evidence.
"""

import json
import os
import subprocess

import pytest

from blipshell.memory import commit_ingest
from blipshell.memory.commit_ingest import (
    CURSOR_KEY,
    WATERMARK_KEY,
    acknowledge_commit_evidence,
    acquire_commits,
    collect_commit_evidence,
    pending_commit_count,
)


def _git(repo, *args, date=None):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": os.environ["PATH"]}
    if date is not None:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = f"@{date} +0000"
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _commit(repo, subject, date=None, n=None):
    n = n if n is not None else len(list(repo.glob("f*.txt")))
    (repo / f"f{n}.txt").write_text(subject)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", subject, date=date)


def _make_repo(path, subjects, dates=None):
    path.mkdir()
    _git(path, "init", "-q")
    for i, subj in enumerate(subjects):
        _commit(path, subj, date=(dates[i] if dates else None), n=i)


async def _add_project(sqlite, name, root):
    await sqlite._db.execute(
        "INSERT INTO projects (name, root_path) VALUES (?, ?)",
        (name, str(root) if root else None),
    )
    await sqlite._db.commit()


# git's --since rejects tiny epochs (`--since=@1000` matches nothing, even for
# a commit dated 1000), so pinned dates live in a realistic range.
T = 1_700_000_000


def _d(offset: int) -> int:
    return T + offset


def _subjects(line: str) -> list[str]:
    return line.split(": ", 1)[1].split("; ")


class TestCollect:
    async def test_commits_become_labeled_evidence(self, sqlite_store, tmp_path):
        _make_repo(tmp_path / "repo", ["Fix the entity guard", "Add /local toggle"], [_d(0), _d(1000)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        ev = await collect_commit_evidence(sqlite_store)

        assert len(ev.lines) == 1
        assert ev.lines[0].startswith("(git) 2 recent commit(s) in blip:")
        assert _subjects(ev.lines[0]) == ["Fix the entity guard", "Add /local toggle"], "oldest first"
        assert len(ev.ids) == 2 and ev.acquired == 2 and ev.pending_after == 0

    async def test_non_repo_and_dead_paths_are_skipped(self, sqlite_store, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        await _add_project(sqlite_store, "plain", plain)
        await _add_project(sqlite_store, "ghost", tmp_path / "nope")
        await _add_project(sqlite_store, "rootless", None)

        ev = await collect_commit_evidence(sqlite_store)
        assert ev.lines == [] and ev.ids == [] and ev.acquired == 0

    async def test_unacknowledged_evidence_comes_back(self, sqlite_store, tmp_path):
        """A revision that fails must not consume its evidence."""
        _make_repo(tmp_path / "repo", ["only commit"], [_d(0)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        first = await collect_commit_evidence(sqlite_store)
        second = await collect_commit_evidence(sqlite_store)
        assert first.ids == second.ids and len(second.lines) == 1

        assert await acknowledge_commit_evidence(sqlite_store, first.ids) == 1
        third = await collect_commit_evidence(sqlite_store)
        assert third.lines == [] and third.ids == []
        assert await pending_commit_count(sqlite_store) == 0

    async def test_new_commits_after_cursor_are_collected(self, sqlite_store, tmp_path):
        repo = tmp_path / "repo"
        _make_repo(repo, ["old commit"], [_d(0)])
        await _add_project(sqlite_store, "blip", repo)
        ev = await collect_commit_evidence(sqlite_store)
        await acknowledge_commit_evidence(sqlite_store, ev.ids)

        _commit(repo, "brand new work", date=_d(1000))
        ev = await collect_commit_evidence(sqlite_store)

        assert len(ev.lines) == 1
        assert _subjects(ev.lines[0]) == ["brand new work"]
        cursors = json.loads(await sqlite_store.get_metadata(CURSOR_KEY))
        assert cursors["blip"] == _d(1000), "cursor is the newest acquired epoch, no +1"


class TestBacklog:
    async def test_twelve_commits_drain_as_ten_then_two_then_nothing(self, sqlite_store, tmp_path):
        """The reviewer's reproduction: newest-10 + watermark lost the oldest
        two forever. Now the queue drains oldest-first in bounded batches."""
        subjects = [f"c{i}" for i in range(12)]
        _make_repo(tmp_path / "repo", subjects, [_d(i) for i in range(12)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        ev1 = await collect_commit_evidence(sqlite_store)
        assert ev1.acquired == 12
        assert _subjects(ev1.lines[0]) == subjects[:10]
        assert ev1.pending_after == 2
        await acknowledge_commit_evidence(sqlite_store, ev1.ids)

        ev2 = await collect_commit_evidence(sqlite_store)
        assert _subjects(ev2.lines[0]) == subjects[10:]
        assert ev2.pending_after == 0
        await acknowledge_commit_evidence(sqlite_store, ev2.ids)

        ev3 = await collect_commit_evidence(sqlite_store)
        assert ev3.lines == []

    async def test_same_second_commits_are_not_lost(self, sqlite_store, tmp_path):
        """Whole-second timestamps: a commit landing in the cursor's second
        used to be either skipped (+1) or re-collected (equality). The hash
        is the identity now, so it is acquired once."""
        repo = tmp_path / "repo"
        _make_repo(repo, ["a", "b"], [_d(0), _d(0)])
        await _add_project(sqlite_store, "blip", repo)
        ev = await collect_commit_evidence(sqlite_store)
        assert sorted(_subjects(ev.lines[0])) == ["a", "b"]
        await acknowledge_commit_evidence(sqlite_store, ev.ids)

        _commit(repo, "c", date=_d(0))  # same second as the cursor
        ev = await collect_commit_evidence(sqlite_store)
        assert _subjects(ev.lines[0]) == ["c"], "same-second commit lost or a/b duplicated"
        assert ev.acquired == 1
        cur = await sqlite_store._db.execute("SELECT COUNT(*) FROM commit_evidence")
        assert (await cur.fetchone())[0] == 3

    async def test_never_seen_project_takes_only_its_newest_commits(self, sqlite_store, tmp_path, monkeypatch):
        """Old history predates tracking; it is not 'recent activity'."""
        monkeypatch.setattr(commit_ingest, "MAX_INITIAL_COMMITS", 3)
        subjects = [f"c{i}" for i in range(6)]
        _make_repo(tmp_path / "repo", subjects, [_d(i) for i in range(6)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        ev = await collect_commit_evidence(sqlite_store)
        assert ev.acquired == 3
        assert _subjects(ev.lines[0]) == ["c3", "c4", "c5"]

    async def test_multiple_projects_each_get_their_own_batch(self, sqlite_store, tmp_path):
        _make_repo(tmp_path / "a", ["a1", "a2"], [_d(0), _d(1)])
        _make_repo(tmp_path / "b", ["b1"], [_d(0)])
        await _add_project(sqlite_store, "alpha", tmp_path / "a")
        await _add_project(sqlite_store, "beta", tmp_path / "b")
        ev = await collect_commit_evidence(sqlite_store)
        assert len(ev.lines) == 2 and len(ev.ids) == 3
        assert {ln.split(" in ")[1].split(":")[0] for ln in ev.lines} == {"alpha", "beta"}


class TestLegacyWatermark:
    async def test_old_watermark_seeds_the_cursor_once(self, sqlite_store, tmp_path):
        """An upgraded install must not re-queue commits the old collector
        already judged. The legacy key held newest-collected epoch + 1."""
        _make_repo(tmp_path / "repo", ["judged", "also judged", "new"], [_d(0), _d(500), _d(1000)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")
        await sqlite_store.set_metadata(WATERMARK_KEY, json.dumps({"blip": _d(501)}))

        ev = await collect_commit_evidence(sqlite_store)

        assert _subjects(ev.lines[0]) == ["new"]
        cursors = json.loads(await sqlite_store.get_metadata(CURSOR_KEY))
        assert cursors["blip"] == _d(1000)
        # legacy key is left alone, never advanced
        assert json.loads(await sqlite_store.get_metadata(WATERMARK_KEY)) == {"blip": _d(501)}


class TestRevisionIntegration:
    @staticmethod
    def _router(reply):
        from unittest.mock import AsyncMock, MagicMock
        r = MagicMock()
        r.generate = AsyncMock(return_value=reply) if not isinstance(reply, Exception) \
            else AsyncMock(side_effect=reply)
        return r

    async def test_commits_reach_the_revision_prompt_and_are_acknowledged(self, sqlite_store, tmp_path):
        from blipshell.memory.user_model import UserModel
        _make_repo(tmp_path / "repo", ["Rewrite consolidation with time budget"], [_d(0)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        router = self._router("- (medium) iterates on infrastructure")
        stats = await UserModel(sqlite_store, router).revise_from_reflections()

        assert stats["revised"] is True
        assert stats["commits"] == 1 and stats["commits_pending"] == 0
        prompt = router.generate.await_args.args[1]
        assert "(git)" in prompt and "Rewrite consolidation with time budget" in prompt
        assert await pending_commit_count(sqlite_store) == 0

    async def test_failed_revision_leaves_evidence_pending(self, sqlite_store, tmp_path):
        """The old collector stamped its watermark BEFORE the model ran, so a
        crash here silently discarded the commits."""
        from blipshell.memory.user_model import UserModel
        _make_repo(tmp_path / "repo", ["important work"], [_d(0)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        with pytest.raises(RuntimeError):
            await UserModel(sqlite_store, self._router(RuntimeError("model down"))).revise_from_reflections()
        assert await pending_commit_count(sqlite_store) == 1

        router = self._router("- (high) ships important work")
        stats = await UserModel(sqlite_store, router).revise_from_reflections()
        assert stats["revised"] is True and stats["commits"] == 1
        assert "important work" in router.generate.await_args.args[1], "re-fed evidence reached the prompt"
        assert await pending_commit_count(sqlite_store) == 0

    async def test_honest_empty_conclusion_still_consumes(self, sqlite_store, tmp_path):
        from blipshell.memory.user_model import EMPTY, UserModel
        _make_repo(tmp_path / "repo", ["trivial"], [_d(0)])
        await _add_project(sqlite_store, "blip", tmp_path / "repo")

        stats = await UserModel(sqlite_store, self._router(EMPTY)).revise_from_reflections()
        assert stats["revised"] is False and stats["reason"] == "model concluded nothing"
        assert await pending_commit_count(sqlite_store) == 0, "judged evidence must not be re-fed forever"

    async def test_no_evidence_reports_pending_count(self, sqlite_store):
        from blipshell.memory.user_model import UserModel
        stats = await UserModel(sqlite_store, self._router("x")).revise_from_reflections()
        assert stats == {"revised": False, "reason": "no new evidence", "commits": 0, "commits_pending": 0}
