"""Commits as user-model evidence: the input half of the loop.

The user model is revised from session reflections — the system's own
output about its own conversations. Compressing that loop raises density
but adds no material (review, 2026-08-10). Git history is the cheapest
EXTERNAL signal available: what the user actually shipped, in their own
words, timestamped by a tool that doesn't care what BlipShell thinks
happened. A week of commit subjects says more about working style than a
week of self-reflections.

Collection is bounded and skeptical: only projects whose root_path is a
real git repo, only commits since the last collection, capped per project,
one subprocess per repo with a timeout. Git being absent, a repo being
broken, or a log call hanging must never take the nightly job down — this
is garnish, not load-bearing.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

WATERMARK_KEY = "commit_ingest_watermarks"
MAX_SUBJECTS_PER_PROJECT = 10
GIT_TIMEOUT = 15.0


async def _git_log(repo: Path, since_epoch: Optional[int]) -> Optional[list[tuple[int, str]]]:
    """[(committer_epoch, subject)] in `repo` past the watermark, newest
    first. None = not a usable repo."""
    if not (repo / ".git").exists():
        return None
    cmd = ["git", "-C", str(repo), "log", "--pretty=%ct%x09%s",
           f"--max-count={MAX_SUBJECTS_PER_PROJECT}"]
    if since_epoch:
        cmd.append(f"--since=@{since_epoch}")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=GIT_TIMEOUT)
        if proc.returncode != 0:
            return None
        commits = []
        for line in out.decode("utf-8", "replace").splitlines():
            epoch, _, subject = line.partition("\t")
            if subject.strip() and epoch.isdigit():
                commits.append((int(epoch), subject.strip()))
        return commits
    except (FileNotFoundError, asyncio.TimeoutError, OSError) as e:
        logger.debug("git log failed for %s: %s", repo, e)
        return None


async def collect_commit_evidence(sqlite) -> list[str]:
    """One evidence line per project with new commits.

    Per-project watermarks, advanced to the newest COLLECTED commit's epoch
    plus one second — not "now". Two reasons, both learned the hard way this
    week: stamping now skips anything between the cap and the stamp, and git
    timestamps are whole seconds, so a same-second stamp re-collects the
    commits it just read (equality passes --since).
    """
    try:
        raw = await sqlite.get_metadata(WATERMARK_KEY)
        marks: dict = json.loads(raw) if raw else {}
        if not isinstance(marks, dict):
            marks = {}
        projects = await sqlite.list_projects()
    except Exception as e:
        logger.warning("Commit ingest could not read state: %s", e)
        return []

    evidence: list[str] = []
    changed = False
    for project in projects:
        name, root = project.get("name"), project.get("root_path")
        if not name or not root:
            continue
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        commits = await _git_log(root_path, marks.get(name))
        if not commits:
            continue
        evidence.append(
            f"(git) {len(commits)} recent commit(s) in {name}: "
            + "; ".join(subject for _, subject in commits)
        )
        marks[name] = max(epoch for epoch, _ in commits) + 1
        changed = True

    if changed:
        try:
            await sqlite.set_metadata(WATERMARK_KEY, json.dumps(marks))
        except Exception as e:
            logger.warning("Commit ingest could not stamp watermarks: %s", e)

    return evidence
