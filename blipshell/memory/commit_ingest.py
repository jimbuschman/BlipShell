"""Commits as user-model evidence: the input half of the loop.

The user model is revised from session reflections — the system's own
output about its own conversations. Compressing that loop raises density
but adds no material (review, 2026-08-10). Git history is the cheapest
EXTERNAL signal available: what the user actually shipped, in their own
words, timestamped by a tool that doesn't care what BlipShell thinks
happened. A week of commit subjects says more about working style than a
week of self-reflections.

Collection is bounded and skeptical: only projects whose root_path is a
real git repo, one subprocess per repo with a timeout. Git being absent, a
repo being broken, or a log call hanging must never take the nightly job
down — this is garnish, not load-bearing.

Two positions, kept separate (V3_PLAN A4, review F7):

- **Acquisition** (`acquire_commits`): per-project cursor = the newest
  acquired commit's epoch. `git log --since=@cursor` (inclusive) with NO
  count cap, into the `commit_evidence` table, unique on (repo_root, sha).
  Same-second commits are no longer a problem because the hash is the key,
  not the timestamp; re-reading the cursor second costs an ignored insert.
  A never-seen project takes its newest MAX_INITIAL_COMMITS only — older
  history predates tracking and is not "recent activity".
- **Revision** (`collect_commit_evidence` / `acknowledge_commit_evidence`):
  pending rows are drained oldest-first, MAX_SUBJECTS_PER_PROJECT per project
  per revision, and marked consumed ONLY after the caller has persisted the
  derived update. A failed revision leaves its evidence pending.

Before this the collector ran `--max-count=10` newest-first and stamped the
watermark past the newest collected commit — twelve commits produced ten,
then zero, and the oldest two were excluded forever — and it stamped before
the revision model ran, so a failed revision consumed its evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Legacy (pre-A4) per-project watermarks: newest collected epoch + 1. Read
# ONCE as the initial acquisition cursor so an upgraded install does not
# re-queue the commits it already judged; never written again.
WATERMARK_KEY = "commit_ingest_watermarks"
CURSOR_KEY = "commit_ingest_cursors"

MAX_SUBJECTS_PER_PROJECT = 10   # drained per project per revision
MAX_INITIAL_COMMITS = 50        # first acquisition for a never-seen project
GIT_TIMEOUT = 15.0


@dataclass
class CommitEvidence:
    """What one revision gets to see, and how to acknowledge it."""
    lines: list[str] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)
    acquired: int = 0        # rows queued by this call's acquisition pass
    pending_after: int = 0   # rows still pending once these `ids` are consumed


async def _git_log(
    repo: Path, since_epoch: Optional[int], max_count: Optional[int],
) -> Optional[list[tuple[str, int, str]]]:
    """[(sha, committer_epoch, subject)] newest first. None = not a usable repo."""
    if not (repo / ".git").exists():
        return None
    cmd = ["git", "-C", str(repo), "log", "--pretty=%H%x09%ct%x09%s"]
    if max_count:
        cmd.append(f"--max-count={max_count}")
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
            sha, _, rest = line.partition("\t")
            epoch, _, subject = rest.partition("\t")
            if sha and epoch.isdigit() and subject.strip():
                commits.append((sha, int(epoch), subject.strip()))
        return commits
    except (FileNotFoundError, asyncio.TimeoutError, OSError) as e:
        logger.debug("git log failed for %s: %s", repo, e)
        return None


async def _load_json_map(sqlite, key: str) -> dict:
    raw = await sqlite.get_metadata(key)
    try:
        obj = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        obj = {}
    return obj if isinstance(obj, dict) else {}


async def acquire_commits(sqlite) -> int:
    """Queue new commits from every project repo. Returns rows inserted.

    Never raises for git problems; a state read/write failure is logged and
    counts as nothing acquired.
    """
    try:
        cursors = await _load_json_map(sqlite, CURSOR_KEY)
        legacy = await _load_json_map(sqlite, WATERMARK_KEY)
        projects = await sqlite.list_projects()
    except Exception as e:
        logger.warning("Commit ingest could not read state: %s", e)
        return 0

    inserted = 0
    changed = False
    for project in projects:
        name, root = project.get("name"), project.get("root_path")
        if not name or not root:
            continue
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        cursor = cursors.get(name)
        if cursor is None and isinstance(legacy.get(name), int):
            cursor = legacy[name]  # epoch+1 under the old scheme: "anything from here on"
        commits = await _git_log(root_path, cursor, None if cursor else MAX_INITIAL_COMMITS)
        if not commits:
            continue
        for sha, epoch, subject in commits:
            try:
                cur = await sqlite._db.execute(
                    "INSERT OR IGNORE INTO commit_evidence (project, repo_root, sha, epoch, subject) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (name, str(root_path), sha, epoch, subject),
                )
                inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            except Exception as e:
                logger.warning("Commit ingest could not queue %s in %s: %s", sha[:8], name, e)
        cursors[name] = max(epoch for _, epoch, _ in commits)
        changed = True

    try:
        if inserted:
            await sqlite._db.commit()
        if changed:
            await sqlite.set_metadata(CURSOR_KEY, json.dumps(cursors))
    except Exception as e:
        logger.warning("Commit ingest could not persist state: %s", e)
    return inserted


async def pending_commit_count(sqlite) -> int:
    cur = await sqlite._db.execute("SELECT COUNT(*) FROM commit_evidence WHERE status = 'pending'")
    row = await cur.fetchone()
    return int(row[0]) if row else 0


async def collect_commit_evidence(
    sqlite, per_project: int = MAX_SUBJECTS_PER_PROJECT,
) -> CommitEvidence:
    """Acquire, then drain: one evidence line per project with pending commits.

    Oldest pending first, `per_project` per project. Nothing is marked
    consumed here — call `acknowledge_commit_evidence(ids)` after the derived
    update is persisted, or the same rows come back next time (which is the
    point).
    """
    try:
        acquired = await acquire_commits(sqlite)
    except Exception as e:  # belt and braces: acquisition is best-effort
        logger.warning("Commit acquisition failed: %s", e)
        acquired = 0

    try:
        cur = await sqlite._db.execute(
            "SELECT id, project, subject FROM commit_evidence WHERE status = 'pending' "
            "ORDER BY project, epoch, id",
        )
        rows = await cur.fetchall()
    except Exception as e:
        logger.warning("Commit ingest could not read the queue: %s", e)
        return CommitEvidence(acquired=acquired)

    by_project: dict[str, list[tuple[int, str]]] = {}
    for row_id, project, subject in rows:
        by_project.setdefault(project, []).append((int(row_id), subject))

    lines: list[str] = []
    ids: list[int] = []
    for project, items in by_project.items():
        batch = items[:per_project]
        lines.append(
            f"(git) {len(batch)} recent commit(s) in {project}: "
            + "; ".join(subject for _, subject in batch)
        )
        ids.extend(row_id for row_id, _ in batch)

    return CommitEvidence(
        lines=lines, ids=ids, acquired=acquired,
        pending_after=len(rows) - len(ids),
    )


async def acknowledge_commit_evidence(sqlite, ids: list[int]) -> int:
    """Mark drained rows consumed. Call only after the derived update persisted."""
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    cur = await sqlite._db.execute(
        f"UPDATE commit_evidence SET status = 'consumed', consumed_at = CURRENT_TIMESTAMP "
        f"WHERE status = 'pending' AND id IN ({placeholders})",
        list(ids),
    )
    await sqlite._db.commit()
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
