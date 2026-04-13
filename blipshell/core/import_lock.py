"""File-based lock for long-running imports.

Used by the import CLI commands to signal that a heavy operation is in
progress, so the nightly job runner can skip itself rather than fight
for the SQLite write lock and create orphan vectors.

Lock file location: <db_dir>/import.lock
Format: JSON with start time, pid, and operation name.
A stale lock (older than max_age_hours) is treated as released so a
crashed import doesn't block nightly runs forever.
"""

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Treat lock files older than this as stale (crashed process)
_STALE_AFTER_HOURS = 12


def _lock_path(db_path: str) -> Path:
    return Path(db_path).parent / "import.lock"


def is_import_active(db_path: str, max_age_hours: float = _STALE_AFTER_HOURS) -> bool:
    """Return True if a non-stale import lock file exists."""
    lock = _lock_path(db_path)
    if not lock.exists():
        return False

    try:
        age_s = time.time() - lock.stat().st_mtime
        if age_s > max_age_hours * 3600:
            logger.warning(
                "Import lock %s is %.1fh old — treating as stale",
                lock, age_s / 3600,
            )
            return False
    except OSError:
        return False

    return True


def read_lock_info(db_path: str) -> Optional[dict]:
    """Return the lock metadata if a lock exists, else None."""
    lock = _lock_path(db_path)
    if not lock.exists():
        return None
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@contextmanager
def import_lock(db_path: str, operation: str = "import"):
    """Context manager that creates/removes the import lock file.

    Multiple imports can technically coexist (the file is overwritten),
    but the goal is just to signal "something heavy is running" to the
    nightly runner. The import lock auto-cleans on context exit even
    if an exception occurs.
    """
    lock = _lock_path(db_path)
    lock.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "operation": operation,
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    try:
        lock.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write import lock %s: %s", lock, e)

    try:
        yield
    finally:
        try:
            if lock.exists():
                lock.unlink()
        except OSError as e:
            logger.warning("Could not remove import lock %s: %s", lock, e)
