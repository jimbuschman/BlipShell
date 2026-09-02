"""Sweep the anti-pattern store for correction-detector false positives.

The correction detector minted permanent ANTI-PATTERN lessons from innocent
introspective questions ("so, why do you think you didn't see the mechanism?"
matched `you didn't` — seen live 2026-09-02). The detector now suppresses
weak signals inside introspective framings (core/guardrails.py); this script
re-judges every existing correction lesson under the corrected detector and
removes the ones that would no longer fire.

Each correction lesson stores the full `User said: "..."` text, so the
re-judgment runs on the original message, not the truncated signal. Lessons
whose content predates that format fall back to the Signal snippet and are
flagged as lower-confidence.

Dry-run by default (lists, changes nothing). --apply deletes the false
positives — deletion, not archival, following the clean_junk_lessons
precedent: lessons are derived data, recomputable from sessions, and a false
"the user corrected you" claim is harmful to keep even archived.

Usage (on the box holding the live DB):
    python -m scripts.sweep_correction_lessons          # list
    python -m scripts.sweep_correction_lessons --apply  # delete
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.core.guardrails import detect_correction  # noqa: E402

USER_SAID = re.compile(r'User said: "(.*)"\s*$', re.DOTALL)
SIGNAL = re.compile(r'Signal: "(.*?)"\.')


def judge(content: str) -> tuple[str, str] | None:
    """Re-judge one lesson. Returns (verdict, evidence_text) where verdict is
    'false_positive' or 'false_positive_lowconf', or None when the lesson
    still fires under the corrected detector (a real correction)."""
    m = USER_SAID.search(content)
    if m:
        text = m.group(1)
        return None if detect_correction(text) else ("false_positive", text)
    m = SIGNAL.search(content)
    if m:
        text = m.group(1)
        return None if detect_correction(text) else ("false_positive_lowconf", text)
    return None  # unrecognized format — leave it alone


async def run(apply: bool, db_override: str | None) -> int:
    from blipshell.core.config import ConfigManager
    from blipshell.memory.sqlite_store import SQLiteStore

    config = ConfigManager().load()
    db_path = db_override or config.database.path
    if not Path(db_path).exists():
        print(f"No database at {db_path}")
        return 1

    sqlite = SQLiteStore(db_path)
    await sqlite.initialize()
    try:
        rows = await sqlite._db.execute_fetchall(
            "SELECT id, content FROM lessons "
            "WHERE content LIKE 'ANTI-PATTERN: User corrected%'"
        )
        print(f"{len(rows)} correction lessons in the store")
        doomed: list[tuple[int, str, str]] = []
        for lid, content in rows:
            verdict = judge(content)
            if verdict:
                doomed.append((lid, verdict[0], verdict[1]))

        if not doomed:
            print("No false positives under the corrected detector.")
            return 0

        print(f"{len(doomed)} would no longer fire (false positives):")
        for lid, conf, text in doomed:
            tag = " (low-confidence: judged on truncated signal)" \
                if conf == "false_positive_lowconf" else ""
            safe = text[:100].encode("ascii", "replace").decode("ascii")
            print(f"  lesson {lid}{tag}: {safe}")

        if not apply:
            print("\nDry run - nothing changed. Re-run with --apply to delete.")
            return 0

        # Receipts before deletion: the full doomed rows land in a JSON file
        # next to the database, so the delete is reversible by hand even
        # though the lessons schema has no archive flag. (The SOURCE
        # conversations these lessons were minted from are untouched in the
        # memory store either way — only the derived claim is removed.)
        import json
        from datetime import datetime, timezone
        doomed_ids = [lid for lid, _, _ in doomed]
        full_rows = await sqlite._db.execute_fetchall(
            f"SELECT id, content, summary, timestamp, source_session_id, project "
            f"FROM lessons WHERE id IN ({','.join('?' for _ in doomed_ids)})",
            doomed_ids,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        receipt = Path(db_path).parent / f"lessons_swept_{stamp}.json"
        receipt.write_text(json.dumps(
            [dict(zip(("id", "content", "summary", "timestamp",
                       "source_session_id", "project"), r)) for r in full_rows],
            indent=1), encoding="utf-8")
        print(f"Receipts written: {receipt}")

        db = sqlite._db
        for lid, _, _ in doomed:
            await db.execute("DELETE FROM lesson_tags WHERE lesson_id = ?", (lid,))
            await db.execute("DELETE FROM lessons WHERE id = ?", (lid,))
        await db.commit()
        # Vectors swept after the SQL commit (same lock-contention care as
        # nightly's clean_junk_lessons).
        try:
            from blipshell.memory.vector_store import VectorStore
            from blipshell.models.config import get_ollama_url
            vectors = VectorStore(
                db_path=db_path,
                embedding_model=config.models.embedding,
                ollama_url=get_ollama_url(config.endpoints),
                embedding_dim=config.database.embedding_dimensions,
            )
            vectors.initialize()
            for lid, _, _ in doomed:
                try:
                    vectors.delete_lesson(lid)
                except Exception:
                    pass  # orphan vectors are swept by nightly maintenance
        except Exception as e:
            print(f"(vector sweep skipped: {e} - nightly will catch orphans)")
        print(f"Deleted {len(doomed)} false-positive lessons.")
        return 0
    finally:
        await sqlite.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="delete the false positives (default: list only)")
    parser.add_argument("--db", help="database path (default: config.yaml's)")
    args = parser.parse_args()
    return asyncio.run(run(args.apply, args.db))


if __name__ == "__main__":
    sys.exit(main())
