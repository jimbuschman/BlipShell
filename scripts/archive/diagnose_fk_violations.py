"""Find foreign-key orphans that would cause worker process_message to fail.

Run on the Ollama PC where the FK error fires:

    python scripts/diagnose_fk_violations.py

Reports every dangling reference touching the memory pipeline so we know
whether the FK culprit is memories.session_id, memory_tags.memory_id, etc.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import yaml


def main() -> int:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    db_path = cfg["database"]["path"]
    if not Path(db_path).is_absolute():
        db_path = str(cfg_path.parent / db_path)

    print(f"DB: {db_path}")
    if not Path(db_path).exists():
        print("DB does not exist")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    # PRAGMA foreign_key_check returns every existing FK violation in the DB.
    print("\n=== PRAGMA foreign_key_check ===")
    cur = conn.execute("PRAGMA foreign_key_check")
    violations = cur.fetchall()
    if not violations:
        print("  (no violations reported by PRAGMA)")
    else:
        for v in violations:
            # columns: table, rowid, parent, fkid
            print(f"  table={v[0]} rowid={v[1]} parent={v[2]} fkid={v[3]}")

    # Targeted checks for the tables process_message touches.
    print("\n=== memories with dangling session_id ===")
    cur = conn.execute("""
        SELECT m.id, m.session_id, m.role,
               COALESCE(m.is_processed, 1) AS is_processed,
               m.is_archived,
               substr(m.content, 1, 80) AS preview
          FROM memories m
     LEFT JOIN sessions s ON s.id = m.session_id
         WHERE m.session_id IS NOT NULL
           AND s.id IS NULL
         ORDER BY m.id DESC
         LIMIT 20
    """)
    rows = cur.fetchall()
    print(f"  count: {len(rows)} (showing up to 20)")
    for r in rows:
        print(f"  mem={r['id']} sess={r['session_id']} role={r['role']} "
              f"proc={r['is_processed']} arch={r['is_archived']} | {r['preview']!r}")

    cur = conn.execute("""
        SELECT COUNT(*) AS n FROM memories m
   LEFT JOIN sessions s ON s.id = m.session_id
       WHERE m.session_id IS NOT NULL AND s.id IS NULL
    """)
    print(f"  total orphaned memories: {cur.fetchone()['n']}")

    print("\n=== memory_tags pointing to missing memories ===")
    cur = conn.execute("""
        SELECT mt.memory_id, COUNT(*) AS n
          FROM memory_tags mt
     LEFT JOIN memories m ON m.id = mt.memory_id
         WHERE m.id IS NULL
      GROUP BY mt.memory_id
         LIMIT 20
    """)
    rows = cur.fetchall()
    print(f"  distinct orphaned memory_ids in memory_tags: {len(rows)} (showing up to 20)")
    for r in rows:
        print(f"  memory_id={r['memory_id']} tag_rows={r['n']}")

    print("\n=== memory_tags pointing to missing tags ===")
    cur = conn.execute("""
        SELECT mt.tag_id, COUNT(*) AS n
          FROM memory_tags mt
     LEFT JOIN tags t ON t.id = mt.tag_id
         WHERE t.id IS NULL
      GROUP BY mt.tag_id
         LIMIT 20
    """)
    rows = cur.fetchall()
    print(f"  distinct orphaned tag_ids in memory_tags: {len(rows)} (showing up to 20)")
    for r in rows:
        print(f"  tag_id={r['tag_id']} tag_rows={r['n']}")

    print("\n=== unprocessed memories (worker would pick these up) ===")
    try:
        cur = conn.execute("""
            SELECT m.id, m.session_id, m.role,
                   substr(m.content, 1, 60) AS preview,
                   (s.id IS NULL) AS sess_missing
              FROM memories m
         LEFT JOIN sessions s ON s.id = m.session_id
             WHERE COALESCE(m.is_processed, 1) = 0
             ORDER BY m.id DESC
             LIMIT 20
        """)
        rows = cur.fetchall()
        print(f"  count: {len(rows)} (showing up to 20)")
        for r in rows:
            warn = "  <-- SESSION MISSING" if r["sess_missing"] else ""
            print(f"  mem={r['id']} sess={r['session_id']} role={r['role']}"
                  f" | {r['preview']!r}{warn}")
    except sqlite3.OperationalError as e:
        print(f"  (skipped: {e})")

    print("\n=== sessions / memories totals ===")
    cur = conn.execute("SELECT COUNT(*) AS n FROM sessions")
    print(f"  sessions: {cur.fetchone()['n']}")
    cur = conn.execute("SELECT COUNT(*) AS n FROM memories")
    print(f"  memories: {cur.fetchone()['n']}")
    cur = conn.execute("SELECT COUNT(*) AS n FROM memory_tags")
    print(f"  memory_tags: {cur.fetchone()['n']}")

    print("\n=== last 5 sessions ===")
    cur = conn.execute("""
        SELECT id, title, project, message_count, created_at, last_active
          FROM sessions
      ORDER BY id DESC
         LIMIT 5
    """)
    for r in cur.fetchall():
        print(f"  sess={r['id']} msgs={r['message_count']} project={r['project']!r}"
              f" created={r['created_at']} title={r['title']!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
