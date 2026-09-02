"""Consolidate near-duplicate lesson families. Reviewed-by-Claude 2026-09-02.

All 127 multi-row families (456 lessons) were read before this was written.
Policy: within each family keep ONE member (highest importance, then longest
content, then newest) and delete the rest (rows, tags, vectors), receipts
written next to the database. Six human-judged FALSE families are skipped.
No content is rewritten, so surviving vectors stay valid.

Usage:
    python consolidate_lessons.py [path\to\blipshell.db]
Default DB path: the dev-box Downloads copy. Run from anywhere; the BlipShell
repo is auto-located (needed for blipshell.memory.themes).
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Locate the BlipShell repo for the themes import: cwd first, then known spots.
for candidate in [Path.cwd(),
                  Path(r"C:\Users\[user]\source\repos\jimbuschman\BlipShell"),
                  Path(r"C:\Users\JimBu\source\repos\blipshell")]:
    if (candidate / "blipshell" / "memory" / "themes.py").exists():
        sys.path.insert(0, str(candidate))
        break
else:
    sys.exit("Could not find the BlipShell repo (run from inside it, and "
             "make sure it's pulled to a commit that has memory/themes.py).")

import sqlite_vec  # noqa: E402
from blipshell.memory.themes import family_sizes  # noqa: E402

DEFAULT_DB = r"C:\Users\[user]\Downloads\blipshell.db"
DB = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
if not Path(DB).exists():
    sys.exit(f"No database at {DB} — pass the path as an argument.")
RECEIPTS = Path(DB).parent / "lesson_consolidation_receipts_2026-09-02.json"

# Human-reviewed false families: members stay untouched.
SKIP_SETS = [
    {97, 322, 1181, 1194},   # memory-systems expertise vs mental-health framing
    {24, 256},               # generic confirmation-seeking vs cat health
    {36, 1197},              # weightlifting routine vs coding preferences
    {185, 886},              # hardware clarifying vs fitness reassurance
    {196, 863},              # equipment pain-points vs spending reassurance
    {353, 726},              # buy-vs-build vs concept exploration
]
SKIP_IDS = set().union(*SKIP_SETS)

con = sqlite3.connect(DB)
con.enable_load_extension(True)
sqlite_vec.load(con)
con.row_factory = sqlite3.Row

rows = con.execute(
    "SELECT id, importance, timestamp, content FROM lessons ORDER BY id").fetchall()
ids = [r["id"] for r in rows]
by_id = {r["id"]: r for r in rows}
fam, _ = family_sizes([r["content"] or "" for r in rows], link="representative")

from collections import defaultdict  # noqa: E402
groups = defaultdict(list)
for i, f in enumerate(fam):
    groups[f].append(ids[i])

doomed: list[int] = []
kept: list[dict] = []
skipped_families = 0
for members in groups.values():
    if len(members) < 2:
        continue
    if set(members) & SKIP_IDS:
        skipped_families += 1
        continue
    keeper = max(members, key=lambda lid: (
        by_id[lid]["importance"] or 0,
        len(by_id[lid]["content"] or ""),
        lid,
    ))
    kept.append({"keeper": keeper,
                 "deleted": [m for m in members if m != keeper]})
    doomed.extend(m for m in members if m != keeper)

if not doomed:
    print("Nothing to consolidate (already ran?).")
    sys.exit(0)

receipt_rows = con.execute(
    f"SELECT id, content, summary, timestamp, source_session_id, project "
    f"FROM lessons WHERE id IN ({','.join('?' * len(doomed))})",
    doomed).fetchall()
receipts = {
    "applied_at": datetime.now(timezone.utc).isoformat(),
    "policy": "keep max(importance, content length, id) per family; "
              "6 human-judged false families skipped",
    "skipped_family_ids": [sorted(s) for s in SKIP_SETS],
    "families": kept,
    "deleted_lessons": [dict(r) for r in receipt_rows],
}

for lid in doomed:
    con.execute("DELETE FROM lesson_tags WHERE lesson_id = ?", (lid,))
    con.execute("DELETE FROM lessons WHERE id = ?", (lid,))
    con.execute("DELETE FROM vec_lessons WHERE rowid = ?", (lid,))
con.commit()

print(f"consolidated {len(kept)} families, deleted {len(doomed)} lessons, "
      f"skipped {skipped_families} false families")
print("lessons remaining:",
      con.execute("SELECT COUNT(*) FROM lessons").fetchone()[0])
print("vec rows:",
      con.execute("SELECT COUNT(*) FROM vec_lessons_rowids").fetchone()[0])
print("integrity:", con.execute("PRAGMA integrity_check").fetchone()[0])
con.close()

RECEIPTS.write_text(json.dumps(receipts, indent=1, ensure_ascii=False),
                    encoding="utf-8")
print(f"receipts: {RECEIPTS}")
