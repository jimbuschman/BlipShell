"""Verify ChatGPT export completeness against the SQLite database.

Parses conversations.json using the codebase parser, then compares
each conversation against sessions/memories in the DB.
"""

import importlib
import io
import json
import sqlite3
import sys
import types
from pathlib import Path

# Force UTF-8 stdout to handle Unicode conversation titles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path so we can import blipshell modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We need parse_conversations from import_chatgpt, but that module
# imports heavy dependencies (chromadb, etc.) at module level.
# Instead, we create stub modules for the heavy deps and only import
# what we actually need: the parser + noise filter.

# Import noise first (before we stub anything), since it has no heavy deps
from blipshell.memory.noise import should_skip_memory

# Now stub the heavy dependencies that import_chatgpt needs but we don't
# We must NOT overwrite blipshell, blipshell.memory, or blipshell.memory.noise
# since those are already loaded.
_stub_modules = [
    "blipshell.core",
    "blipshell.core.config",
    "blipshell.llm",
    "blipshell.llm.prompts",
    "blipshell.llm.router",
    "blipshell.memory.chroma_store",
    "blipshell.memory.processor",
    "blipshell.memory.sqlite_store",
    "blipshell.memory.tagger",
    "blipshell.models",
    "blipshell.models.config",
    "blipshell.models.memory",
]
for mod_name in _stub_modules:
    if mod_name in sys.modules:
        continue  # don't overwrite already-loaded real modules
    stub = types.ModuleType(mod_name)
    # Add common attributes that import_chatgpt references at import time
    for attr in [
        "ConfigManager", "ask_importance", "rank_memory", "summarize_memory",
        "LLMRouter", "TaskType", "ChromaStore", "MemoryProcessor",
        "tag_message", "MemoryConfig", "Memory", "MemoryType", "SQLiteStore",
    ]:
        setattr(stub, attr, None)
    sys.modules[mod_name] = stub

from blipshell.import_chatgpt import parse_conversations

# Paths
EXPORT_PATH = Path(r"C:\Users\[user]\Downloads\conversations.json")
DB_PATH = Path(r"C:\Users\[user]\Downloads\blipshell.db")


def main():
    # ── 1. Parse the ChatGPT export ──────────────────────────────────────
    print(f"Parsing export: {EXPORT_PATH}")
    conversations = parse_conversations(EXPORT_PATH)
    print(f"  Parsed {len(conversations)} conversations with messages\n")

    # Also count the raw conversations (including those with no messages)
    with open(EXPORT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw conversations in JSON: {len(raw)}")
    empty_convs = len(raw) - len(conversations)
    print(f"  Conversations with no parseable messages (skipped by parser): {empty_convs}\n")

    # ── 2. Connect to the DB ─────────────────────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Get all sessions
    cur.execute("SELECT id, title, message_count FROM sessions")
    db_sessions = cur.fetchall()
    db_by_title = {}
    for row in db_sessions:
        title = row["title"]
        # Handle duplicate titles by keeping them all in a list
        if title not in db_by_title:
            db_by_title[title] = []
        db_by_title[title].append(dict(row))

    total_db_sessions = len(db_sessions)
    complete_db = sum(1 for r in db_sessions if r["message_count"] > 0)
    incomplete_db = sum(1 for r in db_sessions if r["message_count"] == 0)

    print(f"DB sessions: {total_db_sessions} total, {complete_db} complete (msg_count>0), {incomplete_db} incomplete (msg_count=0)\n")

    # ── 3. Match and compare ─────────────────────────────────────────────
    matched = 0
    unmatched_export = []  # in export but not DB
    count_mismatches = []
    suspicious_defaults = []

    for conv in conversations:
        title = conv.title

        # Count non-noise messages in the export
        non_noise = 0
        for msg in conv.messages:
            if not should_skip_memory(msg.content):
                non_noise += 1

        if title not in db_by_title:
            unmatched_export.append((title, len(conv.messages), non_noise))
            continue

        # Use the first matching session (or the one with message_count > 0)
        sessions = db_by_title[title]
        best = None
        for s in sessions:
            if s["message_count"] > 0:
                best = s
                break
        if best is None:
            best = sessions[0]

        matched += 1
        session_id = best["id"]
        db_msg_count = best["message_count"]

        # Actual memory rows in DB for this session
        cur.execute("SELECT COUNT(*) FROM memories WHERE session_id = ?", (session_id,))
        actual_mem_count = cur.fetchone()[0]

        # Check for mismatches. The import pipeline also filters via
        # summarization (SKIP responses), so db count can be <= non_noise.
        # Flag if DB count is 0 (incomplete) or actual row count differs
        # from the session message_count.
        issues = []
        if db_msg_count == 0:
            issues.append(f"session.message_count=0 (incomplete import)")
        if actual_mem_count != db_msg_count:
            issues.append(f"memory rows ({actual_mem_count}) != session.message_count ({db_msg_count})")
        if actual_mem_count > non_noise:
            issues.append(f"memory rows ({actual_mem_count}) > non-noise export msgs ({non_noise})")

        if issues:
            count_mismatches.append((title, non_noise, db_msg_count, actual_mem_count, issues))

        # Check for suspicious default values in memories
        cur.execute("""
            SELECT id, role,
                   COALESCE(summary, '') as summary,
                   COALESCE(rank, 0) as rank,
                   COALESCE(importance, 0.0) as importance
            FROM memories
            WHERE session_id = ?
        """, (session_id,))
        memories = cur.fetchall()

        missing_summary = 0
        zero_rank = 0
        zero_importance = 0
        for mem in memories:
            if not mem["summary"] or mem["summary"].strip() == "":
                missing_summary += 1
            if mem["rank"] == 0:
                zero_rank += 1
            if mem["importance"] == 0.0:
                zero_importance += 1

        if missing_summary > 0 or zero_rank > 0 or zero_importance > 0:
            suspicious_defaults.append((
                title, session_id, len(memories),
                missing_summary, zero_rank, zero_importance
            ))

    # ── 4. Check for DB sessions not in the export ───────────────────────
    export_titles = {c.title for c in conversations}
    db_not_in_export = []
    for title, sessions in db_by_title.items():
        if title not in export_titles:
            for s in sessions:
                db_not_in_export.append((title, s["id"], s["message_count"]))

    # ── 5. Print summary report ──────────────────────────────────────────
    print("=" * 80)
    print("VERIFICATION SUMMARY REPORT")
    print("=" * 80)

    print(f"\n--- Counts ---")
    print(f"  Conversations in export (with messages): {len(conversations)}")
    print(f"  Raw conversations in export JSON:        {len(raw)}")
    print(f"  Sessions in DB:                          {total_db_sessions}")
    print(f"    Complete (message_count > 0):           {complete_db}")
    print(f"    Incomplete (message_count = 0):         {incomplete_db}")
    print(f"  Matched by title:                        {matched}")
    print(f"  In export but NOT in DB:                 {len(unmatched_export)}")
    print(f"  In DB but NOT in export:                 {len(db_not_in_export)}")

    print(f"\n--- Count Mismatches ({len(count_mismatches)}) ---")
    if count_mismatches:
        for title, non_noise, db_count, actual_count, issues in count_mismatches:
            print(f"  [{title[:60]}]")
            print(f"    export non-noise={non_noise}, session.message_count={db_count}, actual memory rows={actual_count}")
            for issue in issues:
                print(f"    ** {issue}")
    else:
        print("  None - all matched sessions have consistent counts.")

    print(f"\n--- Suspicious Defaults ({len(suspicious_defaults)} sessions) ---")
    if suspicious_defaults:
        total_missing_summary = sum(s[3] for s in suspicious_defaults)
        total_zero_rank = sum(s[4] for s in suspicious_defaults)
        total_zero_importance = sum(s[5] for s in suspicious_defaults)
        print(f"  Total memories with missing summary:   {total_missing_summary}")
        print(f"  Total memories with rank=0:            {total_zero_rank}")
        print(f"  Total memories with importance=0.0:    {total_zero_importance}")
        print()
        for title, sid, mem_count, ms, zr, zi in suspicious_defaults:
            print(f"  [{title[:60]}] (session_id={sid}, {mem_count} mems)")
            if ms > 0:
                print(f"    missing summary: {ms}")
            if zr > 0:
                print(f"    rank=0: {zr}")
            if zi > 0:
                print(f"    importance=0.0: {zi}")
    else:
        print("  None - all memories have non-default summary/rank/importance.")

    if unmatched_export:
        print(f"\n--- Conversations in Export but NOT in DB ({len(unmatched_export)}) ---")
        for title, total_msgs, non_noise in unmatched_export:
            print(f"  [{title[:70]}] total_msgs={total_msgs}, non_noise={non_noise}")

    if db_not_in_export:
        print(f"\n--- Sessions in DB but NOT in Export ({len(db_not_in_export)}) ---")
        for title, sid, mc in db_not_in_export:
            print(f"  [{title[:70]}] session_id={sid}, message_count={mc}")

    # ── 6. Duplicate title analysis ──────────────────────────────────────
    dup_titles = {t: ss for t, ss in db_by_title.items() if len(ss) > 1}
    if dup_titles:
        print(f"\n--- Duplicate Titles in DB ({len(dup_titles)} titles) ---")
        for title, sessions in dup_titles.items():
            ids = [(s["id"], s["message_count"]) for s in sessions]
            print(f"  [{title[:60]}] -> {ids}")

    # Export title duplicates
    from collections import Counter
    export_title_counts = Counter(c.title for c in conversations)
    export_dups = {t: c for t, c in export_title_counts.items() if c > 1}
    if export_dups:
        print(f"\n--- Duplicate Titles in Export ({len(export_dups)} titles) ---")
        for title, count in sorted(export_dups.items(), key=lambda x: -x[1]):
            print(f"  [{title[:60]}] x{count}")

    conn.close()
    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
