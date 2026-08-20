"""Forensics for the self-thought store -- the alive layer's readout input.

READ-ONLY. Safe to run against a live database while BlipShell is up.

Why this exists: `/thoughts` renders ACTIVE rows only (`get_self_thoughts`
filters `is_archived = 0`), so a corpus that collapsed into archive looks
identical to a corpus that was never written. On 2026-08-20 `/thoughts`
showed ONE thought where the 2026-07-09 readout had 24, and the survivor's
`echo_count` was 0 -- meaning it had not absorbed them, because
`_fold_duplicates` credits a winner one echo per fold. That distinction is
invisible from the active side alone.

This dumps the archived side with its provenance (`folded_into`,
`archived_at`) and cross-checks the fold ledger against `echo_count`, which
is how a fold that failed to credit its winner becomes visible at all.

The headline number is GENERATION RATE, not weight. Self-gravity step 2 is
gated on ">= 10 NEW thoughts", and a layer producing one thought a week
cannot supply that no matter how the weighting is tuned. Throughput is
upstream of gravity.

Usage:
    python scripts/diagnose_self_thoughts.py [--db PATH]

Exit codes: 0 clean, 1 anomalies found, 2 no self_thoughts table.
Output is ASCII-only (Windows console is cp1252).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
import sys
from datetime import datetime, timezone

DEFAULT_DB = "data/blipshell.db"

# _fold_duplicates() in core/self_reflection.py; kept here only to explain
# reconstructed weights in the report, never to recompute them.
RECUR_BOOST = 0.5
FATIGUE = 0.6


def _connect_readonly(path: str) -> sqlite3.Connection:
    """Open read-only so this can never mutate a live corpus.

    mode=ro still reads the WAL, which is correct on the machine that owns
    the file. Do NOT point this at a database over a network share.
    """
    db = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    db.row_factory = sqlite3.Row
    return db


def _has_table(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _parse_dt(value):
    """ISO -> aware datetime, or None. Undated rows are a finding, not a crash."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_days(value, now):
    dt = _parse_dt(value)
    if dt is None:
        return None
    return (now - dt).total_seconds() / 86400.0


def _fmt_age(days):
    if days is None:
        return "?"
    if days < 1:
        return "%dh" % int(days * 24)
    if days < 14:
        return "%dd" % int(days)
    return "%dw" % int(days / 7)


def section(title):
    print("")
    print("== %s " % title + "=" * max(0, 66 - len(title)))


def report_totals(db):
    total = db.execute("SELECT COUNT(*) FROM self_thoughts").fetchone()[0]
    active = db.execute(
        "SELECT COUNT(*) FROM self_thoughts WHERE is_archived = 0"
    ).fetchone()[0]
    section("TOTALS")
    print("rows in table      : %d" % total)
    print("active (/thoughts) : %d" % active)
    print("archived           : %d" % (total - active))
    if total and active == 0:
        print("NOTE: every thought is archived -- the layer looks empty but is not.")
    return total, active


def report_generation_rate(db, now):
    """The actual question: is the layer still producing thoughts?"""
    section("GENERATION RATE (by ISO week of created_at)")
    rows = db.execute(
        """SELECT strftime('%Y-W%W', created_at) AS wk, COUNT(*) AS n
           FROM self_thoughts
           WHERE created_at IS NOT NULL
           GROUP BY wk ORDER BY wk"""
    ).fetchall()
    if not rows:
        print("no dated rows at all -- created_at is NULL everywhere.")
        return
    for r in rows:
        print("  %-10s %s (%d)" % (r["wk"], "#" * min(40, r["n"]), r["n"]))

    undated = db.execute(
        "SELECT COUNT(*) FROM self_thoughts WHERE created_at IS NULL"
    ).fetchone()[0]
    if undated:
        print("  UNDATED: %d row(s) -- decay math is meaningless for these." % undated)

    newest = db.execute(
        "SELECT MAX(created_at) FROM self_thoughts"
    ).fetchone()[0]
    age = _age_days(newest, now)
    print("")
    print("newest thought : %s (%s ago)" % (newest, _fmt_age(age)))
    if age is not None and age > 7:
        print("STALL: nothing new in over a week. Reflection is not firing, or")
        print("       its output is being discarded before it lands.")


def report_archive_forensics(db, now):
    section("ARCHIVE FORENSICS")
    rows = db.execute(
        """SELECT is_archived,
                  CASE WHEN folded_into IS NOT NULL THEN 1 ELSE 0 END AS folded,
                  COUNT(*) AS n,
                  MIN(created_at) AS born_first, MAX(created_at) AS born_last,
                  MIN(archived_at) AS arch_first, MAX(archived_at) AS arch_last
           FROM self_thoughts GROUP BY 1, 2 ORDER BY 1, 2"""
    ).fetchall()
    for r in rows:
        kind = "ACTIVE" if not r["is_archived"] else (
            "ARCHIVED (folded)" if r["folded"] else "ARCHIVED (evicted)"
        )
        print("%-20s n=%-4d born %s .. %s" % (
            kind, r["n"], r["born_first"], r["born_last"]))
        if r["is_archived"]:
            print("%-20s      archived %s .. %s" % (
                "", r["arch_first"], r["arch_last"]))

    # One timestamp cluster => a single event (a fold run, a migration).
    # A spread => ongoing attrition. Different bugs entirely.
    days = db.execute(
        """SELECT substr(archived_at, 1, 10) AS d, COUNT(*) AS n
           FROM self_thoughts
           WHERE archived_at IS NOT NULL
           GROUP BY d ORDER BY d"""
    ).fetchall()
    if days:
        print("")
        print("archive events by day:")
        for r in days:
            print("  %s  n=%d" % (r["d"], r["n"]))
        if len(days) == 1:
            print("  -> single event: one run did this, not gradual attrition.")

    evicted = db.execute(
        """SELECT COUNT(*) FROM self_thoughts
           WHERE is_archived = 1 AND folded_into IS NULL"""
    ).fetchone()[0]
    if evicted:
        print("")
        print("%d archived row(s) have NO folded_into -- not folds." % evicted)
        print("That points at max_keep eviction (max_keep is 50) or a bulk")
        print("archive path, NOT duplicate-folding.")
    return evicted


def report_fold_ledger(db):
    """Cross-check absorbed-count against the winner's echo_count.

    _fold_duplicates does `winner.echo_count += loser.echo_count + 1` per
    fold, so a winner that absorbed N losers must carry at least N echoes.
    Less than that means the fold ran but the credit never persisted -- and
    since echo_count IS the gravity signal, a silent shortfall here is
    exactly what makes the step-1 readout unreadable.
    """
    section("FOLD LEDGER (absorbed vs credited)")
    rows = db.execute(
        """SELECT folded_into AS winner, COUNT(*) AS absorbed
           FROM self_thoughts
           WHERE folded_into IS NOT NULL
           GROUP BY folded_into ORDER BY absorbed DESC"""
    ).fetchall()
    if not rows:
        print("no folds recorded -- nothing was merged into anything.")
        return []

    anomalies = []
    print("%-8s %-9s %-8s %-9s %s" % (
        "winner", "absorbed", "echoes", "state", "verdict"))
    for r in rows:
        w = db.execute(
            """SELECT id, echo_count, is_archived, folded_into
               FROM self_thoughts WHERE id = ?""", (r["winner"],)
        ).fetchone()
        if w is None:
            print("%-8s %-9d %-8s %-9s ORPHAN: winner row missing" % (
                r["winner"], r["absorbed"], "-", "-"))
            anomalies.append("fold winner %s does not exist" % r["winner"])
            continue
        state = "archived" if w["is_archived"] else "active"
        if w["is_archived"] and w["folded_into"]:
            state = "chained"
        verdict = "ok"
        # A row folded into itself is corruption: it makes the winner its own
        # loser, so the fold can never be followed back to a live thought.
        if w["folded_into"] == w["id"]:
            verdict = "SELF-REFERENTIAL fold"
            anomalies.append("thought %d is folded into itself" % w["id"])
        elif w["echo_count"] < r["absorbed"]:
            verdict = "UNDER-CREDITED (echo < absorbed)"
            anomalies.append(
                "winner %d absorbed %d but carries %d echo(es)"
                % (w["id"], r["absorbed"], w["echo_count"]))
        print("%-8d %-9d %-8d %-9s %s" % (
            w["id"], r["absorbed"], w["echo_count"], state, verdict))
    return anomalies


def report_active_roster(db, now):
    section("ACTIVE ROSTER")
    rows = db.execute(
        """SELECT id, created_at, weight, surfaced, echo_count, surface_count,
                  embedding IS NOT NULL AS has_emb, text
           FROM self_thoughts WHERE is_archived = 0 ORDER BY id"""
    ).fetchall()
    if not rows:
        print("no active thoughts.")
        return []

    anomalies = []
    no_emb = 0
    print("%-5s %-6s %-7s %-5s %-5s %-4s %s" % (
        "id", "age", "weight", "echo", "surf", "vec", "text"))
    for r in rows:
        if not r["has_emb"]:
            no_emb += 1
        text = " ".join(str(r["text"]).split())
        print("%-5d %-6s %-7.2f %-5d %-5d %-4s %s" % (
            r["id"], _fmt_age(_age_days(r["created_at"], now)), r["weight"],
            r["echo_count"], r["surface_count"],
            "y" if r["has_emb"] else "NO", text[:58]))

    if no_emb:
        print("")
        print("%d active thought(s) have NO embedding." % no_emb)
        print("CONFOUND: echo detection is a cosine comparison, so these can")
        print("never recur and never resurface. A zero echo count on these")
        print("rows says nothing about the layer -- it is mechanical.")
        anomalies.append("%d active thought(s) missing embeddings" % no_emb)

    # A virgin thought sits at 1.0 * FATIGUE ** surface_count. Matching that
    # exactly means it never absorbed a fold, which is worth knowing when the
    # archive shows folds that should have credited someone.
    print("")
    for r in rows:
        expected = 1.0 * (FATIGUE ** r["surface_count"])
        if r["echo_count"] == 0 and abs(r["weight"] - expected) < 0.005:
            print("id %d: weight %.2f == 1.0 * %.1f^%d exactly -- never folded,"
                  % (r["id"], r["weight"], FATIGUE, r["surface_count"]))
            print("       never echoed. A fresh thought, not a merge survivor.")
    return anomalies


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def report_fold_candidates(db, threshold):
    """Pairwise cosine over ACTIVE embedded thoughts.

    Answers the question the fold ledger cannot: when near-identical rows sit
    unfolded, is folding BROKEN or is the threshold simply wrong for this
    corpus? Those need opposite responses, and the counts alone cannot tell
    them apart.

    A pair at or above `threshold` should have folded at write time
    (`add()` -> `_best_echo`) and did not -- that is a live defect. A cluster
    just below it means folding works and 0.85 is mistuned for a corpus this
    thematically narrow, where distinct-but-adjacent thoughts sit high.

    O(n^2) in embedded thoughts, which is fine at `max_keep = 50`.
    """
    section("FOLD CANDIDATES (pairwise cosine, active only)")
    rows = db.execute(
        """SELECT id, embedding, text FROM self_thoughts
           WHERE is_archived = 0 AND embedding IS NOT NULL ORDER BY id"""
    ).fetchall()
    if len(rows) < 2:
        print("fewer than 2 embedded active thoughts -- nothing to compare.")
        return []

    vecs = []
    for r in rows:
        blob = r["embedding"]
        try:
            vecs.append((r["id"], list(struct.unpack("%df" % (len(blob) // 4), blob)),
                         " ".join(str(r["text"]).split())))
        except struct.error:
            print("id %d: embedding blob is unreadable (wrong length?)" % r["id"])

    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            pairs.append((_cosine(vecs[i][1], vecs[j][1]), vecs[i][0], vecs[j][0]))
    pairs.sort(reverse=True)

    over = [p for p in pairs if p[0] >= threshold]
    print("compared %d thought(s), %d pair(s); threshold %.2f"
          % (len(vecs), len(pairs), threshold))
    print("")
    print("top 10 most-similar pairs:")
    texts = {v[0]: v[2] for v in vecs}
    for sim, a, b in pairs[:10]:
        flag = "  <-- SHOULD HAVE FOLDED" if sim >= threshold else ""
        print("  %.4f  id %-3d / id %-3d%s" % (sim, a, b, flag))
        print("          %s" % texts[a][:64])
        print("          %s" % texts[b][:64])

    anomalies = []
    print("")
    if over:
        print("%d pair(s) at or above %.2f are sitting UNFOLDED." % (len(over), threshold))
        print("Folding is failing on the write path -- not a threshold question.")
        anomalies.append("%d unfolded pair(s) >= %.2f" % (len(over), threshold))
    else:
        band = [p for p in pairs if threshold - 0.10 <= p[0] < threshold]
        print("no pair reaches %.2f, so nothing SHOULD have folded --" % threshold)
        print("folding is not broken; there was simply nothing to fold.")
        if band:
            print("%d pair(s) sit in the %.2f-%.2f band. If those read as the"
                  % (len(band), threshold - 0.10, threshold))
            print("same thought to you, the threshold is too high for this corpus.")
    return anomalies


def report_migration_backup(db):
    """Did the JSON-blob -> table migration lose anything?"""
    section("PRE-MIGRATION BACKUP")
    if not _has_table(db, "app_metadata"):
        print("no app_metadata table.")
        return []
    row = db.execute(
        "SELECT value FROM app_metadata WHERE key = 'self_thoughts_pre_migration'"
    ).fetchone()
    if row is None:
        print("no backup key -- migration either never ran or left no copy.")
        return []
    try:
        items = json.loads(row["value"])
    except (ValueError, TypeError) as exc:
        print("backup key present but unparseable: %s" % exc)
        return ["pre-migration backup is corrupt"]

    n = len(items) if isinstance(items, list) else 0
    total = db.execute("SELECT COUNT(*) FROM self_thoughts").fetchone()[0]
    print("backup holds : %d thought(s)" % n)
    print("table holds  : %d row(s)" % total)
    if total < n:
        print("LOSS: the table has FEWER rows than the pre-migration backup.")
        print("The backup is the recovery source.")
        return ["migration lost %d thought(s)" % (n - total)]
    print("no loss -- table count covers the backup.")
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB, help="path (default %s)" % DEFAULT_DB)
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="fold/recur cosine threshold; matches "
                         "reflection.gravity_recur_threshold (default 0.85)")
    args = ap.parse_args(argv)

    try:
        db = _connect_readonly(args.db)
    except sqlite3.OperationalError as exc:
        print("cannot open %s read-only: %s" % (args.db, exc))
        return 2

    print("self-thought forensics -- %s (read-only)" % args.db)
    now = datetime.now(timezone.utc)

    if not _has_table(db, "self_thoughts"):
        print("")
        print("No self_thoughts table. Either this database predates the")
        print("JSON-blob -> table migration, or it is not a BlipShell corpus.")
        return 2

    anomalies = []
    report_totals(db)
    report_generation_rate(db, now)
    evicted = report_archive_forensics(db, now)
    if evicted:
        anomalies.append("%d thought(s) archived without a fold" % evicted)
    anomalies += report_fold_ledger(db)
    anomalies += report_active_roster(db, now)
    anomalies += report_fold_candidates(db, args.threshold)
    anomalies += report_migration_backup(db)

    section("VERDICT")
    if anomalies:
        for a in anomalies:
            print("  - %s" % a)
    else:
        print("  no anomalies detected.")
    print("")
    print("Reminder: self-gravity step 2 is gated on >= 10 NEW thoughts.")
    print("Check GENERATION RATE first -- weighting cannot be judged on a")
    print("corpus the layer is not filling.")
    db.close()
    return 1 if anomalies else 0


if __name__ == "__main__":
    sys.exit(main())
