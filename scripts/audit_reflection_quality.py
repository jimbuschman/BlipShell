"""Audit session-reflection quality by comparing chunked vs single-pass output.

Why this exists: after session_review moved to local qwen3:14b (2026-07-31,
kimi-k2.5:cloud retirement), sessions over ~28K tokens no longer fit one
context window. They go through chunk + merge instead, and a chunk is
reflected by a model that cannot see the other chunks. The benchmark harness
cannot detect the resulting damage -- its SESSION_REVIEW cases are 111-190
tokens, so `benchmark run --jobs session_review` only ever exercises the
single-pass path.

What this does instead is a natural experiment on real data. One nightly run
produces both chunked and single-pass reflections: same model, same corpus,
same night, differing only in whether chunking happened. So a failure
signature that shows up markedly more often in the chunked group is evidence
the chunking is at fault, with no judge model and no baseline required.

Three signatures, matching the three ways a whole-session prompt misjudges a
fragment:

  FABRICATED    "never addressed" / "unresolved" / "abandoned" language. A
                chunk ending on a question whose answer is in the next chunk
                invites an invented finding, which then merges into lessons
                and is later retrieved as fact. The chunk prompt forbids this
                outright, so ANY hit in the chunked group is a prompt miss.
  INEFFECTIVE   effectiveness == 'ineffective'. A middle chunk of hard
                debugging is all dead ends (resolution is in a later part), so
                a fragment judged as a session rates itself badly.
  SKIPPED       effectiveness == 'skipped'. A chunk of pure file reads is
                trivial AS A FRAGMENT; enough such chunks make
                process_reflection drop the entire session.

Read-only. Never writes to the database.

Usage:
    python -m scripts.audit_reflection_quality
    python -m scripts.audit_reflection_quality --since 2026-08-03
    python -m scripts.audit_reflection_quality --context 65536 --examples 3
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.memory.manager import estimate_tokens  # noqa: E402

# Language that asserts something did not happen. A single chunk cannot know
# this -- the evidence would be in a part it never saw.
FABRICATION_RE = re.compile(
    r"\b(never (?:addressed|answered|resolved|completed)"
    r"|un(?:addressed|answered|resolved)"
    r"|not (?:addressed|answered|resolved)"
    r"|left (?:open|unfinished|hanging)"
    r"|(?:was|were) abandoned"
    r"|remained? (?:open|unresolved|incomplete)"
    r"|no (?:answer|response) (?:was )?(?:given|provided))\b",
    re.IGNORECASE,
)

# Mirrors processor.prepare_conversation_for_reflection: reserve ~4K for the
# system prompt and the response, and never drop below half the window.
RESERVE_TOKENS = 4096


def reflection_input_tokens(conn: sqlite3.Connection, session_id: int) -> int:
    """Token count of the text the reflection actually saw.

    Rebuilt the same way prepare_conversation_for_reflection does it: every
    memory row for the session including archived ones, joined as
    "role: content".
    """
    rows = conn.execute(
        "SELECT role, content FROM memories WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    if not rows:
        return 0
    return estimate_tokens("\n".join(f"{r['role']}: {r['content']}" for r in rows))


def classify(row, was_chunked: bool) -> set:
    """Which failure signatures this reflection exhibits."""
    flags = set()
    effectiveness = (row["effectiveness"] or "").strip().lower()
    if effectiveness == "skipped":
        flags.add("SKIPPED")
        return flags  # a skipped row has no reflection body to inspect
    if effectiveness == "ineffective":
        flags.add("INEFFECTIVE")
    haystack = " ".join(
        str(row[c] or "") for c in ("what_didnt_work", "reflection_text")
    )
    if FABRICATION_RE.search(haystack):
        flags.add("FABRICATED")
    return flags


def pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):5.1f}%" if total else "    - "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="data/blipshell.db")
    ap.add_argument("--context", type=int, default=32768,
                    help="Context window session_review runs in (default 32768 "
                         "= the local endpoint). Sets the chunking threshold.")
    ap.add_argument("--since", default=None,
                    help="Only reflections created on/after this date "
                         "(YYYY-MM-DD). Use it to isolate post-migration runs "
                         "from the older single-pass cloud reflections.")
    ap.add_argument("--examples", type=int, default=2,
                    help="Flagged reflections to print per signature (0 = none)")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"ERROR: no database at {db}")
        return 1

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    max_tokens = max(args.context - RESERVE_TOKENS, args.context // 2)

    sql = "SELECT * FROM session_reflections"
    params: tuple = ()
    if args.since:
        sql += " WHERE created_at >= ?"
        params = (args.since,)
    sql += " ORDER BY session_id"

    groups = {"single-pass": [], "chunked": []}
    examples: dict = {}

    for row in conn.execute(sql, params):
        tokens = reflection_input_tokens(conn, row["session_id"])
        was_chunked = tokens > max_tokens
        group = "chunked" if was_chunked else "single-pass"
        flags = classify(row, was_chunked)
        est_chunks = max(1, -(-tokens // max_tokens)) if max_tokens else 1
        groups[group].append((row, flags, tokens, est_chunks))
        for f in flags:
            examples.setdefault((group, f), []).append((row, tokens, est_chunks))

    print(f"Reflection quality audit -- {db}")
    print(f"Context {args.context} tokens -> chunking above {max_tokens} tokens"
          + (f"   (since {args.since})" if args.since else ""))
    print()

    if not any(groups.values()):
        print("No reflections matched. If you just migrated, run the nightly")
        print("job first:  blipshell nightly --job session_reflections")
        return 0

    hdr = f"{'group':12} {'n':>5} {'FABRICATED':>12} {'INEFFECTIVE':>12} {'SKIPPED':>10}"
    print(hdr)
    print("-" * len(hdr))
    rates = {}
    for name in ("single-pass", "chunked"):
        rows = groups[name]
        n = len(rows)
        counts = {
            sig: sum(1 for _, flags, _, _ in rows if sig in flags)
            for sig in ("FABRICATED", "INEFFECTIVE", "SKIPPED")
        }
        rates[name] = {s: (counts[s] / n if n else 0.0) for s in counts}
        print(f"{name:12} {n:>5} "
              f"{counts['FABRICATED']:>5} {pct(counts['FABRICATED'], n)} "
              f"{counts['INEFFECTIVE']:>5} {pct(counts['INEFFECTIVE'], n)} "
              f"{counts['SKIPPED']:>3} {pct(counts['SKIPPED'], n)}")
    print()

    if groups["chunked"]:
        sizes = [c for _, _, _, c in groups["chunked"]]
        print(f"Chunked sessions: {len(sizes)}, "
              f"estimated {min(sizes)}-{max(sizes)} chunks each "
              f"({sum(sizes) + len(sizes)} LLM calls vs {len(sizes)} single-pass)")
        print()

    # The verdict. Only meaningful with chunked sessions to compare.
    print("Verdict")
    print("-" * 7)
    if not groups["chunked"]:
        print("  No chunked reflections in this set, so chunking is untested here.")
        print("  Either no session exceeded the threshold, or the migration has")
        print("  not run yet. Nothing to conclude.")
    elif not groups["single-pass"]:
        print("  No single-pass reflections to compare against -- rates below")
        print("  are absolute, not relative. Re-run without --since for a")
        print("  baseline, or treat any FABRICATED hit as a prompt failure.")
    else:
        verdict_clean = True
        for sig in ("FABRICATED", "INEFFECTIVE", "SKIPPED"):
            c, s = rates["chunked"][sig], rates["single-pass"][sig]
            if c > s * 1.5 and c > 0.05:
                verdict_clean = False
                print(f"  CONCERN {sig}: {c*100:.1f}% chunked vs {s*100:.1f}% "
                      f"single-pass. Chunking looks responsible.")
        if rates["chunked"]["FABRICATED"] > 0:
            verdict_clean = False
            print("  CONCERN FABRICATED is non-zero in the chunked group. The")
            print("  chunk prompt forbids unresolved claims outright, so any hit")
            print("  means the model is ignoring that instruction.")
        if verdict_clean:
            print("  CLEAN. Chunked reflections show no elevated failure rate")
            print("  and no fabricated unresolved claims. Chunking is holding up.")
    print()

    if args.examples:
        for (group, sig), rows in sorted(examples.items()):
            if group != "chunked" and sig != "FABRICATED":
                continue  # chunked is what's under test; show single-pass fabrication too
            print(f"[{group} / {sig}] {len(rows)} flagged, showing "
                  f"{min(args.examples, len(rows))}:")
            for row, tokens, est in rows[:args.examples]:
                print(f"  session {row['session_id']} "
                      f"({tokens} tokens, ~{est} chunks) "
                      f"effectiveness={row['effectiveness']}")
                body = (row["what_didnt_work"] or row["reflection_text"] or "")
                snippet = " ".join(body.split())[:300]
                print(f"    {snippet}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
