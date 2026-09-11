"""Recorder and scorer for the conversation-continuity probe
(docs/CONTINUITY_PROBE.md - frozen protocol).

    python -m scripts.continuity_probe score --thread "<last assistant turn>" [--note "<note>"] --reply "<reply>"
    python -m scripts.continuity_probe record --arm C --variant live --prior-session 1930 --next-session 1931 \
        --ended normally --reply-file reply.txt --thread-file thread.txt [--note-file note.txt] [--operator yes]

`record` appends one row (numbers, clause booleans, ids, arm, ending,
operator verdict) to benchmark_results/continuity_probe__v1.jsonl and keeps
the texts under data/continuity_probe/ (gitignored: real replies are
personal). Nothing here calls a model.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from blipshell.core.claim_check import content_words

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmark_results" / "continuity_probe__v1.jsonl"
TEXTS = REPO / "data" / "continuity_probe"
_DISCLAIM = re.compile(r"\b(i don'?t have|no record|i don'?t remember|i can'?t recall|i do not have|nothing on file)\b", re.I)


def score(reply: str, thread: str, note: str | None = None) -> dict:
    rw = content_words(reply)
    thread_hits = len(rw & content_words(thread))
    note_hits = len(rw & content_words(note)) if note else None
    return {
        "thread_words_shared": thread_hits,
        "picks_up_thread": thread_hits >= 2,
        "note_words_shared": note_hits,
        "picks_up_note": (note_hits >= 2) if note else None,
        "disclaims": bool(_DISCLAIM.search(reply or "")),
    }


def _read(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8") if path else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--reply", required=True); sc.add_argument("--thread", required=True); sc.add_argument("--note")
    rc = sub.add_parser("record")
    rc.add_argument("--arm", required=True, choices=["A", "B", "C"])
    rc.add_argument("--variant", required=True, choices=["live", "synthetic"])
    rc.add_argument("--prior-session", type=int, required=True); rc.add_argument("--next-session", type=int, required=True)
    rc.add_argument("--ended", required=True, choices=["normally", "abnormally"])
    rc.add_argument("--reply-file", required=True); rc.add_argument("--thread-file", required=True); rc.add_argument("--note-file")
    rc.add_argument("--operator", choices=["yes", "no", "unsure"], default="unsure")
    rc.add_argument("--infra-error", default="", help="records the pair as invalid with this reason")
    args = ap.parse_args(argv)

    if args.cmd == "score":
        print(json.dumps(score(args.reply, args.thread, args.note), indent=2))
        return 0

    reply, thread, note = _read(args.reply_file), _read(args.thread_file), _read(args.note_file)
    row = {
        "protocol": "v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm, "variant": args.variant, "prior_session": args.prior_session,
        "next_session": args.next_session, "ended": args.ended, "operator": args.operator,
        "valid": not args.infra_error, "infra_error": args.infra_error or None,
        **(score(reply, thread, note) if not args.infra_error else {}),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    TEXTS.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_{args.arm}_{args.prior_session}_{args.next_session}"
    (TEXTS / f"{stem}.json").write_text(json.dumps({"reply": reply, "thread": thread, "note": note}, indent=2),
                                        encoding="utf-8")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
