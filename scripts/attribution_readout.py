"""Correction-attribution readout (V3 D2a, phase 1). Read-only by default.

Shows every recorded correction with the judge's attribution beside it, the
per-lesson attribution counts, and - once human labels exist - the judge's
agreement per class and its false-positive rate on `lesson_wrong`, which is
the gate for phase 2 (agreement >= 0.8, FP <= 0.1, >= 10-15 genuine
lesson_wrong positives in the labelled set, repeat spread < 0.1).

Usage (from repo root):
    python -m scripts.attribution_readout                 # rows + counts + agreement
    python -m scripts.attribution_readout --db PATH
    python -m scripts.attribution_readout --label 12 lesson_wrong 34
        # human label for correction 12: attribution, and lesson id (omit for unrelated)

Everything but --label opens the database read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.memory.attribution import ATTRIBUTIONS  # noqa: E402

MIN_POSITIVES = 10


def _connect(db_path: str, readonly: bool = True) -> sqlite3.Connection:
    if readonly:
        uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
    else:
        con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def load_rows(con) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM corrections ORDER BY id")]


def agreement(rows: list[dict]) -> dict:
    """Judge vs human, over rows that carry a human label."""
    labelled = [r for r in rows if r.get("human_attribution")]
    out = {"labelled": len(labelled), "positives_lesson_wrong": 0, "per_class": {},
           "lesson_wrong_false_positive_rate": None, "meaningful": False}
    if not labelled:
        return out
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in labelled:
        by_class[r["human_attribution"]].append(r)
    for cls, rs in by_class.items():
        agree = sum(1 for r in rs if r["attribution"] == cls
                    and (cls == "unrelated" or r.get("lesson_id") == r.get("human_lesson_id")))
        out["per_class"][cls] = {"n": len(rs), "agreement": round(agree / len(rs), 3)}
    out["positives_lesson_wrong"] = len(by_class.get("lesson_wrong", []))
    judged_wrong = [r for r in labelled if r["attribution"] == "lesson_wrong"]
    if judged_wrong:
        fp = sum(1 for r in judged_wrong if r["human_attribution"] != "lesson_wrong"
                 or r.get("lesson_id") != r.get("human_lesson_id"))
        out["lesson_wrong_false_positive_rate"] = round(fp / len(judged_wrong), 3)
    out["meaningful"] = out["positives_lesson_wrong"] >= MIN_POSITIVES
    return out


def render(rows: list[dict]) -> str:
    lines = [f"corrections: {len(rows)}"]
    counts = Counter(r["attribution"] for r in rows)
    lines.append("judge attributions: " + ", ".join(f"{k}={counts.get(k, 0)}" for k in ATTRIBUTIONS))
    per_lesson: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        if r.get("lesson_id"):
            per_lesson[r["lesson_id"]][r["attribution"]] += 1
    if per_lesson:
        lines.append("per lesson (judge, record only - nothing acts on these):")
        for lid, c in sorted(per_lesson.items()):
            lines.append(f"  lesson {lid}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    lines.append("")
    for r in rows:
        present = json.loads(r.get("lessons_present") or "[]")
        human = f" | human={r['human_attribution']}" + (f"/{r['human_lesson_id']}" if r.get("human_lesson_id") else "") \
            if r.get("human_attribution") else ""
        conf = f"{r['confidence']:.2f}" if r.get("confidence") is not None else "-"
        lines.append(f"#{r['id']} [{r['attribution']}{'/' + str(r['lesson_id']) if r.get('lesson_id') else ''} "
                     f"conf {conf}] present={present}{human}")
        lines.append(f"    {(r['text'] or '')[:140]!r}")
    a = agreement(rows)
    lines.append("")
    lines.append(f"human-labelled: {a['labelled']}  lesson_wrong positives: {a['positives_lesson_wrong']} "
                 f"(need >= {MIN_POSITIVES} for the gate to mean anything: {'yes' if a['meaningful'] else 'NO'})")
    for cls, st in sorted(a["per_class"].items()):
        lines.append(f"  {cls}: n={st['n']} agreement={st['agreement']}")
    if a["lesson_wrong_false_positive_rate"] is not None:
        lines.append(f"  lesson_wrong false-positive rate: {a['lesson_wrong_false_positive_rate']}")
    lines.append("phase 2 gate: agreement(lesson_wrong) >= 0.8 AND FP <= 0.1 AND positives >= 10 AND user approval")
    return "\n".join(lines)


def label(db_path: str, correction_id: int, attribution: str, lesson_id: int | None) -> None:
    if attribution not in ATTRIBUTIONS or attribution == "unattributed":
        raise SystemExit(f"attribution must be one of {[a for a in ATTRIBUTIONS if a != 'unattributed']}")
    con = _connect(db_path, readonly=False)
    try:
        con.execute(
            "UPDATE corrections SET human_attribution = ?, human_lesson_id = ?, human_labelled_at = ? WHERE id = ?",
            (attribution, lesson_id, datetime.now(timezone.utc).isoformat(), correction_id),
        )
        con.commit()
        print(f"labelled correction {correction_id}: {attribution} lesson={lesson_id}")
    finally:
        con.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None, help="database path (default: from config.yaml)")
    ap.add_argument("--label", nargs="+", metavar=("ID ATTRIBUTION", "LESSON_ID"),
                    help="record a human label: ID ATTRIBUTION [LESSON_ID]")
    args = ap.parse_args(argv)

    db = args.db
    if not db:
        from blipshell.core.config import ConfigManager
        db = ConfigManager().load().database.path

    if args.label:
        if len(args.label) < 2:
            raise SystemExit("--label needs ID ATTRIBUTION [LESSON_ID]")
        lid = int(args.label[2]) if len(args.label) > 2 else None
        label(db, int(args.label[0]), args.label[1], lid)
        return 0

    con = _connect(db)
    try:
        print(render(load_rows(con)))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
