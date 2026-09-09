"""The D2a evaluation boundary: a frozen, hand-labelled correction set and
versioned judge runs against it (approved 2026-09-09).

Rules this module enforces, because the plan depends on them:

- **Build against the CURRENT selection behaviour.** A set is tagged with a
  `generation` (e.g. `pre-D1`) and a description of how lessons were being
  selected when it was built. Sets from different generations are never
  merged.
- **Freeze before running.** Human labels can be edited until `freeze()`;
  after that the set is read-only and the judge runs against it. A run on an
  unfrozen or incompletely labelled set is refused.
- **No tuning against the held labels.** Every run records a hash of the
  judge's system prompt and prompt template. The first run on a generation
  is its baseline; a later run whose judge hash differs is a NEW judge
  version and is reported as such, never averaged into the baseline.
- **Texts stay local.** The set and the full run records live under
  `data/attribution_eval/` (gitignored - they are the user's most personal
  messages). Only a numbers-only summary goes to `benchmark_results/`.

Items come from two sources: `corrections` rows (phase 1, exact
`lessons_present`) and the historical anti-pattern lessons the correction
detector minted before phase 1, whose content carries the user's words and
the previous reply. For those the lesson set in context is RECONSTRUCTED as
the top-30-by-importance lessons that existed at the time (the always-on
pool rule) - flagged `lessons_present_source = reconstructed_top30`, since
per-query Recall hits cannot be recovered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from blipshell.memory import attribution as att

logger = logging.getLogger(__name__)

DEFAULT_DIR = Path("data") / "attribution_eval"
MIN_POSITIVES = 10
GATE_AGREEMENT = 0.8
GATE_FP_RATE = 0.1
GATE_SPREAD = 0.1
TOP_N_RECONSTRUCTED = 30

_USER_SAID = re.compile(r'User said: "(.*)"\s*$', re.DOTALL)
_PREV = re.compile(r'Previous response \(excerpt\): "(.*?)\.\.\."\.', re.DOTALL)


def judge_hash() -> str:
    """Identity of the judge under evaluation: system prompt + prompt template."""
    template = att.judge_prompt("<c>", "<p>", [(0, "<l>")])
    return hashlib.sha256((att.JUDGE_SYSTEM + "\n" + template).encode()).hexdigest()[:12]


@dataclass
class EvalItem:
    item_id: str
    source: str                       # corrections_row | historical_lesson | historical_message
    source_id: int
    text: str
    prev_assistant: str
    lessons_present: list[int]
    lessons_present_source: str       # recorded | reconstructed_top30
    lessons: list[list]               # [[lesson_id, content], ...] - what the judge is shown
    at: str = ""
    human_attribution: Optional[str] = None
    human_lesson_id: Optional[int] = None
    human_labelled_at: Optional[str] = None


@dataclass
class EvalSet:
    generation: str
    selection_behavior: str
    created_at: str
    db_name: str
    items: list[EvalItem] = field(default_factory=list)
    frozen_at: Optional[str] = None

    @property
    def frozen(self) -> bool:
        return self.frozen_at is not None

    def labelled(self) -> list[EvalItem]:
        return [i for i in self.items if i.human_attribution]

    def positives(self) -> int:
        return sum(1 for i in self.items if i.human_attribution == "lesson_wrong")


def set_path(generation: str, base: Path = DEFAULT_DIR) -> Path:
    return base / f"{generation}.json"


def save_set(s: EvalSet, base: Path = DEFAULT_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    p = set_path(s.generation, base)
    payload = {**asdict(s)}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_set(generation: str, base: Path = DEFAULT_DIR) -> EvalSet:
    p = set_path(generation, base)
    data = json.loads(p.read_text(encoding="utf-8"))
    items = [EvalItem(**i) for i in data.pop("items", [])]
    return EvalSet(items=items, **data)


# ---------------------------------------------------------------- build

def parse_anti_pattern(content: str) -> tuple[Optional[str], str]:
    """(user's correction text, previous assistant excerpt) from a detector lesson."""
    m = _USER_SAID.search(content or "")
    if not m:
        return None, ""
    prev = _PREV.search(content or "")
    return m.group(1).strip(), (prev.group(1).strip() if prev else "")


def _pool_at(others, ts):
    """The always-on lesson pool as it stood at `ts` (pre-D1 selection: the
    top-N by importance of every lesson that existed), reconstructed."""
    pool = [o for o in others if o.timestamp and ts and o.timestamp <= ts]
    pool.sort(key=lambda o: o.importance, reverse=True)
    return pool[:TOP_N_RECONSTRUCTED]


async def build_set(sqlite, *, generation: str, selection_behavior: str,
                    db_name: str = "", limit: int = 200) -> EvalSet:
    """Collect candidate items from three sources, most exact first:
    1. `corrections` rows (phase 1 records: exact lessons_present),
    2. historical anti-pattern lessons the detector minted,
    3. the production correction detector replayed over raw user messages
       written while a lesson pool existed (a candidate the detector would
       have flagged; the pool is reconstructed at that time; the previous
       assistant message in the session is the excerpt).
    Source 3 exists because the 2026-09-02 corpus held only THREE detector
    lessons: the detector wrote lessons for a fraction of the corrections it
    saw, and the stage-2 judge (2026-09-02) rejects most candidates as
    recounted dialogue. Its items are CANDIDATES - many will be labelled
    `unrelated`, which is exactly the false-positive population the gate
    needs. Does not label. Never overwrites a frozen set (caller checks)."""
    s = EvalSet(generation=generation, selection_behavior=selection_behavior,
                created_at=datetime.now(timezone.utc).isoformat(), db_name=db_name)

    # 1. phase-1 rows: exact lessons_present
    cur = await sqlite._db.execute("SELECT * FROM corrections ORDER BY id LIMIT ?", (limit,))
    for r in await cur.fetchall():
        r = dict(r)
        try:
            present = [int(i) for i in json.loads(r.get("lessons_present") or "[]")]
        except (ValueError, TypeError):
            present = []
        lessons = []
        for lid in present[:att.MAX_LESSONS_JUDGED]:
            lesson = await sqlite.get_lesson(lid)
            if lesson is not None:
                lessons.append([lid, (lesson.content or "")[:300]])
        s.items.append(EvalItem(
            item_id=f"c{r['id']}", source="corrections_row", source_id=int(r["id"]),
            text=r["text"], prev_assistant=r.get("prev_assistant_excerpt") or "",
            lessons_present=present, lessons_present_source="recorded", lessons=lessons,
            at=r.get("at") or "",
        ))

    # 2. historical detector lessons: reconstruct the always-on pool at the time
    all_lessons = await sqlite.get_all_lessons()
    anti = [l for l in all_lessons if (l.added_by == "correction_detector"
                                        or (l.content or "").startswith("ANTI-PATTERN: User corrected"))]
    others = [l for l in all_lessons if l not in anti]
    seen_texts = {i.text.strip().lower() for i in s.items}
    for l in sorted(anti, key=lambda x: x.timestamp or datetime.min)[:limit]:
        text, prev = parse_anti_pattern(l.content or "")
        if not text:
            continue
        ts = l.timestamp
        pool = _pool_at(others, ts)
        seen_texts.add(text.strip().lower())
        s.items.append(EvalItem(
            item_id=f"h{l.id}", source="historical_lesson", source_id=int(l.id),
            text=text, prev_assistant=prev,
            lessons_present=[o.id for o in pool],
            lessons_present_source="reconstructed_top30",
            lessons=[[o.id, (o.content or "")[:300]] for o in pool[:att.MAX_LESSONS_JUDGED]],
            at=ts.isoformat() if ts else "",
        ))

    # 3. detector replay over raw user messages, only while a pool existed
    from blipshell.core.guardrails import detect_correction
    first_lesson = min((o.timestamp for o in others if o.timestamp), default=None)
    if first_lesson is not None:
        cur = await sqlite._db.execute(
            "SELECT id, session_id, content, timestamp FROM memories "
            "WHERE role = 'user' AND is_archived = 0 AND timestamp >= ? ORDER BY id",
            (first_lesson.isoformat(),))
        rows = [dict(r) for r in await cur.fetchall()]
        added = 0
        for r in rows:
            text = (r.get("content") or "").strip()
            if not text or not detect_correction(text):
                continue
            key = text.lower()
            if key in seen_texts:
                continue  # the detector already minted a lesson for this one (source 2)
            ts = _parse_ts(r.get("timestamp"))
            pool = _pool_at(others, ts)
            if not pool:
                continue
            prev_cur = await sqlite._db.execute(
                "SELECT content FROM memories WHERE session_id = ? AND role = 'assistant' AND id < ? "
                "ORDER BY id DESC LIMIT 1", (r["session_id"], r["id"]))
            prev_row = await prev_cur.fetchone()
            prev = (prev_row[0] or "")[:400] if prev_row else ""
            seen_texts.add(key)
            s.items.append(EvalItem(
                item_id=f"m{r['id']}", source="historical_message", source_id=int(r["id"]),
                text=text[:1000], prev_assistant=prev,
                lessons_present=[o.id for o in pool],
                lessons_present_source="reconstructed_top30",
                lessons=[[o.id, (o.content or "")[:300]] for o in pool[:att.MAX_LESSONS_JUDGED]],
                at=ts.isoformat() if ts else (r.get("timestamp") or ""),
            ))
            added += 1
            if added >= limit:
                break
    return s


def _parse_ts(raw) -> Optional[datetime]:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- label / freeze

def label_item(s: EvalSet, item_id: str, attribution: str, lesson_id: Optional[int]) -> EvalItem:
    if s.frozen:
        raise PermissionError(f"set {s.generation!r} is frozen ({s.frozen_at}); labels are read-only")
    if attribution not in att.ATTRIBUTIONS or attribution == "unattributed":
        raise ValueError(f"attribution must be one of {[a for a in att.ATTRIBUTIONS if a != 'unattributed']}")
    item = next((i for i in s.items if i.item_id == item_id), None)
    if item is None:
        raise KeyError(item_id)
    if attribution != "unrelated" and lesson_id not in item.lessons_present:
        raise ValueError(f"lesson {lesson_id} was not present for item {item_id}: {item.lessons_present}")
    item.human_attribution = attribution
    item.human_lesson_id = None if attribution == "unrelated" else int(lesson_id)
    item.human_labelled_at = datetime.now(timezone.utc).isoformat()
    return item


def freeze(s: EvalSet) -> dict:
    """Freeze once every item is labelled. Returns the label census; raises if
    incomplete. Warns (in the census) when lesson_wrong positives < MIN_POSITIVES."""
    if s.frozen:
        return census(s)
    unlabelled = [i.item_id for i in s.items if not i.human_attribution]
    if unlabelled:
        raise ValueError(f"{len(unlabelled)} item(s) unlabelled: {unlabelled[:10]}")
    s.frozen_at = datetime.now(timezone.utc).isoformat()
    return census(s)


def census(s: EvalSet) -> dict:
    counts: dict[str, int] = {}
    for i in s.items:
        counts[i.human_attribution or "UNLABELLED"] = counts.get(i.human_attribution or "UNLABELLED", 0) + 1
    return {
        "generation": s.generation, "items": len(s.items), "labelled": len(s.labelled()),
        "counts": counts, "positives_lesson_wrong": s.positives(),
        "positives_ok": s.positives() >= MIN_POSITIVES, "frozen": s.frozen,
    }


# ---------------------------------------------------------------- run

@dataclass
class RunResult:
    generation: str
    judge_hash: str
    run_ts: str
    repeats: int
    per_repeat: list[dict]          # [{agreement: {cls: x}, fp_rate: y, unattributed_rate: z}]
    mean_agreement: dict            # cls -> mean
    spread_agreement: dict          # cls -> max-min
    mean_fp_rate: Optional[float]
    spread_fp_rate: Optional[float]
    positives: int
    gate: dict                      # {passes: bool, reasons: [...]}
    verdicts: list[dict]            # per item per repeat: {item_id, repeat, attribution, lesson_id, confidence}


def _score_repeat(items: list[EvalItem], verdicts: dict[str, att.Verdict]) -> dict:
    by_class: dict[str, list[bool]] = {}
    fp_num = fp_den = 0
    unatt = 0
    for i in items:
        v = verdicts[i.item_id]
        if v.attribution == "unattributed":
            unatt += 1
        agree = (v.attribution == i.human_attribution
                 and (i.human_attribution == "unrelated" or v.lesson_id == i.human_lesson_id))
        by_class.setdefault(i.human_attribution, []).append(agree)
        if v.attribution == "lesson_wrong":
            fp_den += 1
            if not (i.human_attribution == "lesson_wrong" and v.lesson_id == i.human_lesson_id):
                fp_num += 1
    return {
        "agreement": {c: round(sum(a) / len(a), 4) for c, a in by_class.items()},
        "fp_rate": round(fp_num / fp_den, 4) if fp_den else None,
        "unattributed_rate": round(unatt / len(items), 4) if items else None,
    }


async def run_judge(s: EvalSet, router, *, repeats: int = 3) -> RunResult:
    """Run the phase-1 judge over a FROZEN set `repeats` times. Never writes to
    the store; never touches a lesson."""
    if not s.frozen:
        raise PermissionError("the set must be frozen (all items labelled) before the judge runs")
    from blipshell.llm.router import TaskType

    verdict_rows: list[dict] = []
    per_repeat: list[dict] = []
    for rep in range(repeats):
        verdicts: dict[str, att.Verdict] = {}
        for i in s.items:
            lessons = [(int(lid), txt) for lid, txt in i.lessons]
            if not lessons:
                v = att.Verdict("unrelated", None, 1.0, "no lessons were present", "")
            else:
                try:
                    raw = await router.generate(
                        TaskType.REASONING, att.judge_prompt(i.text, i.prev_assistant, lessons),
                        system=att.JUDGE_SYSTEM, think=False, use_cache=False,
                    )
                except TypeError:
                    raw = await router.generate(
                        TaskType.REASONING, att.judge_prompt(i.text, i.prev_assistant, lessons),
                        system=att.JUDGE_SYSTEM, think=False,
                    )
                except Exception as e:
                    logger.warning("judge failed on %s: %s", i.item_id, e)
                    raw = ""
                v = att.parse_verdict(raw, [lid for lid, _ in lessons])
            verdicts[i.item_id] = v
            verdict_rows.append({"item_id": i.item_id, "repeat": rep, "attribution": v.attribution,
                                 "lesson_id": v.lesson_id, "confidence": v.confidence})
        per_repeat.append(_score_repeat(s.items, verdicts))

    classes = sorted({c for r in per_repeat for c in r["agreement"]})
    mean_ag = {c: round(statistics.mean(r["agreement"][c] for r in per_repeat if c in r["agreement"]), 4) for c in classes}
    spread_ag = {c: round(max(r["agreement"][c] for r in per_repeat if c in r["agreement"])
                          - min(r["agreement"][c] for r in per_repeat if c in r["agreement"]), 4) for c in classes}
    fps = [r["fp_rate"] for r in per_repeat if r["fp_rate"] is not None]
    mean_fp = round(statistics.mean(fps), 4) if fps else None
    spread_fp = round(max(fps) - min(fps), 4) if fps else None

    reasons = []
    if s.positives() < MIN_POSITIVES:
        reasons.append(f"only {s.positives()} lesson_wrong positives (< {MIN_POSITIVES}); result not meaningful")
    lw = mean_ag.get("lesson_wrong")
    if lw is None or lw < GATE_AGREEMENT:
        reasons.append(f"lesson_wrong agreement {lw} < {GATE_AGREEMENT}")
    if mean_fp is None or mean_fp > GATE_FP_RATE:
        reasons.append(f"lesson_wrong false-positive rate {mean_fp} > {GATE_FP_RATE}")
    if spread_ag.get("lesson_wrong", 0.0) >= GATE_SPREAD or (spread_fp or 0.0) >= GATE_SPREAD:
        reasons.append("repeat spread >= 0.1")
    return RunResult(
        generation=s.generation, judge_hash=judge_hash(),
        run_ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"), repeats=repeats,
        per_repeat=per_repeat, mean_agreement=mean_ag, spread_agreement=spread_ag,
        mean_fp_rate=mean_fp, spread_fp_rate=spread_fp, positives=s.positives(),
        gate={"passes": not reasons, "reasons": reasons}, verdicts=verdict_rows,
    )


def previous_runs(generation: str, base: Path = DEFAULT_DIR) -> list[dict]:
    out = []
    for p in sorted(base.glob(f"{generation}__run__*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    return out


def write_run(result: RunResult, *, base: Path = DEFAULT_DIR,
              summary_dir: Optional[Path] = None, git_sha: str = "nosha") -> tuple[Path, Optional[Path]]:
    """Full record (with per-item verdicts) locally; numbers-only summary for
    the committed results directory. Marks a judge-version change."""
    base.mkdir(parents=True, exist_ok=True)
    prior = previous_runs(result.generation, base)
    baseline_hash = prior[0]["judge_hash"] if prior else result.judge_hash
    judge_version = "baseline" if result.judge_hash == baseline_hash else f"changed-from-{baseline_hash}"
    record = {**asdict(result), "judge_version": judge_version, "git_sha": git_sha,
              "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME")}
    full = base / f"{result.generation}__run__{result.run_ts}.json"
    full.write_text(json.dumps(record, indent=2), encoding="utf-8")
    summary_path = None
    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {k: v for k, v in record.items() if k != "verdicts"}
        summary["kind"] = "attribution_eval"
        summary_path = summary_dir / f"attribution_eval__{result.generation}__{git_sha}__{result.run_ts}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return full, summary_path


def render(result: RunResult) -> str:
    lines = [f"generation={result.generation} judge={result.judge_hash} repeats={result.repeats} "
             f"positives(lesson_wrong)={result.positives}"]
    for c in sorted(result.mean_agreement):
        lines.append(f"  {c:18} agreement={result.mean_agreement[c]:.3f} spread={result.spread_agreement[c]:.3f}")
    lines.append(f"  lesson_wrong FP rate={result.mean_fp_rate} spread={result.spread_fp_rate}")
    lines.append(f"  unattributed rate per repeat: {[r['unattributed_rate'] for r in result.per_repeat]}")
    lines.append("gate: " + ("PASSES (phase 2 still needs the user's explicit approval)" if result.gate["passes"]
                             else "DOES NOT PASS - " + "; ".join(result.gate["reasons"])))
    return "\n".join(lines)
