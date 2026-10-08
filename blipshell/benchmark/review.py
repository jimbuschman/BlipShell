"""Offline external review packets for open-ended benchmark responses.

BlipShell never calls a judge model.  It exports blinded prompt/response pairs
with explicit rubrics; the user may give the packet to ChatGPT, Claude, or a
human, then import the returned JSON without rerunning candidate inference.
"""

from __future__ import annotations

import json
import re
import secrets
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from blipshell.benchmark.manifest import REVIEW_SCHEMA_VERSION


RUBRICS = {
    "summarization": (
        "Score 0.0-1.0 for factual faithfulness, preservation of salient details, "
        "concision, and neutral third-person memory-note style. Do not reward verbosity."
    ),
    "lessons": (
        "Score 0.0-1.0 for grounding in the conversation, reusability, specificity, "
        "and concision. Penalize invented lessons and generic advice."
    ),
    "reasoning": (
        "Score 0.0-1.0 for correctness, completeness, calibrated uncertainty, and "
        "actionability. Do not reward length by itself."
    ),
    "code_gen": (
        "Score 0.0-1.0 for correctness, completeness, idiomatic implementation, "
        "edge-case handling, and compliance with the requested interface."
    ),
    "session_review": (
        "Score 0.0-1.0 for accurate synthesis, useful reflection, coverage of what "
        "worked and failed, and absence of invented events. Do not reward verbosity."
    ),
}


def _now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _load_sources(root: Path, models: Optional[set[str]] = None) -> list[dict]:
    source_dir = root / "review-sources"
    latest: dict[str, dict] = {}
    for path in sorted(source_dir.glob("*.json")) if source_dir.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = data.get("model")
        if not model or (models and model not in models):
            continue
        if data.get("items"):
            latest[model] = data
    return list(latest.values())


def export_review_packet(
    root: str | Path,
    *,
    models: Optional[set[str]] = None,
    packet_name: Optional[str] = None,
) -> dict[str, Path]:
    """Create blinded Markdown/JSON plus a private alias key for later import."""
    root = Path(root)
    sources = _load_sources(root, models)
    if not sources:
        raise ValueError("No open-ended review sources found for the requested models")

    packet_id = packet_name or f"review-{_now_compact()}"
    aliases = [f"Candidate {chr(65 + i)}" if i < 26 else f"Candidate {i + 1}"
               for i in range(len(sources))]
    secrets.SystemRandom().shuffle(sources)
    model_alias = {source["model"]: aliases[i] for i, source in enumerate(sources)}

    items = []
    key_items = {}
    for source in sources:
        alias = model_alias[source["model"]]
        for index, raw in enumerate(source.get("items") or [], 1):
            category = raw["category"]
            item_id = f"{packet_id}-{len(items) + 1:04d}"
            items.append({
                "item_id": item_id,
                "candidate": alias,
                "category": category,
                "case": raw.get("case") or str(index),
                "task": raw.get("task") or "",
                "response": raw.get("response") or "",
                "rubric": RUBRICS[category],
            })
            key_items[item_id] = {
                "model": source["model"],
                "run_group": source.get("run_group"),
                "run_ts": source.get("run_ts"),
                "category": category,
                "case": raw.get("case") or str(index),
            }

    pending = root / "external-reviews" / "pending"
    keys = root / "external-reviews" / "keys"
    pending.mkdir(parents=True, exist_ok=True)
    keys.mkdir(parents=True, exist_ok=True)
    packet_json = pending / f"{packet_id}.json"
    packet_md = pending / f"{packet_id}.md"
    key_path = keys / f"{packet_id}.json"

    packet = {
        "schema": REVIEW_SCHEMA_VERSION,
        "kind": "blipshell_external_review_packet",
        "packet_id": packet_id,
        "instructions": (
            "Review every item independently. Return JSON only with: "
            "{packet_id, reviewer, reviews:[{item_id, score, reason}]}. "
            "score must be a number from 0.0 to 1.0."
        ),
        "items": items,
    }
    packet_json.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    key_path.write_text(json.dumps({
        "schema": REVIEW_SCHEMA_VERSION,
        "packet_id": packet_id,
        "items": key_items,
    }, indent=2), encoding="utf-8")

    lines = [
        "# BlipShell external benchmark review",
        "",
        packet["instructions"],
        "",
        "Candidate names are blinded. Do not infer identity from writing style.",
    ]
    for item in items:
        lines += [
            "", f"## {item['item_id']} - {item['candidate']} - {item['category']}",
            "", f"Case: {item['case']}", "", f"Rubric: {item['rubric']}",
            "", "### Task", "", item["task"], "", "### Response", "",
            item["response"],
        ]
    lines += [
        "", "## Required response", "", "Return JSON only:", "", "```json",
        json.dumps({
            "packet_id": packet_id,
            "reviewer": "ChatGPT, Claude, or human reviewer",
            "reviews": [{"item_id": items[0]["item_id"], "score": 0.0,
                         "reason": "brief evidence-based explanation"}],
        }, indent=2),
        "```", "",
    ]
    packet_md.write_text("\n".join(lines), encoding="utf-8")
    return {"markdown": packet_md, "json": packet_json, "key": key_path}


def import_review(root: str | Path, response_path: str | Path) -> Path:
    """Validate a returned review and persist model/category score rows."""
    root = Path(root)
    response_path = Path(response_path)
    response_text = response_path.read_text(encoding="utf-8").strip()
    response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text,
                           flags=re.IGNORECASE)
    response = json.loads(response_text)
    packet_id = response.get("packet_id")
    if not packet_id:
        raise ValueError("Review response has no packet_id")
    key_path = root / "external-reviews" / "keys" / f"{packet_id}.json"
    if not key_path.exists():
        raise ValueError(f"No local review key found for packet {packet_id}")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    known = key.get("items") or {}
    reviews = response.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("Review response must contain a non-empty reviews list")

    accepted = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen = set()
    for review in reviews:
        item_id = review.get("item_id")
        if item_id in seen:
            raise ValueError(f"Duplicate review item: {item_id}")
        seen.add(item_id)
        if item_id not in known:
            raise ValueError(f"Unknown review item: {item_id}")
        score = review.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise ValueError(f"Score for {item_id} must be between 0.0 and 1.0")
        meta = known[item_id]
        accepted.append({**review, **meta})
        grouped[(meta["model"], meta["category"])].append(float(score))

    missing = sorted(set(known) - seen)
    rows_by_model: dict[str, list[dict]] = defaultdict(list)
    for (model, category), scores in grouped.items():
        rows_by_model[model].append({
            "suite": "external_review",
            "task_type": category,
            "metric": "quality",
            "value": round(sum(scores) / len(scores), 4),
            "unit": "ratio",
            "raw": {"reviewed": len(scores), "reviewer": response.get("reviewer")},
            "review_packet": packet_id,
        })

    imported = root / "external-reviews" / "imported"
    imported.mkdir(parents=True, exist_ok=True)
    out = imported / f"{_now_compact()}__{packet_id}__{secrets.token_hex(3)}.json"
    out.write_text(json.dumps({
        "schema": REVIEW_SCHEMA_VERSION,
        "kind": "blipshell_external_review",
        "packet_id": packet_id,
        "reviewer": response.get("reviewer") or "unspecified",
        "imported_ts": datetime.now(timezone.utc).isoformat(),
        "missing_items": missing,
        "reviews": accepted,
        "rows_by_model": rows_by_model,
    }, indent=2), encoding="utf-8")
    return out


def load_review_rows(root: str | Path) -> dict[str, list[dict]]:
    """Aggregate every independent imported review by model/category."""
    imported = Path(root) / "external-reviews" / "imported"
    grouped: dict[tuple[str, str], dict] = {}
    for path in sorted(imported.glob("*.json")) if imported.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for model, rows in (data.get("rows_by_model") or {}).items():
            for row in rows:
                key = (model, row.get("task_type"))
                reviewed = int((row.get("raw") or {}).get("reviewed") or 1)
                bucket = grouped.setdefault(key, {
                    "weighted": 0.0, "reviewed": 0, "reviewers": [],
                    "run_ts": None, "row": row,
                })
                bucket["weighted"] += float(row["value"]) * reviewed
                bucket["reviewed"] += reviewed
                bucket["reviewers"].append(data.get("reviewer") or "unspecified")
                bucket["run_ts"] = data.get("imported_ts") or bucket["run_ts"]
    out: dict[str, list[dict]] = defaultdict(list)
    for (model, category), bucket in grouped.items():
        row = dict(bucket["row"])
        row.update({
            "model": model,
            "task_type": category,
            "value": round(bucket["weighted"] / bucket["reviewed"], 4),
            "run_ts": bucket["run_ts"],
            "external_review": True,
            "raw": {"reviewed": bucket["reviewed"],
                    "reviewers": sorted(set(bucket["reviewers"]))},
        })
        out[model].append(row)
    return dict(out)
