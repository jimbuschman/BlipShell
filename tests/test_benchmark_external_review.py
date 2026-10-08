"""Judge-free benchmark review export/import and v2 manifest behavior."""

import json

import pytest

from blipshell.benchmark.manifest import REQUIRED_CATEGORIES, manifest_dict, resolve_tier
from blipshell.benchmark.report import build_report
from blipshell.benchmark.results import ResultsStore
from blipshell.benchmark.review import export_review_packet, import_review, load_review_rows


def _source(store, model, stamp, response):
    store.write_review_source(
        model=model, run_ts=stamp, run_group=f"{model}@{stamp}",
        benchmark_manifest=manifest_dict(),
        items=[{
            "category": "reasoning", "case": "case-1",
            "task": "Explain the failure.", "response": response,
            "rubric": "correct and actionable",
        }],
    )


def test_tiers_have_explicit_jobs_and_repeat_defaults():
    jobs, repeats = resolve_tier("smoke")
    assert jobs == {"dedup", "reasoning"} and repeats == 1
    jobs, repeats = resolve_tier("decision", {"pipeline"}, 2)
    assert jobs == {"pipeline"} and repeats == 2


def test_export_is_blinded_and_import_restores_model_scores(tmp_path):
    store = ResultsStore(tmp_path, structured=True)
    _source(store, "model/a", "2026-09-23T10:00:00+00:00", "A response")
    _source(store, "model/b", "2026-09-23T10:01:00+00:00", "B response")

    paths = export_review_packet(tmp_path, packet_name="cohort-one")
    packet = json.loads(paths["json"].read_text(encoding="utf-8"))
    text = paths["markdown"].read_text(encoding="utf-8")
    assert "model/a" not in text and "model/b" not in text
    assert {i["candidate"] for i in packet["items"]} == {"Candidate A", "Candidate B"}

    response = tmp_path / "returned.json"
    response.write_text(json.dumps({
        "packet_id": "cohort-one", "reviewer": "manual-test",
        "reviews": [
            {"item_id": item["item_id"], "score": 0.75, "reason": "grounded"}
            for item in packet["items"]
        ],
    }), encoding="utf-8")
    imported = import_review(tmp_path, response)
    assert imported.exists()
    rows = load_review_rows(tmp_path)
    assert set(rows) == {"model/a", "model/b"}
    assert all(r[0]["value"] == 0.75 for r in rows.values())

    second = tmp_path / "returned-second.json"
    second.write_text(json.dumps({
        "packet_id": "cohort-one", "reviewer": "second-reviewer",
        "reviews": [
            {"item_id": item["item_id"], "score": 0.25, "reason": "independent"}
            for item in packet["items"]
        ],
    }), encoding="utf-8")
    import_review(tmp_path, second)
    rows = load_review_rows(tmp_path)
    assert all(r[0]["value"] == 0.5 for r in rows.values())
    assert all(len(r[0]["raw"]["reviewers"]) == 2 for r in rows.values())


def test_import_rejects_unknown_items(tmp_path):
    store = ResultsStore(tmp_path, structured=True)
    _source(store, "m", "2026-09-23T10:00:00+00:00", "answer")
    export_review_packet(tmp_path, packet_name="p")
    response = tmp_path / "bad.json"
    response.write_text(json.dumps({
        "packet_id": "p", "reviews": [{"item_id": "unknown", "score": 1.0}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown review item"):
        import_review(tmp_path, response)


def test_versioned_report_uses_fixed_coverage_and_effective_score():
    rows = [{
        "suite": "reasoning", "task_type": "tool_calling",
        "metric": "tool_pass_rate", "value": 1.0,
    }, {
        "suite": "reasoning", "task_type": "tool_calling",
        "metric": "completion_rate", "value": 0.5,
    }]
    report = build_report(
        {"m": rows}, required_categories=set(REQUIRED_CATEGORIES),
    )
    assert report["coverage"]["m"] == 1
    assert report["max_coverage"] == len(REQUIRED_CATEGORIES)
    assert report["composite"]["m"] == 0.5
