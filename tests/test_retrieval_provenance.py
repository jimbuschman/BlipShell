"""Aggregation logic of scripts/retrieval_provenance.py (the pure half).

The script itself needs live models (query rephrasing + embedding) and runs
on the Ollama PC; the aggregation that turns per-probe role counts into the
exhaust-vs-world verdict numbers is pure and pinned here.
"""

from scripts.retrieval_provenance import summarize


def test_empty_probe_set():
    assert summarize([], 0.5) == {"probes": 0}


def test_balanced_retrieval_matches_baseline():
    rows = [
        {"assistant_in_topk": 4, "k": 10, "top1_role": "user", "first_user_rank": 1},
        {"assistant_in_topk": 4, "k": 10, "top1_role": "user", "first_user_rank": 1},
    ]
    stats = summarize(rows, baseline_assistant_share=0.4)
    assert stats["mean_assistant_share_topk"] == 0.4
    assert stats["over_representation"] == 1.0
    assert stats["top1_assistant_count"] == 0
    assert stats["median_first_user_rank"] == 1
    assert stats["probes_with_no_user_result"] == 0


def test_exhaust_pathology_is_visible():
    # Wisp-shaped failure: exhaust owns the top of every list, first user
    # memory buried or absent entirely.
    rows = [
        {"assistant_in_topk": 9, "k": 10, "top1_role": "assistant",
         "first_user_rank": 8},
        {"assistant_in_topk": 10, "k": 10, "top1_role": "assistant",
         "first_user_rank": None},
        {"assistant_in_topk": 8, "k": 10, "top1_role": "assistant",
         "first_user_rank": 9},
    ]
    stats = summarize(rows, baseline_assistant_share=0.45)
    assert stats["mean_assistant_share_topk"] == 0.9
    assert stats["over_representation"] == 2.0
    assert stats["top1_assistant_count"] == 3
    assert stats["median_first_user_rank"] == 9
    assert stats["probes_with_no_user_result"] == 1


def test_zero_baseline_reports_no_ratio():
    rows = [
        {"assistant_in_topk": 0, "k": 10, "top1_role": "user", "first_user_rank": 1},
    ]
    stats = summarize(rows, baseline_assistant_share=0.0)
    assert stats["over_representation"] is None
