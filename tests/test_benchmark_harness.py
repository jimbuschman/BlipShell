"""Unit tests for the unified benchmark harness — logic/wiring only.

Per the project's validation split, these run on the dev box with NO Ollama:
they cover normalization, scoring math, the switch-verdict, judge response
parsing (via a fake client), discovery parsing (canned JSON), and the store.
Model quality/behavior is validated separately on the Ollama PC.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blipshell.benchmark import discovery, embedding_bench, harness, report
from blipshell.benchmark.judge import LLMJudge
from blipshell.benchmark.store import BenchmarkStore


# ---------------------------------------------------------------------------
# Pure scorers
# ---------------------------------------------------------------------------

def test_score_ranking_correlation_discriminates():
    truth = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    perfect = [{"parsed": t, "truth_rank": t} for t in truth]
    inverted = [{"parsed": 6 - t, "truth_rank": t} for t in truth]
    flat = [{"parsed": 3, "truth_rank": t} for t in truth]
    assert harness.score_ranking(perfect) == pytest.approx(1.0)
    assert harness.score_ranking(inverted) == pytest.approx(0.0)
    # rated everything the same while truth varies -> failed to discriminate
    assert harness.score_ranking(flat) == pytest.approx(0.0)


def test_score_ranking_ignores_errors():
    results = [{"parsed": -1, "truth_rank": 3}] * 8  # all errored -> <2 valid
    assert harness.score_ranking(results) is None


def test_score_importance_calibration():
    truth = [0.1, 0.5, 0.9, 0.3, 0.7]
    perfect = [{"parsed": t, "truth_importance": t} for t in truth]
    assert harness.score_importance(perfect) == pytest.approx(1.0)
    # flat 0.5 guess: poor correlation + ~0.24 MAE -> well below 1.0
    flat = [{"parsed": 0.5, "truth_importance": t} for t in truth]
    assert harness.score_importance(flat) < 0.6


def test_score_contradiction_only_valid():
    results = [
        {"parsed": "YES", "correct": True},
        {"parsed": "NO", "correct": True},
        {"parsed": "INVALID", "correct": False},  # excluded
        {"parsed": "YES", "correct": False},
    ]
    assert harness.score_contradiction(results) == pytest.approx(2 / 3)


def test_score_entity_f1():
    results = [
        {"extracted": ["user", "desk robot"], "expected": {"user", "desk robot"}},  # F1 1.0
        {"extracted": ["wrong"], "expected": {"right"}},                              # F1 0.0
        {"extracted": [], "expected": set()},                                          # both empty -> 1.0
    ]
    # mean(1.0, 0.0, 1.0) = 0.6667
    assert harness.score_entity(results) == pytest.approx(2 / 3, abs=1e-3)


def test_score_entity_partial_f1_and_case_insensitive():
    # extracted {a,b,c} vs expected {A,B}: precision 2/3, recall 1.0 -> F1 0.8
    results = [{"extracted": ["A", "B", "C"], "expected": {"a", "b"}}]
    assert harness.score_entity(results) == pytest.approx(0.8, abs=1e-3)


def test_score_entity_skips_errors():
    results = [{"raw": "ERROR: boom", "expected": {"x"}}] * 5
    assert harness.score_entity(results) is None


def test_score_tool_calling():
    results = [
        {"correct": True, "content": "ok"},
        {"correct": False, "content": "ok"},
        {"correct": True, "content": "ERROR: x"},  # excluded
    ]
    assert harness.score_tool_calling(results) == pytest.approx(1 / 2)


def test_score_rank_and_importance_average():
    # perfect on both channels -> 1.0
    results = [
        {"rank": t, "importance": t / 5.0, "truth_rank": t, "truth_importance": t / 5.0}
        for t in [1, 2, 3, 4, 5]
    ]
    assert harness.score_rank_and_importance(results) == pytest.approx(1.0)


def test_realdata_agreement_tolerance():
    results = [
        {"original_rank": 3, "new_rank": 3},   # agree
        {"original_rank": 3, "new_rank": 4},   # within tol=1
        {"original_rank": 1, "new_rank": 5},   # disagree
        {"original_rank": 2, "new_rank": -1},  # invalid, skipped
    ]
    assert harness.realdata_agreement(results, "original_rank", "new_rank", tol=1) == pytest.approx(2 / 3)


def test_score_reflection_completeness():
    full = {
        "effectiveness": "effective",
        "what_worked": "x", "what_didnt_work": "y",
        "technical_insights": "z", "process_insights": "w",
    }
    assert harness.score_reflection_completeness(full) == pytest.approx(1.0)
    half = {"what_worked": "x", "what_didnt_work": "y",
            "technical_insights": None, "process_insights": None}
    assert harness.score_reflection_completeness(half) == pytest.approx(0.5)
    empty = {"what_worked": None, "what_didnt_work": None,
             "technical_insights": None, "process_insights": None}
    assert harness.score_reflection_completeness(empty) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Embedding retrieval metrics
# ---------------------------------------------------------------------------

def test_embedding_precision_recall_mrr():
    ranked = [10, 20, 30, 40, 50]   # ids in rank order
    expected = {30, 50}
    # P@5: 2 relevant in top5 / min(5,2)=2 -> 1.0
    assert embedding_bench.precision_at_k(ranked, expected, 5) == pytest.approx(1.0)
    # R@10: both found / 2 -> 1.0
    assert embedding_bench.recall_at_k(ranked, expected, 10) == pytest.approx(1.0)
    # first relevant at rank 3 -> MRR 1/3
    assert embedding_bench.mrr(ranked, expected) == pytest.approx(1 / 3)
    # P@2: only id 10,20 considered, neither relevant -> 0
    assert embedding_bench.precision_at_k(ranked, expected, 2) == pytest.approx(0.0)


def test_embedding_metrics_empty_expected():
    assert embedding_bench.precision_at_k([1, 2], set(), 5) == 0.0
    assert embedding_bench.recall_at_k([1, 2], set(), 10) == 0.0
    assert embedding_bench.mrr([1, 2], set()) == 0.0


def test_embedding_cosine():
    assert embedding_bench.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert embedding_bench.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert embedding_bench.cosine([1, 0], [0, 0]) == 0.0  # zero-norm guard


def test_embedding_aggregate():
    per_query = [
        {"p_at_5": 1.0, "r_at_10": 1.0, "mrr": 1.0},
        {"p_at_5": 0.0, "r_at_10": 0.0, "mrr": 0.0},
    ]
    agg = embedding_bench.aggregate_retrieval(per_query)
    assert agg["p_at_5"] == pytest.approx(0.5)
    assert agg["headline"] == pytest.approx(0.5)
    assert agg["n_queries"] == 2
    assert embedding_bench.aggregate_retrieval([]) is None


# ---------------------------------------------------------------------------
# Judge parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"score": 0.8, "reason": "good"}', 0.8),
    ('```json\n{"score": 0.5}\n```', 0.5),
    ('garbage score: 0.42 trailing', 0.42),
    ('I rate this 7/10', 0.7),          # 0-10 normalization
    ('score = 85', 0.85),               # 0-100 normalization
    ('0.33', 0.33),
    ('', None),
    ('no number here', None),
    ('{"score": 9999}', None),          # absurd -> rejected
])
def test_judge_parse_score(raw, expected):
    got = LLMJudge.parse_score(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


class _FakeClient:
    """Minimal duck-typed client for the judge: returns queued responses."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def generate(self, prompt, model, system=None, use_cache=True, **kw):
        self.calls += 1
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


async def test_judge_grade_returns_score():
    judge = LLMJudge("judge-model", _FakeClient(['{"score": 0.9}']))
    assert await judge.grade_summarization("src", "summary") == pytest.approx(0.9)


async def test_judge_grade_survives_failure():
    judge = LLMJudge("judge-model", _FakeClient([RuntimeError("api down")]))
    assert await judge.grade_reasoning("task", "resp") is None


# ---------------------------------------------------------------------------
# Report builder (the shareable deliverable — numbers only, no verdict).
# ---------------------------------------------------------------------------

def _rows(model, scores, is_baseline=False):
    """scores: {task_type: (metric, value)}."""
    return [
        {
            "run_group": "g", "model": model, "suite": "pipeline",
            "task_type": tt, "metric": metric, "value": val, "unit": "ratio",
            "tier": "deep", "is_baseline": is_baseline, "run_ts": "t", "raw_json": None,
        }
        for tt, (metric, val) in scores.items()
    ]


def _rows_with_latency(model, scores, lats):
    rows = _rows(model, scores)
    rows += [{
        "run_group": "g", "model": model, "suite": "s", "task_type": suite,
        "metric": "latency_s", "value": v, "unit": "seconds", "tier": "deep",
        "is_baseline": False, "run_ts": "t", "raw_json": None,
    } for suite, v in lats.items()]
    return rows


def test_report_quality_matrix_and_best_per_job():
    model_rows = {
        "a": _rows("a", {"coding": ("accuracy", 0.80), "ranking": ("accuracy", 0.60)}),
        "b": _rows("b", {"coding": ("accuracy", 0.40), "ranking": ("accuracy", 1.00)}),
    }
    rep = report.build_report(model_rows, task_weights={"coding": 3.0, "ranking": 1.0})
    coding = next(c for c in rep["categories"] if c["key"] == "coding")
    assert coding["scores"] == {"a": pytest.approx(0.80), "b": pytest.approx(0.40)}
    assert coding["best_model"] == "a"
    # composite a = (3*0.8 + 1*0.6)/4 = 0.75 ; b = (3*0.4 + 1*1.0)/4 = 0.55
    assert rep["composite"]["a"] == pytest.approx(0.75)
    assert rep["composite"]["b"] == pytest.approx(0.55)
    # known job ordering: ranking before coding
    assert [c["key"] for c in rep["categories"]] == ["ranking", "coding"]
    # NO verdict/incumbent/switch concepts anywhere
    assert "switch" not in rep and "incumbent" not in str(rep["categories"])


def test_report_latency_and_catalog_round_trip():
    model_rows = {
        "local": _rows_with_latency("local", {"ranking": ("accuracy", 0.8)},
                                    {"pipeline": 0.6, "reasoning_suite": 28.9}),
        "cloud": _rows_with_latency("cloud", {"ranking": ("accuracy", 0.85)},
                                    {"pipeline": 2.9, "reasoning_suite": 11.0}),
    }
    catalog = {"cloud": {"price_in": 0.3, "price_out": 1.2, "tok_per_s": 95, "context_length": 131072}}
    rep = report.build_report(model_rows, catalog=catalog, judge_model="judge-x")
    assert rep["latency"]["local"]["reasoning_suite"] == pytest.approx(28.9)
    assert rep["catalog"]["cloud"]["price_in"] == pytest.approx(0.3)
    assert rep["catalog"]["local"]["price_in"] is None
    assert rep["judge_model"] == "judge-x"


def test_report_markdown_is_self_contained():
    model_rows = {"a": _rows_with_latency("a", {"reasoning": ("quality", 0.7)}, {"reasoning_suite": 5.0})}
    md = report.render_markdown(report.build_report(model_rows, judge_model="judge-x", generated_ts="2026-06-23"))
    # the reading LLM needs context: how-to-read, methodology, judge, caveats
    assert "How to read this" in md
    assert "Methodology" in md
    assert "judge-x" in md
    assert "Caveats" in md
    assert "Quality by job" in md


def test_report_excludes_informational_metrics_from_composite():
    # realdata 'agreement' and latency must not enter the quality composite.
    rows = _rows("a", {"ranking": ("accuracy", 1.0)})
    rows.append({"run_group": "g", "model": "a", "suite": "realdata", "task_type": "ranking",
                 "metric": "agreement", "value": 0.0, "unit": "ratio", "tier": "deep",
                 "is_baseline": False, "run_ts": "t", "raw_json": None})
    rep = report.build_report({"a": rows})
    assert rep["composite"]["a"] == pytest.approx(1.0)  # agreement 0.0 excluded


def test_report_empty():
    rep = report.build_report({})
    assert rep["categories"] == []
    assert rep["composite"] == {}
    md = report.render_markdown(rep)
    assert "No benchmarked models" in md


# ---------------------------------------------------------------------------
# Dataset loader — must NOT depend on a top-level `tests` package (collides
# with an unrelated site-packages `tests` under an editable install).
# ---------------------------------------------------------------------------

def test_load_dataset_resolves_suite_modules_by_path():
    bm = harness._load_dataset("benchmark_models")
    # the functions/data run_pipeline reuses must be present
    for attr in ("benchmark_ranking", "benchmark_summarization",
                 "build_conversation_text", "TEST_MESSAGES", "TEST_CONVERSATIONS"):
        assert hasattr(bm, attr), f"benchmark_models missing {attr}"
    # loaded from the project's tests/, not site-packages
    assert bm.__file__.replace("\\", "/").endswith("tests/benchmark_models.py")


def test_load_dataset_is_cached():
    assert harness._load_dataset("benchmark_models") is harness._load_dataset("benchmark_models")


def test_load_dataset_missing_raises_actionable_error():
    with pytest.raises(ImportError, match="Benchmark dataset not found"):
        harness._load_dataset("benchmark_does_not_exist")


def test_harness_does_not_import_top_level_tests_package():
    # Guards the regression: `from tests import ...` resolves to the wrong package
    # on the Ollama PC. The harness must load suites by path instead.
    src = Path(harness.__file__).read_text(encoding="utf-8")
    assert "from tests import" not in src
    assert "import tests\n" not in src


# ---------------------------------------------------------------------------
# Benchmark DB path resolution — must be cwd-independent (the store is the same
# file no matter which folder `blipshell` is invoked from).
# ---------------------------------------------------------------------------

from blipshell.benchmark import runner  # noqa: E402
from blipshell.core.config import DEFAULT_CONFIG_PATH  # noqa: E402


def test_resolve_db_path_relative_anchors_to_repo_root_not_cwd():
    # Default config_path -> repo root (where config.yaml/data live), regardless of cwd.
    resolved = Path(runner._resolve_db_path("data/benchmark.db", None))
    assert resolved.is_absolute()
    assert resolved == (DEFAULT_CONFIG_PATH.parent / "data" / "benchmark.db").resolve()


def test_resolve_db_path_relative_anchors_to_config_dir(tmp_path):
    cfg = tmp_path / "myconfig.yaml"
    resolved = Path(runner._resolve_db_path("data/benchmark.db", str(cfg)))
    assert resolved == (tmp_path / "data" / "benchmark.db").resolve()


def test_resolve_db_path_absolute_passthrough(tmp_path):
    abs_db = tmp_path / "explicit.db"
    assert runner._resolve_db_path(str(abs_db), None) == str(abs_db)


# ---------------------------------------------------------------------------
# Discovery parsing
# ---------------------------------------------------------------------------

def test_parse_openrouter_price_and_vision():
    payload = {"data": [
        {
            "id": "vendor/model-a",
            "context_length": 128000,
            "pricing": {"prompt": "0.0000007", "completion": "0.0000028"},
            "architecture": {"input_modalities": ["text", "image"]},
            "created": 1700000000,
        },
        {
            "id": "vendor/model-b",
            "context_length": 8192,
            "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"modality": "text->text"},
        },
    ]}
    entries = discovery.parse_openrouter(payload, "ts")
    a, b = entries
    assert a["model"] == "vendor/model-a"
    assert a["price_in"] == pytest.approx(0.7)        # 0.0000007 * 1e6
    assert a["price_out"] == pytest.approx(2.8)
    assert a["vision"] is True
    assert b["vision"] is False


def test_parse_artificial_analysis_field_fallbacks():
    payload = {"data": [
        {
            "slug": "gpt-x",
            "context_window": 200000,
            "price_1m_input_tokens": 5.0,
            "price_1m_output_tokens": 15.0,
            "artificial_analysis_intelligence_index": 59,
            "median_output_tokens_per_second": 62.0,
            "median_time_to_first_token_seconds": 0.4,
        },
    ]}
    entries = discovery.parse_artificial_analysis(payload, "ts")
    e = entries[0]
    assert e["model"] == "gpt-x"
    assert e["intelligence_index"] == 59
    assert e["tok_per_s"] == pytest.approx(62.0)
    assert e["price_in"] == pytest.approx(5.0)


def test_shortlist_filters_and_new_flag():
    entries = [
        {"model": "big-cheap", "source": "openrouter", "context_length": 200000, "price_in": 0.5, "vision": False},
        {"model": "small", "source": "openrouter", "context_length": 4096, "price_in": 0.1, "vision": False},
        {"model": "pricey", "source": "openrouter", "context_length": 128000, "price_in": 10.0, "vision": True},
    ]
    short = discovery.shortlist(
        entries, min_context=8192, max_price=5.0, vision_only=False,
        known_keys={("big-cheap", "openrouter")},
    )
    models = [e["model"] for e in short]
    assert "small" not in models      # below min_context
    assert "pricey" not in models     # above max_price
    assert models == ["big-cheap"]
    assert short[0]["is_new"] is False


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------

async def test_store_record_and_baseline(tmp_path):
    db = str(tmp_path / "bench.db")
    store = await BenchmarkStore(db).initialize()
    try:
        await store.record_run(
            run_group="g1", model="m1", suite="pipeline", task_type="ranking",
            metric="accuracy", value=0.8, run_ts="t1", is_baseline=True,
        )
        await store.record_run(
            run_group="g2", model="m2", suite="pipeline", task_type="ranking",
            metric="accuracy", value=0.9, run_ts="t2",
        )
        base = await store.baseline_metrics()
        assert len(base) == 1 and base[0]["model"] == "m1"

        # New baseline supersedes the old one.
        await store.clear_baseline()
        assert await store.baseline_metrics() == []

        assert await store.latest_run_group("m2") == "g2"
        assert set(await store.models_with_runs()) == {"m1", "m2"}
    finally:
        await store.close()


async def test_store_catalog_upsert(tmp_path):
    db = str(tmp_path / "bench.db")
    store = await BenchmarkStore(db).initialize()
    try:
        await store.upsert_catalog({
            "model": "vendor/x", "source": "openrouter", "context_length": 1000,
            "price_in": 1.0, "price_out": 2.0, "vision": True,
            "intelligence_index": None, "tok_per_s": None, "ttft_s": None,
            "created_ts": None, "fetched_ts": "t1", "raw": {"a": 1},
        })
        # Upsert again with new price — should overwrite, not duplicate.
        await store.upsert_catalog({
            "model": "vendor/x", "source": "openrouter", "context_length": 1000,
            "price_in": 0.5, "price_out": 2.0, "vision": True,
            "intelligence_index": None, "tok_per_s": None, "ttft_s": None,
            "created_ts": None, "fetched_ts": "t2", "raw": None,
        })
        rows = await store.catalog_models("openrouter")
        assert len(rows) == 1
        assert rows[0]["price_in"] == pytest.approx(0.5)
        keys = await store.known_catalog_keys()
        assert ("vendor/x", "openrouter") in keys
    finally:
        await store.close()
