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

from blipshell.benchmark import discovery, embedding_bench, harness, scoreboard
from blipshell.benchmark.judge import LLMJudge
from blipshell.benchmark.store import BenchmarkStore


# ---------------------------------------------------------------------------
# Pure scorers
# ---------------------------------------------------------------------------

def test_score_ranking_band():
    # 8 items aligned to EXPECTED_RANK_BANDS; make 6/8 land in band.
    parsed = [1, 1, 1, 3, 4, 3, 5, 5]  # last (sanding) band is (1,3); 5 is out
    # index 7 expected (1,3) -> 5 out; all others in band -> 7/8
    results = [{"parsed": p} for p in parsed]
    assert harness.score_ranking(results) == pytest.approx(7 / 8)


def test_score_ranking_ignores_errors():
    results = [{"parsed": -1}] * 8  # all errored
    assert harness.score_ranking(results) is None


def test_score_importance_band():
    vals = [0.1, 0.1, 0.1, 0.5, 0.6, 0.5, 0.7, 0.9]  # last band (0,0.45) -> 0.9 out
    results = [{"parsed": v} for v in vals]
    assert harness.score_importance(results) == pytest.approx(7 / 8)


def test_score_contradiction_only_valid():
    results = [
        {"parsed": "YES", "correct": True},
        {"parsed": "NO", "correct": True},
        {"parsed": "INVALID", "correct": False},  # excluded
        {"parsed": "YES", "correct": False},
    ]
    assert harness.score_contradiction(results) == pytest.approx(2 / 3)


def test_score_entity_presence_match():
    # EXPECTED_ENTITY_HAS = [T,T,T,T,F]
    results = [
        {"triple_count": 2, "raw": "a|b|c"},   # T ok
        {"triple_count": 0, "raw": ""},          # T miss
        {"triple_count": 1, "raw": "x"},        # T ok
        {"triple_count": 3, "raw": "y"},        # T ok
        {"triple_count": 0, "raw": "NONE"},     # F ok
    ]
    assert harness.score_entity(results) == pytest.approx(4 / 5)


def test_score_entity_skips_errors():
    results = [{"triple_count": 0, "raw": "ERROR: boom"}] * 5
    assert harness.score_entity(results) is None


def test_score_tool_calling():
    results = [
        {"correct": True, "content": "ok"},
        {"correct": False, "content": "ok"},
        {"correct": True, "content": "ERROR: x"},  # excluded
    ]
    assert harness.score_tool_calling(results) == pytest.approx(1 / 2)


def test_score_rank_and_importance_average():
    results = [
        {"rank": lo, "importance": (b[0] + b[1]) / 2}
        for lo, b in zip(
            [b[0] for b in harness.EXPECTED_RANK_BANDS],
            harness.EXPECTED_IMPORTANCE_BANDS,
        )
    ]
    # all rank at band-low, all importance mid -> both fractions 1.0 -> 1.0
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
# Scoreboard verdict math
# ---------------------------------------------------------------------------

def _rows(model, scores, is_baseline=False):
    """scores: {task_type: (metric, value)}."""
    return [
        {
            "run_group": "g", "model": model, "suite": "pipeline",
            "task_type": tt, "metric": metric, "value": val, "unit": "ratio",
            "tier": "quick", "is_baseline": is_baseline, "run_ts": "t", "raw_json": None,
        }
        for tt, (metric, val) in scores.items()
    ]


def test_scoreboard_better_worse_tie():
    cand = _rows("cand", {
        "ranking": ("accuracy", 0.90),     # +0.20 better
        "coding": ("quality", 0.50),       # -0.20 worse
        "tool_calling": ("tool_pass_rate", 0.80),  # +0.02 tie
    })
    base = _rows("base", {
        "ranking": ("accuracy", 0.70),
        "coding": ("quality", 0.70),
        "tool_calling": ("tool_pass_rate", 0.78),
    }, is_baseline=True)
    sb = scoreboard.build_scoreboard(cand, base, verdict_delta=0.05)
    verdicts = {t["task_type"]: t["verdict"] for t in sb["tasks"]}
    assert verdicts == {"ranking": "better", "coding": "worse", "tool_calling": "tie"}
    assert sb["have_baseline"] is True
    # equal weights: composite cand = (0.9+0.5+0.8)/3, base = (0.7+0.7+0.78)/3
    # composite is rounded to 4dp for display
    assert sb["composite_candidate"] == pytest.approx((0.9 + 0.5 + 0.8) / 3, abs=1e-3)
    assert sb["composite_baseline"] == pytest.approx((0.7 + 0.7 + 0.78) / 3, abs=1e-3)


def test_scoreboard_no_baseline():
    cand = _rows("cand", {"ranking": ("accuracy", 0.8)})
    sb = scoreboard.build_scoreboard(cand, [], verdict_delta=0.05)
    assert sb["have_baseline"] is False
    assert sb["overall"] == "no_baseline"
    assert sb["composite_candidate"] == pytest.approx(0.8)


def test_scoreboard_weights_and_latency_excluded():
    cand = _rows("cand", {
        "ranking": ("accuracy", 1.0),
        "coding": ("quality", 0.0),
    })
    # add a latency row that must NOT enter the composite
    cand.append({
        "run_group": "g", "model": "cand", "suite": "pipeline", "task_type": "pipeline",
        "metric": "latency_s", "value": 12.3, "unit": "seconds", "tier": "quick",
        "is_baseline": False, "run_ts": "t", "raw_json": None,
    })
    sb = scoreboard.build_scoreboard(cand, [], task_weights={"ranking": 3.0, "coding": 1.0})
    # weighted: (3*1.0 + 1*0.0)/4 = 0.75
    assert sb["composite_candidate"] == pytest.approx(0.75)
    assert sb["latency"] == {"pipeline": 12.3}
    # only two scoring tasks
    assert {t["task_type"] for t in sb["tasks"]} == {"ranking", "coding"}


# ---------------------------------------------------------------------------
# Leaderboard (cross-model, per-job incumbent comparison)
# ---------------------------------------------------------------------------

from blipshell.benchmark import leaderboard  # noqa: E402


def test_leaderboard_flags_switch_when_candidate_beats_incumbent():
    model_rows = {
        "incumbent-model": _rows("incumbent-model", {"coding": ("quality", 0.70)}),
        "challenger": _rows("challenger", {"coding": ("quality", 0.80)}),  # +0.10
    }
    lb = leaderboard.build_leaderboard(
        model_rows, {"coding": "incumbent-model"}, verdict_delta=0.05,
    )
    task = next(t for t in lb["tasks"] if t["task_type"] == "coding")
    assert task["incumbent"] == "incumbent-model"
    assert task["incumbent_score"] == pytest.approx(0.70)
    assert task["best_model"] == "challenger"
    assert task["switch"] is True
    assert task["delta"] == pytest.approx(0.10)
    assert lb["switch_suggestions"] == [
        {"task_type": "coding", "from": "incumbent-model", "to": "challenger", "delta": pytest.approx(0.10)}
    ]


def test_leaderboard_no_switch_within_threshold():
    model_rows = {
        "incumbent-model": _rows("incumbent-model", {"coding": ("quality", 0.70)}),
        "challenger": _rows("challenger", {"coding": ("quality", 0.73)}),  # +0.03 < 0.05
    }
    lb = leaderboard.build_leaderboard(
        model_rows, {"coding": "incumbent-model"}, verdict_delta=0.05,
    )
    task = next(t for t in lb["tasks"] if t["task_type"] == "coding")
    # challenger is still "best", but the margin doesn't clear the switch threshold.
    assert task["best_model"] == "challenger"
    assert task["switch"] is False
    assert lb["switch_suggestions"] == []


def test_leaderboard_incumbent_not_benchmarked():
    # config routes coding to a model we never benchmarked -> no delta, no switch.
    model_rows = {
        "challenger": _rows("challenger", {"coding": ("quality", 0.90)}),
    }
    lb = leaderboard.build_leaderboard(
        model_rows, {"coding": "some-unbenchmarked-model"}, verdict_delta=0.05,
    )
    task = next(t for t in lb["tasks"] if t["task_type"] == "coding")
    assert task["incumbent"] == "some-unbenchmarked-model"
    assert task["incumbent_benchmarked"] is False
    assert task["incumbent_score"] is None
    assert task["delta"] is None
    assert task["switch"] is False
    assert lb["switch_suggestions"] == []


def test_leaderboard_incumbent_wins_no_suggestion():
    model_rows = {
        "incumbent-model": _rows("incumbent-model", {"coding": ("quality", 0.90)}),
        "challenger": _rows("challenger", {"coding": ("quality", 0.70)}),
    }
    lb = leaderboard.build_leaderboard(
        model_rows, {"coding": "incumbent-model"}, verdict_delta=0.05,
    )
    task = next(t for t in lb["tasks"] if t["task_type"] == "coding")
    assert task["best_model"] == "incumbent-model"
    assert task["switch"] is False
    assert lb["switch_suggestions"] == []


def test_leaderboard_composite_and_known_job_ordering():
    model_rows = {
        "a": _rows("a", {
            "coding": ("quality", 0.80),
            "ranking": ("accuracy", 0.60),
        }),
        "b": _rows("b", {
            "coding": ("quality", 0.40),
            "ranking": ("accuracy", 1.00),
        }),
    }
    lb = leaderboard.build_leaderboard(
        model_rows, {"coding": "a", "ranking": "a"},
        task_weights={"coding": 3.0, "ranking": 1.0},
    )
    # composite a = (3*0.8 + 1*0.6)/4 = 0.75 ; b = (3*0.4 + 1*1.0)/4 = 0.55
    assert lb["composite"]["a"] == pytest.approx(0.75)
    assert lb["composite"]["b"] == pytest.approx(0.55)
    # known jobs ordered per TASK_TO_CONFIG_FIELD (ranking before coding)
    order = [t["task_type"] for t in lb["tasks"]]
    assert order == ["ranking", "coding"]


def test_leaderboard_empty():
    lb = leaderboard.build_leaderboard({}, {}, verdict_delta=0.05)
    assert lb["tasks"] == []
    assert lb["switch_suggestions"] == []
    assert lb["composite"] == {}


def _rows_with_latency(model, scores, lats, is_baseline=False):
    rows = _rows(model, scores, is_baseline)
    rows += [{
        "run_group": "g", "model": model, "suite": "s", "task_type": suite,
        "metric": "latency_s", "value": v, "unit": "seconds", "tier": "full",
        "is_baseline": is_baseline, "run_ts": "t", "raw_json": None,
    } for suite, v in lats.items()]
    return rows


def test_leaderboard_surfaces_suite_latency_per_job():
    model_rows = {
        "incumbent-model": _rows_with_latency(
            "incumbent-model",
            {"ranking": ("accuracy", 0.80), "reasoning": ("quality", 0.80)},
            {"pipeline": 1.4, "reasoning_suite": 2.1},
        ),
        "challenger": _rows_with_latency(
            "challenger",
            {"ranking": ("accuracy", 0.90), "reasoning": ("quality", 0.82)},
            {"pipeline": 6.0, "reasoning_suite": 8.2},
        ),
    }
    incumbents = {"ranking": "incumbent-model", "reasoning": "incumbent-model"}
    lb = leaderboard.build_leaderboard(model_rows, incumbents, verdict_delta=0.05)

    ranking = next(t for t in lb["tasks"] if t["task_type"] == "ranking")
    # ranking is a pipeline-suite job -> shares the pipeline latency
    assert ranking["incumbent_latency"] == pytest.approx(1.4)
    assert ranking["best_latency"] == pytest.approx(6.0)   # challenger won on quality
    assert ranking["switch"] is True

    reasoning = next(t for t in lb["tasks"] if t["task_type"] == "reasoning")
    # +0.02 within threshold -> no switch; latency still surfaced for the best model
    assert reasoning["switch"] is False
    assert reasoning["best_latency"] is not None
    # latency map round-trips per model/suite
    assert lb["latency"]["challenger"]["reasoning_suite"] == pytest.approx(8.2)


def test_job_latency_falls_back_and_handles_missing():
    # coding has no own latency row -> falls back to reasoning_suite
    assert leaderboard._job_latency({"reasoning_suite": 5.0}, "coding") == 5.0
    # coding's own latency wins when present
    assert leaderboard._job_latency({"coding": 9.0, "reasoning_suite": 5.0}, "coding") == 9.0
    # embedding has no measured latency
    assert leaderboard._job_latency({"pipeline": 1.0}, "embedding") is None
    # nothing measured
    assert leaderboard._job_latency({}, "ranking") is None


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
