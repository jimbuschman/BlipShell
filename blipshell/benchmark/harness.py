"""Benchmark harness — runs a candidate model through the existing suites,
normalizes their output into metric rows, and grades open-ended outputs.

Design split (so logic is unit-testable on the dev box, per the project's
validation split — model behavior is validated only on the Ollama PC):

  * Pure scoring functions (`score_*`) take a suite's raw runner output and
    return {metric: value}. No LLM, no I/O — fully unit-testable with canned data.
  * Async `run_*` methods execute the reused suite runners, then call the pure
    scorers and the (optional) judge, and emit metric-row dicts ready for
    BenchmarkStore.record_run.

The suite runners themselves are imported and reused from tests/benchmark_*.py
(they each take a router) — this harness never reimplements task execution.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from blipshell.benchmark.judge import LLMJudge
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig, resolve_env_vars

logger = logging.getLogger(__name__)

# Job groups `run()` can execute, in run order. A run may be scoped to a subset
# (e.g. skip the slow, cloud-routed `coding` suite when comparing local
# background models). "pipeline" covers ranking/importance/contradiction/entity/
# summarization/lessons; realdata + embedding need a db_path.
BENCHMARK_JOBS = (
    "pipeline", "reasoning", "session_review", "realdata", "embedding", "coding",
)

# Repo root = <repo>/blipshell/benchmark/harness.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dataset(name: str):
    """Load tests/<name>.py (a benchmark suite runner + its data) by file path.

    The suites live in tests/, but we deliberately do NOT `import tests.<name>`:
    'tests' is a generic top-level name that collides with unrelated `tests`
    packages installed in site-packages, and an editable install of BlipShell
    only exposes the `blipshell` package on the path — so `import tests` resolves
    to the wrong package (or fails). Loading by absolute path anchored to the
    repo root is install-mode-independent and collision-proof. These modules
    import only `blipshell.*` + stdlib/rich, so path-loading is self-contained.
    """
    path = _REPO_ROOT / "tests" / f"{name}.py"
    if not path.exists():
        raise ImportError(
            f"Benchmark dataset not found: {path}. The benchmark harness reuses the "
            f"suite runners in tests/; run from a source checkout of BlipShell."
        )
    mod_name = f"_blipshell_bench_{name}"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # register before exec so any self-reference resolves
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module

ALL_ROLES = [
    "reasoning", "tool_calling", "coding", "summarization",
    "ranking", "importance", "ranking_importance", "embedding", "session_review",
]

# Metric names that count toward the composite verdict (everything else —
# latency, agreement, cost — is shown but informational).
SCORING_METRICS = {"accuracy", "quality", "tool_pass_rate"}

# Scoring is now ground-truth based (per-item), not band-based: each dataset
# item carries its own gold value (truth_rank, truth_importance, expected
# entities, expected tool+args), and the score_* functions below compare the
# model's output against it. This discriminates capable models instead of
# saturating everyone at 1.0 the way the old wide bands did.

# Synthetic substantive sessions for session_review (a non-trivial session should
# yield a filled 5-section reflection; SKIP would be wrong here). Kept self-contained
# (no DB dependency) so the benchmark is deterministic and safe.
SESSION_REVIEW_CASES = [
    {
        "summary": (
            "Debugged terrible audio from a MAX98357 on an ESP32; root cause was an "
            "impedance mismatch and a loose JST connector. Switched to JST-PH connectors "
            "and a matched 4-ohm speaker, which fixed the noise."
        ),
        "transcript": (
            "User: the MAX98357 on my esp32 sounds like garbage, mostly noise.\n"
            "Assistant: That's often impedance mismatch or loose wiring. What speaker ohms?\n"
            "User: not sure, and the connectors are dupont and pretty loose.\n"
            "Assistant: Dupont wires on audio are unreliable — switch to JST-PH 2-pin, and "
            "confirm the speaker is 4ohm to match the MAX98357. A wrong-impedance speaker "
            "also distorts.\n"
            "User: swapped to JST-PH and a 4ohm speaker — clean now. the speaker was 8ohm before.\n"
            "Assistant: Good — the 8ohm mismatch plus loose contacts explains the noise."
        ),
    },
    {
        "summary": (
            "Diagnosed a duplicate-message bug: assistant messages were saved twice because "
            "_persist_message was fire-and-forget, so _memory_db_ids was empty when the "
            "enqueue ran. Fixed by tracking persist tasks and awaiting them before enqueue."
        ),
        "transcript": (
            "User: every assistant message is getting saved twice in the DB.\n"
            "Assistant: Sounds like a race. Let me check how messages persist.\n"
            "Assistant: Found it — _persist_message uses asyncio.ensure_future (fire-and-forget), "
            "so _memory_db_ids isn't populated when _enqueue_undumped_messages runs. The processor "
            "gets memory_id=None and inserts a duplicate row.\n"
            "User: how do we fix it cleanly?\n"
            "Assistant: Track persist tasks in a list and await them in flush_pending_persists() "
            "before the enqueue reads _memory_db_ids. Make _enqueue_undumped_messages async and "
            "await the flush first."
        ),
    },
    {
        "summary": (
            "Fixed search returning almost nothing at 31k memories: compounding filters "
            "(similarity 0.5 → rank ≥3 → score floor) removed ~85% of candidates. Replaced "
            "the rank filter with an importance floor, lowered the similarity threshold, and "
            "added a recency boost."
        ),
        "transcript": (
            "User: search barely returns anything now that the corpus is huge.\n"
            "Assistant: Your filters compound — similarity 0.5, then rank≥3, then a score floor. "
            "On 31k memories that strips ~85% of candidates.\n"
            "User: what should change?\n"
            "Assistant: Drop the rank filter for a continuous importance floor (0.25), lower "
            "similarity to ~0.35, give FTS-only hits a baseline similarity, and add a recency boost.\n"
            "User: recall is way better now."
        ),
    },
    {
        "summary": (
            "Resolved cascading 429s on cloud summarization. The free tier's 20-request burst "
            "limit caused failures; added per-endpoint rate limiting and a RateLimitExhausted "
            "cascade to the next endpoint, with a local fallback that passes num_ctx."
        ),
        "transcript": (
            "User: cloud summarization keeps dying with 429s during import.\n"
            "Assistant: Free tier has a 20-request burst cap. We should rate-limit per endpoint "
            "and cascade to the next on RateLimitExhausted rather than failing.\n"
            "User: and when everything cloud is exhausted?\n"
            "Assistant: Fall back to local — but the fallback path must pass num_ctx from the "
            "endpoint config or it silently truncates context.\n"
            "User: import ran clean after that."
        ),
    },
    {
        "summary": (
            "Made the project-mode executor finish reliably. It was making 37 tool calls without "
            "completing; added a task_complete tool, per-turn state injection, and an 80%-budget "
            "wind-down message."
        ),
        "transcript": (
            "User: the executor just spins — 37 tool calls and never finishes.\n"
            "Assistant: It lacks a completion signal and state awareness. Add a task_complete tool "
            "instead of a magic string, inject a [STATE] block each turn, and a wind-down nudge at 80% budget.\n"
            "User: that's the Claude Code pattern?\n"
            "Assistant: Yes — explicit completion, state awareness, bounded tool output.\n"
            "User: it completes in ~12 steps now."
        ),
    },
    {
        "summary": (
            "Decided to migrate the vector store from ChromaDB to sqlite-vec to eliminate "
            "dual-store sync drift, accepting a one-time re-embedding cost."
        ),
        "transcript": (
            "User: the chroma/sqlite sync keeps drifting and throwing FK errors.\n"
            "Assistant: Two stores means two sources of truth. Moving vectors into sqlite-vec "
            "(vec0 tables in the same DB file) removes the drift class entirely.\n"
            "User: downside?\n"
            "Assistant: A one-time full re-embed and losing Chroma's prebuilt index, but for a "
            "single-user corpus that's a clear win.\n"
            "User: let's do it."
        ),
    },
    {
        "summary": (
            "Built a deterministic loop-integration test harness so chat-loop behavior can be "
            "validated without Ollama, using a scripted LLM client feeding canned turns."
        ),
        "transcript": (
            "User: I keep breaking the chat loop and only finding out on the Ollama PC.\n"
            "Assistant: Separate wiring from model behavior. A scripted LLM client can feed canned "
            "tool-call/text turns into the real loop and assert completion detection deterministically.\n"
            "User: so logic tests here, quality tests there?\n"
            "Assistant: Exactly — that split is the key mental model.\n"
            "User: caught two regressions already."
        ),
    },
    {
        "summary": (
            "Fixed an entity graph full of near-duplicate entities by adding a lexical-variant "
            "merge pass and an embedding-similarity merge with a version guard, archiving rather "
            "than deleting so re-mentions can revive."
        ),
        "transcript": (
            "User: the entity graph has 'esp32', 'ESP32', 'Esp32' all separate.\n"
            "Assistant: Run a lexical-variant merge first (case/punctuation), then an embedding "
            "merge with a version guard so concurrent writes don't clobber.\n"
            "User: delete the duplicates?\n"
            "Assistant: Archive, don't delete — a merged name can reappear later and you want to "
            "revive it on re-mention, not strand the mention.\n"
            "User: 7k merged, none lost."
        ),
    },
]


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------

def build_candidate_router(
    model: str,
    *,
    provider: str = "ollama",
    url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    context_tokens: Optional[int] = None,
    llm_config: Optional[LLMConfig] = None,
) -> LLMRouter:
    """Build a router that pins every task type to one candidate model+endpoint.

    Fallback is disabled — a benchmark must measure THE candidate, never silently
    drift to a fallback model.

    `llm_config` MUST be the real loaded config, not left to default. Passing
    LLMConfig() here silently ran every candidate at the 120s built-in timeout
    while config.yaml deliberately sets 300 ("5min for slower local models") —
    so a 14B local model benchmarking reasoning or coding would time out and
    score 0 on capability it actually has (hit live 2026-08-03, qwen3:14b).

    `context_tokens` is likewise load-bearing: when None, no num_ctx is sent and
    Ollama falls back to its own default, silently truncating long prompts. That
    corrupts scores rather than just slowing them, so the caller should pass the
    window the model would really get in production.
    """
    models = ModelsConfig(
        reasoning=model, tool_calling=model, coding=model, summarization=model,
        ranking=model, importance=model, ranking_importance=model,
        session_review=model, embedding=model,
    )
    ep = EndpointConfig(
        name="benchmark-candidate",
        url=url,
        provider=provider,
        api_key=api_key,
        roles=list(ALL_ROLES),
        priority=1,
        max_concurrent=1,
        context_tokens=context_tokens,
        # Per-endpoint model map ensures the candidate is used for every role,
        # even on an openai-compat endpoint (which only honors its models map).
        models={role: model for role in ALL_ROLES},
    )
    manager = EndpointManager([ep], llm_config or LLMConfig())
    return LLMRouter(models, manager, pii_enabled=False, disable_fallback=True)


# ---------------------------------------------------------------------------
# Pure scorers — take raw suite output, return {metric_key: value}.
# `metric_key` is the task_type; value is the 0-1 deterministic score.
# These are the unit-test surface.
# ---------------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    """Pearson correlation, or None if undefined (n<2 or zero variance)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / ((sxx * syy) ** 0.5)


def _corr_score(pred: list[float], truth: list[float]) -> Optional[float]:
    """Correlation mapped to 0-1 ((r+1)/2). A model that tracks the gold ordering
    scores ~1.0; random ~0.5; inverted ~0.0. If the model gave no usable variance
    (e.g. rated everything the same) while the truth does vary, that's a failure to
    discriminate -> 0.0. None only when there's nothing to score."""
    if len(pred) < 2:
        return None
    r = _pearson(pred, truth)
    if r is None:
        truth_varies = len(set(truth)) > 1
        pred_varies = len(set(pred)) > 1
        if truth_varies and not pred_varies:
            return 0.0  # flat predictions can't track a varying target
        return None
    return round(max(0.0, (r + 1.0) / 2.0), 4)


def _mae_score(pred: list[float], truth: list[float]) -> Optional[float]:
    """1 - mean absolute error, for values already on a 0-1 scale (importance)."""
    if not pred:
        return None
    mae = sum(abs(p - t) for p, t in zip(pred, truth)) / len(pred)
    return round(max(0.0, 1.0 - mae), 4)


def score_ranking(results: list[dict]) -> Optional[float]:
    """Rank-correlation of predicted rank vs gold rank (order is what matters)."""
    pairs = [(float(r["parsed"]), float(r["truth_rank"]))
             for r in results if r.get("parsed", -1) >= 0 and "truth_rank" in r]
    if len(pairs) < 2:
        return None
    return _corr_score([p for p, _ in pairs], [t for _, t in pairs])


def score_importance(results: list[dict]) -> Optional[float]:
    """Average of ordering (correlation) and calibration (1-MAE) vs gold importance."""
    pairs = [(float(r["parsed"]), float(r["truth_importance"]))
             for r in results if r.get("parsed", -1) >= 0 and "truth_importance" in r]
    if len(pairs) < 2:
        return None
    pred = [p for p, _ in pairs]
    truth = [t for _, t in pairs]
    parts = [s for s in (_corr_score(pred, truth), _mae_score(pred, truth)) if s is not None]
    return round(sum(parts) / len(parts), 4) if parts else None


def score_rank_and_importance(results: list[dict]) -> Optional[float]:
    """Combined: rank correlation + importance (corr & calibration)."""
    rank = score_ranking([
        {"parsed": r.get("rank", -1), "truth_rank": r["truth_rank"]}
        for r in results if "truth_rank" in r
    ])
    imp = score_importance([
        {"parsed": r.get("importance", -1.0), "truth_importance": r["truth_importance"]}
        for r in results if "truth_importance" in r
    ])
    parts = [x for x in (rank, imp) if x is not None]
    return round(sum(parts) / len(parts), 4) if parts else None


def score_contradiction(results: list[dict]) -> Optional[float]:
    """Mean correctness over items that produced a valid YES/NO."""
    valid = [r for r in results if r.get("parsed") in ("YES", "NO")]
    if not valid:
        return None
    return sum(1 for r in valid if r.get("correct")) / len(valid)


def _f1(predicted: set, expected: set) -> float:
    """F1 of two sets. Both empty -> 1.0 (correctly extracted nothing)."""
    if not expected and not predicted:
        return 1.0
    if not predicted or not expected:
        return 0.0
    tp = len(predicted & expected)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_entity(results: list[dict]) -> Optional[float]:
    """Mean F1 of extracted entities vs the expected entity set per item.

    Rewards extracting the RIGHT entities, not merely producing some output.
    Each result carries `extracted` (set/list of entity names) and `expected`."""
    scored = []
    for r in results:
        if str(r.get("raw", "")).startswith("ERROR:") or "expected" not in r:
            continue
        predicted = {str(e).strip().lower() for e in r.get("extracted", [])}
        expected = {str(e).strip().lower() for e in r.get("expected", [])}
        scored.append(_f1(predicted, expected))
    return round(sum(scored) / len(scored), 4) if scored else None


def score_tool_calling(results: list[dict]) -> Optional[float]:
    """Mean over non-errored items of (correct tool name AND required args present).

    `correct` is set by the dataset runner: the called tool must match
    `expected_tool` and include every key in `expected_args` with matching values."""
    valid = [r for r in results if not str(r.get("content", "")).startswith("ERROR:")]
    if not valid:
        return None
    return round(sum(1 for r in valid if r.get("correct")) / len(valid), 4)


def score_reflection_completeness(parsed: dict) -> float:
    """0-1 completeness of a parsed session reflection: the 4 content sections
    filled. (Effectiveness always parses to a default, so it isn't scored.)"""
    keys = ("what_worked", "what_didnt_work", "technical_insights", "process_insights")
    filled = sum(1 for k in keys if parsed.get(k))
    return filled / len(keys)


def realdata_agreement(results: list[dict], orig_key: str, new_key: str, tol: float) -> Optional[float]:
    """Fraction where the candidate's value agrees (within tol) with the stored
    (prior-model) value. This is a DRIFT signal, not ground truth — recorded for
    drill-down, excluded from the composite verdict."""
    hits = total = 0
    for r in results:
        new = r.get(new_key)
        orig = r.get(orig_key)
        if new is None or orig is None or new < 0:
            continue
        total += 1
        if abs(new - orig) <= tol:
            hits += 1
    return (hits / total) if total else None


def _mean_latency(*result_lists: list[dict]) -> Optional[float]:
    times = [r["time"] for results in result_lists for r in results if "time" in r]
    return round(sum(times) / len(times), 3) if times else None


def _mean(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _mean_words(responses: list[str]) -> Optional[float]:
    """Mean word count over non-errored responses — a verbosity signal so a
    longer-but-not-better model can't silently win on judged jobs."""
    counts = [len(str(r).split()) for r in responses if r and not str(r).startswith("ERROR:")]
    return round(sum(counts) / len(counts), 1) if counts else None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """Executes suites for one candidate and emits normalized metric rows."""

    def __init__(
        self,
        *,
        model: str,
        router: LLMRouter,
        run_group: str,
        run_ts: str,
        tier: str = "quick",
        judge: Optional[LLMJudge] = None,
        is_baseline: bool = False,
    ):
        self.model = model
        self.router = router
        self.run_group = run_group
        self.run_ts = run_ts
        self.tier = tier
        self.judge = judge
        self.is_baseline = is_baseline

    def _row(self, suite: str, task_type: str, metric: str, value, unit: str = "ratio", raw=None) -> dict:
        return {
            "run_group": self.run_group,
            "model": self.model,
            "suite": suite,
            "task_type": task_type,
            "metric": metric,
            "value": value,
            "unit": unit,
            "tier": self.tier,
            "is_baseline": self.is_baseline,
            "run_ts": self.run_ts,
            "raw": raw,
        }

    # -- pipeline (synthetic, always run) ---------------------------------

    async def run_pipeline(self, on_status=None) -> list[dict]:
        bm = _load_dataset("benchmark_models")

        def status(msg):
            if on_status:
                on_status(msg)

        rows: list[dict] = []
        r = self.router

        status("pipeline: ranking")
        ranking = await bm.benchmark_ranking(r)
        rows.append(self._row("pipeline", "ranking", "accuracy", score_ranking(ranking)))

        status("pipeline: importance")
        importance = await bm.benchmark_importance(r)
        rows.append(self._row("pipeline", "importance", "accuracy", score_importance(importance)))

        status("pipeline: rank+importance")
        rank_imp = await bm.benchmark_rank_and_importance(r)
        rows.append(self._row("pipeline", "rank_importance", "accuracy", score_rank_and_importance(rank_imp)))

        status("pipeline: contradiction")
        contradiction = await bm.benchmark_contradiction(r)
        rows.append(self._row("pipeline", "contradiction", "accuracy", score_contradiction(contradiction)))

        status("pipeline: entity extraction")
        entity = await bm.benchmark_entity_extraction(r)
        rows.append(self._row("pipeline", "entity", "accuracy", score_entity(entity)))

        status("pipeline: summarization")
        summ = await bm.benchmark_summarization(r)
        summ_q = await self._judge_summaries(
            [m["content"] for m in bm.TEST_MESSAGES],
            [s["response"] for s in summ],
        )
        rows.append(self._row("pipeline", "summarization", "quality", summ_q))
        rows.append(self._row("pipeline", "summarization", "length_words",
                              _mean_words([s["response"] for s in summ]), unit="words"))

        status("pipeline: lessons")
        lessons = await bm.benchmark_lessons(r)
        conv_texts = [bm.build_conversation_text(c) for c in bm.TEST_CONVERSATIONS]
        less_q = await self._judge_lessons(conv_texts, [l["response"] for l in lessons])
        rows.append(self._row("pipeline", "lessons", "quality", less_q))
        rows.append(self._row("pipeline", "lessons", "length_words",
                              _mean_words([l["response"] for l in lessons]), unit="words"))

        lat = _mean_latency(ranking, importance, rank_imp, contradiction, entity, summ, lessons)
        rows.append(self._row("pipeline", "pipeline", "latency_s", lat, unit="seconds"))
        return rows

    # -- reasoning / coding-gen / tool-calling (full tier) ----------------

    async def run_reasoning(self, on_status=None) -> list[dict]:
        br = _load_dataset("benchmark_reasoning")

        def status(msg):
            if on_status:
                on_status(msg)

        rows: list[dict] = []
        r = self.router

        status("reasoning: plans/analysis")
        reasoning = await br.benchmark_reasoning(r)
        reason_tasks = [self._reasoning_task_text(t) for t in br.REASONING_TESTS]
        reason_q = await self._judge_reasoning(reason_tasks, [x["response"] for x in reasoning])
        rows.append(self._row("reasoning", "reasoning", "quality", reason_q))
        rows.append(self._row("reasoning", "reasoning", "length_words",
                              _mean_words([x["response"] for x in reasoning]), unit="words"))

        status("reasoning: code generation")
        coding = await br.benchmark_coding(r)
        code_tasks = [t.get("prompt", "") for t in br.CODING_TESTS]
        code_q = await self._judge_reasoning(code_tasks, [x["response"] for x in coding])
        # task_type is "code_gen", NOT "coding": run_coding() also reported
        # task_type "coding" (its sandbox pass rate), and report._scoring_map
        # keys by task_type alone — so on a full run the two silently collapsed
        # and whichever row came last won. This judged generation score was
        # computed, spent judge tokens, and was then discarded.
        rows.append(self._row("reasoning", "code_gen", "quality", code_q))
        rows.append(self._row("reasoning", "code_gen", "length_words",
                              _mean_words([x["response"] for x in coding]), unit="words"))

        status("reasoning: tool calling")
        tools = await br.benchmark_tool_calling(r)
        rows.append(self._row("reasoning", "tool_calling", "tool_pass_rate", score_tool_calling(tools)))

        lat = _mean_latency(reasoning, coding, tools)
        rows.append(self._row("reasoning", "reasoning_suite", "latency_s", lat, unit="seconds"))
        return rows

    # -- real-data (full tier) --------------------------------------------

    async def run_realdata(self, db_path: str, sample: int, on_status=None) -> list[dict]:
        rd = _load_dataset("benchmark_realdata")

        def status(msg):
            if on_status:
                on_status(msg)

        rows: list[dict] = []
        r = self.router
        try:
            messages = rd.load_sample_messages(db_path, sample)
        except Exception as e:  # noqa: BLE001 — missing/empty DB shouldn't kill the run
            logger.warning("realdata: could not load sample from %s: %s", db_path, e)
            status(f"realdata: skipped ({e})")
            return rows
        if not messages:
            status("realdata: skipped (no messages)")
            return rows

        status(f"realdata: ranking ({len(messages)} msgs)")
        ranking = await rd.benchmark_ranking(r, messages)
        rows.append(self._row(
            "realdata", "ranking", "agreement",
            realdata_agreement(ranking, "original_rank", "new_rank", tol=1),
            raw={"n": len(ranking)},
        ))

        status("realdata: importance")
        importance = await rd.benchmark_importance(r, messages)
        rows.append(self._row(
            "realdata", "importance", "agreement",
            realdata_agreement(importance, "original_importance", "new_importance", tol=0.2),
        ))

        status("realdata: summarization (heuristics)")
        summ = await rd.benchmark_summarization(r, messages)
        # The realdata runner returns the prior stored summary, not the raw source
        # message, so faithfulness judging needs no source pair — instead we record
        # the runner's own heuristic compliance (concision + 3rd-person voice + no
        # echo/error) as a real, source-free quality signal.
        ok = [s for s in summ if not s["is_error"]]
        if ok:
            compliant = sum(
                1 for s in ok
                if s["under_30_words"] and not s["third_person"] and not s["is_echo"]
            )
            heuristic = compliant / len(ok)
        else:
            heuristic = None
        rows.append(self._row(
            "realdata", "summarization", "heuristic", heuristic,
            raw={"n": len(ok), "errors": len(summ) - len(ok)},
        ))

        lat = _mean_latency(ranking, importance, summ)
        rows.append(self._row("realdata", "realdata_suite", "latency_s", lat, unit="seconds"))
        return rows

    # -- session review (full tier) --------------------------------------

    async def run_session_review(self, on_status=None) -> list[dict]:
        from blipshell.llm.prompts import reflect_on_session
        from blipshell.memory.processor import MemoryProcessor

        def status(msg):
            if on_status:
                on_status(msg)

        status("session_review: reflecting on sessions")
        completeness_scores = []
        judge_scores = []
        latencies = []
        responses = []
        for case in SESSION_REVIEW_CASES:
            system, user = reflect_on_session(case["summary"], case["transcript"])
            import time
            start = time.perf_counter()
            try:
                raw = await self.router.generate(
                    TaskType.SESSION_REVIEW, user, system=system, think=False,
                )
            except Exception as e:  # noqa: BLE001
                raw = f"ERROR: {e}"
            latencies.append(time.perf_counter() - start)
            if str(raw).startswith("ERROR:"):
                continue
            responses.append(raw)
            parsed = MemoryProcessor._parse_reflection(raw)
            completeness_scores.append(score_reflection_completeness(parsed))
            if self.judge:
                judge_scores.append(await self.judge.grade_reasoning(
                    f"Review this session and produce a 5-section reflection.\n\n"
                    f"Summary: {case['summary']}\n\n{case['transcript']}",
                    raw,
                ))

        completeness = _mean(completeness_scores)
        quality = _mean(judge_scores) if self.judge else completeness
        rows = [self._row("session_review", "session_review", "quality", quality,
                          raw={"completeness": completeness, "judged": bool(self.judge)})]
        rows.append(self._row("session_review", "session_review", "length_words",
                              _mean_words(responses), unit="words"))
        lat = round(sum(latencies) / len(latencies), 3) if latencies else None
        rows.append(self._row("session_review", "session_review", "latency_s", lat, unit="seconds"))
        return rows

    # -- embedding retrieval (all tier) -----------------------------------

    async def run_embedding(self, db_path: str, ollama_url: str, on_status=None) -> list[dict]:
        from blipshell.benchmark.embedding_bench import run_embedding_benchmark

        def status(msg):
            if on_status:
                on_status(msg)

        status("embedding: re-embedding ground-truth set + scoring retrieval")
        agg = await run_embedding_benchmark(
            model=self.model, ollama_url=ollama_url, db_path=db_path,
        )
        if agg is None:
            return [self._row("embedding", "embedding", "accuracy", None,
                              raw={"note": "embedding benchmark unavailable (model/db/ground-truth)"})]
        return [self._row(
            "embedding", "embedding", "accuracy", agg["headline"],
            raw={"p_at_5": agg["p_at_5"], "r_at_10": agg["r_at_10"],
                 "mrr": agg["mrr"], "n_queries": agg["n_queries"]},
        )]

    # -- agentic coding executor (all tier; heavy) ------------------------

    async def run_coding(self, coding_tier: str = "standard", timeout: float = 300.0, on_status=None) -> list[dict]:
        import shutil

        bc = _load_dataset("benchmark_coding")

        def status(msg):
            if on_status:
                on_status(msg)

        task_list = list(bc.CODING_TASKS)
        if coding_tier in ("hard", "all"):
            task_list += list(bc.HARD_TASKS)
        if coding_tier in ("expert", "all"):
            task_list += list(bc.EXPERT_TASKS)

        status(f"coding: building sandbox ({len(task_list)} tasks — this is slow)")
        try:
            sandbox = bc.create_project_sandbox()
            context = bc.build_project_context(sandbox)
        except Exception as e:  # noqa: BLE001
            logger.warning("coding: sandbox setup failed: %s", e)
            return [self._row("coding", "coding_agentic", "accuracy", None,
                              raw={"note": f"sandbox setup failed: {e}"})]

        passed = total = 0
        times = []
        completed = 0
        try:
            for task in task_list:
                status(f"coding: {task['name']}")
                try:
                    m = await bc.run_task(self.model, task, sandbox, context, timeout=timeout)
                    passed += m.checks_passed
                    total += m.checks_total
                    times.append(getattr(m, "total_time", 0.0))
                    completed += 1
                except Exception as e:  # noqa: BLE001 — one bad task shouldn't kill the suite
                    logger.warning("coding: task '%s' errored: %s", task.get("name"), e)
                finally:
                    try:
                        bc.reset_sandbox(sandbox)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("coding: sandbox reset failed: %s", e)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

        score = (passed / total) if total else None
        # "coding_agentic" so the sandbox pass rate stops colliding with the
        # reasoning suite's judged "code_gen" score (see run_reasoning).
        rows = [self._row("coding", "coding_agentic", "accuracy", score,
                          raw={"checks_passed": passed, "checks_total": total,
                               "tasks_completed": completed, "tasks_total": len(task_list)})]
        lat = round(sum(times) / len(times), 2) if times else None
        # Latency stays keyed on the SUITE name ("coding") — report.LATENCY_SUITES
        # matches suites, not jobs, so renaming this would drop the row silently.
        rows.append(self._row("coding", "coding", "latency_s", lat, unit="seconds"))
        return rows

    async def run(
        self,
        *,
        db_path: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        full_sample: int = 50,
        coding_timeout: float = 300.0,
        jobs: Optional[set] = None,
        on_status=None,
    ) -> list[dict]:
        """Run the deep test across the requested job groups.

        `jobs=None` runs every category at full depth (the default deep run):
        pipeline + reasoning + session_review + real-data + embedding + the full
        agentic-coding suite. Intentionally heavy (~30-90 min/model); shallow
        runs don't discriminate.

        Pass a subset of `BENCHMARK_JOBS` to scope the run — e.g. comparing
        local background models (ranking/importance/summarization/reasoning/
        session_review/embedding) doesn't need the cloud-routed coding suite,
        which is the slowest part. The report merges partial runs, so a scoped
        run still shows those jobs side-by-side with other models.
        """
        if jobs is None:
            jobs = set(BENCHMARK_JOBS)
        else:
            unknown = jobs - set(BENCHMARK_JOBS)
            if unknown:
                raise ValueError(
                    f"Unknown benchmark job(s): {sorted(unknown)}. "
                    f"Valid: {sorted(BENCHMARK_JOBS)}"
                )

        rows: list[dict] = []
        if "pipeline" in jobs:
            rows += await self.run_pipeline(on_status=on_status)
        if "reasoning" in jobs:
            rows += await self.run_reasoning(on_status=on_status)
        if "session_review" in jobs:
            rows += await self.run_session_review(on_status=on_status)
        if "realdata" in jobs:
            if db_path:
                rows += await self.run_realdata(db_path, full_sample, on_status=on_status)
            else:
                logger.info("realdata skipped (no db_path)")
        if "embedding" in jobs:
            if db_path:
                rows += await self.run_embedding(db_path, ollama_url, on_status=on_status)
            else:
                logger.info("embedding skipped (no db_path)")
        if "coding" in jobs:
            rows += await self.run_coding("all", coding_timeout, on_status=on_status)
        return rows

    # -- judge helpers (no-op when judge is None) -------------------------

    async def _judge_summaries(self, sources: list[str], summaries: list[str]) -> Optional[float]:
        if not self.judge:
            return None
        scores = []
        for src, summ in zip(sources, summaries):
            if str(summ).startswith("ERROR:"):
                continue
            scores.append(await self.judge.grade_summarization(src, summ))
        return _mean(scores)

    async def _judge_lessons(self, convs: list[str], lessons: list[str]) -> Optional[float]:
        if not self.judge:
            return None
        scores = []
        for conv, lesson in zip(convs, lessons):
            if str(lesson).startswith("ERROR:"):
                continue
            scores.append(await self.judge.grade_lesson(conv, lesson))
        return _mean(scores)

    async def _judge_reasoning(self, tasks: list[str], responses: list[str]) -> Optional[float]:
        if not self.judge:
            return None
        scores = []
        for task, resp in zip(tasks, responses):
            if str(resp).startswith("ERROR:"):
                continue
            scores.append(await self.judge.grade_reasoning(task, resp))
        return _mean(scores)

    @staticmethod
    def _reasoning_task_text(test: dict) -> str:
        """Best-effort human-readable task text for a REASONING_TESTS entry."""
        return str(test.get("input") or test.get("prompt") or test.get("user_message") or test.get("name", ""))
