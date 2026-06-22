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

import logging
from typing import Optional

from blipshell.benchmark.judge import LLMJudge
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig, resolve_env_vars

logger = logging.getLogger(__name__)

ALL_ROLES = [
    "reasoning", "tool_calling", "coding", "summarization",
    "ranking", "importance", "ranking_importance", "embedding", "session_review",
]

# Metric names that count toward the composite verdict (everything else —
# latency, agreement, cost — is shown but informational).
SCORING_METRICS = {"accuracy", "quality", "tool_pass_rate"}

# Calibration bands for the curated synthetic TEST_MESSAGES (benchmark_models.py),
# aligned to that file's MESSAGE_LABELS order. A rank/importance is "correct" when
# it lands inside the band. Deliberately tolerant — we reward calibration, not
# pinpoint matching.
EXPECTED_RANK_BANDS = [
    (1, 2),  # hey (greeting)
    (1, 2),  # ok thanks (filler)
    (1, 2),  # system noise
    (3, 4),  # ESP32 audio issue
    (3, 5),  # daughter's minecraft
    (3, 4),  # code review (asst)
    (4, 5),  # desk robot decision
    (1, 3),  # sanding paint
]
EXPECTED_IMPORTANCE_BANDS = [
    (0.0, 0.35),
    (0.0, 0.35),
    (0.0, 0.35),
    (0.30, 0.75),
    (0.40, 0.90),
    (0.30, 0.80),
    (0.40, 1.00),
    (0.0, 0.45),
]
# ENTITY_TEST_SUMMARIES order: first 4 contain entities, last ("User said hello") is NONE.
EXPECTED_ENTITY_HAS = [True, True, True, True, False]

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
) -> LLMRouter:
    """Build a router that pins every task type to one candidate model+endpoint.

    Fallback is disabled — a benchmark must measure THE candidate, never silently
    drift to a fallback model.
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
    manager = EndpointManager([ep], LLMConfig())
    return LLMRouter(models, manager, pii_enabled=False, disable_fallback=True)


# ---------------------------------------------------------------------------
# Pure scorers — take raw suite output, return {metric_key: value}.
# `metric_key` is the task_type; value is the 0-1 deterministic score.
# These are the unit-test surface.
# ---------------------------------------------------------------------------

def _band_fraction(values, bands, valid=lambda v: v >= 0) -> Optional[float]:
    """Fraction of values landing inside their band. None if no valid items."""
    hits = total = 0
    for v, (lo, hi) in zip(values, bands):
        if not valid(v):
            continue
        total += 1
        if lo <= v <= hi:
            hits += 1
    return (hits / total) if total else None


def score_ranking(results: list[dict]) -> Optional[float]:
    return _band_fraction([r.get("parsed", -1) for r in results], EXPECTED_RANK_BANDS)


def score_importance(results: list[dict]) -> Optional[float]:
    return _band_fraction(
        [r.get("parsed", -1.0) for r in results], EXPECTED_IMPORTANCE_BANDS,
        valid=lambda v: v >= 0.0,
    )


def score_rank_and_importance(results: list[dict]) -> Optional[float]:
    """Average of the rank-in-band and importance-in-band fractions."""
    rank_frac = _band_fraction([r.get("rank", -1) for r in results], EXPECTED_RANK_BANDS)
    imp_frac = _band_fraction(
        [r.get("importance", -1.0) for r in results], EXPECTED_IMPORTANCE_BANDS,
        valid=lambda v: v >= 0.0,
    )
    parts = [x for x in (rank_frac, imp_frac) if x is not None]
    return (sum(parts) / len(parts)) if parts else None


def score_contradiction(results: list[dict]) -> Optional[float]:
    """Mean correctness over items that produced a valid YES/NO."""
    valid = [r for r in results if r.get("parsed") in ("YES", "NO")]
    if not valid:
        return None
    return sum(1 for r in valid if r.get("correct")) / len(valid)


def score_entity(results: list[dict]) -> Optional[float]:
    """Fraction where presence-of-entities matches expectation."""
    hits = total = 0
    for r, expect_has in zip(results, EXPECTED_ENTITY_HAS):
        if str(r.get("raw", "")).startswith("ERROR:"):
            continue
        total += 1
        has = r.get("triple_count", 0) > 0
        if has == expect_has:
            hits += 1
    return (hits / total) if total else None


def score_tool_calling(results: list[dict]) -> Optional[float]:
    """tool_pass_rate = mean(correct) over non-errored items."""
    valid = [r for r in results if not str(r.get("content", "")).startswith("ERROR:")]
    if not valid:
        return None
    return sum(1 for r in valid if r.get("correct")) / len(valid)


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
        from tests import benchmark_models as bm

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

        status("pipeline: lessons")
        lessons = await bm.benchmark_lessons(r)
        conv_texts = [bm.build_conversation_text(c) for c in bm.TEST_CONVERSATIONS]
        less_q = await self._judge_lessons(conv_texts, [l["response"] for l in lessons])
        rows.append(self._row("pipeline", "lessons", "quality", less_q))

        lat = _mean_latency(ranking, importance, rank_imp, contradiction, entity, summ, lessons)
        rows.append(self._row("pipeline", "pipeline", "latency_s", lat, unit="seconds"))
        return rows

    # -- reasoning / coding-gen / tool-calling (full tier) ----------------

    async def run_reasoning(self, on_status=None) -> list[dict]:
        from tests import benchmark_reasoning as br

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

        status("reasoning: code generation")
        coding = await br.benchmark_coding(r)
        code_tasks = [t.get("prompt", "") for t in br.CODING_TESTS]
        code_q = await self._judge_reasoning(code_tasks, [x["response"] for x in coding])
        rows.append(self._row("reasoning", "coding", "quality", code_q))

        status("reasoning: tool calling")
        tools = await br.benchmark_tool_calling(r)
        rows.append(self._row("reasoning", "tool_calling", "tool_pass_rate", score_tool_calling(tools)))

        lat = _mean_latency(reasoning, coding, tools)
        rows.append(self._row("reasoning", "reasoning_suite", "latency_s", lat, unit="seconds"))
        return rows

    # -- real-data (full tier) --------------------------------------------

    async def run_realdata(self, db_path: str, sample: int, on_status=None) -> list[dict]:
        from tests import benchmark_realdata as rd

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

        from tests import benchmark_coding as bc

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
            return [self._row("coding", "coding", "accuracy", None,
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
        rows = [self._row("coding", "coding", "accuracy", score,
                          raw={"checks_passed": passed, "checks_total": total,
                               "tasks_completed": completed, "tasks_total": len(task_list)})]
        lat = round(sum(times) / len(times), 2) if times else None
        rows.append(self._row("coding", "coding", "latency_s", lat, unit="seconds"))
        return rows

    async def run(
        self,
        *,
        db_path: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        full_sample: int = 50,
        coding_tier: str = "standard",
        coding_timeout: float = 300.0,
        on_status=None,
    ) -> list[dict]:
        """Run the configured tier and return all metric rows.

        Tiers: quick = pipeline; full = + reasoning + session_review + real-data;
        all = full + embedding + agentic coding executor (heavy).
        """
        rows = await self.run_pipeline(on_status=on_status)
        if self.tier in ("full", "all"):
            rows += await self.run_reasoning(on_status=on_status)
            rows += await self.run_session_review(on_status=on_status)
            if db_path:
                rows += await self.run_realdata(db_path, full_sample, on_status=on_status)
        if self.tier == "all":
            if db_path:
                rows += await self.run_embedding(db_path, ollama_url, on_status=on_status)
            else:
                logger.info("embedding: skipped (no db_path)")
            rows += await self.run_coding(coding_tier, coding_timeout, on_status=on_status)
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
