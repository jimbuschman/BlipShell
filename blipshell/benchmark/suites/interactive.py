"""Interactive suite — tool calling and coding (multi-turn harness).

Matches benchmark-spec.md Section 2, Roles 15-16.

Two tiers per role:
  Tier 1 (router-only): Works with any router including mock.
  Tier 2 (agent harness): Requires config_path for full Agent bootstrap. Live only.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Callable

from blipshell.benchmark.models import SuiteResult, TaskScore
from blipshell.benchmark.suites.base import BenchmarkSuite

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

# Roles in this suite and their task_name keys
ROLES = ["tool_calling", "coding"]


# ---------------------------------------------------------------------------
# Lightweight stream collector (simplified from scripts/test_executor.py)
# ---------------------------------------------------------------------------

class _StreamCollector:
    """Captures tool calls and errors from agent.chat() streaming output."""

    def __init__(self):
        self.tool_calls: list[dict] = []
        self.errors: list[str] = []
        self.raw_output: list[str] = []

    def on_token(self, chunk: str):
        self.raw_output.append(chunk)

        # Parse tool call markers: [Tool: name ...]
        tool_match = re.search(r"\[Tool: (\w+)", chunk)
        if tool_match:
            self.tool_calls.append({"name": tool_match.group(1), "success": True})

        # Parse tool result errors: [Result: Error: ...]
        result_match = re.search(r"\[Result: (.+?)\]", chunk)
        if result_match and result_match.group(1).startswith("Error:"):
            self.errors.append(result_match.group(1)[:200])
            if self.tool_calls:
                self.tool_calls[-1]["success"] = False

        # Fatal errors in stream
        if "FATAL ERROR" in chunk:
            self.errors.append(chunk.strip()[:200])

    @property
    def raw_text(self) -> str:
        return "".join(self.raw_output)

    @property
    def completed(self) -> bool:
        raw = self.raw_text
        return (
            "[Task complete signal received]" in raw
            or "[No tool calls — treating as complete]" in raw
            or "[Inline text used as completion]" in raw
        )


# ---------------------------------------------------------------------------
# Agent bootstrap helper
# ---------------------------------------------------------------------------

async def _bootstrap_agent(config_path: str, model: str):
    """Create and initialize an Agent with overridden model for tool_calling/coding.

    Kills the memory worker immediately after init to avoid "database is locked"
    errors and the 270s shutdown timeout.  The benchmark doesn't need background
    memory processing.

    Returns (agent, session_id). Caller must clean up via _cleanup_agent().
    """
    from blipshell.core.agent import Agent
    from blipshell.core.config import ConfigManager

    config_manager = ConfigManager(config_path)
    config = config_manager.load()

    # Override model for interactive task types
    config.models.tool_calling = model
    config.models.coding = model
    config.models.reasoning = model  # some tools trigger reasoning

    # Disable fallback models — benchmark must test the specified model only.
    # Without this, models that don't support tools (e.g. glm4) silently
    # fall back to gpt-oss and get credited with gpt-oss's scores.
    config.models.tool_calling_fallback = None
    config.models.coding_fallback = None
    config.models.reasoning_fallback = None
    config.models.summarization_fallback = None
    config.models.ranking_importance_fallback = None

    # Strip cloud endpoints — benchmark must only use local Ollama.
    # Without this, the Agent's router may route sub-calls (e.g. summarization
    # during memory processing) to Groq/Gemini endpoints from config.yaml.
    config.endpoints = [ep for ep in config.endpoints if ep.provider == "ollama"]

    agent = Agent(config, config_manager)

    # Prevent the memory worker from ever starting — benchmark doesn't need
    # background processing.  Previous approach (shutdown after init) left
    # in-flight work items that hit "ChromaStore is closed" errors.
    from blipshell.memory.worker import MemoryWorker
    _orig_start = MemoryWorker.start
    MemoryWorker.start = lambda self: None  # no-op
    try:
        await agent.initialize()
    finally:
        MemoryWorker.start = _orig_start  # restore for other code

    # The worker was created but never started — null it out
    agent._memory_worker = None

    # Disable the router's endpoint-level fallback too
    agent.router._disable_fallback = True

    # Headless ask_user callback
    async def _headless_ask_user(question: str) -> str:
        return "Make your best judgment."

    agent.set_ask_user_callback(_headless_ask_user)

    session_id = await agent.start_session()
    return agent, session_id


async def _cleanup_agent(agent):
    """Safely clean up an Agent after benchmark testing.

    Uses force_cleanup() instead of end_session() to avoid the full
    session-close pipeline (summaries, lessons, worker drain).
    """
    try:
        await agent.force_cleanup()
    except Exception:
        logger.debug("force_cleanup error", exc_info=True)


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

class InteractiveSuite(BenchmarkSuite):
    name = "interactive"
    description = "Tool calling accuracy and coding task completion (multi-turn)"
    task_types = ["tool_calling", "coding"]
    needs_db = False
    needs_router = True
    quick_samples = 0  # fixed test set
    thorough_samples = 0

    async def run(
        self,
        models: list[str],
        *,
        router_factory: Callable[[str], LLMRouter] | None = None,
        config: BlipShellConfig | None = None,
        db_path: str | None = None,
        ollama_url: str = "http://localhost:11434",
        thorough: bool = False,
        on_status: Callable[[str], None] | None = None,
        config_path: str | None = None,
        on_model_done: Callable[[SuiteResult], None] | None = None,
    ) -> list[SuiteResult]:
        results = []
        for model in models:
            if on_status:
                on_status(f"[interactive] Testing {model}")
            router = router_factory(model) if router_factory else None

            scores = []
            total_start = time.monotonic()

            scores.append(await self._score_tool_calling(
                model, router, config_path, on_status,
            ))
            scores.append(await self._score_coding(
                model, router, config_path, thorough, on_status,
            ))

            elapsed = time.monotonic() - total_start
            sr = SuiteResult(
                suite_name=self.name, model=model, scores=scores,
                elapsed_s=round(elapsed, 1),
            )
            results.append(sr)

            # Incremental callback — lets runner save after each model
            if on_model_done:
                on_model_done(sr)

        return results

    # ------------------------------------------------------------------
    # Role 15: Tool Calling
    # Spec: tool_selection, no_errors, completion, efficiency
    # ------------------------------------------------------------------
    async def _score_tool_calling(
        self,
        model: str,
        router: LLMRouter | None,
        config_path: str | None,
        on_status: Callable | None,
    ) -> TaskScore:
        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        # --- Tier 1: Tool selection via API (4 tests) ---
        tier1_scores = await self._tier1_tool_selection(router, times, detail_rows)
        case_scores.extend(tier1_scores)

        # --- Tier 2: Agent tool tests (14 tests, live only) ---
        if config_path:
            tier2_scores, tier2_errors = await self._tier2_agent_tools(
                model, config_path, times, detail_rows, on_status,
            )
            case_scores.extend(tier2_scores)
            errors += tier2_errors
        else:
            if on_status:
                on_status("  tool_calling tier 2: skipped (no config_path)")

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  tool_calling: {avg_quality:.2f} quality, "
                      f"{len(case_scores)} cases, {errors} errors")

        return TaskScore(
            task_name="tool_calling",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(case_scores),
            errors=errors,
            detail={"cases": detail_rows},
        )

    async def _tier1_tool_selection(
        self,
        router: LLMRouter | None,
        times: list[float],
        detail_rows: list[dict],
    ) -> list[float]:
        """Tier 1: Tool selection accuracy via client.chat() with tool schemas."""
        if router is None:
            return []

        # Check if router supports get_model_and_client (mock doesn't)
        if not hasattr(router, "get_model_and_client"):
            return []

        try:
            from tests.benchmark_reasoning import (
                MOCK_TOOLS,
                TOOL_CALLING_TESTS,
                extract_response,
                extract_tool_call_info,
            )
            from blipshell.llm.router import TaskType
        except ImportError:
            logger.debug("Could not import tool calling test data")
            return []

        try:
            model, client = await router.get_model_and_client(TaskType.TOOL_CALLING)
        except Exception:
            return []

        if not client:
            return []

        scores = []
        for test in TOOL_CALLING_TESTS:
            messages = [{"role": "user", "content": test["message"]}]
            try:
                start = time.monotonic()
                response = await client.chat(
                    messages=messages, model=model, tools=MOCK_TOOLS,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)

                content, tool_calls = extract_response(response)

                called_tools = []
                if tool_calls:
                    for tc in tool_calls:
                        name, args = extract_tool_call_info(tc)
                        called_tools.append({"name": name, "args": args})

                checks = {}
                # 1. Tool selection: correct tool called
                checks["tool_selection"] = 1.0 if any(
                    t["name"] == test["expected_tool"] for t in called_tools
                ) else 0.0
                # 2. No errors
                checks["no_errors"] = 1.0
                # 3. Completion: non-empty response or tool called
                checks["completion"] = 1.0 if (content or called_tools) else 0.0
                # 4. Efficiency: <=2 tool calls
                checks["efficiency"] = 1.0 if len(called_tools) <= 2 else 0.0

                score = sum(checks.values()) / 4
                scores.append(score)
                detail_rows.append({
                    "id": f"t1_{test['name']}",
                    "tier": 1,
                    "output": (content or "")[:60],
                    "tools_called": [t["name"] for t in called_tools],
                    "checks": {k: round(v, 2) for k, v in checks.items()},
                    "score": round(score, 2),
                })
            except Exception as e:
                logger.debug("tier1 tool selection error: %s", e)
                scores.append(0.0)
                detail_rows.append({
                    "id": f"t1_{test['name']}",
                    "tier": 1,
                    "output": f"ERROR: {e}"[:60],
                    "checks": {},
                    "score": 0.0,
                })
        return scores

    async def _tier2_agent_tools(
        self,
        model: str,
        config_path: str,
        times: list[float],
        detail_rows: list[dict],
        on_status: Callable | None,
    ) -> tuple[list[float], int]:
        """Tier 2: Full agent harness tool tests (14 simple-chat tests)."""
        try:
            from scripts.test_executor import SIMPLE_CHAT_TESTS
        except ImportError:
            logger.debug("Could not import SIMPLE_CHAT_TESTS")
            return [], 0

        scores = []
        errors = 0

        agent = None
        try:
            agent, session_id = await _bootstrap_agent(config_path, model)
        except Exception as e:
            logger.debug("Agent bootstrap failed: %s", e)
            return [], 1

        try:
            for test in SIMPLE_CHAT_TESTS:
                collector = _StreamCollector()
                force_plan = test.get("force_plan", False)

                try:
                    start = time.monotonic()
                    result = await agent.chat(
                        user_message=test["task"],
                        on_token=collector.on_token,
                        force_plan=force_plan,
                    )
                    elapsed = time.monotonic() - start
                    times.append(elapsed)
                except Exception as e:
                    logger.debug("tier2 agent error on %s: %s", test["name"], e)
                    errors += 1
                    scores.append(0.0)
                    detail_rows.append({
                        "id": f"t2_{test['name']}",
                        "tier": 2,
                        "output": f"ERROR: {e}"[:60],
                        "checks": {},
                        "score": 0.0,
                    })
                    continue

                tool_names = [tc["name"] for tc in collector.tool_calls]
                checks = {}
                total_checks = 0

                # Completion check
                if test.get("expect_complete"):
                    completed = (
                        collector.completed
                        or (result and len(result.strip()) > 0)
                    )
                    checks["completion"] = 1.0 if completed else 0.0
                    total_checks += 1

                # Expected tools
                for expected_tool in test.get("expect_tools", []):
                    found = expected_tool in tool_names
                    checks[f"used_{expected_tool}"] = 1.0 if found else 0.0
                    total_checks += 1

                # No-tools check (conversational tests)
                if test.get("expect_no_tools"):
                    checks["no_tools"] = 1.0 if len(tool_names) == 0 else 0.0
                    total_checks += 1

                # Max tool calls budget
                max_tc = test.get("expect_max_tool_calls")
                if max_tc is not None:
                    checks["efficiency"] = 1.0 if len(tool_names) <= max_tc else 0.0
                    total_checks += 1

                # No errors
                no_err = len(collector.errors) == 0
                checks["no_errors"] = 1.0 if no_err else 0.0
                total_checks += 1

                score = sum(checks.values()) / total_checks if total_checks > 0 else 0.0
                scores.append(score)
                detail_rows.append({
                    "id": f"t2_{test['name']}",
                    "tier": 2,
                    "category": test.get("category", ""),
                    "output": (result or "")[:60],
                    "tools_called": tool_names[:5],
                    "checks": {k: round(v, 2) for k, v in checks.items()},
                    "score": round(score, 2),
                })

                if on_status and len(scores) % 5 == 0:
                    on_status(f"  tool_calling tier 2: {len(scores)}/{len(SIMPLE_CHAT_TESTS)} tests")
        finally:
            if agent:
                await _cleanup_agent(agent)

        return scores, errors

    # ------------------------------------------------------------------
    # Role 16: Coding
    # Spec: completion, check_accuracy, efficiency, no_errors
    # ------------------------------------------------------------------
    async def _score_coding(
        self,
        model: str,
        router: LLMRouter | None,
        config_path: str | None,
        thorough: bool,
        on_status: Callable | None,
    ) -> TaskScore:
        times: list[float] = []
        case_scores: list[float] = []
        errors = 0
        detail_rows: list[dict] = []

        # --- Tier 1: Code generation (8 tasks, router-only) ---
        tier1_scores = await self._tier1_code_gen(router, times, detail_rows)
        case_scores.extend(tier1_scores)

        # --- Tier 2: Executor tasks (65 stress tests, live + thorough) ---
        if config_path and thorough:
            tier2_scores, tier2_errors = await self._tier2_executor(
                model, config_path, times, detail_rows, on_status,
            )
            case_scores.extend(tier2_scores)
            errors += tier2_errors
        elif config_path and not thorough:
            if on_status:
                on_status("  coding tier 2: skipped (use --sample >30 for thorough)")
        else:
            if on_status:
                on_status("  coding tier 2: skipped (no config_path)")

        avg_quality = sum(case_scores) / len(case_scores) if case_scores else 0
        avg_speed = sum(times) / len(times) if times else 0

        if on_status:
            on_status(f"  coding: {avg_quality:.2f} quality, "
                      f"{len(case_scores)} cases, {errors} errors")

        return TaskScore(
            task_name="coding",
            quality=round(avg_quality, 3),
            speed_s=round(avg_speed, 2),
            samples=len(case_scores),
            errors=errors,
            detail={"cases": detail_rows},
        )

    async def _tier1_code_gen(
        self,
        router: LLMRouter | None,
        times: list[float],
        detail_rows: list[dict],
    ) -> list[float]:
        """Tier 1: Single-turn code generation (8 tasks from coding.py)."""
        if router is None:
            return []

        try:
            from blipshell.benchmark.suites.coding import CODE_TASKS, _extract_code
            from blipshell.llm.router import TaskType
        except ImportError:
            logger.debug("Could not import CODE_TASKS")
            return []

        scores = []
        for task in CODE_TASKS:
            try:
                start = time.monotonic()
                raw = await router.generate(
                    TaskType.CODING, task["prompt"], think=False,
                )
                elapsed = time.monotonic() - start
                times.append(elapsed)
            except Exception as e:
                logger.debug("tier1 code gen error on %s: %s", task["name"], e)
                scores.append(0.0)
                detail_rows.append({
                    "id": f"t1_{task['name']}",
                    "tier": 1,
                    "output": f"ERROR: {e}"[:60],
                    "checks": {},
                    "score": 0.0,
                })
                continue

            code = _extract_code(raw)
            passed = 0
            total = len(task["checks"])
            check_results = {}

            for check_name, check_fn in task["checks"]:
                try:
                    ok = check_fn(code)
                except Exception:
                    ok = False
                check_results[check_name] = 1.0 if ok else 0.0
                if ok:
                    passed += 1

            score = passed / total if total > 0 else 0.0
            scores.append(score)
            detail_rows.append({
                "id": f"t1_{task['name']}",
                "tier": 1,
                "output": code[:60],
                "checks": check_results,
                "score": round(score, 2),
            })
        return scores

    async def _tier2_executor(
        self,
        model: str,
        config_path: str,
        times: list[float],
        detail_rows: list[dict],
        on_status: Callable | None,
    ) -> tuple[list[float], int]:
        """Tier 2: Full executor coding tasks (65 stress tests)."""
        try:
            from scripts.test_executor import STRESS_TESTS
        except ImportError:
            logger.debug("Could not import STRESS_TESTS")
            return [], 0

        scores = []
        errors = 0

        agent = None
        try:
            agent, session_id = await _bootstrap_agent(config_path, model)
        except Exception as e:
            logger.debug("Agent bootstrap failed: %s", e)
            return [], 1

        try:
            for i, test in enumerate(STRESS_TESTS):
                collector = _StreamCollector()
                force_plan = test.get("force_plan", True)

                try:
                    start = time.monotonic()
                    result = await agent.chat(
                        user_message=test["task"],
                        on_token=collector.on_token,
                        force_plan=force_plan,
                    )
                    elapsed = time.monotonic() - start
                    times.append(elapsed)
                except Exception as e:
                    logger.debug("tier2 executor error on %s: %s", test["name"], e)
                    errors += 1
                    scores.append(0.0)
                    detail_rows.append({
                        "id": f"t2_{test['name']}",
                        "tier": 2,
                        "output": f"ERROR: {e}"[:60],
                        "checks": {},
                        "score": 0.0,
                    })
                    continue

                tool_names = [tc["name"] for tc in collector.tool_calls]
                checks = {}

                # 1. Completion: task_complete or substantial response
                completed = (
                    collector.completed
                    or (result and len(result.strip()) > 50)
                )
                checks["completion"] = 1.0 if completed else 0.0

                # 2. Accuracy: expected tools used
                expected_tools = test.get("expect_tools", [])
                if expected_tools:
                    found = sum(1 for et in expected_tools if et in tool_names)
                    checks["accuracy"] = found / len(expected_tools)
                else:
                    # No specific tools expected — count as pass if completed
                    checks["accuracy"] = 1.0 if completed else 0.0

                # 3. Efficiency: tool_calls <= 2x expected max (default 20)
                expected_max = test.get("expect_max_tool_calls", 20)
                checks["efficiency"] = 1.0 if len(tool_names) <= expected_max * 2 else 0.0

                # 4. No FATAL errors
                has_fatal = any("FATAL" in e for e in collector.errors)
                checks["no_errors"] = 0.0 if has_fatal else 1.0

                score = sum(checks.values()) / 4
                scores.append(score)
                detail_rows.append({
                    "id": f"t2_{test['name']}",
                    "tier": 2,
                    "category": test.get("category", ""),
                    "output": (result or "")[:60],
                    "tool_count": len(tool_names),
                    "checks": {k: round(v, 2) for k, v in checks.items()},
                    "score": round(score, 2),
                })

                if on_status and (i + 1) % 10 == 0:
                    on_status(f"  coding tier 2: {i + 1}/{len(STRESS_TESTS)} tests")

        finally:
            if agent:
                await _cleanup_agent(agent)

        return scores, errors
