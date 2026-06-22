"""Neutral cloud LLM-as-judge for grading open-ended benchmark outputs.

Deterministic parsers can grade ranking/contradiction (structured outputs) but
they cannot grade whether a *summary is actually good* or a *plan is sound*.
The judge fills that gap: a strong cloud model — explicitly NOT one of the
candidate models, to avoid self-grading bias — scores each open-ended response
0.0-1.0 against a rubric.

The judge routes through an endpoint already defined in config.yaml (looked up
by name), reusing its client (Ollama or OpenAI-compat — both expose the same
duck-typed `generate`). Every call is wrapped in asyncio.wait_for per the
project's timeout discipline. On any failure the judge returns None (not an
exception) so a flaky judge never aborts a benchmark run.
"""

import asyncio
import json
import logging
import re
from typing import Optional

from blipshell.llm import prompts

logger = logging.getLogger(__name__)


class JudgeUnavailable(RuntimeError):
    """Raised at construction when the configured judge endpoint can't be found."""


class LLMJudge:
    """Grades open-ended outputs 0.0-1.0 via a configured cloud judge model."""

    def __init__(self, model: str, client, timeout: float = 60.0):
        if not model:
            raise JudgeUnavailable("no judge_model configured")
        if client is None:
            raise JudgeUnavailable("judge endpoint client unavailable")
        self.model = model
        self._client = client
        self.timeout = timeout

    # -- public grading API ------------------------------------------------

    async def grade_summarization(self, source_text: str, summary: str) -> Optional[float]:
        return await self._grade(*prompts.judge_summarization(source_text, summary))

    async def grade_reasoning(self, task: str, response: str) -> Optional[float]:
        return await self._grade(*prompts.judge_reasoning(task, response))

    async def grade_lesson(self, conversation: str, lesson: str) -> Optional[float]:
        return await self._grade(*prompts.judge_lesson(conversation, lesson))

    # -- internals ---------------------------------------------------------

    async def _grade(self, system: str, user: str) -> Optional[float]:
        """Call the judge and parse a 0.0-1.0 score. None on any failure."""
        try:
            raw = await asyncio.wait_for(
                self._client.generate(
                    prompt=user, model=self.model, system=system, use_cache=False,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Judge call timed out after %.0fs", self.timeout)
            return None
        except Exception as e:  # noqa: BLE001 — judge must never crash a run
            logger.warning("Judge call failed: %s", e)
            return None
        return self.parse_score(raw)

    @staticmethod
    def parse_score(raw: str) -> Optional[float]:
        """Extract a 0.0-1.0 score from a judge response.

        Prefers strict JSON {"score": ...}; falls back to the first float in
        the text. Returns None if nothing parseable / out of range.
        """
        if not raw:
            return None
        text = raw.strip()
        # Strip ```json fences if the model added them despite instructions.
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

        # 1) Try to locate a JSON object and read its "score".
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            try:
                obj = json.loads(obj_match.group(0))
                if isinstance(obj, dict) and "score" in obj:
                    return _clamp_unit(float(obj["score"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 2) Regex for a "score": <num> pattern even if JSON is malformed.
        kv = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        if kv:
            try:
                return _clamp_unit(float(kv.group(1)))
            except ValueError:
                pass

        # 3) Last resort: first standalone float in the text.
        num = re.search(r"[0-9]*\.?[0-9]+", text)
        if num:
            try:
                return _clamp_unit(float(num.group(0)))
            except ValueError:
                pass
        return None


def _clamp_unit(value: float) -> Optional[float]:
    """Clamp to [0,1]; reject obviously-out-of-range values (e.g. a 1-5 rank)."""
    if value != value:  # NaN
        return None
    if value > 1.0:
        # A model might answer on a 0-100 or 0-10 scale; normalize common cases.
        if value <= 10.0:
            value = value / 10.0
        elif value <= 100.0:
            value = value / 100.0
        else:
            return None
    if value < 0.0:
        return None
    return round(value, 4)


def build_judge(config, endpoint_manager) -> Optional[LLMJudge]:
    """Construct an LLMJudge from BenchmarkConfig + an EndpointManager.

    Returns None (judging disabled) when no judge_model is configured. Raises
    JudgeUnavailable when a judge IS configured but its endpoint is missing —
    a misconfiguration the caller should surface, not silently ignore.
    """
    bench = config.benchmark
    if not bench.judge_model:
        return None
    client = endpoint_manager.get_client_by_name(bench.judge_endpoint)
    if client is None:
        raise JudgeUnavailable(
            f"judge_endpoint '{bench.judge_endpoint}' not found or disabled in endpoints"
        )
    return LLMJudge(bench.judge_model, client, timeout=bench.judge_timeout)
