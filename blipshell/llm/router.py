"""Task-type to model + endpoint routing.

Maps task types (reasoning, coding, summarization, ranking, importance, embedding)
to the appropriate model and endpoint based on configuration.
"""

import logging
import time
from typing import Optional

from blipshell.llm.client import LLMClient
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.exceptions import (
    is_bad_request_error,
    is_model_error,
    is_rate_limit_error,
)
from blipshell.models.config import ModelsConfig

logger = logging.getLogger(__name__)


class TaskType:
    """Known task types for routing."""
    REASONING = "reasoning"
    TOOL_CALLING = "tool_calling"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    RANKING = "ranking"
    IMPORTANCE = "importance"
    RANKING_IMPORTANCE = "ranking_importance"
    EMBEDDING = "embedding"
    SESSION_REVIEW = "session_review"
    # Idle self-reflection (lingering thoughts). A dedicated key because the
    # reflection task is unusually model-sensitive: Wisp measured (2026-08-31,
    # same corpus, same prompt, only the model varied) phi4:14b producing 3
    # distinct themes across 20 reflections where gemma4:31b-cloud produced 14.
    # Routing it through REASONING would tie thought diversity to whatever
    # model entity extraction happens to need.
    REFLECTION = "reflection"


class LLMRouter:
    """Routes tasks to the appropriate model + endpoint.

    Uses config to determine which model handles each task type,
    and EndpointManager to select the best endpoint for the role.
    """

    def __init__(self, models_config: ModelsConfig, endpoint_manager: EndpointManager, *, pii_enabled: bool = True, disable_fallback: bool = False):
        self._models = models_config
        self._endpoint_manager = endpoint_manager
        self._failed_models: dict[str, float] = {}  # model_name → failure timestamp
        self._pii_enabled = pii_enabled
        self._disable_fallback = disable_fallback

    def get_model(self, task_type: str) -> str:
        """Get the configured model name for a task type."""
        model_map = {
            TaskType.REASONING: self._models.reasoning,
            TaskType.TOOL_CALLING: self._models.tool_calling,
            TaskType.CODING: self._models.coding,
            TaskType.SUMMARIZATION: self._models.summarization,
            TaskType.RANKING: self._models.ranking,
            TaskType.IMPORTANCE: self._models.importance,
            TaskType.RANKING_IMPORTANCE: self._models.ranking_importance or self._models.ranking,
            TaskType.EMBEDDING: self._models.embedding,
            TaskType.SESSION_REVIEW: self._models.session_review or self._models.reasoning,
            TaskType.REFLECTION: self._models.reflection or self._models.reasoning,
        }
        return model_map.get(task_type, self._models.reasoning)

    def get_fallback_model(self, task_type: str) -> Optional[str]:
        """Get the fallback model for a task type (used when cloud is down)."""
        fallback_map = {
            TaskType.REASONING: self._models.reasoning_fallback,
            TaskType.TOOL_CALLING: self._models.tool_calling_fallback,
            TaskType.CODING: self._models.coding_fallback,
            TaskType.SUMMARIZATION: self._models.summarization_fallback,
            TaskType.RANKING: self._models.ranking_fallback,
            TaskType.IMPORTANCE: self._models.importance_fallback,
            TaskType.RANKING_IMPORTANCE: self._models.ranking_importance_fallback,
            TaskType.SESSION_REVIEW: self._models.session_review_fallback,
            TaskType.REFLECTION: self._models.reflection_fallback,
        }
        return fallback_map.get(task_type)

    def _resolve_fallback_model(self, fallback_ep, task_type: str) -> Optional[str]:
        """Resolve the model to use on a fallback endpoint.

        Returns None if the endpoint can't handle this task type.
        Cloud endpoints only use their explicitly configured per-endpoint models.
        Local (ollama) endpoints can use global fallback model names.
        """
        fb_model = fallback_ep.models.get(task_type)
        if fb_model:
            return fb_model
        # Global fallback models are local model names — never send to cloud
        if fallback_ep.provider == "ollama":
            return self.get_fallback_model(task_type) or self.get_model(task_type)
        return None

    def mark_model_failed(self, model: str):
        """Mark a model as failed. Subsequent calls skip straight to fallback."""
        self._failed_models[model] = time.time()
        logger.warning("Model '%s' marked as failed, will use fallback", model)

    def clear_failed_models(self):
        """Clear all model failures so primaries get retried."""
        if self._failed_models:
            logger.info("Clearing %d failed model(s), will retry primaries", len(self._failed_models))
            self._failed_models.clear()

    def is_model_failed(self, model: str) -> bool:
        """Check if a model is currently marked as failed."""
        return model in self._failed_models

    async def get_client(self, task_type: str) -> Optional[LLMClient]:
        """Get the LLMClient for the best endpoint matching a task type."""
        return await self._endpoint_manager.get_client_for_role(task_type)

    async def get_model_and_client(self, task_type: str) -> tuple[str, Optional[LLMClient]]:
        """Get both model name and client for a task type.

        Resolves per-endpoint model overrides: if the selected endpoint has a
        model configured for this task type, that model is used instead of the
        global default from ModelsConfig.
        """
        endpoint = await self._endpoint_manager.get_endpoint_for_role(task_type)
        if not endpoint:
            return self.get_model(task_type), None
        model = endpoint.models.get(task_type) or self.get_model(task_type)
        return model, endpoint.client

    async def get_context_tokens(self, task_type: str, min_context_tokens: int | None = None) -> int:
        """Get context window size for the endpoint that would handle this task type."""
        endpoint = await self._endpoint_manager.get_endpoint_for_role(task_type, min_context_tokens=min_context_tokens)
        if endpoint and endpoint.context_tokens:
            return endpoint.context_tokens
        return 32768  # safe default

    @staticmethod
    def _estimate_request_tokens(prompt: str, system: str | None = None) -> int:
        """Estimate total tokens for a request (prompt + system)."""
        from blipshell.memory.manager import estimate_tokens
        total = estimate_tokens(prompt)
        if system:
            total += estimate_tokens(system)
        return total

    async def _gated_generate(self, endpoint, prompt: str, model: str,
                              system: Optional[str], gen_kwargs: dict) -> str:
        """Call client.generate(), gating local Ollama calls via OllamaGate."""
        if endpoint.provider == "ollama":
            from blipshell.llm.ollama_gate import get_gate
            gate = get_gate()
            async with gate.async_gate(gate.infer_priority()):
                return await endpoint.client.generate(
                    prompt=prompt, model=model, system=system, **gen_kwargs,
                )
        return await endpoint.client.generate(
            prompt=prompt, model=model, system=system, **gen_kwargs,
        )

    async def generate(self, task_type: str, prompt: str, system: Optional[str] = None, think: Optional[bool | str] = None, min_context_tokens: int | None = None) -> str:
        """Route a generate request to the appropriate model/endpoint.

        If the primary model/endpoint fails and a fallback model is configured,
        retries with the fallback model on any available endpoint.

        Args:
            min_context_tokens: If set, prefer endpoints with at least this many
                context tokens. Used by session_review to route large sessions to
                cloud endpoints with bigger context windows.
        """
        endpoint = await self._endpoint_manager.get_endpoint_for_role(task_type, min_context_tokens=min_context_tokens)
        if not endpoint:
            raise RuntimeError(f"No available endpoint for task type: {task_type}")

        # Use per-endpoint model override if configured
        model = endpoint.models.get(task_type) or self.get_model(task_type)
        client = endpoint.client
        use_fallback = False

        # Pre-flight token check: skip cloud endpoint if request would exceed TPM
        estimated_tokens = self._estimate_request_tokens(prompt, system)
        if not self._disable_fallback and endpoint.would_exceed_tpm(estimated_tokens):
            # min_context_tokens must survive every fallback hop: a 100K-token
            # session review that lands on a 32K endpoint is silently truncated
            # by num_ctx rather than failing.
            fallback_ep = await self._endpoint_manager.get_endpoint_for_role(
                task_type, exclude=endpoint.name,
                min_context_tokens=min_context_tokens,
            )
            if fallback_ep:
                fb_model = self._resolve_fallback_model(fallback_ep, task_type)
                if fb_model:
                    logger.info(
                        "Request ~%dK tokens would exceed %s TPM budget, routing to %s",
                        estimated_tokens // 1000, endpoint.name, fallback_ep.name,
                    )
                    endpoint = fallback_ep
                    model = fb_model
                    client = fallback_ep.client
                    use_fallback = True

        # Skip straight to fallback if primary model is known to be down
        if not self._disable_fallback and self.is_model_failed(model):
            # Must also switch endpoint — can't send a local model name to a cloud API
            fallback_ep = await self._endpoint_manager.get_endpoint_for_role(
                task_type, exclude=endpoint.name,
                min_context_tokens=min_context_tokens,
            )
            if fallback_ep:
                fb_model = self._resolve_fallback_model(fallback_ep, task_type)
                if fb_model:
                    logger.info(
                        "Model '%s' is failed, skipping to '%s' on '%s'",
                        model, fb_model, fallback_ep.name,
                    )
                    endpoint = fallback_ep
                    model = fb_model
                    client = fallback_ep.client
                    use_fallback = True
            if not use_fallback:
                # No alternative endpoint — try fallback model on same endpoint as last resort
                fallback_model = self.get_fallback_model(task_type)
                if fallback_model:
                    model = fallback_model
                    use_fallback = True

        # Sanitize PII before sending to cloud endpoints. Keep the originals:
        # a local fallback must not inherit cloud-scrubbed text (it loses the
        # names and dates that make a summary or lesson specific, for no
        # privacy benefit — the data never leaves the machine).
        raw_prompt, raw_system = prompt, system
        if self._pii_enabled and endpoint.should_sanitize_pii:
            from blipshell.llm.pii import sanitize_text
            prompt = sanitize_text(prompt)
            if system:
                system = sanitize_text(system)

        await self._endpoint_manager.start_request_atomic(endpoint, token_count=estimated_tokens)
        try:
            gen_kwargs = {}
            if endpoint.context_tokens:
                gen_kwargs["options"] = {"num_ctx": endpoint.context_tokens}
            if think is not None:
                gen_kwargs["think"] = think
            if use_fallback and not self._models.fallback_think:
                gen_kwargs["think"] = False
            result = await self._gated_generate(endpoint, prompt, model, system, gen_kwargs)
            endpoint.record_success(0)
            return result
        except Exception as primary_err:
            if is_bad_request_error(primary_err):
                # 400 = per-request issue (content filter, malformed input).
                # Don't penalize endpoint or mark model as permanently failed.
                logger.warning(
                    "Bad request on '%s'/'%s' (not penalizing): %s",
                    model, endpoint.name, primary_err,
                )
            elif is_rate_limit_error(primary_err):
                # Throttling penalizes NOTHING: the endpoint is healthy and the
                # model works — we just asked too often. Blacklisting the model
                # here took it out of service on every OTHER endpoint too.
                logger.warning(
                    "Rate limited on '%s' (penalizing nothing, trying next): %s",
                    endpoint.name, primary_err,
                )
            elif is_model_error(primary_err):
                logger.warning(
                    "Model-level error on endpoint '%s' (not penalizing): %s",
                    endpoint.name, primary_err,
                )
                self.mark_model_failed(model)
            else:
                endpoint.record_failure()
            # Try fallback: get next endpoint (excluding failed one), use its model
            if not use_fallback and not self._disable_fallback:
                try:
                    fallback_ep = await self._endpoint_manager.get_endpoint_for_role(
                        task_type, exclude=endpoint.name,
                        min_context_tokens=min_context_tokens,
                    )
                    fb_model = None
                    if fallback_ep:
                        fb_model = self._resolve_fallback_model(fallback_ep, task_type)
                    if fallback_ep and fb_model:
                        logger.warning(
                            "Primary '%s' on '%s' failed, trying '%s' on '%s'",
                            model, endpoint.name, fb_model, fallback_ep.name,
                        )
                        # Start from the UNSANITIZED text: if the primary was a
                        # cloud endpoint the prompt was scrubbed in place, and a
                        # local fallback would otherwise inherit that damage.
                        fb_prompt, fb_system = raw_prompt, raw_system
                        if self._pii_enabled and fallback_ep.should_sanitize_pii:
                            from blipshell.llm.pii import sanitize_text
                            fb_prompt = sanitize_text(raw_prompt)
                            if fb_system:
                                fb_system = sanitize_text(fb_system)
                        fb_kwargs = {}
                        if not self._models.fallback_think:
                            fb_kwargs["think"] = False
                        if fallback_ep.context_tokens:
                            fb_kwargs["options"] = {"num_ctx": fallback_ep.context_tokens}
                        # Register the request on the endpoint that actually
                        # serves it. Without this, fallback traffic was invisible
                        # to rate-limit accounting and max_concurrent, and its
                        # success/failure never reached endpoint health — while
                        # the PRIMARY got a complete_request() for a call that
                        # ran somewhere else. That matters most on exactly this
                        # path, which is taken when a cloud primary is flaking.
                        await self._endpoint_manager.start_request_atomic(
                            fallback_ep, token_count=estimated_tokens,
                        )
                        try:
                            result = await self._gated_generate(
                                fallback_ep, fb_prompt, fb_model, fb_system, fb_kwargs,
                            )
                            fallback_ep.record_success(0)
                            return result
                        except Exception as fb_err:
                            # Same classification as the primary path: only a
                            # genuine endpoint failure counts against health.
                            if not is_rate_limit_error(fb_err) and not is_model_error(fb_err):
                                fallback_ep.record_failure()
                            raise
                        finally:
                            fallback_ep.complete_request()
                except Exception as fallback_err:
                    logger.error("Fallback also failed: %s", fallback_err)
            raise primary_err
        finally:
            endpoint.complete_request()
