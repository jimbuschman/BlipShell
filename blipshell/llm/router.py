"""Task-type to model + endpoint routing.

Maps task types (reasoning, coding, summarization, ranking, importance, embedding)
to the appropriate model and endpoint based on configuration.
"""

import logging
import time
from typing import Optional

from blipshell.llm.client import LLMClient
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.exceptions import is_model_error
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


class LLMRouter:
    """Routes tasks to the appropriate model + endpoint.

    Uses config to determine which model handles each task type,
    and EndpointManager to select the best endpoint for the role.
    """

    def __init__(self, models_config: ModelsConfig, endpoint_manager: EndpointManager):
        self._models = models_config
        self._endpoint_manager = endpoint_manager
        self._failed_models: dict[str, float] = {}  # model_name → failure timestamp

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
        }
        return fallback_map.get(task_type)

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

    async def generate(self, task_type: str, prompt: str, system: Optional[str] = None, think: Optional[bool | str] = None) -> str:
        """Route a generate request to the appropriate model/endpoint.

        If the primary model/endpoint fails and a fallback model is configured,
        retries with the fallback model on any available endpoint.
        """
        endpoint = await self._endpoint_manager.get_endpoint_for_role(task_type)
        if not endpoint:
            raise RuntimeError(f"No available endpoint for task type: {task_type}")

        # Use per-endpoint model override if configured
        model = endpoint.models.get(task_type) or self.get_model(task_type)
        client = endpoint.client
        use_fallback = False

        # Skip straight to fallback if primary model is known to be down
        if self.is_model_failed(model):
            # Must also switch endpoint — can't send a local model name to a cloud API
            fallback_ep = await self._endpoint_manager.get_endpoint_for_role(
                task_type, exclude=endpoint.name,
            )
            if fallback_ep:
                fb_model = fallback_ep.models.get(task_type)
                if not fb_model:
                    fb_model = self.get_fallback_model(task_type) or self.get_model(task_type)
                logger.info(
                    "Model '%s' is failed, skipping to '%s' on '%s'",
                    model, fb_model, fallback_ep.name,
                )
                endpoint = fallback_ep
                model = fb_model
                client = fallback_ep.client
                use_fallback = True
            else:
                # No alternative endpoint — try fallback model on same endpoint as last resort
                fallback_model = self.get_fallback_model(task_type)
                if fallback_model:
                    model = fallback_model
                    use_fallback = True

        endpoint.start_request()
        try:
            gen_kwargs = {}
            if endpoint.context_tokens:
                gen_kwargs["options"] = {"num_ctx": endpoint.context_tokens}
            if think is not None:
                gen_kwargs["think"] = think
            if use_fallback and not self._models.fallback_think:
                gen_kwargs["think"] = False
            result = await client.generate(prompt=prompt, model=model, system=system, **gen_kwargs)
            endpoint.record_success(0)
            return result
        except Exception as primary_err:
            if is_model_error(primary_err):
                logger.warning(
                    "Model-level error on endpoint '%s' (not penalizing): %s",
                    endpoint.name, primary_err,
                )
                self.mark_model_failed(model)
            else:
                endpoint.record_failure()
            # Try fallback: get next endpoint (excluding failed one), use its model
            if not use_fallback:
                try:
                    fallback_ep = await self._endpoint_manager.get_endpoint_for_role(
                        task_type, exclude=endpoint.name,
                    )
                    if fallback_ep:
                        # Use the fallback endpoint's own model for this task,
                        # or the global fallback model if it's a local endpoint
                        fb_model = fallback_ep.models.get(task_type)
                        if not fb_model:
                            fb_model = self.get_fallback_model(task_type) or self.get_model(task_type)
                        logger.warning(
                            "Primary '%s' on '%s' failed, trying '%s' on '%s'",
                            model, endpoint.name, fb_model, fallback_ep.name,
                        )
                        fb_kwargs = {}
                        if not self._models.fallback_think:
                            fb_kwargs["think"] = False
                        if fallback_ep.context_tokens:
                            fb_kwargs["options"] = {"num_ctx": fallback_ep.context_tokens}
                        result = await fallback_ep.client.generate(
                            prompt=prompt, model=fb_model, system=system,
                            **fb_kwargs,
                        )
                        return result
                except Exception as fallback_err:
                    logger.error("Fallback also failed: %s", fallback_err)
            raise primary_err
        finally:
            endpoint.complete_request()
