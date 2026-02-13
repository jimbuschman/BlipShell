"""Task-type to model + endpoint routing.

Maps task types (reasoning, coding, summarization, ranking, embedding)
to the appropriate model and endpoint based on configuration.
"""

import logging
from typing import Optional

from blipshell.llm.client import LLMClient
from blipshell.llm.endpoints import EndpointManager
from blipshell.models.config import ModelsConfig

logger = logging.getLogger(__name__)


class TaskType:
    """Known task types for routing."""
    REASONING = "reasoning"
    TOOL_CALLING = "tool_calling"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    RANKING = "ranking"
    EMBEDDING = "embedding"


class LLMRouter:
    """Routes tasks to the appropriate model + endpoint.

    Uses config to determine which model handles each task type,
    and EndpointManager to select the best endpoint for the role.
    """

    def __init__(self, models_config: ModelsConfig, endpoint_manager: EndpointManager):
        self._models = models_config
        self._endpoint_manager = endpoint_manager

    def get_model(self, task_type: str) -> str:
        """Get the configured model name for a task type."""
        model_map = {
            TaskType.REASONING: self._models.reasoning,
            TaskType.TOOL_CALLING: self._models.tool_calling,
            TaskType.CODING: self._models.coding,
            TaskType.SUMMARIZATION: self._models.summarization,
            TaskType.RANKING: self._models.ranking,
            TaskType.EMBEDDING: self._models.embedding,
        }
        return model_map.get(task_type, self._models.reasoning)

    def get_fallback_model(self, task_type: str) -> Optional[str]:
        """Get the fallback model for a task type (used when cloud is down)."""
        fallback_map = {
            TaskType.REASONING: self._models.reasoning_fallback,
            TaskType.TOOL_CALLING: self._models.tool_calling_fallback,
            TaskType.CODING: self._models.coding_fallback,
        }
        return fallback_map.get(task_type)

    async def get_client(self, task_type: str) -> Optional[LLMClient]:
        """Get the LLMClient for the best endpoint matching a task type."""
        return await self._endpoint_manager.get_client_for_role(task_type)

    async def get_model_and_client(self, task_type: str) -> tuple[str, Optional[LLMClient]]:
        """Get both model name and client for a task type."""
        return self.get_model(task_type), await self.get_client(task_type)

    async def generate(self, task_type: str, prompt: str, system: Optional[str] = None) -> str:
        """Route a generate request to the appropriate model/endpoint.

        If the primary model/endpoint fails and a fallback model is configured,
        retries with the fallback model on any available endpoint.
        """
        model = self.get_model(task_type)
        client = await self.get_client(task_type)
        if not client:
            raise RuntimeError(f"No available endpoint for task type: {task_type}")

        endpoint = await self._endpoint_manager.get_endpoint_for_role(task_type)
        endpoint.start_request()
        try:
            # Pass context window size to Ollama
            gen_kwargs = {}
            if endpoint and endpoint.context_tokens:
                gen_kwargs["options"] = {"num_ctx": endpoint.context_tokens}
            result = await client.generate(prompt=prompt, model=model, system=system, **gen_kwargs)
            endpoint.record_success(0)
            return result
        except Exception as primary_err:
            endpoint.record_failure()
            # Try fallback model if available
            fallback_model = self.get_fallback_model(task_type)
            if fallback_model and fallback_model != model:
                logger.warning("Primary model '%s' failed, trying fallback '%s'", model, fallback_model)
                try:
                    fallback_client = await self._endpoint_manager.get_client_for_role(task_type)
                    if fallback_client:
                        fb_kwargs = {}
                        if not self._models.fallback_think:
                            fb_kwargs["think"] = False
                        result = await fallback_client.generate(
                            prompt=prompt, model=fallback_model, system=system,
                            **fb_kwargs,
                        )
                        return result
                except Exception as fallback_err:
                    logger.error("Fallback model '%s' also failed: %s", fallback_model, fallback_err)
            raise primary_err
        finally:
            endpoint.complete_request()
