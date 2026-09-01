"""TaskType.REFLECTION routing (llm/router.py + models/config.py).

Reflection got its own routing key because the task is unusually
model-sensitive (Wisp, 2026-08-31: 3 vs 14 distinct themes varying only the
model). These tests pin the resolution order: models.reflection when set,
reasoning otherwise — so existing configs without the key behave exactly as
before the key existed.
"""

from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter, TaskType
from blipshell.models.config import EndpointConfig, ModelsConfig


def _router(models: ModelsConfig) -> LLMRouter:
    endpoints = EndpointManager([
        EndpointConfig(name="local", url="http://localhost:11434",
                       roles=["reasoning", "reflection"]),
    ])
    return LLMRouter(models, endpoints)


def test_reflection_falls_back_to_reasoning_when_unset():
    router = _router(ModelsConfig(reasoning="qwen3:14b"))
    assert router.get_model(TaskType.REFLECTION) == "qwen3:14b"


def test_reflection_uses_its_own_model_when_set():
    router = _router(ModelsConfig(reasoning="qwen3:14b", reflection="gemma4:31b-cloud"))
    assert router.get_model(TaskType.REFLECTION) == "gemma4:31b-cloud"
    # And setting it does not leak into the reasoning jobs.
    assert router.get_model(TaskType.REASONING) == "qwen3:14b"


def test_reflection_fallback_field():
    router = _router(ModelsConfig(
        reasoning="qwen3:14b",
        reflection="gemma4:31b-cloud",
        reflection_fallback="qwen3:14b",
    ))
    assert router.get_fallback_model(TaskType.REFLECTION) == "qwen3:14b"


def test_reflection_fallback_unset_is_none():
    router = _router(ModelsConfig(reasoning="qwen3:14b"))
    assert router.get_fallback_model(TaskType.REFLECTION) is None
