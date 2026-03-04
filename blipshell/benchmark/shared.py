"""Shared utilities for the unified benchmark system."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from blipshell.core.config import ConfigManager
from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter
from blipshell.models.config import EndpointConfig, LLMConfig, ModelsConfig, get_ollama_url

if TYPE_CHECKING:
    from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

# All LLM task types that can be benchmarked
ALL_TASK_TYPES = [
    "reasoning", "summarization", "ranking", "importance",
    "ranking_importance", "tool_calling", "coding",
]


def make_router(
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    task_types: list[str] | None = None,
    context_tokens: int | None = None,
) -> LLMRouter:
    """Create a router that routes specified task types to a single model.

    Consolidates make_router/make_local_router from existing benchmark scripts.
    """
    types = task_types or ALL_TASK_TYPES
    model_kwargs = {t: model_name for t in types}
    models = ModelsConfig(**model_kwargs)

    ep_kwargs: dict = {
        "name": "bench",
        "url": ollama_url,
        "roles": types,
        "priority": 1,
        "max_concurrent": 1,
    }
    if context_tokens:
        ep_kwargs["context_tokens"] = context_tokens

    ep = EndpointConfig(**ep_kwargs)
    return LLMRouter(models, EndpointManager([ep], LLMConfig()), disable_fallback=True)


def make_cloud_router(
    endpoint_name: str,
    model_name: str,
    config: BlipShellConfig,
    task_types: list[str] | None = None,
) -> LLMRouter | None:
    """Create a router using a cloud endpoint from config."""
    types = task_types or ALL_TASK_TYPES

    ep_config = None
    for ep in config.endpoints:
        if ep.name == endpoint_name and ep.enabled:
            ep_config = ep
            break
    if not ep_config:
        return None

    ep_copy = ep_config.model_copy()
    ep_copy.roles = types
    ep_copy.models = {t: model_name for t in types}

    model_kwargs = {t: model_name for t in types}
    models = ModelsConfig(**model_kwargs)
    return LLMRouter(models, EndpointManager([ep_copy], config.llm), disable_fallback=True)


async def check_model_availability(
    models: list[str], ollama_url: str,
) -> tuple[list[str], list[str]]:
    """Check which models are available on the Ollama server.

    Returns (available, unavailable).
    """
    available = []
    unavailable = []

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            installed = {m["name"] for m in resp.json().get("models", [])}

            for model in models:
                # Match with or without :latest suffix
                if model in installed or f"{model}:latest" in installed:
                    available.append(model)
                else:
                    unavailable.append(model)
    except Exception as e:
        logger.warning("Could not check model availability: %s", e)
        # Assume all available if we can't check
        return models, []

    return available, unavailable


def load_stratified_sample(db_path: str, n: int = 20) -> list[dict]:
    """Pull a stratified sample of memories with known scores.

    Gets an even mix across rank levels (1-5) for accuracy testing.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    messages = []
    for rank in [1, 2, 3, 4, 5]:
        per_rank = max(n // 5, 2)
        rows = conn.execute(
            """SELECT id, content, summary, role, rank, importance, memory_type
               FROM memories
               WHERE rank = ? AND summary IS NOT NULL AND length(content) > 20
               ORDER BY RANDOM() LIMIT ?""",
            (rank, per_rank),
        ).fetchall()
        for row in rows:
            messages.append(dict(row))

    conn.close()

    # Deduplicate
    seen = set()
    unique = []
    for m in messages:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)
    return unique[:n]


def load_dedup_pairs(db_path: str, n: int = 10) -> list[tuple[dict, list[str]]]:
    """Pull memory pairs for dedup testing."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """SELECT id, content, summary, role, rank
           FROM memories
           WHERE summary IS NOT NULL AND length(summary) > 20
           ORDER BY RANDOM() LIMIT ?""",
        (n * 3,),
    ).fetchall()
    conn.close()

    pairs = []
    items = [dict(r) for r in rows]
    for i in range(0, len(items) - 1, 2):
        if len(pairs) >= n:
            break
        pairs.append((items[i], [items[i + 1]["summary"]]))

    return pairs


def get_config_and_db(
    config_path: str | None = None,
    db_path: str | None = None,
) -> tuple[BlipShellConfig, str, str]:
    """Load config, resolve DB path and Ollama URL.

    Returns (config, db_path, ollama_url).
    """
    config_mgr = ConfigManager(config_path)
    config = config_mgr.load()
    resolved_db = db_path or config.database.path
    ollama_url = get_ollama_url(config.endpoints)
    return config, resolved_db, ollama_url


# Known context window sizes per model
CONTEXT_TOKENS: dict[str, int] = {
    # qwen3.5 — all 256K
    "qwen3.5:0.8b": 262144,
    "qwen3.5:2b": 262144,
    "qwen3.5:4b": 262144,
    "qwen3.5:9b": 262144,
    "qwen3.5:27b": 262144,
    "qwen3.5:cloud": 262144,
    # qwen3
    "qwen3:4b": 40960,
    "qwen3:14b": 32768,
    # older models
    "qwen2.5:7b": 32768,
    "qwen2.5:14b": 32768,
    "glm4:latest": 32768,
    "gpt-oss:latest": 32768,
    "glm-5:cloud": 196608,
}


def get_context_tokens(model: str) -> int | None:
    """Get known context window size for a model, or None if unknown."""
    return CONTEXT_TOKENS.get(model)
