"""One place where a router is built with the configured privacy settings.

Every production path that calls a model needs the same three settings from
`config.pii` applied at construction:

- `local_mode_default` -> `EndpointManager.local_only`, which hides every
  endpoint that relays off the machine from selection.
- `enabled` -> `LLMRouter(pii_enabled=)`, the sanitize-before-cloud switch.
- `require_ner` -> `LLMRouter(require_ner=)`, which keeps the FULL-sanitize
  background path off sanitizing endpoints while Presidio is unavailable.

`Agent._build_subsystems` applied all three; nothing else did. `blipshell
repair --blank-summaries`, the nightly runner, both importers and both
reprocess commands each built a bare `LLMRouter(cfg.models,
EndpointManager(cfg.endpoints, cfg.llm))`, so on a machine configured for
local-only operation they would still select a cloud endpoint, and the NER
gate (default `False` in the constructor) was never armed on the very jobs it
was written for — session review, lessons, summaries, whole transcripts.

Maintenance commands read the same corpus as chat. They get the same rules.
"""

from __future__ import annotations

from typing import Optional

from blipshell.llm.endpoints import EndpointManager
from blipshell.llm.router import LLMRouter
from blipshell.models.config import BlipShellConfig, EndpointConfig, ModelsConfig


def local_model_or_fallback(manager, endpoint, model: str, fallback: str | None) -> str:
    """A local Ollama URL can still proxy a cloud model; check both boundaries.

    Known cloud endpoint model names also cover provider-specific names without
    a ':cloud' suffix. A configured local fallback is required when rejected.
    """
    if not manager.local_only:
        return model
    cloud_models = {name for ep in manager.endpoints if ep.should_sanitize_pii
                    for name in ep.models.values()}

    def allowed(name):
        return bool(name) and not (
            name in cloud_models or name.lower().endswith((':cloud', '-cloud'))
        )

    if not endpoint.should_sanitize_pii and allowed(model):
        return model
    if not endpoint.should_sanitize_pii and allowed(fallback):
        return fallback
    raise RuntimeError(f"Local mode has no local model configured for '{model}'")


def build_endpoint_manager(
    config: BlipShellConfig,
    *,
    endpoints: Optional[list[EndpointConfig]] = None,
    local_only: Optional[bool] = None,
) -> EndpointManager:
    """An EndpointManager with local mode applied from `config.pii`.

    `endpoints` overrides which endpoint configs are used (the nightly runner
    disables cloud endpoints on the config objects first). `local_only=True`
    forces local mode on; it never turns OFF what the config asked for —
    a caller may tighten routing, never loosen it.
    """
    manager = EndpointManager(
        endpoints if endpoints is not None else config.endpoints, config.llm,
    )
    manager.local_only = bool(config.pii.local_mode_default) or bool(local_only)
    return manager


def build_router(
    config: BlipShellConfig,
    endpoint_manager: EndpointManager,
    *,
    models: Optional[ModelsConfig] = None,
    disable_fallback: bool = False,
) -> LLMRouter:
    """An LLMRouter carrying the configured PII settings."""
    return LLMRouter(
        models if models is not None else config.models,
        endpoint_manager,
        pii_enabled=config.pii.enabled,
        require_ner=config.pii.require_ner,
        disable_fallback=disable_fallback,
    )


def build_routing(
    config: BlipShellConfig,
    *,
    endpoints: Optional[list[EndpointConfig]] = None,
    local_only: Optional[bool] = None,
    models: Optional[ModelsConfig] = None,
    disable_fallback: bool = False,
) -> tuple[EndpointManager, LLMRouter]:
    """The pair, built together — the usual call."""
    manager = build_endpoint_manager(
        config, endpoints=endpoints, local_only=local_only,
    )
    return manager, build_router(
        config, manager, models=models, disable_fallback=disable_fallback,
    )
