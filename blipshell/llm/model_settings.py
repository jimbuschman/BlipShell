"""Per-model behavioral settings.

Inspired by Aider's model-settings.yml: each model can have custom
behavioral configuration that affects how the agent interacts with it.
Settings are loaded from config.yaml under the `model_settings` key.

Example config:
  model_settings:
    qwen3-coder:480b-cloud:
      max_tool_calls: 20
      use_repo_map: true
      think: false
      extra_instructions: "Be concise. Prefer editing existing files over creating new ones."
    devstral-2:cloud:
      max_tool_calls: 25
      use_repo_map: true
    gpt-oss:latest:
      max_tool_calls: 10
      think: false

When no settings exist for a model, sensible defaults are used.
Model name matching is flexible: "qwen3-coder:480b-cloud" matches
both the exact string and the base name "qwen3-coder".
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Defaults for models without explicit settings
_DEFAULTS = {
    "max_tool_calls": 15,
    "use_repo_map": True,
    "think": None,  # None = use agent-level toggle
    "extra_instructions": "",
    "num_ctx_buffer": 8192,  # extra tokens added to context for overhead
}


@dataclass
class ModelSettings:
    """Behavioral settings for a specific model."""
    max_tool_calls: int = 15
    use_repo_map: bool = True
    think: bool | None = None  # None = defer to agent toggle
    extra_instructions: str = ""
    num_ctx_buffer: int = 8192

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSettings":
        """Create from a config dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class ModelSettingsRegistry:
    """Registry of per-model behavioral settings.

    Loaded from config.yaml's model_settings section.
    Supports flexible name matching for model variants.
    """

    def __init__(self):
        self._settings: dict[str, ModelSettings] = {}

    def load(self, config_data: dict[str, dict]):
        """Load settings from config data (model_name -> settings dict)."""
        self._settings.clear()
        for model_name, settings_dict in config_data.items():
            try:
                self._settings[model_name] = ModelSettings.from_dict(settings_dict)
                logger.debug("Loaded settings for model '%s'", model_name)
            except Exception as e:
                logger.warning("Invalid settings for model '%s': %s", model_name, e)

        logger.info("Loaded settings for %d models", len(self._settings))

    def get(self, model_name: str) -> ModelSettings:
        """Get settings for a model, with flexible name matching.

        Tries:
        1. Exact match
        2. Base name match (before first ':' or '/')
        3. Defaults
        """
        # Exact match
        if model_name in self._settings:
            return self._settings[model_name]

        # Base name match (e.g., "qwen3-coder:480b-cloud" -> "qwen3-coder")
        base = model_name.split(":")[0].split("/")[0]
        if base in self._settings:
            return self._settings[base]

        # Check if any setting key is a prefix of the model name
        for key, settings in self._settings.items():
            if model_name.startswith(key):
                return settings

        # Defaults
        return ModelSettings()

    def has_settings(self, model_name: str) -> bool:
        """Check if explicit settings exist for a model."""
        if model_name in self._settings:
            return True
        base = model_name.split(":")[0].split("/")[0]
        return base in self._settings
