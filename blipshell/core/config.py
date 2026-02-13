"""YAML config manager with get/set/save for self-modification."""

import logging
from pathlib import Path
from typing import Any

import yaml

from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


class ConfigManager:
    """Manages BlipShell configuration with YAML persistence.

    Supports self-modification by the agent (e.g., changing models,
    adjusting pool percentages).
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._raw: dict = {}
        self.config: BlipShellConfig = BlipShellConfig()

    def load(self) -> BlipShellConfig:
        """Load config from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self._raw = yaml.safe_load(f) or {}
            self.config = BlipShellConfig(**self._raw)
            logger.info("Config loaded from %s", self.config_path)
        else:
            self.config = BlipShellConfig()
            logger.info("Using default config (no file at %s)", self.config_path)
        return self.config

    def save(self):
        """Save current config to YAML file."""
        self._raw = self.config.model_dump()
        with open(self.config_path, "w") as f:
            yaml.dump(self._raw, f, default_flow_style=False, sort_keys=False)
        logger.info("Config saved to %s", self.config_path)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value using dotted notation (e.g., 'models.reasoning')."""
        keys = dotted_key.split(".")
        obj = self._raw
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj

    def set(self, dotted_key: str, value: Any):
        """Set a config value using dotted notation and reload."""
        keys = dotted_key.split(".")
        obj = self._raw
        for key in keys[:-1]:
            if key not in obj or not isinstance(obj[key], dict):
                obj[key] = {}
            obj = obj[key]
        obj[keys[-1]] = value

        # Reload Pydantic model from updated raw dict
        self.config = BlipShellConfig(**self._raw)

    def get_config(self) -> BlipShellConfig:
        """Get the current config object."""
        return self.config

    def to_dict(self) -> dict:
        """Get config as a plain dict."""
        return self.config.model_dump()
