"""YAML config manager with get/set/save for self-modification."""

import logging
from pathlib import Path
from typing import Any

import yaml

from blipshell.models.config import BlipShellConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def resolve_config_relative(path: str, config_path: str | Path | None = None) -> str:
    """Anchor a relative data path to the config file's directory, NOT the cwd.

    `blipshell` is an installed console script, so it runs from whatever folder
    the user happens to be standing in, while the config file is always found
    relative to the install (see DEFAULT_CONFIG_PATH). Two different anchors
    for one pair of settings is how you get a config that loads reliably and
    then names a database that moves around.

    That is not hypothetical. Between 2026-08-11 and 2026-08-20 the live
    instance was started one directory deep, in `<repo>/blipshell/`, so
    `data/blipshell.db` resolved to `<repo>/blipshell/data/blipshell.db` --
    SQLite created a fresh 16MB database and nine days of conversation went
    there while the real 491MB corpus sat untouched. Nothing errored: an
    absent SQLite file is a creation, not a failure. `benchmark/runner.py`
    had already hit the same class of bug and solved it locally; this is that
    fix generalized to the chokepoint so every consumer inherits it.

    Absolute paths pass through untouched, which is what keeps `--db` and the
    simulate temp-DB override working -- those are applied AFTER load().
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    base = Path(config_path).resolve().parent if config_path else DEFAULT_CONFIG_PATH.parent
    return str((base / p).resolve())


class ConfigManager:
    """Manages BlipShell configuration with YAML persistence.

    Supports self-modification by the agent (e.g., changing models,
    adjusting pool percentages).
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._raw: dict = {}
        self.config: BlipShellConfig = BlipShellConfig()
        # The database path exactly as authored in YAML, kept only when
        # anchoring rewrote it. save() puts it back — see _anchor_paths.
        self._authored_db_path: str | None = None

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
        self._anchor_paths()
        self._guard_missing_database()
        return self.config

    def _guard_missing_database(self) -> None:
        """Fail LOUDLY when an expected database file is absent.

        SQLite treats an absent file as a creation, not a failure, so a wrong
        path costs history silently (nine days of it, 2026-08-11..08-20 —
        see resolve_config_relative). Anchoring fixed the known cause; this
        guard catches the CLASS: with `database.require_existing: true`, any
        launch that resolves to a nonexistent file stops here with the paths
        spelled out, instead of quietly starting a parallel corpus.

        Enforced at the same chokepoint that anchors the path, so every
        consumer inherits it. Overrides applied AFTER load() (`simulate --db`,
        benchmark temp DBs) are untouched — they are deliberate choices of a
        different file, which is exactly not the failure mode.
        """
        db = getattr(self.config, "database", None)
        if not db or not getattr(db, "require_existing", False):
            return
        target = Path(db.path)
        if target.exists():
            return
        raise FileNotFoundError(
            f"database.require_existing is true and no database exists at the "
            f"resolved path:\n    {target}\n"
            f"(config: {self.config_path.resolve()}, cwd: {Path.cwd()})\n"
            f"If this instance genuinely has no corpus yet (fresh install), "
            f"set database.require_existing: false, or create the file "
            f"deliberately. If it HAS a corpus, this launch was about to "
            f"start a new empty database somewhere else — find the real one "
            f"before running anything."
        )

    def _anchor_paths(self) -> None:
        """Make `database.path` independent of the working directory.

        Done here, at load, rather than at the ~20 `config.database.path`
        consumers (agent.py, nightly.py, cli.py, import_lock): a fix applied
        per-call-site is one that a future call site silently opts out of, and
        this is the bug where opting out costs nine days of memory.
        """
        authored = self.config.database.path
        resolved = resolve_config_relative(authored, self.config_path)
        if resolved == authored:
            return
        self._authored_db_path = authored
        self.config.database.path = resolved
        logger.info("Database path anchored: %r -> %s", authored, resolved)

    def save(self):
        """Save current config to YAML file."""
        self._raw = self.config.model_dump()
        # Write back the relative path the file was authored with. The resolved
        # absolute path is machine-specific, and config.yaml is tracked and
        # synced between the dev box and the Ollama PC — persisting an absolute
        # path here would point one machine at the other's directory layout.
        if self._authored_db_path is not None:
            self._raw.setdefault("database", {})["path"] = self._authored_db_path
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
