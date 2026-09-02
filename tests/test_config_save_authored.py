"""save() persists the AUTHORED config, not the expanded model dump.

The old save() wrote model_dump(): every default expanded into the file, all
comments gone, and the anchored absolute database path needing a special
restore hack. Two live callers (web /api/config, personality import) meant one
programmatic save away from a 500-line machine file. These tests pin the new
contract: the file keeps only what was authored plus set() changes; unset keys
stay unset so code defaults keep ruling them.
"""

import yaml

from blipshell.core.config import ConfigManager


def _write(tmp_path, content: dict):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump(content), encoding="utf-8")
    return cfg


def test_save_does_not_expand_defaults(tmp_path):
    cfg = _write(tmp_path, {"llm": {"timeout": 300}})
    mgr = ConfigManager(cfg)
    mgr.load()
    mgr.save()
    on_disk = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    # Only the authored key survives — no models:, memory:, endpoints: blocks
    # materialized out of the defaults.
    assert on_disk == {"llm": {"timeout": 300}}


def test_set_then_save_persists_the_change(tmp_path):
    cfg = _write(tmp_path, {"llm": {"timeout": 300}})
    mgr = ConfigManager(cfg)
    mgr.load()
    mgr.set("models.tool_calling", "gemma4:31b-cloud")
    mgr.save()
    on_disk = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert on_disk == {
        "llm": {"timeout": 300},
        "models": {"tool_calling": "gemma4:31b-cloud"},
    }
    # And the in-memory config reflects it too.
    assert mgr.config.models.tool_calling == "gemma4:31b-cloud"


def test_direct_config_mutation_is_runtime_only(tmp_path):
    """simulate --db style overrides mutate .config after load; save() must
    not leak them into the tracked file."""
    cfg = _write(tmp_path, {"database": {"path": "data/blipshell.db"}})
    mgr = ConfigManager(cfg)
    config = mgr.load()
    config.database.path = str(tmp_path / "throwaway.db")
    mgr.save()
    on_disk = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert on_disk["database"]["path"] == "data/blipshell.db"


def test_slim_config_loads_with_defaults_intact(tmp_path):
    """The pruned-config premise: unset keys mean the code default."""
    cfg = _write(tmp_path, {"models": {"tool_calling": "gemma4:31b-cloud"}})
    config = ConfigManager(cfg).load()
    assert config.models.tool_calling == "gemma4:31b-cloud"
    # Untouched keys come from ModelsConfig defaults.
    assert config.models.reasoning == "qwen3:14b"
    assert config.memory.consolidation_similarity == 0.92
    assert config.reflection.enabled is True
