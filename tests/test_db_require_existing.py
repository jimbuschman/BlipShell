"""database.require_existing guard (core/config.py).

An absent SQLite file is a creation, not a failure — which is how nine days
of live history went into a phantom database (2026-08-11..08-20). Path
anchoring fixed the known cause; this guard catches the class. It lives at
the same ConfigManager chokepoint as anchoring, so every consumer inherits
it, and overrides applied AFTER load() (simulate --db, benchmark temp DBs)
are deliberately untouched.
"""

import pytest

from blipshell.core.config import ConfigManager


def _write_config(tmp_path, *, require_existing: bool) -> str:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "database:\n"
        "  path: \"data/blipshell.db\"\n"
        f"  require_existing: {'true' if require_existing else 'false'}\n",
        encoding="utf-8",
    )
    return str(cfg)


def test_missing_db_with_guard_on_refuses_to_load(tmp_path):
    cfg_path = _write_config(tmp_path, require_existing=True)
    with pytest.raises(FileNotFoundError) as exc:
        ConfigManager(cfg_path).load()
    # The message must name the RESOLVED target — the whole failure mode is
    # that the authored string looks right while the file it names moved.
    message = str(exc.value)
    assert str(tmp_path / "data" / "blipshell.db") in message
    # And it must not have created anything on the way out.
    assert not (tmp_path / "data" / "blipshell.db").exists()


def test_existing_db_with_guard_on_loads(tmp_path):
    cfg_path = _write_config(tmp_path, require_existing=True)
    db = tmp_path / "data" / "blipshell.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    config = ConfigManager(cfg_path).load()
    assert config.database.require_existing is True


def test_missing_db_with_guard_off_loads(tmp_path):
    """Fresh installs and temp-DB test configs keep working."""
    cfg_path = _write_config(tmp_path, require_existing=False)
    config = ConfigManager(cfg_path).load()
    assert config.database.require_existing is False


def test_guard_defaults_off(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("database:\n  path: \"data/blipshell.db\"\n", encoding="utf-8")
    config = ConfigManager(str(cfg)).load()
    assert config.database.require_existing is False


def test_guard_checks_resolved_target_not_cwd(tmp_path, monkeypatch):
    """The guard runs AFTER anchoring: a wrong-cwd launch with the database
    present at the config-anchored path must load, and one where only a
    cwd-relative decoy exists must refuse. This is the wrong-cwd launch that
    cost nine days, replayed as a test."""
    cfg_path = _write_config(tmp_path, require_existing=True)
    real_db = tmp_path / "data" / "blipshell.db"
    real_db.parent.mkdir(parents=True)
    real_db.write_bytes(b"")

    elsewhere = tmp_path / "somewhere" / "deeper"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    ConfigManager(cfg_path).load()  # must not raise

    # Now the inverse: a decoy at the cwd-relative path, nothing at the
    # anchored one. Passing here would mean the guard used the cwd.
    real_db.unlink()
    decoy = elsewhere / "data" / "blipshell.db"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        ConfigManager(cfg_path).load()
