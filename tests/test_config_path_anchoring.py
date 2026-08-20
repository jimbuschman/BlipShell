"""Config path anchoring — `database.path` must not depend on the cwd.

Regression cover for the 2026-08-11..08-20 split-database incident: `blipshell`
is an installed console script, the config file is found relative to the
install, but `database.path` was handed to SQLite as authored. Started one
directory deep, `data/blipshell.db` resolved under that directory, SQLite
CREATED a fresh database, and nine days of conversation went into it while the
real corpus sat untouched. Nothing raised — an absent SQLite file is a
creation, not an error — so only a size/mtime comparison ever revealed it.

The load-bearing test here is `test_same_path_from_any_working_directory`.
"""

import os

import pytest
import yaml

from blipshell.core.config import (
    DEFAULT_CONFIG_PATH,
    ConfigManager,
    resolve_config_relative,
)


def _write_config(tmp_path, db_path="data/blipshell.db"):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"database": {"path": db_path}}), encoding="utf-8")
    return cfg


class TestResolveConfigRelative:
    def test_relative_anchors_to_config_directory(self, tmp_path):
        cfg = _write_config(tmp_path)
        assert resolve_config_relative("data/blipshell.db", cfg) == str(
            (tmp_path / "data" / "blipshell.db").resolve()
        )

    def test_absolute_passes_through_untouched(self, tmp_path):
        cfg = _write_config(tmp_path)
        absolute = str((tmp_path / "elsewhere" / "x.db").resolve())
        assert resolve_config_relative(absolute, cfg) == absolute

    def test_no_config_path_falls_back_to_install_root(self):
        """Callers that pass nothing get the repo root, not the cwd."""
        got = resolve_config_relative("data/blipshell.db", None)
        assert got == str((DEFAULT_CONFIG_PATH.parent / "data" / "blipshell.db").resolve())


class TestConfigManagerAnchoring:
    def test_load_resolves_database_path_to_absolute(self, tmp_path):
        cfg = _write_config(tmp_path)
        config = ConfigManager(cfg).load()
        assert os.path.isabs(config.database.path)
        assert config.database.path == str((tmp_path / "data" / "blipshell.db").resolve())

    def test_same_path_from_any_working_directory(self, tmp_path, monkeypatch):
        """THE incident: the cwd must not change which database is opened.

        `nested/` stands in for `<repo>/blipshell/` — the package directory the
        live instance was actually launched from.
        """
        cfg = _write_config(tmp_path)
        nested = tmp_path / "nested"
        nested.mkdir()

        # abspath() is the point: it is what SQLite does with a relative path.
        # Comparing config.database.path directly is VACUOUS — unanchored it
        # stays the literal "data/blipshell.db" from either directory, so the
        # strings match while the FILES OPENED differ. Caught by mutation
        # testing; do not "simplify" this back.
        monkeypatch.chdir(tmp_path)
        from_root = os.path.abspath(ConfigManager(cfg).load().database.path)

        monkeypatch.chdir(nested)
        from_nested = os.path.abspath(ConfigManager(cfg).load().database.path)

        assert from_root == from_nested
        # And specifically NOT the stray file the bug would have created.
        assert os.path.abspath(nested) not in from_nested

    def test_absolute_path_in_yaml_is_left_alone(self, tmp_path):
        absolute = str((tmp_path / "somewhere" / "real.db").resolve())
        cfg = _write_config(tmp_path, db_path=absolute)
        assert ConfigManager(cfg).load().database.path == absolute

    def test_missing_config_file_still_anchors(self, tmp_path):
        """No file on disk = defaults, which must be anchored too."""
        config = ConfigManager(tmp_path / "absent.yaml").load()
        assert os.path.isabs(config.database.path)
        assert config.database.path == str((tmp_path / "data" / "blipshell.db").resolve())

    def test_post_load_override_is_not_re_anchored(self, tmp_path):
        """`simulate --db` and the tests assign after load(); that must stick."""
        cfg = _write_config(tmp_path)
        mgr = ConfigManager(cfg)
        config = mgr.load()
        override = str(tmp_path / "throwaway.db")
        config.database.path = override
        assert config.database.path == override


class TestSaveKeepsAuthoredPath:
    """config.yaml is tracked and synced between two machines — an absolute
    path written back would point one box at the other's directory layout."""

    def test_save_writes_back_the_relative_path(self, tmp_path):
        cfg = _write_config(tmp_path)
        mgr = ConfigManager(cfg)
        config = mgr.load()
        assert os.path.isabs(config.database.path)  # anchored in memory

        mgr.save()

        on_disk = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert on_disk["database"]["path"] == "data/blipshell.db"

    def test_save_preserves_an_authored_absolute_path(self, tmp_path):
        """Nothing was rewritten, so nothing should be restored either."""
        absolute = str((tmp_path / "real.db").resolve())
        cfg = _write_config(tmp_path, db_path=absolute)
        mgr = ConfigManager(cfg)
        mgr.load()
        mgr.save()
        on_disk = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        assert on_disk["database"]["path"] == absolute

    def test_save_round_trip_is_stable(self, tmp_path):
        """Two saves must not drift the stored value."""
        cfg = _write_config(tmp_path)
        mgr = ConfigManager(cfg)
        mgr.load()
        mgr.save()
        first = yaml.safe_load(cfg.read_text(encoding="utf-8"))["database"]["path"]
        mgr.save()
        second = yaml.safe_load(cfg.read_text(encoding="utf-8"))["database"]["path"]
        assert first == second == "data/blipshell.db"


class TestBenchmarkHelperStillDelegatesIdentically:
    """`_resolve_db_path` now delegates; semantics must be unchanged."""

    @pytest.mark.parametrize("db_path", ["data/benchmark.db", "nested/dir/b.db"])
    def test_matches_shared_helper(self, tmp_path, db_path):
        from blipshell.benchmark.runner import _resolve_db_path

        cfg = _write_config(tmp_path)
        assert _resolve_db_path(db_path, str(cfg)) == resolve_config_relative(
            db_path, str(cfg)
        )

    def test_absolute_benchmark_path_untouched(self, tmp_path):
        from blipshell.benchmark.runner import _resolve_db_path

        absolute = str((tmp_path / "b.db").resolve())
        assert _resolve_db_path(absolute, None) == absolute
