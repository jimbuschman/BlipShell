"""The scratchpad path is anchored to the config file, not the cwd (V3 A6).

Same class of bug as the split-database incident: `blipshell` is an installed
console script, so a cwd-relative `data/scratchpad.md` was whatever folder
the user was standing in. The database path was anchored at the ConfigManager
chokepoint in August; the scratchpad read still used os.path.join("data", ...).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from blipshell.core.agent_chat import ChatMixin


class _Host(ChatMixin):
    def __init__(self, config_path, active_project=None):
        self.config_manager = SimpleNamespace(config_path=config_path)
        self.active_project = active_project


def test_scratchpad_is_read_relative_to_the_config_file(tmp_path, monkeypatch):
    install = tmp_path / "install"
    (install / "data").mkdir(parents=True)
    (install / "data" / "scratchpad.md").write_text("remember the milk", encoding="utf-8")
    (install / "data" / "scratchpad_blip.md").write_text("project note", encoding="utf-8")
    config_path = install / "config.yaml"
    config_path.write_text("", encoding="utf-8")

    # Stand somewhere else entirely, with a decoy data/ dir to catch a cwd read.
    elsewhere = tmp_path / "elsewhere" / "data"
    elsewhere.mkdir(parents=True)
    (elsewhere / "scratchpad.md").write_text("WRONG FILE", encoding="utf-8")
    monkeypatch.chdir(tmp_path / "elsewhere")

    host = _Host(config_path, active_project={"name": "blip"})
    text = host._read_scratchpad()

    assert "remember the milk" in text
    assert "project note" in text
    assert "WRONG FILE" not in text


def test_missing_scratchpad_is_empty_not_an_error(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert _Host(config_path)._read_scratchpad() == ""


def test_absolute_anchor_ignores_cwd_changes(tmp_path, monkeypatch):
    install = tmp_path / "i"
    (install / "data").mkdir(parents=True)
    (install / "data" / "scratchpad.md").write_text("stable", encoding="utf-8")
    cfg = install / "config.yaml"
    cfg.write_text("", encoding="utf-8")
    host = _Host(cfg)
    monkeypatch.chdir(tmp_path)
    first = host._read_scratchpad()
    monkeypatch.chdir(install)
    second = host._read_scratchpad()
    assert first == second and "stable" in first
