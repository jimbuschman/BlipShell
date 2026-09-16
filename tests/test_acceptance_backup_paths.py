"""Backup isolation must include both startup and unattended maintenance."""
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from blipshell.core.config import ConfigManager
from blipshell.core.nightly import NightlyRunner


@pytest.mark.parametrize('absolute_db', [False, True])
def test_backup_path_anchors_even_when_database_already_absolute(tmp_path, absolute_db):
    authored_db = str(tmp_path / 'corpus.db') if absolute_db else 'corpus.db'
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump({'database': {
        'path': authored_db, 'backup_dir': 'isolated-backups',
    }}))
    manager = ConfigManager(config_file)
    config = manager.load()
    assert Path(config.database.backup_dir) == tmp_path / 'isolated-backups'
    manager.save()
    assert yaml.safe_load(config_file.read_text())['database']['backup_dir'] == 'isolated-backups'


async def test_nightly_backup_is_quiet_and_restorable_in_configured_directory(tmp_path, monkeypatch):
    from scripts import backup_db
    source = tmp_path / 'corpus.db'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE sentinel(value TEXT)')
        db.execute("INSERT INTO sentinel VALUES ('preserved')")
    config_file = tmp_path / 'config.yaml'
    config_file.write_text(yaml.safe_dump({'database': {
        'path': str(source), 'backup_dir': 'isolated-backups',
    }}))
    runner = object.__new__(NightlyRunner)
    runner.config = ConfigManager(config_file).load()
    console = Mock()
    console.print.side_effect = AssertionError('unattended backup must not print unicode to console')
    monkeypatch.setattr(backup_db, 'console', console)
    result = await runner._job_backup(lambda _: None)
    folder = Path(result['backup_path'])
    assert folder.parent == tmp_path / 'isolated-backups'
    with sqlite3.connect(folder / source.name) as restored:
        assert restored.execute('SELECT value FROM sentinel').fetchone()[0] == 'preserved'
        assert restored.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
