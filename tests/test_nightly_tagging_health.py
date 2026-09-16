"""Nightly history and tagging measurements must reflect outcomes, not exits."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from blipshell.core.nightly import NightlyRunner
from blipshell.core.nightly_history import HISTORY_KEY, growing_pool_warning, save_run
from blipshell.memory.batch_tagger import BatchTagger
from blipshell.memory.centroid_tagger import CentroidTagger
from blipshell.memory.tag_health import tag_health_async
from blipshell.models.config import MemoryConfig
from blipshell.models.memory import Memory


async def seed(store, tags):
    mid = await store.create_memory(Memory(role='user', content='content', summary='summary'))
    if tags:
        await store.tag_memory(mid, tags)
    return mid


def runner(store, path):
    result = NightlyRunner(SimpleNamespace(database=SimpleNamespace(path=path)),
                          store, MagicMock(), MagicMock(), MagicMock())
    result._check_ollama_health = AsyncMock(return_value=True)
    return result


async def test_health_separates_queue_exits_from_coverage(sqlite_store):
    for tags in ([], ['neutral'], ['_skip'], ['python', '_skip'], ['python', 'sql']):
        await seed(sqlite_store, tags)
    stats = await tag_health_async(sqlite_store)
    assert stats == dict(active=5, untagged=1, without_topic_tags=3,
                         neutral_only=1, marked_skip=2, skipped_low_coverage=2, pending=2)
    assert stats['pending'] == await sqlite_store.count_poorly_tagged_memories()


async def test_history_retains_runs_and_checkpoints_interruption(sqlite_store, temp_db_path):
    r = runner(sqlite_store, temp_db_path)
    r.run_job = AsyncMock(return_value={'checked': 1})
    await r.run(jobs=['batch_tag'])
    await r.run(jobs=['batch_tag'])
    history = json.loads(await sqlite_store.get_metadata(HISTORY_KEY))
    assert len(history) == 2
    assert all(h['status'] == 'completed' for h in history)
    assert history[0]['jobs']['batch_tag']['checked'] == 1
    assert history[0]['tagging']['after']['pending'] == 0

    r.run_job = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await r.run(jobs=['batch_tag'])
    history = json.loads(await sqlite_store.get_metadata(HISTORY_KEY))
    assert len(history) == 3
    assert history[-1]['status'] == 'running'
    assert history[-1]['active_job'] == 'batch_tag'


async def test_history_is_bounded_and_checkpoint_does_not_add_a_run(sqlite_store, monkeypatch):
    import blipshell.core.nightly_history as module
    monkeypatch.setattr(module, 'HISTORY_LIMIT', 3)
    for stamp in range(5):
        await save_run(sqlite_store, {'started_at': stamp, 'status': 'running'})
    await save_run(sqlite_store, {'started_at': 4, 'status': 'completed'})
    history = json.loads(await sqlite_store.get_metadata(HISTORY_KEY))
    assert [h['started_at'] for h in history] == [2, 3, 4]
    assert history[-1]['status'] == 'completed'


def test_trend_uses_separate_consecutive_days_not_loop_iterations():
    def observation(day, count):
        return {'status': 'completed', 'completed_at': day * 86400,
                'tagging': {'after': {'pending': count}}}
    assert growing_pool_warning([observation(d, d * 10) for d in range(1, 5)])
    assert not growing_pool_warning([observation(1, n) for n in range(4)])
    assert not growing_pool_warning([observation(d, d) for d in (1, 3, 4, 5)])
    assert not growing_pool_warning([observation(d, 100 - d) for d in range(1, 5)])


async def test_audit_lowercase_severities_reach_saved_report(sqlite_store, temp_db_path):
    r = runner(sqlite_store, temp_db_path)
    await r._build_and_store_report({'jobs': {'health_check': {'status': 'ok', 'findings': [
        {'severity': 'warn', 'check': 'tagging_backlog', 'message': '100 pending'},
        {'severity': 'error', 'check': 'integrity', 'message': 'broken'},
    ]}}})
    report = json.loads(await sqlite_store.get_metadata('nightly_report'))
    assert any('100 pending' in w for w in report['warnings'])
    assert any('broken' in e for e in report['errors'])


async def test_tag_storage_failure_stays_retryable(sqlite_store, monkeypatch):
    mid = await seed(sqlite_store, [])
    await sqlite_store._ensure_tags_exist(['python', 'sql'])
    router = MagicMock(generate=AsyncMock(return_value='1: python, sql'))
    tagger = BatchTagger(sqlite_store, router, MemoryConfig())
    original = sqlite_store.tag_memory

    async def fail_real_tags(mid, tags):
        if 'python' in tags:
            raise RuntimeError('write failed')
        return await original(mid, tags)

    monkeypatch.setattr(sqlite_store, 'tag_memory', fail_real_tags)
    stats = await tagger.tag_batch()
    assert stats['failed'] == 1
    assert stats['memories_marked_skip'] == 0
    assert mid in await sqlite_store.get_poorly_tagged_memory_ids()


async def test_centroids_never_propagate_skip_or_neutral():
    store = MagicMock()
    store.get_tag_member_counts = AsyncMock(return_value={'_skip': 30, 'neutral': 30, 'python': 30})
    store.get_memory_ids_for_tag = AsyncMock(return_value=[1])
    vectors = MagicMock()
    vectors.get_embeddings_by_ids.return_value = {1: [1., 0.]}
    tagger = CentroidTagger(store, vectors, MemoryConfig())
    assert set(await tagger.build_centroids()) == {'python'}
    store.get_memory_ids_for_tag.assert_awaited_once_with('python', limit=200)


async def test_centroid_cap_selects_strongest_matches():
    store = MagicMock()
    store.get_poorly_tagged_memory_ids = AsyncMock(return_value=[1])
    store.get_tags_for_memories = AsyncMock(return_value={1: []})
    store.tag_memory = AsyncMock()
    vectors = MagicMock()
    vectors.get_embeddings_by_ids.return_value = {1: [1., 0.]}
    tagger = CentroidTagger(store, vectors, MemoryConfig(centroid_tag_similarity=0.5))
    centroids = {str(i): np.array([0.6 + i * 0.05, 0.]) for i in range(7)}
    centroids['invalid'] = np.array([np.nan, 0.])
    await tagger.tag_poorly_tagged(centroids)
    assert store.tag_memory.call_args.args[1] == ['6', '5', '4', '3', '2']


async def test_batch_inner_budget_uses_outer_job_override(sqlite_store, temp_db_path, monkeypatch):
    import blipshell.core.nightly as module
    r = runner(sqlite_store, temp_db_path)
    r.config.memory = MemoryConfig()
    fake = MagicMock()
    fake.tag_all = AsyncMock(return_value={})
    monkeypatch.setattr(module, 'BatchTagger', lambda *a: fake)
    monkeypatch.setitem(module._JOB_TIMEOUTS, 'batch_tag', 600)
    await r._job_batch_tag(lambda _: None)
    assert fake.tag_all.call_args.kwargs['time_budget_seconds'] == 570


def test_history_cli_is_read_only_and_does_not_boot_models(tmp_path, monkeypatch):
    import sqlite3
    import yaml
    from click.testing import CliRunner
    from blipshell.ui.cli import main
    db = tmp_path / 'history.db'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE app_metadata(key TEXT PRIMARY KEY, value TEXT)')
        connection.execute('INSERT INTO app_metadata VALUES (?, ?)', (HISTORY_KEY, '[]'))
    cfg = tmp_path / 'config.yaml'
    cfg.write_text(yaml.safe_dump({'database': {'path': str(db)}}))
    monkeypatch.setattr(NightlyRunner, 'create_from_config', AsyncMock(side_effect=AssertionError('boot')))
    before = db.read_bytes()
    result = CliRunner().invoke(main, ['--config-path', str(cfg), 'nightly', '--history', '--quiet'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []
    assert db.read_bytes() == before
