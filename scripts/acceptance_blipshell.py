"""Resumable acceptance checks on isolated copies, using actual local models.

Run with the development Python environment. `prepare` prints the run folder;
subsequent stages take --root pointing there. Never use the production DB as root.
The audit hook rejects Python file writes/SQLite opens outside that folder,
subprocesses, and non-loopback network access. It is a test guard, not an OS sandbox.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True


def emit(root, key, value):
    payload = {'at': time.time(), 'check': key, **value}
    with (root / 'events.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(payload, default=str) + '\n')
    # Keep private content and job result bodies in the isolated artifact only.
    print(json.dumps({k: payload[k] for k in ('at', 'check', 'status', 'elapsed_s') if k in payload}), flush=True)


def contains_release_code(text):
    # Models sometimes typeset the same identifier using a non-breaking hyphen.
    # Accept typographic hyphens, but require the exact name and number.
    import re
    return bool(re.search(r'\bMARIGOLD[-\u2010-\u2015\u2212]742\b', text, re.IGNORECASE))


def inside(root, path):
    try:
        Path(path).resolve().relative_to(root)
        return True
    except (ValueError, TypeError):
        return False


def install_guard(root):
    import tempfile
    temporary = root / 'tmp'
    temporary.mkdir(exist_ok=True)
    tempfile.tempdir = str(temporary)
    os.environ['TMP'] = os.environ['TEMP'] = str(temporary)
    os.environ['HF_HOME'] = str(root / 'model_cache')
    os.environ['TIKTOKEN_CACHE_DIR'] = str(root / 'tokenizer_cache')

    def reject(event, path):
        emit(root, 'isolation_violation', {'status': 'BLOCKED', 'event': event, 'path': str(path)})
        raise PermissionError(f'Acceptance guard blocked {event} outside the isolated run')

    def guard(event, args):
        if event == 'open':
            path, mode, flags = args
            if str(path).lower() == os.devnull.lower():
                return  # The OS null device is not a filesystem write.
            write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if write and not isinstance(path, int) and not inside(root, path):
                reject(event, path)
        elif event == 'sqlite3.connect':
            path = args[0]
            if str(path) != ':memory:' and not inside(root, path):
                reject(event, path)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod'):
            if not inside(root, args[0]):
                reject(event, args[0])
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            if not inside(root, args[0]) or not inside(root, args[1]):
                reject(event, args[:2])
        elif event in ('subprocess.Popen', 'os.system'):
            reject(event, 'subprocess execution is disabled')
        elif event == 'socket.getaddrinfo':
            host = args[0].decode('ascii') if isinstance(args[0], bytes) else args[0]
            if host not in ('localhost', '127.0.0.1', '::1', None):
                reject(event, args[0])
        elif event == 'socket.connect':
            address = args[1]
            if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1'):
                reject(event, address)

    sys.addaudithook(guard)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def copy_db(source, destination):
    with sqlite3.connect(Path(source).resolve().as_uri() + '?mode=ro', uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)


def counts(path):
    with sqlite3.connect(path) as db:
        return {name: db.execute(f'SELECT count(*) FROM {name}').fetchone()[0]
                for name in ('memories', 'memory_tags', 'sessions', 'lessons', 'core_memories', 'entities')}


def isolate_project_paths(root, db_path):
    """Copied project rows still name real repositories; redirect their exports."""
    with sqlite3.connect(db_path) as db:
        for project_id, in db.execute('SELECT id FROM projects').fetchall():
            target = root / 'projects' / Path(db_path).stem / str(project_id)
            target.mkdir(parents=True, exist_ok=True)
            db.execute('UPDATE projects SET root_path=? WHERE id=?', (str(target), project_id))


def prepare():
    from blipshell.core.config import ConfigManager
    import yaml
    original = ConfigManager().load()
    root = REPO / 'data' / ('acceptance_' + time.strftime('%Y%m%d_%H%M%S'))
    root.mkdir(exist_ok=False)
    source = Path(original.database.path).resolve()
    before = fingerprint(source)
    for phase in ('baseline', 'nightly', 'agent'):
        copy_db(source if phase == 'baseline' else root / 'baseline.db', root / f'{phase}.db')
    for phase in ('nightly', 'agent'):
        isolate_project_paths(root, root / f'{phase}.db')
    base = original.model_copy(deep=True)
    base.endpoints = [e for e in base.endpoints if e.provider == 'ollama' and not e.pii_sanitize]
    assert base.endpoints and all(e.url.rstrip('/') == 'http://localhost:11434' for e in base.endpoints)
    for ep in base.endpoints:
        ep.api_key = None
        ep.roles = list(set(ep.roles) | {'tool_calling', 'coding'})
    base.models.tool_calling = original.models.tool_calling_fallback
    base.models.coding = original.models.coding_fallback
    base.pii.local_mode_default = True
    base.database.backup_dir = str(root / 'backups')
    base.benchmark.db_path = str(root / 'benchmark.db')
    base.agent.auto_approve_tools = False
    base.robotics.enabled = False
    base.telegram.enabled = False
    for phase in ('nightly', 'agent'):
        cfg = base.model_copy(deep=True)
        cfg.database.path = str(root / f'{phase}.db')
        if phase == 'agent':
            # Avoid a second entity-maintenance run competing with conversation
            # tests; real entity extraction is exercised in nightly separately.
            cfg.memory.entity_extraction_batch_size = 0
            cfg.reflection.enabled = False  # Nightly exercises reflection separately.
        raw = cfg.model_dump()
        for field in ('token', 'bot_token', 'api_key'):
            if field in raw.get('telegram', {}):
                raw['telegram'][field] = ''
        (root / f'{phase}.yaml').write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    with sqlite3.connect(root / 'agent.db') as db:
        db.execute('INSERT OR REPLACE INTO app_metadata(key,value) VALUES (?,?)',
                   ('last_tag_discovery', datetime.now(timezone.utc).isoformat()))
    manifest = {'root': str(root), 'source': str(source), 'source_fingerprint': before,
                'baseline_counts': counts(root / 'baseline.db'), 'created_at': time.time(),
                'overrides': ['local models only', 'isolated backup/benchmark paths',
                              'tool approval required', 'robotics/telegram/idle reflection disabled',
                              'agent startup tag discovery deferred; startup entity batch size zero']}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (REPO / 'data' / 'acceptance_latest.txt').write_text(str(root), encoding='utf-8')
    assert fingerprint(source) == before
    emit(root, 'prepare', {'status': 'PASS', 'baseline_counts': manifest['baseline_counts']})
    print(str(root), flush=True)


async def restore(root):
    from scripts.backup_db import run_backup, backup_sqlite
    from blipshell.memory.sqlite_store import SQLiteStore
    import sqlite_vec
    before = counts(root / 'baseline.db')
    backup = run_backup(root / 'baseline.db', out_dir=root / 'restore_backups', quiet=True)
    assert backup
    restored = root / 'restored.db'
    backup_sqlite(str(backup / 'baseline.db'), str(restored))
    assert counts(restored) == before
    with sqlite3.connect(restored) as db:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        db.execute("INSERT INTO memories_fts(memories_fts,rank) VALUES('integrity-check',1)")
        db.rollback()
    store = SQLiteStore(str(restored))
    await store.initialize()
    assert await store.list_sessions(limit=1)
    await store.close()
    emit(root, 'backup_restore', {'status': 'PASS', 'counts': before})


async def nightly(root, jobs=None):
    from blipshell.core.nightly import NightlyRunner, JOB_ORDER
    runner = await NightlyRunner.create_from_config(str(root / 'nightly.yaml'), local_only=True)
    try:
        for job in jobs or JOB_ORDER:
            start = time.monotonic()
            emit(root, 'nightly:' + job, {'status': 'RUNNING'})
            before = counts(Path(runner.config.database.path))
            result = await runner.run(jobs=[job])
            job_result = result.get('jobs', {}).get(job, result)
            state = 'PASS' if job_result.get('status') == 'ok' else 'FAIL'
            if any(job_result.get(k) for k in ('failed', 'errors', 'error', 'warning')):
                state = 'FAIL'
            if job_result.get('disabled') or job_result.get('skipped') or job_result.get('skipped_reason'):
                state = 'NOT_EXERCISED'
            emit(root, 'nightly:' + job, {'status': state, 'elapsed_s': round(time.monotonic()-start, 2),
                 'result': job_result, 'before': before, 'after': counts(Path(runner.config.database.path))})
    finally:
        await runner.close()
        runner.vectors.close()


async def lifecycle(root):
    from blipshell.core.config import ConfigManager
    from blipshell.core.agent import Agent
    from blipshell.models.session import MessageRole
    from blipshell.models.tools import ToolCall
    cm = ConfigManager(root / 'agent.yaml')
    agent = Agent(cm.load(), cm)
    session_id = None
    try:
        emit(root, 'agent_start', {'status': 'RUNNING'})
        await asyncio.wait_for(agent.initialize(), 600)
        await web_auth(root, agent, cm)
        # Check actual approval dispatch without permitting any shell/file effect.
        forbidden = root / 'approval-must-not-exist.txt'
        denied = await agent.tool_registry.execute_tool_call(ToolCall(name='write_file',
            arguments={'path': str(forbidden), 'content': 'should not be written'}))
        assert not denied.success and not forbidden.exists()
        emit(root, 'tool_approval', {'status': 'PASS'})
        session_id = await asyncio.wait_for(agent.start_session(project='acceptance-isolated'), 600)
        emit(root, 'agent_start', {'status': 'PASS', 'session_id': session_id})
        marker = 'The acceptance project Lumen Orchard uses release code MARIGOLD-742.'
        for turn, prompt in enumerate((
            'This is an isolated acceptance test. Remember this project fact: ' + marker + ' Briefly acknowledge.',
            'What is the release code for the Lumen Orchard project? Answer briefly.',
            'Keep the release code unchanged. Briefly confirm the project name and its code.',
        )):
            start = time.monotonic()
            reply = await asyncio.wait_for(agent.chat(prompt), 600)
            assert reply.strip()
            if turn:
                assert contains_release_code(reply), 'Conversation failed to retain the release code'
            emit(root, 'chat_turn', {'status': 'PASS', 'elapsed_s': round(time.monotonic()-start, 2),
                 'contains_marker': contains_release_code(reply), 'endpoint': agent.last_endpoint_used})
        sm = agent.session_manager
        await sm.flush_pending_persists()
        before_ids = dict(sm._memory_db_ids)
        # Wait for actual background processing, without calling it a mock pass.
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            cursor = await agent.sqlite._db.execute('SELECT count(*) FROM memories WHERE session_id=? AND is_processed=0', (session_id,))
            if (await cursor.fetchone())[0] == 0:
                break
            await asyncio.sleep(2)
        else:
            raise TimeoutError('new session memories were not processed within ten minutes')
        matches = await asyncio.wait_for(agent.search.search('Lumen Orchard MARIGOLD-742'), 180)
        assert any(r.memory_id in before_ids.values() for r in matches)
        emit(root, 'persist_and_recall', {'status': 'PASS', 'session_id': session_id, 'memory_ids': list(before_ids.values())})
        older = len(sm.get_messages()) - 4
        compact = await asyncio.wait_for(agent.compact_conversation(), 300)
        assert compact.startswith('Compacted'), compact
        assert sm._memory_db_ids == {i-older+1: mid for i, mid in before_ids.items() if i >= older}
        emit(root, 'compaction_identity', {'status': 'PASS'})
        await asyncio.wait_for(agent.end_session(), 900)
        emit(root, 'session_close', {'status': 'PASS'})
    finally:
        await agent.force_cleanup()
    # New objects and a reopened store; no carried-over conversation state.
    resumed = Agent(cm.load(), cm)
    try:
        await asyncio.wait_for(resumed.start_session(resume_session_id=session_id), 600)
        assert resumed.session_manager.session_id == session_id
        assert any('MARIGOLD-742' in m.content for m in resumed.session_manager.get_messages())
        result = await asyncio.wait_for(resumed.chat('What release code did we record for Lumen Orchard? Answer briefly.'), 600)
        assert contains_release_code(result)
        emit(root, 'restart_and_resume', {'status': 'PASS', 'session_id': session_id})
        await asyncio.wait_for(resumed.end_session(), 900)
    finally:
        await resumed.force_cleanup()


async def web_auth(root, agent, config_manager):
    """Real FastAPI routes and auth dependency, without a second Agent startup."""
    import httpx
    from blipshell.ui.web import app as web
    from blipshell.models.config import AuthConfig
    saved = web._agent, web._config_manager, web._auth_config
    web._agent, web._config_manager = agent, config_manager
    web._auth_config = AuthConfig(enabled=True, api_key='acceptance-synthetic-key')
    try:
        application = web.create_app(str(root / 'agent.yaml'))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url='http://localhost') as client:
            assert (await client.get('/v1/models')).status_code == 401
            assert (await client.get('/v1/models', headers={'Authorization': 'Bearer wrong'})).status_code == 401
            assert (await client.get('/v1/models', headers={'Authorization': 'Bearer acceptance-synthetic-key'})).status_code == 200
            web._auth_config.api_key = ''
            assert (await client.get('/v1/models')).status_code == 503
        emit(root, 'web_auth_routes', {'status': 'PASS', 'transport': 'in-process ASGI; real Agent and route dependencies'})
    finally:
        web._agent, web._config_manager, web._auth_config = saved


async def resume(root):
    """Test reopening the persisted session even when its close step failed."""
    from blipshell.core.config import ConfigManager
    from blipshell.core.agent import Agent
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    session_id = [e['session_id'] for e in events
                  if e['check'] == 'persist_and_recall' and e['status'] == 'PASS'][-1]
    cm = ConfigManager(root / 'agent.yaml')
    agent = Agent(cm.load(), cm)
    try:
        await asyncio.wait_for(agent.start_session(resume_session_id=session_id), 600)
        assert agent.session_manager.session_id == session_id
        assert any(contains_release_code(m.content) for m in agent.session_manager.get_messages())
        result = await asyncio.wait_for(agent.chat('What release code did we record for Lumen Orchard? Answer briefly.'), 600)
        assert contains_release_code(result)
        await agent.session_manager.flush_pending_persists()
        emit(root, 'restart_after_interrupted_close', {'status': 'PASS', 'session_id': session_id})
        started = time.monotonic()
        close_outcomes = {}
        original_close = agent.session_manager.end_session
        async def observed_close(*args, **kwargs):
            result = await original_close(*args, **kwargs)
            close_outcomes.update(result)
            return result
        agent.session_manager.end_session = observed_close
        await asyncio.wait_for(agent.end_session(), 900)
        assert close_outcomes and all(v == 'ok' or v.startswith('skipped:') for v in close_outcomes.values()), close_outcomes
        emit(root, 'session_close_after_fix', {'status': 'PASS', 'elapsed_s': round(time.monotonic()-started, 2), 'outcomes': close_outcomes})
    finally:
        await agent.force_cleanup()


async def outage(root):
    from blipshell.core.config import ConfigManager
    from blipshell.llm.routing import build_routing
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.memory.batch_tagger import BatchTagger
    from blipshell.models.memory import Memory
    cfg = ConfigManager(root / 'agent.yaml').load()
    cfg.llm.timeout, cfg.llm.max_retries = 2, 0
    cfg.endpoints[0].url = 'http://127.0.0.1:1'
    _, offline = build_routing(cfg, local_only=True, disable_fallback=True)
    store = SQLiteStore(str(root / 'outage.db'))
    await store.initialize()
    try:
        for name in ('python', 'database', 'testing'):
            await store._db.execute('INSERT INTO tags(name) VALUES (?)', (name,))
        await store._db.commit()
        mid = await store.create_memory(Memory(role='user', content='Testing a Python database backup.',
            summary='Testing a Python database backup.'))
        failed = await BatchTagger(store, offline, cfg.memory).tag_batch()
        assert failed['failed'] == 1 and failed['memories_marked_skip'] == 0
        assert await store.count_poorly_tagged_memories() == 1
        emit(root, 'model_outage_retryable', {'status': 'PASS'})
        cfg = ConfigManager(root / 'agent.yaml').load()
        _, online = build_routing(cfg, local_only=True, disable_fallback=True)
        tagger = BatchTagger(store, online, cfg.memory)
        interrupted = await tagger.tag_all(time_budget_seconds=0.3)
        assert interrupted['interrupted_batches'] == 1, interrupted
        assert interrupted['remaining_pool'] == 1
        assert interrupted['memories_marked_skip'] == 0
        emit(root, 'live_batch_deadline_retryable', {'status': 'PASS', 'result': interrupted})
        result = await tagger.tag_batch()
        assert result['failed'] == 0 and result['memories_tagged'] == 1
        assert await store.count_poorly_tagged_memories() == 0
        emit(root, 'model_recovered', {'status': 'PASS'})
    finally:
        await store.close()


async def crash_worker(root):
    """Child process terminated by crash_probe while a real model call is pending."""
    from blipshell.core.config import ConfigManager
    from blipshell.llm.routing import build_routing
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.memory.batch_tagger import BatchTagger
    from blipshell.models.memory import Memory
    store = SQLiteStore(str(root / 'crash.db'))
    await store.initialize()
    for name in ('python', 'database', 'testing'):
        await store._db.execute('INSERT INTO tags(name) VALUES (?)', (name,))
    await store._db.commit()
    await store.create_memory(Memory(role='user', content='Testing a Python database backup.',
                                    summary='Testing a Python database backup.'))
    cfg = ConfigManager(root / 'agent.yaml').load()
    _, router = build_routing(cfg, local_only=True, disable_fallback=True)
    # Instrument entry only; the real router, transport, and model remain in use.
    generate = router.generate
    async def observed_generate(*args, **kwargs):
        (root / 'crash-ready').write_text(str(os.getpid()))
        return await generate(*args, **kwargs)
    router.generate = observed_generate
    await BatchTagger(store, router, cfg.memory).tag_batch()
    await store.close()


def crash_probe(root):
    """Own and terminate only the test child, then reopen its committed database."""
    import subprocess
    import signal
    assert not (root / 'crash.db').exists() and not (root / 'crash-ready').exists(), 'Use a fresh crash fixture per attempt'
    worker_pid = None
    with (root / 'crash-console.log').open('w') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'crash_worker',
                                  '--root', str(root)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while not (root / 'crash-ready').exists():
                assert child.poll() is None, 'Crash fixture failed before model dispatch'
                if time.monotonic() >= deadline:
                    raise TimeoutError('Crash fixture did not reach model dispatch')
                time.sleep(0.25)
            time.sleep(2)
            assert child.poll() is None, 'Model finished before interruption; retry with a fresh fixture'
            # Windows venv executables can be launchers. Terminate the actual
            # Python worker that wrote the readiness marker, not just its launcher.
            worker_pid = int((root / 'crash-ready').read_text())
            assert worker_pid > 0 and worker_pid != os.getpid()
            os.kill(worker_pid, signal.SIGTERM)
            child.wait(timeout=15)
        finally:
            if child.poll() is None:
                if worker_pid is not None:
                    try:
                        os.kill(worker_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                child.kill()
                child.wait(timeout=15)
    import sqlite_vec
    with sqlite3.connect(root / 'crash.db') as db:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not db.execute('PRAGMA foreign_key_check').fetchall()
        assert db.execute('SELECT count(*) FROM memories').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM memory_tags').fetchone()[0] == 0
    emit(root, 'hard_process_interruption', {'status': 'PASS', 'child_exit_code': child.returncode,
         'scope': 'synthetic SQLite fixture, real tagger dispatch, forced process termination; committed input retained'})


async def recover_crash(root):
    from blipshell.core.config import ConfigManager
    from blipshell.llm.routing import build_routing
    from blipshell.memory.sqlite_store import SQLiteStore
    from blipshell.memory.batch_tagger import BatchTagger
    cfg = ConfigManager(root / 'agent.yaml').load()
    _, router = build_routing(cfg, local_only=True, disable_fallback=True)
    store = SQLiteStore(str(root / 'crash.db'))
    await store.initialize()
    try:
        assert await store.count_poorly_tagged_memories() == 1
        result = await BatchTagger(store, router, cfg.memory).tag_batch()
        assert result['failed'] == 0 and result['memories_tagged'] == 1, result
        assert await store.count_poorly_tagged_memories() == 0
        emit(root, 'hard_crash_work_recovered', {'status': 'PASS'})
    finally:
        await store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'restore', 'nightly', 'lifecycle', 'resume', 'outage', 'crash', 'crash_worker', 'recover_crash', 'verify_source'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--jobs', nargs='+')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
        return
    root = args.root.resolve()
    assert inside(REPO / 'data', root) and (root / 'manifest.json').is_file()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if args.stage == 'crash':
        crash_probe(root)
        return
    if args.stage == 'verify_source':
        assert fingerprint(manifest['source']) == manifest['source_fingerprint']
        emit(root, 'production_unchanged', {'status': 'PASS'})
        return
    install_guard(root)
    logging.basicConfig(filename=root / (args.stage + '.log'), level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    try:
        asyncio.run(nightly(root, args.jobs) if args.stage == 'nightly' else globals()[args.stage](root))
    except BaseException as error:
        emit(root, args.stage, {'status': 'FAIL', 'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
