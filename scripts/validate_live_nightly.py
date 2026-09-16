"""Bounded live Ollama validation. Source corpus is opened read-only.

Copies at most twenty pending summaries and the tag vocabulary to a new
scratch database. All tagging and run-history writes go to that database.
No cloud request is allowed. Results contain metrics, never source text.
"""
import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blipshell.core.config import ConfigManager
from blipshell.core.nightly import NightlyRunner
from blipshell.core.nightly_history import HISTORY_KEY
from blipshell.llm.routing import build_routing
from blipshell.memory.sqlite_store import SQLiteStore
from blipshell.memory.tag_health import tag_health_async
from blipshell.models.memory import Memory


async def main():
    config = ConfigManager().load().model_copy(deep=True)
    source = Path(config.database.path).resolve()
    output = Path('data') / ('live_validation_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    scratch = (output / 'sample.db').resolve()
    report = {'started_at': time.time(), 'sample_database': str(scratch), 'requests': []}

    def event(name, **values):
        report[name] = values
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(name, json.dumps(values), flush=True)

    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original:
        rows = original.execute("""SELECT m.id, m.summary FROM memories m
            LEFT JOIN memory_tags mt ON mt.memory_id=m.id
            WHERE m.is_archived=0 AND m.summary IS NOT NULL
            AND m.id NOT IN (SELECT mt2.memory_id FROM memory_tags mt2
                JOIN tags t ON t.id=mt2.tag_id WHERE t.name='_skip')
            GROUP BY m.id HAVING count(mt.id)<=1 ORDER BY m.timestamp DESC LIMIT 20""").fetchall()
        vocabulary = original.execute('SELECT DISTINCT name FROM tags').fetchall()
        original_tags = {mid: [r[0] for r in original.execute(
            'SELECT t.name FROM tags t JOIN memory_tags mt ON mt.tag_id=t.id WHERE mt.memory_id=?', (mid,))]
            for mid, _ in rows}

    config.database.path = str(scratch)
    config.llm.max_retries = 0  # one bounded attempt; do not retry a stressed server
    config.llm.timeout = 180
    store = SQLiteStore(str(scratch))
    await store.initialize()
    for name, in vocabulary:
        await store._db.execute('INSERT OR IGNORE INTO tags(name) VALUES (?)', (name,))
    await store._db.commit()
    for old_id, summary in rows:
        mid = await store.create_memory(Memory(role='user', content=summary, summary=summary, is_processed=True))
        if original_tags[old_id]:
            await store.tag_memory(mid, original_tags[old_id])

    manager, router = build_routing(config, local_only=True, disable_fallback=True)
    original_generate = router._gated_generate

    async def traced(endpoint, prompt, model, system, kwargs):
        assert not endpoint.should_sanitize_pii, 'Live private sample must stay local'
        start = time.monotonic()
        record = {'endpoint': endpoint.name, 'model': model, 'started_at': time.time()}
        report['requests'].append(record)
        try:
            result = await original_generate(endpoint, prompt, model, system, {**kwargs, 'use_cache': False})
            record['response_chars'] = len(result)
            return result
        finally:
            record['elapsed_s'] = round(time.monotonic() - start, 3)

    router._gated_generate = traced
    runner = NightlyRunner(config, store, None, router, None)
    event('sample', rows=len(rows), vocabulary=len(vocabulary), health=await tag_health_async(store))

    async def foreground(label):
        start = time.monotonic()
        reply = await asyncio.wait_for(router.generate('reasoning',
            'Reply with exactly LIVE_OK and nothing else.', think=False, use_cache=False), 210)
        event(label, elapsed_s=round(time.monotonic()-start, 3), expected_reply='LIVE_OK' in reply,
              last_endpoint=manager._last_routed.get('reasoning'))

    try:
        await foreground('baseline_foreground')
        # Exercise a real in-flight nightly cancellation, then reopen its DB.
        interrupted = asyncio.create_task(runner.run(jobs=['batch_tag']))
        await asyncio.sleep(3)
        interrupted.cancel()
        await asyncio.gather(interrupted, return_exceptions=True)
        history = json.loads(await store.get_metadata(HISTORY_KEY) or '[]')
        event('interruption', history_status=history[-1].get('status') if history else None,
              active_job=history[-1].get('active_job') if history else None,
              health=await tag_health_async(store))
        await store.close()
        store = SQLiteStore(str(scratch))
        await store.initialize()
        runner = NightlyRunner(config, store, None, router, None)
        # A genuine foreground router request while the batch holds the gate.
        nightly = asyncio.create_task(runner.run(jobs=['batch_tag']))
        await asyncio.sleep(2)
        foreground_task = asyncio.create_task(foreground('foreground_during_nightly'))
        result, _ = await asyncio.wait_for(asyncio.gather(nightly, foreground_task), 420)
        event('resumed_run', elapsed_s=result['elapsed_s'], jobs=result['jobs'],
              health=await tag_health_async(store))
        # Repeat a drained run to confirm it does not repeat model work.
        if (await tag_health_async(store))['pending'] == 0:
            calls_before = len(report['requests'])
            repeat = await runner.run(jobs=['batch_tag'])
            event('drained_rerun', model_calls=len(report['requests'])-calls_before,
                  jobs=repeat['jobs'])
        import ollama
        client = ollama.AsyncClient(host='http://localhost:11434', timeout=60)
        start = time.monotonic()
        vector = await client.embed(model=config.models.embedding, input='Synthetic live embedding validation')
        event('embedding', elapsed_s=round(time.monotonic()-start, 3), dimension=len(vector.embeddings[0]))
        history = json.loads(await store.get_metadata(HISTORY_KEY) or '[]')
        event('finished', history_statuses=[r['status'] for r in history], integrity=(
            await (await store._db.execute('PRAGMA quick_check')).fetchone())[0])
    except Exception as error:
        event('failure', kind=type(error).__name__, message=str(error))
        raise
    finally:
        await store.close()
        event('closed', completed_at=time.time())


if __name__ == '__main__':
    asyncio.run(main())
