"""Archived graph husks must not inflate the active orphan warning."""

import sqlite3

import pytest

from scripts.audit_db import AuditResult, check_entity_quality


@pytest.mark.parametrize('active_orphans', [0, 1])
def test_orphan_severity_uses_active_population(tmp_path, active_orphans):
    db = tmp_path / 'audit.db'
    with sqlite3.connect(db) as conn:
        conn.executescript('''
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY, name TEXT, entity_type TEXT, is_archived INTEGER
            );
            CREATE TABLE entity_mentions (entity_id INTEGER);
            CREATE TABLE entity_relationships (subject_id INTEGER, object_id INTEGER);
        ''')
        conn.executemany('INSERT INTO entities VALUES (?, ?, ?, ?)',
                         [(i, f'entity{i}', 'concept', int(i > 10)) for i in range(1, 111)])
        conn.executemany('INSERT INTO entity_mentions VALUES (?)',
                         [(i,) for i in range(1 + active_orphans, 11)])
    result = AuditResult()
    check_entity_quality(str(db), result)
    findings = {f['check']: f for f in result.findings}
    active = findings['orphans']
    assert active['severity'] == ('warn' if active_orphans else 'ok')
    assert active['message'] == (
        '1 active orphaned entities (10.0% of active entities)' if active_orphans
        else 'No active orphaned entities'
    )
    assert findings['archived_orphans']['severity'] == 'info'
    assert findings['archived_orphans']['message'].startswith('100 archived entities')
