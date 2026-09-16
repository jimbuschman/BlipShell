import sqlite3

from scripts.audit_db import AuditResult, check_fts_sync


def corpus(path):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE memories(id INTEGER PRIMARY KEY, summary TEXT, content TEXT);
            CREATE VIRTUAL TABLE memories_fts USING fts5(summary, content, content=memories, content_rowid=id);
            INSERT INTO memories VALUES (1, NULL, 'raw content only'), (2, 'summary', 'message');
            INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');
        ''')


def test_unsummarized_raw_content_is_correctly_indexed(tmp_path):
    path = tmp_path / 'test.db'
    corpus(path)
    report = AuditResult()
    check_fts_sync(str(path), report)
    assert report.findings[0]['severity'] == 'ok'


def test_equal_counts_do_not_hide_wrong_index_membership(tmp_path):
    path = tmp_path / 'test.db'
    corpus(path)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO memories_fts(memories_fts, rowid, summary, content) VALUES ('delete', 1, NULL, 'raw content only')")
        db.execute("INSERT INTO memories_fts(rowid, summary, content) VALUES (3, 'unexpected', 'row')")
        assert db.execute('SELECT count(*) FROM memories_fts').fetchone()[0] == 2
        assert db.execute('SELECT count(*) FROM memories_fts_docsize').fetchone()[0] == 2
    report = AuditResult()
    check_fts_sync(str(path), report)
    assert report.findings[0]['severity'] == 'warn'
    assert '1 missing, 1 extra' in report.findings[0]['message']
