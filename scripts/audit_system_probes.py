"""Run the regression tests replacing the original defect reproductions.

Uses mocked clients and temporary test databases, never the live corpus.
Run with the project's development Python environment.
"""
from pathlib import Path
import sys
import tempfile
import pytest

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    with tempfile.TemporaryDirectory(prefix='blipshell-audit-') as temporary:
        raise SystemExit(pytest.main([
            str(root / 'tests' / 'test_system_audit_fixes.py'),
            '-q', '-p', 'no:cacheprovider', '--basetemp', str(Path(temporary) / 'tests'),
        ]))
