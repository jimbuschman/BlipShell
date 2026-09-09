"""A model reply with an emoji must not crash the CLI on a cp1252 console.

Tested in a subprocess with the stream encoding forced, because the point is
the process-level stream state, not a mocked print.
"""

from __future__ import annotations

import os
import subprocess
import sys

PRINT = "print('\\u2705 checkmark ok')"


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0"}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, timeout=60)


def test_unhardened_cp1252_stream_crashes_on_an_emoji():
    """Negative control: without hardening the failure is real."""
    r = _run(PRINT)
    assert r.returncode != 0 and b"UnicodeEncodeError" in r.stderr


def test_hardened_stream_substitutes_instead():
    r = _run("from blipshell.ui.encoding import harden_stdio; harden_stdio(); " + PRINT)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert b"checkmark ok" in r.stdout


def test_utf8_stream_is_left_alone():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c",
                        "import sys; from blipshell.ui.encoding import harden_stdio; harden_stdio(); "
                        "print(sys.stdout.errors); " + PRINT],
                       capture_output=True, env=env, timeout=60)
    assert r.returncode == 0
    assert r.stdout.split(b"\n")[0].strip() == b"strict"
