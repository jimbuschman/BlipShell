"""The committed benchmark artifacts must not carry personal data.

`benchmark_results/` and `agent_eval_results/` are deliberately carved out of
.gitignore so two machines can share measurements, and the repo is public.
On 2026-09-02 a transcripts file recorded the `realdata` suite verbatim: 732
distinct real user messages, plus home-directory paths and the Ollama cloud
account name, all pushed. History was rewritten on 2026-09-04. This test is
the guard that fails BEFORE the commit next time.

It scans every committed artifact as text, so a new file type is covered by
default, and it runs from the repo root on either machine.
"""

import json
import re
from pathlib import Path

import pytest

from blipshell.benchmark.harness import LIVE_CORPUS_SUITES
from blipshell.benchmark.recording import LIVE_CORPUS_MARKER

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_DIRS = [REPO / "benchmark_results", REPO / "agent_eval_results"]

# A home directory with a real account name after it. Models write generic
# paths like `/home/user/project` in ordinary answers, and prose contains
# `terms/users/browsers`, so: `Users` must sit right after a drive letter or
# path root (case-sensitive; URLs use lowercase), and the next segment must
# not be a placeholder. Backslashes appear doubled inside JSON, hence `+`.
_PLACEHOLDER = r"(?!(?:\[user\]|someone|user|username|your\w*|<)(?:[\\/]|\b))"
_HOME_PATH = re.compile(
    r"(?:(?<![A-Za-z0-9_./-])[A-Za-z]:[\\/]+Users[\\/]+"
    r"|(?<![A-Za-z0-9_.-])/(?:c/)?Users/"
    r"|(?<![A-Za-z0-9_.-])/home/)" + _PLACEHOLDER
)

# Each pattern is something that has actually leaked, or the class it belongs to.
FORBIDDEN = {
    "home directory path": _HOME_PATH,
    "Tailscale address": re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    "account name in quota error": re.compile(r"you \((?!\[account\])[^)]{1,64}\) have reached"),
}


@pytest.mark.parametrize("leak", [
    r"C:\Users\somebody\app.py",
    r"C:\\Users\\somebody\\Downloads",          # as it appears inside JSON
    "/c/Users/somebody/repo",                    # Git Bash form
    "/Users/somebody/Documents",
    "/home/somebody/.ssh",
    "ResponseError: you (somebody) have reached your session usage limit",
    "url: http://100.97.1.2:11434",
])
def test_rules_catch_the_shapes_that_leaked(leak):
    assert any(p.search(leak) for p in FORBIDDEN.values()), leak


@pytest.mark.parametrize("benign", [
    "/Users/user/Documents/deploy_config.yaml",  # model-invented example path
    "/home/user/project",
    "terms/users/browsers",
    r"C:\Users\[user]\x",
    "/c/Users/[user]/repo",
    "/home/<name>/x",
    "/home/username/x",
    r"C:\Users\someone\app.py",
    "you ([account]) have reached your session usage limit",
    "http://10.0.0.5:11434 and 192.168.1.20",
])
def test_rules_ignore_placeholders_and_prose(benign):
    hits = {k for k, p in FORBIDDEN.items() if p.search(benign)}
    assert not hits, (benign, hits)


def _artifact_files():
    for d in ARTIFACT_DIRS:
        if d.is_dir():
            yield from (p for p in d.rglob("*") if p.is_file())


@pytest.mark.parametrize("path", sorted(_artifact_files()), ids=lambda p: p.name)
def test_artifact_has_no_personal_markers(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = {name: pat.search(text) for name, pat in FORBIDDEN.items()}
    hits = {k: v.group(0) for k, v in hits.items() if v}
    assert not hits, f"{path.relative_to(REPO)} contains {hits}"


@pytest.mark.parametrize(
    "path",
    sorted(p for p in _artifact_files() if p.name.endswith("__transcripts.json")),
    ids=lambda p: p.name,
)
def test_live_corpus_suites_are_redacted_in_transcripts(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    leaked = [
        c for c in data.get("calls", [])
        if c.get("suite") in LIVE_CORPUS_SUITES
        and any(c.get(k) not in (None, LIVE_CORPUS_MARKER)
                for k in ("prompt", "system", "response"))
    ]
    assert not leaked, (
        f"{path.name}: {len(leaked)} live-corpus call(s) recorded with text; "
        f"first task_type={leaked[0].get('task_type')!r}"
    )
