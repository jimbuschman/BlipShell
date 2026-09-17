"""`blipshell nightly --job X --loop` applies core/nightly_loop.decide.

Drives the real Click command with a scripted NightlyRunner (no models, no
database), the way test_history_cli_is_read_only does. The policy itself is
tested in test_nightly_loop_policy.py; this pins that the CLI retries,
stops honestly and exits nonzero instead of announcing "done".
"""

import json
from unittest.mock import AsyncMock

import yaml
from click.testing import CliRunner

from blipshell.core import nightly_loop
from blipshell.core.nightly import NightlyRunner
from blipshell.ui.cli import main


class ScriptedRunner:
    def __init__(self, passes):
        self._passes = list(passes)
        self.calls = 0
        self.closed = False

    async def run(self, jobs=None, on_status=None):
        self.calls += 1
        stats = self._passes.pop(0) if self._passes else _tag_pass(remaining_pool=0, stopped_early=False)
        return {"jobs": {"batch_tag": stats}, "elapsed_s": 1.0}

    async def close(self):
        self.closed = True


def _tag_pass(**over):
    base = {"batches": 1, "checked": 10, "memories_tagged": 8, "memories_marked_skip": 2,
            "errors": 0, "stopped_early": True, "stop_reason": "time budget reached",
            "interrupted_batches": 0, "remaining_pool": 14515, "status": "ok", "elapsed_s": 1.0}
    base.update(over)
    return base


def _invoke(tmp_path, monkeypatch, runner):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"database": {"path": str(tmp_path / "x.db")}}))
    monkeypatch.setattr(NightlyRunner, "create_from_config", AsyncMock(return_value=runner))
    monkeypatch.setattr(nightly_loop, "RETRY_BACKOFF_SECONDS", 0.0)
    return CliRunner().invoke(main, ["--config-path", str(cfg), "nightly",
                                     "--job", "batch_tag", "--loop", "--quiet"])


def test_interrupted_first_batch_is_retried_and_the_loop_drains(tmp_path, monkeypatch):
    """The overnight shape, then recovery: pass 1 overran inside its first
    batch (checked 0, 14,515 remain) - the old loop stopped here and said
    done. Now it retries; pass 2 tags; pass 3 reports the pool drained."""
    runner = ScriptedRunner([
        _tag_pass(checked=0, memories_tagged=0, memories_marked_skip=0, interrupted_batches=1),
        _tag_pass(remaining_pool=14505),
        _tag_pass(remaining_pool=0, stopped_early=False, stop_reason=None),
    ])
    result = _invoke(tmp_path, monkeypatch, runner)
    assert result.exit_code == 0, result.output
    assert runner.calls == 3, "the loop stopped before the pool was drained"
    assert runner.closed


def test_three_no_progress_passes_stop_honestly_and_exit_nonzero(tmp_path, monkeypatch):
    runner = ScriptedRunner([
        {"status": "timeout", "error": "Timed out after 300s", "elapsed_s": 300.0},
        _tag_pass(checked=0, memories_tagged=0, memories_marked_skip=0, interrupted_batches=1),
        {"status": "error", "error": "connection refused", "elapsed_s": 0.5},
        _tag_pass(remaining_pool=0),   # never reached
    ])
    result = _invoke(tmp_path, monkeypatch, runner)
    assert result.exit_code == 1, result.output
    assert runner.calls == nightly_loop.MAX_STALLED_PASSES
    last = json.loads(result.output.strip().splitlines()[-1])
    assert last["loop"] == "stalled" and "connection refused" in last["reason"]


def test_progress_resets_the_stall_counter(tmp_path, monkeypatch):
    """Two failures, progress, two failures: never three in a row, so the
    loop keeps going until the pool drains."""
    runner = ScriptedRunner([
        {"status": "error", "error": "e1", "elapsed_s": 0.1},
        {"status": "error", "error": "e2", "elapsed_s": 0.1},
        _tag_pass(remaining_pool=14500),
        {"status": "error", "error": "e3", "elapsed_s": 0.1},
        {"status": "error", "error": "e4", "elapsed_s": 0.1},
        _tag_pass(remaining_pool=0, stopped_early=False, stop_reason=None),
    ])
    result = _invoke(tmp_path, monkeypatch, runner)
    assert result.exit_code == 0, result.output
    assert runner.calls == 6
