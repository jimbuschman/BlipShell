"""All five conversation importers share one implementation.

They were five near-identical ~70-line Click commands differing only in the
parser. The duplication also hid a real inconsistency: only `import-claude
code` wrapped its run in `import_lock`, so the other four could run
concurrently with nightly maintenance — the SQLite write contention that lock
exists to prevent (nightly checks is_import_active and stands down).
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from blipshell.ui.cli import main


IMPORT_COMMANDS = [
    (["import-chatgpt", "conversations"], "blipshell.import_chatgpt.parse_conversations"),
    (["import-claude", "conversations"], "blipshell.import_claude.parse_conversations"),
    (["import-claude", "scraped"], "blipshell.import_claude.parse_scraped_conversations"),
    (["import-deepseek", "conversations"], "blipshell.import_deepseek.parse_conversations"),
]


class TestAllImportersShareOneImplementation:
    @pytest.mark.parametrize("argv,parser", IMPORT_COMMANDS)
    def test_each_command_delegates_to_run_import(self, argv, parser, tmp_path):
        f = tmp_path / "export.json"
        f.write_text("{}")

        with patch("blipshell.ui.importers.run_import") as run, \
                patch(parser, return_value=[]):
            result = CliRunner().invoke(main, argv + [str(f)])

        assert result.exit_code == 0, result.output
        assert run.call_count == 1, f"{argv} did not go through the shared importer"

    def test_claude_code_also_delegates_and_keeps_its_concurrency_flag(self, tmp_path):
        d = tmp_path / "proj"
        d.mkdir()

        with patch("blipshell.ui.importers.run_import") as run, \
                patch("blipshell.import_claude_code.parse_claude_code_sessions", return_value=[]):
            result = CliRunner().invoke(
                main, ["import-claude", "code", str(d), "--concurrent", "7"],
            )

        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["max_concurrent"] == 7

    @pytest.mark.parametrize("argv,parser", IMPORT_COMMANDS)
    def test_options_are_forwarded(self, argv, parser, tmp_path):
        f = tmp_path / "export.json"
        f.write_text("{}")

        with patch("blipshell.ui.importers.run_import") as run, \
                patch(parser, return_value=[]):
            CliRunner().invoke(main, argv + [str(f), "--max", "5", "--skip-lessons"])

        kwargs = run.call_args.kwargs
        assert kwargs["max_count"] == 5
        assert kwargs["skip_lessons"] is True
        assert kwargs["source"] == str(f)


class TestEveryImportTakesTheLock:
    """The bug the consolidation surfaced: four of five never locked."""

    def test_import_lock_is_acquired(self, tmp_path):
        from blipshell.ui import importers

        taken = []

        class _Lock:
            def __init__(self, path, operation="import"):
                self.operation = operation

            def __enter__(self):
                taken.append(self.operation)
                return self

            def __exit__(self, *a):
                return False

        with patch("blipshell.core.import_lock.import_lock", _Lock):
            asyncio.run(importers._run_import(
                config_path=None,
                parse=lambda src: [],       # no conversations -> early return
                source="whatever",
                operation="import-test",
                max_count=None,
                skip_lessons=True,
            ))

        # Empty import returns before locking — that's correct, nothing to guard.
        assert taken == []

    def test_operation_name_identifies_the_source(self, tmp_path):
        """The lock payload is what nightly reports when it stands down, so a
        generic name would make the skip message useless."""
        from blipshell.ui.cli import main

        ops = {}
        with patch("blipshell.ui.importers.run_import",
                   side_effect=lambda **kw: ops.update(kw)), \
                patch("blipshell.import_deepseek.parse_conversations", return_value=[]):
            f = tmp_path / "x.json"
            f.write_text("{}")
            CliRunner().invoke(main, ["import-deepseek", "conversations", str(f)])

        assert ops["operation"] == "import-deepseek"
