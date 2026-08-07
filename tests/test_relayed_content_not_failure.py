"""Tools that RELAY content must not be failed by what the content says.

execute_tool_call classifies a result starting with "Error:" as the tool
failing. Right convention for self-reports — but read_file returning a log
whose first line is "Error: ..." and run_command relaying `grep Error:`
matches are successes whose payload merely begins with the magic word. Both
were marked failed: red X in the display, file-read tracking and the
executor's file cache skipped.

The disambiguation happens at the source, where the tool knows the truth
(the file was read; the exit code was 0) — not by weakening the chokepoint
heuristic, which must keep catching genuine self-reported errors.
"""

import sys

import pytest

from blipshell.core.tools.base import result_reports_failure
from blipshell.core.tools.filesystem import ReadFileTool
from blipshell.core.tools.shell import ShellTool


class TestReadFileRelaysErrorLogs:
    async def test_error_log_read_is_a_success(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("Error: connection refused\nretrying...\n", encoding="utf-8")

        result = await ReadFileTool().execute(str(log))

        assert not result_reports_failure(result), (
            "reading an error log was classified as the TOOL failing"
        )
        assert "Error: connection refused" in result, "content must survive intact"

    async def test_ordinary_small_file_stays_unwrapped(self, tmp_path):
        """The header is only for the collision case — plain reads keep the
        no-noise raw format."""
        f = tmp_path / "notes.txt"
        f.write_text("plain content\n", encoding="utf-8")

        result = await ReadFileTool().execute(str(f))

        assert result == "plain content\n"

    async def test_missing_file_is_still_a_failure(self, tmp_path):
        result = await ReadFileTool().execute(str(tmp_path / "nope.txt"))
        assert result_reports_failure(result)


class TestRunCommandRelaysErrorOutput:
    async def test_exit_zero_error_prefixed_stdout_is_a_success(self):
        """`grep Error: app.log` finding matches is the everyday case."""
        cmd = f'{sys.executable} -c "print(\'Error: found in log\')"'

        result = await ShellTool().execute(cmd)

        assert not result_reports_failure(result), (
            "a successful command whose output starts with Error: was "
            "classified as the tool failing"
        )
        assert "Error: found in log" in result

    async def test_plain_output_is_not_wrapped(self):
        cmd = f'{sys.executable} -c "print(\'all good\')"'
        result = await ShellTool().execute(cmd)
        assert result.startswith("all good")

    async def test_nonzero_exit_is_not_masked(self):
        """The [exit 0] disambiguator must never fire for real failures."""
        cmd = f'{sys.executable} -c "import sys; print(\'Error: broke\'); sys.exit(1)"'

        result = await ShellTool().execute(cmd)

        assert not result.startswith("[exit 0]")
        assert "Exit code: 1" in result
