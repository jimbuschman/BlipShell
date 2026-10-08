"""Live, read-only self-inspection for BlipShell.

The architecture card explains how BlipShell is designed.  This tool answers
the complementary operational questions: which installation/config/database
is this process using, is the in-process scheduler alive, and what actually
happened in recent nightly runs.  Keeping this behind a tool avoids putting a
changing status dump in every prompt while giving the model an authoritative
alternative to guessing paths or searching its own source tree.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from blipshell.core.nightly_history import HISTORY_KEY, decode_history
from blipshell.core.tools.base import Tool
from blipshell.models.tools import ToolDefinition


def _when(value) -> str:
    """Render an epoch timestamp in the machine's local timezone."""
    if not isinstance(value, (int, float)):
        return "unknown"
    return datetime.fromtimestamp(value).astimezone().strftime("%Y-%m-%d %I:%M %p %Z")


class InspectRuntimeTool(Tool):
    """Report live paths, scheduler state, and maintenance history."""

    read_only = True

    def __init__(self, config, config_manager, sqlite, state_provider: Callable[[], dict]):
        self._config = config
        self._config_manager = config_manager
        self._sqlite = sqlite
        self._state_provider = state_provider

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="inspect_runtime",
            description=(
                "Inspect YOUR CURRENT BLIPSHELL PROCESS and installation. Use this first "
                "when the user asks whether nightly maintenance ran or is scheduled, where "
                "your config/database/project files are, which instance you are using, or "
                "whether your own runtime is working. Returns authoritative live paths, "
                "scheduler state, the last report, and recent nightly history. Do not guess "
                "paths or search source code for these facts."
            ),
            parameters=[],
        )

    async def execute(self, **kwargs) -> str:
        state = self._state_provider() or {}
        config_path = Path(self._config_manager.config_path).resolve()
        db_path = Path(self._config.database.path).resolve()
        install_root = Path(__file__).resolve().parents[3]

        project = state.get("active_project")
        if project:
            project_text = f"{project.get('name', 'unnamed')} at {project.get('root_path', 'unknown')}"
        else:
            project_text = "none (file tools use their non-project defaults)"

        scheduler = state.get("nightly_scheduler")
        if scheduler == "running":
            scheduler_text = (
                "running in this process; checks around 2 AM local time every 15 minutes, "
                "requires the app to be running and the user idle for 5 minutes"
            )
        elif scheduler == "stopped":
            scheduler_text = "not running in this process"
        else:
            scheduler_text = "state unavailable"

        lines = [
            "BLIPSHELL RUNTIME — live read-only inspection",
            f"- Installation root: {install_root}",
            f"- Config file: {config_path}",
            f"- Database: {db_path} ({'exists' if db_path.exists() else 'MISSING'})",
            f"- Process working directory: {Path.cwd()}",
            f"- Active project: {project_text}",
            f"- Nightly scheduler: {scheduler_text}",
        ]

        try:
            raw = await self._sqlite.get_metadata("nightly_last_run")
            last = json.loads(raw) if raw else None
            if last:
                status = last.get("status", "completed")
                elapsed = last.get("elapsed_s")
                elapsed_text = f", {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
                lines.append(
                    f"- Last nightly completion: {_when(last.get('completed_at'))} "
                    f"(status={status}{elapsed_text})"
                )
            else:
                lines.append("- Last nightly completion: never recorded")
        except Exception as exc:
            lines.append(f"- Last nightly completion: unreadable ({type(exc).__name__})")

        try:
            raw = await self._sqlite.get_metadata("nightly_report")
            report = json.loads(raw) if raw else None
            if report:
                errors = report.get("errors", [])
                warnings = report.get("warnings", [])
                statuses = report.get("job_statuses", {})
                jobs = sum(value for value in statuses.values() if isinstance(value, int))
                ok = statuses.get("ok", 0)
                lines.append(
                    f"- Last report: {ok}/{jobs} jobs ok; "
                    f"{len(warnings)} warning(s), {len(errors)} error(s)"
                )
                for item in errors[:3]:
                    lines.append(f"  error: {item}")
                for item in warnings[:3]:
                    lines.append(f"  warning: {item}")
            else:
                lines.append("- Last report: none recorded")
        except Exception as exc:
            lines.append(f"- Last report: unreadable ({type(exc).__name__})")

        try:
            history = decode_history(await self._sqlite.get_metadata(HISTORY_KEY))
            completed = [item for item in history if item.get("status") == "completed"][-5:]
            if completed:
                lines.append("- Recent completed runs (oldest to newest):")
                for item in completed:
                    tagging = item.get("tagging", {})
                    before = tagging.get("before", {}).get("pending")
                    after = tagging.get("after", {}).get("pending")
                    pool = f", tagging pending {before}->{after}" if before is not None and after is not None else ""
                    lines.append(f"  {_when(item.get('completed_at'))}{pool}")
            else:
                lines.append("- Recent completed runs: none recorded")
        except Exception as exc:
            lines.append(f"- Recent completed runs: unreadable ({type(exc).__name__})")

        lines.append(
            "This snapshot is the source of truth for runtime state. "
            "describe_architecture explains the design rather than current status."
        )
        return "\n".join(lines)
