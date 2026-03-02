"""Scenario collection and filtering."""

from __future__ import annotations

from blipshell.simulate.models import SimScenario

# Import all scenario modules
from blipshell.simulate.scenarios.tool_registration import get_scenarios as _tool_reg
from blipshell.simulate.scenarios.slash_commands import get_scenarios as _slash
from blipshell.simulate.scenarios.mode_transitions import get_scenarios as _modes
from blipshell.simulate.scenarios.regression import get_scenarios as _regression
from blipshell.simulate.scenarios.chat_workflows import get_scenarios as _chat
from blipshell.simulate.scenarios.project_workflows import get_scenarios as _project
from blipshell.simulate.scenarios.error_recovery import get_scenarios as _errors


def collect_all_scenarios() -> list[SimScenario]:
    """Collect all scenarios from all modules."""
    all_scenarios: list[SimScenario] = []
    for getter in [_tool_reg, _slash, _modes, _regression, _chat, _project, _errors]:
        all_scenarios.extend(getter())
    return all_scenarios


def filter_by_category(scenarios: list[SimScenario], category: str) -> list[SimScenario]:
    """Filter scenarios by category."""
    return [s for s in scenarios if s.category == category]


def filter_by_name(scenarios: list[SimScenario], name: str) -> list[SimScenario]:
    """Filter scenarios to a specific name."""
    return [s for s in scenarios if s.name == name]
