"""Automated user simulation system for BlipShell.

Usage:
    blipshell simulate                    # run all scenarios
    blipshell simulate -s <name>          # single scenario by name
    blipshell simulate -c regression      # all scenarios in a category
    blipshell simulate --list             # list all scenarios
    blipshell simulate --quiet -o out.json  # JSON output for automation
"""

from blipshell.simulate.models import SimScenario, SimSuiteResult
from blipshell.simulate.runner import SimRunner
from blipshell.simulate.scenarios import (
    collect_all_scenarios,
    filter_by_category,
    filter_by_name,
)

__all__ = [
    "SimRunner",
    "SimScenario",
    "SimSuiteResult",
    "collect_all_scenarios",
    "filter_by_category",
    "filter_by_name",
]
