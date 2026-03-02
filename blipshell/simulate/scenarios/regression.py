"""Scenarios: Regression tests for known bugs.

One scenario per previously-found bug, with a reference to the fix commit.
These are pure state assertions — no LLM calls needed.
"""

from blipshell.simulate.models import SimScenario, SimStep, StepAction


def get_scenarios() -> list[SimScenario]:
    return [
        _glob_grep_after_deactivate(),
        _glob_grep_in_simple_chat(),
        _session_required_for_chat(),
        _think_state_persists(),
        _project_tools_dont_leak(),
    ]


def _glob_grep_after_deactivate() -> SimScenario:
    """Regression: glob_files/grep_files vanish after project deactivation.

    Fix: commit 5270b36 — registered globally in agent_tools.py.
    But deactivate_project() still calls unregister() on them (lines 123-124).
    This scenario catches if they don't get re-registered.
    """
    return SimScenario(
        name="regression_glob_grep_after_deactivate",
        description="glob_files/grep_files must survive project deactivation",
        category="regression",
        steps=[
            # Verify present before project
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Before project: grep_files and glob_files present",
                expect_tools_registered=["grep_files", "glob_files"],
            ),
            # Activate project
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Activate project",
                expect_tools_registered=["grep_files", "glob_files"],
            ),
            # Deactivate
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate project",
            ),
            # REGRESSION CHECK: must still be present
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="REGRESSION: grep_files/glob_files must survive deactivation",
                expect_tools_registered=["grep_files", "glob_files"],
            ),
        ],
    )


def _glob_grep_in_simple_chat() -> SimScenario:
    """Regression: glob_files/grep_files missing in simple chat entirely.

    Fix: commit 5270b36 — registered in _register_tools().
    """
    return SimScenario(
        name="regression_glob_grep_simple_chat",
        description="glob_files/grep_files present in simple chat from the start",
        category="regression",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Simple chat: grep_files registered at startup",
                expect_tools_registered=["grep_files"],
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Simple chat: glob_files registered at startup",
                expect_tools_registered=["glob_files"],
            ),
        ],
    )


def _session_required_for_chat() -> SimScenario:
    """Regression: chat() should guard against missing session_manager.

    Fix: commit in code review fixes (feature 38).
    Tests that the session exists before chat is attempted.
    """
    return SimScenario(
        name="regression_session_exists",
        description="Session manager exists after start_session()",
        category="regression",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="session_manager is not None",
                custom_validator=lambda agent: (
                    []
                    if agent.session_manager is not None
                    else ["session_manager is None after start_session"]
                ),
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="session_id is set",
                custom_validator=lambda agent: (
                    []
                    if agent.session_manager and agent.session_manager.session_id
                    else ["session_id not set"]
                ),
            ),
        ],
    )


def _think_state_persists() -> SimScenario:
    """Verify think state toggles correctly and doesn't reset unexpectedly."""
    return SimScenario(
        name="regression_think_toggle_state",
        description="Think toggle state is consistent",
        category="regression",
        steps=[
            # Record initial state
            SimStep(
                action=StepAction.SLASH,
                input="/think off",
                description="Set think off",
                expect_think_enabled=False,
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/think on",
                description="Set think on",
                expect_think_enabled=True,
            ),
            # Do something unrelated — state should persist
            SimStep(
                action=StepAction.SLASH,
                input="/status",
                description="Run unrelated command",
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Think still on after unrelated command",
                expect_think_enabled=True,
            ),
            # Reset
            SimStep(
                action=StepAction.SLASH,
                input="/think off",
                description="Clean up: think off",
                expect_think_enabled=False,
            ),
        ],
    )


def _project_tools_dont_leak() -> SimScenario:
    """Project-only tools should never appear in simple chat mode."""
    return SimScenario(
        name="regression_project_tools_dont_leak",
        description="Project-only tools never appear without active project",
        category="regression",
        steps=[
            # No project
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="No project: git tools absent",
                expect_project_inactive=True,
                expect_tools_not_registered=[
                    "git_status", "git_diff", "git_add", "git_commit",
                    "ask_user", "task_complete",
                ],
            ),
            # Activate and deactivate
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Activate",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate",
            ),
            # After deactivation: project tools gone
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="After deactivation: git tools absent again",
                expect_tools_not_registered=[
                    "git_status", "git_diff", "git_add", "git_commit",
                    "ask_user", "task_complete",
                ],
            ),
        ],
    )
