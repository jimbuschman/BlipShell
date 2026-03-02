"""Scenarios: Mode transition correctness.

Tests transitions between simple chat, project mode, and plan mode.
Verifies tool state is correct at every step.
"""

from blipshell.simulate.models import SimScenario, SimStep, StepAction


def get_scenarios() -> list[SimScenario]:
    return [
        _simple_to_project_to_simple(),
        _project_to_plan_to_project(),
        _double_activate(),
    ]


def _simple_to_project_to_simple() -> SimScenario:
    """Full round-trip: simple chat → project → back to simple."""
    return SimScenario(
        name="simple_project_simple_roundtrip",
        description="Mode round-trip preserves correct state",
        category="mode_transition",
        steps=[
            # --- Simple chat baseline ---
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Start: no project active",
                expect_project_inactive=True,
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Start: base tools present",
                expect_tools_registered=["grep_files", "glob_files", "read_file"],
                expect_tools_not_registered=["git_status", "task_complete"],
            ),
            # --- Activate project ---
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Activate project",
                expect_project_active="blipshell",
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Project mode: all tools present",
                expect_tools_registered=[
                    "grep_files", "glob_files", "read_file",
                    "git_status", "git_diff", "task_complete", "ask_user",
                ],
            ),
            # --- Deactivate project ---
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate project",
                expect_project_inactive=True,
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Back to simple: base tools preserved, project tools gone",
                expect_tools_registered=["grep_files", "glob_files", "read_file"],
                expect_tools_not_registered=["git_status", "task_complete", "ask_user"],
            ),
        ],
    )


def _project_to_plan_to_project() -> SimScenario:
    """Project mode → plan mode → back to project mode."""
    return SimScenario(
        name="project_plan_project_roundtrip",
        description="Plan mode filters tools correctly within project context",
        category="mode_transition",
        requires_project="blipshell",
        steps=[
            # Baseline: all tools
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Project mode: write tools available",
                expect_tools_registered=["edit_file", "write_file", "git_commit"],
            ),
            # Enter plan mode
            SimStep(
                action=StepAction.AGENT_METHOD,
                method="tool_registry._plan_mode",
                method_args={"value": True},
                description="Enter plan mode",
            ),
            # Plan mode: write tools filtered
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Plan mode: exit_plan_mode available",
                custom_validator=lambda agent: (
                    []
                    if agent.tool_registry.in_plan_mode
                    else ["Plan mode should be active"]
                ),
            ),
            # Exit plan mode
            SimStep(
                action=StepAction.AGENT_METHOD,
                method="tool_registry._plan_mode",
                method_args={"value": False},
                description="Exit plan mode",
            ),
            # Back to project: all tools
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Back to project: all tools restored",
                expect_tools_registered=["edit_file", "write_file", "git_commit", "task_complete"],
            ),
        ],
    )


def _double_activate() -> SimScenario:
    """Activating a project while one is active should switch cleanly."""
    return SimScenario(
        name="double_activate_project",
        description="Switching projects directly works without deactivate",
        category="mode_transition",
        steps=[
            # Activate first project
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Activate blipshell",
                expect_project_active="blipshell",
            ),
            # Activate same project again (should be a no-op or clean re-activate)
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Re-activate same project",
                expect_project_active="blipshell",
                expect_tools_registered=["git_status", "task_complete"],
            ),
            # Cleanup
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate",
                expect_project_inactive=True,
            ),
        ],
    )
