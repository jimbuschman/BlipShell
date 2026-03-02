"""Scenarios: Error recovery and failure handling.

Tests graceful degradation when things go wrong — bad inputs, missing
resources, edge cases in command parsing, recovery after errors.
"""

from blipshell.simulate.models import SimScenario, SimStep, StepAction


def get_scenarios() -> list[SimScenario]:
    return [
        _unknown_slash_command(),
        _activate_nonexistent_project(),
        _double_deactivate(),
        _slash_commands_with_bad_args(),
        _chat_after_failed_project(),
        _rapid_mode_cycling(),
        _empty_and_whitespace_inputs(),
        _project_commands_without_project(),
    ]


def _unknown_slash_command() -> SimScenario:
    """Unknown commands should return an error message, not crash."""
    return SimScenario(
        name="unknown_slash_command",
        description="Unknown slash commands handled gracefully",
        category="error_recovery",
        steps=[
            SimStep(
                action=StepAction.SLASH,
                input="/nonexistent_command",
                description="Unknown command",
                custom_validator=lambda agent: [],  # just verify no crash
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/",
                description="Bare slash",
                custom_validator=lambda agent: [],
            ),
            # Verify agent is still functional after bad commands
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Agent functional after bad slash commands",
                expect_tools_registered=["read_file", "grep_files", "glob_files"],
            ),
        ],
    )


def _activate_nonexistent_project() -> SimScenario:
    """Activating a project that doesn't exist should fail gracefully."""
    return SimScenario(
        name="activate_nonexistent_project",
        description="Non-existent project activation doesn't crash",
        category="error_recovery",
        steps=[
            SimStep(
                action=StepAction.SLASH,
                input="/project this_project_definitely_does_not_exist_xyz",
                description="Activate non-existent project",
                expect_project_inactive=True,
            ),
            # Agent should still be functional
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Agent still functional after failed activation",
                expect_tools_registered=["read_file", "grep_files"],
                expect_tools_not_registered=["git_status", "task_complete"],
            ),
            # Should be able to chat after failed activation
            SimStep(
                action=StepAction.CHAT,
                input="Are you still working?",
                description="Chat works after failed project activation",
                expect_no_error=True,
            ),
        ],
    )


def _double_deactivate() -> SimScenario:
    """Deactivating when no project is active should be a no-op, not crash."""
    return SimScenario(
        name="double_deactivate",
        description="Double deactivation is a safe no-op",
        category="error_recovery",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="No project active",
                expect_project_inactive=True,
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate with no project — should be safe",
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Still no project, still functional",
                expect_project_inactive=True,
                expect_tools_registered=["read_file", "grep_files"],
            ),
            # Double deactivate
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Second deactivate — also safe",
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Tools still correct after double deactivate",
                expect_tools_registered=[
                    "read_file", "write_file", "edit_file",
                    "grep_files", "glob_files", "list_directory",
                ],
                expect_tools_not_registered=["git_status", "task_complete"],
            ),
        ],
    )


def _slash_commands_with_bad_args() -> SimScenario:
    """Slash commands with unexpected arguments shouldn't crash."""
    return SimScenario(
        name="slash_commands_bad_args",
        description="Slash commands with bad arguments are handled gracefully",
        category="error_recovery",
        steps=[
            SimStep(
                action=StepAction.SLASH,
                input="/think maybe",
                description="/think with invalid arg",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/reflect blah",
                description="/reflect with invalid arg",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/approve invalid_option",
                description="/approve with bad arg",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project info",
                description="/project info with no active project",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project delete nonexistent_project_xyz",
                description="/project delete non-existent project",
            ),
            # Still functional after all that
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Agent still functional after bad args",
                expect_tools_registered=["read_file", "grep_files", "glob_files"],
                expect_project_inactive=True,
            ),
        ],
    )


def _chat_after_failed_project() -> SimScenario:
    """Chat should still work correctly after project activation fails."""
    return SimScenario(
        name="chat_after_failed_project",
        description="Chat pipeline survives failed project activation attempt",
        category="error_recovery",
        steps=[
            # Try to activate non-existent project
            SimStep(
                action=StepAction.SLASH,
                input="/project nonexistent_project_xyz",
                description="Failed activation attempt",
                expect_project_inactive=True,
            ),
            # Chat should still work fine
            SimStep(
                action=StepAction.CHAT,
                input="What is Python?",
                description="Simple chat after failed activation",
                expect_no_error=True,
            ),
            # Tool-using chat should also work
            SimStep(
                action=StepAction.CHAT,
                input="Read the file blipshell/__init__.py",
                description="Tool chat after failed activation",
                expect_tools=["read_file"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            # Verify tools are correct
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Tools intact after failed activation + chat",
                expect_tools_registered=[
                    "read_file", "write_file", "grep_files", "glob_files",
                ],
                expect_tools_not_registered=["git_status", "task_complete"],
            ),
        ],
    )


def _rapid_mode_cycling() -> SimScenario:
    """Rapidly cycling between modes shouldn't corrupt tool state."""
    return SimScenario(
        name="rapid_mode_cycling",
        description="Rapid activate/deactivate cycles don't corrupt tool state",
        category="error_recovery",
        steps=[
            # Cycle 1
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Cycle 1: activate",
                expect_project_active="blipshell",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Cycle 1: deactivate",
                expect_project_inactive=True,
            ),
            # Cycle 2
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Cycle 2: activate",
                expect_project_active="blipshell",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Cycle 2: deactivate",
                expect_project_inactive=True,
            ),
            # Cycle 3
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Cycle 3: activate",
                expect_project_active="blipshell",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Cycle 3: deactivate",
                expect_project_inactive=True,
            ),
            # Final check: everything still correct
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Tools correct after 3 rapid activate/deactivate cycles",
                expect_tools_registered=[
                    "read_file", "write_file", "edit_file",
                    "grep_files", "glob_files", "list_directory",
                    "run_command", "search_memories",
                ],
                expect_tools_not_registered=[
                    "git_status", "git_diff", "git_add", "git_commit",
                    "ask_user", "task_complete",
                ],
            ),
        ],
    )


def _empty_and_whitespace_inputs() -> SimScenario:
    """Empty or whitespace-only inputs shouldn't crash slash dispatch."""
    return SimScenario(
        name="empty_whitespace_inputs",
        description="Empty/whitespace inputs handled gracefully",
        category="error_recovery",
        steps=[
            SimStep(
                action=StepAction.SLASH,
                input="/  ",
                description="Slash with only spaces",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project  ",
                description="/project with trailing spaces",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/think  ",
                description="/think with trailing spaces",
            ),
            # Everything should still be fine
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Agent functional after whitespace inputs",
                expect_tools_registered=["read_file", "grep_files"],
            ),
        ],
    )


def _project_commands_without_project() -> SimScenario:
    """Project-specific slash commands without an active project shouldn't crash."""
    return SimScenario(
        name="project_commands_without_project",
        description="Project-specific commands without active project are safe",
        category="error_recovery",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="No project active",
                expect_project_inactive=True,
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project info",
                description="/project info with no project",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/project digest",
                description="/project digest with no project",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/changes",
                description="/changes with no project",
            ),
            # Agent should still be functional
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Still functional after project commands with no project",
                expect_project_inactive=True,
                expect_tools_registered=["read_file", "grep_files", "glob_files"],
            ),
        ],
    )
