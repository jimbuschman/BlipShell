"""Scenarios: Every slash command exercised.

Verifies no command crashes and state mutations work correctly.
No LLM needed.
"""

from blipshell.simulate.models import SimScenario, SimStep, StepAction


def get_scenarios() -> list[SimScenario]:
    return [
        _all_display_commands(),
        _state_mutation_commands(),
        _project_subcommands(),
    ]


def _all_display_commands() -> SimScenario:
    """Run every display/read-only slash command — none should crash."""
    return SimScenario(
        name="all_display_slash_commands",
        description="Every display slash command runs without crashing",
        category="slash_commands",
        steps=[
            SimStep(action=StepAction.SLASH, input="/status", description="/status"),
            SimStep(action=StepAction.SLASH, input="/memory", description="/memory"),
            SimStep(action=StepAction.SLASH, input="/context", description="/context"),
            SimStep(action=StepAction.SLASH, input="/tokens", description="/tokens"),
            SimStep(action=StepAction.SLASH, input="/core", description="/core"),
            SimStep(action=StepAction.SLASH, input="/flow", description="/flow"),
            SimStep(action=StepAction.SLASH, input="/changes", description="/changes"),
            SimStep(action=StepAction.SLASH, input="/projects", description="/projects"),
            SimStep(action=StepAction.SLASH, input="/plan", description="/plan"),
            SimStep(action=StepAction.SLASH, input="/plans", description="/plans"),
            SimStep(action=StepAction.SLASH, input="/tasks", description="/tasks"),
            SimStep(action=StepAction.SLASH, input="/approve", description="/approve"),
            SimStep(action=StepAction.SLASH, input="/help", description="/help"),
        ],
    )


def _state_mutation_commands() -> SimScenario:
    """Test commands that change agent state."""
    return SimScenario(
        name="state_mutation_slash_commands",
        description="Slash commands that mutate state work correctly",
        category="slash_commands",
        steps=[
            # Think toggle
            SimStep(
                action=StepAction.SLASH, input="/think on",
                description="Think on",
                expect_think_enabled=True,
            ),
            SimStep(
                action=StepAction.SLASH, input="/think off",
                description="Think off",
                expect_think_enabled=False,
            ),
            SimStep(
                action=StepAction.SLASH, input="/think",
                description="Think toggle (should flip to on)",
                expect_think_enabled=True,
            ),
            SimStep(
                action=StepAction.SLASH, input="/think",
                description="Think toggle (should flip back to off)",
                expect_think_enabled=False,
            ),
            # Reflect toggle
            SimStep(
                action=StepAction.SLASH, input="/reflect on",
                description="Reflect on",
                expect_reflect_enabled=True,
            ),
            SimStep(
                action=StepAction.SLASH, input="/reflect off",
                description="Reflect off",
                expect_reflect_enabled=False,
            ),
            # Approve
            SimStep(
                action=StepAction.SLASH, input="/approve all",
                description="Approve all tools",
            ),
            SimStep(
                action=StepAction.SLASH, input="/approve reset",
                description="Reset approvals",
            ),
            # Save (just verify it doesn't crash)
            SimStep(
                action=StepAction.SLASH, input="/save",
                description="Save session to memory",
            ),
        ],
    )


def _project_subcommands() -> SimScenario:
    """Test /project subcommands in sequence."""
    return SimScenario(
        name="project_subcommands",
        description="Project activation, info, deactivation via slash commands",
        category="slash_commands",
        steps=[
            # No project initially
            SimStep(
                action=StepAction.SLASH, input="/project",
                description="/project with no active project",
                expect_project_inactive=True,
            ),
            # List projects
            SimStep(
                action=StepAction.SLASH, input="/projects",
                description="List all projects",
            ),
            # Activate
            SimStep(
                action=StepAction.SLASH, input="/project blipshell",
                description="Activate blipshell project",
                expect_project_active="blipshell",
            ),
            # Info
            SimStep(
                action=StepAction.SLASH, input="/project info",
                description="Show project info",
                expect_project_active="blipshell",
            ),
            # Deactivate
            SimStep(
                action=StepAction.SLASH, input="/project off",
                description="Deactivate project",
                expect_project_inactive=True,
            ),
            # Verify deactivation stuck
            SimStep(
                action=StepAction.SLASH, input="/project",
                description="Confirm no active project",
                expect_project_inactive=True,
            ),
        ],
    )
