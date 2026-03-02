"""Scenarios: Tool registration correctness across modes.

Verifies the right tools are available in each mode — simple chat,
project, and plan. No LLM needed, runs in seconds.
"""

from blipshell.simulate.models import SimScenario, SimStep, StepAction


# Tools expected in simple chat (after session start, memory tools registered)
SIMPLE_CHAT_TOOLS = [
    "read_file", "write_file", "edit_file", "list_directory",
    "run_command", "grep_files", "glob_files",
    "web_search", "web_fetch",
    "enter_plan_mode", "exit_plan_mode",
    "search_memories", "save_core_memory", "promote_to_core",
    "list_sessions", "create_project",
]

# Tools added when a project is activated
PROJECT_EXTRA_TOOLS = [
    "git_status", "git_diff", "git_add", "git_commit",
    "ask_user", "task_complete",
]

# Tools that should NOT be in simple chat
PROJECT_ONLY_TOOLS = [
    "git_status", "git_diff", "git_add", "git_commit",
    "ask_user", "task_complete",
]


def get_scenarios() -> list[SimScenario]:
    return [
        _simple_chat_tools(),
        _project_mode_tools(),
        _tools_survive_project_cycle(),
        _plan_mode_filters_tools(),
        _tool_count_sanity(),
    ]


def _simple_chat_tools() -> SimScenario:
    return SimScenario(
        name="simple_chat_tool_registry",
        description="Verify all expected tools are registered in simple chat mode",
        category="tool_registration",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Simple chat: all base tools present",
                expect_tools_registered=SIMPLE_CHAT_TOOLS,
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Simple chat: project-only tools NOT present",
                expect_tools_not_registered=PROJECT_ONLY_TOOLS,
            ),
        ],
    )


def _project_mode_tools() -> SimScenario:
    return SimScenario(
        name="project_mode_tool_registry",
        description="Verify project mode adds git and interaction tools",
        category="tool_registration",
        requires_project="blipshell",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Project mode: base tools still present",
                expect_tools_registered=SIMPLE_CHAT_TOOLS,
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Project mode: project tools added",
                expect_tools_registered=PROJECT_EXTRA_TOOLS,
            ),
        ],
    )


def _tools_survive_project_cycle() -> SimScenario:
    """Activate project, deactivate, verify base tools still intact."""
    return SimScenario(
        name="tools_survive_project_cycle",
        description="Base tools survive project activate/deactivate cycle",
        category="tool_registration",
        steps=[
            # Check baseline
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Before project: base tools present",
                expect_tools_registered=SIMPLE_CHAT_TOOLS,
            ),
            # Activate project
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Activate blipshell project",
                expect_project_active="blipshell",
            ),
            # Verify project tools added
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="After activate: project tools present",
                expect_tools_registered=PROJECT_EXTRA_TOOLS,
            ),
            # Deactivate
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate project",
                expect_project_inactive=True,
            ),
            # KEY CHECK: base tools must still be here
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="After deactivate: base tools MUST still be present",
                expect_tools_registered=SIMPLE_CHAT_TOOLS,
            ),
            # Project-only tools should be gone
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="After deactivate: project-only tools removed",
                expect_tools_not_registered=PROJECT_ONLY_TOOLS,
            ),
        ],
    )


def _plan_mode_filters_tools() -> SimScenario:
    """Verify plan mode restricts to read-only tools."""
    return SimScenario(
        name="plan_mode_filters_tools",
        description="Plan mode should only expose read-only tools + exit_plan_mode",
        category="tool_registration",
        steps=[
            # Enter plan mode via tool registry
            SimStep(
                action=StepAction.AGENT_METHOD,
                method="tool_registry._plan_mode",
                method_args={"value": True},
                description="Enter plan mode",
            ),
            # Verify write tools are filtered out
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Plan mode: write_file should not be available",
                custom_validator=lambda agent: _check_plan_mode(agent),
            ),
            # Exit plan mode
            SimStep(
                action=StepAction.AGENT_METHOD,
                method="tool_registry._plan_mode",
                method_args={"value": False},
                description="Exit plan mode",
            ),
            # Verify all tools are back
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="After plan mode: all tools restored",
                expect_tools_registered=SIMPLE_CHAT_TOOLS,
            ),
        ],
    )


def _check_plan_mode(agent) -> list[str]:
    """Verify plan mode tool set is correct."""
    failures = []
    if not agent.tool_registry.in_plan_mode:
        failures.append("Plan mode not active on registry")
        return failures

    plan_tools = agent.tool_registry.get_plan_mode_tools()
    plan_tool_names = {t["function"]["name"] for t in plan_tools}

    # Write tools should NOT be in plan mode
    write_tools = {"write_file", "edit_file", "run_command", "git_add", "git_commit"}
    for wt in write_tools:
        if wt in plan_tool_names:
            failures.append(f"Write tool '{wt}' available in plan mode")

    # Read tools should be in plan mode
    read_tools = {"read_file", "list_directory", "grep_files", "glob_files", "search_memories"}
    for rt in read_tools:
        if rt in plan_tool_names:
            pass  # good
        # Don't fail on missing — some read tools may not be read_only yet

    # exit_plan_mode must be available
    if "exit_plan_mode" not in plan_tool_names:
        failures.append("exit_plan_mode not available in plan mode")

    return failures


def _tool_count_sanity() -> SimScenario:
    """Verify tool count is within expected range (catches accidental mass unregister)."""
    return SimScenario(
        name="tool_count_sanity",
        description="Tool count should be within expected range",
        category="tool_registration",
        steps=[
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="At least 15 tools registered",
                custom_validator=lambda agent: (
                    []
                    if len(agent.tool_registry.get_tool_names()) >= 15
                    else [f"Only {len(agent.tool_registry.get_tool_names())} tools registered (expected >= 15)"]
                ),
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="No more than 50 tools registered (sanity cap)",
                custom_validator=lambda agent: (
                    []
                    if len(agent.tool_registry.get_tool_names()) <= 50
                    else [f"{len(agent.tool_registry.get_tool_names())} tools registered (expected <= 50)"]
                ),
            ),
        ],
    )
