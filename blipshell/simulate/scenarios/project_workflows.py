"""Scenarios: Full project workflows.

Tests real project mode workflows: activate, use tools, review code,
check status, verify state, deactivate. These exercise the executor
and project-mode tool registration.
"""

from blipshell.simulate.models import SimScenario, SimStep, StepAction


def get_scenarios() -> list[SimScenario]:
    return [
        _full_project_lifecycle(),
        _project_file_operations(),
        _project_status_commands_all(),
        _project_executor_creates_file(),
        _project_git_tools(),
    ]


def _full_project_lifecycle() -> SimScenario:
    """Full lifecycle: activate, check tools, chat, check status, deactivate, verify cleanup."""
    return SimScenario(
        name="full_project_lifecycle",
        description="Complete project lifecycle: activate, use, verify, deactivate",
        category="project_workflow",
        steps=[
            # Pre-activation state
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="No project active before start",
                expect_project_inactive=True,
                expect_tools_not_registered=["git_status", "task_complete"],
            ),
            # Activate
            SimStep(
                action=StepAction.SLASH,
                input="/project blipshell",
                description="Activate blipshell project",
                expect_project_active="blipshell",
            ),
            # Verify full tool set
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="All project tools registered",
                expect_tools_registered=[
                    "read_file", "write_file", "edit_file", "list_directory",
                    "run_command", "grep_files", "glob_files",
                    "git_status", "git_diff", "git_add", "git_commit",
                    "ask_user", "task_complete",
                    "search_memories", "enter_plan_mode", "exit_plan_mode",
                ],
            ),
            # Use project in chat
            SimStep(
                action=StepAction.CHAT,
                input="What files are in the blipshell/simulate/ directory?",
                description="Chat about project files",
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            # Check status commands
            SimStep(
                action=StepAction.SLASH,
                input="/project info",
                description="Project info",
                expect_project_active="blipshell",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/status",
                description="Status in project mode",
            ),
            SimStep(
                action=StepAction.SLASH,
                input="/changes",
                description="Changes in project mode",
            ),
            # Deactivate
            SimStep(
                action=StepAction.SLASH,
                input="/project off",
                description="Deactivate project",
                expect_project_inactive=True,
            ),
            # Verify cleanup
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Project tools removed after deactivation",
                expect_tools_not_registered=[
                    "git_status", "git_diff", "git_add", "git_commit",
                    "ask_user", "task_complete",
                ],
            ),
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Base tools preserved after deactivation",
                expect_tools_registered=[
                    "read_file", "write_file", "edit_file", "list_directory",
                    "run_command", "grep_files", "glob_files",
                    "search_memories",
                ],
            ),
        ],
    )


def _project_file_operations() -> SimScenario:
    """Use all file-related tools in project mode."""
    return SimScenario(
        name="project_file_operations",
        description="All file tools work in project mode: read, grep, glob, list",
        category="project_workflow",
        requires_project="blipshell",
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="List the Python files in the blipshell/core/tools/ directory using glob.",
                description="Glob files in project",
                expect_tools=["glob_files"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Search for 'class Tool' in the blipshell/core/tools/ directory.",
                description="Grep in project",
                expect_tools=["grep_files"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Read the first 30 lines of blipshell/core/tools/base.py.",
                description="Read file in project",
                expect_tools=["read_file"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="List what's in the blipshell/simulate/scenarios/ directory.",
                description="List directory in project",
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
        ],
    )


def _project_status_commands_all() -> SimScenario:
    """Run ALL status/display commands while project is active — none should crash."""
    return SimScenario(
        name="project_all_status_commands",
        description="Every status command works with active project",
        category="project_workflow",
        requires_project="blipshell",
        steps=[
            SimStep(action=StepAction.SLASH, input="/status", description="/status"),
            SimStep(action=StepAction.SLASH, input="/memory", description="/memory"),
            SimStep(action=StepAction.SLASH, input="/context", description="/context"),
            SimStep(action=StepAction.SLASH, input="/tokens", description="/tokens"),
            SimStep(action=StepAction.SLASH, input="/changes", description="/changes"),
            SimStep(action=StepAction.SLASH, input="/project info", description="/project info"),
            SimStep(action=StepAction.SLASH, input="/core", description="/core"),
            SimStep(action=StepAction.SLASH, input="/flow", description="/flow"),
            SimStep(action=StepAction.SLASH, input="/plans", description="/plans"),
            SimStep(action=StepAction.SLASH, input="/tasks", description="/tasks"),
            SimStep(action=StepAction.SLASH, input="/approve", description="/approve"),
            SimStep(action=StepAction.SLASH, input="/help", description="/help"),
            # Verify project is still active after all those commands
            SimStep(
                action=StepAction.ASSERT_STATE,
                description="Project still active after all status commands",
                expect_project_active="blipshell",
                expect_tools_registered=["git_status", "task_complete"],
            ),
        ],
    )


def _project_executor_creates_file() -> SimScenario:
    """Use executor to create a file, verify it exists, then clean up."""
    return SimScenario(
        name="project_executor_creates_file",
        description="Executor creates a file via force_plan and file exists on disk",
        category="project_workflow",
        requires_project="blipshell",
        cleanup_files=["blipshell/simulate/_test_scratch.py"],
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input=(
                    "Create a file called blipshell/simulate/_test_scratch.py with this content:\n"
                    "# test scratch file\n"
                    "def hello():\n"
                    "    return 'hello from simulation test'\n"
                ),
                description="Executor creates a scratch file",
                force_plan=True,
                expect_tools=["write_file"],
                expect_no_error=True,
                expect_files_exist=["blipshell/simulate/_test_scratch.py"],
                timeout_seconds=180.0,
            ),
            # Verify the file via read
            SimStep(
                action=StepAction.CHAT,
                input="Read the file blipshell/simulate/_test_scratch.py and confirm it contains 'hello from simulation test'.",
                description="Read back the created file",
                expect_tools=["read_file"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
        ],
    )


def _project_git_tools() -> SimScenario:
    """Verify git tools work in project mode (read-only operations)."""
    return SimScenario(
        name="project_git_tools",
        description="Git status and diff tools work in project mode",
        category="project_workflow",
        requires_project="blipshell",
        steps=[
            SimStep(
                action=StepAction.CHAT,
                input="Show me the current git status of the project.",
                description="Trigger git_status",
                expect_tools=["git_status"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
            SimStep(
                action=StepAction.CHAT,
                input="Show me the git diff of any staged or unstaged changes.",
                description="Trigger git_diff",
                expect_tools=["git_diff"],
                expect_no_error=True,
                timeout_seconds=90.0,
            ),
        ],
    )
