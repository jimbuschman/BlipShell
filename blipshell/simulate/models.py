"""Data classes for the simulation system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ResultStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"  # soft assertion failed (LLM-dependent content checks)
    FAIL = "fail"


class StepAction(str, Enum):
    CHAT = "chat"
    SLASH = "slash"
    AGENT_METHOD = "agent_method"
    ASSERT_STATE = "assert_state"
    WAIT = "wait"


@dataclass
class SimStep:
    """A single step in a simulation scenario."""

    action: StepAction
    description: str = ""

    # Input for chat/slash actions
    input: str = ""

    # For agent_method actions
    method: str = ""
    method_args: dict[str, Any] = field(default_factory=dict)

    # Chat options
    force_plan: bool = False

    # Wait options
    wait_seconds: float = 1.0

    # --- Expectations (only checked if set) ---

    # Tool call expectations
    expect_tools: list[str] | None = None
    expect_no_tools: bool = False
    expect_max_tool_calls: int | None = None

    # Response content expectations (soft — LLM-dependent)
    expect_response_contains: list[str] | None = None
    expect_response_not_contains: list[str] | None = None
    expect_no_error: bool = True

    # State expectations (hard — deterministic)
    expect_tools_registered: list[str] | None = None
    expect_tools_not_registered: list[str] | None = None
    expect_project_active: str | None = None
    expect_project_inactive: bool = False
    expect_think_enabled: bool | None = None
    expect_reflect_enabled: bool | None = None

    # File expectations
    expect_files_exist: list[str] | None = None
    expect_files_not_exist: list[str] | None = None

    # Custom validator: (SimContext) -> list[str] (failure messages, empty = pass)
    custom_validator: Optional[Callable] = None

    # Timeout
    timeout_seconds: float = 120.0


@dataclass
class SimScenario:
    """A complete simulation scenario — a sequence of steps mimicking a user."""

    name: str
    description: str
    category: str
    steps: list[SimStep]

    # Setup
    requires_project: str | None = None
    requires_project_path: str | None = None
    fresh_session: bool = True

    # Teardown
    cleanup_files: list[str] = field(default_factory=list)


@dataclass
class SimStepResult:
    """Result of executing one step."""

    step_index: int
    description: str
    action: StepAction
    status: ResultStatus
    response: str = ""
    tools_called: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    hard_failures: list[str] = field(default_factory=list)
    soft_failures: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class SimScenarioResult:
    """Result of running a full scenario."""

    name: str
    category: str
    status: ResultStatus
    step_results: list[SimStepResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None  # scenario-level error (bootstrap failure, etc.)


@dataclass
class SimSuiteResult:
    """Result of running the full simulation suite."""

    scenario_results: list[SimScenarioResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.scenario_results if r.status == ResultStatus.PASS)

    @property
    def warned(self) -> int:
        return sum(1 for r in self.scenario_results if r.status == ResultStatus.WARN)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.scenario_results if r.status == ResultStatus.FAIL)

    @property
    def total(self) -> int:
        return len(self.scenario_results)


@dataclass
class SlashResult:
    """Result of executing a slash command programmatically."""

    command: str
    output: str
    success: bool
    error: str | None = None
