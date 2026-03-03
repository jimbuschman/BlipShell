"""Simulation runner — bootstraps Agent, runs scenarios, collects results."""

from __future__ import annotations

import difflib
import logging
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from blipshell.core.agent import Agent
from blipshell.core.config import BlipShellConfig, ConfigManager
from blipshell.simulate.models import (
    ResultStatus,
    SimScenario,
    SimScenarioResult,
    SimStepResult,
    SimSuiteResult,
)
from blipshell.simulate.slash_dispatcher import SlashCommandDispatcher
from blipshell.simulate.step_executor import SimStepExecutor

logger = logging.getLogger(__name__)


class SimContext:
    """Mutable state passed between steps in a scenario."""

    def __init__(
        self,
        agent: Agent,
        config: BlipShellConfig,
        config_manager: ConfigManager,
        slash_dispatcher: SlashCommandDispatcher,
    ):
        self.agent = agent
        self.config = config
        self.config_manager = config_manager
        self.slash_dispatcher = slash_dispatcher

        # Accumulated results
        self.step_results: list[SimStepResult] = []
        self.responses: list[str] = []
        self.all_tool_calls: list[dict] = []

    @property
    def last_response(self) -> str:
        return self.responses[-1] if self.responses else ""


class SimRunner:
    """Runs simulation scenarios against a real Agent."""

    def __init__(
        self,
        config_path: str | None = None,
        quiet: bool = False,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.config_path = config_path
        self.quiet = quiet
        self.on_status = on_status or (lambda msg: None)
        self._executor = SimStepExecutor()

    async def run_suite(
        self,
        scenarios: list[SimScenario],
    ) -> SimSuiteResult:
        """Run a list of scenarios and return aggregated results."""
        suite_result = SimSuiteResult()
        t0 = time.monotonic()

        # Pre-flight: validate scenario tool names against actual registry
        preflight_errors = await self._preflight_validate(scenarios)
        if preflight_errors:
            for err in preflight_errors:
                self.on_status(f"  PREFLIGHT ERROR: {err}")
            # Abort — these are scenario definition bugs, not LLM issues
            suite_result.elapsed_seconds = round(time.monotonic() - t0, 2)
            # Add a synthetic failed result so the report clearly shows what happened
            suite_result.scenario_results.append(SimScenarioResult(
                name="__preflight_validation__",
                category="preflight",
                status=ResultStatus.FAIL,
                error=f"Scenario definition errors ({len(preflight_errors)}): "
                      + "; ".join(preflight_errors),
            ))
            return suite_result

        for scenario in scenarios:
            self.on_status(f"Running: {scenario.name}")
            result = await self.run_scenario(scenario)
            suite_result.scenario_results.append(result)

            # Log progress
            status_str = result.status.value.upper()
            self.on_status(f"  {status_str}: {scenario.name} ({result.elapsed_seconds:.1f}s)")

        suite_result.elapsed_seconds = round(time.monotonic() - t0, 2)
        return suite_result

    async def run_scenario(
        self,
        scenario: SimScenario,
    ) -> SimScenarioResult:
        """Run a single scenario end-to-end."""
        t0 = time.monotonic()
        result = SimScenarioResult(
            name=scenario.name,
            category=scenario.category,
            status=ResultStatus.PASS,
        )

        # Bootstrap agent
        try:
            agent, config, config_manager = await self._bootstrap_agent()
        except Exception as e:
            result.status = ResultStatus.FAIL
            result.error = f"Bootstrap failed: {type(e).__name__}: {e}"
            result.elapsed_seconds = round(time.monotonic() - t0, 2)
            return result

        slash_dispatcher = SlashCommandDispatcher(agent, config)
        ctx = SimContext(agent, config, config_manager, slash_dispatcher)

        try:
            # Start session
            if scenario.fresh_session:
                await agent.start_session()

            # Setup: activate project if needed
            if scenario.requires_project:
                try:
                    await agent.activate_project(scenario.requires_project)
                except KeyError:
                    # Project doesn't exist — create it if path provided
                    if scenario.requires_project_path:
                        await agent.sqlite.create_project(
                            name=scenario.requires_project,
                            root_path=scenario.requires_project_path,
                            language="Python",
                        )
                        await agent.activate_project(scenario.requires_project)
                    else:
                        result.status = ResultStatus.FAIL
                        result.error = f"Project '{scenario.requires_project}' not found and no path provided"
                        return result

            # Run steps
            for i, step in enumerate(scenario.steps):
                step_result = await self._executor.execute(step, i, ctx)
                result.step_results.append(step_result)
                ctx.step_results.append(step_result)

                # Stop on hard failure
                if step_result.status == ResultStatus.FAIL:
                    result.status = ResultStatus.FAIL
                    break

            # Determine overall scenario status
            if result.status != ResultStatus.FAIL:
                has_warns = any(
                    sr.status == ResultStatus.WARN for sr in result.step_results
                )
                result.status = ResultStatus.WARN if has_warns else ResultStatus.PASS

        except Exception as e:
            result.status = ResultStatus.FAIL
            result.error = f"Scenario error: {type(e).__name__}: {e}"
            logger.exception("Scenario '%s' failed", scenario.name)

        finally:
            # Cleanup
            await self._cleanup(agent, scenario)
            result.elapsed_seconds = round(time.monotonic() - t0, 2)

        return result

    async def _preflight_validate(
        self,
        scenarios: list[SimScenario],
    ) -> list[str]:
        """Validate all scenario tool name expectations against the actual registry.

        Bootstraps a throwaway agent, collects all expected tool names from all
        scenarios, and checks them against what's actually registered. Returns
        a list of error messages (empty = all good).
        """
        errors: list[str] = []
        self.on_status("Pre-flight: validating scenario definitions...")

        try:
            agent, config, _ = await self._bootstrap_agent()
            await agent.start_session()
        except Exception as e:
            errors.append(f"Could not bootstrap agent for pre-flight: {e}")
            return errors

        registered = set(agent.tool_registry.get_tool_names())

        # Also get project-mode tools by temporarily activating a project
        project_tools: set[str] = set()
        try:
            # Try to find any existing project for validation
            projects = await agent.sqlite.list_projects()
            if projects:
                await agent.activate_project(projects[0]["name"])
                project_tools = set(agent.tool_registry.get_tool_names())
                await agent.deactivate_project()
        except Exception:
            pass  # No projects available — can only validate base tools

        all_known = registered | project_tools

        # Collect all expected tool names from all scenarios
        for scenario in scenarios:
            for step in scenario.steps:
                names_to_check: list[str] = []
                if step.expect_tools_registered:
                    names_to_check.extend(step.expect_tools_registered)
                if step.expect_tools:
                    names_to_check.extend(step.expect_tools)

                for name in names_to_check:
                    if name not in all_known:
                        # Find close matches
                        close = difflib.get_close_matches(name, all_known, n=3, cutoff=0.6)
                        suggestion = f" (did you mean: {', '.join(close)}?)" if close else ""
                        errors.append(
                            f"Scenario '{scenario.name}': unknown tool '{name}'{suggestion}"
                        )

        # Cleanup
        try:
            await agent.end_session()
        except Exception:
            pass

        if not errors:
            self.on_status(f"  Pre-flight OK: {len(all_known)} tools validated")
        return errors

    async def _bootstrap_agent(self) -> tuple[Agent, BlipShellConfig, ConfigManager]:
        """Bootstrap a real Agent instance (same pattern as test_executor.py)."""
        config_manager = ConfigManager(self.config_path)
        config = config_manager.load()
        agent = Agent(config, config_manager)

        def _on_status(msg: str):
            if not self.quiet:
                self.on_status(f"  [init] {msg}")

        await agent.initialize(on_status=_on_status)

        # Wire a headless ask_user callback (auto-respond during simulation)
        async def _headless_ask_user(question: str) -> str:
            return "Make your best judgment"

        agent.set_ask_user_callback(_headless_ask_user)

        # Auto-approve all tools in simulation (no interactive prompts)
        if config.agent.tools_requiring_approval:
            async def _auto_approve(tool_name: str, arguments: dict, force: bool = False) -> bool:
                return True
            agent.tool_registry.set_approval_callback(
                callback=_auto_approve,
                tools_requiring_approval=set(config.agent.tools_requiring_approval),
            )

        return agent, config, config_manager

    async def _cleanup(self, agent: Agent, scenario: SimScenario):
        """Clean up after a scenario."""
        # Deactivate project if active
        if agent.active_project:
            try:
                await agent.deactivate_project()
            except Exception:
                pass

        # End session
        try:
            if agent.session_manager and agent.session_manager.session_id:
                await agent.end_session()
        except Exception:
            pass

        # Clean up files created during scenario
        for fpath in scenario.cleanup_files:
            p = Path(fpath)
            try:
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                pass
