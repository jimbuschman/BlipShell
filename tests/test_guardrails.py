"""Tests for the guardrails system.

Tests cover:
- Correction detection (regex patterns)
- GuardrailsEngine state management
- Trajectory injection timing and content
- Completion audit validation
- Confirm plan tool
- Context pinning
- Config defaults and toggling
"""

import asyncio
import pytest

from blipshell.core.guardrails import (
    CORRECTION_PATTERNS,
    GuardrailsEngine,
    detect_correction,
)
from blipshell.models.config import GuardrailsConfig


# ── Correction Detection ────────────────────────────────────────────────────


class TestCorrectionDetection:
    """Test regex patterns for detecting user corrections."""

    @pytest.mark.parametrize("message,should_detect", [
        # Positive: corrections the assistant should catch
        ("I already told you not to do that", True),
        ("that's not what I asked for", True),
        ("no, I said to use pytest not unittest", True),
        ("you missed the third requirement", True),
        ("you forgot to add the import", True),
        ("you ignored what I said about the config", True),
        ("stop doing that", True),
        ("stop adding docstrings", True),
        ("I didn't ask for tests", True),
        ("I didn't ask you to refactor", True),
        ("that's wrong", True),
        ("please actually read the file", True),
        ("you keep making the same mistake", True),
        ("how many times do I have to say this", True),
        ("read what I said earlier", True),
        ("I said dark mode, not light mode", True),
        ("I asked for Python not JavaScript", True),
        ("you didn't follow the instructions", True),
        ("you skipped step 3", True),
        # Negative: normal messages that aren't corrections
        ("can you help me with this?", False),
        ("please add a function to parse JSON", False),
        ("what's the difference between async and sync?", False),
        ("looks good, thanks!", False),
        ("I want to refactor the search module", False),
        ("the tests are passing now", False),
        ("I think we should use Redis for caching", False),
        ("let me know when you're done", False),
        ("I told my boss about the project", False),  # "I told" but not a correction
    ])
    def test_correction_patterns(self, message, should_detect):
        result = detect_correction(message)
        if should_detect:
            assert result is not None, f"Expected correction detection for: {message}"
        else:
            assert result is None, f"False positive correction for: {message}"

    def test_correction_returns_context(self):
        """Detected corrections should return surrounding context."""
        msg = "I already told you not to add docstrings to every function"
        result = detect_correction(msg)
        assert result is not None
        assert "told you" in result

    def test_case_insensitive(self):
        """Patterns should match regardless of case."""
        assert detect_correction("THAT'S NOT RIGHT") is not None
        assert detect_correction("You Missed the config") is not None

    def test_patterns_compile(self):
        """All patterns should be valid compiled regexes."""
        assert len(CORRECTION_PATTERNS) > 0
        for p in CORRECTION_PATTERNS:
            assert hasattr(p, 'search'), f"Pattern not compiled: {p}"


# ── GuardrailsConfig ────────────────────────────────────────────────────────


class TestGuardrailsConfig:
    """Test config model defaults and behavior."""

    def test_defaults(self):
        cfg = GuardrailsConfig()
        assert cfg.enabled is False
        assert cfg.completion_audit is True
        assert cfg.correction_detector is True
        assert cfg.trajectory_monitor is True
        assert cfg.context_pinning is True
        assert cfg.requirement_checklist is True
        assert cfg.monitor_interval == 5
        assert cfg.max_audit_retries == 2

    def test_toggle(self):
        cfg = GuardrailsConfig(enabled=True)
        assert cfg.enabled is True
        cfg.enabled = False
        assert cfg.enabled is False

    def test_selective_disable(self):
        cfg = GuardrailsConfig(
            enabled=True,
            completion_audit=False,
            trajectory_monitor=False,
        )
        assert cfg.enabled is True
        assert cfg.completion_audit is False
        assert cfg.correction_detector is True
        assert cfg.trajectory_monitor is False


# ── GuardrailsEngine ────────────────────────────────────────────────────────


class TestGuardrailsEngine:
    """Test the guardrails engine state management."""

    def _make_engine(self, **overrides):
        cfg = GuardrailsConfig(enabled=True, **overrides)
        return GuardrailsEngine(cfg, router=None)

    def test_initial_state(self):
        engine = self._make_engine()
        assert engine.original_request == ""
        assert engine.checklist == []
        assert engine.postconditions == []
        assert engine.audit_retries == 0

    def test_record_checklist(self):
        engine = self._make_engine()
        engine.record_checklist(
            ["Read config", "Add class", "Wire up"],
            ["Tests pass", "Config loads"],
        )
        assert len(engine.checklist) == 3
        assert len(engine.postconditions) == 2
        assert "Add class" in engine.checklist

    # ── Trajectory Monitor ───────────────────────────────────────────────

    def test_trajectory_injection_not_due(self):
        """No injection when tool_call_count isn't a multiple of interval."""
        engine = self._make_engine(monitor_interval=5)
        engine.original_request = "Fix the bug"
        result = engine.build_trajectory_injection(3, 50, ["read_file", "edit_file", "read_file"])
        assert result is None

    def test_trajectory_injection_at_zero(self):
        """No injection at tool_call_count=0."""
        engine = self._make_engine()
        engine.original_request = "Fix the bug"
        assert engine.build_trajectory_injection(0, 50, []) is None

    def test_trajectory_injection_due(self):
        """Injection fires at the configured interval."""
        engine = self._make_engine(monitor_interval=5)
        engine.original_request = "Add a guardrails toggle to config"
        result = engine.build_trajectory_injection(
            5, 50,
            ["read_file", "read_file", "edit_file", "read_file", "write_file"],
        )
        assert result is not None
        assert "CHECKPOINT" in result
        assert "Add a guardrails toggle" in result
        assert "5/50" in result
        assert "90% budget remaining" in result

    def test_trajectory_injection_includes_checklist(self):
        """If a plan was confirmed, the checklist appears in the injection."""
        engine = self._make_engine(monitor_interval=5)
        engine.original_request = "Build feature X"
        engine.record_checklist(["Step 1", "Step 2"])
        result = engine.build_trajectory_injection(5, 50, ["a", "b", "c", "d", "e"])
        assert "Plan (2 steps)" in result
        assert "Step 1" in result
        assert "Step 2" in result

    def test_trajectory_disabled(self):
        """No injection when trajectory_monitor is off."""
        engine = self._make_engine(trajectory_monitor=False)
        engine.original_request = "Fix the bug"
        assert engine.build_trajectory_injection(5, 50, ["a"] * 5) is None

    # ── Context Pinning ──────────────────────────────────────────────────

    def test_pinned_context(self):
        engine = self._make_engine()
        engine.original_request = "Refactor the search module"
        pinned = engine.pinned_context
        assert "[PINNED]" in pinned
        assert "Refactor the search module" in pinned

    def test_pinned_context_with_checklist(self):
        engine = self._make_engine()
        engine.original_request = "Build X"
        engine.record_checklist(["Read files", "Write code"])
        pinned = engine.pinned_context
        assert "Confirmed plan" in pinned
        assert "Read files" in pinned

    def test_pinned_context_disabled(self):
        engine = self._make_engine(context_pinning=False)
        engine.original_request = "Something"
        assert engine.pinned_context == ""

    # ── Completion Audit ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_audit_disabled(self):
        """When audit is off, always passes."""
        engine = self._make_engine(completion_audit=False)
        engine.original_request = "Test"
        valid, feedback = await engine.validate_completion("Done", "")
        assert valid is True
        assert feedback == ""

    @pytest.mark.asyncio
    async def test_audit_max_retries(self):
        """After max retries, accepts regardless."""
        engine = self._make_engine(max_audit_retries=2)
        engine.original_request = "Test"
        engine.audit_retries = 2
        valid, feedback = await engine.validate_completion("Done", "")
        assert valid is True

    @pytest.mark.asyncio
    async def test_audit_no_router_accepts(self):
        """If router is None (no LLM available), accepts gracefully."""
        engine = self._make_engine()
        engine.original_request = "Test"
        # router is None — the LLM call will fail
        valid, feedback = await engine.validate_completion("Done", "file.py")
        # Should accept on error (graceful degradation)
        assert valid is True


# ── Confirm Plan Tool ────────────────────────────────────────────────────────


class TestConfirmPlanTool:
    """Test the confirm_plan tool."""

    def test_definition(self):
        from blipshell.core.tools.interaction_tools import ConfirmPlanTool
        tool = ConfirmPlanTool()
        defn = tool.definition()
        assert defn.name == "confirm_plan"
        assert len(defn.parameters) == 2

    @pytest.mark.asyncio
    async def test_auto_approve_no_callback(self):
        """Without a callback, auto-approves."""
        from blipshell.core.tools.interaction_tools import ConfirmPlanTool
        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=None)
        tool = ConfirmPlanTool(callback=None, guardrails_engine=engine)
        result = await tool.execute(plan="1. Read\n2. Write\n3. Test")
        assert "approved" in result.lower()
        assert len(engine.checklist) == 3

    @pytest.mark.asyncio
    async def test_stores_checklist_on_engine(self):
        """Approved plan stores steps on the guardrails engine."""
        from blipshell.core.tools.interaction_tools import ConfirmPlanTool
        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=None)
        tool = ConfirmPlanTool(callback=None, guardrails_engine=engine)
        await tool.execute(
            plan="1. Add config model\n2. Create guardrails.py\n3. Wire into executor",
            postconditions="Tests pass\nCLI shows /guardrails command",
        )
        assert "Add config model" in engine.checklist
        assert len(engine.postconditions) == 2

    @pytest.mark.asyncio
    async def test_with_callback_approval(self):
        """User approves via callback."""
        from blipshell.core.tools.interaction_tools import ConfirmPlanTool

        async def mock_callback(question):
            return "yes"

        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=None)
        tool = ConfirmPlanTool(callback=mock_callback, guardrails_engine=engine)
        result = await tool.execute(plan="1. Do thing")
        assert "approved" in result.lower()
        assert len(engine.checklist) == 1

    @pytest.mark.asyncio
    async def test_with_callback_rejection(self):
        """User rejects via callback."""
        from blipshell.core.tools.interaction_tools import ConfirmPlanTool

        async def mock_callback(question):
            return "reject: I want a different approach"

        tool = ConfirmPlanTool(callback=mock_callback)
        result = await tool.execute(plan="1. Do thing")
        assert "rejected" in result.lower()

    @pytest.mark.asyncio
    async def test_with_callback_modification(self):
        """User modifies the plan via callback."""
        from blipshell.core.tools.interaction_tools import ConfirmPlanTool

        async def mock_callback(question):
            return "add step 4: run the tests"

        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=None)
        tool = ConfirmPlanTool(callback=mock_callback, guardrails_engine=engine)
        result = await tool.execute(plan="1. Read\n2. Edit\n3. Write")
        assert "feedback" in result.lower()
        # Original checklist still stored (user can modify verbally)
        assert len(engine.checklist) == 3

    @pytest.mark.asyncio
    async def test_missing_plan_returns_error(self):
        from blipshell.core.tools.interaction_tools import ConfirmPlanTool
        tool = ConfirmPlanTool()
        result = await tool.execute()
        assert "error" in result.lower()


# ── Prompt: validate_task_completion ─────────────────────────────────────────


class TestCompletionAuditPrompt:
    """Test the completion audit prompt generation."""

    def test_basic_prompt(self):
        from blipshell.llm.prompts import validate_task_completion
        system, user = validate_task_completion(
            original_request="Add a /guardrails CLI command",
            summary="Added /guardrails toggle command to cli.py",
            files_modified="blipshell/ui/cli.py",
        )
        assert "PASS" in system
        assert "FAIL" in system
        assert "/guardrails" in user
        assert "cli.py" in user

    def test_prompt_with_checklist(self):
        from blipshell.llm.prompts import validate_task_completion
        system, user = validate_task_completion(
            original_request="Build guardrails",
            summary="Done",
            checklist=["Add config", "Create engine", "Wire tests"],
        )
        assert "Confirmed plan" in user
        assert "Add config" in user
        assert "Create engine" in user

    def test_prompt_without_files(self):
        from blipshell.llm.prompts import validate_task_completion
        _, user = validate_task_completion(
            original_request="Explain how X works",
            summary="X works by...",
        )
        assert "Files modified" not in user


# ── Integration: LoopConfig guardrails field ─────────────────────────────────


class TestLoopConfigGuardrails:
    """Test that LoopConfig accepts and passes guardrails."""

    def test_default_none(self):
        from blipshell.core.chat_loop import LoopConfig
        config = LoopConfig()
        assert config.guardrails is None

    def test_with_engine(self):
        from blipshell.core.chat_loop import LoopConfig
        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=None)
        config = LoopConfig(guardrails=engine)
        assert config.guardrails is engine
