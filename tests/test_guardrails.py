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


class _RecordingRouter:
    """Stub router that counts generate() calls (to assert LLM-call avoidance)."""

    def __init__(self, result: str = "PASS"):
        self.calls = 0
        self._result = result

    async def generate(self, *args, **kwargs):
        self.calls += 1
        return self._result


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
        assert cfg.enabled is True
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
        """If router is None (no LLM available), accepts gracefully — for a
        non-trivial task that actually reaches the LLM audit."""
        engine = self._make_engine()
        engine.original_request = "Test"
        # Non-trivial (high tool count) + a real edit, so it clears the
        # deterministic + difficulty gates and reaches the (None) router.
        valid, feedback = await engine.validate_completion(
            "Done", "file.py", tool_call_names=["edit_file"], tool_call_count=8,
        )
        # Should accept on error (graceful degradation)
        assert valid is True

    # ── Deterministic completion gate (zero LLM) ─────────────────────────

    def test_det_empty_summary_rejects(self):
        engine = self._make_engine()
        assert engine._deterministic_completion_check("", "", ["edit_file"]) is not None
        assert engine._deterministic_completion_check("   ", "", ["edit_file"]) is not None

    def test_det_files_claimed_no_mutator_rejects(self):
        engine = self._make_engine()
        msg = engine._deterministic_completion_check("Done", "a.py", ["read_file", "grep_files"])
        assert msg is not None and "no edits" in msg

    def test_det_files_claimed_with_edit_passes(self):
        engine = self._make_engine()
        assert engine._deterministic_completion_check("Done", "a.py", ["edit_file"]) is None

    def test_det_no_files_claimed_passes(self):
        # Q&A task: no files modified, summary present → nothing to flag.
        engine = self._make_engine()
        assert engine._deterministic_completion_check("Here's the answer", "", ["read_file"]) is None

    # ── Difficulty gate ──────────────────────────────────────────────────

    def test_difficulty_checklist_triggers_audit(self):
        engine = self._make_engine()
        engine.record_checklist(["step 1", "step 2"])
        assert engine._completion_needs_llm_audit("", ["read_file"], 1) is True

    def test_difficulty_high_tool_count_triggers_audit(self):
        engine = self._make_engine(completion_audit_min_tool_calls=5)
        assert engine._completion_needs_llm_audit("", ["read_file"], 6) is True

    def test_difficulty_multifile_triggers_audit(self):
        engine = self._make_engine()
        assert engine._completion_needs_llm_audit("a.py, b.py", ["edit_file"], 1) is True

    def test_difficulty_trivial_skips_audit(self):
        engine = self._make_engine(completion_audit_min_tool_calls=5)
        assert engine._completion_needs_llm_audit("a.py", ["edit_file"], 2) is False

    # ── validate_completion routing (LLM call only when warranted) ───────

    @pytest.mark.asyncio
    async def test_trivial_clean_task_skips_llm(self):
        """The core win: a trivial, deterministic-clean completion accepts with
        NO LLM call."""
        router = _RecordingRouter(result="PASS")
        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=router)
        engine.original_request = "rename a var"
        valid, _ = await engine.validate_completion(
            "Renamed the variable", "a.py", tool_call_names=["edit_file"], tool_call_count=2,
        )
        assert valid is True
        assert router.calls == 0  # difficulty gate skipped the LLM

    @pytest.mark.asyncio
    async def test_nontrivial_task_calls_llm_once(self):
        router = _RecordingRouter(result="PASS")
        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=router)
        engine.original_request = "implement feature"
        valid, _ = await engine.validate_completion(
            "Implemented it", "a.py", tool_call_names=["edit_file"], tool_call_count=9,
        )
        assert valid is True
        assert router.calls == 1

    @pytest.mark.asyncio
    async def test_deterministic_reject_skips_llm(self):
        router = _RecordingRouter(result="PASS")
        engine = GuardrailsEngine(GuardrailsConfig(enabled=True), router=router)
        engine.original_request = "do the thing"
        valid, feedback = await engine.validate_completion(
            "", "", tool_call_names=["edit_file"], tool_call_count=9,
        )
        assert valid is False
        assert router.calls == 0  # rejected before any LLM call
        assert "No summary" in feedback


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


# ── Critique Provider Config ─────────────────────────────────────────────────


class TestStaleFileDetection:
    """Test that ReadFileTool allows re-reading stale files."""

    @pytest.fixture
    def read_tool(self, tmp_path):
        """Create a ReadFileTool with tracking sets."""
        from blipshell.core.tools.filesystem import ReadFileTool
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")
        files_read = set()
        file_cache = {}
        stale_files = set()
        tool = ReadFileTool(
            root_path=str(tmp_path),
            files_read=files_read,
            file_cache=file_cache,
            stale_files=stale_files,
        )
        return tool, str(test_file), files_read, file_cache, stale_files

    def test_normal_reread_blocked(self, read_tool):
        """Re-reads should be blocked when file is not stale."""
        tool, path, files_read, file_cache, stale_files = read_tool
        files_read.add(path)
        result = asyncio.run(
            tool.execute(path)
        )
        assert "ALREADY READ" in result

    def test_stale_file_reread_allowed(self, read_tool):
        """Stale files should be re-read from disk."""
        tool, path, files_read, file_cache, stale_files = read_tool
        files_read.add(path)
        stale_files.add(path)
        result = asyncio.run(
            tool.execute(path)
        )
        assert "ALREADY READ" not in result
        assert "line 1" in result

    def test_stale_bypasses_cache(self, read_tool):
        """Stale files should read from disk, not cache."""
        tool, path, files_read, file_cache, stale_files = read_tool
        files_read.add(path)
        file_cache[path] = "old cached content"
        stale_files.add(path)
        result = asyncio.run(
            tool.execute(path)
        )
        assert "old cached content" not in result
        assert "line 1" in result

    def test_non_stale_uses_cache(self, read_tool):
        """Non-stale files should still return cached content."""
        tool, path, files_read, file_cache, stale_files = read_tool
        files_read.add(path)
        file_cache[path] = "cached content here"
        result = asyncio.run(
            tool.execute(path)
        )
        assert "cached content" in result


# ── Per-Tool-Type Compaction ─────────────────────────────────────────────────


class TestPerToolTypeCompaction:
    """Test that compact_messages uses different strategies per tool type."""

    def _make_messages(self, tool_calls_and_results):
        """Build a message list with system, user, then tool call/result pairs.

        tool_calls_and_results: list of (tool_name, result_content) tuples
        """
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Do the task"},
        ]
        for i, (name, content) in enumerate(tool_calls_and_results):
            tc_id = f"tc_{i}"
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": tc_id,
                    "function": {"name": name, "arguments": "{}"},
                }],
            })
            messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tc_id,
            })
        return messages

    def test_read_file_dropped(self):
        from blipshell.core.chat_loop import compact_messages
        file_content = "\n".join(f"    {i}\tcode line {i}" for i in range(1, 101))
        msgs = self._make_messages([
            ("read_file", file_content),
            ("read_file", "short file"),
            ("edit_file", "OK — edited"),
            ("read_file", "another file"),
            ("edit_file", "OK — edited again"),
            ("read_file", "final read"),
        ])
        result = compact_messages(msgs, keep_last_n=2)
        # Find compacted read_file results (should say "re-read from disk")
        compacted_reads = [
            m for m in result
            if m.get("role") == "tool" and "re-read from disk" in m.get("content", "")
        ]
        assert len(compacted_reads) >= 1  # At least the big file got compacted

    def test_grep_collapsed_to_files(self):
        from blipshell.core.chat_loop import compact_messages
        grep_output = "src/foo.py:10:match1\nsrc/bar.py:20:match2\nsrc/foo.py:30:match3"
        msgs = self._make_messages([
            ("grep_files", grep_output),
            ("read_file", "content"),
            ("edit_file", "OK"),
            ("read_file", "more content"),
            ("edit_file", "OK again"),
            ("read_file", "final"),
        ])
        result = compact_messages(msgs, keep_last_n=2)
        compacted_greps = [
            m for m in result
            if m.get("role") == "tool" and "grep_files" in m.get("content", "")
        ]
        assert len(compacted_greps) >= 1
        assert "2 files" in compacted_greps[0]["content"] or "src/foo.py" in compacted_greps[0]["content"]

    def test_shell_keeps_first_last(self):
        from blipshell.core.chat_loop import compact_messages
        shell_output = "\n".join(f"output line {i}" for i in range(50))
        msgs = self._make_messages([
            ("run_command", shell_output),
            ("read_file", "content"),
            ("edit_file", "OK"),
            ("read_file", "more"),
            ("edit_file", "OK 2"),
            ("read_file", "final"),
        ])
        result = compact_messages(msgs, keep_last_n=2)
        compacted_shells = [
            m for m in result
            if m.get("role") == "tool" and "run_command" in m.get("content", "")
        ]
        assert len(compacted_shells) >= 1
        content = compacted_shells[0]["content"]
        assert "output line 0" in content  # first line preserved
        assert "output line 49" in content  # last line preserved
        assert "50 lines" in content

    def test_recent_results_kept_intact(self):
        from blipshell.core.chat_loop import compact_messages
        msgs = self._make_messages([
            ("read_file", "old content " * 100),
            ("read_file", "recent content"),
            ("edit_file", "OK"),
        ])
        result = compact_messages(msgs, keep_last_n=2)
        # Last 2 tool results should be untouched
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert any("[Compacted]" not in m["content"] for m in tool_msgs[-2:])


# ── Doom-Loop Detector ───────────────────────────────────────────────────────


class TestDoomLoopDetector:
    """Test the counter-based doom-loop detection."""

    def _make_engine(self, **overrides):
        cfg = GuardrailsConfig(enabled=True, **overrides)
        return GuardrailsEngine(cfg, router=None)

    def test_disabled_returns_none(self):
        engine = self._make_engine(doom_loop_detector=False)
        result = engine.check_doom_loop([("read_file", {"path": "foo.py"})] * 5)
        assert result is None

    def test_no_warning_below_threshold(self):
        engine = self._make_engine(doom_loop_read_threshold=3)
        engine.original_request = "Fix the bug"
        # 2 reads of same file — below threshold
        result = engine.check_doom_loop([
            ("read_file", {"path": "foo.py"}),
            ("read_file", {"path": "foo.py"}),
        ])
        assert result is None

    def test_read_warning_at_threshold(self):
        engine = self._make_engine(doom_loop_read_threshold=3)
        engine.original_request = "Fix the bug"
        # 3 reads of same file — hits threshold
        result = engine.check_doom_loop([
            ("read_file", {"path": "foo.py"}),
            ("read_file", {"path": "foo.py"}),
            ("read_file", {"path": "foo.py"}),
        ])
        assert result is not None
        assert "DOOM-LOOP" in result
        assert "foo.py" in result
        assert "3 times" in result

    def test_read_warning_across_batches(self):
        engine = self._make_engine(doom_loop_read_threshold=3)
        # Spread across multiple batches
        engine.check_doom_loop([("read_file", {"path": "foo.py"})])
        engine.check_doom_loop([("read_file", {"path": "foo.py"})])
        result = engine.check_doom_loop([("read_file", {"path": "foo.py"})])
        assert result is not None
        assert "foo.py" in result

    def test_different_files_no_warning(self):
        engine = self._make_engine(doom_loop_read_threshold=3)
        result = engine.check_doom_loop([
            ("read_file", {"path": "foo.py"}),
            ("read_file", {"path": "bar.py"}),
            ("read_file", {"path": "baz.py"}),
        ])
        assert result is None

    def test_edit_warning_at_threshold(self):
        engine = self._make_engine(doom_loop_edit_threshold=3)
        result = engine.check_doom_loop([
            ("edit_file", {"path": "foo.py", "old_text": "a", "new_text": "b"}),
            ("edit_file", {"path": "foo.py", "old_text": "b", "new_text": "c"}),
            ("edit_file", {"path": "foo.py", "old_text": "c", "new_text": "d"}),
        ])
        assert result is not None
        assert "edited" in result.lower() or "edit" in result.lower()

    def test_readonly_streak_warning(self):
        engine = self._make_engine(doom_loop_readonly_streak=5)
        # 5 consecutive read-only tools
        result = engine.check_doom_loop([
            ("read_file", {"path": "a.py"}),
            ("read_file", {"path": "b.py"}),
            ("grep_files", {"pattern": "foo"}),
            ("glob_files", {"pattern": "*.py"}),
            ("list_directory", {"path": "."}),
        ])
        assert result is not None
        assert "read-only" in result.lower() or "without making any changes" in result

    def test_write_resets_readonly_streak(self):
        engine = self._make_engine(doom_loop_readonly_streak=5)
        # 4 reads, then a write, then 4 more reads — never hits 5
        engine.check_doom_loop([
            ("read_file", {"path": "a.py"}),
            ("read_file", {"path": "b.py"}),
            ("read_file", {"path": "c.py"}),
            ("read_file", {"path": "d.py"}),
        ])
        engine.check_doom_loop([
            ("edit_file", {"path": "a.py", "old_text": "x", "new_text": "y"}),
        ])
        result = engine.check_doom_loop([
            ("read_file", {"path": "e.py"}),
            ("read_file", {"path": "f.py"}),
            ("read_file", {"path": "g.py"}),
            ("read_file", {"path": "h.py"}),
        ])
        # Should be None — streak reset by edit
        readonly_warnings = [
            line for line in (result or "").split("\n")
            if "read-only" in line.lower() or "without making" in line.lower()
        ]
        assert len(readonly_warnings) == 0

    def test_warning_deduplicated(self):
        engine = self._make_engine(doom_loop_read_threshold=3)
        # First hit — warning
        result1 = engine.check_doom_loop([
            ("read_file", {"path": "foo.py"}),
            ("read_file", {"path": "foo.py"}),
            ("read_file", {"path": "foo.py"}),
        ])
        assert result1 is not None
        # Second hit same file — no duplicate warning
        result2 = engine.check_doom_loop([
            ("read_file", {"path": "foo.py"}),
        ])
        assert result2 is None

    def test_multiple_warnings_combined(self):
        engine = self._make_engine(
            doom_loop_read_threshold=2,
            doom_loop_edit_threshold=2,
        )
        result = engine.check_doom_loop([
            ("read_file", {"path": "foo.py"}),
            ("read_file", {"path": "foo.py"}),
            ("edit_file", {"path": "bar.py", "old_text": "a", "new_text": "b"}),
            ("edit_file", {"path": "bar.py", "old_text": "b", "new_text": "c"}),
        ])
        assert result is not None
        assert "foo.py" in result
        assert "bar.py" in result
