"""Guardrails engine for instruction adherence.

Toggleable system that reduces specification drift, forgotten requirements,
and repeated mistakes during LLM execution.

Seven sub-capabilities:
1. Requirement checklist — confirm_plan tool before execution
2. Trajectory monitor — periodic state injection with original task reminder
3. Completion audit — re-check original request before accepting task_complete
4. Correction detector — detect user corrections → anti-pattern lessons
5. Context pinning — original task survives compaction
6. Critique provider — active LLM-based quality review (edits, trajectory, completion)
7. Doom-loop detector — cheap counter-based stuck-pattern detection (no LLM cost)
"""

import logging
import re
from collections import Counter
from typing import TYPE_CHECKING

from blipshell.core.intent_detection import detect_review_intent

if TYPE_CHECKING:
    from blipshell.llm.router import LLMRouter
    from blipshell.models.config import GuardrailsConfig

logger = logging.getLogger(__name__)


# ── Correction Detection ────────────────────────────────────────────────────

# Patterns that indicate a user is correcting the assistant.
# Ordered roughly by specificity. Compiled once at import time.
CORRECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bi (already|just) told you\b",
        r"\bthat'?s not (what|right|correct)\b",
        r"\bno,?\s+(i said|i meant|i want|you should)\b",
        r"\byou (missed|forgot|ignored|skipped|didn'?t)\b",
        r"\bstop (doing|adding|changing|removing)\b",
        r"\bi didn'?t ask (for|you to)\b",
        r"\bthat'?s wrong\b",
        r"\bplease (actually|just)\b",
        r"\byou keep\b",
        r"\bhow many times\b",
        r"\bread (what i|my) (said|wrote|asked)\b",
        r"\bi (said|asked|wanted|meant)\b.{5,50}\bnot\b",
    ]
]


def detect_correction(user_message: str) -> str | None:
    """Check if a user message is correcting the assistant.

    Returns a short description of the correction signal, or None.
    Cheap regex check — safe to run on every message.
    """
    for pattern in CORRECTION_PATTERNS:
        match = pattern.search(user_message)
        if match:
            # Return the matched portion + surrounding context for the lesson
            start = max(0, match.start() - 20)
            end = min(len(user_message), match.end() + 80)
            context = user_message[start:end].strip()
            return context
    return None


# ── Guardrails Engine ────────────────────────────────────────────────────────


class GuardrailsEngine:
    """Manages guardrails state for a single execution.

    Created per-execution in _chat_planned(). Passed to ChatLoop via LoopConfig.
    """

    def __init__(self, config: "GuardrailsConfig", router: "LLMRouter"):
        self.config = config
        self.router = router

        # State — set by caller before loop starts
        self.original_request: str = ""

        # Populated by confirm_plan tool (if used)
        self.checklist: list[str] = []
        self.postconditions: list[str] = []

        # Completion audit retry tracking
        self.audit_retries: int = 0

        # Doom-loop detector state
        self._file_read_counts: Counter = Counter()   # path → read count
        self._file_edit_counts: Counter = Counter()   # path → edit count
        self._readonly_streak: int = 0                # consecutive read-only tool calls
        self._doom_warnings_sent: set[str] = set()    # deduplicate warnings

        # Look-before-review gate state — fire at most once per execution so a
        # stubborn model isn't blocked forever.
        self._review_gate_fired: bool = False

    def record_checklist(self, plan_steps: list[str], postconditions: list[str] | None = None):
        """Store the confirmed plan for later completion audit."""
        self.checklist = plan_steps
        self.postconditions = postconditions or []

    # ── Trajectory Monitor (#2) ──────────────────────────────────────────

    def build_trajectory_injection(
        self,
        tool_call_count: int,
        budget: int,
        tool_call_names: list[str],
    ) -> str | None:
        """Build a state injection message if due (every N tool calls).

        Returns the message to inject as a user message, or None if not due.
        Includes original task reminder and progress summary.
        """
        if not self.config.trajectory_monitor:
            return None
        if tool_call_count == 0:
            return None
        if tool_call_count % self.config.monitor_interval != 0:
            return None

        parts = [f"[CHECKPOINT — {tool_call_count}/{budget} tool calls used]"]
        parts.append(f"\nOriginal task: {self.original_request}")

        if tool_call_names:
            recent = tool_call_names[-7:]
            parts.append(f"Recent actions: {' -> '.join(recent)}")

        # Checklist progress (if confirm_plan was used)
        if self.checklist:
            parts.append(f"\nPlan ({len(self.checklist)} steps):")
            for step in self.checklist:
                parts.append(f"  - {step}")

        remaining_pct = int((1 - tool_call_count / max(budget, 1)) * 100)
        parts.append(
            f"\n{remaining_pct}% budget remaining. "
            "Are you still on task? If done, call task_complete."
        )

        return "\n".join(parts)

    # ── Completion Audit (#3) ────────────────────────────────────────────

    # Tools that actually mutate state — used to sanity-check completion claims.
    _MUTATING_TOOLS = frozenset({"edit_file", "write_file", "run_command", "shell"})

    def _deterministic_completion_check(
        self, summary: str, files_modified: str, tool_call_names: list[str] | None,
    ) -> str | None:
        """Zero-LLM completion sanity checks. Returns a rejection message or None.

        Conservative by design — only fires on unambiguous cases so it never
        false-trips a legitimate completion.
        """
        if not (summary or "").strip():
            return ("No summary provided. Call task_complete with a short summary of "
                    "what you actually did.")
        names = tool_call_names or []
        if files_modified and files_modified.strip():
            if not any(n in self._MUTATING_TOOLS for n in names):
                return (
                    f"You reported files_modified ({files_modified.strip()}) but made no "
                    "edits this turn (no edit_file/write_file/run_command). Make the change "
                    "first, then call task_complete — or clear files_modified if nothing changed."
                )
        return None

    def _completion_needs_llm_audit(
        self, files_modified: str, tool_call_names: list[str] | None, tool_call_count: int,
    ) -> bool:
        """Whether a deterministic-clean completion warrants an LLM audit.

        Trivial tasks skip it — always-on self-critique degrades easy work
        (Snorkel "self-critique paradox"). Non-trivial = had a confirmed plan,
        ran enough tools, or touched more than one file.
        """
        if self.checklist:
            return True
        if tool_call_count >= self.config.completion_audit_min_tool_calls:
            return True
        if files_modified:
            files = [f for f in re.split(r"[,\n]", files_modified) if f.strip()]
            if len(files) > 1:
                return True
        return False

    async def validate_completion(
        self,
        summary: str,
        files_modified: str = "",
        tool_call_names: list[str] | None = None,
        tool_call_count: int = 0,
    ) -> tuple[bool, str]:
        """Grounded, difficulty-gated completion check.

        Order (cheap → expensive): master toggle → retry cap → deterministic
        checks (zero LLM) → difficulty gate (skip the LLM audit on trivial,
        deterministic-clean tasks) → a single LLM PASS/FAIL audit. Returns
        (is_valid, feedback); if invalid, feedback is injected so the model can fix it.
        """
        if not self.config.completion_audit:
            return True, ""

        if self.audit_retries >= self.config.max_audit_retries:
            logger.info("Completion audit: max retries reached, accepting")
            return True, ""

        # Deterministic gate — catch obvious "not done" cases with no LLM call.
        det = self._deterministic_completion_check(summary, files_modified, tool_call_names)
        if det:
            self.audit_retries += 1
            return False, (
                f"[COMPLETION CHECK — attempt {self.audit_retries}/"
                f"{self.config.max_audit_retries}]\n{det}"
            )

        # Difficulty gate — don't spend an LLM call critiquing a trivial task that
        # already passed the deterministic checks.
        if not self._completion_needs_llm_audit(files_modified, tool_call_names, tool_call_count):
            return True, ""

        from blipshell.llm.prompts import validate_task_completion
        from blipshell.llm.router import TaskType

        system, user = validate_task_completion(
            original_request=self.original_request,
            summary=summary,
            files_modified=files_modified,
            checklist=self.checklist,
        )

        try:
            result = await self.router.generate(
                TaskType.REASONING, user, system=system,
            )
        except Exception as e:
            logger.error("Completion audit LLM call failed: %s — accepting", e)
            return True, ""

        result = result.strip()

        # Parse result: starts with PASS or FAIL
        if result.upper().startswith("PASS"):
            return True, ""

        # FAIL — extract feedback
        self.audit_retries += 1
        feedback = result
        if feedback.upper().startswith("FAIL"):
            feedback = feedback[4:].lstrip(":").lstrip()

        rejection = (
            f"[COMPLETION AUDIT — attempt {self.audit_retries}/{self.config.max_audit_retries}]\n"
            f"Your task_complete was rejected. The original request was:\n"
            f"{self.original_request}\n\n"
            f"Issues found:\n{feedback}\n\n"
            "Fix the issues and call task_complete again, or use ask_user if you're unsure."
        )

        return False, rejection

    # ── Doom-Loop Detector (#7) ─────────────────────────────────────────

    # Tools that modify state — if we see these, reset the readonly streak
    _WRITE_TOOLS = frozenset({
        "edit_file", "write_file", "run_command", "shell", "task_complete",
        "confirm_plan", "ask_user",
    })

    def check_doom_loop(
        self,
        tool_calls: list[tuple[str, dict]],
    ) -> str | None:
        """Check a batch of tool calls for stuck/repetitive patterns.

        Call this after each tool batch. Updates internal counters and returns
        a warning message if a doom-loop pattern is detected, or None.

        Zero LLM cost — pure counter logic.

        Args:
            tool_calls: list of (tool_name, arguments) from the batch
        """
        if not self.config.doom_loop_detector:
            return None

        warnings = []

        for name, args in tool_calls:
            path = args.get("path", "") or args.get("directory", "")

            # Track file reads
            if name == "read_file" and path:
                self._file_read_counts[path] += 1
                count = self._file_read_counts[path]
                if (count >= self.config.doom_loop_read_threshold
                        and f"read:{path}" not in self._doom_warnings_sent):
                    self._doom_warnings_sent.add(f"read:{path}")
                    warnings.append(
                        f"You've read '{path}' {count} times. The content hasn't changed — "
                        "it's in the conversation above. Work with what you have or "
                        "try a different approach."
                    )

            # Track file edits
            if name == "edit_file" and path:
                self._file_edit_counts[path] += 1
                count = self._file_edit_counts[path]
                if (count >= self.config.doom_loop_edit_threshold
                        and f"edit:{path}" not in self._doom_warnings_sent):
                    self._doom_warnings_sent.add(f"edit:{path}")
                    warnings.append(
                        f"You've edited '{path}' {count} times. If your edits keep failing, "
                        "re-read the file to see current state or use ask_user for help."
                    )

            # Readonly streak tracking
            if name in self._WRITE_TOOLS:
                self._readonly_streak = 0
            else:
                self._readonly_streak += 1

        # Check readonly streak
        if (self._readonly_streak >= self.config.doom_loop_readonly_streak
                and "readonly_streak" not in self._doom_warnings_sent):
            self._doom_warnings_sent.add("readonly_streak")
            warnings.append(
                f"You've made {self._readonly_streak} consecutive read-only tool calls "
                "without making any changes. If you have enough information, "
                "start making changes. If you're stuck, use ask_user."
            )

        if not warnings:
            return None

        return "[DOOM-LOOP WARNING]\n" + "\n".join(f"- {w}" for w in warnings)

    # ── Look-Before-Review gate ──────────────────────────────────────────

    # Tools that count as "actually looking at the code" this turn. grep/glob
    # don't add to the executor's files_read set, so tool_call_names is the
    # correct signal here (not files_read).
    _GROUNDING_TOOLS = frozenset({
        "read_file", "read_multiple_files", "grep_files", "glob_files",
        "list_directory", "repo_map",
    })

    def check_review_grounding(self, tool_call_names: list[str]) -> str | None:
        """Refuse a review/critique completion that never looked at the code.

        Called when task_complete fires. If the original request was a review
        request and the model made NO grounding tool call (read/grep/glob/etc.)
        this turn, returns actionable feedback so the loop rejects the
        completion and the model goes and looks. Fires at most once per
        execution. Zero LLM cost — pure intent + tool-name check.
        """
        if not self.config.review_grounding or self._review_gate_fired:
            return None
        if not detect_review_intent(self.original_request):
            return None
        if any(name in self._GROUNDING_TOOLS for name in (tool_call_names or [])):
            return None

        self._review_gate_fired = True
        return (
            "[REVIEW NOT GROUNDED]\n"
            "This is a review/critique request, but you haven't read or searched "
            "any code this turn (no read_file/grep_files/glob_files/list_directory "
            "calls). Findings stated without looking are guesses. Read or grep the "
            "relevant files, then ground each finding in a specific file and line "
            "before calling task_complete. If you look and find nothing, say so."
        )

    # ── Critique Provider (#6) ──────────────────────────────────────────

    async def critique_edit(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
    ) -> str | None:
        """Review an edit for correctness. Returns critique message or None if OK.

        Calls the REASONING model to check whether the edit is correct.
        Only runs when critique_edits is enabled.
        """
        if not self.config.critique_edits:
            return None

        from blipshell.llm.prompts import critique_edit as critique_edit_prompt
        from blipshell.llm.router import TaskType

        system, user = critique_edit_prompt(
            original_task=self.original_request,
            file_path=file_path,
            old_text=old_text,
            new_text=new_text,
        )

        try:
            result = await self.router.generate(
                TaskType.REASONING, user, system=system,
            )
        except Exception as e:
            logger.warning("Critique edit LLM call failed: %s — skipping", e)
            return None

        result = result.strip()
        if result.upper().startswith("OK"):
            return None

        # Extract the issue
        feedback = result
        if feedback.upper().startswith("ISSUE"):
            feedback = feedback[5:].lstrip(":").lstrip()

        return (
            f"[CRITIQUE — edit review for {file_path}]\n"
            f"{feedback}\n"
            "Consider re-reading the file and verifying your change."
        )

    async def critique_trajectory(
        self,
        tool_call_count: int,
        budget: int,
        tool_call_names: list[str],
    ) -> str | None:
        """Evaluate whether the current approach is productive.

        Heavier than trajectory_monitor (makes an LLM call) but provides
        actual analysis instead of just a task reminder.
        Returns critique message or None if on track.
        """
        if not self.config.critique_trajectory:
            return None
        if tool_call_count == 0:
            return None
        if tool_call_count % self.config.monitor_interval != 0:
            return None

        from blipshell.llm.prompts import critique_trajectory as critique_traj_prompt
        from blipshell.llm.router import TaskType

        system, user = critique_traj_prompt(
            original_task=self.original_request,
            recent_actions=tool_call_names,
            tool_call_count=tool_call_count,
            budget=budget,
        )

        try:
            result = await self.router.generate(
                TaskType.REASONING, user, system=system,
            )
        except Exception as e:
            logger.warning("Critique trajectory LLM call failed: %s — skipping", e)
            return None

        result = result.strip()
        if result.upper().startswith("ON TRACK"):
            return None

        feedback = result
        if feedback.upper().startswith("CONCERN"):
            feedback = feedback[7:].lstrip(":").lstrip()

        return (
            f"[CRITIQUE — approach review at {tool_call_count}/{budget} tool calls]\n"
            f"{feedback}"
        )

    async def critique_completion(
        self,
        summary: str,
        files_modified: str = "",
        tool_call_names: list[str] | None = None,
    ) -> str | None:
        """Rich pre-completion quality review. Supplements completion_audit.

        Returns critique message or None if quality is acceptable.
        Runs BEFORE the standard completion_audit pass/fail check.
        """
        if not self.config.critique_completion:
            return None

        from blipshell.llm.prompts import critique_completion as critique_comp_prompt
        from blipshell.llm.router import TaskType

        system, user = critique_comp_prompt(
            original_task=self.original_request,
            summary=summary,
            files_modified=files_modified,
            recent_actions=tool_call_names,
            checklist=self.checklist,
        )

        try:
            result = await self.router.generate(
                TaskType.REASONING, user, system=system,
            )
        except Exception as e:
            logger.warning("Critique completion LLM call failed: %s — skipping", e)
            return None

        result = result.strip()
        if result.upper().startswith("PASS"):
            return None

        feedback = result
        if feedback.upper().startswith("ISSUE"):
            feedback = feedback[5:].lstrip(":").lstrip()

        self.audit_retries += 1
        return (
            f"[CRITIQUE — completion quality review, "
            f"attempt {self.audit_retries}/{self.config.max_audit_retries}]\n"
            f"{feedback}\n\n"
            "Fix the issues and call task_complete again."
        )

    # ── Context Pinning (#5) ─────────────────────────────────────────────

    @property
    def pinned_context(self) -> str:
        """Build the pinned context block that should survive compaction.

        Injected into system prompt so it's always visible.
        """
        if not self.config.context_pinning:
            return ""

        parts = [f"[PINNED] Original task: {self.original_request}"]
        if self.checklist:
            parts.append("Confirmed plan:")
            for step in self.checklist:
                parts.append(f"  - {step}")
        return "\n".join(parts)
