"""Guardrails engine for instruction adherence.

Toggleable system that reduces specification drift, forgotten requirements,
and repeated mistakes during LLM execution.

Five sub-capabilities:
1. Requirement checklist — confirm_plan tool before execution
2. Trajectory monitor — periodic state injection with original task reminder
3. Completion audit — re-check original request before accepting task_complete
4. Correction detector — detect user corrections → anti-pattern lessons
5. Context pinning — original task survives compaction
"""

import logging
import re
from typing import TYPE_CHECKING

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

        remaining_pct = int((1 - tool_call_count / budget) * 100)
        parts.append(
            f"\n{remaining_pct}% budget remaining. "
            "Are you still on task? If done, call task_complete."
        )

        return "\n".join(parts)

    # ── Completion Audit (#3) ────────────────────────────────────────────

    async def validate_completion(
        self,
        summary: str,
        files_modified: str = "",
    ) -> tuple[bool, str]:
        """Validate that task_complete matches the original request.

        Returns (is_valid, feedback). If invalid, feedback is a message
        to inject back into the conversation so the model can fix it.
        """
        if not self.config.completion_audit:
            return True, ""

        if self.audit_retries >= self.config.max_audit_retries:
            logger.info("Completion audit: max retries reached, accepting")
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
