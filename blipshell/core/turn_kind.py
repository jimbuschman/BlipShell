"""Deterministic classification of a user turn for the authorization rule
(V3 Stage E, decided 2026-09-10).

Three kinds, three consequences:
- question    - discussion. State what is recorded (decisions, reasons); change nothing.
- instruction - authorization. Act; disclose any decision in force the action overrides.
- declarative - a stated requirement or new fact. Update project state
                (follow-up, decision revision with disclosure) and propose;
                do NOT modify files or run commands unless a standing
                implementation mandate already covers the task (the
                executor path, `!plan`) - the authorization then comes from
                the standing task, not from the declarative wording.

Regex on purpose: this is instrumentation and a prompt rule, not a
permission framework. The classification is recorded per turn and a write
tool called on a declarative turn without a mandate is logged as an event,
so the distinction is testable and observable; nothing is blocked here.
"""

from __future__ import annotations

import re

QUESTION = "question"
INSTRUCTION = "instruction"
DECLARATIVE = "declarative"

_QUESTION_OPEN = re.compile(
    r"^\s*(should|shall|could|would|can|do|does|did|is|are|was|were|will|what|which|who|where|when|why|how|"
    r"any|anything|remind me|give me|tell me|show me|list|summari[sz]e|recap|walk me through|where did we)\b", re.I)
_INSTRUCTION = re.compile(
    r"(\b(can|could|would|will) you\b|\bplease\b|\bgo ahead\b|\bset (that|it|this) up\b|\bmake (it|this|that)\b|"
    r"\blet'?s (just )?(make|switch|change|move|do|wire|add|build|implement|write|create|set|run|fix)\b|"
    r"^\s*(implement|add|create|write|fix|change|switch|update|wire|build|run|install|remove|delete|rename|refactor|"
    r"move|set|configure|enable|disable|deploy|migrate|generate|draft|do)\b|\bfor me\b|\bdo (it|that|this)\b)",
    re.I)
# An information request anywhere in the message ("Back from a break. Give me
# the state of this project") is a question about the record, not a change.
_INFO_REQUEST = re.compile(
    r"\b(give me|tell me|show me|remind me|walk me through|where did we|where are we|where do we stand|"
    r"what'?s (decided|done|open|left|next|the state)|summari[sz]e|recap|status of|state of)\b", re.I)
WRITE_TOOLS = ("edit_file", "write_file", "delete_file", "run_command", "git_add", "git_commit")


def classify_turn(text: str) -> str:
    """question | instruction | declarative. An explicit ask anywhere in the
    message wins over a trailing question mark ("Let's make it hourly, can you
    set that up?" is an instruction)."""
    t = (text or "").strip()
    if not t:
        return DECLARATIVE
    if _INSTRUCTION.search(t):
        return INSTRUCTION
    if t.endswith("?") or _QUESTION_OPEN.match(t) or _INFO_REQUEST.search(t):
        return QUESTION
    return DECLARATIVE


DECLARATIVE_RULE = (
    "AUTHORIZATION FOR THIS TURN: the user's message states a requirement or a new fact; it is not a request "
    "to change anything. Record it (follow-up, or a decision revision with the overridden decision and its "
    "reason disclosed) and propose the next steps. Do not modify files or run commands on this turn - no "
    "implementation task is in progress. If the user wants it built, they will say so."
)


def mutations_in(tool_names) -> list[str]:
    return [t for t in (tool_names or []) if t in WRITE_TOOLS]
