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
# Politeness is not authorization (review 2026-09-10, finding 6): an
# instruction needs an ACTION verb - "can you implement this" is one, "could
# you explain why" is a question, "please do not change any files" is a
# stated constraint. The requested action, its negation and information
# intent all survive classification.
_ACTION = (r"(implement|add|create|write|fix|change|switch|update|wire|build|run|install|remove|delete|rename|"
           r"refactor|move|set (it|that|this) up|set up|configure|enable|disable|deploy|migrate|generate|draft|"
           r"make (it|this|that|the)|rewrite|edit|apply|commit|push|revert|schedule|start|kick off|go ahead)")
_INSTRUCTION = re.compile(
    r"(\b(can|could|would|will) you\b( please)?( just)?( go ahead and)? " + _ACTION + r"\b"
    r"|^\s*(please )?" + _ACTION + r"\b"
    r"|\bplease " + _ACTION + r"\b"
    r"|\blet'?s (just )?" + _ACTION + r"\b"
    r"|\bgo ahead\b|\bset (that|it|this) up\b|\bdo (it|that|this)\b)",
    re.I)
# A prohibition or constraint is a statement, never authorization to perform
# the prohibited operation.
_PROHIBITION = re.compile(r"^\s*(please )?(do not|don'?t|never|stop|avoid|refrain from)\b", re.I)
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
    if _PROHIBITION.match(t):
        return DECLARATIVE  # a constraint: update state, change nothing
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
