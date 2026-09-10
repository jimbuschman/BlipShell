"""Decision tools (V3 E1): record, revise, reopen, list.

The model records a decision when the user makes one (or, flagged, when it
proposes one itself), with the reason and the condition under which it
should be reconsidered. Revising writes a supersession record from the old
decision to the new; reopening undoes it. Storage lives in
memory/decisions.py - these tools are thin.
"""

from __future__ import annotations

from blipshell.core.tools.base import Tool, ToolFailure
from blipshell.memory import decisions
from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType


def _fmt(d: decisions.Decision) -> str:
    tail = f" (revisit when: {d.revisit_when})" if d.revisit_when else ""
    return f"Decision #{d.id} [{d.status}, {d.decided_by}]: {d.decision} - because {d.reason or 'n/a'}{tail}"


class RecordDecisionTool(Tool):
    read_only = False

    def __init__(self, sqlite, vectors, session_id=None, project=None):
        self._sqlite, self._vectors, self._session_id, self._project = sqlite, vectors, session_id, project

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="record_decision",
            description=(
                "Record a decision WITH its reason and the condition under which it should be "
                "reconsidered. Use when the user decides or rejects something ('let's go with X', "
                "'no, not Y because Z'). The reason matters more than the decision: it is what "
                "lets a rejected idea be raised again when the constraint changes, and not before."
            ),
            parameters=[
                ToolParameter(name="decision", type=ToolParameterType.STRING,
                              description="What was decided or rejected, in one sentence."),
                ToolParameter(name="reason", type=ToolParameterType.STRING,
                              description="The constraint or argument behind it.", required=False),
                ToolParameter(name="revisit_when", type=ToolParameterType.STRING,
                              description="The condition that should reopen it (e.g. 'a free tier appears', 'the model runs on a Pi').",
                              required=False),
                ToolParameter(name="decided_by", type=ToolParameterType.STRING,
                              description="'user' if the user decided (default); 'assistant' if this is your own proposal.",
                              required=False, enum=["user", "assistant"]),
            ],
        )

    async def execute(self, decision: str, reason: str = "", revisit_when: str = "",
                      decided_by: str = "user", **kwargs) -> str:
        if not decision or not decision.strip():
            return ToolFailure("record_decision needs a decision.")
        d = await decisions.record_decision(
            self._sqlite, self._vectors, decision=decision, reason=reason, revisit_when=revisit_when,
            project=self._project, session_id=self._session_id, decided_by=decided_by,
        )
        return "Recorded. " + _fmt(d)


class ReviseDecisionTool(Tool):
    read_only = False

    def __init__(self, sqlite, vectors, session_id=None):
        self._sqlite, self._vectors, self._session_id = sqlite, vectors, session_id

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="revise_decision",
            description=(
                "Replace an earlier decision with a new one. The old decision is kept and marked "
                "superseded (it still answers 'how did we get here'); the new one is current."
            ),
            parameters=[
                ToolParameter(name="decision_id", type=ToolParameterType.INTEGER,
                              description="ID of the decision being replaced (from list_decisions or context)."),
                ToolParameter(name="decision", type=ToolParameterType.STRING, description="The new decision."),
                ToolParameter(name="reason", type=ToolParameterType.STRING, description="Why it changed.", required=False),
                ToolParameter(name="revisit_when", type=ToolParameterType.STRING,
                              description="Condition to reconsider the new decision.", required=False),
                ToolParameter(name="decided_by", type=ToolParameterType.STRING, required=False,
                              description="'user' (default) or 'assistant'.", enum=["user", "assistant"]),
            ],
        )

    async def execute(self, decision_id: int, decision: str, reason: str = "",
                      revisit_when: str = "", decided_by: str = "user", **kwargs) -> str:
        old = await decisions.get_decision(self._sqlite, int(decision_id))
        new = await decisions.revise_decision(
            self._sqlite, self._vectors, int(decision_id), decision=decision, reason=reason,
            revisit_when=revisit_when, session_id=self._session_id, decided_by=decided_by,
        )
        if new is None:
            return ToolFailure(f"Decision {decision_id} not found.")
        # The disclosure material: what was overridden and why it stood. The
        # reply must say this to the user (production batch 2026-09-09: an
        # imperative request flipped a decision in force 4/5 without it).
        if old is not None:
            return (f"Revised. This OVERRIDES decision #{old.id} '{old.decision}', now superseded (it was in force "
                    f"because: {old.reason or 'no reason recorded'}). Tell the user that. Now: " + _fmt(new))
        return f"Decision #{decision_id} superseded. " + _fmt(new)


class ReopenDecisionTool(Tool):
    read_only = False

    def __init__(self, sqlite):
        self._sqlite = sqlite

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="reopen_decision",
            description=(
                "Reopen a superseded decision because its revisit condition was met (or the "
                "replacement turned out wrong). The decision becomes current again; nothing is deleted."
            ),
            parameters=[
                ToolParameter(name="decision_id", type=ToolParameterType.INTEGER, description="ID of the decision to reopen."),
                ToolParameter(name="reason", type=ToolParameterType.STRING, description="What changed.", required=False),
                ToolParameter(name="restore", type=ToolParameterType.BOOLEAN, required=False,
                              description="false (default): reopen for DISCUSSION only - the current decision stays in "
                                          "force. true: RESTORE this decision as the governing one and retire its "
                                          "replacement (only when the user has said so)."),
            ],
        )

    async def execute(self, decision_id: int, reason: str = "", restore: bool = False, **kwargs) -> str:
        d = await decisions.reopen_decision(self._sqlite, int(decision_id), reason=reason, restore=bool(restore))
        if d is None:
            return ToolFailure(f"Decision {decision_id} not found.")
        return ("Restored as the governing decision. " if restore else
                "Reopened for discussion (not in force until restored or revised). ") + _fmt(d)


class ListDecisionsTool(Tool):
    read_only = True

    def __init__(self, sqlite, project=None):
        self._sqlite, self._project = sqlite, project

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_decisions",
            description="List recorded decisions (current by default) with their reasons and revisit conditions.",
            parameters=[
                ToolParameter(name="status", type=ToolParameterType.STRING, required=False,
                              description="active (default) | superseded | reopened | all",
                              enum=["active", "superseded", "reopened", "all"]),
                ToolParameter(name="all_projects", type=ToolParameterType.BOOLEAN, required=False,
                              description="Include decisions from other projects (default: current project only)."),
            ],
        )

    async def execute(self, status: str = "active", all_projects: bool = False, **kwargs) -> str:
        st = None if status == "all" else status
        items = await decisions.list_decisions(
            self._sqlite, project=None if all_projects else self._project, status=st,
        )
        if not items:
            return "No decisions recorded" + (f" with status {status}." if st else ".")
        return "\n".join(_fmt(d) for d in items)
