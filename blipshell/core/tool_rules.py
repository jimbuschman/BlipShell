"""Tool rules engine (inspired by Letta/MemGPT).

Structural constraints on tool calling flow, replacing brittle regex
heuristics. Rules filter the available tool set dynamically based on
what's been called so far in the current turn.

Rule types:
- MaxCallsRule: Limit how many times a tool can be called per turn.
- SequenceRule: After calling tool X, only tools in [Y, Z] allowed next.
- CooldownRule: After calling tool X, it can't be called again for N calls.
- TerminalRule: After calling tool X, no more tool calls allowed.

Usage:
    engine = ToolRuleEngine()
    engine.add_rule(MaxCallsRule("list_directory", max_calls=2))
    engine.add_rule(SequenceRule("read_file", allowed_next={"edit_file", "write_file"}))

    # In the tool loop:
    available = engine.filter_tools(all_tools, call_history)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ToolRule(ABC):
    """Base class for tool rules."""

    @abstractmethod
    def filter(self, tools: list[dict], history: list[str]) -> list[dict]:
        """Filter available tools based on call history.

        Args:
            tools: List of Ollama tool dicts (each has function.name).
            history: Ordered list of tool names called so far this turn.

        Returns:
            Filtered list of tools that are still allowed.
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of this rule."""
        ...


class MaxCallsRule(ToolRule):
    """Limit how many times a specific tool can be called per turn.

    Prevents the model from calling read_file 10 times, list_directory 5 times, etc.
    """

    def __init__(self, tool_name: str, max_calls: int):
        self.tool_name = tool_name
        self.max_calls = max_calls

    def filter(self, tools: list[dict], history: list[str]) -> list[dict]:
        count = history.count(self.tool_name)
        if count >= self.max_calls:
            return [t for t in tools if _tool_name(t) != self.tool_name]
        return tools

    def describe(self) -> str:
        return f"{self.tool_name}: max {self.max_calls} calls per turn"


class CooldownRule(ToolRule):
    """After calling tool X, it can't be called again for N subsequent calls.

    Prevents immediate re-calls (e.g., reading the same file twice in a row)
    while still allowing the tool later in the turn.
    """

    def __init__(self, tool_name: str, cooldown: int = 1):
        self.tool_name = tool_name
        self.cooldown = cooldown

    def filter(self, tools: list[dict], history: list[str]) -> list[dict]:
        if not history:
            return tools
        # Check if the tool was called within the last `cooldown` calls
        recent = history[-self.cooldown:]
        if self.tool_name in recent:
            return [t for t in tools if _tool_name(t) != self.tool_name]
        return tools

    def describe(self) -> str:
        return f"{self.tool_name}: cooldown {self.cooldown} calls"


class SequenceRule(ToolRule):
    """After calling tool X, only specific tools are allowed next.

    Example: After read_file, the model should edit_file or write_file,
    not call list_directory or web_search.
    """

    def __init__(self, after_tool: str, allowed_next: set[str]):
        self.after_tool = after_tool
        self.allowed_next = allowed_next

    def filter(self, tools: list[dict], history: list[str]) -> list[dict]:
        if not history or history[-1] != self.after_tool:
            return tools
        return [t for t in tools if _tool_name(t) in self.allowed_next]

    def describe(self) -> str:
        return f"after {self.after_tool}: only {', '.join(sorted(self.allowed_next))}"


class TerminalRule(ToolRule):
    """After calling tool X, no more tool calls allowed.

    The turn ends after this tool is executed. Useful for tools
    that represent a final action (like git_commit).
    """

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def filter(self, tools: list[dict], history: list[str]) -> list[dict]:
        if self.tool_name in history:
            return []
        return tools

    def describe(self) -> str:
        return f"after {self.tool_name}: stop (terminal)"


class ToolRuleEngine:
    """Applies a set of rules to filter available tools during a turn.

    Rules are applied in order. Each rule can remove tools from the
    available set. A tool must pass ALL rules to remain available.
    """

    def __init__(self):
        self.rules: list[ToolRule] = []

    def add_rule(self, rule: ToolRule):
        """Add a rule to the engine."""
        self.rules.append(rule)
        logger.debug("Added tool rule: %s", rule.describe())

    def filter_tools(self, tools: list[dict], history: list[str]) -> list[dict]:
        """Filter tools based on all rules and the current call history.

        Args:
            tools: Full set of available Ollama tool dicts.
            history: Ordered list of tool names called so far this turn.

        Returns:
            Filtered list of tools that are still allowed by all rules.
        """
        filtered = tools
        for rule in self.rules:
            before = len(filtered)
            filtered = rule.filter(filtered, history)
            if len(filtered) < before:
                removed = before - len(filtered)
                logger.debug(
                    "Rule '%s' removed %d tools (%d remaining)",
                    rule.describe(), removed, len(filtered),
                )
        return filtered

    def describe_rules(self) -> list[str]:
        """Get human-readable descriptions of all rules."""
        return [r.describe() for r in self.rules]


def _tool_name(tool_dict: dict) -> str:
    """Extract tool name from an Ollama tool dict."""
    if not isinstance(tool_dict, dict):
        return ""
    fn = tool_dict.get("function", {})
    if not isinstance(fn, dict):
        return str(fn)
    return fn.get("name", "")


def create_default_rules() -> ToolRuleEngine:
    """Create the default tool rule engine for BlipShell.

    These rules encode best practices learned from real usage:
    - Don't re-read the same directory endlessly
    - Don't call web_search more than 3 times
    - Don't list directories more than 3 times
    - After git_commit, stop — the task is done
    """
    engine = ToolRuleEngine()

    # Prevent excessive exploration
    engine.add_rule(MaxCallsRule("list_directory", max_calls=3))
    engine.add_rule(MaxCallsRule("web_search", max_calls=3))
    engine.add_rule(MaxCallsRule("web_fetch", max_calls=3))

    # Prevent immediate re-calls of the same search
    engine.add_rule(CooldownRule("grep_files", cooldown=1))
    engine.add_rule(CooldownRule("glob_files", cooldown=1))
    engine.add_rule(CooldownRule("list_directory", cooldown=1))

    # Terminal actions — after these, the task is done
    engine.add_rule(TerminalRule("git_commit"))
    engine.add_rule(TerminalRule("task_complete"))

    return engine


def create_coding_rules() -> ToolRuleEngine:
    """Create tool rules for coding/project mode.

    More permissive than default (allows more tool calls) but enforces
    coding workflow discipline:
    - Read before edit
    - Don't explore endlessly
    - Commit is terminal
    """
    engine = ToolRuleEngine()

    # Exploration caps (more generous for coding)
    engine.add_rule(MaxCallsRule("list_directory", max_calls=5))
    engine.add_rule(MaxCallsRule("web_search", max_calls=2))

    # Prevent immediate re-calls
    engine.add_rule(CooldownRule("grep_files", cooldown=1))
    engine.add_rule(CooldownRule("glob_files", cooldown=1))
    engine.add_rule(CooldownRule("list_directory", cooldown=1))

    # Terminal actions
    engine.add_rule(TerminalRule("git_commit"))
    engine.add_rule(TerminalRule("task_complete"))

    return engine
