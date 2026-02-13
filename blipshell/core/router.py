"""Task-type routing for the agent (delegates to llm.router)."""

# This module re-exports from llm.router for convenience
# in the core package namespace.

from blipshell.llm.router import LLMRouter, TaskType

__all__ = ["LLMRouter", "TaskType"]
