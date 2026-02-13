"""Tool-related Pydantic models for native Ollama tool calling."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolParameterType(str, Enum):
    """JSON Schema types for tool parameters."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """A single parameter for a tool function."""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    enum: Optional[list[str]] = None
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Definition of a tool in Ollama native format."""
    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)

    def to_ollama_tool(self) -> dict:
        """Convert to Ollama's native tool format."""
        properties = {}
        required = []
        for param in self.parameters:
            prop = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolCall(BaseModel):
    """A tool call from the LLM (native Ollama format)."""
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolResult(BaseModel):
    """Result from executing a tool."""
    tool_call_id: str = ""
    name: str
    result: str
    success: bool = True
    execution_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_ollama_message(self) -> dict:
        """Convert to Ollama tool response message format."""
        return {
            "role": "tool",
            "content": self.result,
        }
