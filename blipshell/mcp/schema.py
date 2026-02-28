"""Convert MCP tool schemas to BlipShell ToolDefinitions."""

from blipshell.models.tools import ToolDefinition, ToolParameter, ToolParameterType

# Map JSON Schema type strings to ToolParameterType
_TYPE_MAP = {
    "string": ToolParameterType.STRING,
    "integer": ToolParameterType.INTEGER,
    "number": ToolParameterType.NUMBER,
    "boolean": ToolParameterType.BOOLEAN,
    "array": ToolParameterType.ARRAY,
    "object": ToolParameterType.OBJECT,
}


def mcp_tool_to_definition(mcp_tool, server_name: str) -> ToolDefinition:
    """Convert an MCP tool definition to a BlipShell ToolDefinition.

    Tool names are prefixed with ``mcp_{server_name}_`` to avoid collisions
    with native tools.

    Args:
        mcp_tool: An MCP Tool object (from ``session.list_tools()``).
        server_name: Name of the MCP server (used for prefixing).

    Returns:
        A BlipShell ToolDefinition ready for registration.
    """
    prefixed_name = f"mcp_{server_name}_{mcp_tool.name}"

    parameters: list[ToolParameter] = []
    input_schema = getattr(mcp_tool, "inputSchema", None) or {}
    properties = input_schema.get("properties", {})
    required_set = set(input_schema.get("required", []))

    for prop_name, prop_schema in properties.items():
        json_type = prop_schema.get("type", "string")
        param = ToolParameter(
            name=prop_name,
            type=_TYPE_MAP.get(json_type, ToolParameterType.STRING),
            description=prop_schema.get("description", ""),
            required=prop_name in required_set,
            enum=prop_schema.get("enum"),
        )
        parameters.append(param)

    description = getattr(mcp_tool, "description", None) or f"MCP tool from {server_name}"

    return ToolDefinition(
        name=prefixed_name,
        description=f"[MCP:{server_name}] {description}",
        parameters=parameters,
    )
