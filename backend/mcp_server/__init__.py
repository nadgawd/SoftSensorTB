"""MCP tool server package for Soft Sensor Toolbox."""

from backend.mcp_server.eda_tools import (
    LLM_TOOL_SCHEMAS as EDA_TOOL_SCHEMAS,
    TOOL_FUNCTIONS as EDA_TOOL_FUNCTIONS,
)
from backend.mcp_server.modeling_tools import (
    LLM_TOOL_SCHEMAS as MODELING_TOOL_SCHEMAS,
    TOOL_FUNCTIONS as MODELING_TOOL_FUNCTIONS,
)

LLM_TOOL_SCHEMAS = [*EDA_TOOL_SCHEMAS, *MODELING_TOOL_SCHEMAS]
TOOL_FUNCTIONS = {**EDA_TOOL_FUNCTIONS, **MODELING_TOOL_FUNCTIONS}

__all__ = [
    "EDA_TOOL_SCHEMAS",
    "LLM_TOOL_SCHEMAS",
    "MODELING_TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
]
