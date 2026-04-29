"""In-process MCP server exposing Benson's 27 tools to claude-agent-sdk.

Wraps each function in `agent_tools.IMPL` with the `@tool` decorator,
returning MCP-shaped responses. The wrappers are thin: they just call
the existing async impls and JSON-encode the result for the model.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool

from agent_tools import IMPL, TOOLS

logger = logging.getLogger("benson.mcp")


def _make_wrapper(name: str, description: str, input_schema: dict):
    """Build a Python coroutine wrapping IMPL[name] for the @tool decorator."""

    @tool(name, description, input_schema)
    async def wrapped(args: dict[str, Any]) -> dict[str, Any]:
        impl = IMPL.get(name)
        if impl is None:
            return {
                "content": [{"type": "text", "text": json.dumps({"error": f"impl missing for {name}"})}],
                "isError": True,
            }
        try:
            result = await impl(**(args or {}))
        except TypeError as e:
            # Most likely cause: extra/missing kwargs in args
            return {
                "content": [{"type": "text", "text": json.dumps({"error": f"{name} arg mismatch: {e}"})}],
                "isError": True,
            }
        except Exception as e:
            logger.exception(f"tool {name} raised")
            return {
                "content": [{"type": "text", "text": json.dumps({"error": f"{type(e).__name__}: {e}"})}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

    return wrapped


# Build the SdkMcpTool list by iterating the existing TOOLS schema entries.
_sdk_tools: list[SdkMcpTool] = [
    _make_wrapper(t["name"], t["description"], t["input_schema"])
    for t in TOOLS
]

SERVER = create_sdk_mcp_server(name="benson", version="1.0.0", tools=_sdk_tools)
ALLOWED_TOOL_NAMES: list[str] = [f"mcp__benson__{t['name']}" for t in TOOLS]
