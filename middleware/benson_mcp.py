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


def _coerce_numeric_args(args: dict, schema: dict) -> dict:
    """Defensively coerce string-encoded numbers to their declared type.
    Sonnet routinely emits integer params as strings (e.g. '50' instead
    of 50). The CLI's strict JSON-Schema validator rejects with
    'Input validation error', the model retries — each retry burns a
    turn. 4 of yesterday's exit-1 crashes traced to this turn-budget
    exhaustion (2026-05-02 incident)."""
    if not isinstance(args, dict) or not isinstance(schema, dict):
        return args
    props = schema.get("properties") or {}
    out = dict(args)
    for key, val in args.items():
        prop = props.get(key)
        if not isinstance(prop, dict) or not isinstance(val, str):
            continue
        types = prop.get("type")
        if isinstance(types, str):
            types = [types]
        if not types:
            continue
        s = val.strip()
        if not s:
            continue
        if "integer" in types:
            try:
                out[key] = int(s)
            except ValueError:
                pass
        elif "number" in types:
            try:
                out[key] = float(s)
            except ValueError:
                pass
    return out


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
        # Pre-coerce stringy numbers ('50' → 50) so a model quirk
        # doesn't waste a turn on the round-trip retry.
        args = _coerce_numeric_args(args or {}, input_schema)
        try:
            result = await impl(**args)
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
