"""Deterministic memory hooks — fire regardless of model state.

Agent-initiated STM writes (the model calling stm_append itself) are
unreliable under chained tools, error recovery, and mid-session crashes
because they require the model to remember to remember. These hooks
fire from the harness instead, so they execute even when the model
forgets, errors out, or never gets the chance.

Two hook points:
  * post_tool_use(name, result) — fired by benson_mcp._make_wrapper()
    after any side-effect tool in SIDE_EFFECT_TOOLS returns ok=True.
    Writes a one-line trace to STM so future turns know what happened.
  * session_stop_hook(response)  — fired by main.handle_conversation()
    after run_agent() returns. Asks haiku to distill 0-2 durable facts
    from the assistant's response and appends them to STM. Gated by
    the MEMORY_STOP_HOOK_ENABLED env var (default true).

Both hooks run as background asyncio tasks and must never raise — a
broken memory hook should never break a user-facing turn.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from short_term import stm_append
from oauth_oneshot import ask as oauth_ask

logger = logging.getLogger("benson.memory_hooks")

# Tools whose successful completion is always worth recording. These all
# have observable household side effects (state change, message sent,
# event scheduled) — i.e. things future turns may reasonably ask about.
SIDE_EFFECT_TOOLS = {
    "propose_change", "announce", "send_signal",
    "create_calendar_event", "update_calendar_event", "delete_calendar_event",
    "schedule_meal", "unschedule_meal",
    "mark_chore_done", "add_chore", "delete_chore",
    "log_event", "list_add", "list_remove",
}


def _summarize_result(tool_name: str, result: dict) -> str:
    """Extract the most useful single-line summary from a tool result."""
    # Prefer explicit summary fields in the order most likely to be informative.
    for key in ("branch", "message", "event_id", "meal", "item", "chore", "id", "summary"):
        if key in result:
            return f"{result[key]}"
    return "ok"


async def post_tool_use(tool_name: str, result: Any) -> None:
    """Called after any tool in SIDE_EFFECT_TOOLS returns ok=True."""
    try:
        if tool_name not in SIDE_EFFECT_TOOLS:
            return
        if not isinstance(result, dict) or not result.get("ok"):
            return
        summary = _summarize_result(tool_name, result)
        stm_append("today", f"[auto] {tool_name}: {summary}")
    except Exception:
        # Hooks must never raise — a broken hook can't break a turn.
        logger.exception("post_tool_use hook failed")


_STOP_HOOK_PROMPT = """\
You are a memory extractor. Given the assistant's response below, identify 0-2 short facts that would still be useful to know in the NEXT conversation turn (durable household state, corrections, decisions made). Write each as a single short sentence. If nothing is worth noting, output the single word NONE.

Assistant response:
{response}
"""


async def session_stop_hook(response: str) -> None:
    """Called after run_agent() returns. Extracts 0-2 STM facts via haiku."""
    try:
        if os.environ.get("MEMORY_STOP_HOOK_ENABLED", "true").lower() == "false":
            return
        if not response or len(response) < 40:
            return
        prompt = _STOP_HOOK_PROMPT.format(response=response[:2000])
        text = await oauth_ask(prompt, model="haiku", timeout_s=30)
        if not text or text.strip().upper() == "NONE":
            return
        for line in text.strip().splitlines():
            line = line.strip().lstrip("-•123456789. ")
            if line:
                stm_append("today", f"[stop-hook] {line}")
    except Exception:
        logger.exception("session_stop_hook failed")
