"""Benson agent — Claude tool-use loop.

One entry: `run_agent(user_text, speaker, room, system_prompt) -> (response, tier, meta)`.

Implements Anthropic's tool-use protocol directly:
  1. Send user message + tools to Claude.
  2. If response.stop_reason == 'tool_use', execute tools concurrently,
     append `{role: assistant, content}` and `{role: user, tool_results}`,
     loop.
  3. If 'end_turn' or max iterations, return assembled text.

Session continuity is handled by `agent_session`: messages persist on
disk per (speaker, room) and are reloaded on each call.

Falls through to Ollama on Anthropic API failure (insufficient credit,
network, etc.) — same fallback contract as the prior router.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from agent_session import Session, load_or_create, lock_for
from agent_tools import IMPL, TOOLS
from claude_models import MODEL_ID, ModelTier, select as select_model
from ollama_client import ask_ollama

logger = logging.getLogger("benson.agent")

MAX_AGENT_ITERATIONS = 6  # ceiling on tool-use loops per request


_client: AsyncAnthropic | None = None


def _anthropic() -> AsyncAnthropic:
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _client = AsyncAnthropic(api_key=key)
    return _client


def _build_system_with_context(
    base_prompt: str,
    *,
    speaker: str | None,
    room: str | None,
) -> str:
    now = datetime.now().strftime("%A, %Y-%m-%d %H:%M %Z")
    ctx = (
        f"Current speaker: {speaker or 'unknown'}.\n"
        f"Speaking from: {room or 'unknown'}.\n"
        f"Time: {now}.\n"
        "Use the available tools when the user asks for information or "
        "actions. Use search_memory before answering 'do you remember' or "
        "preference questions. Use remember_this only for durable facts. "
        "Keep spoken/Telegram replies concise and in Benson's voice."
    )
    return f"{base_prompt}\n\n--- session context ---\n{ctx}"


async def _execute_tool(name: str, args: dict) -> tuple[bool, Any]:
    fn = IMPL.get(name)
    if fn is None:
        return False, {"error": f"unknown tool: {name}"}
    try:
        result = await fn(**args)
        return True, result
    except Exception as e:
        logger.exception(f"tool {name} raised")
        return False, {"error": f"{type(e).__name__}: {e}"}


def _content_to_text(content_blocks) -> str:
    parts: list[str] = []
    for b in content_blocks:
        if b.type == "text":
            parts.append(b.text)
    return "\n".join(parts).strip()


async def run_agent(
    user_text: str,
    *,
    speaker: str | None,
    room: str | None,
    system_prompt: str,
    timeout_s: float = 180.0,
) -> tuple[str, str, dict]:
    """Returns (response_text, tier_used, metadata)."""
    session = load_or_create(speaker, room)
    lock = lock_for(session.session_id)

    async with lock:
        choice = select_model(user_text)
        full_system = _build_system_with_context(
            system_prompt, speaker=speaker, room=room
        )

        # Append the new user turn to session history.
        session.messages.append({"role": "user", "content": user_text})
        session.trim()

        meta: dict[str, Any] = {
            "session_id": session.session_id,
            "session_messages_before": len(session.messages),
            "model": choice.model_id,
            "tool_calls": [],
        }

        try:
            text, tier = await asyncio.wait_for(
                _agent_loop(session, full_system, choice, meta), timeout=timeout_s
            )
        except (
            anthropic.APIError,
            anthropic.APIConnectionError,
            anthropic.APIStatusError,
            asyncio.TimeoutError,
        ) as e:
            logger.warning(
                f"agent fell back to Ollama ({type(e).__name__}: {e})"
            )
            # Roll back the user turn we appended — session shouldn't
            # remember a turn that didn't actually run.
            if session.messages and session.messages[-1].get("role") == "user":
                session.messages.pop()
            try:
                text = await ask_ollama(
                    user_text, system_prompt, timeout_s=int(timeout_s)
                )
                tier = "ollama_fallback"
            except Exception as e2:
                logger.exception("Ollama fallback failed too")
                text = (
                    f"I had trouble reaching both Claude and the local "
                    f"model ({type(e2).__name__}). Try again in a moment."
                )
                tier = "all_failed"
            meta["fallback"] = True

        session.last_active = time.time()
        session.save()
        meta["session_messages_after"] = len(session.messages)
        return text, tier, meta


async def _agent_loop(
    session: Session,
    full_system: str,
    choice,
    meta: dict[str, Any],
) -> tuple[str, str]:
    client = _anthropic()
    iterations = 0

    while iterations < MAX_AGENT_ITERATIONS:
        iterations += 1
        kwargs: dict[str, Any] = {
            "model": choice.model_id,
            "max_tokens": choice.max_tokens,
            "system": full_system,
            "tools": TOOLS,
            "messages": session.messages,
        }
        if choice.thinking_tokens > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": choice.thinking_tokens,
            }
            if choice.max_tokens <= choice.thinking_tokens:
                kwargs["max_tokens"] = choice.thinking_tokens + 2048

        response = await client.messages.create(**kwargs)
        # Append assistant turn to session (raw content blocks for tool_use roundtrip)
        assistant_content = [b.model_dump() for b in response.content]
        session.messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            text = _content_to_text(response.content)
            return text or "(no response)", choice.tier.value

        # Execute all tool_use blocks (concurrently).
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        async def _run(tu):
            ok, result = await _execute_tool(tu.name, tu.input or {})
            meta["tool_calls"].append(
                {"name": tu.name, "ok": ok, "input": tu.input}
            )
            return tu.id, ok, result

        results = await asyncio.gather(*(_run(tu) for tu in tool_uses))
        tool_results = []
        for tool_use_id, ok, result in results:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(result, default=str),
                "is_error": not ok,
            })
        session.messages.append({"role": "user", "content": tool_results})

    # Hit iteration ceiling — return whatever the last assistant message had
    last = session.messages[-1] if session.messages else None
    if last and last.get("role") == "assistant":
        text_parts = [
            b.get("text", "")
            for b in last["content"]
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(text_parts).strip()
        if text:
            return text, choice.tier.value
    return (
        "I worked on that but ran out of steps before finishing. "
        "Try rephrasing or breaking it into pieces.",
        choice.tier.value,
    )
