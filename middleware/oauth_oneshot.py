"""One-shot OAuth-authenticated calls (text + vision).

Wraps `claude_agent_sdk.query()` for stateless single-turn prompts.
Uses the same OAuth token as the main agent (Casey's Claude Code
subscription), so calls cost nothing per-token — they consume
subscription quota only.

Vision: `ask_with_image()` passes the image path to Claude Code, which
loads it via its built-in Read tool. The model receives image content
in context exactly like a direct multi-modal API call would, but on
the OAuth subscription quota.
"""
from __future__ import annotations

import asyncio
import logging

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

logger = logging.getLogger("benson.oauth_oneshot")


async def ask(
    prompt: str,
    system_prompt: str = "",
    *,
    model: str = "haiku",
    timeout_s: float = 180.0,
) -> str:
    """Single-turn OAuth call. Returns the assistant text or '' on failure."""
    stderr_buf: list[str] = []

    def _stderr_cb(line: str) -> None:
        s = line.rstrip()
        if s:
            stderr_buf.append(s)
            logger.warning(f"[oneshot-stderr] {s}")

    options = ClaudeAgentOptions(
        system_prompt=system_prompt or None,
        model=model,
        max_turns=1,
        permission_mode="bypassPermissions",
        stderr=_stderr_cb,
    )
    text_parts: list[str] = []

    async def _run():
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

    try:
        await asyncio.wait_for(_run(), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(f"oauth_oneshot timed out after {timeout_s}s")
        return ""
    except Exception as e:
        tail = "\n".join(stderr_buf[-20:]) if stderr_buf else "(empty)"
        logger.warning(
            f"oauth_oneshot failed: {type(e).__name__}: {e}; stderr-tail: {tail}"
        )
        return ""

    return "\n".join(text_parts).strip()


async def ask_with_image(
    image_path: str,
    prompt: str,
    *,
    model: str = "sonnet",
    timeout_s: float = 240.0,
) -> str:
    """Vision call via OAuth. Lets Claude Code Read the image into context,
    then asks the question. Returns the FINAL assistant text (the answer
    after the Read tool turn — intermediate 'let me look' narration is
    dropped automatically by keeping only the last AssistantMessage).
    """
    stderr_buf: list[str] = []

    def _stderr_cb(line: str) -> None:
        s = line.rstrip()
        if s:
            stderr_buf.append(s)
            logger.warning(f"[oneshot-img-stderr] {s}")

    options = ClaudeAgentOptions(
        model=model,
        max_turns=4,
        permission_mode="bypassPermissions",
        allowed_tools=["Read"],
        stderr=_stderr_cb,
    )
    full_prompt = (
        f"Use the Read tool to view the image at {image_path}, then answer "
        f"the request below. Respond with the answer only — no narration "
        f"like 'Let me look at this' or 'Here is what I see'.\n\n"
        f"Request: {prompt}"
    )
    last_turn_texts: list[str] = []

    async def _run():
        nonlocal last_turn_texts
        async for msg in query(prompt=full_prompt, options=options):
            if isinstance(msg, AssistantMessage):
                turn_texts: list[str] = []
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        turn_texts.append(block.text)
                if turn_texts:
                    last_turn_texts = turn_texts

    try:
        await asyncio.wait_for(_run(), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(f"oauth_oneshot.ask_with_image timed out after {timeout_s}s")
        return ""
    except Exception as e:
        tail = "\n".join(stderr_buf[-20:]) if stderr_buf else "(empty)"
        logger.warning(
            f"oauth_oneshot.ask_with_image failed: {type(e).__name__}: {e}; "
            f"stderr-tail: {tail}"
        )
        return ""

    return "\n".join(last_turn_texts).strip()
