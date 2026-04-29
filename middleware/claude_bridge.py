"""Tier 2 escalation: invoke the Claude Code CLI as a subprocess.

The CLI handles authentication via ANTHROPIC_API_KEY (passed through the
systemd unit's EnvironmentFile). We use --bare to skip OAuth/keychain
attempts and force API-key auth, --print for non-interactive output, and
--effort to pick the deliberation level.
"""
from __future__ import annotations

import asyncio
import logging
import os

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_CLI,
    CLAUDE_DEFAULT_EFFORT,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT_S,
)

logger = logging.getLogger("benson.claude")


class ClaudeUnavailable(RuntimeError):
    """Raised when the Claude CLI can't be reached or auth is missing."""


async def ask_claude(
    prompt: str,
    effort: str = CLAUDE_DEFAULT_EFFORT,
    timeout_s: int = CLAUDE_TIMEOUT_S,
) -> str:
    """Run a single-shot Claude prompt and return stdout text."""
    if not ANTHROPIC_API_KEY:
        raise ClaudeUnavailable(
            "ANTHROPIC_API_KEY is not set; can't escalate to Claude."
        )

    cmd = [
        CLAUDE_CLI,
        "--bare",
        "-p",
        "--model",
        CLAUDE_MODEL,
        "--effort",
        effort,
        prompt,
    ]
    env = {**os.environ, "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY}

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise ClaudeUnavailable(f"Claude CLI timed out after {timeout_s}s")

    if proc.returncode != 0:
        msg = (stderr or b"").decode(errors="replace").strip()
        raise ClaudeUnavailable(
            f"Claude CLI exited {proc.returncode}: {msg[:500]}"
        )
    return stdout.decode(errors="replace").strip()
