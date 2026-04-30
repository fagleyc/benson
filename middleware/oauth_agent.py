"""Benson agent — claude-agent-sdk path (OAuth via Claude Code subscription).

Replaces the direct-API `agent.run_agent` with a subprocess-based path
that uses Casey's Claude Code OAuth instead of the Anthropic API key.

Same external contract:
    run_agent(user_text, *, speaker, room, system_prompt) -> (text, tier, meta)

Falls back to the API-key agent (if a key is set) and ultimately to
Ollama if both Claude paths fail.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from agent_session import IDLE_TIMEOUT_MIN, SESSIONS_DIR, lock_for, _make_id
from benson_mcp import ALLOWED_TOOL_NAMES, SERVER
from claude_models import ModelTier, select as select_model

logger = logging.getLogger("benson.oauth_agent")

# Map our internal tier choices to claude-agent-sdk model strings.
_MODEL_FOR_TIER = {
    ModelTier.HAIKU: "haiku",
    ModelTier.SONNET: "sonnet",
    ModelTier.OPUS: "opus",
}


def _load_speaker_memory(speaker: str | None) -> str:
    """Read the speaker's persistent memory file + household memory.
    These are injected into the system prompt every turn so Benson never
    has to call memory_read just to know basic facts about the person."""
    from pathlib import Path
    MEMORY_DIR = Path("/opt/benson/memory")
    chunks: list[str] = []
    if speaker:
        sp_path = MEMORY_DIR / f"{speaker.lower()}.md"
        if sp_path.exists():
            try:
                chunks.append(f"### {speaker}\n" + sp_path.read_text(errors="replace").strip())
            except Exception:
                pass
    h_path = MEMORY_DIR / "household.md"
    if h_path.exists():
        try:
            chunks.append("### Household\n" + h_path.read_text(errors="replace").strip())
        except Exception:
            pass
    return "\n\n".join(chunks)


def _relevant_ltm(user_text: str, k: int = 5, max_distance: float = 0.30) -> str:
    """Semantic-search the LTM (memory_index pgvector table) for memories
    relevant to this turn's user_text, and format as a system-prompt
    section. Returns '' if nothing relevant or on any failure — this
    must NEVER block the agent loop.
    """
    if not user_text or len(user_text.strip()) < 4:
        return ""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        from memory import _embedding_model
        from config import PG_DSN

        emb = _embedding_model().encode(
            user_text, normalize_embeddings=True
        ).tolist()
        emb_lit = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        with psycopg2.connect(**PG_DSN) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT source_type, title, content, "
                    "embedding <=> %s::vector AS distance "
                    "FROM memory_index WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> %s::vector LIMIT %s",
                    (emb_lit, emb_lit, k),
                )
                rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.warning(f"LTM injection failed (non-fatal): {e}")
        return ""

    kept = [r for r in rows if r.get("distance", 1.0) <= max_distance]
    if not kept:
        return ""
    lines = ["--- relevant memories from long-term storage ---"]
    for r in kept:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()
        if not content:
            continue
        snip = content[:600] + ("…" if len(content) > 600 else "")
        prefix = f"[{r['source_type']}]"
        if title:
            prefix += f" {title}"
        lines.append(f"\n{prefix}\n{snip}")
    lines.append("\n--- end relevant memories ---")
    return "".join(lines)


def _build_system_prompt(
    base: str, *, speaker: str | None, room: str | None,
    user_text: str = "",
) -> str:
    now = datetime.now().strftime("%A, %Y-%m-%d %H:%M %Z")
    addendum = (
        f"\n\n--- session context ---\n"
        f"Current speaker: {speaker or 'unknown'}.\n"
        f"Time: {now}.\n"
        "Conversation history is unified per speaker — every channel this "
        "person uses (hub web chat, Signal DM, Signal group, voice on Sonos) "
        "shares the same recent history, which is provided to you in the "
        "user message under '--- recent conversation history ---'. ALWAYS "
        "scan that history before answering — when the user references "
        "something they said earlier ('add to that recipe', 'change those "
        "events', 'what was the second one'), the answer is in there.\n"
        "Channel verbosity: voice/Sonos = ≤2 short sentences, Signal DM/group = "
        "a few sentences, hub web chat = up to a paragraph if useful.\n\n"
        "PERSISTENT MEMORY — your durable per-person knowledge is loaded "
        "below. You can rely on this without re-reading. To ADD to it: any "
        "time a household member says any form of 'remember X', 'note that "
        "X', 'keep in mind', 'don't forget', 'save this', or volunteers a "
        "durable fact about themselves (preferences, allergies, schedules, "
        "projects, family details), CALL memory_append IMMEDIATELY in the "
        "same turn — to '<speaker>.md' for personal facts, 'household.md' "
        "for shared facts. Then confirm tersely. Don't ask permission to "
        "remember — just do it and confirm. To REMOVE/EDIT: use memory_read "
        "→ memory_write with cleaned content."
    )
    mem = _load_speaker_memory(speaker)
    if mem:
        addendum += (
            "\n\n--- persistent memory loaded for this speaker ---\n"
            + mem
            + "\n--- end memory ---"
        )

    # Short-term memory (Benson's own working notes from recent turns).
    # Auto-loads so future-you doesn't repeat today's mistakes.
    try:
        from short_term import stm_for_prompt
        stm = stm_for_prompt()
        if stm:
            addendum += "\n\n" + stm
    except Exception as e:
        logger.warning(f"STM load failed (non-fatal): {e}")

    # Long-term memory (semantic search over indexed history).
    if user_text:
        ltm = _relevant_ltm(user_text)
        if ltm:
            addendum += "\n\n" + ltm

    return base + addendum


def _channel_prefix(room: str | None) -> str:
    return f"[{room or 'unknown'}] " if room else ""


def _fetch_recent_history(speaker: str | None, limit: int = 12) -> list[dict]:
    """Pull the speaker's last N turns from `conversations` so the agent
    has explicit context — independent of any SDK session state."""
    if not speaker:
        return []
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from config import PG_DSN
    try:
        with psycopg2.connect(**PG_DSN) as c, c.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_text, benson_response, room, created_at
                FROM conversations
                WHERE speaker = %s
                  AND created_at > NOW() - INTERVAL '6 hours'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (speaker, limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"history fetch failed: {e}")
        return []


def _format_history(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines = ["--- recent conversation history (this conversation continues — read carefully) ---"]
    for r in rows:
        room = r.get("room") or "unknown"
        u = (r.get("user_text") or "").strip()
        a = (r.get("benson_response") or "").strip()
        if u:
            snippet = u if len(u) <= 800 else u[:800] + "…"
            lines.append(f"USER [{room}]: {snippet}")
        if a:
            snippet = a if len(a) <= 1200 else a[:1200] + "…"
            lines.append(f"BENSON: {snippet}")
    lines.append("--- end of history; new turn from user follows ---\n")
    return "\n".join(lines) + "\n"


def _session_id_for(speaker: str | None, room: str | None) -> tuple[str, bool]:
    """Return (session_id, is_resume). is_resume=True means we have a
    previously-saved session id within idle window."""
    import json
    sid_path = SESSIONS_DIR / f"oauth_{_make_id(speaker, room)}.json"
    now = time.time()
    if sid_path.exists():
        try:
            d = json.loads(sid_path.read_text())
            if now - d.get("last_active", 0) < IDLE_TIMEOUT_MIN * 60:
                return d["session_id"], True
        except Exception:
            pass
    new_id = str(uuid.uuid4())
    return new_id, False


def _save_session_id(speaker: str | None, room: str | None, sid: str) -> None:
    import json
    sid_path = SESSIONS_DIR / f"oauth_{_make_id(speaker, room)}.json"
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid_path.write_text(json.dumps({
        "session_id": sid,
        "speaker": speaker,
        "room": room,
        "last_active": time.time(),
    }))


# Per-speaker rolling failure counter — tool ok=false in last N turns.
# Bumps the tier up next turn (Casey 2026-04-30: opus after failure).
_FAILURE_WINDOW_TURNS = 3
_recent_failures: dict[str, list[float]] = {}


def _record_tool_failure(speaker: str | None) -> None:
    if not speaker:
        return
    buf = _recent_failures.setdefault(speaker, [])
    buf.append(time.time())
    # Trim: keep only the most recent N entries.
    if len(buf) > _FAILURE_WINDOW_TURNS:
        del buf[: len(buf) - _FAILURE_WINDOW_TURNS]


def _failure_count(speaker: str | None) -> int:
    if not speaker:
        return 0
    buf = _recent_failures.get(speaker, [])
    cutoff = time.time() - 600  # 10 min window
    return sum(1 for t in buf if t >= cutoff)


def recent_failures_snapshot() -> dict:
    """Read-only view of the per-speaker failure ring buffers, used by
    the /admin/benson observability page."""
    cutoff = time.time() - 600
    out: dict[str, dict] = {}
    for speaker, buf in _recent_failures.items():
        recent = [t for t in buf if t >= cutoff]
        if recent:
            out[speaker] = {
                "count_10min": len(recent),
                "last_failure_age_seconds": int(time.time() - recent[-1]),
            }
    return out


def _clear_failures_on_success(speaker: str | None) -> None:
    if speaker:
        _recent_failures.pop(speaker, None)


async def run_agent(
    user_text: str,
    *,
    speaker: str | None,
    room: str | None,
    system_prompt: str,
    timeout_s: float = 240.0,
) -> tuple[str, str, dict]:
    """OAuth-based agent. Returns (text, tier, meta).

    Each turn is a FRESH SDK session (no resume) — context comes from
    explicit history injection drawn from the `conversations` table.
    This makes memory robust to SDK session corruption, model switching,
    and falls-back-to-API paths."""
    session_id = str(uuid.uuid4())
    lock = lock_for(speaker or "anon")  # serialize per-speaker

    async with lock:
        recent_fail = _failure_count(speaker)
        choice = select_model(
            user_text, recent_failures=recent_fail
        )
        sys_prompt = _build_system_prompt(
            system_prompt, speaker=speaker, room=room, user_text=user_text
        )

        # Capture the bundled CLI's stderr into the same conversation log
        # so the next 'Fatal error in message reader: exit code 1' isn't
        # opaque. The SDK only routes stderr to a callback if one is set —
        # without this, "Check stderr output for details" resolves to
        # nothing, which is what we've been seeing all day.
        stderr_buf: list[str] = []

        def _stderr_cb(line: str) -> None:
            stripped = line.rstrip()
            if stripped:
                stderr_buf.append(stripped)
                logger.warning(f"[sdk-stderr {session_id[:8]}] {stripped}")

        options = ClaudeAgentOptions(
            system_prompt=sys_prompt,
            mcp_servers={"benson": SERVER},
            allowed_tools=ALLOWED_TOOL_NAMES,
            permission_mode="bypassPermissions",
            model=_MODEL_FOR_TIER.get(choice.tier, "haiku"),
            max_turns=8,
            session_id=session_id,
            stderr=_stderr_cb,
        )

        # Pull history BEFORE this turn is logged (main.py logs after run_agent returns)
        history = await asyncio.to_thread(_fetch_recent_history, speaker, 12)
        history_block = _format_history(history)

        meta: dict[str, Any] = {
            "session_id": session_id,
            "resumed": False,
            "history_turns": len(history),
            "model": _MODEL_FOR_TIER.get(choice.tier, "haiku"),
            "tool_calls": [],
        }

        text_parts: list[str] = []
        prefixed_user_text = history_block + _channel_prefix(room) + user_text
        try:
            async def _run():
                async for msg in query(prompt=prefixed_user_text, options=options):
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                text_parts.append(block.text)
                            elif isinstance(block, ToolUseBlock):
                                meta["tool_calls"].append(
                                    {"name": block.name, "input": block.input}
                                )
                    elif isinstance(msg, ResultMessage):
                        meta["result_subtype"] = getattr(msg, "subtype", None)
                        meta["duration_ms"] = getattr(msg, "duration_ms", None)
                        meta["num_turns"] = getattr(msg, "num_turns", None)
            await asyncio.wait_for(_run(), timeout=timeout_s)
            response = "\n".join(p for p in text_parts if p).strip() or "(no response)"
            return response, "oauth_" + choice.tier.value, meta
        except Exception as e:
            captured_stderr = "\n".join(stderr_buf[-20:]) if stderr_buf else "(no stderr captured)"
            logger.warning(
                f"OAuth agent failed ({type(e).__name__}: {e}); session={session_id[:8]} "
                f"stderr-tail:\n{captured_stderr}"
            )
            meta["oauth_error"] = f"{type(e).__name__}: {e}"
            meta["stderr_tail"] = captured_stderr
            # Mark the speaker as having hit a failure — the next turn
            # will route up one tier (failure-recency bump in select()).
            _record_tool_failure(speaker)

    # No API fallback — Casey wants everything on OAuth. If the SDK call
    # failed, surface the failure honestly so we can fix the root cause
    # rather than hide it behind a stateless API call.
    return (
        f"I hit a problem on the OAuth path ({meta.get('oauth_error', 'unknown')}). "
        "Try again in a moment, or check `sudo journalctl -u benson.service -n 50` "
        "for the full traceback.",
        "oauth_failed",
        meta,
    )
