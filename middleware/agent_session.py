"""Per-(speaker, room) session store for the Benson agent.

Sessions hold the running message history so multi-turn chats keep
context across separate /conversation POSTs. Each session is a JSON file
on disk; idle sessions older than `IDLE_TIMEOUT_MIN` minutes are pruned
on access.

Concurrency: a single asyncio.Lock per session_id prevents two
overlapping requests from the same speaker/room from interleaving turns.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("benson.session")

SESSIONS_DIR = Path("/opt/benson/middleware/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

IDLE_TIMEOUT_MIN = 30
MAX_TURNS = 12   # rolling window — drop oldest user/assistant pairs beyond this


@dataclass
class Session:
    session_id: str
    speaker: str | None
    room: str | None
    messages: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def path(self) -> Path:
        return SESSIONS_DIR / f"{self.session_id}.json"

    def save(self) -> None:
        self.path.write_text(json.dumps(asdict(self)))

    def trim(self) -> None:
        """Keep at most MAX_TURNS user+assistant pairs, drop older."""
        # tool_use/tool_result blocks live inside assistant/user messages
        # respectively; trim by message count rather than turn pairs.
        cap = MAX_TURNS * 4
        if len(self.messages) > cap:
            self.messages = self.messages[-cap:]


_locks: dict[str, asyncio.Lock] = {}


def _make_id(speaker: str | None, room: str | None = None) -> str:
    """Session key is per-speaker only — Casey's hub, Signal, and voice
    conversations all share one session so context is unified. The `room`
    parameter is accepted for backward compatibility but ignored. The
    channel itself is communicated to the model via a per-message prefix.
    """
    raw = f"{speaker or 'anon'}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def lock_for(session_id: str) -> asyncio.Lock:
    lock = _locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[session_id] = lock
    return lock


def load_or_create(speaker: str | None, room: str | None) -> Session:
    sid = _make_id(speaker, room)
    p = SESSIONS_DIR / f"{sid}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            now = time.time()
            if now - d.get("last_active", 0) > IDLE_TIMEOUT_MIN * 60:
                # idle expired — start fresh
                logger.info(f"session {sid} idle-expired; starting fresh")
                p.unlink(missing_ok=True)
            else:
                return Session(**d)
        except Exception as e:
            logger.warning(f"session {sid} unreadable ({e}); starting fresh")
    return Session(session_id=sid, speaker=speaker, room=room)


def forget(speaker: str | None, room: str | None) -> bool:
    sid = _make_id(speaker, room)
    p = SESSIONS_DIR / f"{sid}.json"
    if p.exists():
        p.unlink()
        return True
    return False


def all_active() -> list[dict[str, Any]]:
    """Return summary of all on-disk sessions for diagnostics."""
    out = []
    now = time.time()
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            out.append({
                "session_id": d.get("session_id"),
                "speaker": d.get("speaker"),
                "room": d.get("room"),
                "messages": len(d.get("messages", [])),
                "idle_minutes": int((now - d.get("last_active", now)) / 60),
            })
        except Exception:
            continue
    return out
