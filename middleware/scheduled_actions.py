"""Persistent, process-independent scheduler for future household actions.

Why this exists: the Claude Code `CronCreate` tool is session-scoped — when
the agent session ends (or the harness restarts), every scheduled job
silently evaporates. On 2026-05-18 at 05:41 Lindsey asked for a 6:20 AM
announcement; the session ended before 6:20 and the announcement never
fired. No error, no fallback, just nothing.

This module gives Benson a place to park any one-time future action so it
survives session restarts and gets dispatched by the long-running
``benson.service`` systemd unit instead of a transient agent session.

Surface:
  - ``ensure_schema()``         — idempotent migration (called at startup).
  - ``schedule(...)``           — insert a pending row, return its id.
  - ``list_actions(...)``       — list pending (or all) rows.
  - ``cancel(id)``              — mark a pending row cancelled.
  - ``start_worker()``          — kick off the asyncio polling loop.

The dispatcher knows how to fire a small whitelist of action types:
``announce``, ``play_music``, ``send_signal``. Adding more is a single
``DISPATCHERS`` entry — but each one is an explicit decision so the
scheduler can never be coerced into running arbitrary tools.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from config import PG_DSN

logger = logging.getLogger("benson.scheduled_actions")

_POLL_INTERVAL_S = 30
_SCHEMA_PATH = Path(__file__).parent / "sql" / "scheduled_actions.sql"


def _conn():
    return psycopg2.connect(**PG_DSN)


# ─── Schema bootstrap ────────────────────────────────────────────────────
def ensure_schema() -> None:
    try:
        sql = _SCHEMA_PATH.read_text()
    except FileNotFoundError:
        logger.error(f"scheduled_actions schema missing at {_SCHEMA_PATH}")
        return
    try:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except Exception:
        logger.exception("scheduled_actions: ensure_schema failed")


# ─── CRUD helpers (sync; tool wrappers run them on a thread) ────────────
def _insert(action_type: str, action_params: dict, fire_at: datetime, created_by: str) -> int:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO scheduled_actions (action_type, action_params, fire_at, created_by) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (action_type, Json(action_params), fire_at, created_by),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return int(new_id)


def _select(include_fired: bool) -> list[dict]:
    sql = (
        "SELECT id, action_type, action_params, fire_at, created_by, "
        "created_at, fired_at, status, last_error FROM scheduled_actions"
    )
    if not include_fired:
        sql += " WHERE fired_at IS NULL"
    sql += " ORDER BY fire_at"
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]


def _cancel(row_id: int) -> bool:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE scheduled_actions SET fired_at = now(), status = 'cancelled' "
            "WHERE id = %s AND fired_at IS NULL",
            (row_id,),
        )
        changed = cur.rowcount
        conn.commit()
        return changed > 0


def _due() -> list[dict]:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, action_type, action_params FROM scheduled_actions "
            "WHERE fired_at IS NULL AND fire_at <= now() ORDER BY fire_at LIMIT 50"
        )
        return [dict(r) for r in cur.fetchall()]


def _mark_fired(row_id: int, status: str, error: str | None = None) -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE scheduled_actions SET fired_at = now(), status = %s, last_error = %s WHERE id = %s",
            (status, error, row_id),
        )
        conn.commit()


# ─── Public async API ───────────────────────────────────────────────────
def _parse_fire_at(fire_at: str | datetime) -> datetime:
    if isinstance(fire_at, datetime):
        return fire_at
    s = str(fire_at).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt


async def schedule(
    action_type: str,
    action_params: dict,
    fire_at: str | datetime,
    created_by: str = "benson",
) -> dict:
    if action_type not in DISPATCHERS:
        return {
            "ok": False,
            "error": f"unknown action_type {action_type!r}; supported: {sorted(DISPATCHERS.keys())}",
        }
    if not isinstance(action_params, dict):
        return {"ok": False, "error": "action_params must be an object"}
    try:
        dt = _parse_fire_at(fire_at)
    except Exception as e:
        return {"ok": False, "error": f"invalid fire_at: {e}"}
    try:
        new_id = await asyncio.to_thread(_insert, action_type, action_params, dt, created_by)
    except Exception as e:
        logger.exception("scheduled_actions: insert failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "id": new_id, "fire_at": dt.isoformat()}


async def list_actions(include_fired: bool = False) -> dict:
    try:
        rows = await asyncio.to_thread(_select, include_fired)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"ok": True, "count": len(rows), "actions": rows}


async def cancel(id: int) -> dict:
    try:
        ok = await asyncio.to_thread(_cancel, int(id))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not ok:
        return {"ok": False, "error": f"id {id} not found or already fired/cancelled"}
    return {"ok": True, "id": int(id)}


# ─── Dispatch table ──────────────────────────────────────────────────────
async def _dispatch_announce(params: dict) -> dict:
    from agent_tools import announce
    return await announce(**params)


async def _dispatch_play_music(params: dict) -> dict:
    from agent_tools import play_music
    return await play_music(**params)


async def _dispatch_send_signal(params: dict) -> dict:
    from agent_tools import send_signal
    return await send_signal(**params)


DISPATCHERS: dict[str, Callable[[dict], Awaitable[dict]]] = {
    "announce": _dispatch_announce,
    "play_music": _dispatch_play_music,
    "send_signal": _dispatch_send_signal,
}


async def _fire_one(row: dict) -> None:
    rid = row["id"]
    atype = row["action_type"]
    params = row["action_params"] or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {}
    dispatcher = DISPATCHERS.get(atype)
    if dispatcher is None:
        await asyncio.to_thread(_mark_fired, rid, "failed", f"no dispatcher for {atype!r}")
        return
    try:
        result = await dispatcher(params)
    except Exception as e:
        logger.exception(f"scheduled_actions: dispatch {atype} id={rid} raised")
        await asyncio.to_thread(_mark_fired, rid, "failed", f"{type(e).__name__}: {e}")
        return
    ok = isinstance(result, dict) and result.get("ok", True) is not False
    status = "fired" if ok else "failed"
    err = None if ok else json.dumps(result, default=str)[:500]
    await asyncio.to_thread(_mark_fired, rid, status, err)
    logger.info(f"scheduled_actions: id={rid} {atype} → {status}" + (f" ({err})" if err else ""))


async def _worker_loop() -> None:
    logger.info(f"scheduled_actions: worker started (poll every {_POLL_INTERVAL_S}s)")
    while True:
        try:
            due_rows = await asyncio.to_thread(_due)
            for row in due_rows:
                await _fire_one(row)
        except Exception:
            logger.exception("scheduled_actions: poll iteration failed")
        await asyncio.sleep(_POLL_INTERVAL_S)


def start_worker() -> None:
    """Start the background polling task. Called from main.py on startup."""
    asyncio.create_task(_worker_loop())
