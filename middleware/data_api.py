"""Data read APIs for the Hub frontend.

Thin wrappers around Postgres queries. Keep these JSON-shaped (not HTML)
so the frontend (and any future iOS/Android client) can consume them
identically. The Hub's HTML routes call these internally.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

import psycopg2
from fastapi import APIRouter, HTTPException, Query, Request
from psycopg2.extras import RealDictCursor

from config import PG_DSN

logger = logging.getLogger("benson.data_api")
router = APIRouter(prefix="/api", tags=["data"])


def _conn():
    return psycopg2.connect(**PG_DSN)


def _query(sql: str, params: tuple = ()) -> list[dict]:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def _query_one(sql: str, params: tuple = ()) -> dict | None:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        r = cur.fetchone()
        return dict(r) if r else None


# ─── Recipes ─────────────────────────────────────────────────────────────
@router.get("/recipes")
async def list_recipes(
    q: str | None = Query(None, description="title substring search"),
    course: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    where = []
    params: list[Any] = []
    if q:
        where.append("title ILIKE %s")
        params.append(f"%{q}%")
    if course:
        where.append("course ILIKE %s")
        params.append(course)
    sql = "SELECT id, title, course, prep_time, image_url, user_rating FROM recipes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY title LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    rows = await asyncio.to_thread(_query, sql, tuple(params))
    return {"recipes": rows, "count": len(rows)}


@router.get("/recipes/{recipe_id}")
async def get_recipe(recipe_id: int) -> dict[str, Any]:
    row = await asyncio.to_thread(
        _query_one, "SELECT * FROM recipes WHERE id = %s", (recipe_id,)
    )
    if not row:
        raise HTTPException(404, "recipe not found")
    return row


# ─── Weekly plan ─────────────────────────────────────────────────────────
@router.get("/weekly-plan")
async def weekly_plan(
    days_ahead: int = Query(7, ge=1, le=30),
) -> dict[str, Any]:
    today = date.today()
    end = today + timedelta(days=days_ahead)
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT wp.plan_date, wp.status, wp.recipe_id,
               r.title, r.course, r.image_url
        FROM weekly_plan wp
        LEFT JOIN recipes r ON r.id = wp.recipe_id
        WHERE wp.plan_date BETWEEN %s AND %s
        ORDER BY wp.plan_date
        """,
        (today, end),
    )
    return {"start": today.isoformat(), "end": end.isoformat(), "plan": rows}


@router.get("/today")
async def today_dashboard() -> dict[str, Any]:
    """Composite endpoint for the home page: today's meal + chore counts."""
    today = date.today()
    meal = await asyncio.to_thread(
        _query_one,
        """
        SELECT wp.plan_date, wp.status, r.id AS recipe_id, r.title,
               r.course, r.image_url, r.source_url
        FROM weekly_plan wp
        LEFT JOIN recipes r ON r.id = wp.recipe_id
        WHERE wp.plan_date = %s
        """,
        (today,),
    )
    chore_summary = await asyncio.to_thread(
        _query,
        """
        SELECT person, count(*) FILTER (WHERE NOT done) AS open,
               count(*) FILTER (WHERE done) AS done_count
        FROM chores
        WHERE chore_date = %s OR chore_date IS NULL
        GROUP BY person
        ORDER BY person
        """,
        (today,),
    )
    return {
        "date": today.isoformat(),
        "meal": meal,
        "chores": chore_summary,
    }


# ─── Chores ──────────────────────────────────────────────────────────────
@router.get("/chores")
async def list_chores(
    person: str | None = None,
    when: str = Query("today", regex="^(today|all|open)$"),
) -> dict[str, Any]:
    today = date.today()
    where = []
    params: list[Any] = []
    if person:
        where.append("LOWER(person) = LOWER(%s)")
        params.append(person)
    if when == "today":
        where.append("(chore_date = %s OR chore_date IS NULL)")
        params.append(today)
    elif when == "open":
        where.append("done = FALSE")
    sql = "SELECT id, person, chore_date, chore_name, done FROM chores"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY done, person, chore_name"
    rows = await asyncio.to_thread(_query, sql, tuple(params))
    return {"chores": rows, "count": len(rows)}


@router.post("/chores/{chore_id}/toggle")
async def toggle_chore(chore_id: int) -> dict[str, Any]:
    row = await asyncio.to_thread(
        _query_one,
        "UPDATE chores SET done = NOT done WHERE id = %s RETURNING id, done",
        (chore_id,),
    )
    if not row:
        raise HTTPException(404, "chore not found")
    return row


@router.post("/chores")
async def create_chore(request: Request) -> dict[str, Any]:
    body = await request.json()
    person = (body.get("person") or "").strip()
    chore_name = (body.get("chore_name") or "").strip()
    chore_date = body.get("chore_date") or None
    if not person or not chore_name:
        raise HTTPException(400, "person and chore_name required")
    row = await asyncio.to_thread(
        _query_one,
        """
        INSERT INTO chores (person, chore_name, chore_date, done)
        VALUES (%s, %s, %s, FALSE)
        RETURNING id, person, chore_name, chore_date, done
        """,
        (person, chore_name, chore_date),
    )
    return row or {}


@router.delete("/chores/{chore_id}")
async def delete_chore(chore_id: int) -> dict[str, Any]:
    row = await asyncio.to_thread(
        _query_one,
        "DELETE FROM chores WHERE id = %s RETURNING id",
        (chore_id,),
    )
    if not row:
        raise HTTPException(404, "chore not found")
    return {"ok": True, "id": chore_id}


@router.post("/chores/{chore_id}/update")
async def update_chore_route(chore_id: int, request: Request) -> dict[str, Any]:
    body = await request.json()
    fields = {}
    for k in ("chore_name", "person", "chore_date", "done"):
        if k in body:
            fields[k] = body[k]
    if not fields:
        raise HTTPException(400, "no fields to update")
    cols = ", ".join(f"{k} = %s" for k in fields)
    params = list(fields.values()) + [chore_id]
    row = await asyncio.to_thread(
        _query_one,
        f"UPDATE chores SET {cols} WHERE id = %s RETURNING id, person, chore_name, chore_date, done",
        tuple(params),
    )
    if not row:
        raise HTTPException(404, "chore not found")
    return row


# ─── Recipe edits, rating, comments, delete ──────────────────────────────
@router.post("/recipes/{recipe_id}/rating")
async def set_recipe_rating(recipe_id: int, request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        rating = float(body.get("rating", 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "rating must be a number 0-5")
    if rating < 0 or rating > 5:
        raise HTTPException(400, "rating out of range")
    row = await asyncio.to_thread(
        _query_one,
        "UPDATE recipes SET user_rating = %s WHERE id = %s RETURNING id, user_rating",
        (rating, recipe_id),
    )
    if not row:
        raise HTTPException(404, "recipe not found")
    return row


@router.post("/recipes/{recipe_id}/update")
async def update_recipe_route(recipe_id: int, request: Request) -> dict[str, Any]:
    from psycopg2.extras import Json
    body = await request.json()
    allowed = {"title", "course", "prep_time", "notes", "user_comments",
               "image_url", "source_url", "ingredients", "steps", "tags"}
    fields: dict = {k: body[k] for k in allowed if k in body}
    if not fields:
        raise HTTPException(400, "no fields to update")
    # JSON-shaped fields
    sql_parts: list[str] = []
    params: list = []
    for k, v in fields.items():
        if k in ("ingredients", "steps", "tags"):
            sql_parts.append(f"{k} = %s")
            params.append(Json(v))
        else:
            sql_parts.append(f"{k} = %s")
            params.append(v)
    params.append(recipe_id)
    row = await asyncio.to_thread(
        _query_one,
        f"UPDATE recipes SET {', '.join(sql_parts)} WHERE id = %s RETURNING *",
        tuple(params),
    )
    if not row:
        raise HTTPException(404, "recipe not found")
    return row


@router.delete("/recipes/{recipe_id}")
async def delete_recipe_route(recipe_id: int) -> dict[str, Any]:
    row = await asyncio.to_thread(
        _query_one,
        "DELETE FROM recipes WHERE id = %s RETURNING id",
        (recipe_id,),
    )
    if not row:
        raise HTTPException(404, "recipe not found")
    return {"ok": True, "id": recipe_id}


# ─── Weekly plan edits ───────────────────────────────────────────────────
@router.post("/weekly/{plan_date}/set")
async def set_weekly_meal(plan_date: str, request: Request) -> dict[str, Any]:
    """Set or clear a meal for a specific date. Body: {"recipe_id": int} or
    {"recipe_id": null} (clears) or {"status": "leftover"}."""
    body = await request.json()
    recipe_id = body.get("recipe_id")
    status = body.get("status") or ("cook" if recipe_id else "leftover")
    if recipe_id in (None, "", "leftover"):
        recipe_id = None
        status = "leftover"
    row = await asyncio.to_thread(
        _query_one,
        """
        INSERT INTO weekly_plan (plan_date, recipe_id, status)
        VALUES (%s, %s, %s)
        ON CONFLICT (plan_date) DO UPDATE SET
            recipe_id = EXCLUDED.recipe_id,
            status = EXCLUDED.status
        RETURNING plan_date, recipe_id, status
        """,
        (plan_date, recipe_id, status),
    )
    return row or {}


# ─── Memory recent ───────────────────────────────────────────────────────
@router.get("/memory/recent")
async def memory_recent(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT id, content, source, speaker, room, importance, created_at
        FROM memories
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return {"memories": rows, "count": len(rows)}


# ─── Conversations log ───────────────────────────────────────────────────
@router.get("/conversations")
async def conversations_recent(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT id, speaker, room, user_text, benson_response, tier, created_at
        FROM conversations
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return {"conversations": rows, "count": len(rows)}


# ─── Calendar (synced from Google) ───────────────────────────────────────
# ─── Message of the Day ──────────────────────────────────────────────────
import json as _json
from pathlib import Path as _Path

_MOTD_DIR = _Path("/tmp/benson-motd")


async def _gather_motd_context() -> dict:
    """Snapshot the household state Benson should reflect on this morning."""
    from datetime import date as _date, timedelta as _td
    today = _date.today()
    ctx: dict[str, Any] = {
        "date": today.isoformat(),
        "day_of_week": today.strftime("%A"),
        "month_day": today.strftime("%B %d"),
    }
    # Weather
    try:
        from ha_client import get_state as _ha_get
        s = await _ha_get("weather.fagley_home")
        a = s.get("attributes", {})
        ctx["weather"] = {
            "condition": s.get("state"),
            "temp": a.get("temperature"),
            "unit": a.get("temperature_unit", "°F"),
        }
    except Exception:
        pass
    # Today's meal
    try:
        meal = await asyncio.to_thread(
            _query_one,
            """
            SELECT r.title, r.course
            FROM weekly_plan wp LEFT JOIN recipes r ON r.id = wp.recipe_id
            WHERE wp.plan_date = %s AND wp.recipe_id IS NOT NULL
            """,
            (today,),
        )
        if meal:
            ctx["meal"] = {"title": meal["title"], "course": meal["course"]}
    except Exception:
        pass
    # Today's calendar events
    try:
        events = await asyncio.to_thread(
            _query,
            """
            SELECT person, title, starts_at, all_day
            FROM calendar_events
            WHERE starts_at::date = %s
              AND COALESCE(status, 'confirmed') != 'cancelled'
            ORDER BY starts_at LIMIT 12
            """,
            (today,),
        )
        ctx["events"] = [
            {
                "person": e["person"],
                "title": e["title"],
                "time": e["starts_at"].isoformat() if e["starts_at"] else None,
                "all_day": e["all_day"],
            } for e in events
        ]
    except Exception:
        ctx["events"] = []
    # Open chores count
    try:
        oc = await asyncio.to_thread(
            _query_one,
            "SELECT COUNT(*) AS n FROM chores WHERE NOT done AND (chore_date = %s OR chore_date IS NULL)",
            (today,),
        )
        ctx["open_chores"] = (oc or {}).get("n", 0)
    except Exception:
        pass
    return ctx


async def _generate_motd() -> dict:
    """Generate today's reflection via Haiku over OAuth (no API charge)."""
    from oauth_oneshot import ask as oauth_ask
    from config import PROMPT_PATH
    base_prompt = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else ""
    ctx = await _gather_motd_context()
    user_msg = (
        "Write today's opening line for the Fagley House Hub home page. "
        "ONE or TWO sentences, total. In your voice — chief-of-staff, "
        "observational, mildly witty (never mean), grounded in the actual "
        "context below. Avoid greetings ('Good morning!'), clichés, and "
        "motivational fluff. No quotes from famous people unless one is "
        "genuinely apt. Don't sign or address anyone — this is read by the "
        "whole household.\n\n"
        f"Context:\n{_json.dumps(ctx, indent=2, default=str)}"
    )
    text = await oauth_ask(user_msg, base_prompt, model="haiku", timeout_s=90)
    return {
        "message": (text or "").strip(),
        "generated_at": ctx["date"],
        "tier": "oauth_haiku",
        "context_summary": {
            "events": len(ctx.get("events", [])),
            "open_chores": ctx.get("open_chores"),
            "meal": (ctx.get("meal") or {}).get("title"),
            "weather": (ctx.get("weather") or {}).get("condition"),
        },
    }


@router.get("/motd")
async def motd_today(refresh: bool = False) -> dict:
    """Today's message — cached per-day. ?refresh=1 forces regeneration."""
    from datetime import date as _date
    today = _date.today().isoformat()
    _MOTD_DIR.mkdir(parents=True, exist_ok=True)
    cache = _MOTD_DIR / f"{today}.json"
    if cache.exists() and not refresh:
        try:
            return _json.loads(cache.read_text())
        except Exception:
            pass
    try:
        m = await _generate_motd()
    except Exception as e:
        return {"message": f"(motd generation failed: {e})", "generated_at": today}
    try:
        cache.write_text(_json.dumps(m))
    except Exception:
        pass
    return m


@router.post("/memory/reindex")
async def memory_reindex() -> dict:
    """Manually trigger a full reindex of the deep memory store.
    The nightly cron runs this automatically at 4am."""
    from memory_index import reindex_all
    return await asyncio.to_thread(reindex_all)


@router.get("/memory/index/stats")
async def memory_index_stats() -> dict:
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT source_type, COUNT(*) AS n,
               MIN(occurred_at) AS oldest,
               MAX(occurred_at) AS newest
        FROM memory_index GROUP BY source_type ORDER BY n DESC
        """,
    )
    total = await asyncio.to_thread(
        _query_one, "SELECT COUNT(*) AS n FROM memory_index"
    )
    return {"total": (total or {}).get("n", 0), "by_source": rows}


@router.get("/calendar/upcoming")
async def calendar_upcoming(days: int = Query(2, ge=1, le=30)) -> dict:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT user_name, google_event_id, calendar_summary, person, title,
               location, starts_at, ends_at, all_day, status
        FROM calendar_events
        WHERE starts_at >= %s AND starts_at < %s
          AND COALESCE(status, 'confirmed') != 'cancelled'
        ORDER BY starts_at, person
        """,
        (now - timedelta(hours=1), end),
    )
    linked = await asyncio.to_thread(
        _query,
        "SELECT user_name, email FROM oauth_tokens WHERE provider = 'google' ORDER BY user_name",
    )

    # Dedupe by (title_lower, starts_at). The same family event often
    # appears across multiple synced calendars/accounts with different
    # `person` derivations — e.g., "Karate" lands on both "Lindsey
    # Personal" (person=Lindsey) and the "Zander" calendar (person=Zander)
    # because Lindsey drives Zander to it. Merge those into one row and
    # carry the union of person tags so the widget can show every
    # affected family member.
    GENERIC = {"Family", "Holiday", "Household", "Unknown"}

    def _best_person(persons: set[str]) -> str:
        specific = {p for p in persons if p and p not in GENERIC}
        if specific:
            return sorted(specific)[0]
        for g in ("Family", "Household", "Holiday", "Unknown"):
            if g in persons:
                return g
        return next(iter(persons), "Unknown")

    groups: dict[tuple, dict] = {}
    for r in rows:
        title = (r["title"] or "").strip()
        starts = r["starts_at"].isoformat() if r["starts_at"] else ""
        key = (title.lower(), starts)
        person = r["person"] or r["user_name"] or "Unknown"
        if key not in groups:
            g = dict(r)
            g["_persons"] = {person}
            g["_calendars"] = {r["calendar_summary"]} if r["calendar_summary"] else set()
            groups[key] = g
        else:
            groups[key]["_persons"].add(person)
            if r["calendar_summary"]:
                groups[key]["_calendars"].add(r["calendar_summary"])

    deduped = sorted(groups.values(), key=lambda g: (g["starts_at"] or now))

    # Strip generic tags when at least one specific person is present.
    for g in deduped:
        persons = g["_persons"]
        specific = {p for p in persons if p not in GENERIC}
        g["_display_persons"] = sorted(specific) if specific else sorted(persons)
        g["_best_person"] = _best_person(persons)

    return {
        "events": [
            {
                "user_name": r["user_name"],
                "person": r["_best_person"],
                "persons": r["_display_persons"],
                "calendar": r["calendar_summary"],
                "calendars": sorted(r["_calendars"]),
                "title": r["title"],
                "location": r["location"],
                "starts_at": r["starts_at"].isoformat() if r["starts_at"] else None,
                "ends_at": r["ends_at"].isoformat() if r["ends_at"] else None,
                "all_day": r["all_day"],
            }
            for r in deduped
        ],
        "raw_count": len(rows),
        "deduped_count": len(deduped),
        "linked_users": [r["user_name"] for r in linked],
    }


# ─── Weather (current + 5-day forecast) ──────────────────────────────────
@router.get("/weather")
async def weather_now(days: int = 5) -> dict:
    """Live weather from HA's weather.fagley_home + 5-day daily forecast."""
    from ha_client import get_state as ha_get_state, call_service as ha_call
    try:
        s = await ha_get_state("weather.fagley_home")
    except Exception as e:
        raise HTTPException(503, f"weather entity unavailable: {e}")
    a = s.get("attributes", {})
    current = {
        "condition": s.get("state"),
        "temperature": a.get("temperature"),
        "temperature_unit": a.get("temperature_unit"),
        "humidity": a.get("humidity"),
        "wind_speed": a.get("wind_speed"),
        "wind_speed_unit": a.get("wind_speed_unit"),
        "pressure": a.get("pressure"),
    }
    forecast: list[dict] = []
    try:
        result = await ha_call(
            "weather", "get_forecasts",
            {"entity_id": "weather.fagley_home", "type": "daily"},
            timeout_s=10, return_response=True,
        )
        sr = (result or {}).get("service_response") or {}
        days_data = sr.get("weather.fagley_home", {}).get("forecast", [])
        forecast = days_data[:days]
    except Exception:
        pass
    return {"current": current, "forecast": forecast}


# ─── Music (via Music Assistant + HA) ────────────────────────────────────
_MASS_ZONES = {
    "kitchen": "media_player.kitchen_2",
    "family_room": "media_player.family_room_2",
    "tv_room": "media_player.tv_room_2",
    "master_bedroom": "media_player.bathroom_2",
    "move": "media_player.move_2",
}


async def _ma_entry_id() -> str | None:
    from ha_client import _headers
    from config import HA_BASE_URL
    import httpx
    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(
            f"{HA_BASE_URL}/api/config/config_entries/entry",
            headers=_headers(),
        )
    for e in resp.json():
        if e.get("domain") == "music_assistant":
            return e.get("entry_id")
    return None


def _sonos_counterpart(ma_entity: str) -> str:
    """MA-adopted entities end in `_2`; Sonos integration entities don't."""
    return ma_entity[:-2] if ma_entity.endswith("_2") else ma_entity


@router.get("/music/players")
async def music_players() -> dict:
    """Return the MA-controlled zones with state + Sonos group membership."""
    from ha_client import get_state as ha_get_state
    out = []
    for room_id, entity in _MASS_ZONES.items():
        try:
            s = await ha_get_state(entity)
            attrs = s.get("attributes", {})
            sonos_eid = _sonos_counterpart(entity)
            sonos_group: list[str] = []
            try:
                sonos_state = await ha_get_state(sonos_eid)
                sonos_group = sonos_state.get("attributes", {}).get("group_members", []) or []
            except Exception:
                pass
            out.append({
                "room": room_id,
                "entity_id": entity,
                "sonos_entity_id": sonos_eid,
                "label": attrs.get("friendly_name", entity).rstrip(" 2").strip(),
                "state": s.get("state"),
                "volume": attrs.get("volume_level"),
                "muted": attrs.get("is_volume_muted"),
                "media_title": attrs.get("media_title"),
                "media_artist": attrs.get("media_artist"),
                "media_album_name": attrs.get("media_album_name"),
                "entity_picture": attrs.get("entity_picture"),
                "media_position": attrs.get("media_position"),
                "media_duration": attrs.get("media_duration"),
                "group_members": sonos_group,  # list of Sonos entity_ids in this group
                "is_group_leader": bool(sonos_group) and sonos_group[0] == sonos_eid,
            })
        except Exception:
            out.append({"room": room_id, "entity_id": entity, "state": "unavailable"})
    return {"players": out}


@router.post("/music/group")
async def music_group(request: Request) -> dict:
    """Group Sonos zones together. Body: {primary: <ma_entity>, members: [<ma_entity>, ...]}.
    The primary becomes the group leader; members join its stream."""
    body = await request.json()
    primary = body.get("primary")
    members = body.get("members") or []
    if not primary or not members:
        raise HTTPException(400, "primary and non-empty members required")
    from ha_client import call_service as ha_call
    sonos_primary = _sonos_counterpart(primary)
    sonos_members = [_sonos_counterpart(m) for m in members]
    try:
        await ha_call(
            "media_player", "join",
            {"entity_id": sonos_primary, "group_members": sonos_members},
            timeout_s=10,
        )
    except Exception as e:
        raise HTTPException(502, f"join failed: {e}")
    return {"ok": True, "primary": sonos_primary, "members": sonos_members}


@router.post("/music/ungroup")
async def music_ungroup(request: Request) -> dict:
    """Remove a single zone from its current group. Body: {entity_id: <ma_or_sonos>}."""
    body = await request.json()
    entity_id = body.get("entity_id")
    if not entity_id:
        raise HTTPException(400, "entity_id required")
    sonos = _sonos_counterpart(entity_id)
    from ha_client import call_service as ha_call
    try:
        await ha_call("media_player", "unjoin", {"entity_id": sonos}, timeout_s=10)
    except Exception as e:
        raise HTTPException(502, f"unjoin failed: {e}")
    return {"ok": True, "entity_id": sonos}


@router.get("/music/playlists")
async def music_playlists(
    favorite: bool = False, limit: int = 100
) -> dict:
    from ha_client import call_service as ha_call
    cfg_id = await _ma_entry_id()
    if not cfg_id:
        raise HTTPException(503, "Music Assistant not configured")
    try:
        result = await ha_call(
            "music_assistant", "get_library",
            {"config_entry_id": cfg_id, "media_type": "playlist",
             "favorite": favorite, "limit": limit, "offset": 0},
            timeout_s=20, return_response=True,
        )
    except Exception as e:
        raise HTTPException(502, f"MA get_library failed: {e}")
    sr = (result or {}).get("service_response", {}) or {}
    items = sr.get("items", []) if isinstance(sr, dict) else []
    return {"playlists": items, "count": len(items)}


@router.post("/music/search")
async def music_search(request: Request) -> dict:
    from ha_client import call_service as ha_call
    body = await request.json()
    query = (body.get("query") or "").strip()
    media_type = body.get("media_type") or "playlist"  # playlist, album, artist, track
    limit = int(body.get("limit") or 8)
    if not query:
        raise HTTPException(400, "query required")
    cfg_id = await _ma_entry_id()
    if not cfg_id:
        raise HTTPException(503, "Music Assistant not configured")
    try:
        result = await ha_call(
            "music_assistant", "search",
            {
                "config_entry_id": cfg_id,
                "name": query,
                "media_type": [media_type] if isinstance(media_type, str) else media_type,
                "limit": limit,
                "library_only": False,
            },
            timeout_s=20, return_response=True,
        )
    except Exception as e:
        raise HTTPException(502, f"MA search failed: {e}")
    sr = (result or {}).get("service_response", {}) or {}
    return {"results": sr, "query": query, "media_type": media_type}


@router.post("/music/play")
async def music_play(request: Request) -> dict:
    from ha_client import call_service as ha_call
    body = await request.json()
    entity = body.get("entity_id")
    if not entity:
        room = body.get("room")
        entity = _MASS_ZONES.get(room) if room else None
    if not entity:
        raise HTTPException(400, "entity_id or known room required")
    media_id = (body.get("media_id") or body.get("uri") or body.get("query") or "").strip()
    if not media_id:
        raise HTTPException(400, "media_id (uri or query) required")
    media_type = body.get("media_type") or "playlist"
    enqueue = body.get("enqueue") or "replace"
    radio_mode = bool(body.get("radio_mode") or False)
    try:
        await ha_call(
            "music_assistant", "play_media",
            {
                "entity_id": entity,
                "media_id": media_id,
                "media_type": media_type,
                "enqueue": enqueue,
                "radio_mode": radio_mode,
            },
            timeout_s=30,
        )
    except Exception as e:
        raise HTTPException(502, f"MA play failed: {e}")
    return {"ok": True, "entity_id": entity, "media_id": media_id, "media_type": media_type}


@router.post("/music/control")
async def music_control(request: Request) -> dict:
    """play/pause/stop/next/previous/volume on a Sonos zone."""
    from ha_client import call_service as ha_call
    body = await request.json()
    entity = body.get("entity_id") or _MASS_ZONES.get(body.get("room", ""))
    if not entity:
        raise HTTPException(400, "entity_id or known room required")
    action = body.get("action")
    svc_map = {
        "play": "media_play", "pause": "media_pause", "stop": "media_stop",
        "next": "media_next_track", "previous": "media_previous_track",
    }
    data: dict = {"entity_id": entity}
    if action == "volume":
        level = float(body.get("level", 0.4))
        await ha_call("media_player", "volume_set",
                      {**data, "volume_level": max(0.0, min(1.0, level))})
        return {"ok": True, "entity_id": entity, "volume": level}
    if action not in svc_map:
        raise HTTPException(400, f"unknown action: {action}")
    await ha_call("media_player", svc_map[action], data)
    return {"ok": True, "entity_id": entity, "action": action}


# ─── Status ──────────────────────────────────────────────────────────────
@router.get("/status")
async def system_status() -> dict[str, Any]:
    """Quick check of what's live and what's awaiting Casey's setup."""
    import os
    rec_count = (await asyncio.to_thread(_query_one, "SELECT count(*) FROM recipes"))["count"]
    chore_open = (await asyncio.to_thread(
        _query_one, "SELECT count(*) FROM chores WHERE NOT done"
    ))["count"]
    mem_count = (await asyncio.to_thread(_query_one, "SELECT count(*) FROM memories"))["count"]
    return {
        "db": {
            "recipes": rec_count,
            "open_chores": chore_open,
            "memories": mem_count,
        },
        "secrets": {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "ha_token": bool(os.environ.get("HA_LONG_LIVED_TOKEN")),
            "bond_token": bool(os.environ.get("BOND_BRIDGE_TOKEN")),
            "telegram_token": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
            "instacart": bool(os.environ.get("INSTACART_API_KEY")),
        },
    }
