"""Tool registry exposed to the Benson agent.

Each tool is a Python coroutine plus a JSON-schema definition. The agent
loop in `agent.py` dispatches tool_use blocks here. Tool descriptions are
read by the model — keep them clear and use-case-oriented (not API-doc
flavoured).

Tools wrap existing modules (`ha_client`, `db_tools`, `memory`, `recipes`)
so the agent never sees raw SQL / HA HTTP details — it sees domain
operations.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any, Awaitable, Callable

import psycopg2
from psycopg2.extras import RealDictCursor

from config import PG_DSN
from ha_client import call_service as ha_call_service, get_state as ha_get_state
from memory import MemoryStore

logger = logging.getLogger("benson.agent_tools")

_memory = MemoryStore()


def _conn():
    return psycopg2.connect(**PG_DSN)


def _query(sql: str, params: tuple = ()) -> list[dict]:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _query_one(sql: str, params: tuple = ()) -> dict | None:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        r = cur.fetchone()
        return dict(r) if r else None


# ─── Tool registry ────────────────────────────────────────────────────────
# `TOOLS` is the list of {name, description, input_schema} blocks sent to
# the API. `IMPL` maps name → async impl. Add a tool by adding an entry to
# both.

TOOLS: list[dict] = []
IMPL: dict[str, Callable[..., Awaitable[Any]]] = {}


def _register(
    name: str,
    description: str,
    input_schema: dict,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    def deco(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        TOOLS.append(
            {"name": name, "description": description, "input_schema": input_schema}
        )
        IMPL[name] = fn
        return fn

    return deco


# ─── Device control ──────────────────────────────────────────────────────
@_register(
    "control_light",
    "Turn a light on or off, or set its brightness as a percentage. "
    "Use this for any of the four ceiling-fan lights "
    "(light.master_bedroom_fan, light.office_fan, light.zanders_room_fan, "
    "light.kitchen_fan).",
    {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "HA entity ID, e.g. 'light.master_bedroom_fan'.",
            },
            "action": {
                "type": "string",
                "enum": ["on", "off", "set_brightness"],
                "description": "What to do.",
            },
            "brightness_pct": {
                "type": ["integer", "string"],
                "minimum": 1,
                "maximum": 100,
                "description": "Required when action='on' or 'set_brightness'. 100 = full.",
            },
        },
        "required": ["entity_id", "action"],
    },
)
async def control_light(entity_id: str, action: str, brightness_pct: int | None = None) -> dict:
    if action == "off":
        await ha_call_service("light", "turn_off", {"entity_id": entity_id})
        return {"ok": True, "entity_id": entity_id, "state": "off"}
    pct = brightness_pct if brightness_pct is not None else 100
    pct = max(1, min(100, pct))
    await ha_call_service(
        "light", "turn_on", {"entity_id": entity_id, "brightness_pct": pct}
    )
    return {"ok": True, "entity_id": entity_id, "state": "on", "brightness_pct": pct}


@_register(
    "control_fan",
    "Turn a ceiling fan on/off or set its speed. Speeds: 'low' (33%), "
    "'medium' (66%), 'high' (100%). Available fans: fan.master_bedroom_fan, "
    "fan.office_fan, fan.zanders_room_fan, fan.kitchen_fan.",
    {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string"},
            "action": {"type": "string", "enum": ["on", "off", "set_speed", "reverse"]},
            "speed": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["entity_id", "action"],
    },
)
async def control_fan(entity_id: str, action: str, speed: str | None = None) -> dict:
    if action == "off":
        await ha_call_service("fan", "turn_off", {"entity_id": entity_id})
        return {"ok": True, "entity_id": entity_id, "state": "off"}
    if action == "reverse":
        # Toggle direction
        st = await ha_get_state(entity_id)
        cur = (st.get("attributes", {}) or {}).get("direction", "forward")
        new = "reverse" if cur == "forward" else "forward"
        await ha_call_service(
            "fan", "set_direction", {"entity_id": entity_id, "direction": new}
        )
        return {"ok": True, "entity_id": entity_id, "direction": new}
    pct_map = {"low": 33, "medium": 66, "high": 100}
    pct = pct_map.get(speed or "medium", 66) if action == "set_speed" else 66
    if action == "on":
        await ha_call_service("fan", "turn_on", {"entity_id": entity_id})
        return {"ok": True, "entity_id": entity_id, "state": "on"}
    await ha_call_service(
        "fan", "set_percentage", {"entity_id": entity_id, "percentage": pct}
    )
    return {"ok": True, "entity_id": entity_id, "speed": speed, "percentage": pct}


@_register(
    "control_shades",
    "Open or close the upstairs living-room shades (cover.upstairs_shades). "
    "These cover the main door upstairs.",
    {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["open", "close"]}},
        "required": ["action"],
    },
)
async def control_shades(action: str) -> dict:
    svc = "open_cover" if action == "open" else "close_cover"
    await ha_call_service("cover", svc, {"entity_id": "cover.upstairs_shades"})
    return {"ok": True, "action": action}


@_register(
    "control_waterfall",
    "Turn the outdoor waterfall on or off. Note: requires Bond remote "
    "codes to be trained — may fail if Bond doesn't have the codes yet.",
    {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["on", "off"]}},
        "required": ["action"],
    },
)
async def control_waterfall(action: str) -> dict:
    svc = "turn_on" if action == "on" else "turn_off"
    try:
        await ha_call_service("switch", svc, {"entity_id": "switch.waterfall"})
        return {"ok": True, "action": action}
    except Exception as e:
        return {"ok": False, "error": str(e), "hint": "Bond may not have learned the waterfall remote codes yet."}


# ─── Media (Sonos) ───────────────────────────────────────────────────────
@_register(
    "control_media",
    "Play / pause / resume / skip media on a Sonos zone. Zones: "
    "media_player.kitchen, media_player.family_room, media_player.tv_room, "
    "media_player.bathroom (this one is physically in the master bedroom), "
    "media_player.move (portable, often on the patio).",
    {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["play", "pause", "stop", "next", "previous"],
            },
        },
        "required": ["entity_id", "action"],
    },
)
async def control_media(entity_id: str, action: str) -> dict:
    svc_map = {
        "play": "media_play", "pause": "media_pause", "stop": "media_stop",
        "next": "media_next_track", "previous": "media_previous_track",
    }
    await ha_call_service("media_player", svc_map[action], {"entity_id": entity_id})
    return {"ok": True, "entity_id": entity_id, "action": action}


# ─── Music Assistant (Apple Music) ───────────────────────────────────────
# Each Sonos zone has BOTH a native HA Sonos entity (media_player.<room>)
# and a Music Assistant entity (media_player.<room>_2). MA-side is the
# only one that can play Apple Music / search by name / browse playlists.
MASS_ZONES = {
    "kitchen":         "media_player.kitchen_2",
    "family_room":     "media_player.family_room_2",
    "tv_room":         "media_player.tv_room_2",
    "master_bedroom":  "media_player.bathroom_2",
    "bathroom":        "media_player.bathroom_2",
    "coles_room":      "media_player.family_room_2",
    "zanders_room":    "media_player.family_room_2",
    "office":          "media_player.kitchen_2",
    "patio":           "media_player.move_2",
    "deck":            "media_player.move_2",
    "outdoor":         "media_player.move_2",
}


def _resolve_mass_zone(room: str | None) -> str | None:
    if not room:
        return None
    norm = room.lower().replace(" ", "_").replace("-", "_").replace("'", "").strip()
    if norm in MASS_ZONES:
        return MASS_ZONES[norm]
    for k, v in MASS_ZONES.items():
        if k in norm:
            return v
    return None


@_register(
    "play_music",
    "Play Apple Music content (playlist / album / artist / song) on a "
    "Sonos zone via Music Assistant. `query` is the title/name to search "
    "(e.g., 'dinner jazz', 'Miles Davis', 'Kind of Blue'). `room` is one "
    "of: kitchen, family_room, tv_room, master_bedroom, coles_room, "
    "zanders_room, office, patio. `content_type` defaults to playlist; "
    "use 'album', 'artist', or 'track' for those.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "room": {"type": "string"},
            "content_type": {
                "type": "string",
                "enum": ["playlist", "album", "artist", "track"],
                "default": "playlist",
            },
        },
        "required": ["query", "room"],
    },
)
async def play_music(query: str, room: str, content_type: str = "playlist") -> dict:
    entity = _resolve_mass_zone(room)
    if not entity:
        return {"ok": False, "error": f"unknown room: {room}"}
    # MA accepts a free-text media_id and resolves it as the given media_type
    try:
        await ha_call_service(
            "music_assistant",
            "play_media",
            {
                "entity_id": entity,
                "media_id": query,
                "media_type": content_type,
                "enqueue": "replace",
                "radio_mode": False,
            },
            timeout_s=30,
        )
    except Exception as e:
        return {"ok": False, "error": f"music_assistant.play_media failed: {e}"}
    return {
        "ok": True, "room": room, "zone": entity,
        "query": query, "content_type": content_type,
    }


@_register(
    "list_playlists",
    "Return the user's Apple Music / Music Assistant playlists by name. "
    "Useful when the user asks 'what playlists do I have' or to "
    "disambiguate fuzzy play_music requests.",
    {
        "type": "object",
        "properties": {
            "limit": {"type": ["integer", "string"], "default": 50, "minimum": 1, "maximum": 200},
            "favorite": {"type": "boolean", "default": False},
        },
    },
)
async def _ma_config_entry_id() -> str | None:
    """Look up the music_assistant config_entry_id from HA."""
    import httpx
    from config import HA_BASE_URL, HA_TOKEN
    if not HA_TOKEN:
        return None
    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(
            f"{HA_BASE_URL}/api/config/config_entries/entry",
            headers={"Authorization": f"Bearer {HA_TOKEN}"},
        )
        for e in resp.json():
            if e.get("domain") == "music_assistant":
                return e.get("entry_id")
    return None


async def list_playlists(limit: int = 50, favorite: bool = False) -> dict:
    cfg_id = await _ma_config_entry_id()
    if not cfg_id:
        return {"ok": False, "error": "Music Assistant config entry not found in HA"}
    try:
        result = await ha_call_service(
            "music_assistant",
            "get_library",
            {
                "config_entry_id": cfg_id,
                "media_type": "playlist",
                "favorite": favorite,
                "limit": limit,
                "offset": 0,
            },
            timeout_s=20,
            return_response=True,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    sr = result.get("service_response", {}) if isinstance(result, dict) else {}
    items = sr.get("items", []) if isinstance(sr, dict) else []
    out = []
    for it in items:
        if isinstance(it, dict):
            uri = it.get("uri", "")
            provider = uri.split("://")[0] if "://" in uri else (it.get("provider") or "")
            out.append({
                "name": it.get("name") or "(untitled)",
                "uri": uri,
                "provider": provider,
                "favorite": it.get("favorite", False),
            })
    return {"ok": True, "count": len(out), "playlists": out}


@_register(
    "stop_music",
    "Stop / pause music playback in a room (the MA-controlled side). "
    "Use this when the user says 'stop the music in X' or 'pause in X'.",
    {
        "type": "object",
        "properties": {
            "room": {"type": "string"},
            "action": {"type": "string", "enum": ["pause", "stop"], "default": "pause"},
        },
        "required": ["room"],
    },
)
async def stop_music(room: str, action: str = "pause") -> dict:
    entity = _resolve_mass_zone(room)
    if not entity:
        return {"ok": False, "error": f"unknown room: {room}"}
    svc = "media_pause" if action == "pause" else "media_stop"
    await ha_call_service("media_player", svc, {"entity_id": entity})
    return {"ok": True, "room": room, "zone": entity, "action": action}


@_register(
    "set_volume",
    "Set Sonos zone volume (0.0-1.0).",
    {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string"},
            "level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["entity_id", "level"],
    },
)
async def set_volume(entity_id: str, level: float) -> dict:
    level = max(0.0, min(1.0, float(level)))
    await ha_call_service(
        "media_player", "volume_set", {"entity_id": entity_id, "volume_level": level}
    )
    return {"ok": True, "entity_id": entity_id, "volume": level}


@_register(
    "announce",
    "Speak a short message through a Sonos zone using Piper TTS. The "
    "message should be 1-3 sentences of plain spoken text — no markdown, "
    "no stage directions, no asterisks. Use this whenever the household "
    "should hear something out loud. To play the same announcement in "
    "sync on multiple Sonos zones, pass the additional zone entity ids "
    "in `also_play_on` — the function will temporarily group them under "
    "`zone_entity_id`, play once, then ungroup. Prefer this over "
    "per-zone announce loops for 'all speakers / everywhere / whole "
    "house' requests.",
    {
        "type": "object",
        "properties": {
            "zone_entity_id": {
                "type": "string",
                "description": "media_player.* zone to speak through.",
            },
            "message": {
                "type": "string",
                "description": "Plain spoken text. 1-3 sentences max.",
            },
            "also_play_on": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional. Additional media_player.* zone entity ids to "
                    "play this announcement on in lockstep with "
                    "`zone_entity_id`. The zones are temporarily joined "
                    "under `zone_entity_id` (the coordinator), the message "
                    "plays once, then they are ungrouped."
                ),
            },
        },
        "required": ["zone_entity_id", "message"],
    },
)
async def announce(
    zone_entity_id: str,
    message: str,
    also_play_on: list[str] | None = None,
) -> dict:
    from kokoro_tts import speak_on_zone

    if not also_play_on:
        return await speak_on_zone(zone_entity_id, message)

    # Multi-zone synced announcement: join → play once → unjoin.
    await ha_call_service(
        "media_player",
        "join",
        {"entity_id": zone_entity_id, "group_members": also_play_on},
    )
    try:
        await asyncio.sleep(0.7)  # let the group settle before playback
        result = await speak_on_zone(zone_entity_id, message)
    finally:
        await asyncio.sleep(0.3)
        try:
            await ha_call_service(
                "media_player",
                "unjoin",
                {"entity_id": [zone_entity_id, *also_play_on]},
            )
        except Exception:
            logger.exception("unjoin failed after grouped announce")

    if isinstance(result, dict):
        result = {**result, "grouped_with": list(also_play_on)}
    return result


@_register(
    "group_sonos",
    "Group Sonos speakers so they play in sync. The coordinator is the "
    "speaker that drives playback; members mirror it bit-perfect. Use "
    "this BEFORE announce/play_music when the user wants 'all speakers "
    "/ everywhere / the whole house' to play synced. Pair with "
    "ungroup_sonos afterward to release the group.",
    {
        "type": "object",
        "properties": {
            "coordinator": {
                "type": "string",
                "description": (
                    "HA media_player entity id that will drive playback, "
                    "e.g. 'media_player.kitchen'."
                ),
            },
            "members": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Entity ids to join under the coordinator.",
            },
        },
        "required": ["coordinator", "members"],
        "additionalProperties": False,
    },
)
async def group_sonos(coordinator: str, members: list[str]) -> dict:
    await ha_call_service(
        "media_player",
        "join",
        {"entity_id": coordinator, "group_members": members},
    )
    return {"ok": True, "coordinator": coordinator, "members": members}


@_register(
    "ungroup_sonos",
    "Release Sonos speakers from a temporary group, returning each to "
    "independent playback. Call this after a synced announcement or "
    "grouped playback finishes.",
    {
        "type": "object",
        "properties": {
            "members": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Entity ids to detach from their current group.",
            },
        },
        "required": ["members"],
        "additionalProperties": False,
    },
)
async def ungroup_sonos(members: list[str]) -> dict:
    await ha_call_service("media_player", "unjoin", {"entity_id": members})
    return {"ok": True, "members": members}


# ─── Data lookups ────────────────────────────────────────────────────────
@_register(
    "search_recipes",
    "Search the family recipe database by title substring and optional "
    "course. Returns up to `limit` recipes with id, title, course, and "
    "prep_time. The DB has 58 recipes carried over from the prior "
    "Fagley home server.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Title substring; '' = all."},
            "course": {
                "type": "string",
                "description": "Optional: 'Main', 'Sauce', 'Other', etc.",
            },
            "limit": {"type": ["integer", "string"], "default": 10, "minimum": 1, "maximum": 30},
        },
        "required": ["query"],
    },
)
async def search_recipes(query: str, course: str | None = None, limit: int = 10) -> dict:
    where = []
    params: list = []
    if query:
        where.append("title ILIKE %s")
        params.append(f"%{query}%")
    if course:
        where.append("course ILIKE %s")
        params.append(course)
    sql = "SELECT id, title, course, prep_time, image_url FROM recipes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY title LIMIT %s"
    params.append(limit)
    rows = await asyncio.to_thread(_query, sql, tuple(params))
    return {"recipes": rows, "count": len(rows)}


@_register(
    "get_recipe",
    "Fetch full ingredients and steps for one recipe by id.",
    {
        "type": "object",
        "properties": {"id": {"type": ["integer", "string"]}},
        "required": ["id"],
    },
)
async def get_recipe(id: int) -> dict:
    row = await asyncio.to_thread(
        _query_one,
        "SELECT id, title, course, prep_time, ingredients, steps, "
        "tags, source_url, notes FROM recipes WHERE id = %s",
        (id,),
    )
    if not row:
        return {"error": f"recipe {id} not found"}
    return row


@_register(
    "lookup_chores",
    "Look up the chores table. Filter by person ('Cole' / 'Zander' / "
    "'General') and/or `when` ('today' / 'open' / 'all'). Returns "
    "rewards (dollars for Cole, points for Zander) so when a kid asks "
    "what they have to do, you can also tell them what each chore "
    "pays. ALWAYS surface the price when reporting Cole's chores: "
    "'walk Bluey ($2.50), trash cans ($1)'.",
    {
        "type": "object",
        "properties": {
            "person": {"type": "string"},
            "when": {"type": "string", "enum": ["today", "open", "all"], "default": "today"},
        },
    },
)
async def lookup_chores(person: str | None = None, when: str = "today") -> dict:
    today = date.today()
    where = []
    params: list = []
    if person:
        where.append("LOWER(person) = LOWER(%s)")
        params.append(person)
    if when == "today":
        where.append(
            "(chore_date = %s OR chore_date IS NULL "
            " OR (chore_date < %s AND done = FALSE))"
        )
        params.extend([today, today])
    elif when == "open":
        where.append("done = FALSE")
    sql = (
        "SELECT id, person, chore_date, chore_name, done, recurring, "
        "dollars, points FROM chores"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY done, person, chore_name LIMIT 100"
    rows = await asyncio.to_thread(_query, sql, tuple(params))
    return {"chores": rows, "count": len(rows), "today": today.isoformat()}


@_register(
    "get_weekly_plan",
    "Return the weekly meal plan for the next N days.",
    {
        "type": "object",
        "properties": {
            "days_ahead": {"type": ["integer", "string"], "default": 7, "minimum": 1, "maximum": 30}
        },
    },
)
async def get_weekly_plan(days_ahead: int = 7) -> dict:
    today = date.today()
    end = today + timedelta(days=days_ahead)
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT wp.plan_date, wp.status, wp.recipe_id, r.title, r.course
        FROM weekly_plan wp
        LEFT JOIN recipes r ON r.id = wp.recipe_id
        WHERE wp.plan_date BETWEEN %s AND %s
        ORDER BY wp.plan_date
        """,
        (today, end),
    )
    return {"plan": rows, "start": today.isoformat(), "end": end.isoformat()}


@_register(
    "get_weather",
    "Current Colorado Springs weather pulled live from Home Assistant's "
    "Open-Meteo integration. Returns condition, temperature, humidity, "
    "wind, pressure.",
    {"type": "object", "properties": {}},
)
async def get_weather() -> dict:
    try:
        s = await ha_get_state("weather.fagley_home")
    except Exception as e:
        return {"error": f"weather unavailable: {e}"}
    a = s.get("attributes", {})
    return {
        "condition": s.get("state"),
        "temperature": a.get("temperature"),
        "temperature_unit": a.get("temperature_unit"),
        "humidity": a.get("humidity"),
        "wind_speed": a.get("wind_speed"),
        "wind_speed_unit": a.get("wind_speed_unit"),
        "pressure": a.get("pressure"),
    }


# ─── Memory ──────────────────────────────────────────────────────────────
@_register(
    "search_memory",
    "Semantic search over Benson's long-term household memory. Returns "
    "memories ranked by relevance to the query. Use this whenever the "
    "user asks about household preferences, history, names, routines, "
    "or anything 'do you remember...'-shaped.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": ["integer", "string"], "default": 8, "minimum": 1, "maximum": 30},
        },
        "required": ["query"],
    },
)
async def search_memory(query: str, limit: int = 8) -> dict:
    results = await _memory.search(query, limit=limit)
    return {"memories": results, "count": len(results)}


@_register(
    "remember_this",
    "Store a durable household fact in long-term memory. Use ONLY for "
    "things worth remembering across sessions: preferences, allergies, "
    "routines, family-history facts, recurring schedules. SKIP generic "
    "small talk, polite filler, or one-off events.",
    {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "The fact, in one complete sentence."},
            "speaker": {"type": "string", "description": "Who told you (optional)."},
            "importance": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.7,
            },
        },
        "required": ["fact"],
    },
)
async def remember_this(
    fact: str, speaker: str | None = None, importance: float = 0.7
) -> dict:
    new_id = await _memory.store(
        fact, speaker=speaker, source="agent_explicit", importance=importance
    )
    return {"ok": True, "id": new_id, "stored": fact}


# ─── Write helpers ───────────────────────────────────────────────────────
def _write(sql: str, params: tuple = ()) -> int:
    """Execute a write and return rowcount."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rc = cur.rowcount
        conn.commit()
        return rc


def _write_returning(sql: str, params: tuple = ()) -> dict | None:
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else None


# ─── Recipe writes ───────────────────────────────────────────────────────
@_register(
    "add_recipe",
    "Manually add a new recipe to the family database. Use this when "
    "the user dictates a recipe (no photo / no URL). For photo or video "
    "ingestion, those go through dedicated endpoints, not this tool.",
    {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "ingredients": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of ingredient strings (one per line/item).",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered cooking steps.",
            },
            "course": {"type": "string", "description": "Main / Side / Sauce / Other"},
            "prep_time": {"type": ["integer", "string"], "description": "Total minutes."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
            "source_url": {"type": "string"},
            "image_url": {"type": "string"},
        },
        "required": ["title"],
    },
)
async def add_recipe(
    title: str,
    ingredients: list[str] | None = None,
    steps: list[str] | None = None,
    course: str | None = None,
    prep_time: int | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    source_url: str | None = None,
    image_url: str | None = None,
) -> dict:
    from psycopg2.extras import Json
    ing_jsonb = [{"text": s} for s in (ingredients or [])]
    row = await asyncio.to_thread(
        _write_returning,
        """
        INSERT INTO recipes
            (title, source, source_url, ingredients, steps, tags,
             image_url, course, prep_time, notes)
        VALUES (%s, 'manual', %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id, title, course
        """,
        (
            title,
            source_url,
            Json(ing_jsonb),
            Json(steps or []),
            Json(tags or []),
            image_url,
            course,
            prep_time,
            notes,
        ),
    )
    return {"ok": True, "recipe": row}


@_register(
    "update_recipe",
    "Update fields of an existing recipe by id. Pass only the fields "
    "to change. Set a field to null to clear it.",
    {
        "type": "object",
        "properties": {
            "id": {"type": ["integer", "string"]},
            "title": {"type": "string"},
            "course": {"type": "string"},
            "prep_time": {"type": ["integer", "string"]},
            "notes": {"type": "string"},
            "user_rating": {"type": "number", "minimum": 0, "maximum": 5},
            "user_comments": {"type": "string"},
            "image_url": {"type": "string"},
            "source_url": {"type": "string"},
        },
        "required": ["id"],
    },
)
async def update_recipe(id: int, **fields) -> dict:
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        return {"ok": False, "error": "no fields to update"}
    cols = ", ".join(f"{k} = %s" for k in fields)
    params = list(fields.values()) + [id]
    rc = await asyncio.to_thread(
        _write, f"UPDATE recipes SET {cols} WHERE id = %s", tuple(params)
    )
    return {"ok": rc > 0, "id": id, "rows": rc, "updated": list(fields.keys())}


@_register(
    "delete_recipe",
    "Permanently delete a recipe by id. Confirm with the user before "
    "calling — this is irreversible.",
    {
        "type": "object",
        "properties": {"id": {"type": ["integer", "string"]}},
        "required": ["id"],
    },
)
async def delete_recipe(id: int) -> dict:
    # Cascade: clear any weekly_plan references first
    await asyncio.to_thread(
        _write, "UPDATE weekly_plan SET recipe_id = NULL WHERE recipe_id = %s", (id,)
    )
    rc = await asyncio.to_thread(_write, "DELETE FROM recipes WHERE id = %s", (id,))
    return {"ok": rc > 0, "id": id, "rows_deleted": rc}


@_register(
    "rate_recipe",
    "Set a user's rating (0-5) and optional comments on a recipe.",
    {
        "type": "object",
        "properties": {
            "id": {"type": ["integer", "string"]},
            "rating": {"type": "number", "minimum": 0, "maximum": 5},
            "comments": {"type": "string"},
        },
        "required": ["id", "rating"],
    },
)
async def rate_recipe(id: int, rating: float, comments: str | None = None) -> dict:
    rc = await asyncio.to_thread(
        _write,
        "UPDATE recipes SET user_rating = %s, user_comments = COALESCE(%s, user_comments), last_made = CURRENT_DATE WHERE id = %s",
        (rating, comments, id),
    )
    return {"ok": rc > 0, "id": id, "rating": rating}


# ─── Chore writes ────────────────────────────────────────────────────────
_VALID_RECURRING_CHORE = {"daily", "weekly", "weekdays", "weekends"}


@_register(
    "add_chore",
    "Add a chore. Person: 'Cole' / 'Zander' / 'General'. Date defaults "
    "to today. `recurring` auto-regenerates on done-toggle and rolls "
    "forward when unfinished. `dollars` = Cole's reward (numeric); "
    "`points` = Zander's reward (integer); both default 0.",
    {
        "type": "object",
        "properties": {
            "person": {"type": "string"},
            "chore_name": {"type": "string"},
            "chore_date": {"type": "string", "description": "ISO YYYY-MM-DD. Defaults to today."},
            "recurring": {
                "type": "string",
                "enum": ["daily", "weekly", "weekdays", "weekends"],
            },
            "dollars": {"type": ["number", "string"]},
            "points": {"type": ["integer", "string"]},
        },
        "required": ["person", "chore_name"],
    },
)
async def add_chore(
    person: str, chore_name: str, chore_date: str | None = None,
    recurring: str | None = None,
    dollars: float | str | None = None,
    points: int | str | None = None,
) -> dict:
    cd = chore_date or date.today().isoformat()
    if recurring and recurring not in _VALID_RECURRING_CHORE:
        return {"ok": False, "error": f"recurring must be one of {sorted(_VALID_RECURRING_CHORE)} or omitted"}
    try:
        d = round(float(dollars), 2) if dollars not in (None, "") else 0.0
        p = int(points) if points not in (None, "") else 0
    except (TypeError, ValueError):
        return {"ok": False, "error": "dollars must be a number, points must be an integer"}
    if d < 0 or p < 0:
        return {"ok": False, "error": "rewards cannot be negative"}
    row = await asyncio.to_thread(
        _write_returning,
        """
        INSERT INTO chores (person, chore_name, chore_date, done, recurring, dollars, points)
        VALUES (%s, %s, %s, FALSE, %s, %s, %s)
        RETURNING id, person, chore_name, chore_date, done, recurring, dollars, points
        """,
        (person, chore_name, cd, recurring, d, p),
    )
    # Save as template too — Benson-added chores autocomplete on the
    # hub form and remember the dollar value next time.
    try:
        await asyncio.to_thread(
            _write,
            """
            INSERT INTO chore_templates
                (person, chore_name, default_dollars, default_points, use_count, archived_at)
            VALUES (%s, %s, %s, %s, 1, NOW())
            ON CONFLICT (person, chore_name) DO UPDATE SET
                use_count = chore_templates.use_count + 1,
                default_dollars = CASE WHEN EXCLUDED.default_dollars > 0
                    THEN EXCLUDED.default_dollars ELSE chore_templates.default_dollars END,
                default_points = CASE WHEN EXCLUDED.default_points > 0
                    THEN EXCLUDED.default_points ELSE chore_templates.default_points END,
                archived_at = NOW()
            """,
            (person, chore_name.lower().strip(), d, p),
        )
    except Exception as e:
        logger.warning(f"chore template upsert failed (non-fatal): {e}")
    return {"ok": True, "chore": row}


@_register(
    "mark_chore_done",
    "Mark a chore complete (or set it back to open with done=false).",
    {
        "type": "object",
        "properties": {
            "id": {"type": ["integer", "string"]},
            "done": {"type": "boolean", "default": True},
        },
        "required": ["id"],
    },
)
async def mark_chore_done(id: int, done: bool = True) -> dict:
    rc = await asyncio.to_thread(
        _write, "UPDATE chores SET done = %s WHERE id = %s", (done, id)
    )
    return {"ok": rc > 0, "id": id, "done": done, "rows": rc}


@_register(
    "delete_chore",
    "Permanently delete a chore by id.",
    {
        "type": "object",
        "properties": {"id": {"type": ["integer", "string"]}},
        "required": ["id"],
    },
)
async def delete_chore(id: int) -> dict:
    rc = await asyncio.to_thread(_write, "DELETE FROM chores WHERE id = %s", (id,))
    return {"ok": rc > 0, "id": id, "rows_deleted": rc}


@_register(
    "update_chore",
    "Edit an existing chore's person / name / date / recurrence / "
    "rewards. Only pass fields to change. recurring='' clears it.",
    {
        "type": "object",
        "properties": {
            "id": {"type": ["integer", "string"]},
            "person": {"type": "string"},
            "chore_name": {"type": "string"},
            "chore_date": {"type": "string", "description": "ISO YYYY-MM-DD"},
            "recurring": {
                "type": ["string", "null"],
                "enum": ["daily", "weekly", "weekdays", "weekends", None, ""],
            },
            "dollars": {"type": ["number", "string"]},
            "points": {"type": ["integer", "string"]},
        },
        "required": ["id"],
    },
)
async def update_chore(id: int, **fields) -> dict:
    if "recurring" in fields:
        v = fields["recurring"]
        if v in ("", None):
            fields["recurring"] = None
        elif v not in _VALID_RECURRING_CHORE:
            return {"ok": False, "error": f"recurring must be one of {sorted(_VALID_RECURRING_CHORE)} / null / ''"}
    if "dollars" in fields and fields["dollars"] not in (None, ""):
        try:
            d = round(float(fields["dollars"]), 2)
        except (TypeError, ValueError):
            return {"ok": False, "error": "dollars must be a number"}
        if d < 0:
            return {"ok": False, "error": "dollars cannot be negative"}
        fields["dollars"] = d
    if "points" in fields and fields["points"] not in (None, ""):
        try:
            p = int(fields["points"])
        except (TypeError, ValueError):
            return {"ok": False, "error": "points must be an integer"}
        if p < 0:
            return {"ok": False, "error": "points cannot be negative"}
        fields["points"] = p
    fields = {k: v for k, v in fields.items() if k == "recurring" or v is not None}
    if not fields:
        return {"ok": False, "error": "no fields to update"}
    cols = ", ".join(f"{k} = %s" for k in fields)
    params = list(fields.values()) + [id]
    rc = await asyncio.to_thread(
        _write, f"UPDATE chores SET {cols} WHERE id = %s", tuple(params)
    )
    return {"ok": rc > 0, "id": id, "rows": rc, "updated": list(fields.keys())}


@_register(
    "weekly_chore_summary",
    "Tally one person's (or the family's) chore rewards for a week. "
    "Use when a kid asks 'what have I earned so far?' or for the "
    "Sunday roll-up. Returns earned vs possible dollars + points and "
    "a per-chore breakdown. `week_start` defaults to current Monday; "
    "pass any ISO date inside a past week to scope back. `person` is "
    "optional — omit for everyone.",
    {
        "type": "object",
        "properties": {
            "person": {"type": "string"},
            "week_start": {"type": "string"},
        },
        "required": [],
    },
)
async def weekly_chore_summary(
    person: str | None = None, week_start: str | None = None
) -> dict:
    from datetime import date as _d, datetime as _dt, timedelta as _td
    anchor = _dt.strptime(week_start, "%Y-%m-%d").date() if week_start else _d.today()
    monday = anchor - _td(days=anchor.weekday())
    next_monday = monday + _td(days=7)

    sql = (
        "SELECT person, "
        "COALESCE(SUM(dollars) FILTER (WHERE done), 0) AS earned_dollars, "
        "COALESCE(SUM(points) FILTER (WHERE done), 0) AS earned_points, "
        "COALESCE(SUM(dollars), 0) AS possible_dollars, "
        "COALESCE(SUM(points), 0) AS possible_points, "
        "COUNT(*) FILTER (WHERE done) AS done_count, "
        "COUNT(*) AS total_count "
        "FROM chores WHERE chore_date >= %s AND chore_date < %s"
    )
    params: list = [monday, next_monday]
    if person:
        sql += " AND LOWER(person) = LOWER(%s)"
        params.append(person)
    sql += " GROUP BY person ORDER BY person"
    rows = await asyncio.to_thread(_query, sql, tuple(params))

    bsql = (
        "SELECT id, person, chore_name, chore_date, done, dollars, points "
        "FROM chores WHERE chore_date >= %s AND chore_date < %s"
    )
    bparams: list = [monday, next_monday]
    if person:
        bsql += " AND LOWER(person) = LOWER(%s)"
        bparams.append(person)
    bsql += " ORDER BY chore_date, chore_name"
    items = await asyncio.to_thread(_query, bsql, tuple(bparams))

    return {
        "ok": True,
        "week_start": monday.isoformat(),
        "week_end": (next_monday - _td(days=1)).isoformat(),
        "summary": rows,
        "items": items,
    }


@_register(
    "list_chore_templates",
    "Suggested chore catalog from past assignments — most-used "
    "(person, chore_name) pairs with default rewards. Query before "
    "add_chore to reuse existing names + their default rewards.",
    {
        "type": "object",
        "properties": {
            "person": {"type": "string"},
        },
        "required": [],
    },
)
async def list_chore_templates(person: str | None = None) -> dict:
    sql = (
        "SELECT id, person, chore_name, default_dollars, default_points, "
        "category, use_count FROM chore_templates"
    )
    params: tuple = ()
    if person:
        sql += " WHERE LOWER(person) = LOWER(%s)"
        params = (person,)
    sql += " ORDER BY use_count DESC, chore_name LIMIT 100"
    rows = await asyncio.to_thread(_query, sql, params)
    return {"ok": True, "templates": rows, "count": len(rows)}


# ─── Weekly plan writes ──────────────────────────────────────────────────
@_register(
    "schedule_meal",
    "Schedule a recipe as the meal for a specific date. Replaces any "
    "existing entry for that date.",
    {
        "type": "object",
        "properties": {
            "plan_date": {"type": "string", "description": "ISO YYYY-MM-DD"},
            "recipe_id": {"type": ["integer", "string"]},
            "status": {
                "type": "string",
                "default": "planned",
                "description": "e.g. 'planned', 'made', 'skipped'",
            },
        },
        "required": ["plan_date", "recipe_id"],
    },
)
async def schedule_meal(
    plan_date: str, recipe_id: int, status: str = "planned"
) -> dict:
    row = await asyncio.to_thread(
        _write_returning,
        """
        INSERT INTO weekly_plan (plan_date, recipe_id, status)
        VALUES (%s, %s, %s)
        ON CONFLICT (plan_date) DO UPDATE
            SET recipe_id = EXCLUDED.recipe_id, status = EXCLUDED.status
        RETURNING plan_date, recipe_id, status
        """,
        (plan_date, recipe_id, status),
    )
    return {"ok": True, "scheduled": row}


@_register(
    "unschedule_meal",
    "Remove the meal scheduled for a specific date (clears the slot).",
    {
        "type": "object",
        "properties": {"plan_date": {"type": "string"}},
        "required": ["plan_date"],
    },
)
async def unschedule_meal(plan_date: str) -> dict:
    rc = await asyncio.to_thread(
        _write, "DELETE FROM weekly_plan WHERE plan_date = %s", (plan_date,)
    )
    return {"ok": rc > 0, "plan_date": plan_date, "rows": rc}


# ─── Memory writes (read + remember_this already exist above) ────────────
@_register(
    "forget_memory",
    "Delete a memory by id. Use when the user explicitly asks Benson "
    "to forget something. Confirm before calling — irreversible.",
    {
        "type": "object",
        "properties": {"id": {"type": ["integer", "string"]}},
        "required": ["id"],
    },
)
async def forget_memory(id: int) -> dict:
    rc = await asyncio.to_thread(_write, "DELETE FROM memories WHERE id = %s", (id,))
    return {"ok": rc > 0, "id": id, "rows_deleted": rc}


@_register(
    "update_memory",
    "Edit an existing memory's content or importance. Useful when a "
    "fact changes (e.g., a preference is updated).",
    {
        "type": "object",
        "properties": {
            "id": {"type": ["integer", "string"]},
            "content": {"type": "string"},
            "importance": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["id"],
    },
)
async def update_memory(
    id: int, content: str | None = None, importance: float | None = None
) -> dict:
    if content is None and importance is None:
        return {"ok": False, "error": "no fields to update"}
    # If content changes, also recompute the embedding so search stays accurate.
    if content is not None:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL_NAME
        # Reuse memory module's lazy model rather than reloading
        from memory import _embedding_model
        emb = _embedding_model().encode(content).tolist()
        rc = await asyncio.to_thread(
            _write,
            "UPDATE memories SET content = %s, embedding = %s::vector"
            + (", importance = %s" if importance is not None else "")
            + " WHERE id = %s",
            ((content, emb, importance, id) if importance is not None else (content, emb, id)),
        )
    else:
        rc = await asyncio.to_thread(
            _write,
            "UPDATE memories SET importance = %s WHERE id = %s",
            (importance, id),
        )
    return {"ok": rc > 0, "id": id, "rows": rc}


# ─── Signal ──────────────────────────────────────────────────────────────
_SIGNAL_FILE_PREFIXES = ("/home/casey/Benson/", "/tmp/benson-")


@_register(
    "send_signal",
    "Send a Signal message via Benson's Signal bridge. `to` is a phone "
    "number in E.164 ('+15551234567') for a direct message, or a Signal "
    "group base64 id for a group. Optionally include images or documents "
    "via `file_paths` (each must be under /home/casey/Benson/* or "
    "/tmp/benson-*). Use only when the user asks Benson to message "
    "someone explicitly. Always confirm the destination first.",
    {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "message": {"type": "string"},
            "file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of local file paths to attach as images or "
                    "documents.  Each path must start with '/home/casey/Benson/' "
                    "or '/tmp/benson-'."
                ),
            },
        },
        "required": ["to", "message"],
    },
)
async def send_signal(
    to: str,
    message: str,
    file_paths: list[str] | None = None,
) -> dict:
    from pathlib import Path as _Path
    from signal_handler import send_signal_message

    if file_paths:
        allowed_desc = " or ".join(f"'{p}'" for p in _SIGNAL_FILE_PREFIXES)
        for fp in file_paths:
            # Prefix check on the raw path
            if not any(fp.startswith(pfx) for pfx in _SIGNAL_FILE_PREFIXES):
                return {
                    "ok": False,
                    "error": (
                        f"file_path {fp!r} is not allowed — must be under {allowed_desc}"
                    ),
                }
            # Resolve and re-check to block path-traversal tricks
            try:
                resolved = _Path(fp).resolve()
            except Exception as exc:
                return {"ok": False, "error": f"cannot resolve path {fp!r}: {exc}"}
            if not any(str(resolved).startswith(pfx) for pfx in _SIGNAL_FILE_PREFIXES):
                return {
                    "ok": False,
                    "error": (
                        f"resolved path {str(resolved)!r} escapes allowed prefixes "
                        f"({allowed_desc})"
                    ),
                }

    return await send_signal_message(to, message, file_paths=file_paths or None)


# ─── HA state inspection ─────────────────────────────────────────────────
@_register(
    "get_entity_state",
    "Read the live state of any HA entity. Useful for 'is the X on?' "
    "questions or to check before acting.",
    {
        "type": "object",
        "properties": {"entity_id": {"type": "string"}},
        "required": ["entity_id"],
    },
)
async def get_entity_state(entity_id: str) -> dict:
    try:
        s = await ha_get_state(entity_id)
        return {
            "entity_id": s.get("entity_id"),
            "state": s.get("state"),
            "attributes": s.get("attributes", {}),
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Recipe ingestion (video URL, image) ─────────────────────────────────
@_register(
    "import_recipe_from_url",
    "Import a recipe from a video URL — TikTok, Instagram Reel, YouTube "
    "Short, YouTube video, etc. Downloads the video, transcribes the "
    "narration with Whisper, reads the platform caption/description, and "
    "uses Claude to extract a structured recipe (title, ingredients, "
    "steps, course, prep time, tags). Saves it to the recipes database "
    "and returns the new recipe id + a summary. Use this whenever the "
    "user shares a cooking-video link or asks to add a recipe from a URL.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Direct link to the video."},
        },
        "required": ["url"],
    },
)
async def import_recipe_from_url(url: str) -> dict:
    from recipes import RecipeIngester
    try:
        recipe = await RecipeIngester().from_video_url(url)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    ing = recipe.get("ingredients") or []
    steps = recipe.get("steps") or []
    return {
        "ok": True,
        "recipe_id": recipe.get("id"),
        "title": recipe.get("title"),
        "course": recipe.get("course"),
        "prep_time": recipe.get("prep_time"),
        "ingredients_count": len(ing),
        "steps_count": len(steps),
        "transcript_chars": recipe.get("transcript_chars"),
        "caption_chars": recipe.get("caption_chars"),
        "source_url": url,
    }


@_register(
    "import_recipe_from_image",
    "Import a recipe from a photo — handwritten card, cookbook page, "
    "screenshot, etc. Sends the image to Claude vision, extracts a "
    "structured recipe, and saves it. Use when the user attaches a photo "
    "of a recipe.",
    {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Local filesystem path to the image (e.g., /tmp/benson-attachments/abc.jpg).",
            },
        },
        "required": ["image_path"],
    },
)
async def import_recipe_from_image(image_path: str) -> dict:
    from recipes import RecipeIngester
    try:
        recipe = await RecipeIngester().from_image(image_path)
    except Exception as e:
        logger.exception(f"import_recipe_from_image({image_path}) failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {
        "ok": True,
        "recipe_id": recipe.get("id"),
        "title": recipe.get("title"),
        "ingredients_count": len(recipe.get("ingredients") or []),
        "steps_count": len(recipe.get("steps") or []),
        "image_path": image_path,
    }


@_register(
    "analyze_image",
    "Look at an image and answer a question about it. Generic vision tool "
    "for non-recipe images: receipts, screenshots, calendar pages, "
    "handwritten notes, scenes, packaging, etc. Provide an image_path "
    "(usually from a Signal attachment) and a `query` describing what to "
    "extract or assess. Returns the model's text response.",
    {
        "type": "object",
        "properties": {
            "image_path": {"type": "string"},
            "query": {
                "type": "string",
                "description": "What the user wants to know or extract from the image.",
            },
        },
        "required": ["image_path", "query"],
    },
)
async def analyze_image(image_path: str, query: str) -> dict:
    from pathlib import Path
    from oauth_oneshot import ask_with_image

    p = Path(image_path)
    if not p.exists():
        return {"ok": False, "error": f"image not found: {image_path}"}

    try:
        text = await ask_with_image(str(p), query, model="sonnet", timeout_s=120)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not text:
        return {"ok": False, "error": "vision call returned empty (timeout or auth?)"}
    return {"ok": True, "answer": text}


# ─── Deep memory search (vector RAG over conversations + events + recipes + chores) ─
@_register(
    "search_history",
    "Semantic search across the household's deep history — every past "
    "conversation, every calendar event (past and future), every recipe, "
    "every chore (open or completed), and every memory file. Returns the "
    "top semantic matches. Use this for questions like 'what did Cole "
    "say about baseball last month', 'when did we last make tacos', "
    "'what was that doctor's name Lindsey mentioned'. Prefer this over "
    "memory_read when looking for older or non-curated info.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for. Be descriptive — semantic match works better than keywords."},
            "source_type": {
                "type": "string",
                "enum": ["conversation", "event", "recipe", "chore", "memory_file"],
                "description": "Optional filter to one source. Omit for everything.",
            },
            "speaker": {"type": "string", "description": "Optional: filter to one person."},
            "days_back": {"type": ["integer", "string"], "description": "Optional: only items from the last N days."},
            "limit": {"type": ["integer", "string"], "default": 8, "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
    },
)
async def search_history(
    query: str,
    source_type: str | None = None,
    speaker: str | None = None,
    days_back: int | None = None,
    limit: int = 8,
) -> dict:
    from memory_index import search
    rows = await asyncio.to_thread(
        search,
        query, source_type=source_type, speaker=speaker, limit=limit, days_back=days_back,
    )
    return {
        "query": query,
        "count": len(rows),
        "results": [
            {
                "source_type": r["source_type"],
                "title": r.get("title"),
                "speaker": r.get("speaker"),
                "occurred_at": r["occurred_at"].isoformat() if r.get("occurred_at") else None,
                "snippet": (r.get("content") or "")[:600],
                "distance": round(float(r["distance"]), 4),
            } for r in rows
        ],
    }


# ─── Listen-in mode (passive memory enrichment) ──────────────────────────
@_register(
    "listen_in",
    "Start passive-listening mode: a connected iPad in `room` (default "
    "'kitchen') captures audio for `duration_min` (default 90, max 180), "
    "then automatically transcribes with Whisper, extracts durable "
    "per-person facts via Claude, and appends them to the household's "
    "memory files. Use when the user explicitly asks to listen, listen "
    "in, capture dinner, eavesdrop on the table, etc. CONFIRM the room "
    "and duration before calling — this is privacy-sensitive. The iPad "
    "shows a clearly visible indicator while active.",
    {
        "type": "object",
        "properties": {
            "duration_min": {"type": ["integer", "string"], "default": 90, "minimum": 5, "maximum": 180},
            "room": {"type": "string", "default": "kitchen"},
            "started_by": {"type": "string", "description": "Speaker who initiated."},
        },
    },
)
async def listen_in(
    duration_min: int = 90,
    room: str = "kitchen",
    started_by: str | None = None,
) -> dict:
    import httpx as _httpx
    payload = {"duration_min": duration_min, "room": room, "started_by": started_by or ""}
    async with _httpx.AsyncClient(timeout=10) as client:
        r = await client.post("http://localhost:8100/listening/start", json=payload)
    return r.json()


@_register(
    "stop_listening",
    "Stop an active listening session. If `session_id` is omitted, stops "
    "the most recent active session. Triggers transcription + memory "
    "extraction in the background.",
    {
        "type": "object",
        "properties": {
            "session_id": {"type": ["integer", "string"]},
        },
    },
)
async def stop_listening(session_id: int | None = None) -> dict:
    import httpx as _httpx
    if session_id is None:
        async with _httpx.AsyncClient(timeout=5) as client:
            r = await client.get("http://localhost:8100/listening/status")
        rows = r.json().get("recent", [])
        active = [r for r in rows if r.get("status") == "active"]
        if not active:
            return {"ok": False, "error": "no active listening session"}
        session_id = active[0]["id"]
    async with _httpx.AsyncClient(timeout=10) as client:
        r = await client.post(f"http://localhost:8100/listening/stop/{session_id}")
    return r.json()


# ─── Camera (iPad-as-eyeball) ────────────────────────────────────────────
@_register(
    "look_at_camera",
    "Take a real-time snapshot through one of the household's stationed "
    "iPad cameras and describe what you see. Use this when the user "
    "asks 'what's on the counter', 'is anyone in the kitchen', 'check "
    "the stove', etc. Cameras must be registered (the iPad has the "
    "Benson Eye page open). Argument `camera` defaults to 'kitchen'; "
    "`query` is what to look for. Returns a dict with `answer` (the "
    "vision model's text description) and `image_path` (the saved JPEG "
    "on disk, e.g. '/tmp/benson-cameras/kitchen.jpg', or null on camera "
    "error). When the user wants to share the snapshot, pass that "
    "`image_path` directly to send_signal's `file_paths` parameter.",
    {
        "type": "object",
        "properties": {
            "camera": {"type": "string", "default": "kitchen"},
            "query": {
                "type": "string",
                "description": "What to extract or assess. Be specific — e.g., 'is the stove on', 'what ingredients are on the counter', 'count the dishes in the sink'.",
            },
        },
        "required": ["query"],
    },
)
async def look_at_camera(query: str, camera: str = "kitchen") -> dict:
    from camera_handler import trigger_snapshot
    path = await trigger_snapshot(camera, source="agent")
    if not path:
        return {
            "ok": False,
            "image_path": None,
            "error": f"camera '{camera}' not connected. Open https://192.168.0.240/camera/{camera}/page on that iPad in Safari.",
        }
    result = await analyze_image(str(path), query)
    # Surface the saved snapshot path alongside the answer so the caller can
    # chain it into send_signal's file_paths (its allowlist already permits
    # /tmp/benson-* via _SIGNAL_FILE_PREFIXES).
    if isinstance(result, dict):
        result.setdefault("image_path", str(path))
    return result


# ─── File-based Memory (replaces pgvector MemoryStore) ───────────────────
@_register(
    "memory_list",
    "List every file in long-term memory with its path and a one-line "
    "preview. Read this first when you need context about a household "
    "member or a topic — it tells you which files exist before you "
    "decide what to read in full.",
    {"type": "object", "properties": {}},
)
async def memory_list_tool() -> dict:
    from memory_tools import memory_list
    return memory_list()


@_register(
    "memory_read",
    "Read one memory file in full. Use after memory_list when you need "
    "the actual content. Path is relative to the memory directory — "
    "e.g., 'casey.md', 'lindsey.md', 'household.md'.",
    {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
async def memory_read_tool(path: str) -> dict:
    from memory_tools import memory_read
    return memory_read(path)


@_register(
    "memory_write",
    "Create or overwrite a memory file with new content. Use sparingly — "
    "prefer memory_append for adding facts. Use memory_write when "
    "restructuring a whole file (e.g., reorganizing casey.md after a "
    "cleanup pass). Path is relative; content is the full new file body.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)
async def memory_write_tool(path: str, content: str) -> dict:
    from memory_tools import memory_write
    return memory_write(path, content)


@_register(
    "memory_append",
    "Append a fact (or several lines) to an existing memory file. This "
    "is the primary way you record new things you learn. Pick the right "
    "file: per-person facts go in '<name>.md', household-wide facts in "
    "'household.md', topical files like 'preferences/cooking.md' are "
    "fine to create. Only record durable facts — preferences, "
    "constraints, recurring patterns. NOT transient state (today's "
    "weather, current music) and NOT meta-commentary about whether "
    "something is worth remembering.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "The fact(s) to append. Use markdown bullet format ('- fact') for lists."},
        },
        "required": ["path", "content"],
    },
)
async def memory_append_tool(path: str, content: str) -> dict:
    from memory_tools import memory_append
    return memory_append(path, content)


@_register(
    "memory_delete",
    "Delete a memory file entirely. Use only when explicitly asked to "
    "forget something at file granularity. To remove individual lines, "
    "use memory_read → memory_write with the cleaned content instead.",
    {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
async def memory_delete_tool(path: str) -> dict:
    from memory_tools import memory_delete
    return memory_delete(path)


# ─── Structured memory: events (time-series) + lists (collections) ───────
# Use these instead of memory_append when the data is timestamped (events)
# or list-shaped (collections). MD files stay clean; queries stay easy.

@_register(
    "log_event",
    "Record a timestamped occurrence — workouts, meals, moods, "
    "observations, anything that has a 'when' and accumulates over time. "
    "Use this INSTEAD of memory_append for time-series data. Example: "
    "Casey says he did 50 pushups → log_event(category='workout', "
    "person='Casey', content='50 pushups', metadata={'reps': 50, "
    "'exercise': 'pushups'}). Pick a short snake_case category name "
    "('workout', 'meal', 'mood', 'observation', 'sleep', 'reading'). "
    "Re-use existing categories — call query_events first if unsure.",
    {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "snake_case category, e.g. 'workout', 'meal', 'mood'.",
            },
            "content": {
                "type": "string",
                "description": "Short natural-language description of what happened.",
            },
            "person": {
                "type": "string",
                "description": "Who it's about (Casey/Lindsey/Cole/Zander). Omit for household-level events.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional structured details (reps, duration, location, etc.).",
            },
            "occurred_at": {
                "type": "string",
                "description": "ISO timestamp if NOT 'now' (e.g. '2026-04-26T18:30:00-06:00'). Omit for current time.",
            },
        },
        "required": ["category", "content"],
    },
)
async def log_event_tool(
    category: str,
    content: str,
    person: str | None = None,
    metadata: dict | None = None,
    occurred_at: str | None = None,
) -> dict:
    from memory_structured import log_event
    return log_event(
        category=category,
        content=content,
        person=person,
        metadata=metadata,
        source="agent",
        occurred_at=occurred_at,
    )


@_register(
    "query_events",
    "Read back logged events, newest first. Use when asked 'what workouts "
    "have I done this month', 'what did Cole eat for breakfast last week', "
    "etc. Filter by category, person, and how far back to look. Defaults "
    "return the 50 most recent events across all categories.",
    {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Category filter (e.g. 'workout')."},
            "person": {"type": "string", "description": "Person filter."},
            "days_back": {"type": ["integer", "string"], "description": "Look this many days back (omit for all-time)."},
            "limit": {"type": ["integer", "string"], "description": "Max rows (default 50, max 200)."},
        },
    },
)
async def query_events_tool(
    category: str | None = None,
    person: str | None = None,
    days_back: int | None = None,
    limit: int = 50,
) -> dict:
    from memory_structured import query_events
    return query_events(
        category=category,
        person=person,
        days_back=days_back,
        limit=min(int(limit or 50), 200),
    )


@_register(
    "list_add",
    "Add an item to a named list (a topic-scoped collection that "
    "accumulates over time — gift ideas, books to read, packing list, "
    "household projects). Auto-creates the list if it doesn't exist. "
    "Use this INSTEAD of memory_append for collection-shaped data. "
    "Example: 'add silk pajamas to mom's mother's day list' → list_add("
    "name='mothers_day_2026', item='silk pajamas'). Pick stable "
    "snake_case names; call list_all first if unsure what exists.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "List name (snake_case slug, e.g. 'mothers_day_2026')."},
            "item": {"type": "string", "description": "The item to add."},
            "title": {"type": "string", "description": "Human-readable title (set on first add only)."},
            "added_by": {"type": "string", "description": "Who added it (defaults to current speaker)."},
            "metadata": {"type": "object", "description": "Optional structured details (price, link, etc.)."},
        },
        "required": ["name", "item"],
    },
)
async def list_add_tool(
    name: str,
    item: str,
    title: str | None = None,
    added_by: str | None = None,
    metadata: dict | None = None,
) -> dict:
    from memory_structured import list_add
    return list_add(name=name, item=item, title=title, added_by=added_by, metadata=metadata)


@_register(
    "list_read",
    "Read every item in a named list. Returns items oldest-first with "
    "their done status. Use when asked 'what's on the mother's day list', "
    "'show me the gift ideas for Cole', etc.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "include_done": {"type": "boolean", "description": "Include checked-off items (default true)."},
        },
        "required": ["name"],
    },
)
async def list_read_tool(name: str, include_done: bool = True) -> dict:
    from memory_structured import list_read
    return list_read(name=name, include_done=include_done)


@_register(
    "list_all",
    "Show every named list with item counts. Call this when you need to "
    "find which list to add to, or to give the user an overview of what "
    "they have going on.",
    {
        "type": "object",
        "properties": {
            "include_archived": {"type": "boolean", "description": "Include archived lists (default false)."},
        },
    },
)
async def list_all_tool(include_archived: bool = False) -> dict:
    from memory_structured import list_all
    return list_all(include_archived=include_archived)


@_register(
    "list_check",
    "Mark a list item done (or un-done). Use when an idea has been bought, "
    "a task completed, etc. Pass the item_id from list_read.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "item_id": {"type": ["integer", "string"]},
            "done": {"type": "boolean", "description": "True to check off, false to un-check (default true)."},
        },
        "required": ["name", "item_id"],
    },
)
async def list_check_tool(name: str, item_id: int, done: bool = True) -> dict:
    from memory_structured import list_check
    return list_check(name=name, item_id=item_id, done=done)


@_register(
    "list_remove",
    "Permanently remove an item from a list. Pass the item_id from "
    "list_read. Use list_check (done=true) for completed-but-keep-as-record "
    "items; only use list_remove if the user wants it gone entirely.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "item_id": {"type": ["integer", "string"]},
        },
        "required": ["name", "item_id"],
    },
)
async def list_remove_tool(name: str, item_id: int) -> dict:
    from memory_structured import list_remove
    return list_remove(name=name, item_id=item_id)


# ─── Google Calendar / Gmail ─────────────────────────────────────────────
@_register(
    "query_calendar",
    "Read upcoming or recent calendar events from linked Google Calendars "
    "(synced into the local DB). Use this whenever someone asks 'what's "
    "on the schedule', 'what's today look like', 'do I have anything "
    "tomorrow', 'what's Lindsey doing Friday'. Pass `user_name` "
    "(Casey/Lindsey/Cole/...) to scope to one person; OMIT `user_name` to "
    "see all linked household members' events merged together — use that "
    "for the kids (Cole/Zander), who don't have linked accounts but "
    "appear on Casey's and Lindsey's calendars. Defaults to the next 7 days.",
    {
        "type": "object",
        "properties": {
            "user_name": {"type": "string", "description": "Optional. Omit for the whole family."},
            "days": {"type": ["integer", "string"], "default": 7, "description": "Days from now to look ahead."},
            "search": {"type": "string", "description": "Optional case-insensitive title/description filter."},
        },
        "required": [],
    },
)
async def query_calendar(
    user_name: str | None = None, days: int = 7, search: str | None = None
) -> dict:
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)
    sql = (
        "SELECT user_name, person, title, location, starts_at, ends_at, "
        "all_day, description, google_event_id "
        "FROM calendar_events WHERE starts_at >= %s AND starts_at < %s "
        "AND COALESCE(status,'confirmed') != 'cancelled'"
    )
    params: list = [now - timedelta(hours=1), end]
    if user_name:
        sql += " AND user_name = %s"
        params.append(user_name)
    if search:
        sql += " AND (title ILIKE %s OR description ILIKE %s)"
        params.extend([f"%{search}%", f"%{search}%"])
    sql += " ORDER BY starts_at LIMIT 100"
    rows = await asyncio.to_thread(_query, sql, tuple(params))
    return {
        "ok": True,
        "user_name": user_name or "all_linked",
        "count": len(rows),
        "events": [
            {
                "user_name": r["user_name"],
                "person": r["person"],
                "title": r["title"],
                "location": r["location"],
                "starts_at": r["starts_at"].isoformat() if r["starts_at"] else None,
                "ends_at": r["ends_at"].isoformat() if r["ends_at"] else None,
                "all_day": r["all_day"],
                "description": (r["description"] or "")[:300],
                "event_id": r["google_event_id"],
            }
            for r in rows
        ],
    }


@_register(
    "create_calendar_event",
    "Create a new event on one of a household member's Google calendars. "
    "`user_name` is who owns the linked account (Casey/Lindsey/...). "
    "`calendar` is a name hint — pass the person/category the event "
    "belongs to ('Cole', 'Zander', 'Family', 'Lindsey Personal'); "
    "Benson matches it to the right calendar. If `calendar` is omitted, "
    "Benson uses that user's stored default calendar (set on /admin/google). "
    "`start` and `end` are ISO format: 'YYYY-MM-DD' for all-day, "
    "'YYYY-MM-DDTHH:MM' for timed. Times are interpreted in America/Denver. "
    "If `end` is omitted, defaults to +1 hour for timed events or +1 day "
    "for all-day. CONFIRM date and time with the user before calling.",
    {
        "type": "object",
        "properties": {
            "user_name": {"type": "string"},
            "title": {"type": "string"},
            "start": {"type": "string", "description": "YYYY-MM-DD or YYYY-MM-DDTHH:MM"},
            "end": {"type": "string"},
            "calendar": {"type": "string", "description": "Calendar name hint (Cole/Zander/Family/etc.)"},
            "description": {"type": "string"},
            "location": {"type": "string"},
        },
        "required": ["user_name", "title", "start"],
    },
)
async def create_calendar_event(
    user_name: str, title: str, start: str,
    end: str | None = None, calendar: str | None = None,
    description: str | None = None, location: str | None = None,
) -> dict:
    from google_handler import create_event
    return await asyncio.to_thread(
        create_event,
        user_name, title, start, end,
        calendar=calendar, description=description, location=location,
    )


@_register(
    "update_calendar_event",
    "Modify an existing calendar event by id (get the id from "
    "query_calendar). Pass only the fields to change.",
    {
        "type": "object",
        "properties": {
            "user_name": {"type": "string"},
            "event_id": {"type": "string"},
            "title": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "calendar": {"type": "string"},
            "description": {"type": "string"},
            "location": {"type": "string"},
        },
        "required": ["user_name", "event_id"],
    },
)
async def update_calendar_event(
    user_name: str, event_id: str,
    title: str | None = None, start: str | None = None, end: str | None = None,
    calendar: str | None = None, description: str | None = None, location: str | None = None,
) -> dict:
    from google_handler import update_event
    return await asyncio.to_thread(
        update_event,
        user_name, event_id,
        title=title, start=start, end=end,
        calendar=calendar, description=description, location=location,
    )


@_register(
    "delete_calendar_event",
    "Delete a calendar event by id (from query_calendar). Confirm with "
    "the user before calling — this is irreversible from Benson's side.",
    {
        "type": "object",
        "properties": {
            "user_name": {"type": "string"},
            "event_id": {"type": "string"},
            "calendar": {"type": "string", "description": "Optional hint; Benson will search across calendars if omitted."},
        },
        "required": ["user_name", "event_id"],
    },
)
async def delete_calendar_event(
    user_name: str, event_id: str, calendar: str | None = None,
) -> dict:
    from google_handler import delete_event
    return await asyncio.to_thread(delete_event, user_name, event_id, calendar=calendar)


@_register(
    "search_email",
    "Search a household member's Gmail inbox (read-only). Returns matching "
    "messages with sender, subject, snippet, and date. Use Gmail search "
    "syntax in `query` ('from:school', 'subject:appointment', "
    "'newer_than:7d', etc.). Useful for 'did the school email anything', "
    "'find Lindsey's airline confirmation', etc.",
    {
        "type": "object",
        "properties": {
            "user_name": {"type": "string"},
            "query": {"type": "string", "description": "Gmail search query."},
            "max_results": {"type": ["integer", "string"], "default": 10},
        },
        "required": ["user_name", "query"],
    },
)
async def search_email(user_name: str, query: str, max_results: int = 10) -> dict:
    from google_handler import get_credentials_with_status
    from googleapiclient.discovery import build

    def _run() -> dict:
        creds, status = get_credentials_with_status(user_name)
        if not creds:
            if status == "no_row":
                return {
                    "ok": False,
                    "error": f"{user_name} has not linked Gmail. Go to /admin/google to authorize.",
                    "status": status,
                }
            return {
                "ok": False,
                "error": (
                    f"{user_name}'s Gmail token was rejected by Google "
                    f"({status}). The OAuth grant exists in our DB but "
                    f"Google revoked the refresh token (common causes: "
                    f"password change, signed out of devices, or grant "
                    f"removed at myaccount.google.com/permissions). "
                    f"Re-authorize at /admin/google to fix."
                ),
                "status": status,
            }
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
        listing = svc.users().messages().list(
            userId="me", q=query, maxResults=max(1, min(max_results, 30))
        ).execute()
        ids = [m["id"] for m in listing.get("messages", [])]
        out = []
        for mid in ids:
            m = svc.users().messages().get(
                userId="me", id=mid, format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in m.get("payload", {}).get("headers", [])}
            out.append({
                "id": mid,
                "from": headers.get("From"),
                "to": headers.get("To"),
                "subject": headers.get("Subject"),
                "date": headers.get("Date"),
                "snippet": m.get("snippet", ""),
            })
        return {"ok": True, "count": len(out), "messages": out}

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@_register(
    "read_email",
    "Read the full body of a single email by message id (returned from "
    "search_email). Returns plain-text body + headers.",
    {
        "type": "object",
        "properties": {
            "user_name": {"type": "string"},
            "message_id": {"type": "string"},
        },
        "required": ["user_name", "message_id"],
    },
)
async def read_email(user_name: str, message_id: str) -> dict:
    from google_handler import gmail_service
    import base64 as _b64

    def _walk(part) -> str:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return _b64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", "replace")
        for sub in part.get("parts", []) or []:
            t = _walk(sub)
            if t:
                return t
        return ""

    def _run() -> dict:
        from google_handler import get_credentials_with_status
        from googleapiclient.discovery import build
        creds, status = get_credentials_with_status(user_name)
        if not creds:
            if status == "no_row":
                return {"ok": False, "error": f"{user_name} has not linked Gmail. Go to /admin/google to authorize.", "status": status}
            return {
                "ok": False,
                "error": f"{user_name}'s Gmail token was rejected by Google ({status}). Re-authorize at /admin/google.",
                "status": status,
            }
        svc = build("gmail", "v1", credentials=creds, cache_discovery=False)
        m = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
        headers = {h["name"]: h["value"] for h in m.get("payload", {}).get("headers", [])}
        body = _walk(m.get("payload", {}))[:10000]
        return {
            "ok": True,
            "from": headers.get("From"),
            "to": headers.get("To"),
            "subject": headers.get("Subject"),
            "date": headers.get("Date"),
            "body": body,
        }

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ─── Generic URL fetcher ─────────────────────────────────────────────────
@_register(
    "fetch_url",
    "Fetch a non-video URL and answer a question about its content. "
    "Handles HTML pages (articles, blogs, event listings, schedules, "
    "store pages), PDFs, plain text, and image URLs. For video URLs "
    "(TikTok/Reel/YouTube) use import_recipe_from_url instead. After "
    "extracting content, the model answers `query` based on what it "
    "found. Pair with downstream write tools (remember_this, add_chore, "
    "schedule_meal, etc.) to actually act on the info.",
    {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "query": {
                "type": "string",
                "description": "What to extract or answer. Be specific (e.g., 'list every event date and title' or 'summarize the recipe with ingredient quantities').",
            },
        },
        "required": ["url", "query"],
    },
)
async def fetch_url(url: str, query: str) -> dict:
    import os
    from pathlib import Path

    import httpx as _httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
    }
    try:
        async with _httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as client:
            r = await client.get(url)
    except Exception as e:
        return {"ok": False, "error": f"fetch failed: {e}"}
    if r.status_code >= 400:
        return {"ok": False, "error": f"HTTP {r.status_code}", "url": str(r.url)}

    ctype = (r.headers.get("content-type") or "").lower()
    final_url = str(r.url)

    # Image URLs → save and route through analyze_image
    if ctype.startswith("image/"):
        ext = ctype.split("/", 1)[1].split(";", 1)[0].strip() or "jpg"
        if ext == "jpeg":
            ext = "jpg"
        os.makedirs("/tmp/benson-attachments", exist_ok=True)
        safe = "".join(c for c in url if c.isalnum())[:30]
        path = f"/tmp/benson-attachments/url_{safe}.{ext}"
        Path(path).write_bytes(r.content)
        return await analyze_image(path, query)

    # Extract text content
    text = ""
    kind = "unknown"
    if "application/pdf" in ctype or final_url.lower().endswith(".pdf"):
        kind = "pdf"
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(r.content))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            return {"ok": False, "error": f"pdf parse failed: {e}"}
    elif "html" in ctype or "xml" in ctype or r.content[:200].lstrip().startswith(b"<"):
        kind = "html"
        try:
            import trafilatura
            extracted = trafilatura.extract(
                r.text,
                include_links=False,
                include_tables=True,
                favor_recall=True,
            )
            text = extracted or ""
        except Exception:
            text = ""
        if not text:
            # Fallback: strip tags crudely
            import re as _re
            text = _re.sub(r"<[^>]+>", " ", r.text)
            text = _re.sub(r"\s+", " ", text).strip()[:30000]
    else:
        kind = "text"
        text = (r.text or "")[:30000]

    if not text.strip():
        return {"ok": False, "error": "no extractable content", "kind": kind, "url": final_url}

    # Send to Claude (via OAuth, no API charge)
    from oauth_oneshot import ask as oauth_ask

    prompt = (
        f"URL: {final_url}\n"
        f"CONTENT TYPE: {kind}\n"
        f"USER QUERY: {query}\n\n"
        f"CONTENT:\n{text[:25000]}\n\n"
        "Answer the user query precisely. If the content is a recipe, list "
        "title, ingredients (with quantities), and steps. If it's an "
        "article, summarize the key points. If it's a listing of events or "
        "items, return them as structured lines. Be concise but complete."
    )

    try:
        answer = await oauth_ask(prompt, model="haiku", timeout_s=45)
        if not answer:
            return {"ok": False, "error": "OAuth call returned empty response"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return {
        "ok": True,
        "url": final_url,
        "kind": kind,
        "content_chars": len(text),
        "answer": answer,
    }


# ─── Short-term memory (Benson's own working notes) ──────────────────
@_register(
    "stm_append",
    "Append a timestamped note to your short-term-memory. Use this "
    "AUTONOMOUSLY when something nontrivial happens: a tool call "
    "fails, Casey corrects you, you diagnose a problem, you successfully "
    "use a side-effect tool (propose_change, send_signal, announce, "
    "create_calendar_event), or you learn a household pattern. The "
    "topic name routes to a file: 'today' (default daily journal), "
    "'inbox' (scratch), or any topic under topics/ (lowercase letters/"
    "digits/underscores). Common topics: tool_caveats, proposal_outcomes, "
    "household_patterns, open_questions. STM auto-loads into your system "
    "prompt next turn — write here so future-you doesn't repeat today's "
    "mistakes. One short sentence per entry.",
    {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "today | inbox | tool_caveats | proposal_outcomes | household_patterns | open_questions | <new-snake_case-name>"},
            "content": {"type": "string", "description": "The note (plain text, one sentence is ideal)."},
        },
        "required": ["topic", "content"],
    },
)
async def _tool_stm_append(topic: str, content: str) -> dict:
    from short_term import stm_append
    return stm_append(topic=topic, content=content)


@_register(
    "stm_read",
    "Read your short-term-memory. With topic=None, returns aggregated "
    "recent files (the prompt already gets this — call only if you "
    "need the full unabridged version). With topic=<name>, returns the "
    "named topic's full content.",
    {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Optional: today | inbox | <topic name>"},
            "days_back": {"type": ["integer", "string"], "default": 1},
        },
        "required": [],
    },
)
async def _tool_stm_read(topic: str | None = None, days_back: int = 1) -> dict:
    from short_term import stm_read
    return stm_read(topic=topic, days_back=days_back)


@_register(
    "stm_list",
    "List all short-term-memory files with sizes + last-modified. Useful "
    "for sanity-checking what you've written and seeing if any topic is "
    "ballooning past its cap.",
    {"type": "object", "properties": {}, "required": []},
)
async def _tool_stm_list() -> dict:
    from short_term import stm_list
    return stm_list()


@_register(
    "stm_tidy",
    "Trigger a Sonnet-driven dedupe/merge pass over STM files that "
    "exceed their size cap. Use sparingly — this rewrites files. Each "
    "rewritten file keeps a .bak alongside.",
    {"type": "object", "properties": {}, "required": []},
)
async def _tool_stm_tidy() -> dict:
    from short_term import stm_tidy
    return await stm_tidy()


# ─── Self-awareness + self-modification ─────────────────────────────────
@_register(
    "read_my_conversations",
    "Read your own recent conversations from the local DB. Use this when "
    "reflecting on your own past behavior — what you said, what failed, "
    "what someone keeps having to repeat to you. Returns a list of "
    "(speaker, room, user_text, your_response, created_at) rows. Filter "
    "by speaker (Casey/Lindsey/Cole/Zander) and/or a substring search.",
    {
        "type": "object",
        "properties": {
            "days_back": {"type": ["integer", "string"], "default": 7, "minimum": 1, "maximum": 60},
            "speaker": {"type": "string", "description": "Optional. Filter to one person."},
            "search": {"type": "string", "description": "Optional case-insensitive substring filter on user_text or response."},
            "limit": {"type": ["integer", "string"], "default": 50, "minimum": 1, "maximum": 200},
        },
        "required": [],
    },
)
async def _tool_read_my_conversations(
    days_back: int = 7,
    speaker: str | None = None,
    search: str | None = None,
    limit: int = 50,
) -> dict:
    from self_modify import read_my_conversations
    return await read_my_conversations(
        days_back=days_back, speaker=speaker, search=search, limit=limit
    )


@_register(
    "read_my_logs",
    "Read your own service logs (journalctl -u benson). Use this when "
    "diagnosing why a tool failed or a request didn't go through. `since` "
    "accepts strings like '1 hour ago', 'today', '2026-04-28 12:00'. "
    "Defaults to the last 100 lines.",
    {
        "type": "object",
        "properties": {
            "lines": {"type": ["integer", "string"], "default": 100, "minimum": 1, "maximum": 500},
            "since": {"type": "string", "description": "Optional. Journalctl-style relative time."},
        },
        "required": [],
    },
)
async def _tool_read_my_logs(lines: int = 100, since: str | None = None) -> dict:
    from self_modify import read_my_logs
    return await read_my_logs(lines=lines, since=since)


@_register(
    "list_my_tools",
    "Return a list of every tool currently registered in this Benson "
    "instance — names + one-line descriptions. Use this when deciding "
    "whether a capability already exists before proposing a new one, "
    "or when describing your own surface to the household.",
    {"type": "object", "properties": {}, "required": []},
)
async def _tool_list_my_tools() -> dict:
    from self_modify import list_my_tools
    return await list_my_tools()


@_register(
    "read_my_source",
    "Read a file from your own source tree at /opt/benson. Use this when "
    "the user asks where something lives or whether a capability already "
    "exists — DO NOT claim something is missing without grepping/reading "
    "the source first. `path` may be relative ('middleware/data_api.py') "
    "or absolute under /opt/benson. Files larger than 500KB are refused; "
    "use grep_my_source for those.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to /opt/benson or absolute."},
            "max_lines": {"type": ["integer", "string"], "default": 400, "minimum": 1, "maximum": 2000},
        },
        "required": ["path"],
    },
)
async def _tool_read_my_source(path: str, max_lines: int = 400) -> dict:
    from self_modify import read_my_source
    return await read_my_source(path=path, max_lines=max_lines)


@_register(
    "grep_my_source",
    "Search your own source tree at /opt/benson with a regex. Use this "
    "FIRST when the user asks where something is implemented, whether a "
    "capability exists, or how some endpoint works. Default glob is "
    "'**/*.py' — pass 'middleware/templates/*.html' etc to scope. Skips "
    "venv, caches, model bundles, and user data automatically.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regex."},
            "path_glob": {"type": "string", "default": "**/*.py", "description": "Glob relative to /opt/benson."},
            "max_results": {"type": ["integer", "string"], "default": 60, "minimum": 1, "maximum": 300},
        },
        "required": ["pattern"],
    },
)
async def _tool_grep_my_source(
    pattern: str, path_glob: str = "**/*.py", max_results: int = 60
) -> dict:
    from self_modify import grep_my_source
    return await grep_my_source(pattern=pattern, path_glob=path_glob, max_results=max_results)


@_register(
    "write_local_file",
    "Save text to a file under /tmp/benson-*. Use this when the household "
    "asks you to capture logs, save a note, dump debug output, or similar. "
    "Path-locked to /tmp/benson-* — anything else is refused. For changes "
    "to your own source code, use propose_change instead. NEVER claim you "
    "saved a file without calling this tool and confirming ok=true.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Must start with '/tmp/benson-'."},
            "content": {"type": "string", "description": "File contents (max 1MB)."},
            "append": {"type": "boolean", "default": False, "description": "If true, append; otherwise overwrite."},
        },
        "required": ["path", "content"],
    },
)
async def _tool_write_local_file(path: str, content: str, append: bool = False) -> dict:
    from self_modify import write_local_file
    return await write_local_file(path=path, content=content, append=append)


@_register(
    "propose_change",
    "Open a self-modification proposal: spawn a coding session that "
    "edits Benson's own source in /opt/benson and commits to a fresh "
    "git branch. Casey reviews the rationale + diff on /admin/proposals "
    "and clicks merge to apply (auto-restart). ONLY call this AFTER "
    "Casey has confirmed your diagnosis in chat — see the DIAGNOSIS "
    "FLOW in your system prompt. Don't call this on speculation, on a "
    "follow-up about an existing proposal, or for stylistic tweaks.",
    {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": (
                    "Three labeled sections, written so the dashboard card is "
                    "self-explanatory:\n"
                    "  Intuition: what you think happened and why (cite "
                    "files/lines/log timestamps if you investigated).\n"
                    "  What I'd fix: the structural gap that allowed it — "
                    "not the incident itself.\n"
                    "  How: which file(s)/function(s)/tool(s)/memory shape "
                    "would change, and what new behavior emerges.\n"
                    "Ideally 4-10 lines total. The card title on "
                    "/admin/proposals is the rationale; if it's a slug, "
                    "Casey can't review without diving into the diff."
                ),
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Step-by-step edit plan for the SDK session: which "
                    "files, which functions, what to add/remove/rename, "
                    "what to leave alone. Be specific enough that someone "
                    "reading just this could implement it."
                ),
            },
        },
        "required": ["rationale", "instructions"],
    },
)
async def _tool_propose_change(rationale: str, instructions: str) -> dict:
    from self_modify import propose_change
    return await propose_change(rationale=rationale, instructions=instructions)


# ─── Strip legacy DB-backed memory tools (replaced by file-based memory_*) ─
_LEGACY_MEMORY_TOOLS = {
    "search_memory",
    "remember_this",
    "forget_memory",
    "update_memory",
}
TOOLS = [t for t in TOOLS if t["name"] not in _LEGACY_MEMORY_TOOLS]
for _n in _LEGACY_MEMORY_TOOLS:
    IMPL.pop(_n, None)


def list_tool_names() -> list[str]:
    return [t["name"] for t in TOOLS]
