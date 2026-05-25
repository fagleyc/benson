"""Homebrew Pandora — music dashboard backend.

Surface (everything mounted under /api/music/...):

  GET    /api/music/now-playing          — list of zones + state
  GET    /api/music/library/playlists    — MA library playlists (cached 2min)
  GET    /api/music/search?q=&type=      — proxy MA search
  GET    /api/music/similar?artist=      — similar-artist radio seed list
  POST   /api/music/play                 — play media on a zone
  POST   /api/music/transport            — play/pause/skip/prev/volume_set
  GET    /api/music/stations             — list saved stations
  POST   /api/music/stations             — create a station
  DELETE /api/music/stations/{id}        — delete a station
  POST   /api/music/stations/{id}/play   — start the station on a zone

The page (/music) and the rest of the app are intentionally additive: the
existing /api/music/players, /api/music/playlists, /api/music/search,
/api/music/play, /api/music/control endpoints in data_api.py stay
untouched so we never break anything callers already depend on.

Music Assistant capability notes:
  • `music_assistant.play_media` accepts a media_id (URI or free text) plus
    `radio_mode: true` to start a similar-artist station. That's our
    preferred "similar artist radio" path — single-shot, MA does the work.
  • `music_assistant.search` returns categorised results
    {artists: [...], albums: [...], tracks: [...], playlists: [...]}.
    There is no direct "get similar artists" service; we use search to
    surface artist hits and then start `radio_mode` from the chosen URI.
  • Stations are a Benson-side construct: a JSON seed bundle. To play a
    station we shuffle through the seed_artists (or run a search keyed by
    the first available facet), pick a track URI, and call play_media with
    radio_mode=true so MA fans it out. Documented fallback below.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import psycopg2
from fastapi import APIRouter, HTTPException, Request
from psycopg2.extras import Json, RealDictCursor

from config import PG_DSN
from ha_client import call_service as ha_call, get_state as ha_get_state

logger = logging.getLogger("benson.music_handler")

router = APIRouter()

_SCHEMA_PATH = Path(__file__).parent / "sql" / "music_stations.sql"


# ─── Sonos / MA zone map ────────────────────────────────────────────────
# Keep this synced with data_api._MASS_ZONES. Duplicated rather than
# imported so this module remains independently importable.
_ZONES: dict[str, dict[str, str]] = {
    "kitchen":        {"entity": "media_player.kitchen_2",       "label": "Kitchen"},
    "family_room":    {"entity": "media_player.family_room_2",   "label": "Family Room"},
    "tv_room":        {"entity": "media_player.tv_room_2",       "label": "TV Room"},
    "master_bedroom": {"entity": "media_player.bathroom_2",      "label": "Master Bedroom"},
    "move":           {"entity": "media_player.move_2",          "label": "Move (Patio)"},
    # Music-Assistant queue endpoint on the kitchen satellite.
    "respeaker_kitchen": {"entity": "media_player.respeaker_kitchen", "label": "Kitchen Satellite"},
}


def _entity_from_request(body: dict) -> str:
    entity = body.get("zone") or body.get("entity_id")
    if not entity:
        room = body.get("room")
        z = _ZONES.get(room or "")
        entity = z["entity"] if z else None
    if not entity:
        raise HTTPException(400, "zone / entity_id / room required")
    return entity


# ─── Schema bootstrap + default-station seeding ──────────────────────────
DEFAULT_STATIONS: list[dict] = [
    {
        "name": "80s Rock",
        "seeds": {
            "genres": ["Rock"], "decades": ["80s"], "moods": ["Upbeat"],
            "seed_artists": ["Bon Jovi", "Def Leppard", "Journey", "Heart"],
            "seed_tracks": [],
        },
        "cover_palette": {"hex_from": "#d35400", "hex_to": "#7d3c98"},
    },
    {
        "name": "Coffee + Focus",
        "seeds": {
            "genres": ["Jazz", "Folk/Acoustic"], "decades": [],
            "moods": ["Focused", "Mellow"],
            "seed_artists": ["Bill Evans", "Nick Drake", "Iron & Wine"],
            "seed_tracks": [],
        },
        "cover_palette": {"hex_from": "#34495e", "hex_to": "#16a085"},
    },
    {
        "name": "Kids' Breakfast",
        "seeds": {
            "genres": ["Pop", "Folk/Acoustic"], "decades": ["2010s", "2020s"],
            "moods": ["Upbeat"],
            "seed_artists": ["Jack Johnson", "Lumineers", "Vance Joy"],
            "seed_tracks": [],
        },
        "cover_palette": {"hex_from": "#f1c40f", "hex_to": "#e67e22"},
    },
    {
        "name": "Garage Workout",
        "seeds": {
            "genres": ["Rock", "Metal", "Hip-Hop"], "decades": ["90s", "2000s"],
            "moods": ["Workout", "Party"],
            "seed_artists": ["Metallica", "Rage Against the Machine", "Run-D.M.C."],
            "seed_tracks": [],
        },
        "cover_palette": {"hex_from": "#c0392b", "hex_to": "#2c3e50"},
    },
    {
        "name": "Dinner Mellow",
        "seeds": {
            "genres": ["Jazz", "R&B", "Folk/Acoustic"], "decades": [],
            "moods": ["Mellow", "Romantic"],
            "seed_artists": ["Norah Jones", "Diana Krall", "Chet Baker"],
            "seed_tracks": [],
        },
        "cover_palette": {"hex_from": "#5d4037", "hex_to": "#8e44ad"},
    },
    {
        "name": "Casey's Throwbacks",
        "seeds": {
            "genres": ["Rock", "Indie"], "decades": ["90s", "2000s"],
            "moods": ["Upbeat"],
            "seed_artists": ["Bush", "Foo Fighters", "Pearl Jam", "Weezer"],
            "seed_tracks": [],
        },
        "cover_palette": {"hex_from": "#2980b9", "hex_to": "#1abc9c"},
    },
    {
        "name": "Saturday Morning Cleanup",
        "seeds": {
            "genres": ["Pop", "Rock", "Indie"], "decades": ["2000s", "2010s"],
            "moods": ["Upbeat", "Party"],
            "seed_artists": ["Coldplay", "Imagine Dragons", "fun.", "Florence + The Machine"],
            "seed_tracks": [],
        },
        "cover_palette": {"hex_from": "#16a085", "hex_to": "#f39c12"},
    },
]


def ensure_schema() -> None:
    """Create the music_stations table + seed defaults on first boot.

    Called from main.py startup. Idempotent — the schema file uses
    CREATE TABLE IF NOT EXISTS, and the seed step only runs when the
    table is empty.
    """
    try:
        sql = _SCHEMA_PATH.read_text()
    except FileNotFoundError:
        logger.error(f"music_stations schema missing at {_SCHEMA_PATH}")
        return
    try:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(sql)
            cur.execute("SELECT count(*) FROM music_stations")
            (n,) = cur.fetchone()
            if n == 0:
                for st in DEFAULT_STATIONS:
                    cur.execute(
                        "INSERT INTO music_stations (name, seeds, cover_palette) "
                        "VALUES (%s, %s, %s)",
                        (st["name"], Json(st["seeds"]), Json(st["cover_palette"])),
                    )
                logger.info(f"music_stations: seeded {len(DEFAULT_STATIONS)} defaults")
            conn.commit()
    except Exception:
        logger.exception("music_handler: ensure_schema failed")


# ─── MA config-entry lookup (matches the helper in data_api.py) ──────────
async def _ma_entry_id() -> str | None:
    import httpx
    from config import HA_BASE_URL
    from ha_client import _headers
    async with httpx.AsyncClient(timeout=8) as client:
        resp = await client.get(
            f"{HA_BASE_URL}/api/config/config_entries/entry",
            headers=_headers(),
        )
    for e in resp.json():
        if e.get("domain") == "music_assistant":
            return e.get("entry_id")
    return None


# ─── Now-playing ─────────────────────────────────────────────────────────
@router.get("/api/music/now-playing")
async def now_playing() -> dict:
    """Per-zone snapshot for the now-playing strip."""
    out = []
    for room_id, info in _ZONES.items():
        entity = info["entity"]
        try:
            s = await ha_get_state(entity)
        except Exception as e:
            out.append({
                "room": room_id, "entity_id": entity, "label": info["label"],
                "state": "unavailable", "error": str(e)[:120],
            })
            continue
        a = s.get("attributes", {}) or {}
        out.append({
            "room": room_id,
            "entity_id": entity,
            "label": info["label"],
            "state": s.get("state"),
            "playing": s.get("state") in ("playing", "buffering"),
            "media_title": a.get("media_title"),
            "media_artist": a.get("media_artist"),
            "media_album_name": a.get("media_album_name"),
            "entity_picture": a.get("entity_picture"),
            "media_position": a.get("media_position"),
            "media_duration": a.get("media_duration"),
            "volume": a.get("volume_level"),
            "muted": a.get("is_volume_muted"),
        })
    return {"zones": out, "now": time.time()}


# ─── Library playlists (cached 2 min) ────────────────────────────────────
_PLAYLIST_CACHE: dict[str, Any] = {"at": 0.0, "items": []}
_PLAYLIST_TTL_S = 120


@router.get("/api/music/library/playlists")
async def library_playlists(refresh: bool = False) -> dict:
    now = time.time()
    if not refresh and (now - _PLAYLIST_CACHE["at"]) < _PLAYLIST_TTL_S and _PLAYLIST_CACHE["items"]:
        return {"playlists": _PLAYLIST_CACHE["items"], "cached": True}

    cfg_id = await _ma_entry_id()
    if not cfg_id:
        raise HTTPException(503, "Music Assistant not configured")
    try:
        result = await ha_call(
            "music_assistant", "get_library",
            {"config_entry_id": cfg_id, "media_type": "playlist",
             "favorite": False, "limit": 200, "offset": 0},
            timeout_s=20, return_response=True,
        )
    except Exception as e:
        raise HTTPException(502, f"MA get_library failed: {e}")
    sr = (result or {}).get("service_response", {}) or {}
    raw = sr.get("items", []) if isinstance(sr, dict) else []
    items = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        uri = it.get("uri", "")
        img = it.get("image")
        if not img:
            md = it.get("metadata") or {}
            imgs = md.get("images") or []
            if imgs and isinstance(imgs[0], dict):
                img = imgs[0].get("path")
        items.append({
            "name": it.get("name") or "(untitled)",
            "uri": uri,
            "image": img,
            "provider": uri.split("://")[0] if "://" in uri else (it.get("provider") or ""),
            "track_count": it.get("num_items") or it.get("track_count"),
            "favorite": bool(it.get("favorite", False)),
        })
    _PLAYLIST_CACHE["items"] = items
    _PLAYLIST_CACHE["at"] = now
    return {"playlists": items, "cached": False}


# ─── Search (categorised) ────────────────────────────────────────────────
@router.get("/api/music/search")
async def music_search_get(q: str, type: str = "artist,album,track,playlist", limit: int = 8) -> dict:
    q = (q or "").strip()
    if not q:
        return {"results": {}, "query": ""}
    cfg_id = await _ma_entry_id()
    if not cfg_id:
        raise HTTPException(503, "Music Assistant not configured")
    media_types = [t.strip() for t in type.split(",") if t.strip()]
    try:
        result = await ha_call(
            "music_assistant", "search",
            {
                "config_entry_id": cfg_id,
                "name": q,
                "media_type": media_types,
                "limit": max(1, min(int(limit), 20)),
                "library_only": False,
            },
            timeout_s=20, return_response=True,
        )
    except Exception as e:
        raise HTTPException(502, f"MA search failed: {e}")
    sr = (result or {}).get("service_response", {}) or {}
    return {"results": sr, "query": q, "media_types": media_types}


# ─── Similar artist radio seed ──────────────────────────────────────────
@router.get("/api/music/similar")
async def similar_artists(artist: str) -> dict:
    """Return MA artist hits for the given query.

    Music Assistant doesn't expose a `get_similar_artists` service in HA;
    we use `search` to surface artist URIs and rely on
    `play_media + radio_mode=true` (called from /api/music/play with
    `radio_mode=true`) to fan out into a similar-artist stream. The UI
    presents the top hits so Casey can pick the canonical artist before
    starting the radio.
    """
    artist = (artist or "").strip()
    if not artist:
        return {"artists": []}
    cfg_id = await _ma_entry_id()
    if not cfg_id:
        raise HTTPException(503, "Music Assistant not configured")
    try:
        result = await ha_call(
            "music_assistant", "search",
            {
                "config_entry_id": cfg_id,
                "name": artist,
                "media_type": ["artist"],
                "limit": 8,
                "library_only": False,
            },
            timeout_s=15, return_response=True,
        )
    except Exception as e:
        raise HTTPException(502, f"MA search failed: {e}")
    sr = (result or {}).get("service_response", {}) or {}
    artists = sr.get("artists") or []
    cleaned = []
    for a in artists:
        if not isinstance(a, dict):
            continue
        uri = a.get("uri", "")
        img = a.get("image")
        if not img:
            md = a.get("metadata") or {}
            imgs = md.get("images") or []
            if imgs and isinstance(imgs[0], dict):
                img = imgs[0].get("path")
        cleaned.append({
            "name": a.get("name") or "(unknown)",
            "uri": uri,
            "image": img,
            "provider": uri.split("://")[0] if "://" in uri else "",
        })
    return {"artists": cleaned, "query": artist, "radio_mode_supported": True}


# ─── Play / transport ───────────────────────────────────────────────────
@router.post("/api/music/play")
async def play_endpoint(request: Request) -> dict:
    body = await request.json()
    entity = _entity_from_request(body)
    media_id = (body.get("media_id") or body.get("media_id_or_query") or body.get("uri") or body.get("query") or "").strip()
    if not media_id:
        raise HTTPException(400, "media_id_or_query required")
    content_type = body.get("content_type") or body.get("media_type") or "playlist"
    queue_mode = body.get("queue_mode") or body.get("enqueue") or "replace"
    radio_mode = bool(body.get("radio_mode") or False)
    try:
        await ha_call(
            "music_assistant", "play_media",
            {
                "entity_id": entity,
                "media_id": media_id,
                "media_type": content_type,
                "enqueue": queue_mode,
                "radio_mode": radio_mode,
            },
            timeout_s=30,
        )
    except Exception as e:
        raise HTTPException(502, f"MA play_media failed: {e}")
    return {
        "ok": True, "entity_id": entity, "media_id": media_id,
        "media_type": content_type, "enqueue": queue_mode,
        "radio_mode": radio_mode,
    }


@router.post("/api/music/transport")
async def transport_endpoint(request: Request) -> dict:
    """play/pause/skip/prev/volume_set on a zone.
    Body: {zone|entity_id, action, [level]}."""
    body = await request.json()
    entity = _entity_from_request(body)
    action = (body.get("action") or "").strip()
    svc_map = {
        "play": "media_play",
        "pause": "media_pause",
        "stop": "media_stop",
        "skip": "media_next_track",
        "next": "media_next_track",
        "prev": "media_previous_track",
        "previous": "media_previous_track",
    }
    if action in ("volume_set", "volume"):
        level = float(body.get("level", 0.4))
        await ha_call("media_player", "volume_set",
                      {"entity_id": entity,
                       "volume_level": max(0.0, min(1.0, level))})
        return {"ok": True, "entity_id": entity, "action": "volume_set", "level": level}
    if action == "toggle":
        # Use HA's media_play_pause for true toggle semantics.
        await ha_call("media_player", "media_play_pause", {"entity_id": entity})
        return {"ok": True, "entity_id": entity, "action": "toggle"}
    if action not in svc_map:
        raise HTTPException(400, f"unknown action: {action}")
    await ha_call("media_player", svc_map[action], {"entity_id": entity})
    return {"ok": True, "entity_id": entity, "action": action}


# ─── Stations (CRUD + play) ─────────────────────────────────────────────
def _stations_query(sql: str, params: tuple = ()) -> list[dict]:
    with psycopg2.connect(**PG_DSN) as conn, conn.cursor(
        cursor_factory=RealDictCursor
    ) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _row_to_station(r: dict) -> dict:
    out = dict(r)
    # JSONB columns come back as dict already (psycopg2); strings only if
    # someone inserted as TEXT — tolerate that.
    for col in ("seeds", "cover_palette"):
        v = out.get(col)
        if isinstance(v, str):
            try:
                out[col] = json.loads(v)
            except Exception:
                pass
    # Stringify timestamps for JSON.
    for col in ("created_at", "last_played_at"):
        v = out.get(col)
        if v is not None and not isinstance(v, str):
            out[col] = v.isoformat(timespec="seconds")
    return out


@router.get("/api/music/stations")
async def stations_list() -> dict:
    rows = await asyncio.to_thread(
        _stations_query,
        "SELECT id, name, seeds, cover_palette, created_at, last_played_at, "
        "play_count FROM music_stations ORDER BY play_count DESC, name",
    )
    return {"stations": [_row_to_station(r) for r in rows]}


@router.post("/api/music/stations")
async def stations_create(request: Request) -> dict:
    body = await request.json()
    name = (body.get("name") or "").strip()
    seeds = body.get("seeds") or {}
    palette = body.get("cover_palette") or _palette_for(seeds)
    if not name:
        raise HTTPException(400, "name required")
    # Normalise + defensive defaults so the UI never has to send a full bundle.
    seeds = {
        "genres":       list(seeds.get("genres", []))[:12],
        "decades":      list(seeds.get("decades", []))[:8],
        "moods":        list(seeds.get("moods", []))[:8],
        "seed_artists": [a.strip() for a in seeds.get("seed_artists", []) if a and a.strip()][:20],
        "seed_tracks":  [t.strip() for t in seeds.get("seed_tracks", []) if t and t.strip()][:20],
    }

    def _insert() -> dict:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            cur.execute(
                "INSERT INTO music_stations (name, seeds, cover_palette) "
                "VALUES (%s, %s, %s) "
                "RETURNING id, name, seeds, cover_palette, created_at, "
                "last_played_at, play_count",
                (name, Json(seeds), Json(palette)),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return row

    row = await asyncio.to_thread(_insert)
    return {"station": _row_to_station(row)}


@router.delete("/api/music/stations/{station_id}")
async def stations_delete(station_id: int) -> dict:
    def _del() -> int:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM music_stations WHERE id = %s", (station_id,))
            n = cur.rowcount
            conn.commit()
            return n

    n = await asyncio.to_thread(_del)
    if not n:
        raise HTTPException(404, "station not found")
    return {"ok": True, "deleted": station_id}


def _palette_for(seeds: dict) -> dict:
    """Pick a deterministic gradient pair from the seed's first mood/genre.
    Keeps the cards visually varied even if the user didn't set one."""
    palette_choices = [
        {"hex_from": "#d35400", "hex_to": "#7d3c98"},
        {"hex_from": "#16a085", "hex_to": "#f39c12"},
        {"hex_from": "#2980b9", "hex_to": "#1abc9c"},
        {"hex_from": "#c0392b", "hex_to": "#2c3e50"},
        {"hex_from": "#5d4037", "hex_to": "#8e44ad"},
        {"hex_from": "#34495e", "hex_to": "#16a085"},
        {"hex_from": "#f1c40f", "hex_to": "#e67e22"},
    ]
    key = (
        ((seeds.get("moods") or [""])[0] or "")
        + "|" + ((seeds.get("genres") or [""])[0] or "")
        + "|" + ((seeds.get("decades") or [""])[0] or "")
    ) or "default"
    h = sum(ord(c) for c in key) % len(palette_choices)
    return palette_choices[h]


# ─── Station playback ────────────────────────────────────────────────────
async def _resolve_station_seed(seeds: dict) -> tuple[str, str] | None:
    """Pick a media_id + media_type from a station's seed bundle.

    Strategy (in order):
      1. If seed_tracks present → search the first one as a track and use
         its URI.
      2. If seed_artists present → pick a random one, search MA for the
         artist, use its URI with media_type=artist.
      3. Otherwise → build a free-text query of (genre + mood + decade)
         and search as a playlist.

    Returned URI is then handed to `music_assistant.play_media` with
    `radio_mode=true` so MA fans the seed out into a stream. This is the
    documented fallback for "no direct similar-artist endpoint."
    """
    cfg_id = await _ma_entry_id()
    if not cfg_id:
        return None

    async def _search(name: str, media_types: list[str]) -> dict | None:
        try:
            r = await ha_call(
                "music_assistant", "search",
                {"config_entry_id": cfg_id, "name": name,
                 "media_type": media_types, "limit": 5, "library_only": False},
                timeout_s=15, return_response=True,
            )
        except Exception as e:
            logger.warning(f"station seed search failed: {e}")
            return None
        return (r or {}).get("service_response") or {}

    # 1. seed_tracks
    tracks = list(seeds.get("seed_tracks") or [])
    random.shuffle(tracks)
    for t in tracks:
        sr = await _search(t, ["track"])
        hits = (sr or {}).get("tracks") or []
        if hits and isinstance(hits[0], dict) and hits[0].get("uri"):
            return hits[0]["uri"], "track"

    # 2. seed_artists
    artists = list(seeds.get("seed_artists") or [])
    random.shuffle(artists)
    for a in artists:
        sr = await _search(a, ["artist"])
        hits = (sr or {}).get("artists") or []
        if hits and isinstance(hits[0], dict) and hits[0].get("uri"):
            return hits[0]["uri"], "artist"

    # 3. fall back to a free-text playlist query
    bits: list[str] = []
    if seeds.get("genres"):
        bits.append(seeds["genres"][0])
    if seeds.get("moods"):
        bits.append(seeds["moods"][0])
    if seeds.get("decades"):
        bits.append(seeds["decades"][0])
    name = " ".join(bits).strip() or "mix"
    sr = await _search(name, ["playlist", "track"])
    for bucket in ("playlists", "tracks"):
        hits = (sr or {}).get(bucket) or []
        if hits and isinstance(hits[0], dict) and hits[0].get("uri"):
            return hits[0]["uri"], bucket[:-1]  # "playlists" → "playlist"
    return None


@router.post("/api/music/stations/{station_id}/play")
async def stations_play(station_id: int, request: Request) -> dict:
    body = await request.json()
    entity = _entity_from_request(body)

    row = await asyncio.to_thread(
        _stations_query,
        "SELECT id, name, seeds FROM music_stations WHERE id = %s",
        (station_id,),
    )
    if not row:
        raise HTTPException(404, "station not found")
    station = _row_to_station(row[0])
    seeds = station.get("seeds") or {}

    resolved = await _resolve_station_seed(seeds)
    if not resolved:
        raise HTTPException(
            502,
            "Couldn't resolve a seed for this station — try adding a seed "
            "artist or track.",
        )
    media_id, media_type = resolved

    try:
        await ha_call(
            "music_assistant", "play_media",
            {
                "entity_id": entity, "media_id": media_id,
                "media_type": media_type, "enqueue": "replace",
                "radio_mode": True,
            },
            timeout_s=30,
        )
    except Exception as e:
        raise HTTPException(502, f"MA play_media failed: {e}")

    def _bump() -> None:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE music_stations SET last_played_at = now(), "
                "play_count = play_count + 1 WHERE id = %s",
                (station_id,),
            )
            conn.commit()

    try:
        await asyncio.to_thread(_bump)
    except Exception:
        logger.warning("station play_count bump failed", exc_info=True)

    return {
        "ok": True, "station_id": station_id, "entity_id": entity,
        "seed_media_id": media_id, "seed_media_type": media_type,
        "radio_mode": True,
    }
