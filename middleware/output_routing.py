"""Pick which Sonos zone(s) to use for speaking a response, given the
room the request came from.

Rules:
  - Known room with a bound Sonos → use that zone first.
  - Rooms with no Sonos (Cole's, Zander's, office) → fall back to a
    nearby zone that does have one.
  - Unknown / 'hub' / browser-from-anywhere → default chain is
    [Move (if available), Kitchen, Family Room].

The helper returns an ordered list. Caller tries each until one is
reachable (state != 'unavailable'/'unknown').
"""
from __future__ import annotations

import logging
from typing import Iterable

from ha_client import get_state as ha_get_state

logger = logging.getLogger("benson.routing")


# Room → ordered list of Sonos entities to try (first preferred).
# Keys are the `room` strings the frontend / satellites send us.
_ROOM_ZONES: dict[str, list[str]] = {
    "kitchen":         ["media_player.kitchen", "media_player.move"],
    "family_room":     ["media_player.family_room", "media_player.kitchen"],
    "tv_room":         ["media_player.tv_room", "media_player.kitchen"],
    "master_bedroom":  ["media_player.bathroom", "media_player.kitchen"],
    "bathroom":        ["media_player.bathroom"],   # the master ensuite zone
    "coles_room":      ["media_player.family_room", "media_player.kitchen"],
    "zanders_room":    ["media_player.family_room", "media_player.kitchen"],
    "office":          ["media_player.kitchen", "media_player.move"],
    "patio":           ["media_player.move", "media_player.kitchen"],
    "deck":            ["media_player.move", "media_player.kitchen"],
    "outdoor":         ["media_player.move", "media_player.kitchen"],
}

# Default chain when room is unknown or "hub" (browser from any LAN device).
DEFAULT_CHAIN = ["media_player.move", "media_player.kitchen", "media_player.family_room"]


# Room → on-device satellite output (assist_satellite.* entity).
# Used to keep voice replies private to the satellite that woke us, even
# when its room's Sonos is grouped into a music station. Sonos routing
# (pick_speak_zone) remains the fallback for unmiced rooms.
SATELLITES: dict[str, str] = {
    "kitchen": "assist_satellite.respeaker_kitchen_assist_satellite",
}


def _normalize_room(room: str | None) -> str:
    if not room:
        return ""
    return room.lower().replace(" ", "_").replace("-", "_").replace("'", "").strip()


def candidates_for_room(room: str | None) -> list[str]:
    norm = _normalize_room(room)
    if norm and norm in _ROOM_ZONES:
        return list(_ROOM_ZONES[norm])
    # Loose substring match for free-form rooms ("kitchen island", "den")
    for key, zones in _ROOM_ZONES.items():
        if key in norm:
            return list(zones)
    return list(DEFAULT_CHAIN)


async def first_available(zones: Iterable[str]) -> str | None:
    """Return the first zone whose HA state is not unavailable/unknown."""
    for z in zones:
        try:
            st = await ha_get_state(z)
        except Exception as e:
            logger.warning(f"state lookup failed for {z}: {e}")
            continue
        if st.get("state") not in ("unavailable", "unknown", None):
            return z
    return None


async def pick_speak_zone(room: str | None) -> tuple[str | None, list[str]]:
    """For a given input-room, pick the best currently-reachable Sonos.

    Returns (zone_entity_id_or_None, candidate_chain_tried).
    """
    chain = candidates_for_room(room)
    chosen = await first_available(chain)
    return chosen, chain


def satellite_for_room(room: str | None) -> str | None:
    """Return the assist_satellite entity bound to this room, or None."""
    norm = _normalize_room(room)
    if not norm:
        return None
    return SATELLITES.get(norm)


async def pick_satellite_output(
    room: str | None, satellite_id: str | None
) -> str | None:
    """Pick an on-device satellite output for the voice-reply path.

    Prefers an explicit `satellite_id` (when it's a non-empty
    `assist_satellite.*` entity); otherwise falls back to the room→
    satellite mapping. Does not probe availability — assist_satellite
    entities sit in 'idle'/'listening' and won't read as 'unavailable'
    the way Sonos zones do; if the caller cares, let `announce`
    surface the error.
    """
    if satellite_id and isinstance(satellite_id, str):
        sid = satellite_id.strip()
        if sid.startswith("assist_satellite."):
            return sid
    return satellite_for_room(room)
