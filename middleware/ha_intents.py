"""Intent detection + dispatch — turns natural-language commands into HA service calls.

Approach: small set of regex patterns that match common asks. Each pattern
returns an Action (domain, service, data) and a confirmation string. If
no pattern matches, returns None and the caller falls through to LLM.

Deliberately simple. Better to catch the obvious cases reliably than try
to be a general-purpose tool-router. Hard cases (multi-step, conditional,
"but only if X") still go to the LLM.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ha_client import HAUnavailable, call_service, get_state

logger = logging.getLogger("benson.intents")


# ─── Room / device aliases ──────────────────────────────────────────────
# Maps spoken-name fragments to HA entity ID stems.
ROOM_FAN = {
    "master": "master_bedroom_fan",
    "master bedroom": "master_bedroom_fan",
    "masterbedroom": "master_bedroom_fan",
    "bedroom": "master_bedroom_fan",
    "office": "office_fan",
    "zander": "zanders_room_fan",
    "zanders": "zanders_room_fan",
    "zander's": "zanders_room_fan",
    "zander's room": "zanders_room_fan",
    "zanders room": "zanders_room_fan",
    "zandersroom": "zanders_room_fan",
    "kitchen": "kitchen_fan",
}
ROOM_SONOS = {
    "kitchen": "media_player.kitchen",
    "family room": "media_player.family_room",
    "family": "media_player.family_room",
    "tv room": "media_player.tv_room",
    "tv": "media_player.tv_room",
    "bathroom": "media_player.bathroom",
    "master": "media_player.bathroom",  # the "Bathroom" Sonos lives in master
    "master bedroom": "media_player.bathroom",
    "bedroom": "media_player.bathroom",
    "move": "media_player.move",
    "patio": "media_player.move",
    "deck": "media_player.move",
}

# Compiled regexes for the common command shapes.
_RE_FAN_ON_OFF = re.compile(
    r"\bturn\s+(?P<state>on|off)\s+(?:the\s+)?(?P<room>[\w'\s]+?)\s+fans?(?:\s+lights?)?\b",
    re.IGNORECASE,
)
_RE_LIGHT_ON_OFF = re.compile(
    r"\bturn\s+(?P<state>on|off)\s+(?:the\s+)?(?P<room>[\w'\s]+?)\s+(?:fan\s+)?lights?\b",
    re.IGNORECASE,
)
_RE_DIM = re.compile(
    r"\b(?:dim|set)\s+(?:the\s+)?(?P<room>[\w'\s]+?)\s+(?:fan\s+)?lights?(?:\s+to)?\s+(?P<pct>\d{1,3})\s*%?",
    re.IGNORECASE,
)
# Brightness keywords without a number: "full blast", "max", "all the way", "half", etc.
_RE_DIM_KEYWORD = re.compile(
    r"\b(?:turn\s+(?:on\s+)?|set|crank|brighten|dim)\s+(?:the\s+)?(?P<room>[\w'\s]+?)\s+(?:fan\s+)?lights?"
    r"(?:\s+(?:to|at|all\s+the\s+way))?\s+"
    r"(?P<level>full\s*blast|to\s+the\s+max|max(?:imum)?|full\s+brightness|full|bright|brightest|all\s+the\s+way|halfway|half|low|dim|dimmest)\b",
    re.IGNORECASE,
)
_DIM_KEYWORD_PCT = {
    "full blast": 100, "fullblast": 100, "to the max": 100, "max": 100,
    "maximum": 100, "full brightness": 100, "full": 100, "bright": 100,
    "brightest": 100, "all the way": 100,
    "halfway": 50, "half": 50,
    "low": 20, "dim": 20, "dimmest": 10,
}
_RE_FAN_SPEED = re.compile(
    r"\b(?:turn|set|spin|put|crank|kick)\s+(?:the\s+)?(?P<room>[\w'\s]+?)\s+fans?\s+(?:speed\s+)?(?:to\s+)?(?P<level>low|medium|med|high|max|off)\b",
    re.IGNORECASE,
)
_RE_SHADES = re.compile(
    r"\b(?P<action>open|close|raise|lower)\s+(?:the\s+)?(?:upstairs\s+|living\s+room\s+|main\s+door\s+)*shades?\b",
    re.IGNORECASE,
)
_RE_WATERFALL = re.compile(
    r"\bturn\s+(?P<state>on|off)\s+(?:the\s+)?(?:water\s*fall|fountain)\b",
    re.IGNORECASE,
)
_RE_PLAY_MUSIC = re.compile(
    r"\bplay\s+(?P<query>[\w\s]+?)\s+(?:in|on)\s+(?:the\s+)?(?P<room>[\w'\s]+?)\b",
    re.IGNORECASE,
)
_RE_PAUSE = re.compile(
    r"\b(pause|stop)\s+(?:the\s+)?(?P<room>[\w'\s]+?)(?:\s+(?:speakers?|music|sonos))?\b",
    re.IGNORECASE,
)
_RE_VOLUME = re.compile(
    r"\b(?:set\s+)?(?:the\s+)?(?P<room>[\w'\s]+?)\s+(?:volume|sound)\s+(?:to\s+)?(?P<pct>\d{1,3})\s*%?",
    re.IGNORECASE,
)
# announce / say through a Sonos zone
_RE_ANNOUNCE = re.compile(
    r"\b(?:announce|say|broadcast|tell|speak)\s+"
    r"(?:that\s+|to\s+(?:the\s+)?(?P<room1>kitchen|family\s*room|tv\s*room|bathroom|master(?:\s*bedroom)?|move|patio|deck)\s+(?:that\s+)?)?"
    r"(?P<msg>.+?)"
    r"(?:\s+(?:in|on|through|to|over)\s+(?:the\s+)?(?P<room2>kitchen|family\s*room|tv\s*room|bathroom|master(?:\s*bedroom)?|move|patio|deck)(?:\s+sonos)?)?$",
    re.IGNORECASE,
)
# everyone / all speakers
_RE_BROADCAST_ALL = re.compile(
    r"\b(?:announce|say|broadcast|tell\s+everyone|tell\s+the\s+(?:family|household))\s+(?:that\s+)?(?P<msg>.+?)(?:\s+(?:everywhere|to\s+everyone|on\s+all|in\s+the\s+whole\s+house|all\s+rooms))?$",
    re.IGNORECASE,
)


@dataclass
class Action:
    domain: str
    service: str
    data: dict
    confirmation: str  # what Benson should say after firing


@dataclass
class ComposeAndAnnounce:
    """Two-step: ask Llama to compose a short announcement using
    `compose_prompt` as the instruction, then TTS the result through
    `zone_entity`. Handled in main.handle_conversation."""
    zone_entity: str
    zone_label: str
    compose_prompt: str


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _resolve_room_fan(room_text: str) -> str | None:
    key = _norm(room_text)
    # exact match first
    if key in ROOM_FAN:
        return ROOM_FAN[key]
    # substring fallback
    for k, v in ROOM_FAN.items():
        if k in key:
            return v
    return None


def _resolve_room_sonos(room_text: str) -> str | None:
    key = _norm(room_text)
    if key in ROOM_SONOS:
        return ROOM_SONOS[key]
    for k, v in ROOM_SONOS.items():
        if k in key:
            return v
    return None


# Verbs that signal "make a Sonos announcement"
_ANNOUNCE_VERBS = (
    "announce", "broadcast", "make an announcement",
    "tell everyone", "tell the family", "tell the household",
    "introduce", "speak", "talk", "greet", "share", "let everyone know",
    "let the family know", "say",
)
# Verbs whose use ALWAYS implies "compose, don't speak verbatim"
_ALWAYS_COMPOSE_VERBS = ("introduce", "speak", "talk", "greet", "share", "let everyone know", "let the family know")
# Verbs in the BODY that mean "Llama should compose first"
_GENERATIVE_BODY_VERBS = (
    "give", "write", "draft", "compose", "make up", "tell us",
    "introduce", "explain", "describe", "summarize", "create", "come up with",
    "retrieve", "look up", "check", "find", "get me", "fetch",
    "summarise", "share", "what's", "what is", "tell me",
)
# Phrases that mean "this is a request to compute, not a literal phrase"
_GENERATIVE_PREFIXES = (
    "can you", "could you", "would you", "please ",
    "what's", "what is", "what are", "how's", "how is",
    "tell me", "let me know",
)


def _find_sonos_zone(text: str) -> tuple[str, str] | None:
    """Return (entity_id, label) for the Sonos zone the user is targeting.

    Priority:
      1. Zone preceded by an explicit announce-route preposition
         (on / in / through / over / to) — that's the speaker the user
         wants to broadcast on.
      2. Bare zone mention as last-resort.

    Keys are checked longest-first within each tier so 'family room' wins
    over 'family', etc.
    """
    lower = text.lower()
    keys = sorted(ROOM_SONOS.keys(), key=len, reverse=True)
    # Tier 1: prepositional zone reference
    for label in keys:
        for prep in ("on the ", "in the ", "through the ", "over the ",
                     "on ", "in ", "through ", "over "):
            if f"{prep}{label}" in lower:
                return ROOM_SONOS[label], label
    # Tier 2: any "the {zone}" or bare zone mention
    for label in keys:
        if f"the {label}" in lower:
            return ROOM_SONOS[label], label
    for label in keys:
        if re.search(rf"\b{re.escape(label)}\b", lower):
            return ROOM_SONOS[label], label
    return None


def _strip_announce_clause(text: str, zone_label: str) -> str:
    """Remove the 'announce ... on the {zone}' wrapper, leaving the body."""
    # Strip leading verbs
    body = text.strip()
    for v in _ANNOUNCE_VERBS:
        if body.lower().startswith(v + " "):
            body = body[len(v) + 1:]
            break
    # Strip trailing zone phrase
    pat = re.compile(
        rf"\s*(?:and\s+)?(?:announce|broadcast|say|speak|tell|play)?\s*(?:it|this|that)?\s*"
        rf"(?:on|in|through|to|over)\s+(?:the\s+)?{re.escape(zone_label)}(?:\s+(?:sonos|speaker|speakers))?\s*\.?\s*$",
        re.IGNORECASE,
    )
    body = pat.sub("", body).strip(" .,;:!\"")
    return body


def _looks_generative(body: str) -> bool:
    low = body.lower().strip()
    if "?" in body:
        return True
    if any(low.startswith(p) for p in _GENERATIVE_PREFIXES):
        return True
    if any(low.startswith(v + " ") for v in _GENERATIVE_BODY_VERBS):
        return True
    if any(f" {v} " in f" {low} " for v in _GENERATIVE_BODY_VERBS):
        return True
    return len(body.split()) > 14


def _has_announce_verb(text: str) -> bool:
    low = text.lower()
    for v in _ANNOUNCE_VERBS:
        # word-boundary-ish: need verb followed by space or end
        if re.search(rf"\b{re.escape(v)}\b", low):
            return True
    return False


def _uses_always_compose_verb(text: str) -> bool:
    low = text.lower()
    return any(re.search(rf"\b{re.escape(v)}\b", low) for v in _ALWAYS_COMPOSE_VERBS)


def detect(text: str) -> Optional[Action | ComposeAndAnnounce]:
    """Return an Action if the text clearly matches a control intent."""
    if not text:
        return None

    # Waterfall on/off
    m = _RE_WATERFALL.search(text)
    if m:
        on = m.group("state").lower() == "on"
        return Action(
            domain="switch",
            service="turn_on" if on else "turn_off",
            data={"entity_id": "switch.waterfall"},
            confirmation=f"{'Starting' if on else 'Stopping'} the waterfall.",
        )

    # Shades open/close
    m = _RE_SHADES.search(text)
    if m:
        action = m.group("action").lower()
        opening = action in ("open", "raise")
        return Action(
            domain="cover",
            service="open_cover" if opening else "close_cover",
            data={"entity_id": "cover.upstairs_shades"},
            confirmation=f"{'Opening' if opening else 'Closing'} the upstairs shades.",
        )

    # Dim light to N% — must be checked before plain on/off because dim mentions a number
    m = _RE_DIM.search(text)
    if m:
        room = _resolve_room_fan(m.group("room"))
        if room:
            pct = max(1, min(100, int(m.group("pct"))))
            return Action(
                domain="light",
                service="turn_on",
                data={"entity_id": f"light.{room}", "brightness_pct": pct},
                confirmation=f"Dimming the {m.group('room').strip()} light to {pct}%.",
            )

    # Brightness keyword: "full blast", "max", "all the way", "half", "low", etc.
    m = _RE_DIM_KEYWORD.search(text)
    if m:
        room = _resolve_room_fan(m.group("room"))
        if room:
            level = " ".join(m.group("level").lower().split())
            pct = _DIM_KEYWORD_PCT.get(level, 100)
            return Action(
                domain="light",
                service="turn_on",
                data={"entity_id": f"light.{room}", "brightness_pct": pct},
                confirmation=f"Setting the {m.group('room').strip()} light to {pct}%.",
            )

    # Fan speed level
    m = _RE_FAN_SPEED.search(text)
    if m:
        room = _resolve_room_fan(m.group("room"))
        level = m.group("level").lower()
        if room:
            if level == "off":
                return Action(
                    domain="fan",
                    service="turn_off",
                    data={"entity_id": f"fan.{room}"},
                    confirmation=f"Turning off the {m.group('room').strip()} fan.",
                )
            pct = {"low": 33, "medium": 66, "med": 66, "high": 100, "max": 100}[level]
            return Action(
                domain="fan",
                service="set_percentage",
                data={"entity_id": f"fan.{room}", "percentage": pct},
                confirmation=f"Setting the {m.group('room').strip()} fan to {level}.",
            )

    # Fan on/off — explicit "fan" mentioned
    m = _RE_FAN_ON_OFF.search(text)
    if m:
        on = m.group("state").lower() == "on"
        room = _resolve_room_fan(m.group("room"))
        if room:
            # If "fan light" was said, target the light entity instead
            entity = (
                f"light.{room}"
                if "light" in m.group(0).lower()
                else f"fan.{room}"
            )
            domain = entity.split(".")[0]
            return Action(
                domain=domain,
                service="turn_on" if on else "turn_off",
                data={"entity_id": entity},
                confirmation=f"{'Turning on' if on else 'Turning off'} the {m.group('room').strip()} {domain}.",
            )

    # Light on/off — when "fan" not mentioned but "light" is
    m = _RE_LIGHT_ON_OFF.search(text)
    if m and " fan " not in f" {text.lower()} ":
        on = m.group("state").lower() == "on"
        room = _resolve_room_fan(m.group("room"))
        if room:
            return Action(
                domain="light",
                service="turn_on" if on else "turn_off",
                data={"entity_id": f"light.{room}"},
                confirmation=f"{'Turning on' if on else 'Turning off'} the {m.group('room').strip()} light.",
            )

    # Pause music
    m = _RE_PAUSE.search(text)
    if m:
        room = _resolve_room_sonos(m.group("room"))
        if room:
            return Action(
                domain="media_player",
                service="media_pause",
                data={"entity_id": room},
                confirmation=f"Pausing {m.group('room').strip()}.",
            )

    # Volume
    m = _RE_VOLUME.search(text)
    if m:
        room = _resolve_room_sonos(m.group("room"))
        if room:
            pct = max(0, min(100, int(m.group("pct"))))
            return Action(
                domain="media_player",
                service="volume_set",
                data={"entity_id": room, "volume_level": pct / 100.0},
                confirmation=f"Setting {m.group('room').strip()} volume to {pct}%.",
            )

    # ─── Sonos announce (verb + zone match) ──────────────────────────────
    # Detection is permissive: the user mentions an announce verb anywhere
    # AND a Sonos zone with a route preposition (on/in/through/over). Body
    # is whatever remains after stripping the announce clause. If the body
    # looks like a request to compose (starts with imperative verb, asks a
    # question, mentions lookup verbs) OR the user used a verb that always
    # implies composition (introduce, speak, etc.) → ComposeAndAnnounce.
    # Otherwise speak it verbatim.
    if _has_announce_verb(text):
        zone = _find_sonos_zone(text)
        if zone:
            entity, label = zone
            body = _strip_announce_clause(text, label)
            if body and len(body) > 2:
                if _uses_always_compose_verb(text) or _looks_generative(body):
                    return ComposeAndAnnounce(
                        zone_entity=entity,
                        zone_label=label,
                        compose_prompt=body,
                    )
                # Direct speak — strip "that " / "to the family" framing
                clean = re.sub(
                    r"^(that|to\s+(?:the\s+|us|everyone|the\s+family|the\s+household)\s*)\s+",
                    "",
                    body,
                    flags=re.IGNORECASE,
                ).strip(" \"'")
                if len(clean.split()) >= 2:
                    return Action(
                        domain="tts",
                        service="speak",
                        data={
                            "entity_id": "tts.piper",
                            "media_player_entity_id": entity,
                            "message": clean,
                        },
                        confirmation=f'Announcing on the {label} speaker: "{clean}"',
                    )

    # NOTE: "play X music in Y" is now handled by the agent's play_music
    # tool via Music Assistant + Apple Music. Removed the old regex hot
    # path so natural-language queries reach the agent.

    return None


async def execute(action: Action) -> str:
    """Fire the HA service call. Returns the confirmation string on success,
    or an error message string on failure (logged but not raised)."""
    # Special case: tts.speak announces should route through the unified
    # speak_on_zone dispatcher so they honor the active engine (Kokoro/Piper).
    if action.domain == "tts" and action.service == "speak":
        from kokoro_tts import speak_on_zone
        zone = action.data.get("media_player_entity_id")
        msg = action.data.get("message", "")
        if zone and msg:
            result = await speak_on_zone(zone, msg)
            if result.get("ok"):
                return action.confirmation
            return f"Couldn't reach the speaker: {result.get('error','unknown')}"

    try:
        await call_service(action.domain, action.service, action.data)
        logger.info(
            f"executed {action.domain}.{action.service} {action.data}"
        )
        return action.confirmation
    except HAUnavailable as e:
        logger.warning(f"HA action failed: {e}")
        return f"I tried to {action.confirmation.lower()} but Home Assistant didn't accept the call. ({e})"
    except Exception as e:
        logger.exception("HA action raised")
        return f"Something went wrong trying to {action.confirmation.lower()} ({type(e).__name__})."
