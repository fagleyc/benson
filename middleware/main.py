"""Benson middleware: HTTP entrypoint.

Routes:
  GET  /health
  POST /conversation
  POST /recipe/photo       (body: {image_path})
  POST /recipe/video       (body: {url})
  POST /grocery/instacart  (body: {items: [str]})
  POST /memory/search      (body: {query, limit?})
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

import agent_session
import memory_hooks
from oauth_agent import run_agent
from claude_models import ModelTier, select as select_model
from config import PROMPT_PATH, configure_logging
from data_api import router as data_router
from db_tools import gather_context
from ha_client import call_service as ha_call_service
from ha_intents import (
    Action,
    ComposeAndAnnounce,
    detect as detect_intent,
    execute as execute_intent,
)
from hub import router as hub_router
from instacart import InstacartClient
from memory import MemoryStore
from output_routing import pick_speak_zone
from recipes import RecipeIngester

configure_logging()
logger = logging.getLogger("benson.main")

app = FastAPI(title="Benson Middleware", version="0.1.0")
# music_router first so its new /api/music/play, /search, /stations endpoints
# take precedence over the legacy ones in data_api at the same paths.
from music_handler import router as _music_router_early, ensure_schema as _music_ensure_schema
app.include_router(_music_router_early)
app.include_router(data_router)
app.include_router(hub_router)

from signal_handler import router as signal_router, start_poller as start_signal_poller
app.include_router(signal_router)

from google_handler import router as google_router, start_sync_loop as start_google_sync
app.include_router(google_router)

from scheduled_actions import ensure_schema as _sa_ensure_schema, start_worker as start_scheduler
from self_modify import ensure_autofix_schema as _autofix_ensure_schema

from camera_handler import router as camera_router
app.include_router(camera_router)

from listening_handler import router as listening_router
app.include_router(listening_router)

from user_config import router as user_config_router
app.include_router(user_config_router)

from wyoming_whisper import router as wyoming_whisper_router
app.include_router(wyoming_whisper_router)


@app.on_event("startup")
async def _signal_startup():
    start_signal_poller()
    start_google_sync()
    _sa_ensure_schema()
    _autofix_ensure_schema()
    _music_ensure_schema()
    start_scheduler()
    from wyoming_kokoro import start as start_wyoming_kokoro
    start_wyoming_kokoro()
    from wyoming_whisper import start as start_wyoming_whisper
    start_wyoming_whisper()

# Static audio for Kokoro TTS — Sonos fetches WAVs from here.
from fastapi.staticfiles import StaticFiles
from kokoro_tts import AUDIO_DIR as _AUDIO_DIR
app.mount("/audio", StaticFiles(directory=str(_AUDIO_DIR)), name="audio")
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)
# Custom microWakeWord models — fetched by the ReSpeaker on first boot
# after a flash and cached on the ESP32-S3 flash partition.
app.mount(
    "/microwakeword",
    StaticFiles(directory="/opt/benson/microwakeword/models"),
    name="microwakeword",
)

memory = MemoryStore()
recipes = RecipeIngester()
instacart = InstacartClient()


def _system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text()
    return (
        "You are Benson, the household AI assistant for the Fagley home. "
        "Speak warmly, concisely, and slightly dryly — like a butler-engineer."
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "benson"}


@app.post("/conversation")
async def handle_conversation(request: Request) -> dict[str, Any]:
    data = await request.json()
    user_text = data.get("text", "").strip()
    if not user_text:
        raise HTTPException(400, "text required")
    speaker = data.get("speaker") or None
    room = data.get("room") or None
    # voice_input → after the response is composed, also TTS it back to
    # the user via the appropriate Sonos zone (chosen by output_routing).
    voice_input = bool(data.get("voice_input", False))
    output_zone_override = data.get("output_zone")  # e.g. "media_player.kitchen"
    # When a request comes from a satellite (assist_satellite.*), prefer
    # its on-device speaker so a grouped-Sonos music station in the same
    # room doesn't get interrupted by the TTS reply. The HA conversation
    # agent should plumb this through; absent that, output_routing falls
    # back to the room→satellite mapping.
    satellite_id = data.get("satellite_id") or None

    # Try a direct HA-control intent first.
    intent = detect_intent(user_text)
    if isinstance(intent, Action):
        confirmation = await execute_intent(intent)
        await memory.log_conversation(
            speaker, room, user_text, confirmation, "ha_action"
        )
        return {"response": confirmation, "tier": "ha_action"}

    if isinstance(intent, ComposeAndAnnounce):
        # Hand the compose-and-announce job to the agent. The agent can:
        #   - look up live data via tools (search_memory, get_weather,
        #     search_recipes, etc.) before composing,
        #   - call the announce tool, which checks zone availability first,
        #   - recover gracefully if the target speaker is unavailable
        #     (suggest an alternate zone, ask the user).
        agent_prompt = (
            f"Compose and play a Sonos announcement on {intent.zone_entity} "
            f"({intent.zone_label}) based on this user request:\n\n"
            f"\"{intent.compose_prompt}\"\n\n"
            f"Use the available tools to look up any live information you "
            f"need (memories, weather, chores, recipes), compose plain "
            f"spoken text in Benson's voice (no stage directions, no "
            f"markdown, three sentences max for routine asks), then call "
            f"the `announce` tool with zone_entity_id="
            f"\"{intent.zone_entity}\" and your composed message. If "
            f"announce fails (e.g. zone unavailable), tell the user "
            f"what happened and suggest an alternate zone."
        )
        response, tier, _meta = await run_agent(
            agent_prompt,
            speaker=speaker,
            room=room,
            system_prompt=_system_prompt(),
        )
        tier = f"ha_compose_announce[{tier}]"

        # Extract the actual spoken text from the agent's tool calls so
        # 'what did you say' later returns the real announcement instead
        # of the wrapper line. Casey 2026-04-30: Benson said he sent a
        # Move announcement but couldn't recall the text — because we
        # only logged 'Sent it to the Move' as benson_response.
        spoken: list[str] = []
        for tc in (_meta.get("tool_calls") or []):
            if tc.get("name") == "announce":
                msg = (tc.get("input") or {}).get("message", "").strip()
                if msg:
                    spoken.append(msg)
        if spoken:
            logged_response = (
                response.rstrip()
                + "\n\n[Spoken on "
                + (intent.zone_label or intent.zone_entity)
                + ': "'
                + " | ".join(spoken)
                + '"]'
            )
        else:
            logged_response = response

        await memory.log_conversation(speaker, room, user_text, logged_response, tier)
        return {"response": response, "tier": tier}

    # Derive the output channel so the agent can suppress double-TTS:
    #   - satellite: HA's Assist pipeline plays it (voice_input=False from
    #     benson_agent + a satellite_id present)
    #   - sonos:     this handler will play via speak_on_zone after return
    #     (voice_input=True from web hub / Signal voice)
    #   - signal:    typed reply over Signal (room=None, voice_input=False)
    #   - hub:       typed reply in browser (room=None, voice_input=False)
    if satellite_id:
        output_channel = "satellite"
    elif voice_input:
        output_channel = "sonos"
    elif speaker and not room:
        output_channel = "signal"
    else:
        output_channel = "hub"

    # Chat path → Claude agent (tools + sessions). Falls back to Ollama on
    # Anthropic API failure inside run_agent.
    response, tier, _meta = await run_agent(
        user_text,
        speaker=speaker,
        room=room,
        system_prompt=_system_prompt(),
        output_channel=output_channel,
        satellite_id=satellite_id,
    )

    # Voice-input → speak the response. Preference order:
    #   1) explicit output_zone_override (admin / debugging),
    #   2) the room's Sonos zone via pick_speak_zone (kitchen → kitchen
    #      Sonos; unknown / hub → Move first per Casey 2026-05-23 rule).
    #
    # We do NOT route to the assist_satellite entity directly: that's
    # not a media_player, so play_media fails silently. The kitchen
    # ReSpeaker has no connected speaker yet anyway — when it does,
    # HA's own Assist pipeline already plays TTS through it and the
    # benson_agent integration can set voice_input=False to avoid
    # double-playback.
    spoken_on: str | None = None
    if voice_input:
        target = output_zone_override
        if not target:
            target, _chain = await pick_speak_zone(room)
        if target:
            try:
                from kokoro_tts import speak_on_zone
                result = await speak_on_zone(target, response)
                if result.get("ok"):
                    spoken_on = target
                else:
                    logger.warning(f"voice-output failed on {target}: {result.get('error')}")
            except Exception as e:
                logger.warning(f"voice-output TTS failed on {target}: {e}")

    await memory.log_conversation(speaker, room, user_text, response, tier)
    # Don't await extraction — fire-and-log
    try:
        # Auto-extraction disabled 2026-04-26 — replaced by file-based memory
        # tools (memory_list/read/write/append) that the agent curates itself.
        # Deterministic stop-hook (2026-05-10): a haiku call distills 0-2
        # durable facts from `response` into STM out-of-band. Gated by
        # MEMORY_STOP_HOOK_ENABLED (default on); the hook itself is
        # exception-safe so a failed extraction can't affect the turn.
        asyncio.create_task(memory_hooks.session_stop_hook(response))
    except Exception as e:
        logger.warning(f"Memory extraction skipped: {e}")

    out: dict[str, Any] = {"response": response, "tier": tier}
    if voice_input:
        out["spoken_on"] = spoken_on
    return out


@app.post("/recipe/photo")
async def ingest_recipe_photo(request: Request) -> dict[str, Any]:
    data = await request.json()
    image_path = data.get("image_path")
    if not image_path:
        raise HTTPException(400, "image_path required")
    p = Path(image_path)
    if not p.exists():
        raise HTTPException(404, f"image not found: {p}")
    recipe = await recipes.from_image(p)
    return {"recipe": recipe}


@app.post("/recipe/video")
async def ingest_recipe_video(request: Request) -> dict[str, Any]:
    data = await request.json()
    url = data.get("url")
    if not url:
        raise HTTPException(400, "url required")
    recipe = await recipes.from_video_url(url)
    return {"recipe": recipe}


@app.post("/grocery/instacart")
async def send_to_instacart(request: Request) -> dict[str, Any]:
    data = await request.json()
    items = data.get("items") or []
    if not items:
        raise HTTPException(400, "items required (list of strings)")
    link = await instacart.create_shopping_link(items)
    return {"instacart_link": link}


@app.post("/memory/search")
async def memory_search(request: Request) -> dict[str, Any]:
    data = await request.json()
    query = data.get("query", "").strip()
    if not query:
        raise HTTPException(400, "query required")
    limit = int(data.get("limit", 5))
    results = await memory.search(query, limit=limit)
    return {"results": results}


@app.post("/memory/store")
async def memory_store(request: Request) -> dict[str, Any]:
    data = await request.json()
    content = data.get("content", "").strip()
    if not content:
        raise HTTPException(400, "content required")
    new_id = await memory.store(
        content,
        speaker=data.get("speaker"),
        room=data.get("room"),
        source=data.get("source", "manual"),
        importance=float(data.get("importance", 0.5)),
    )
    return {"id": new_id}


@app.post("/agent/forget")
async def agent_forget(request: Request) -> dict[str, Any]:
    """Clear the running session for (speaker, room)."""
    data = await request.json()
    cleared = agent_session.forget(data.get("speaker"), data.get("room"))
    return {"cleared": cleared}


@app.get("/agent/sessions")
async def agent_sessions() -> dict[str, Any]:
    return {"sessions": agent_session.all_active()}
