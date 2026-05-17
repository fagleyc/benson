"""Wyoming TTS server fronting Benson's in-process Kokoro model.

Speaks the Wyoming protocol on a plain TCP socket. HA's Assist pipeline
sees this as just another wyoming TTS engine; under the hood, every
synthesis runs through Benson's existing kokoro_tts module, so the
voice/speed picked in /advanced/voice apply equally to:
  - Sonos announcements (via speak_on_zone)
  - HA Assist responses (via this server)

Wyoming framing (one message per JSON line + optional follow-ups):
    {"type": "<event>", ..., "data_length": N?, "payload_length": M?}\n
    [N bytes of UTF-8 JSON data]
    [M bytes of binary payload]
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import kokoro_tts
from voice_config import load as load_voice_cfg

logger = logging.getLogger("benson.wyoming_kokoro")

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 10201
SERVICE_NAME = "benson-kokoro"


# ─── Wyoming wire helpers ───────────────────────────────────────────────
async def _send_event(
    writer: asyncio.StreamWriter,
    event_type: str,
    data: dict[str, Any] | None = None,
    payload: bytes | None = None,
) -> None:
    header: dict[str, Any] = {"type": event_type}
    data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else b""
    if data_bytes:
        header["data_length"] = len(data_bytes)
    if payload:
        header["payload_length"] = len(payload)
    writer.write((json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"))
    if data_bytes:
        writer.write(data_bytes)
    if payload:
        writer.write(payload)
    await writer.drain()


async def _read_event(
    reader: asyncio.StreamReader,
) -> tuple[str, dict[str, Any], bytes] | None:
    line = await reader.readline()
    if not line:
        return None
    try:
        header = json.loads(line.decode("utf-8").strip())
    except Exception as e:
        logger.warning(f"bad header: {e}")
        return None
    et = header.get("type") or ""
    dlen = int(header.get("data_length") or 0)
    plen = int(header.get("payload_length") or 0)
    data: dict[str, Any] = {}
    if dlen:
        raw = await reader.readexactly(dlen)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.warning(f"bad data: {e}")
    payload = await reader.readexactly(plen) if plen else b""
    # Inline data on the header (some clients)
    inline = header.get("data")
    if isinstance(inline, dict):
        data = {**inline, **data}
    return et, data, payload


# ─── Info describing this service ───────────────────────────────────────
def _info_event_data() -> dict[str, Any]:
    voices = []
    for v in kokoro_tts.list_voices():
        # Map voice prefix → language tag (best-effort)
        prefix = v[0].lower()
        lang = {
            "a": "en_US", "b": "en_GB", "e": "es_ES", "f": "fr_FR",
            "h": "hi_IN", "i": "it_IT", "j": "ja_JP", "p": "pt_BR", "z": "zh_CN",
        }.get(prefix, "en_US")
        voices.append({
            "name": v,
            "description": f"Kokoro {v}",
            "attribution": {"name": "hexgrad", "url": "https://huggingface.co/hexgrad/Kokoro-82M"},
            "installed": True,
            "version": "1.0",
            "languages": [lang],
        })
    return {
        "tts": [
            {
                "name": SERVICE_NAME,
                "description": "Benson Kokoro TTS (CUDA)",
                "attribution": {"name": "hexgrad", "url": "https://huggingface.co/hexgrad/Kokoro-82M"},
                "installed": True,
                "version": "1.0",
                "voices": voices,
                "supports_synthesize_streaming": True,
            }
        ],
        "asr": [],
        "wake": [],
        "intent": [],
        "handle": [],
        "satellite": None,
    }


# ─── Per-connection handler ─────────────────────────────────────────────
async def _handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    peer = writer.get_extra_info("peername")
    logger.info(f"wyoming-kokoro: connection from {peer}")
    try:
        while True:
            evt = await _read_event(reader)
            if evt is None:
                return
            event_type, data, _payload = evt
            if event_type == "describe":
                await _send_event(writer, "info", _info_event_data())
            elif event_type == "synthesize":
                await _handle_synthesize(writer, data)
            elif event_type in ("audio-start", "audio-chunk", "audio-stop"):
                continue  # we don't accept audio
            else:
                logger.debug(f"ignoring event type={event_type}")
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass
    except Exception:
        logger.exception("client handler crashed")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _handle_synthesize(
    writer: asyncio.StreamWriter, data: dict[str, Any]
) -> None:
    text = (data.get("text") or "").strip()
    if not text:
        await _send_event(writer, "audio-start", {
            "rate": kokoro_tts.SAMPLE_RATE, "width": 2, "channels": 1, "timestamp": 0,
        })
        await _send_event(writer, "audio-stop", {"timestamp": 0})
        return
    # Resolve voice/speed from voice_config (so /advanced/voice changes
    # apply to satellite voice instantly).
    cfg = load_voice_cfg()
    voice = (data.get("voice") or {}).get("name") if isinstance(data.get("voice"), dict) else data.get("voice")
    voice = voice or cfg.get("voice") or kokoro_tts.DEFAULT_VOICE
    speed = float(cfg.get("speed", kokoro_tts.DEFAULT_SPEED))

    rate = kokoro_tts.SAMPLE_RATE
    t0 = time.time()
    first_chunk = True
    n_chunks = 0
    n_bytes = 0
    try:
        async for sr, pcm in kokoro_tts.synth_stream(text, voice=voice, speed=speed):
            if first_chunk:
                await _send_event(writer, "audio-start", {
                    "rate": sr, "width": 2, "channels": 1, "timestamp": 0,
                })
                first_chunk = False
                rate = sr
            # Wyoming clients expect chunks below ~64KB; split big ones.
            CHUNK = 16384
            for i in range(0, len(pcm), CHUNK):
                slab = pcm[i : i + CHUNK]
                ts_ms = int((n_bytes / 2) * 1000 / rate)  # 16-bit mono
                await _send_event(
                    writer,
                    "audio-chunk",
                    {"rate": rate, "width": 2, "channels": 1, "timestamp": ts_ms},
                    payload=slab,
                )
                n_bytes += len(slab)
                n_chunks += 1
        if first_chunk:
            # Kokoro produced nothing — emit empty stream
            await _send_event(writer, "audio-start", {
                "rate": rate, "width": 2, "channels": 1, "timestamp": 0,
            })
        total_ms = int((n_bytes / 2) * 1000 / rate) if rate else 0
        await _send_event(writer, "audio-stop", {"timestamp": total_ms})
        logger.info(
            f"synth ok voice={voice} text_len={len(text)} chunks={n_chunks} "
            f"audio_ms={total_ms} wall_ms={int((time.time()-t0)*1000)}"
        )
    except Exception:
        logger.exception("synth failed")
        try:
            await _send_event(writer, "audio-stop", {"timestamp": 0})
        except Exception:
            pass


# ─── Server lifecycle ───────────────────────────────────────────────────
_server_task: asyncio.Task | None = None


async def _serve() -> None:
    server = await asyncio.start_server(_handle_client, LISTEN_HOST, LISTEN_PORT)
    sockets = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    logger.info(f"wyoming-kokoro listening on {sockets}")
    async with server:
        await server.serve_forever()


def start() -> None:
    """Spawn the server task on the current asyncio loop."""
    global _server_task
    if _server_task and not _server_task.done():
        return
    _server_task = asyncio.create_task(_serve())
