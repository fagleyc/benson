"""Wyoming STT server fronting Benson's in-process Whisper + ECAPA-TDNN.

Runs inside the uvicorn process (same pattern as wyoming_kokoro). HA's
Assist pipeline sees this as a normal wyoming STT engine; under the
hood, every transcription also runs ECAPA-TDNN over the same PCM,
identifies the speaker against the voiceprint store, and caches the
result so the benson_agent integration can attach `speaker=<name>`
when it POSTs /conversation moments later.

Wyoming framing (one JSON-line header + optional data + optional payload):
    {"type": "<event>", ..., "data_length": N?, "payload_length": M?}\n
    [N bytes of UTF-8 JSON data]
    [M bytes of binary payload]

A typical STT exchange from HA looks like:
    -> transcribe {language: "en"}
    -> audio-start {rate, width, channels}
    -> audio-chunk (payload=pcm) ... repeated
    -> audio-stop
    <- transcript {text}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile

import voiceprint

logger = logging.getLogger("benson.wyoming_whisper")

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 10301
SERVICE_NAME = "benson-whisper"
MODEL_ID = "openai/whisper-large-v3-turbo"

# ECAPA-TDNN identification thresholds.
SPEAKER_THRESHOLD = 0.45
SPEAKER_GAP = 0.08

# Speaker-cache TTL: how long an identified speaker stays "current"
# for a given room before the HA agent treats it as stale.
CACHE_TTL_S = 90.0

# Minimum audio length to bother running ECAPA over (skip wake words).
MIN_VOICE_SECONDS = 0.4

# Cap on audio buffered per Wyoming connection (30 s @ 16 kHz / 16-bit mono
# = 960 000 bytes). Anything past this is dropped + logged; protects the
# process from a misbehaving client streaming forever.
MAX_AUDIO_BYTES = 960_000

# ── Kitchen-mic enrollment ─────────────────────────────────────────────
# When set, post-wake PCM utterances from HA's pipeline are saved to the
# named user's raw voiceprint dir until samples_needed is reached. Cleared
# on completion, cancel, or expiry.
ENROLL_MIN_SECONDS = 3.0          # ignore short / aborted wakes
ENROLL_MAX_SECONDS = 30.0
ENROLL_TIMEOUT_S = 600.0           # auto-clear after 10 min idle
_enroll_lock = threading.Lock()
_enroll_state: Optional[dict[str, Any]] = None
# {"name": "casey", "needed": 3, "samples": [{"path", "duration_s", "recorded_at"}],
#  "started_at": <epoch>, "expires_at": <epoch>, "embeddings": [np.ndarray, ...]}


def _enroll_get() -> Optional[dict[str, Any]]:
    global _enroll_state
    with _enroll_lock:
        if not _enroll_state:
            return None
        if time.time() > _enroll_state["expires_at"]:
            logger.info(f"kitchen-mic enrollment expired for {_enroll_state['name']}")
            _enroll_state = None
            return None
        return dict(_enroll_state)


def _enroll_clear() -> None:
    global _enroll_state
    with _enroll_lock:
        _enroll_state = None


def _record_enrollment_sample(
    name: str, pcm: bytes, sample_rate: int, duration_s: float
) -> None:
    """Save a post-wake PCM utterance as a voice sample for the named user.

    Embeds via ECAPA, appends to the in-memory enrollment state, and when
    the quota is hit calls merge_voiceprint to fold the new samples into
    the running voiceprint.
    """
    global _enroll_state
    import uuid
    raw_dir = voiceprint.RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"kitchen_{uuid.uuid4().hex[:12]}.wav"
    wav_path = raw_dir / fname
    try:
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        emb = voiceprint.extract_embedding(wav_path)
    except Exception:
        logger.exception(f"enrollment sample save failed for {name}")
        try:
            wav_path.unlink()
        except OSError:
            pass
        return

    done = False
    with _enroll_lock:
        if not _enroll_state or _enroll_state["name"] != name:
            # State was cleared while we were embedding — drop the sample
            try:
                wav_path.unlink()
            except OSError:
                pass
            return
        _enroll_state["samples"].append({
            "path": str(wav_path),
            "duration_s": round(duration_s, 2),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        _enroll_state["embeddings"].append(emb)
        _enroll_state["expires_at"] = time.time() + ENROLL_TIMEOUT_S
        if len(_enroll_state["samples"]) >= _enroll_state["needed"]:
            done = True
        n = len(_enroll_state["samples"])
        needed = _enroll_state["needed"]
    logger.info(f"kitchen-mic enrollment: {name} sample {n}/{needed} ({duration_s:.1f}s)")

    if not done:
        return

    # Merge the new embeddings into the voiceprint + append to JSON metadata.
    with _enroll_lock:
        s = _enroll_state
        if not s:
            return
        embs = list(s["embeddings"])
        samples = list(s["samples"])
        _enroll_state = None

    try:
        avg, count = voiceprint.merge_voiceprint(name, embs)
        meta = voiceprint.load_meta(name) or {}
        existing_samples = meta.get("samples") or []
        meta["samples"] = existing_samples + samples
        meta["sample_count"] = count
        meta["last_updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta.setdefault("enrolled_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        meta.setdefault("display_name", name)
        meta.setdefault("name", name)
        voiceprint.write_meta(name, meta)
        logger.info(
            f"kitchen-mic enrollment complete: {name} merged {len(embs)} new samples "
            f"(total now {count})"
        )
    except Exception:
        logger.exception(f"merge_voiceprint failed at enrollment finish for {name}")


# ─── Speaker cache ──────────────────────────────────────────────────────
# Module-level caches consulted by the HTTP endpoint below. The Wyoming
# STT request from HA doesn't reliably attach a room, so we expose both
# a per-room map (best-effort, populated when transcribe metadata
# carries a hint) and a "latest overall" fallback so the agent can grab
# whoever spoke most recently.
_cache_lock = threading.Lock()
# Tuple shape: (speaker_name|None, ts_monotonic, best_similarity, margin)
LATEST_BY_ROOM: dict[str, tuple[Optional[str], float, float, float]] = {}
LATEST_OVERALL: tuple[Optional[str], float, float, float] = (None, 0.0, 0.0, 0.0)


def _record_speaker(
    room: Optional[str], speaker: Optional[str], best: float, margin: float
) -> None:
    global LATEST_OVERALL
    now = time.monotonic()
    with _cache_lock:
        LATEST_OVERALL = (speaker, now, best, margin)
        if room:
            LATEST_BY_ROOM[room.lower()] = (speaker, now, best, margin)


def _lookup_speaker(room: Optional[str]) -> dict[str, Any]:
    now = time.monotonic()
    with _cache_lock:
        rec: Optional[tuple[Optional[str], float, float, float]] = None
        if room:
            rec = LATEST_BY_ROOM.get(room.lower())
        if rec is None:
            rec = LATEST_OVERALL
    speaker, ts, best, margin = rec
    age = now - ts if ts > 0 else None
    if age is None or age > CACHE_TTL_S:
        return {
            "speaker": None,
            "age_s": age,
            "confidence": 0.0,
            "margin": 0.0,
            "stale": True,
        }
    return {
        "speaker": speaker,
        "age_s": round(age, 2),
        "confidence": round(best, 3),
        "margin": round(margin, 3),
        "stale": False,
    }


# ─── HTTP router (mounted into the FastAPI app) ─────────────────────────
router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.get("/latest-speaker")
async def latest_speaker(
    room: Optional[str] = Query(None, description="HA area name, lowercased"),
) -> dict[str, Any]:
    """Return whoever ECAPA-TDNN most recently identified.

    `speaker` is null when no recent identification (or below threshold).
    `age_s` is the seconds since the last identification (or null if none yet).
    `confidence` is the cosine-similarity margin (best - second best).
    """
    return _lookup_speaker(room)


@router.post("/enroll-start")
async def enroll_start(payload: dict[str, Any]) -> dict[str, Any]:
    """Begin a kitchen-mic enrollment session for {name}.

    Post-wake utterances from HA's pipeline will be saved as voice samples
    for the named user (in addition to being transcribed normally) until
    `needed` samples are collected or the session expires.
    """
    global _enroll_state
    name = (payload or {}).get("name") or ""
    needed = int((payload or {}).get("needed") or 3)
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(400, "name required")
    name = name.strip().lower()
    needed = max(1, min(needed, 6))
    md_path = Path("/opt/benson/memory") / f"{name}.md"
    if not md_path.exists():
        # No memory file → unknown household member. Refuse so we don't
        # quietly create a brand-new identity from kitchen audio.
        raise HTTPException(404, f"{name} is not a known household member")
    raw_dir = voiceprint.RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with _enroll_lock:
        _enroll_state = {
            "name": name,
            "needed": needed,
            "samples": [],
            "embeddings": [],
            "started_at": now,
            "expires_at": now + ENROLL_TIMEOUT_S,
        }
    logger.info(f"kitchen-mic enrollment started: {name} need={needed}")
    return {"ok": True, "name": name, "needed": needed, "expires_in_s": ENROLL_TIMEOUT_S}


@router.get("/enroll-status")
async def enroll_status() -> dict[str, Any]:
    s = _enroll_get()
    if not s:
        return {"active": False}
    return {
        "active": True,
        "name": s["name"],
        "needed": s["needed"],
        "collected": len(s["samples"]),
        "samples": s["samples"],
        "expires_in_s": max(0, int(s["expires_at"] - time.time())),
    }


@router.post("/enroll-cancel")
async def enroll_cancel() -> dict[str, Any]:
    s = _enroll_get()
    _enroll_clear()
    return {"ok": True, "was_active": bool(s)}


@router.post("/transcribe")
async def transcribe_upload(audio: UploadFile) -> dict[str, Any]:
    """Run whisper over an uploaded audio blob and return text.

    Browser MediaRecorder ships WebM/Ogg/MP4 chunks; ffmpeg normalizes
    to 16 kHz mono PCM WAV before handing to the same whisper model
    that serves HA's Assist pipeline.
    """
    import subprocess
    import tempfile

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "audio too large (max 20 MB)")

    in_suffix = (audio.filename or "").lower().split(".")[-1] or "webm"
    if in_suffix not in {"webm", "ogg", "m4a", "mp4", "wav", "mp3"}:
        in_suffix = "webm"

    with tempfile.NamedTemporaryFile(suffix=f".{in_suffix}", delete=False) as f_in:
        in_path = Path(f_in.name)
        f_in.write(raw)
    out_path = in_path.with_name(in_path.stem + "_16k.wav")

    def _convert_and_transcribe() -> str:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_path),
             "-ac", "1", "-ar", "16000", str(out_path)],
            capture_output=True, timeout=20,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"ffmpeg failed: {err}")
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        # PCM int16 LE — strip 44-byte WAV header.
        pcm = wav_bytes[44:]
        return _transcribe_pcm(pcm, 16000, "en")

    try:
        text = await asyncio.to_thread(_convert_and_transcribe)
    except Exception as e:
        logger.exception("transcribe failed")
        raise HTTPException(500, f"transcribe failed: {e}")
    finally:
        try:
            in_path.unlink()
        except OSError:
            pass
        try:
            out_path.unlink()
        except OSError:
            pass

    return {"text": text}


# ─── Whisper model (lazy-loaded on first STT call) ──────────────────────
_whisper_lock = threading.Lock()
_whisper_state: dict[str, Any] = {}


def _load_whisper() -> dict[str, Any]:
    if "model" in _whisper_state:
        return _whisper_state
    with _whisper_lock:
        if "model" in _whisper_state:
            return _whisper_state
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        t0 = time.time()
        # Reuse the same on-disk cache the systemd whisper service warmed.
        cache_dir = os.environ.get(
            "BENSON_WHISPER_CACHE", "/opt/benson/whisper/models"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, cache_dir=cache_dir, torch_dtype=dtype,
        ).to(device)
        model.eval()

        _whisper_state.update({
            "processor": processor,
            "model": model,
            "device": device,
            "dtype": dtype,
            "torch": torch,
        })
        logger.info(
            f"whisper loaded model={MODEL_ID} device={device} dtype={dtype} "
            f"in {time.time() - t0:.2f}s"
        )
        return _whisper_state


def _transcribe_pcm(pcm_int16: bytes, sample_rate: int, language: Optional[str]) -> str:
    """Run whisper-large-v3-turbo over a raw int16 PCM buffer."""
    state = _load_whisper()
    torch = state["torch"]
    processor = state["processor"]
    model = state["model"]
    device = state["device"]
    dtype = state["dtype"]

    audio = torch.frombuffer(pcm_int16, dtype=torch.int16).float() / 32768.0
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    # Cast inputs to model dtype/device (same patch we applied in the
    # systemd unit's transformers_whisper.py).
    inputs = {
        k: (v.to(device=device, dtype=dtype) if v.is_floating_point()
            else v.to(device))
        for k, v in inputs.items()
    }
    generate_args = {**inputs, "num_beams": 5}
    if language:
        try:
            generate_args["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )
        except Exception:
            pass

    with torch.no_grad():
        ids = model.generate(**generate_args)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text.strip()


# ─── ECAPA speaker identification with confidence margin ────────────────
def _identify_speaker(
    pcm_int16: bytes, sample_rate: int
) -> tuple[Optional[str], float, float]:
    """Run ECAPA-TDNN over the PCM, return (name|None, best_similarity, margin).

    Delegates the scoring to `voiceprint.identify_with_confidence` so we
    have one canonical place for thresholds and margin math.
    """
    duration_s = len(pcm_int16) / 2 / max(sample_rate, 1)
    if duration_s < MIN_VOICE_SECONDS:
        return None, 0.0, 0.0

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16)

        emb = voiceprint.extract_embedding(tmp_path)
        return voiceprint.identify_with_confidence(
            emb, threshold=SPEAKER_THRESHOLD, gap=SPEAKER_GAP
        )
    except Exception as e:
        logger.warning(f"ECAPA identify failed: {e}")
        return None, 0.0, 0.0
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


# ─── Wyoming wire helpers (mirroring wyoming_kokoro.py) ─────────────────
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
    inline = header.get("data")
    if isinstance(inline, dict):
        data = {**inline, **data}
    return et, data, payload


# ─── Info describing this service ───────────────────────────────────────
def _info_event_data() -> dict[str, Any]:
    return {
        "asr": [
            {
                "name": SERVICE_NAME,
                "description": "Benson in-process Whisper (large-v3-turbo) + ECAPA speaker ID",
                "attribution": {"name": "OpenAI / SpeechBrain", "url": "https://huggingface.co/openai/whisper-large-v3-turbo"},
                "installed": True,
                "version": "1.0",
                "models": [
                    {
                        "name": MODEL_ID,
                        "description": "Whisper large-v3-turbo (CUDA fp16)",
                        "attribution": {"name": "OpenAI", "url": "https://huggingface.co/openai/whisper-large-v3-turbo"},
                        "installed": True,
                        "languages": ["en"],
                        "version": "1.0",
                    }
                ],
            }
        ],
        "tts": [],
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
    logger.info(f"wyoming-whisper: connection from {peer}")

    pending_language: Optional[str] = None
    pending_room: Optional[str] = None
    audio_rate: int = 16000
    audio_width: int = 2
    audio_channels: int = 1
    audio_buf: list[bytes] = []

    try:
        while True:
            evt = await _read_event(reader)
            if evt is None:
                return
            event_type, data, payload = evt

            if event_type == "describe":
                await _send_event(writer, "info", _info_event_data())
                continue

            if event_type == "transcribe":
                pending_language = data.get("language") or pending_language
                # Best-effort room hint (HA Assist sometimes attaches
                # context). Empty/missing is the common case.
                pending_room = (
                    data.get("room")
                    or data.get("area")
                    or data.get("name")
                    or pending_room
                )
                audio_buf = []
                continue

            if event_type == "audio-start":
                audio_rate = int(data.get("rate") or 16000)
                audio_width = int(data.get("width") or 2)
                audio_channels = int(data.get("channels") or 1)
                audio_buf = []
                continue

            if event_type == "audio-chunk":
                if payload:
                    cur = sum(len(b) for b in audio_buf)
                    if cur + len(payload) <= MAX_AUDIO_BYTES:
                        audio_buf.append(payload)
                    elif cur < MAX_AUDIO_BYTES:
                        audio_buf.append(payload[: MAX_AUDIO_BYTES - cur])
                        logger.warning(
                            f"audio buffer hit {MAX_AUDIO_BYTES}B cap — "
                            f"truncating; peer={peer}"
                        )
                continue

            if event_type == "audio-stop":
                pcm = b"".join(audio_buf)
                audio_buf = []
                await _process_turn(
                    writer,
                    pcm=pcm,
                    sample_rate=audio_rate,
                    width=audio_width,
                    channels=audio_channels,
                    language=pending_language,
                    room=pending_room,
                )
                continue

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


async def _process_turn(
    writer: asyncio.StreamWriter,
    *,
    pcm: bytes,
    sample_rate: int,
    width: int,
    channels: int,
    language: Optional[str],
    room: Optional[str],
) -> None:
    """Run whisper + ECAPA over the buffered PCM, cache speaker, emit transcript."""
    t0 = time.time()

    # Wyoming clients should send 16 kHz / 16-bit / mono — but be defensive.
    if width != 2 or channels != 1:
        logger.warning(
            f"unexpected audio format width={width} channels={channels} — "
            f"trying transcription anyway"
        )

    text = ""
    speaker: Optional[str] = None
    best = 0.0
    margin = 0.0
    t_whisper = 0.0
    t_spk = 0.0
    try:
        loop = asyncio.get_event_loop()
        t_a = time.time()
        text = await loop.run_in_executor(
            None, _transcribe_pcm, pcm, sample_rate, language
        )
        t_whisper = time.time() - t_a

        if pcm:
            t_b = time.time()
            speaker, best, margin = await loop.run_in_executor(
                None, _identify_speaker, pcm, sample_rate
            )
            t_spk = time.time() - t_b
            _record_speaker(room, speaker, best, margin)

            # Kitchen-mic enrollment hook: if a session is active and this
            # utterance is long enough, save it as a voice sample for the
            # named user. Runs in addition to the normal STT path.
            enroll_s = _enroll_get()
            if enroll_s and pcm:
                duration_s = len(pcm) / 2 / max(sample_rate, 1)
                if ENROLL_MIN_SECONDS <= duration_s <= ENROLL_MAX_SECONDS:
                    await loop.run_in_executor(
                        None, _record_enrollment_sample, enroll_s["name"], pcm, sample_rate, duration_s
                    )
    except Exception:
        logger.exception("STT turn failed")

    # Always send a transcript event so the pipeline can continue.
    try:
        await _send_event(writer, "transcript", {"text": text})
    except Exception:
        logger.exception("failed to send transcript")

    logger.info(
        f"stt turn len_pcm={len(pcm)} sr={sample_rate} text_len={len(text)} "
        f"speaker={speaker or 'unknown'} best={best:.3f} margin={margin:.3f} "
        f"room={room or '-'} whisper={t_whisper*1000:.0f}ms "
        f"spk={t_spk*1000:.0f}ms total={int((time.time()-t0)*1000)}ms"
    )


# ─── Server lifecycle ───────────────────────────────────────────────────
_server_task: asyncio.Task | None = None


async def _serve() -> None:
    server = await asyncio.start_server(_handle_client, LISTEN_HOST, LISTEN_PORT)
    sockets = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    logger.info(f"wyoming-whisper listening on {sockets}")
    async with server:
        await server.serve_forever()


def start() -> None:
    """Spawn the server task on the current asyncio loop.

    Honors BENSON_DISABLE_WYOMING_STT=1 as an escape hatch.
    """
    global _server_task
    if os.environ.get("BENSON_DISABLE_WYOMING_STT") == "1":
        logger.info("wyoming-whisper disabled via BENSON_DISABLE_WYOMING_STT")
        return
    if _server_task and not _server_task.done():
        return
    _server_task = asyncio.create_task(_serve())
