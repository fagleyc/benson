"""Local Kokoro TTS on the GB10 GPU.

Uses the PyTorch-based `kokoro` package (not kokoro-onnx) so we can
push the model to CUDA. ~80M params; first synth after load is ~1-2s,
subsequent synths are typically <500 ms on the Spark.

Architecture:
  - One shared `KModel` on CUDA, lazy-loaded.
  - One `KPipeline` per language code (Kokoro is language-specific for
    grapheme→phoneme conversion). Voice prefix maps to lang code:
      a*=American English (lang='a')
      b*=British English (lang='b')
      e*=Spanish, f*=French, h*=Hindi, i*=Italian,
      j*=Japanese, p*=Brazilian Portuguese, z*=Mandarin
  - `synth_to_file(text, voice, ...)` returns the filename of a fresh
    WAV under AUDIO_DIR. FastAPI serves it via /audio/{filename}.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("benson.kokoro")

# Phonemizer emits "words count mismatch" warnings on every punctuation-
# heavy line — harmless and floods the journal. Pin to ERROR.
logging.getLogger("phonemizer").setLevel(logging.ERROR)

AUDIO_DIR = Path("/opt/benson/middleware/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

RETAIN_S = 600  # delete WAVs older than this on each synth call
SAMPLE_RATE = 24000

DEFAULT_VOICE = "bm_george"
DEFAULT_SPEED = 1.0

# ─── Lazy globals ───────────────────────────────────────────────────────
_model = None
_pipelines: dict[str, object] = {}
_lock = asyncio.Lock()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _model
    if _model is None:
        from kokoro import KModel
        t0 = time.time()
        # Pin spaCy / huggingface offline-ish: use cache; allow downloads
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _model = KModel().to(_device()).eval()
        logger.info(f"Kokoro model loaded on {_device()} in {time.time() - t0:.2f}s")
    return _model


def _pipeline_for(lang_code: str):
    pipeline = _pipelines.get(lang_code)
    if pipeline is None:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code=lang_code, model=_load_model())
        _pipelines[lang_code] = pipeline
        logger.info(f"Kokoro pipeline ready for lang='{lang_code}'")
    return pipeline


def _voice_to_lang(voice: str) -> str:
    """Voice prefix → Kokoro lang_code."""
    if not voice:
        return "a"
    return voice[0].lower()  # 'b' for bm_george, 'a' for af_bella, etc.


def list_voices() -> list[str]:
    """Return all known Kokoro voice IDs.

    Kokoro doesn't expose a tidy enum, so we hard-code the v1 catalog
    here. Update if newer voices ship.
    """
    return [
        # American English (a)
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
        "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
        # British English (b)
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        # Spanish (e), French (f), Hindi (h), Italian (i), Japanese (j),
        # Brazilian Portuguese (p), Mandarin (z) — included for completeness.
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
        "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ]


def _cleanup_old() -> None:
    cutoff = time.time() - RETAIN_S
    for p in AUDIO_DIR.glob("*.wav"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


async def speak_on_zone(zone_entity_id: str, message: str) -> dict:
    """Single dispatcher: read voice_config, synth (Kokoro) or speak (Piper),
    fire the HA call, return a result dict.

    Used by all three call sites:
      - agent_tools.announce
      - ha_intents announce Action
      - main.py voice_input post-response TTS

    Dispatches by entity domain:
      - assist_satellite.* → assist_satellite.announce (on-device TTS),
        used by the voice-reply path so a satellite whose room's Sonos
        is grouped into a music station doesn't broadcast TTS across
        the whole group.
      - media_player.*    → Kokoro synth + media_player.play_media (Sonos).
    """
    from ha_client import call_service as ha_call_service, get_state as ha_get_state
    from voice_config import load as load_voice_cfg

    domain = (zone_entity_id or "").split(".", 1)[0]

    # On-device satellite path — let HA's assist_satellite.announce run
    # the announcement through the satellite's own speaker.
    if domain == "assist_satellite":
        try:
            await ha_call_service(
                "assist_satellite",
                "announce",
                {"entity_id": zone_entity_id, "message": message},
                timeout_s=30,
            )
        except Exception as e:
            logger.warning(f"assist_satellite.announce failed on {zone_entity_id}: {e}")
            return {"ok": False, "error": f"assist_satellite.announce failed: {e}"}
        return {
            "ok": True,
            "zone": zone_entity_id,
            "engine": "assist_satellite_announce",
            "spoken": message,
        }

    if domain != "media_player":
        return {"ok": False, "error": f"unsupported output domain: {zone_entity_id}"}

    # Precondition: zone available?
    try:
        st = await ha_get_state(zone_entity_id)
    except Exception as e:
        return {"ok": False, "error": f"could not read state of {zone_entity_id}: {e}"}
    state = st.get("state")
    if state in ("unavailable", "unknown", None):
        return {
            "ok": False,
            "error": f"{zone_entity_id} is currently {state}",
            "zone_state": state,
        }

    cfg = load_voice_cfg()
    try:
        fname = await synth_to_file(
            message,
            voice=cfg.get("voice", DEFAULT_VOICE),
            speed=float(cfg.get("speed", DEFAULT_SPEED)),
        )
    except Exception as e:
        logger.exception("Kokoro synth failed")
        return {"ok": False, "error": f"Kokoro synth failed: {e}"}
    url = f"http://192.168.0.240:8100/audio/{fname}"
    await ha_call_service(
        "media_player",
        "play_media",
        {
            "entity_id": zone_entity_id,
            "media_content_id": url,
            "media_content_type": "music",
            "announce": True,
        },
        timeout_s=30,
    )
    return {
        "ok": True, "zone": zone_entity_id, "engine": "kokoro",
        "voice": cfg.get("voice"), "spoken": message,
    }


async def synth_stream(
    text: str,
    voice: str = DEFAULT_VOICE,
    *,
    speed: float = DEFAULT_SPEED,
    lang: str | None = None,
):
    """Yield (sample_rate, pcm16_bytes) per phrase as Kokoro generates them.

    For the Wyoming TTS server — streaming a sentence-at-a-time lets the
    satellite begin playback well before the full utterance is synthesized.
    """
    async with _lock:
        lang_code = lang or _voice_to_lang(voice)
        loop = asyncio.get_running_loop()

        # Run the (blocking) Kokoro generator in a thread, push chunks into
        # a queue, and yield from the queue here.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def _producer():
            try:
                pipeline = _pipeline_for(lang_code)
                generator = pipeline(text, voice=voice, speed=speed)
                for _gs, _ps, audio in generator:
                    if audio is None:
                        continue
                    if hasattr(audio, "cpu"):
                        audio = audio.cpu().numpy()
                    pcm16 = np.clip(audio, -1.0, 1.0)
                    pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
                    loop.call_soon_threadsafe(queue.put_nowait, pcm16)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        producer = asyncio.create_task(asyncio.to_thread(_producer))
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    return
                if isinstance(item, Exception):
                    raise item
                yield SAMPLE_RATE, item
        finally:
            if not producer.done():
                producer.cancel()


async def synth_to_file(
    text: str,
    voice: str = DEFAULT_VOICE,
    *,
    speed: float = DEFAULT_SPEED,
    lang: str | None = None,
) -> str:
    """Synthesize text → WAV, return filename (not full path).

    Locked: ONNX/torch inference are not parallel-safe with one model.
    """
    async with _lock:
        _cleanup_old()
        lang_code = lang or _voice_to_lang(voice)

        def _do() -> str:
            pipeline = _pipeline_for(lang_code)
            generator = pipeline(text, voice=voice, speed=speed)
            chunks = []
            for _gs, _ps, audio in generator:
                if audio is None:
                    continue
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                chunks.append(audio)
            if not chunks:
                raise RuntimeError("Kokoro returned no audio")
            out = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            fname = f"benson_{uuid.uuid4().hex[:12]}.wav"
            path = AUDIO_DIR / fname
            sf.write(str(path), out, SAMPLE_RATE)
            return fname

        return await asyncio.to_thread(_do)
