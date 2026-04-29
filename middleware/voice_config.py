"""Persistent voice configuration for Benson's TTS.

Stores active engine + voice in JSON under `/opt/benson/middleware/`.
Reload-on-read so updates from the /advanced/voice settings page take
effect on the next announce without a service restart.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("benson.voice_config")

CONFIG_PATH = Path("/opt/benson/middleware/voice_config.json")

# Engines we know about. Keys = engine ids; values = display info.
ENGINES = {
    "kokoro": {
        "label": "Kokoro (local, GPU, 24 kHz)",
        "default_voice": "bm_george",
        "voices_func": "kokoro_tts.list_voices",
    },
}

DEFAULTS: dict[str, Any] = {
    "engine": "kokoro",
    "voice": "bm_george",
    "speed": 1.0,
    "lang": "en-us",
}


def load() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return dict(DEFAULTS)
    try:
        d = json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        logger.warning(f"voice_config unreadable ({e}); using defaults")
        return dict(DEFAULTS)
    merged = dict(DEFAULTS)
    merged.update(d)
    return merged


def save(updates: dict[str, Any]) -> dict[str, Any]:
    cur = load()
    cur.update({k: v for k, v in updates.items() if k in DEFAULTS})
    CONFIG_PATH.write_text(json.dumps(cur, indent=2))
    return cur


def list_voices_for_engine(engine: str) -> list[str]:
    if engine == "kokoro":
        from kokoro_tts import list_voices
        return list_voices()
    return []
