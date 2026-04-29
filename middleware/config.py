"""Centralized config for Benson middleware.

Reads from /etc/benson/env via systemd's EnvironmentFile= (so values are
just os.environ at runtime). Provides typed accessors and the canonical
paths used by other modules.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path("/opt/benson")
CONTEXT_DIR = ROOT / "context"
PROMPT_PATH = ROOT / "middleware" / "benson_prompt.txt"
RECIPE_MEDIA_DIR = ROOT / "recipes"
LOG_DIR = ROOT / "logs"

# ─── Database ──────────────────────────────────────────────────────────
PG_DSN = {
    "dbname": os.environ.get("POSTGRES_DB", "benson"),
    "user": os.environ.get("POSTGRES_USER", "benson"),
    "password": os.environ.get("POSTGRES_PASSWORD", ""),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
}


# ─── Claude CLI (Tier 2) ───────────────────────────────────────────────
CLAUDE_CLI = "/usr/bin/claude"
CLAUDE_MODEL = "claude-opus-4-7"
CLAUDE_DEFAULT_EFFORT = "high"
CLAUDE_TIMEOUT_S = 240

# ─── Embedding model (memory layer) ────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# ─── Home Assistant ────────────────────────────────────────────────────
HA_BASE_URL = os.environ.get("HA_BASE_URL", "http://localhost:8123")
HA_TOKEN = os.environ.get("HA_LONG_LIVED_TOKEN", "")

# ─── Telegram ──────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_chat_ids_raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
TELEGRAM_ALLOWED_CHAT_IDS: set[int] = (
    {int(x) for x in _chat_ids_raw.split(",") if x.strip()} if _chat_ids_raw else set()
)

# ─── Anthropic API key (for vision and CLI) ────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── Logging ───────────────────────────────────────────────────────────
def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
