"""Phase 1.3 — Embed and load seed memories from
`/opt/benson/context/data/seed_memories.json` into the Postgres
`memories` table.

Run after the schema is created and the embedding model is available.
Idempotent: skips memories whose content already exists in the table.

Usage:
    python3 phase1_load_seed_memories.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "context/data/seed_memories.json"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase1.seed_memories")


def _pg_conn():
    return psycopg2.connect(
        dbname=os.environ.get("POSTGRES_DB", "benson"),
        user=os.environ.get("POSTGRES_USER", "benson"),
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
    )


def main() -> int:
    if not SEED_PATH.exists():
        logger.error(f"Seed file not found: {SEED_PATH}")
        return 1
    with open(SEED_PATH) as f:
        data = json.load(f)
    memories = data.get("memories", [])
    logger.info(f"Loading {len(memories)} seed memories")

    logger.info(f"Loading embedding model {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    conn = _pg_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT content FROM memories WHERE source = 'seed'")
        existing = {row[0] for row in cur.fetchall()}
        new = [m for m in memories if m["content"] not in existing]
        if not new:
            logger.info("All seed memories already present; nothing to do.")
            return 0

        contents = [m["content"] for m in new]
        embeddings = model.encode(contents, show_progress_bar=False).tolist()

        rows = [
            (
                m["content"],
                emb,
                m.get("source", "seed"),
                m.get("speaker"),
                m.get("room"),
                float(m.get("importance", 0.5)),
            )
            for m, emb in zip(new, embeddings)
        ]
        execute_values(
            cur,
            """
            INSERT INTO memories
                (content, embedding, source, speaker, room, importance)
            VALUES %s
            """,
            rows,
            template="(%s, %s::vector, %s, %s, %s, %s)",
        )
        conn.commit()
        logger.info(f"Inserted {len(new)} seed memories")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
