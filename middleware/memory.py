"""Benson's persistent memory with pgvector semantic search.

Stores and retrieves conversational facts, household knowledge, and
extracted insights. Embeds with BAAI/bge-large-en-v1.5; searches via
pgvector cosine distance over an HNSW index.

The embedding model is loaded lazily and held in module state for the
lifetime of the process.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, PG_DSN

logger = logging.getLogger("benson.memory")

_model: Optional[SentenceTransformer] = None


def _embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _conn():
    return psycopg2.connect(**PG_DSN)


class MemoryStore:
    """Async-friendly wrapper. The actual DB calls are sync (psycopg2)
    but we run them in a threadpool so they don't block the event loop."""

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        return await asyncio.to_thread(self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[dict]:
        emb = _embedding_model().encode(query).tolist()
        with _conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, content, speaker, room, source, importance,
                           created_at,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM memories
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (emb, emb, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    async def store(
        self,
        content: str,
        speaker: str | None = None,
        room: str | None = None,
        source: str = "voice",
        importance: float = 0.5,
    ) -> int:
        return await asyncio.to_thread(
            self._store_sync, content, speaker, room, source, importance
        )

    def _store_sync(
        self,
        content: str,
        speaker: str | None,
        room: str | None,
        source: str,
        importance: float,
    ) -> int:
        emb = _embedding_model().encode(content).tolist()
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories
                        (content, embedding, source, speaker, room, importance)
                    VALUES (%s, %s::vector, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (content, emb, source, speaker, room, importance),
                )
                conn.commit()
                return cur.fetchone()[0]

    async def log_conversation(
        self,
        speaker: str | None,
        room: str | None,
        user_text: str,
        benson_response: str,
        tier: str,
    ) -> int:
        return await asyncio.to_thread(
            self._log_conversation_sync,
            speaker,
            room,
            user_text,
            benson_response,
            tier,
        )

    def _log_conversation_sync(
        self,
        speaker: str | None,
        room: str | None,
        user_text: str,
        benson_response: str,
        tier: str,
    ) -> int:
        with _conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations
                        (speaker, room, user_text, benson_response, tier)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (speaker, room, user_text, benson_response, tier),
                )
                conn.commit()
                return cur.fetchone()[0]

