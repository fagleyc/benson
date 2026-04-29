"""Vector-indexed deep memory.

Sweeps the household's structured data (conversations, calendar events,
recipes, completed chores) and the file-based memory, embeds each item
with bge-large, and stores in a pgvector-indexed table for fast
semantic retrieval.

Two layers of memory:
  Tier A (always-loaded): /opt/benson/memory/<person>.md — wholesale
    in the system prompt every turn (curated personal facts).
  Tier B (this module): vector-indexed deep history — searched on
    demand via the search_history agent tool when the agent needs
    context older than the last 12 turns or beyond curated facts.

Indexer is idempotent — keyed on (source_type, source_id) UNIQUE so
re-runs are safe. Embeddings are generated only for new/changed rows.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from config import PG_DSN
from memory import _embedding_model

logger = logging.getLogger("benson.memory_index")

MEMORY_DIR = Path("/opt/benson/memory")


def _conn():
    return psycopg2.connect(**PG_DSN)


def _embed(text: str):
    """Returns a list[float] of length 1024 (bge-large)."""
    if not text or not text.strip():
        return None
    model = _embedding_model()
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vec.tolist()


def _embed_to_vector_literal(vec: list[float] | None) -> str | None:
    """Format a vector as the pgvector literal string '[...]'."""
    if vec is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _upsert(rows: list[tuple]) -> int:
    """rows: (source_type, source_id, speaker, title, content, occurred_at, metadata, embedding_literal)"""
    if not rows:
        return 0
    with _conn() as c, c.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO memory_index
                (source_type, source_id, speaker, title, content, occurred_at, metadata, embedding)
            VALUES %s
            ON CONFLICT (source_type, source_id) DO UPDATE SET
                speaker = EXCLUDED.speaker,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                occurred_at = EXCLUDED.occurred_at,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """,
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s, %s::vector)",
        )
    return len(rows)


def _existing_source_ids(source_type: str) -> set[str]:
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT source_id FROM memory_index WHERE source_type = %s",
            (source_type,),
        )
        return {r[0] for r in cur.fetchall()}


# ─── Source-specific indexers ────────────────────────────────────────────
def index_conversations(min_chars: int = 30) -> int:
    """Index every conversation row not already present."""
    seen = _existing_source_ids("conversation")
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, speaker, room, user_text, benson_response, tier, created_at
            FROM conversations
            WHERE LENGTH(COALESCE(user_text, '') || COALESCE(benson_response, '')) >= %s
            ORDER BY created_at DESC
            LIMIT 5000
            """,
            (min_chars,),
        )
        rows = cur.fetchall()
    new_rows = []
    from psycopg2.extras import Json
    for r in rows:
        sid = str(r["id"])
        if sid in seen:
            continue
        text = (
            f"USER ({r['speaker']} via {r['room'] or 'unknown'}): "
            f"{(r['user_text'] or '').strip()}\n"
            f"BENSON: {(r['benson_response'] or '').strip()}"
        )
        emb = _embed(text)
        if emb is None:
            continue
        new_rows.append((
            "conversation",
            sid,
            r["speaker"],
            f"{r['speaker']} chat ({r['room']})",
            text[:2000],
            r["created_at"],
            Json({"tier": r["tier"], "room": r["room"]}),
            _embed_to_vector_literal(emb),
        ))
    return _upsert(new_rows)


def index_calendar_events() -> int:
    seen = _existing_source_ids("event")
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT user_name, google_event_id, person, calendar_summary,
                   title, description, location, starts_at, ends_at, all_day
            FROM calendar_events
            ORDER BY starts_at DESC
            LIMIT 2000
            """
        )
        rows = cur.fetchall()
    from psycopg2.extras import Json
    new_rows = []
    for r in rows:
        sid = f"{r['user_name']}:{r['google_event_id']}"
        if sid in seen:
            continue
        bits = [r["title"] or ""]
        if r["location"]: bits.append(f"at {r['location']}")
        if r["description"]: bits.append(r["description"][:500])
        text = " | ".join(b for b in bits if b)
        if not text.strip():
            continue
        emb = _embed(text)
        if emb is None:
            continue
        new_rows.append((
            "event",
            sid,
            r["person"] or r["user_name"],
            r["title"],
            text[:1500],
            r["starts_at"],
            Json({
                "calendar": r["calendar_summary"],
                "location": r["location"],
                "all_day": r["all_day"],
                "ends_at": r["ends_at"].isoformat() if r["ends_at"] else None,
            }),
            _embed_to_vector_literal(emb),
        ))
    return _upsert(new_rows)


def index_recipes() -> int:
    seen = _existing_source_ids("recipe")
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, title, course, prep_time, ingredients, notes, user_comments, user_rating, created_at FROM recipes"
        )
        rows = cur.fetchall()
    from psycopg2.extras import Json
    new_rows = []
    for r in rows:
        sid = str(r["id"])
        if sid in seen:
            continue
        ingredients = r["ingredients"] or []
        ing_text = ""
        if isinstance(ingredients, list):
            ing_text = ", ".join(
                (i.get("text") if isinstance(i, dict) else str(i))
                for i in ingredients[:30]
            )
        bits = [r["title"] or "(untitled)", r["course"] or "", ing_text]
        if r["notes"]: bits.append(r["notes"][:300])
        if r["user_comments"]: bits.append(f"comments: {r['user_comments'][:300]}")
        if r["user_rating"]: bits.append(f"rated {r['user_rating']}/5")
        text = " | ".join(b for b in bits if b)
        emb = _embed(text)
        if emb is None:
            continue
        new_rows.append((
            "recipe",
            sid,
            None,
            r["title"],
            text[:2000],
            r["created_at"],
            Json({"course": r["course"], "prep_time": r["prep_time"], "rating": r["user_rating"]}),
            _embed_to_vector_literal(emb),
        ))
    return _upsert(new_rows)


def index_chores() -> int:
    """Index completed chores (history) + open chores (current state)."""
    seen = _existing_source_ids("chore")
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, person, chore_name, chore_date, done FROM chores WHERE chore_name IS NOT NULL"
        )
        rows = cur.fetchall()
    from psycopg2.extras import Json
    new_rows = []
    for r in rows:
        sid = str(r["id"])
        if sid in seen:
            continue
        prefix = "Completed" if r["done"] else "Open"
        text = f"{prefix} chore for {r['person']}: {r['chore_name']}"
        if r["chore_date"]:
            text += f" on {r['chore_date'].isoformat()}"
        emb = _embed(text)
        if emb is None:
            continue
        new_rows.append((
            "chore",
            sid,
            r["person"],
            r["chore_name"],
            text[:600],
            r["chore_date"],
            Json({"done": r["done"]}),
            _embed_to_vector_literal(emb),
        ))
    return _upsert(new_rows)


def index_events() -> int:
    """Index every memory_events row not already present."""
    seen = _existing_source_ids("event_log")
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, occurred_at, category, person, content, metadata "
            "FROM memory_events ORDER BY occurred_at DESC LIMIT 5000"
        )
        rows = cur.fetchall()
    from psycopg2.extras import Json
    new_rows = []
    for r in rows:
        sid = str(r["id"])
        if sid in seen:
            continue
        bits = [r["category"]]
        if r["person"]:
            bits.append(f"({r['person']})")
        bits.append(r["content"] or "")
        text = " ".join(b for b in bits if b)
        emb = _embed(text)
        if emb is None:
            continue
        new_rows.append((
            "event_log",
            sid,
            r["person"],
            f"{r['category']} {('— ' + r['person']) if r['person'] else ''}".strip(),
            text[:1500],
            r["occurred_at"],
            Json({"category": r["category"], "metadata": r["metadata"]}),
            _embed_to_vector_literal(emb),
        ))
    return _upsert(new_rows)


def index_list_items() -> int:
    """Index every list item not already present."""
    seen = _existing_source_ids("list_item")
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT i.id, i.content, i.added_at, i.added_by, i.done, i.metadata, "
            "       l.name AS list_name, l.title AS list_title "
            "FROM memory_list_items i "
            "JOIN memory_lists l ON l.id = i.list_id "
            "WHERE l.archived_at IS NULL "
            "ORDER BY i.added_at DESC LIMIT 5000"
        )
        rows = cur.fetchall()
    from psycopg2.extras import Json
    new_rows = []
    for r in rows:
        sid = str(r["id"])
        if sid in seen:
            continue
        text = f"{r['list_title'] or r['list_name']}: {r['content']}"
        if r["done"]:
            text += " (done)"
        emb = _embed(text)
        if emb is None:
            continue
        new_rows.append((
            "list_item",
            sid,
            r["added_by"],
            r["list_title"] or r["list_name"],
            text[:1000],
            r["added_at"],
            Json({"list": r["list_name"], "done": r["done"], "metadata": r["metadata"]}),
            _embed_to_vector_literal(emb),
        ))
    return _upsert(new_rows)


def index_memory_files() -> int:
    """Re-index every chunk of /opt/benson/memory/*.md."""
    if not MEMORY_DIR.exists():
        return 0
    # Wipe-and-replace memory_file rows since files change in place
    with _conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM memory_index WHERE source_type = 'memory_file'")
    from psycopg2.extras import Json
    new_rows = []
    for p in sorted(MEMORY_DIR.rglob("*.md")):
        rel = p.relative_to(MEMORY_DIR).as_posix()
        text = p.read_text(errors="replace").strip()
        if not text:
            continue
        # Chunk by paragraph (memory files are short)
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            sid = f"{rel}#{i}"
            emb = _embed(chunk)
            if emb is None:
                continue
            person = p.stem.capitalize() if p.stem.lower() not in ("household", "index", "preferences") else None
            new_rows.append((
                "memory_file",
                sid,
                person,
                p.stem,
                chunk[:1500],
                None,
                Json({"path": rel}),
                _embed_to_vector_literal(emb),
            ))
    return _upsert(new_rows)


# ─── Public API ──────────────────────────────────────────────────────────
def reindex_all() -> dict:
    """Run every indexer. Returns counts per source."""
    counts = {}
    counts["conversations"] = index_conversations()
    counts["events"]        = index_calendar_events()
    counts["recipes"]       = index_recipes()
    counts["chores"]        = index_chores()
    counts["events"]        = index_events()
    counts["list_items"]    = index_list_items()
    counts["memory_files"]  = index_memory_files()
    counts["total_indexed_rows"] = sum(counts.values())
    with _conn() as c, c.cursor() as cur:
        cur.execute("SELECT source_type, COUNT(*) FROM memory_index GROUP BY source_type")
        counts["totals_by_source"] = dict(cur.fetchall())
    return counts


def search(
    query: str,
    *,
    source_type: str | None = None,
    speaker: str | None = None,
    limit: int = 8,
    days_back: int | None = None,
) -> list[dict]:
    """Semantic search the memory index. Returns matching items with cosine
    distance (lower = better)."""
    if not query.strip():
        return []
    qemb = _embed(query)
    if qemb is None:
        return []
    qlit = _embed_to_vector_literal(qemb)
    sql = (
        "SELECT id, source_type, source_id, speaker, title, content, "
        "occurred_at, metadata, embedding <=> %s::vector AS distance "
        "FROM memory_index WHERE TRUE"
    )
    params: list = [qlit]
    if source_type:
        sql += " AND source_type = %s"
        params.append(source_type)
    if speaker:
        sql += " AND speaker = %s"
        params.append(speaker)
    if days_back:
        sql += " AND (occurred_at IS NULL OR occurred_at > NOW() - INTERVAL '%s days')" % int(days_back)
    sql += " ORDER BY distance ASC LIMIT %s"
    params.append(limit)
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        return [dict(r) for r in cur.fetchall()]
