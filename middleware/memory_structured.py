"""Structured memory: events + lists.

Complements the file-based memory tools (memory_list/read/append/write/delete
in memory_tools.py). MD files are for durable per-person facts; this module
handles two shapes that don't fit a file:

  EVENTS  — timestamped occurrences (workouts, meals, moods, observations).
            Append-only, queryable by date range + category + person.

  LISTS   — named collections of items (gift ideas, books to read, packing
            lists). Items can be checked off and added incrementally.

Both shapes are picked up by the nightly memory_index reindexer, so they're
also semantically searchable via search_history.
"""
from __future__ import annotations

import re
import logging
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor, Json

from config import PG_DSN

logger = logging.getLogger("benson.memory_structured")


def _conn():
    return psycopg2.connect(**PG_DSN)


def _slug(name: str) -> str:
    """Normalize a list name into a stable slug."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s-]+", "_", s)
    return s.strip("_") or "untitled"


# ─── Events ──────────────────────────────────────────────────────────────
def log_event(
    category: str,
    content: str,
    person: str | None = None,
    metadata: dict | None = None,
    source: str | None = None,
    occurred_at: str | None = None,
) -> dict:
    """Record a timestamped event. Returns the new row's id."""
    if not category or not content:
        return {"ok": False, "error": "category and content are required"}
    sql = (
        "INSERT INTO memory_events (category, person, content, metadata, source"
        + (", occurred_at" if occurred_at else "")
        + ") VALUES (%s, %s, %s, %s, %s"
        + (", %s" if occurred_at else "")
        + ") RETURNING id, occurred_at"
    )
    params: list[Any] = [
        category.strip().lower(),
        person.strip() if person else None,
        content.strip(),
        Json(metadata) if metadata else None,
        source,
    ]
    if occurred_at:
        params.append(occurred_at)
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
    return {
        "ok": True,
        "id": row["id"],
        "occurred_at": row["occurred_at"].isoformat(),
        "category": category.strip().lower(),
        "person": person,
    }


def query_events(
    category: str | None = None,
    person: str | None = None,
    days_back: int | None = None,
    limit: int = 50,
) -> dict:
    """Fetch events matching filters, newest first."""
    sql = (
        "SELECT id, occurred_at, category, person, content, metadata "
        "FROM memory_events WHERE TRUE"
    )
    params: list[Any] = []
    if category:
        sql += " AND category = %s"
        params.append(category.strip().lower())
    if person:
        sql += " AND person ILIKE %s"
        params.append(person.strip())
    if days_back:
        sql += " AND occurred_at > NOW() - (%s || ' days')::interval"
        params.append(int(days_back))
    sql += " ORDER BY occurred_at DESC LIMIT %s"
    params.append(int(limit))
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return {
        "ok": True,
        "count": len(rows),
        "events": [
            {
                "id": r["id"],
                "occurred_at": r["occurred_at"].isoformat() if r["occurred_at"] else None,
                "category": r["category"],
                "person": r["person"],
                "content": r["content"],
                "metadata": r["metadata"],
            }
            for r in rows
        ],
    }


def delete_event(event_id: int) -> dict:
    with _conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM memory_events WHERE id = %s", (event_id,))
        deleted = cur.rowcount
    return {"ok": deleted > 0, "id": event_id}


# ─── Lists ───────────────────────────────────────────────────────────────
def _get_or_create_list(
    name: str,
    title: str | None = None,
    description: str | None = None,
    created_by: str | None = None,
) -> dict:
    slug = _slug(name)
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, name, title, description FROM memory_lists WHERE name = %s",
            (slug,),
        )
        row = cur.fetchone()
        if row:
            return {"id": row["id"], "name": row["name"], "title": row["title"], "created": False}
        cur.execute(
            "INSERT INTO memory_lists (name, title, description, created_by) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (slug, title or name.strip(), description, created_by),
        )
        new_id = cur.fetchone()["id"]
    return {"id": new_id, "name": slug, "title": title or name.strip(), "created": True}


def list_add(
    name: str,
    item: str,
    title: str | None = None,
    added_by: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Append an item to a named list. Auto-creates the list if missing."""
    if not item or not item.strip():
        return {"ok": False, "error": "item required"}
    info = _get_or_create_list(name, title=title, created_by=added_by)
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO memory_list_items (list_id, content, metadata, added_by) "
            "VALUES (%s, %s, %s, %s) RETURNING id, added_at",
            (info["id"], item.strip(), Json(metadata) if metadata else None, added_by),
        )
        row = cur.fetchone()
    return {
        "ok": True,
        "list": info["name"],
        "list_created": info["created"],
        "item_id": row["id"],
        "added_at": row["added_at"].isoformat(),
    }


def list_read(name: str, include_done: bool = True) -> dict:
    """Read all items from a list, oldest first."""
    slug = _slug(name)
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT id, title, description FROM memory_lists WHERE name = %s",
            (slug,),
        )
        meta = cur.fetchone()
        if not meta:
            return {"ok": False, "error": f"no list named '{name}'"}
        sql = (
            "SELECT id, content, metadata, added_by, added_at, done, done_at "
            "FROM memory_list_items WHERE list_id = %s"
        )
        if not include_done:
            sql += " AND done = FALSE"
        sql += " ORDER BY added_at ASC"
        cur.execute(sql, (meta["id"],))
        rows = cur.fetchall()
    return {
        "ok": True,
        "name": slug,
        "title": meta["title"],
        "description": meta["description"],
        "count": len(rows),
        "items": [
            {
                "id": r["id"],
                "content": r["content"],
                "added_by": r["added_by"],
                "added_at": r["added_at"].isoformat() if r["added_at"] else None,
                "done": r["done"],
                "done_at": r["done_at"].isoformat() if r["done_at"] else None,
                "metadata": r["metadata"],
            }
            for r in rows
        ],
    }


def list_check(name: str, item_id: int, done: bool = True) -> dict:
    slug = _slug(name)
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE memory_list_items "
            "SET done = %s, done_at = CASE WHEN %s THEN NOW() ELSE NULL END "
            "WHERE id = %s AND list_id = (SELECT id FROM memory_lists WHERE name = %s)",
            (done, done, item_id, slug),
        )
        updated = cur.rowcount
    return {"ok": updated > 0, "list": slug, "item_id": item_id, "done": done}


def list_remove(name: str, item_id: int) -> dict:
    slug = _slug(name)
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "DELETE FROM memory_list_items "
            "WHERE id = %s AND list_id = (SELECT id FROM memory_lists WHERE name = %s)",
            (item_id, slug),
        )
        deleted = cur.rowcount
    return {"ok": deleted > 0, "list": slug, "item_id": item_id}


def list_all(include_archived: bool = False) -> dict:
    """List every named list with item counts."""
    sql = (
        "SELECT l.id, l.name, l.title, l.description, l.created_at, l.archived_at, "
        "       COUNT(i.id) FILTER (WHERE i.done = FALSE) AS open_items, "
        "       COUNT(i.id) AS total_items "
        "FROM memory_lists l "
        "LEFT JOIN memory_list_items i ON i.list_id = l.id "
    )
    if not include_archived:
        sql += "WHERE l.archived_at IS NULL "
    sql += "GROUP BY l.id ORDER BY l.created_at DESC"
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return {
        "ok": True,
        "count": len(rows),
        "lists": [
            {
                "name": r["name"],
                "title": r["title"],
                "description": r["description"],
                "open_items": int(r["open_items"]),
                "total_items": int(r["total_items"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ],
    }


def list_archive(name: str) -> dict:
    slug = _slug(name)
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE memory_lists SET archived_at = NOW() WHERE name = %s",
            (slug,),
        )
        updated = cur.rowcount
    return {"ok": updated > 0, "list": slug}
