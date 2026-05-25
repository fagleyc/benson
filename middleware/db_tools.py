"""Server-side RAG: detect household-data intents, query Postgres, return
a compact text block to inject into the system prompt before Ollama runs.

This is the fix for the "Benson confidently invents chores" failure mode.
Every request that mentions chores/recipes/weekly-plan gets the actual DB
state pulled into the prompt — Ollama composes the response, but it's
working from real data, not guessing.

Keep the intent detection deliberately simple. False positives are cheap
(extra context) and false negatives just mean Benson falls back to memory
search, which is no worse than today.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

from config import PG_DSN

logger = logging.getLogger("benson.db_tools")

_CHORE_RE = re.compile(r"\bchore", re.IGNORECASE)
_RECIPE_RE = re.compile(r"\b(recipe|cook|dinner|meal|menu|breakfast|lunch|dish)\b", re.IGNORECASE)
_PLAN_RE = re.compile(r"\b(this week|weekly plan|week's plan|this week's|tonight|tomorrow|weekly menu)\b", re.IGNORECASE)
_PERSON_RE = re.compile(r"\b(cole|zander|general|casey|lindsey)\b", re.IGNORECASE)
_WEATHER_RE = re.compile(r"\b(weather|temperature|temp|forecast|rain|snow|cold|hot|wind|humidity|outside)\b", re.IGNORECASE)


def _conn():
    return psycopg2.connect(**PG_DSN)


async def gather_context(user_text: str) -> str:
    """Return a multi-line context block keyed to whatever intents the
    user_text matches. Empty string if nothing matches."""
    blocks: list[str] = []
    if _CHORE_RE.search(user_text):
        blocks.append(await asyncio.to_thread(_chores_block, user_text))
    if _PLAN_RE.search(user_text) or (
        _RECIPE_RE.search(user_text) and "tonight" in user_text.lower()
    ):
        blocks.append(await asyncio.to_thread(_weekly_plan_block, user_text))
    if _RECIPE_RE.search(user_text) and not _PLAN_RE.search(user_text):
        blocks.append(await asyncio.to_thread(_recipes_block, user_text))
    if _WEATHER_RE.search(user_text):
        wb = await _weather_block()
        if wb:
            blocks.append(wb)
    return "\n\n".join(b for b in blocks if b)


async def _weather_block() -> str:
    """Pull live weather state from HA. Returns empty if HA isn't reachable."""
    try:
        from ha_client import get_state
        s = await get_state("weather.fagley_home")
    except Exception as e:
        logger.warning(f"weather lookup failed: {e}")
        return ""
    a = s.get("attributes", {})
    parts = [f"Live weather (HA, Open-Meteo for Colorado Springs):"]
    parts.append(f"  condition: {s.get('state')}")
    if "temperature" in a:
        parts.append(f"  temperature: {a['temperature']}{a.get('temperature_unit','')}")
    if "humidity" in a:
        parts.append(f"  humidity: {a['humidity']}%")
    if "wind_speed" in a:
        parts.append(f"  wind: {a['wind_speed']} {a.get('wind_speed_unit','')}")
    if "pressure" in a:
        parts.append(f"  pressure: {a['pressure']} {a.get('pressure_unit','')}")
    return "\n".join(parts)


def _chores_block(user_text: str) -> str:
    person_match = _PERSON_RE.search(user_text)
    today = date.today()
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if person_match:
            person = person_match.group(0).capitalize()
            cur.execute(
                """
                SELECT chore_name, chore_date, done
                FROM chores
                WHERE LOWER(person) = LOWER(%s)
                  AND (chore_date = %s OR chore_date IS NULL)
                ORDER BY done, chore_name
                """,
                (person, today),
            )
            rows = cur.fetchall()
            if not rows:
                return f"Chores DB: {person} has no chores recorded for today."
            lines = [f"Chores for {person} (today, {today.isoformat()}):"]
            for r in rows:
                tick = "[done]" if r["done"] else "[open]"
                lines.append(f"  {tick} {r['chore_name']}")
            return "\n".join(lines)
        # No person specified — summarize per-person counts for today. (tier-1 test)
        cur.execute(
            """
            SELECT person, count(*) FILTER (WHERE NOT done) AS open,
                   count(*) FILTER (WHERE done) AS done_count
            FROM chores
            WHERE chore_date = %s OR chore_date IS NULL
            GROUP BY person
            ORDER BY person
            """,
            (today,),
        )
        rows = cur.fetchall()
        if not rows:
            return "Chores DB: no chores recorded for today."
        lines = [f"Chores summary today ({today.isoformat()}):"]
        for r in rows:
            lines.append(
                f"  {r['person']}: {r['open']} open, {r['done_count']} done"
            )
        return "\n".join(lines)


def _weekly_plan_block(user_text: str) -> str:
    today = date.today()
    week_end = today + timedelta(days=7)
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT wp.plan_date, wp.status, r.title, r.course
            FROM weekly_plan wp
            LEFT JOIN recipes r ON r.id = wp.recipe_id
            WHERE wp.plan_date BETWEEN %s AND %s
            ORDER BY wp.plan_date
            """,
            (today, week_end),
        )
        rows = cur.fetchall()
        if not rows:
            return (
                "Weekly plan DB: nothing scheduled in the next 7 days. "
                "There are 26 historical weekly_plan rows but none upcoming."
            )
        lines = ["Weekly plan (next 7 days):"]
        for r in rows:
            day = r["plan_date"].strftime("%a %m/%d") if r["plan_date"] else "?"
            title = r["title"] or "(no recipe linked)"
            lines.append(f"  {day}: {title} [{r['status'] or 'planned'}]")
        return "\n".join(lines)


def _recipes_block(user_text: str) -> str:
    """Light keyword search of recipe titles for context. Cheap and
    doesn't pretend to be the canonical recipe-search tool — that's a
    future endpoint that will use embeddings."""
    keywords = [
        w
        for w in re.findall(r"[A-Za-z]{4,}", user_text)
        if w.lower() not in {
            "recipe", "recipes", "cook", "make", "dinner", "lunch",
            "breakfast", "dish", "meal", "menu", "what", "have",
            "give", "tell", "show", "with", "from", "find", "want",
            "need", "today", "tonight", "tomorrow",
        }
    ][:4]
    with _conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        if keywords:
            ilike = " OR ".join(["title ILIKE %s"] * len(keywords))
            params = [f"%{k}%" for k in keywords]
            cur.execute(
                f"""
                SELECT title, course, prep_time
                FROM recipes
                WHERE {ilike}
                ORDER BY title
                LIMIT 8
                """,
                params,
            )
            hits = cur.fetchall()
            if hits:
                lines = ["Recipe DB matches (for keywords " +
                         ", ".join(repr(k) for k in keywords) + "):"]
                for r in hits:
                    extra = []
                    if r["course"]:
                        extra.append(r["course"])
                    if r["prep_time"]:
                        extra.append(f"{r['prep_time']} min")
                    suffix = f" ({', '.join(extra)})" if extra else ""
                    lines.append(f"  - {r['title']}{suffix}")
                return "\n".join(lines)
        cur.execute("SELECT count(*) FROM recipes")
        n = cur.fetchone()["count"]
        return f"Recipe DB: {n} recipes available; no specific matches for this query."
