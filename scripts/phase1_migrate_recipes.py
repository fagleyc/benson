"""Phase 1.2 migration: copy recipes / weekly_plan / chores from
`/opt/benson/context/prior_code/current_dashboard/recipes.db` (SQLite)
into Benson's Postgres database.

Run after `phase1_schema.sql` has been applied. Idempotent: skips rows
whose `legacy_recipe_id` already exists.

Usage:
    python3 phase1_migrate_recipes.py
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "context/prior_code/current_dashboard/recipes.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("phase1.migrate")


def _split_text_lines(text: str | None) -> list[str]:
    if not text:
        return []
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _ingredients_to_jsonb(text: str | None) -> list[dict]:
    """Each line becomes {text: <line>}; structured fields filled in by
    future Claude-vision ingestion, not here."""
    return [{"text": ln} for ln in _split_text_lines(text)]


def _steps_to_jsonb(text: str | None) -> list[str]:
    """Preserve each line as a step. Section headers ('Make the cake.')
    remain in line — they're useful anchors for the reader."""
    return _split_text_lines(text)


def _coerce_int(val) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _coerce_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pg_conn():
    return psycopg2.connect(
        dbname=os.environ.get("POSTGRES_DB", "benson"),
        user=os.environ.get("POSTGRES_USER", "benson"),
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
    )


def migrate_recipes(sqlite_conn, pg_conn) -> tuple[int, int]:
    sqlite_conn.row_factory = sqlite3.Row
    cur_src = sqlite_conn.cursor()
    cur_dst = pg_conn.cursor()
    cur_src.execute("SELECT * FROM recipes")
    inserted = 0
    skipped = 0
    for row in cur_src:
        legacy_id = row["recipe_id"]
        cur_dst.execute(
            "SELECT 1 FROM recipes WHERE legacy_recipe_id = %s", (legacy_id,)
        )
        if cur_dst.fetchone():
            skipped += 1
            continue
        cur_dst.execute(
            """
            INSERT INTO recipes
                (title, source, source_url, ingredients, steps, tags,
                 image_url, household_rating, user_rating, user_comments,
                 dish_type, course, prep_time, parse_status,
                 legacy_recipe_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["title"],
                "migrated",
                row["source_link"],
                Json(_ingredients_to_jsonb(row["ingredients"])),
                Json(_steps_to_jsonb(row["instructions"])),
                Json([]),
                row["image_url"],
                _coerce_int(row["rating"]),
                _coerce_float(row["user_rating"]),
                row["user_comments"] or None,
                row["dish_type"] or None,
                row["course"] or None,
                _coerce_int(row["prep_time"]),
                row["parse_status"] or "approved",
                legacy_id,
            ),
        )
        inserted += 1
    pg_conn.commit()
    return inserted, skipped


def migrate_weekly_plan(sqlite_conn, pg_conn) -> tuple[int, int]:
    sqlite_conn.row_factory = sqlite3.Row
    cur_src = sqlite_conn.cursor()
    cur_dst = pg_conn.cursor()
    cur_src.execute("SELECT * FROM weekly_plan")
    inserted = 0
    skipped = 0
    for row in cur_src:
        # Map legacy recipe_id -> new recipes.id via legacy_recipe_id
        legacy_recipe_id = row["recipe_id"]
        new_recipe_id = None
        if legacy_recipe_id is not None:
            cur_dst.execute(
                "SELECT id FROM recipes WHERE legacy_recipe_id = %s",
                (legacy_recipe_id,),
            )
            r = cur_dst.fetchone()
            if r:
                new_recipe_id = r[0]
        cur_dst.execute(
            """
            INSERT INTO weekly_plan (plan_date, recipe_id, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (plan_date) DO NOTHING
            """,
            (row["plan_date"], new_recipe_id, row["status"]),
        )
        if cur_dst.rowcount == 1:
            inserted += 1
        else:
            skipped += 1
    pg_conn.commit()
    return inserted, skipped


def migrate_chores(sqlite_conn, pg_conn) -> tuple[int, int]:
    sqlite_conn.row_factory = sqlite3.Row
    cur_src = sqlite_conn.cursor()
    cur_dst = pg_conn.cursor()
    cur_src.execute("SELECT * FROM chores")
    inserted = 0
    skipped = 0
    cur_dst.execute("SELECT count(*) FROM chores")
    if cur_dst.fetchone()[0] > 0:
        logger.info("chores already populated; skipping")
        return 0, sum(1 for _ in cur_src)
    for row in cur_src:
        cur_dst.execute(
            """
            INSERT INTO chores (person, chore_date, chore_name, done)
            VALUES (%s, %s, %s, %s)
            """,
            (row["person"], row["chore_date"], row["chore_name"], bool(row["done"])),
        )
        inserted += 1
    pg_conn.commit()
    return inserted, skipped


def main() -> int:
    if not SQLITE_PATH.exists():
        logger.error(f"SQLite source not found: {SQLITE_PATH}")
        return 1

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    pg_conn = _pg_conn()
    try:
        ins_r, skp_r = migrate_recipes(sqlite_conn, pg_conn)
        logger.info(f"recipes: {ins_r} inserted, {skp_r} already present")
        ins_w, skp_w = migrate_weekly_plan(sqlite_conn, pg_conn)
        logger.info(f"weekly_plan: {ins_w} inserted, {skp_w} skipped")
        ins_c, skp_c = migrate_chores(sqlite_conn, pg_conn)
        logger.info(f"chores: {ins_c} inserted, {skp_c} skipped")
    finally:
        sqlite_conn.close()
        pg_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
