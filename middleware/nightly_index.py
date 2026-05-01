"""Nightly memory job: reindex deep memory + generate daily digest.

Run by /etc/systemd/system/benson-nightly.timer at 04:00 local time.

Two phases:
  1. reindex_all() — pick up any new conversations, events, recipes, chores
     and re-index memory files.
  2. Generate a daily digest (chief-of-staff voice via OAuth) summarizing
     yesterday's activity, store at /opt/benson/memory/digests/<date>.md
     so Benson can grep his own past digests later.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Resolve middleware imports
sys.path.insert(0, "/opt/benson/middleware")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benson.nightly")

DIGEST_DIR = Path("/opt/benson/memory/digests")


async def _gather_yesterday() -> dict:
    """Pull yesterday's activity for the digest prompt."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from config import PG_DSN
    yesterday = date.today() - timedelta(days=1)
    out: dict = {"date": yesterday.isoformat()}
    with psycopg2.connect(**PG_DSN) as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT speaker, room, user_text, benson_response FROM conversations "
            "WHERE created_at::date = %s ORDER BY created_at",
            (yesterday,),
        )
        out["conversations"] = [dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT person, chore_name, done FROM chores WHERE chore_date = %s",
            (yesterday,),
        )
        out["chores"] = [dict(r) for r in cur.fetchall()]

        cur.execute(
            "SELECT person, title, location, starts_at FROM calendar_events "
            "WHERE starts_at::date = %s ORDER BY starts_at",
            (yesterday,),
        )
        out["events"] = [
            {**dict(r), "starts_at": r["starts_at"].isoformat() if r["starts_at"] else None}
            for r in cur.fetchall()
        ]
    return out


async def _generate_digest(activity: dict) -> str:
    from oauth_oneshot import ask
    from config import PROMPT_PATH
    base = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else ""
    prompt = (
        "Generate yesterday's digest for the Fagley House Hub. Read the "
        "raw activity below and produce a short markdown summary in your "
        "chief-of-staff voice. Include sections: '## Highlights' (4-8 "
        "bullets, what mattered), '## Patterns' (anything you noticed "
        "about routines / preferences / issues that's worth remembering "
        "long-term — these will be re-read later when answering "
        "questions), '## Open threads' (anything unresolved). Be specific "
        "and observational — name people, name events. Skip filler.\n\n"
        f"DATE: {activity['date']}\n\n"
        f"CONVERSATIONS ({len(activity['conversations'])}):\n"
        + "\n".join(
            f"  - {c['speaker']} via {c.get('room','?')}: {(c.get('user_text') or '')[:200]} → {(c.get('benson_response') or '')[:200]}"
            for c in activity["conversations"][:80]
        )
        + f"\n\nCHORES ({len(activity['chores'])}):\n"
        + "\n".join(f"  - {c['person']}: {c['chore_name']} {'✓' if c['done'] else '○'}" for c in activity["chores"])
        + f"\n\nEVENTS ({len(activity['events'])}):\n"
        + "\n".join(f"  - {e.get('person','?')}: {e['title']} @ {e.get('starts_at','?')}" for e in activity["events"])
    )
    return await ask(prompt, base, model="sonnet", timeout_s=180)


def rollover_chores() -> dict:
    """Push undone chores from prior days forward to today, and spawn
    fresh undone copies of recurring chores whose latest instance
    landed before today. Idempotent — running it multiple times in
    one day is safe (the second pass finds no stale rows + no missing
    recurring spawns).

    Two distinct moves:
      1. Stale undone chores: chore_date < today AND done=FALSE → bump
         chore_date to today. The user sees them on today's list
         instead of them silently vanishing from view.
      2. Recurring spawn: for each (person, chore_name, recurring) that
         has no incomplete instance scheduled for today or later, drop
         a new undone row dated today (or the next applicable day for
         weekdays/weekends).
    """
    import psycopg2
    from datetime import date as _d, timedelta as _td
    from config import PG_DSN

    today = _d.today()
    rolled = 0
    spawned = 0

    with psycopg2.connect(**PG_DSN) as c, c.cursor() as cur:
        cur.execute(
            "UPDATE chores SET chore_date = %s "
            "WHERE done = FALSE AND chore_date IS NOT NULL "
            "AND chore_date < %s",
            (today, today),
        )
        rolled = cur.rowcount

        cur.execute(
            "SELECT DISTINCT person, chore_name, recurring "
            "FROM chores WHERE recurring IS NOT NULL"
        )
        schedules = cur.fetchall()
        for person, chore_name, recurring in schedules:
            cur.execute(
                "SELECT 1 FROM chores "
                "WHERE person = %s AND chore_name = %s AND recurring = %s "
                "  AND chore_date >= %s LIMIT 1",
                (person, chore_name, recurring, today),
            )
            if cur.fetchone():
                continue
            d = today
            if recurring == "weekdays":
                while d.weekday() >= 5:
                    d += _td(days=1)
            elif recurring == "weekends":
                while d.weekday() < 5:
                    d += _td(days=1)
            cur.execute(
                "INSERT INTO chores (person, chore_name, chore_date, done, recurring) "
                "VALUES (%s, %s, %s, FALSE, %s)",
                (person, chore_name, d, recurring),
            )
            spawned += 1
        c.commit()

    return {"rolled_forward": rolled, "recurring_spawned": spawned, "today": today.isoformat()}


async def main():
    from memory_index import reindex_all
    logger.info("nightly: rolling chores forward")
    chore_stats = await asyncio.to_thread(rollover_chores)
    logger.info(f"nightly: chores {chore_stats}")

    logger.info("nightly: reindexing deep memory")
    counts = await asyncio.to_thread(reindex_all)
    logger.info(f"nightly: indexed {counts}")

    logger.info("nightly: generating yesterday's digest")
    activity = await _gather_yesterday()
    if not (activity["conversations"] or activity["chores"] or activity["events"]):
        logger.info("nightly: no activity yesterday, skipping digest")
        return
    digest = await _generate_digest(activity)
    if not digest:
        logger.warning("nightly: digest generation returned empty; skipping write")
        return
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIGEST_DIR / f"{activity['date']}.md"
    header = f"# Daily digest — {activity['date']}\n\n"
    out_path.write_text(header + digest.strip() + "\n")
    logger.info(f"nightly: digest written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
