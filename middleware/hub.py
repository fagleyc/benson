"""House Hub — HTML frontend for the Fagley household.

Mounted on the same FastAPI app as the API. Serves Jinja-rendered
Bootswatch-Darkly pages plus an HTMX-driven floating "Talk to Benson"
chat widget. The widget posts to /hub/chat which forwards to the
existing /conversation pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import psycopg2
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from psycopg2.extras import RealDictCursor

from config import PG_DSN
from db_tools import gather_context
from memory import MemoryStore

logger = logging.getLogger("benson.hub")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()

memory = MemoryStore()


def _ctx(active: str, **extra) -> dict:
    return {
        "active": active,
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **extra,
    }


def _query(sql: str, params: tuple = ()) -> list[dict]:
    with psycopg2.connect(**PG_DSN) as conn, conn.cursor(
        cursor_factory=RealDictCursor
    ) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def _query_one(sql: str, params: tuple = ()) -> dict | None:
    with psycopg2.connect(**PG_DSN) as conn, conn.cursor(
        cursor_factory=RealDictCursor
    ) as cur:
        cur.execute(sql, params)
        r = cur.fetchone()
        return dict(r) if r else None


# ─── Home ────────────────────────────────────────────────────────────────
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    from datetime import date
    today = date.today()
    meal = await asyncio.to_thread(
        _query_one,
        """
        SELECT wp.plan_date, wp.status, r.id AS recipe_id, r.title,
               r.course, r.image_url, r.source_url
        FROM weekly_plan wp
        LEFT JOIN recipes r ON r.id = wp.recipe_id
        WHERE wp.plan_date = %s
        """,
        (today,),
    )
    chores = await asyncio.to_thread(
        _query,
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
    return templates.TemplateResponse(request, "home.html", _ctx("home", today={"meal": meal, "chores": chores}),
    )


# ─── Recipes ─────────────────────────────────────────────────────────────
@router.get("/recipes", response_class=HTMLResponse)
async def recipes_page(
    request: Request,
    q: str | None = Query(None),
    course: str | None = Query(None),
):
    # Pull everything once; filtering + sorting happen client-side over
    # the whole 76-row dataset. Q/course server params are kept for
    # back-compat with existing /recipes?course=X bookmarks but the new
    # UI prefers in-page controls.
    where = []
    params: list = []
    if q:
        where.append("title ILIKE %s")
        params.append(f"%{q}%")
    if course:
        where.append("course ILIKE %s")
        params.append(course)
    sql = (
        "SELECT id, title, course, dish_type, prep_time, image_url, "
        "user_rating, tags, last_made, source_url, notes FROM recipes"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY title LIMIT 5000"
    recipes = await asyncio.to_thread(_query, sql, tuple(params))
    courses = await asyncio.to_thread(
        _query,
        "SELECT DISTINCT course FROM recipes WHERE course IS NOT NULL AND course != '' ORDER BY course",
    )
    return templates.TemplateResponse(request, "recipes.html", _ctx("recipes",
            recipes=recipes,
            courses=[c["course"] for c in courses],
            q=q,
            course=course,
        ),
    )


@router.get("/recipes/{recipe_id}", response_class=HTMLResponse)
async def recipe_detail_page(request: Request, recipe_id: int):
    recipe = await asyncio.to_thread(
        _query_one, "SELECT * FROM recipes WHERE id = %s", (recipe_id,)
    )
    if not recipe:
        raise HTTPException(404, "recipe not found")
    return templates.TemplateResponse(request, "recipe_detail.html", _ctx("recipes", recipe=recipe)
    )


@router.get("/recipes/{recipe_id}/cook", response_class=HTMLResponse)
async def recipe_cook_page(request: Request, recipe_id: int):
    recipe = await asyncio.to_thread(
        _query_one, "SELECT * FROM recipes WHERE id = %s", (recipe_id,)
    )
    if not recipe:
        raise HTTPException(404, "recipe not found")
    return templates.TemplateResponse(
        request, "recipe_cook.html", {"recipe": recipe}
    )


# ─── Cook-mode ingredient → step map (Haiku-resolved, cached per recipe) ─
# The /cook page used to highlight ingredients with browser-side substring
# matching, which broke on references like "the dough" / "the sauce" /
# "remaining mixture". This endpoint asks Haiku once (with the full
# ingredient list + all steps in context) to produce a step-index →
# ingredient-indices map, persists it on the recipes row, and returns
# the cached value on every subsequent call.
@router.get("/api/recipes/{recipe_id}/cook_map")
async def recipe_cook_map(recipe_id: int):
    import json as _json
    from fastapi.responses import JSONResponse

    # Defensive idempotent migration — keeps this feature working even
    # if middleware/sql/recipe_step_ingredient_map.sql hasn't been applied
    # by hand yet. ADD COLUMN IF NOT EXISTS is a no-op once the column is
    # present, so this is cheap to run on every call.
    def _ensure_column_sync() -> None:
        try:
            with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE recipes "
                    "ADD COLUMN IF NOT EXISTS step_ingredient_map JSONB DEFAULT NULL"
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"cook_map: ensure column failed: {e}")

    await asyncio.to_thread(_ensure_column_sync)

    recipe = await asyncio.to_thread(
        _query_one,
        "SELECT id, ingredients, steps, step_ingredient_map "
        "FROM recipes WHERE id = %s",
        (recipe_id,),
    )
    if not recipe:
        return JSONResponse({"error": "not found"}, status_code=404)

    # Cache hit — psycopg2 already decodes JSONB to a Python object, but
    # tolerate the str case in case some row was inserted as TEXT.
    cached = recipe.get("step_ingredient_map")
    if cached is not None:
        if isinstance(cached, str):
            try:
                cached = _json.loads(cached)
            except Exception:
                cached = {}
        return JSONResponse(cached)

    # Build human-readable ingredient + step lists from the JSONB columns.
    raw_ings = recipe.get("ingredients") or []
    if isinstance(raw_ings, str):
        try:
            raw_ings = _json.loads(raw_ings)
        except Exception:
            raw_ings = []
    ingredients: list[str] = []
    for ing in raw_ings:
        if isinstance(ing, dict):
            ingredients.append(
                str(ing.get("text") or ing.get("name") or "").strip()
            )
        else:
            ingredients.append(str(ing or "").strip())
    ingredients = [i for i in ingredients if i]

    raw_steps = recipe.get("steps") or []
    if isinstance(raw_steps, str):
        # Could be JSON or newline-separated. Try JSON first.
        try:
            parsed = _json.loads(raw_steps)
            raw_steps = parsed if isinstance(parsed, list) else [
                s for s in raw_steps.splitlines() if s.strip()
            ]
        except Exception:
            raw_steps = [s for s in raw_steps.splitlines() if s.strip()]
    steps: list[str] = [str(s or "").strip() for s in raw_steps if str(s or "").strip()]

    def _persist_sync(payload: dict) -> None:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE recipes SET step_ingredient_map = %s::jsonb WHERE id = %s",
                (_json.dumps(payload), recipe_id),
            )
            conn.commit()

    if not ingredients or not steps:
        await asyncio.to_thread(_persist_sync, {})
        return JSONResponse({})

    system_prompt = (
        "You are a cooking assistant. Given a recipe's ingredient list "
        "and step-by-step directions, return a JSON object mapping each "
        "step index (0-based integer key as a string) to an array of "
        "ingredient indices (0-based integers) that are actively used in "
        "that step. Resolve references like 'the dough', 'the sauce', "
        "'remaining mixture' back to the original ingredients. Return "
        "ONLY the JSON object, no explanation, no markdown."
    )
    user_prompt = (
        "Ingredients (0-based index):\n"
        + "\n".join(f"{i}: {ing}" for i, ing in enumerate(ingredients))
        + "\n\nSteps (0-based index):\n"
        + "\n".join(f"{i}: {step}" for i, step in enumerate(steps))
    )

    parsed_map: dict = {}
    try:
        from oauth_oneshot import ask as oauth_ask
        text = await oauth_ask(
            user_prompt, system_prompt, model="haiku", timeout_s=60
        )
        if text:
            # Strip optional markdown fence — same trick recipes.py uses.
            t = text.strip()
            if t.startswith("```"):
                t = t.split("\n", 1)[1] if "\n" in t else t
                if t.endswith("```"):
                    t = t.rsplit("```", 1)[0]
                t = t.strip()
            try:
                parsed = _json.loads(t)
                if isinstance(parsed, dict):
                    parsed_map = parsed
            except Exception as e:
                logger.warning(
                    f"cook_map: haiku response wasn't parseable JSON: "
                    f"{e}; first 200 chars: {text[:200]!r}"
                )
    except Exception as e:
        logger.warning(f"cook_map: haiku call failed: {type(e).__name__}: {e}")

    await asyncio.to_thread(_persist_sync, parsed_map)
    return JSONResponse(parsed_map)


# ─── Weekly menu ─────────────────────────────────────────────────────────
@router.get("/weekly", response_class=HTMLResponse)
async def weekly_page(request: Request):
    from datetime import date, timedelta
    today = date.today()
    days = [today + timedelta(days=i) for i in range(7)]
    existing = await asyncio.to_thread(
        _query,
        """
        SELECT wp.plan_date, wp.status, wp.recipe_id,
               r.title, r.course, r.image_url
        FROM weekly_plan wp
        LEFT JOIN recipes r ON r.id = wp.recipe_id
        WHERE wp.plan_date BETWEEN %s AND %s
        ORDER BY wp.plan_date
        """,
        (today, today + timedelta(days=6)),
    )
    by_date = {row["plan_date"]: row for row in existing}
    plan = []
    for d in days:
        row = by_date.get(d)
        plan.append({
            "plan_date": d,
            "recipe_id": (row or {}).get("recipe_id"),
            "title": (row or {}).get("title"),
            "course": (row or {}).get("course"),
            "status": (row or {}).get("status") or "leftover",
        })
    recipes = await asyncio.to_thread(
        _query,
        "SELECT id, title, course FROM recipes ORDER BY title",
    )
    return templates.TemplateResponse(
        request, "weekly.html", _ctx("weekly", plan=plan, recipes=recipes)
    )


# ─── Chores ──────────────────────────────────────────────────────────────
@router.get("/chores", response_class=HTMLResponse)
async def chores_page(request: Request):
    """Person × day grid for the next 7 days."""
    from datetime import date, timedelta
    today = date.today()
    days = [today + timedelta(days=i) for i in range(7)]
    all_persons = ["Casey", "Lindsey", "Cole", "Zander", "Household"]
    person_filter = request.query_params.get("person")
    if person_filter:
        match = next(
            (p for p in all_persons if p.lower() == person_filter.lower()), None
        )
        persons = [match] if match else all_persons
    else:
        persons = all_persons
    chores = await asyncio.to_thread(
        _query,
        """
        SELECT id, person, chore_date, chore_name, done, recurring,
               dollars, points
        FROM chores
        WHERE chore_date BETWEEN %s AND %s OR chore_date IS NULL
        ORDER BY person, chore_date NULLS LAST, chore_name
        """,
        (today, today + timedelta(days=6)),
    )
    # Group: chores_map[person][date_str] = list[chore]
    chores_map = {p: {d.isoformat(): [] for d in days} for p in persons}
    for c in chores:
        p = c["person"] if c["person"] in persons else "Household"
        if p not in chores_map:
            # Filtering to a single person — drop rows for other people.
            continue
        d = c["chore_date"].isoformat() if c["chore_date"] else None
        if d and d in chores_map[p]:
            chores_map[p][d].append(c)
        elif not d:
            # Undated chores → put in today's column for the person
            chores_map[p][today.isoformat()].append(c)
    return templates.TemplateResponse(
        request, "chores.html",
        _ctx("chores", days=days, persons=persons, chores_map=chores_map,
             person_filter=person_filter),
    )


# ─── Memory ──────────────────────────────────────────────────────────────
@router.get("/memory", response_class=HTMLResponse)
async def memory_page(request: Request):
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT id, content, source, speaker, room, importance, created_at
        FROM memories
        ORDER BY created_at DESC
        LIMIT 100
        """,
    )
    return templates.TemplateResponse(request, "memory.html", _ctx("memory", memories=rows)
    )


# ─── Status ──────────────────────────────────────────────────────────────
@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request):
    import os
    rec = await asyncio.to_thread(_query_one, "SELECT count(*) FROM recipes")
    op = await asyncio.to_thread(
        _query_one, "SELECT count(*) FROM chores WHERE NOT done"
    )
    mem = await asyncio.to_thread(_query_one, "SELECT count(*) FROM memories")
    status = {
        "db": {
            "recipes": rec["count"],
            "open_chores": op["count"],
            "memories": mem["count"],
        },
        "secrets": {
            "oauth_credentials": (Path.home() / ".claude" / ".credentials.json").exists(),
            "ha_token": bool(os.environ.get("HA_LONG_LIVED_TOKEN")),
            "bond_token": bool(os.environ.get("BOND_BRIDGE_TOKEN")),
            "telegram_token": bool(os.environ.get("TELEGRAM_BOT_TOKEN")),
            "instacart": bool(os.environ.get("INSTACART_API_KEY")),
        },
    }
    return templates.TemplateResponse(request, "status.html", _ctx("status", status=status)
    )


# ─── Chat (full page) ────────────────────────────────────────────────────
@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", _ctx("chat"))


# ─── Signal pairing/admin page ───────────────────────────────────────────
@router.get("/admin/signal", response_class=HTMLResponse)
async def signal_admin(request: Request):
    return templates.TemplateResponse(request, "signal_admin.html", _ctx("advanced"))


# ─── Google (Calendar + Gmail) admin page ────────────────────────────────
@router.get("/admin/google", response_class=HTMLResponse)
async def google_admin(request: Request):
    return templates.TemplateResponse(request, "google_admin.html", _ctx("advanced"))


# ─── Cameras (iPad-as-eyeball) admin page ────────────────────────────────
@router.get("/admin/cameras", response_class=HTMLResponse)
async def cameras_admin(request: Request):
    return templates.TemplateResponse(request, "camera_admin.html", _ctx("advanced"))


# ─── Benson observability dashboard ──────────────────────────────────────
@router.get("/admin/benson", response_class=HTMLResponse)
async def benson_admin(request: Request):
    ctx = await asyncio.to_thread(_gather_benson_admin_sync)
    return templates.TemplateResponse(
        request, "benson_admin.html", _ctx("advanced", **ctx)
    )


def _gather_schedules_sync() -> list[dict]:
    """List all benson-*.timer units with their schedule + last/next
    fire + the service they trigger + that service's last result."""
    import subprocess as _sp

    def _show(unit: str, props: list[str]) -> dict:
        # `-p=Foo` (with equals) is silently ignored by some systemctl
        # versions on Linux 6+ — produces empty stdout. `--property=Foo`
        # works reliably. Found this on the DGX 2026-05-08.
        try:
            r = _sp.run(
                ["systemctl", "show", unit, "--no-pager",
                 *(f"--property={p}" for p in props)],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return {}
        result: dict = {}
        for line in r.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                result[k] = v
        return result

    # Find all benson timer units.
    try:
        r = _sp.run(
            ["systemctl", "list-unit-files", "benson-*.timer",
             "--no-legend", "--no-pager"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []
    timer_units = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].endswith(".timer"):
            timer_units.append(parts[0])

    out: list[dict] = []
    for tu in sorted(timer_units):
        timer = _show(tu, [
            "Unit", "Description", "LastTriggerUSec",
            "NextElapseUSecRealtime", "ActiveState",
        ])
        svc_name = timer.get("Unit") or tu.replace(".timer", ".service")
        svc = _show(svc_name, [
            "Description", "Result", "ActiveState",
            "ExecMainStartTimestamp", "ExecMainExitTimestamp",
        ])
        # "Now" fire info: NextElapseUSecRealtime can be blank when the
        # timer is one-shot or just fired; fall back to "—".
        next_fire = (timer.get("NextElapseUSecRealtime") or "").strip()
        last_fire = (timer.get("LastTriggerUSec") or "").strip()
        # Friendly "what does it do" — prefer the service description
        # (more specific) but fall back to the timer's.
        what = (svc.get("Description") or timer.get("Description") or "").strip()
        out.append({
            "name": tu.replace(".timer", "").replace("benson-", ""),
            "unit_timer": tu,
            "unit_service": svc_name,
            "description": what,
            "last_fire": last_fire if last_fire and last_fire != "n/a" else "—",
            "next_fire": next_fire or "—",
            "last_result": svc.get("Result") or "—",
            "active": svc.get("ActiveState") or "inactive",
        })
    return out


def _gather_benson_admin_sync() -> dict:
    """Collect everything the /admin/benson page renders. Sync calls only —
    wrapped in to_thread by the route handler."""
    import json as _json
    import re as _re
    import subprocess as _sp
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    out: dict = {}

    # ─── Health checks (✓/✗ list at the top) ──────────────────────────
    health: list[dict] = []

    def _check(name: str, ok: bool, detail: str = "") -> None:
        health.append({"name": name, "ok": bool(ok), "detail": detail})

    # Service active
    try:
        r = _sp.run(["systemctl", "is-active", "benson.service"],
                    capture_output=True, text=True, timeout=5)
        _check("service active", r.stdout.strip() == "active", r.stdout.strip())
    except Exception as e:
        _check("service active", False, f"{type(e).__name__}: {e}")

    # OAuth credentials
    creds = Path.home() / ".claude" / ".credentials.json"
    creds_ok = creds.exists()
    expires_str = ""
    if creds_ok:
        try:
            data = _json.loads(creds.read_text())
            exp_ms = data.get("claudeAiOauth", {}).get("expiresAt", 0)
            if exp_ms:
                exp_dt = _dt.fromtimestamp(exp_ms / 1000, _tz.utc)
                still_good = exp_dt > _dt.now(_tz.utc)
                creds_ok = still_good
                expires_str = (
                    f"expires {exp_dt.isoformat(timespec='minutes')} "
                    f"({'valid' if still_good else 'EXPIRED'})"
                )
        except Exception as e:
            creds_ok = False
            expires_str = f"parse error: {e}"
    _check("oauth credentials valid", creds_ok, expires_str or str(creds))

    # ANTHROPIC_API_KEY should NOT be set (we migrated to OAuth)
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    _check(
        "no ANTHROPIC_API_KEY in env",
        not api_key_set,
        "key present — every SDK call may bill API!" if api_key_set else "good (OAuth only)",
    )

    # Indexer freshness — most recent row in memory_index
    try:
        latest = _query_one(
            "SELECT MAX(created_at) AS m FROM memory_index"
        )
        last_idx = (latest or {}).get("m")
        fresh = bool(last_idx) and (
            _dt.now(_tz.utc) - last_idx
        ) < _td(hours=36)
        _check(
            "ltm indexer ran in last 36h",
            fresh,
            f"last row: {last_idx.isoformat(timespec='minutes') if last_idx else 'never'}",
        )
        out["last_indexer_row"] = last_idx.isoformat() if last_idx else None
    except Exception as e:
        _check("ltm indexer ran in last 36h", False, str(e))

    # STM auto-curation activity in last 24h
    try:
        stm_root = Path("/opt/benson/memory/short_term")
        cutoff = _dt.now().timestamp() - 24 * 3600
        active = []
        for fp in list(stm_root.rglob("*.md")):
            if fp.stat().st_mtime > cutoff and fp.stat().st_size > 0:
                active.append(fp.name)
        _check(
            "stm written to in last 24h",
            len(active) > 0,
            f"{len(active)} file(s): {', '.join(active[:5])}" if active else "no entries — Benson hasn't written autonomously yet",
        )
    except Exception as e:
        _check("stm written to in last 24h", False, str(e))

    out["health"] = health

    # ─── Activity (recent entries) ──────────────────────────────────────
    # Recent STM entries — parse last few timestamped lines from today's
    # journal + each topic file.
    stm_entries: list[dict] = []
    try:
        from short_term import STM_ROOT, _today_file, TOPICS_DIR, INBOX_FILE
        line_re = _re.compile(r"^- \[(\d{2}:\d{2})\] (.+)$")
        for src_path in [_today_file(), INBOX_FILE] + (
            sorted(TOPICS_DIR.glob("*.md")) if TOPICS_DIR.exists() else []
        ):
            if not src_path.exists() or src_path.stat().st_size == 0:
                continue
            # Get the last 3 timestamped entries from this file.
            lines = src_path.read_text(errors="replace").splitlines()
            entries = []
            for ln in lines:
                m = line_re.match(ln.strip())
                if m:
                    entries.append({"time": m.group(1), "content": m.group(2)})
            for e in entries[-3:]:
                stm_entries.append({
                    "topic": src_path.stem,
                    "time": e["time"],
                    "content": e["content"][:200],
                    "modified": _dt.fromtimestamp(src_path.stat().st_mtime).isoformat(timespec="minutes"),
                })
        # Sort newest first by modified time + entry time
        stm_entries.sort(key=lambda x: (x["modified"], x["time"]), reverse=True)
    except Exception as e:
        logger.warning(f"stm_entries gather failed: {e}")
    out["stm_entries"] = stm_entries[:10]

    # Recent proposals (open + last few merged)
    proposals_open: list[dict] = []
    try:
        from self_modify import list_open_proposals
        proposals_open = list_open_proposals()
    except Exception as e:
        logger.warning(f"list_open_proposals failed: {e}")
    out["proposals_open"] = proposals_open

    # Recently merged proposals — git log on main for [proposal-meta] tags
    try:
        log_out = _sp.check_output(
            ["git", "-C", "/opt/benson", "log", "--all", "-5",
             "--grep=[proposal-meta]", "--format=%h|%ci|%s"],
            stderr=_sp.STDOUT, timeout=10,
        ).decode("utf-8", errors="replace")
        merged = []
        for line in log_out.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                merged.append({"sha": parts[0], "when": parts[1], "subject": parts[2]})
        out["proposals_merged"] = merged
    except Exception:
        out["proposals_merged"] = []

    # Recent Tier 1 autonomous changes
    try:
        from self_modify import autofix_list, autofix_remote_commit_url
        autofixes = autofix_list(limit=20)
        for af in autofixes:
            af["commit_url"] = autofix_remote_commit_url(af["commit_sha"])
        out["autofixes"] = autofixes
    except Exception as e:
        logger.warning(f"autofix_list failed: {e}")
        out["autofixes"] = []

    # Recent tool failures (per-speaker, last 10 min)
    try:
        from oauth_agent import recent_failures_snapshot
        out["recent_failures"] = recent_failures_snapshot()
    except Exception as e:
        out["recent_failures"] = {"_error": str(e)}

    # Tier mix from last 24h of conversations
    try:
        rows = _query(
            "SELECT tier, COUNT(*) AS n FROM conversations "
            "WHERE created_at > NOW() - INTERVAL '24 hours' "
            "GROUP BY tier ORDER BY n DESC"
        )
        out["tier_mix"] = [{"tier": r["tier"] or "(none)", "n": r["n"]} for r in rows]
    except Exception as e:
        logger.warning(f"tier_mix failed: {e}")
        out["tier_mix"] = []

    # Recent conversation log (last 40 turns) — what the household
    # actually said to Benson and what he replied. Read-only audit
    # trail; same data Benson sees when calling read_my_conversations.
    try:
        rows = _query(
            "SELECT id, speaker, room, user_text, benson_response, tier, "
            "created_at FROM conversations ORDER BY id DESC LIMIT 40"
        )
        out["conversations"] = [
            {
                "id": r["id"],
                "speaker": r["speaker"] or "(unknown)",
                "room": r["room"] or "",
                "user_text": (r["user_text"] or "")[:600],
                "benson_response": (r["benson_response"] or "")[:1200],
                "tier": r["tier"] or "",
                "created_at": r["created_at"].strftime("%Y-%m-%d %H:%M") if r["created_at"] else "",
            }
            for r in rows
        ]
    except Exception as e:
        logger.warning(f"recent conversations failed: {e}")
        out["conversations"] = []

    # ─── Scheduled jobs ──────────────────────────────────────────────────
    # All benson-*.timer units + the service each triggers, with last
    # fire / next fire / outcome. Casey 2026-05-08: 'add a summary of
    # scheduled things so I can see them and manage them.'
    try:
        out["schedules"] = _gather_schedules_sync()
    except Exception as e:
        logger.warning(f"schedule collect failed: {e}")
        out["schedules"] = []

    # ─── Memory structure ────────────────────────────────────────────────
    # STM file inventory
    try:
        from short_term import stm_list
        out["stm_files"] = stm_list().get("files", [])
    except Exception as e:
        out["stm_files"] = []
        logger.warning(f"stm_list failed: {e}")

    # LTM by source_type
    try:
        rows = _query(
            "SELECT source_type, COUNT(*) AS n, MAX(created_at) AS most_recent "
            "FROM memory_index GROUP BY source_type ORDER BY n DESC"
        )
        out["ltm_by_source"] = [
            {
                "source_type": r["source_type"],
                "n": r["n"],
                "most_recent": r["most_recent"].isoformat(timespec="minutes") if r["most_recent"] else "",
            }
            for r in rows
        ]
        total = sum(r["n"] for r in rows)
        out["ltm_total"] = total
    except Exception as e:
        out["ltm_by_source"] = []
        out["ltm_total"] = 0
        logger.warning(f"ltm_by_source failed: {e}")

    return out


# ─── Chore template editor ──────────────────────────────────────────────
@router.get("/admin/chore-templates", response_class=HTMLResponse)
async def chore_templates_admin(request: Request):
    rows = await asyncio.to_thread(
        _query,
        "SELECT id, person, chore_name, default_dollars, default_points, "
        "use_count, archived_at FROM chore_templates "
        "ORDER BY person, use_count DESC, chore_name",
    )
    return templates.TemplateResponse(
        request, "chore_templates_admin.html",
        _ctx("advanced", templates_rows=rows),
    )


# ─── Self-modification proposals ─────────────────────────────────────────
@router.get("/admin/proposals", response_class=HTMLResponse)
async def proposals_admin(request: Request):
    from self_modify import list_open_proposals
    proposals = list_open_proposals()
    return templates.TemplateResponse(
        request,
        "proposals.html",
        _ctx("advanced", proposals=proposals),
    )


@router.get("/admin/proposals/{branch:path}/diff", response_class=HTMLResponse)
async def proposal_diff_view(request: Request, branch: str):
    from self_modify import proposal_diff
    if not branch.startswith("proposal/"):
        raise HTTPException(status_code=400, detail="not a proposal branch")
    try:
        diff = proposal_diff(branch)
    except Exception as e:
        diff = f"(failed to load diff: {e})"
    return HTMLResponse(
        f"<pre style='font-family:ui-monospace,monospace;font-size:12px;"
        f"white-space:pre-wrap;line-height:1.4'>{_html_escape(diff)}</pre>"
    )


@router.post("/admin/proposals/{branch:path}/merge")
async def proposal_merge(branch: str):
    from self_modify import apply_proposal
    if not branch.startswith("proposal/"):
        raise HTTPException(status_code=400, detail="not a proposal branch")
    result = await asyncio.to_thread(apply_proposal, branch)
    return result


@router.post("/admin/proposals/{branch:path}/reject")
async def proposal_reject(branch: str):
    from self_modify import reject_proposal
    if not branch.startswith("proposal/"):
        raise HTTPException(status_code=400, detail="not a proposal branch")
    return await asyncio.to_thread(reject_proposal, branch)


# ─── Tier 1 autonomous changes ───────────────────────────────────────────
@router.post("/admin/benson/autofix/{audit_id:int}/revert")
async def autofix_revert_endpoint(audit_id: int):
    from fastapi.responses import JSONResponse
    from self_modify import autofix_revert
    result = await asyncio.to_thread(autofix_revert, audit_id)
    status = result.pop("status", 200 if result.get("ok") else 400)
    return JSONResponse(result, status_code=status)


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


# ─── Trust the Caddy root CA on iPads ────────────────────────────────────
@router.get("/admin/install-cert", response_class=HTMLResponse)
async def install_cert_page(request: Request):
    return templates.TemplateResponse(request, "install_cert.html", _ctx("advanced"))


@router.get("/cert/benson-caddy-root.crt")
async def serve_cert():
    from fastapi.responses import FileResponse
    return FileResponse(
        "/opt/benson/middleware/static/benson-caddy-root.crt",
        media_type="application/x-x509-ca-cert",
        filename="benson-caddy-root.crt",
    )


# ─── Music (Apple Music via Music Assistant) ────────────────────────────
@router.get("/music", response_class=HTMLResponse)
async def music_page(request: Request):
    return templates.TemplateResponse(request, "music.html", _ctx("music"))


# ─── Advanced settings ───────────────────────────────────────────────────
@router.get("/advanced", response_class=HTMLResponse)
async def advanced_page(request: Request):
    return templates.TemplateResponse(request, "advanced.html", _ctx("advanced"))


@router.get("/advanced/voice", response_class=HTMLResponse)
async def advanced_voice_page(request: Request):
    from voice_config import ENGINES, list_voices_for_engine, load
    cfg = load()
    engines = []
    for eid, info in ENGINES.items():
        engines.append({
            "id": eid,
            "label": info["label"],
            "voices": list_voices_for_engine(eid),
        })
    return templates.TemplateResponse(
        request,
        "advanced_voice.html",
        _ctx("advanced", cfg=cfg, engines=engines),
    )


@router.post("/advanced/voice/save", response_class=HTMLResponse)
async def advanced_voice_save(
    request: Request,
    engine: str = Form(...),
    voice: str = Form(...),
    speed: float = Form(1.0),
):
    from voice_config import save
    cfg = save({"engine": engine, "voice": voice, "speed": float(speed)})
    return templates.TemplateResponse(
        request,
        "_advanced_voice_saved.html",
        {"cfg": cfg},
    )


@router.post("/advanced/voice/test", response_class=HTMLResponse)
async def advanced_voice_test(
    request: Request,
    engine: str = Form(...),
    voice: str = Form(...),
    speed: float = Form(1.0),
    zone: str = Form("media_player.kitchen"),
    text: str = Form("Hello Casey. This is the new Benson voice. How does it sound?"),
):
    from voice_config import save
    # Persist the choice (so the test reflects what gets used in production)
    save({"engine": engine, "voice": voice, "speed": float(speed)})
    from kokoro_tts import speak_on_zone
    result = await speak_on_zone(zone, text)
    return templates.TemplateResponse(
        request,
        "_advanced_voice_tested.html",
        {"result": result, "zone": zone, "voice": voice, "engine": engine},
    )


# ─── HTMX partials ───────────────────────────────────────────────────────
@router.get("/hub/partials/recent-conversations", response_class=HTMLResponse)
async def partial_recent_conv(request: Request):
    rows = await asyncio.to_thread(
        _query,
        """
        SELECT speaker, user_text, benson_response, tier, created_at
        FROM conversations
        ORDER BY created_at DESC
        LIMIT 5
        """,
    )
    return templates.TemplateResponse(request, "_recent_conv.html", {"conversations": rows})


_HUB_ATTACHMENT_DIR = Path("/tmp/benson-attachments")
_HUB_ATTACHMENT_MAX_BYTES = 20 * 1024 * 1024  # 20MB

# Lazy import — UploadFile sits in fastapi but adding it to the top-level
# import keeps the existing imports tidy even when nobody touches it.
from fastapi import File, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402


@router.get("/hub/attachment")
async def hub_attachment(p: str):
    """Serve a hub-uploaded attachment for inline preview in the chat
    log. Path-locked to /tmp/benson-attachments/* — refuses anything
    else so this can't be used as an arbitrary file reader."""
    target = Path(p).resolve()
    base = _HUB_ATTACHMENT_DIR.resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=403, detail="not under attachment dir")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target))


@router.post("/hub/chat", response_class=HTMLResponse)
async def hub_chat(
    request: Request,
    text: str = Form(""),
    speaker: str = Form("Casey"),
    room: str = Form("hub"),
    voice_input: str = Form(""),  # "1" if mic was used
    image: UploadFile | None = File(None),
):
    """Forward to /conversation pipeline; return HTML chat-turn fragment.

    Optionally accepts an image upload (file picker or clipboard paste).
    The image is saved to /tmp/benson-attachments/<uuid>.<ext> and the
    user's text is annotated with `[attachment: <mime> at <path>]` —
    the format the system prompt already documents under 'Attachments
    arrive in messages as bracketed annotations'.
    """
    from main import handle_conversation

    user_text = (text or "").strip()
    saved_path: str | None = None
    saved_mime: str | None = None

    if image is not None and (image.filename or image.size):
        # Validate type — only image/* allowed from the hub chat. Signal
        # already has its own attachment path; this widget is for paste/upload.
        mime = (image.content_type or "").lower()
        if not mime.startswith("image/"):
            return HTMLResponse(
                f'<div class="text-danger small p-2">Unsupported file type '
                f'(got {mime or "unknown"}). Images only on the hub chat.</div>',
                status_code=400,
            )
        body = await image.read()
        if len(body) > _HUB_ATTACHMENT_MAX_BYTES:
            return HTMLResponse(
                f'<div class="text-danger small p-2">File too large '
                f'({len(body)} bytes; cap is {_HUB_ATTACHMENT_MAX_BYTES}).</div>',
                status_code=400,
            )
        ext = (mime.split("/", 1)[1] if "/" in mime else "bin").split("+", 1)[0]
        if ext == "jpeg":
            ext = "jpg"
        _HUB_ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
        import uuid as _uuid
        fname = f"hub-{_uuid.uuid4().hex[:12]}.{ext}"
        saved_path = str(_HUB_ATTACHMENT_DIR / fname)
        with open(saved_path, "wb") as f:
            f.write(body)
        saved_mime = mime

        # Annotate so the agent's existing attachment-handling prompt
        # ('analyze_image with a precise query, then chain into…') fires.
        # If the user didn't supply text, give the agent a sensible default.
        annotation = f"[attachment: {saved_mime} at {saved_path}]"
        if user_text:
            user_text = f"{user_text}\n\n{annotation}"
        else:
            user_text = (
                "Look at this image and tell me what it is and what I most "
                "likely want done with it.\n\n" + annotation
            )

    if not user_text:
        return HTMLResponse(
            '<div class="text-danger small p-2">Need either text or an image.</div>',
            status_code=400,
        )

    payload = {
        "text": user_text,
        "speaker": speaker,
        "room": room,
        "voice_input": voice_input == "1",
    }

    class _FakeReq:
        async def json(self_inner):
            return payload

    result = await handle_conversation(_FakeReq())  # type: ignore[arg-type]
    return templates.TemplateResponse(
        request,
        "_chat_turn.html",
        {
            "user_text": text or ("[image only]" if saved_path else ""),
            "response": result["response"],
            "tier": result["tier"],
            "spoken_on": result.get("spoken_on"),
            "image_path": saved_path,
        },
    )
