"""House Hub — HTML frontend for the Fagley household.

Mounted on the same FastAPI app as the API. Serves Jinja-rendered
Bootswatch-Darkly pages plus an HTMX-driven floating "Talk to Benson"
chat widget. The widget posts to /hub/chat which forwards to the
existing /conversation pipeline.
"""
from __future__ import annotations

import asyncio
import logging
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
    where = []
    params: list = []
    if q:
        where.append("title ILIKE %s")
        params.append(f"%{q}%")
    if course:
        where.append("course ILIKE %s")
        params.append(course)
    sql = "SELECT id, title, course, prep_time, image_url, user_rating FROM recipes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY title LIMIT 200"
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
    persons = ["Casey", "Lindsey", "Cole", "Zander", "Household"]
    chores = await asyncio.to_thread(
        _query,
        """
        SELECT id, person, chore_date, chore_name, done
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
        d = c["chore_date"].isoformat() if c["chore_date"] else None
        if d and d in chores_map[p]:
            chores_map[p][d].append(c)
        elif not d:
            # Undated chores → put in today's column for the person
            chores_map[p][today.isoformat()].append(c)
    return templates.TemplateResponse(
        request, "chores.html",
        _ctx("chores", days=days, persons=persons, chores_map=chores_map),
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
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
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


@router.post("/hub/chat", response_class=HTMLResponse)
async def hub_chat(
    request: Request,
    text: str = Form(...),
    speaker: str = Form("Casey"),
    room: str = Form("hub"),
    voice_input: str = Form(""),  # "1" if mic was used
):
    """Forward to /conversation pipeline; return HTML chat-turn fragment.

    Imported lazily to avoid a circular import when hub is included in main.
    """
    from main import handle_conversation

    payload = {
        "text": text,
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
            "user_text": text,
            "response": result["response"],
            "tier": result["tier"],
            "spoken_on": result.get("spoken_on"),
        },
    )
