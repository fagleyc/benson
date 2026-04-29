"""'Listen in' mode — passive-listening dinner-table memory enrichment.

Casey says "Benson, listen in" → Benson signals the iPad's Benson Eye
page to start MediaRecorder audio capture. The page POSTs ~30-second
WebM/Opus chunks here. When Casey says "stop" (or auto-stop fires),
we concatenate, transcribe with Whisper (turbo on CUDA), send the
transcript to Claude via OAuth to extract durable per-person facts,
and append them to the memory files. Audio is deleted after
transcription — only the transcript persists.

Privacy guardrails:
- Default OFF. Always opt-in.
- iPad page shows a pulsing visible indicator while active.
- A hard auto-stop fires regardless (default 90 min).
- Audio files removed after transcription succeeds.
- Every session logged in `listening_sessions` for audit.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from psycopg2.extras import RealDictCursor

from config import PG_DSN

logger = logging.getLogger("benson.listening")
router = APIRouter()

CHUNK_DIR = Path("/tmp/benson-listening")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

TRANSCRIPT_DIR = Path("/opt/benson/memory/listen-transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def _conn():
    return psycopg2.connect(**PG_DSN)


def _q(sql: str, params: tuple = (), one: bool = False):
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return None
        if one:
            r = cur.fetchone()
            return dict(r) if r else None
        return [dict(r) for r in cur.fetchall()]


# ─── Session lifecycle ───────────────────────────────────────────────────
@router.post("/listening/start")
async def listening_start(request: Request) -> dict:
    body = await request.json()
    room = (body.get("room") or "kitchen").strip()
    duration_min = int(body.get("duration_min") or 90)
    duration_min = max(5, min(duration_min, 180))  # clamp 5min..3hr
    started_by = (body.get("started_by") or "").strip() or "unknown"

    auto_stop = datetime.now(timezone.utc) + timedelta(minutes=duration_min)
    row = _q(
        """
        INSERT INTO listening_sessions
            (started_by, room, duration_min_requested, auto_stop_at, status)
        VALUES (%s, %s, %s, %s, 'active')
        RETURNING id, started_at, auto_stop_at
        """,
        (started_by, room, duration_min, auto_stop),
        one=True,
    )
    session_id = row["id"]
    chunk_dir = CHUNK_DIR / str(session_id)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Tell the iPad to start recording.
    from camera_handler import CONNECTED
    info = CONNECTED.get(room)
    if info:
        try:
            await info["ws"].send_text(json.dumps({
                "action": "start_listening",
                "session_id": session_id,
                "duration_min": duration_min,
            }))
        except Exception as e:
            logger.warning(f"failed to signal {room} iPad: {e}")

    logger.info(f"listening session {session_id} started by {started_by} in {room} for {duration_min}min")
    return {
        "ok": True,
        "session_id": session_id,
        "auto_stop_at": auto_stop.isoformat(),
        "room": room,
        "ipad_signaled": bool(info),
    }


@router.post("/listening/chunk/{session_id}")
async def listening_chunk(session_id: int, request: Request) -> dict:
    body = await request.body()
    if not body or len(body) < 500:
        raise HTTPException(400, f"chunk too small ({len(body)} bytes)")
    s = _q("SELECT status FROM listening_sessions WHERE id = %s", (session_id,), one=True)
    if not s or s["status"] != "active":
        raise HTTPException(409, f"session {session_id} not active")
    chunk_dir = CHUNK_DIR / str(session_id)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    # Use millisecond timestamp so chunks sort chronologically
    ext = "webm"
    ct = (request.headers.get("content-type") or "").lower()
    if "ogg" in ct: ext = "ogg"
    elif "mp4" in ct or "m4a" in ct: ext = "m4a"
    elif "wav" in ct: ext = "wav"
    fn = chunk_dir / f"{int(time.time() * 1000):013d}.{ext}"
    fn.write_bytes(body)
    _q(
        """
        UPDATE listening_sessions
        SET chunks_received = chunks_received + 1,
            bytes_received = bytes_received + %s
        WHERE id = %s
        """,
        (len(body), session_id),
    )
    return {"ok": True, "session_id": session_id, "chunk_bytes": len(body)}


@router.post("/listening/stop/{session_id}")
async def listening_stop(session_id: int) -> dict:
    s = _q("SELECT * FROM listening_sessions WHERE id = %s", (session_id,), one=True)
    if not s:
        raise HTTPException(404, "session not found")
    if s["status"] != "active":
        return {"ok": True, "already": s["status"]}
    _q(
        "UPDATE listening_sessions SET status='processing', ended_at=NOW() WHERE id = %s",
        (session_id,),
    )
    # Tell iPad to stop
    from camera_handler import CONNECTED
    info = CONNECTED.get(s["room"])
    if info:
        try:
            await info["ws"].send_text(json.dumps({"action": "stop_listening", "session_id": session_id}))
        except Exception:
            pass
    # Kick off background processing
    asyncio.create_task(_process_session(session_id))
    return {"ok": True, "session_id": session_id, "status": "processing"}


@router.get("/listening/status")
async def listening_status() -> dict:
    rows = _q(
        """
        SELECT id, started_at, ended_at, started_by, room, status,
               chunks_received, bytes_received, transcript_chars, facts_extracted,
               summary, auto_stop_at
        FROM listening_sessions
        ORDER BY started_at DESC LIMIT 20
        """
    )
    active = [r for r in rows if r["status"] == "active"]
    # Auto-stop expired sessions
    for r in active:
        if r["auto_stop_at"] and datetime.now(timezone.utc) > r["auto_stop_at"]:
            asyncio.create_task(listening_stop(r["id"]))
    return {"recent": rows, "active_count": len(active)}


# ─── Background processing ──────────────────────────────────────────────
async def _process_session(session_id: int) -> None:
    """Concat chunks → Whisper transcribe → Claude extract → memory append."""
    s = _q("SELECT * FROM listening_sessions WHERE id = %s", (session_id,), one=True)
    if not s:
        return
    chunk_dir = CHUNK_DIR / str(session_id)
    chunks = sorted(chunk_dir.glob("*.*")) if chunk_dir.exists() else []
    if not chunks:
        _q(
            "UPDATE listening_sessions SET status='complete', summary='no audio captured' WHERE id = %s",
            (session_id,),
        )
        return

    started_by = s.get("started_by") or "unknown"
    room = s.get("room") or "unknown"
    started_at = s["started_at"]

    # 1. Concat chunks via ffmpeg (Whisper handles container variety, but
    # concatenating to one wav avoids re-loading the model).
    concat_wav = chunk_dir / "session.wav"
    list_file = chunk_dir / "list.txt"
    list_file.write_text("\n".join(f"file '{c.name}'" for c in chunks))
    concat_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-ac", "1", "-ar", "16000",
        str(concat_wav),
    ]
    proc = await asyncio.create_subprocess_exec(
        *concat_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(chunk_dir),
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(f"session {session_id} ffmpeg failed: {stderr[:500].decode(errors='replace')}")
        _q(
            "UPDATE listening_sessions SET status='aborted', summary='ffmpeg concat failed' WHERE id = %s",
            (session_id,),
        )
        return

    # 2. Whisper transcribe (reuse the model loaded for recipes)
    def _transcribe() -> str:
        from recipes import _get_whisper_model
        import torch
        model = _get_whisper_model()
        result = model.transcribe(str(concat_wav), fp16=torch.cuda.is_available())
        return (result.get("text") or "").strip()

    try:
        transcript = await asyncio.to_thread(_transcribe)
    except Exception as e:
        logger.exception(f"session {session_id} whisper failed")
        _q(
            "UPDATE listening_sessions SET status='aborted', summary=%s WHERE id = %s",
            (f"whisper failed: {e}", session_id),
        )
        return

    transcript = (transcript or "").strip()
    if len(transcript) < 50:
        _q(
            "UPDATE listening_sessions SET status='complete', summary='transcript too short — nothing to extract' WHERE id = %s",
            (session_id,),
        )
        _cleanup(chunk_dir)
        return

    # 3. Save transcript
    ts_path = TRANSCRIPT_DIR / f"{started_at.strftime('%Y-%m-%d_%H%M')}_{room}_session{session_id}.md"
    header = (
        f"# Listening session {session_id}\n\n"
        f"- started_by: {started_by}\n"
        f"- room: {room}\n"
        f"- started_at: {started_at.isoformat()}\n"
        f"- chunks: {len(chunks)}\n\n"
        "## Transcript\n\n"
    )
    ts_path.write_text(header + transcript + "\n")

    # 4. Claude extracts per-person facts
    extraction_prompt = (
        "You are reviewing a passive-listening transcript from a Fagley "
        "household dinner conversation. The audio was captured at "
        f"{started_at.strftime('%A %B %d')} from the {room}.\n\n"
        "Your job: pull out DURABLE per-person facts that are worth "
        "remembering long-term. Examples of durable: food preferences, "
        "allergies, recurring schedules, friends mentioned, ongoing "
        "projects, family relationships, opinions stated firmly. NOT "
        "durable: passing comments about today's weather, mood snapshots, "
        "things already in their memory file, transient frustrations.\n\n"
        "Household members: Casey (homeowner, dad), Lindsey (homeowner, "
        "mom), Cole (14, son), Zander (6, son), Sherry (Lindsey's mom, "
        "frequent guest), Bluey (dog).\n\n"
        "Return STRICT JSON only. Schema:\n"
        '{"facts": [{"person": "Casey|Lindsey|Cole|Zander|Sherry|Household", '
        '"fact": "the durable fact in 1 sentence"}, ...]}\n\n'
        "If no durable facts surfaced, return {\"facts\": []}. Don't "
        "invent — only extract what's actually said.\n\n"
        f"TRANSCRIPT:\n{transcript[:20000]}"
    )
    from oauth_oneshot import ask as oauth_ask
    raw = await oauth_ask(extraction_prompt, model="sonnet", timeout_s=120)
    facts: list[dict] = []
    if raw:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned.rsplit("```", 1)[0]
            parsed = json.loads(cleaned.strip())
            facts = parsed.get("facts") or []
        except Exception as e:
            logger.warning(f"session {session_id} fact extraction parse failed: {e}; raw={raw[:300]}")

    # 5. Append facts to per-person memory files
    from memory_tools import memory_append
    appended_count = 0
    for f in facts:
        person = (f.get("person") or "").strip()
        fact = (f.get("fact") or "").strip()
        if not person or not fact:
            continue
        if person.lower() == "household":
            target = "household.md"
        else:
            target = f"{person.lower()}.md"
        try:
            memory_append(target, f"- {fact} _(from listening session {session_id} on {started_at.strftime('%Y-%m-%d')})_")
            appended_count += 1
        except Exception as e:
            logger.warning(f"memory_append failed for {person}: {e}")

    summary_text = f"{len(facts)} facts extracted, {appended_count} appended to memory files"
    _q(
        """
        UPDATE listening_sessions
        SET status='complete', transcript_chars=%s, facts_extracted=%s,
            summary=%s, transcript_path=%s
        WHERE id = %s
        """,
        (len(transcript), appended_count, summary_text, str(ts_path), session_id),
    )
    logger.info(f"session {session_id} complete: {summary_text}")

    # 6. Re-index memory + the transcript itself for future search
    try:
        from memory_index import index_memory_files
        await asyncio.to_thread(index_memory_files)
    except Exception as e:
        logger.warning(f"reindex after listening failed: {e}")

    # 7. Cleanup
    _cleanup(chunk_dir)


def _cleanup(chunk_dir: Path) -> None:
    try:
        for f in chunk_dir.glob("*"):
            f.unlink(missing_ok=True)
        chunk_dir.rmdir()
    except Exception as e:
        logger.warning(f"cleanup failed: {e}")
