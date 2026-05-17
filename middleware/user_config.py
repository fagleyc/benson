"""User Config / Family Enrollment wizard (Slice 1).

Routes mounted at /advanced/user-config:
  GET    /                              page (cards + Add User)
  POST   /start                         {name, role} -> {enrollment_id}
  POST   /{enrollment_id}/voice-sample  multipart audio upload
  POST   /{enrollment_id}/photo         multipart image upload
  POST   /{enrollment_id}/interview     {q: a, ...}
  POST   /{enrollment_id}/complete      finalize -> averaged voiceprint
  GET    /status                        JSON list of enrolled users
  POST   /{name}/delete                 remove voiceprint files (keep .md)

Slice 2 will read /opt/benson/memory/voiceprints/*.{npy,json} and call
voiceprint.identify() to label live audio.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import voiceprint as vp
import wyoming_whisper as ww

logger = logging.getLogger("benson.user_config")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/advanced/user-config", tags=["user-config"])

MEMORY_DIR = Path("/opt/benson/memory")
AVATARS_DIR = Path(__file__).parent / "static" / "avatars"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)

WORK_DIR = Path("/tmp/benson-enrollments")
WORK_DIR.mkdir(parents=True, exist_ok=True)

VALID_ROLES = {"parent", "teen", "child", "guest"}
MIN_SAMPLE_S = 8.0
MAX_SAMPLE_S = 30.0
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")

INTERVIEW_QUESTIONS: list[dict[str, str]] = [
    {"key": "greeting_name", "q": "What name should Benson use when greeting you?"},
    {"key": "diet", "q": "Any food allergies, dietary restrictions, or strong dislikes I should remember?"},
    {"key": "cuisines", "q": "What kinds of meals or cuisines do you love?"},
    {"key": "weekday_routine", "q": "What's your work or school routine on a typical weekday?"},
    {"key": "hobbies", "q": "What are your main hobbies or pastimes?"},
    {"key": "important_people", "q": "Important people in your life I should know about (family, close friends, coworkers)?"},
    {"key": "tone", "q": "How do you like Benson to talk to you — brief and dry, warm and chatty, somewhere in between?"},
    {"key": "media_taste", "q": "Music or shows you'd enjoy a recommendation about?"},
    {"key": "wellness", "q": "Anything I should keep in mind around your sleep, stress, or focus?"},
    {"key": "anything_else", "q": "Anything else you want Benson to remember about you?"},
]

SCRIPTS = [
    (
        "The quick brown fox jumps over the lazy dog by the riverside. "
        "Five sleepy dolphins drifted past, watching the sunlight paint the water gold. "
        "I would happily trade a busy afternoon for one quiet hour outdoors."
    ),
    (
        "Underneath the orange canopy, Theodore stirred his coffee and counted the cars. "
        "Buses, bicycles, scooters, and the occasional skateboard rolled past his bench. "
        "Each year the city felt a little louder, a little brighter, a little stranger."
    ),
    (
        "Pizza, pasta, pancakes, and pickles — pick any pair and I'll be happy. "
        "She whispered that the recipe required exactly three teaspoons of cinnamon, no more, no less. "
        "Tomorrow we'll bake until the kitchen smells like vanilla, honey, and warm bread."
    ),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


RESERVED_SLUGS = {"household", "index", "digests", "raw", "voiceprints"}


def _slugify(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:30] or ""


def _check_reserved(slug: str) -> None:
    if slug in RESERVED_SLUGS:
        raise HTTPException(
            400,
            f"'{slug}' is reserved (collides with Benson's household memory infrastructure)",
        )


def _enrollment_state_path(enrollment_id: str) -> Path:
    return WORK_DIR / enrollment_id / "state.json"


def _load_state(enrollment_id: str) -> dict:
    p = _enrollment_state_path(enrollment_id)
    if not p.exists():
        raise HTTPException(404, f"enrollment {enrollment_id} not found")
    return json.loads(p.read_text())


def _save_state(enrollment_id: str, state: dict) -> None:
    p = _enrollment_state_path(enrollment_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ─── Page ────────────────────────────────────────────────────────────────
@router.get("", response_class=HTMLResponse)
async def user_config_page(request: Request):
    enrolled = await asyncio.to_thread(vp.list_enrolled)
    ctx = {
        "active": "advanced",
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "enrolled": enrolled,
        "questions": INTERVIEW_QUESTIONS,
        "scripts": SCRIPTS,
        "min_s": MIN_SAMPLE_S,
        "max_s": MAX_SAMPLE_S,
    }
    return templates.TemplateResponse(request, "user_config.html", ctx)


# ─── Status (JSON for dashboards) ────────────────────────────────────────
@router.get("/status")
async def user_config_status() -> dict[str, Any]:
    return {"enrolled": await asyncio.to_thread(vp.list_enrolled)}


# ─── Existing details (for the modify flow) ──────────────────────────────
@router.get("/{name}/details")
async def user_details(name: str) -> dict[str, Any]:
    slug = _slugify(name)
    _check_reserved(slug)
    meta = vp.load_meta(slug)
    if not meta:
        raise HTTPException(404, f"{slug} is not enrolled")
    md_path = MEMORY_DIR / f"{slug}.md"
    md_size = md_path.stat().st_size if md_path.exists() else 0
    emb_path = vp._emb_path(slug)
    voiceprint_size = emb_path.stat().st_size if emb_path.exists() else 0
    samples, memory_preview = await asyncio.to_thread(
        _collect_samples_and_preview, slug, md_path
    )
    return {
        "name": slug,
        "display_name": meta.get("display_name") or slug,
        "role": meta.get("role"),
        "photo": meta.get("photo"),
        "sample_count": int(meta.get("sample_count", 0)),
        "enrolled_at": meta.get("enrolled_at"),
        "last_updated_at": meta.get("last_updated_at"),
        "interview": meta.get("interview") or {},
        "memory_file": str(md_path),
        "memory_path": str(md_path),
        "memory_size": md_size,
        "memory_preview": memory_preview,
        "voiceprint_size": voiceprint_size,
        "samples": samples,
    }


def _collect_samples_and_preview(slug: str, md_path: Path) -> tuple[list[dict], str]:
    raw_dir = vp.RAW_DIR / slug
    samples: list[dict] = []
    if raw_dir.is_dir():
        wavs = sorted(
            (p for p in raw_dir.glob("*.wav") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for idx, p in enumerate(wavs):
            try:
                dur = _probe_duration(p)
            except Exception:
                dur = 0.0
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                recorded_at = mtime.isoformat(timespec="seconds")
            except OSError:
                recorded_at = None
            samples.append({
                "idx": idx,
                "filename": p.name,
                "duration_s": round(float(dur), 2),
                "recorded_at": recorded_at,
                "audio_url": f"/advanced/user-config/{slug}/sample/{idx}",
            })
    preview = ""
    try:
        if md_path.exists():
            preview = md_path.read_text(errors="replace")[:400]
    except OSError:
        preview = ""
    return samples, preview


# ─── Raw sample download ─────────────────────────────────────────────────
@router.get("/{name}/sample/{idx}")
async def get_sample(name: str, idx: int):
    slug = _slugify(name)
    _check_reserved(slug)
    if not vp.load_meta(slug):
        raise HTTPException(404, f"{slug} is not enrolled")
    raw_dir = vp.RAW_DIR / slug
    if not raw_dir.is_dir():
        raise HTTPException(404, "no raw samples")

    def _pick() -> Path | None:
        wavs = sorted(
            (p for p in raw_dir.glob("*.wav") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if idx < 0 or idx >= len(wavs):
            return None
        return wavs[idx]

    wav = await asyncio.to_thread(_pick)
    if wav is None:
        raise HTTPException(404, f"sample {idx} not found")
    return FileResponse(
        str(wav), media_type="audio/wav", filename=wav.name
    )


# ─── Inline edit of a single interview answer ────────────────────────────
@router.post("/{name}/interview/{q_key}")
async def edit_interview_answer(
    name: str, q_key: str, request: Request
) -> dict[str, Any]:
    slug = _slugify(name)
    _check_reserved(slug)
    known_keys = {q["key"] for q in INTERVIEW_QUESTIONS}
    if q_key not in known_keys:
        raise HTTPException(400, f"unknown interview key '{q_key}'")
    body = await request.json()
    answer = (body or {}).get("answer")
    if not isinstance(answer, str):
        raise HTTPException(400, "answer must be a string")
    answer = answer.strip()
    if not answer:
        raise HTTPException(400, "answer cannot be empty")
    meta = vp.load_meta(slug)
    if not meta:
        raise HTTPException(404, f"{slug} is not enrolled")

    interview = dict(meta.get("interview") or {})
    interview[q_key] = answer
    meta["interview"] = interview
    meta["last_updated_at"] = _now_iso()
    vp.write_meta(slug, meta)

    q_text = next(q["q"] for q in INTERVIEW_QUESTIONS if q["key"] == q_key)
    md_path = MEMORY_DIR / f"{slug}.md"
    today = date.today().isoformat()
    section_lines = [
        f"## Inline edit {today}",
        "",
        f"- {q_text} — {answer}",
        "",
    ]
    section = "\n".join(section_lines)

    def _append_md() -> None:
        if md_path.exists():
            existing = md_path.read_text()
            if not existing.endswith("\n"):
                existing += "\n"
            md_path.write_text(existing + "\n" + section)
        else:
            header = (
                f"# {meta.get('display_name') or slug}\n\n"
                f"Facts and context about {meta.get('display_name') or slug}, "
                "accumulated from past conversations.\n\n"
            )
            md_path.write_text(header + section)

    await asyncio.to_thread(_append_md)
    return {
        "ok": True,
        "name": slug,
        "key": q_key,
        "answer": answer,
        "last_updated_at": meta["last_updated_at"],
        "memory_path": str(md_path),
    }


# ─── Recent speakers aggregator ──────────────────────────────────────────
@router.get("/recent-speakers")
async def recent_speakers() -> dict[str, Any]:
    enrolled = await asyncio.to_thread(vp.list_enrolled)
    out: dict[str, Any] = {}
    # LATEST_BY_ROOM is keyed by room, not name — best we can do without
    # synthesizing fake per-name cache state is to expose the overall
    # latest and let the UI badge whoever matches.
    overall = ww._lookup_speaker(None)
    for u in enrolled:
        nm = (u.get("name") or "").lower()
        if not nm:
            continue
        if (
            not overall.get("stale")
            and (overall.get("speaker") or "").lower() == nm
        ):
            out[nm] = {
                "speaker": overall.get("speaker"),
                "age_s": overall.get("age_s"),
                "confidence": overall.get("confidence"),
            }
        else:
            out[nm] = {"speaker": None, "age_s": None, "confidence": 0.0}
    return {"latest_overall": overall, "by_name": out}


# ─── Start ───────────────────────────────────────────────────────────────
@router.post("/start")
async def start_enrollment(request: Request) -> dict[str, Any]:
    body = await request.json()
    name_in = (body.get("name") or "").strip()
    role = (body.get("role") or "").strip().lower()
    if not name_in:
        raise HTTPException(400, "name required")
    if role not in VALID_ROLES:
        raise HTTPException(400, f"role must be one of {sorted(VALID_ROLES)}")
    slug = _slugify(name_in)
    if not slug or not NAME_RE.match(slug):
        raise HTTPException(400, "name must be alphanumeric (a-z, 0-9, _, -)")
    _check_reserved(slug)
    enrollment_id = uuid.uuid4().hex[:12]
    state = {
        "enrollment_id": enrollment_id,
        "name": slug,
        "display_name": name_in,
        "role": role,
        "samples": [],
        "interview": {},
        "photo": None,
        "started_at": _now_iso(),
    }
    _save_state(enrollment_id, state)
    raw_dir = vp.RAW_DIR / slug
    raw_dir.mkdir(parents=True, exist_ok=True)
    return {
        "enrollment_id": enrollment_id,
        "name": slug,
        "display_name": name_in,
        "role": role,
        "already_enrolled": vp._meta_path(slug).exists(),
    }


# ─── Voice sample upload ─────────────────────────────────────────────────
@router.post("/{enrollment_id}/voice-sample")
async def upload_voice_sample(
    enrollment_id: str, request: Request, audio: UploadFile = File(...)
) -> dict[str, Any]:
    state = _load_state(enrollment_id)
    body = await audio.read()
    if not body:
        raise HTTPException(400, "empty upload")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            400, f"upload too large ({len(body)} bytes, cap {MAX_UPLOAD_BYTES})"
        )
    name = state["name"]
    raw_dir = vp.RAW_DIR / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_id = uuid.uuid4().hex[:12]
    in_suffix = (audio.filename or "").lower().split(".")[-1] or "webm"
    # Always write input under a distinct .src.<ext> name so even when the
    # caller sends a WAV the ffmpeg in/out paths can't collide.
    in_path = raw_dir / f"{sample_id}.src.{in_suffix}"
    out_path = raw_dir / f"{sample_id}.wav"
    in_path.write_bytes(body)

    # ffmpeg → 16 kHz mono PCM WAV
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(out_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not out_path.exists():
        try:
            in_path.unlink()
        except OSError:
            pass
        raise HTTPException(
            400,
            f"ffmpeg failed: {stderr[:300].decode(errors='replace')}",
        )
    try:
        in_path.unlink()
    except OSError:
        pass

    duration_s = await asyncio.to_thread(_probe_duration, out_path)
    if duration_s < MIN_SAMPLE_S:
        out_path.unlink(missing_ok=True)
        raise HTTPException(
            400, f"sample too short ({duration_s:.1f}s; need >= {MIN_SAMPLE_S}s)"
        )
    if duration_s > MAX_SAMPLE_S:
        out_path.unlink(missing_ok=True)
        raise HTTPException(
            400, f"sample too long ({duration_s:.1f}s; cap {MAX_SAMPLE_S}s)"
        )

    state["samples"].append({
        "path": str(out_path),
        "duration_s": round(duration_s, 2),
        "recorded_at": _now_iso(),
    })
    _save_state(enrollment_id, state)
    return {
        "ok": True,
        "sample_count": len(state["samples"]),
        "duration_s": round(duration_s, 2),
    }


def _probe_duration(wav_path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return float(info.duration)
    except Exception:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 str(wav_path)],
                capture_output=True, text=True, timeout=10,
            )
            return float((r.stdout or "0").strip())
        except Exception:
            return 0.0


# ─── Photo upload ────────────────────────────────────────────────────────
@router.post("/{enrollment_id}/photo")
async def upload_photo(
    enrollment_id: str, photo: UploadFile = File(...)
) -> dict[str, Any]:
    state = _load_state(enrollment_id)
    body = await photo.read()
    if not body:
        raise HTTPException(400, "empty photo")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "photo too large")
    name = state["name"]
    out_path = AVATARS_DIR / f"{name}.png"
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(body))
        img.thumbnail((256, 256))
        # Convert paletted/RGBA to RGB on a neutral background so PNG saves
        # consistently.
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        img.save(out_path, "PNG", optimize=True)
    except Exception as e:
        raise HTTPException(400, f"image decode failed: {e}")
    rel = f"/static/avatars/{name}.png"
    state["photo"] = rel
    _save_state(enrollment_id, state)
    return {"ok": True, "photo": rel}


# ─── Interview ───────────────────────────────────────────────────────────
@router.post("/{enrollment_id}/interview")
async def save_interview(enrollment_id: str, request: Request) -> dict[str, Any]:
    state = _load_state(enrollment_id)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "expected JSON object {key: answer, ...}")
    cleaned: dict[str, str] = {}
    for q in INTERVIEW_QUESTIONS:
        v = body.get(q["key"])
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if v:
            cleaned[q["key"]] = v
    state["interview"] = cleaned
    _save_state(enrollment_id, state)
    return {"ok": True, "answered": len(cleaned)}


# ─── Complete: average voiceprint, write metadata, append memory file ────
@router.post("/{enrollment_id}/complete")
async def complete_enrollment(enrollment_id: str) -> dict[str, Any]:
    state = _load_state(enrollment_id)
    name = state["name"]
    samples = state.get("samples", [])
    # In re-enroll mode we allow zero new voice samples as long as there's
    # an existing voiceprint to keep AND there's at least an interview
    # update — otherwise complete is a no-op.
    prior_meta = vp.load_meta(name)
    has_prior_voice = bool(prior_meta) and (vp._emb_path(name).exists())
    interview = state.get("interview") or {}
    if not samples and not has_prior_voice:
        raise HTTPException(400, "no voice samples uploaded and no prior voiceprint")
    if not samples and not interview:
        raise HTTPException(400, "nothing to update (no new samples, no new interview answers)")

    def _embed_all() -> list[dict]:
        # Collect ALL embeddings first, then fold into the voiceprint in
        # one merge call — otherwise per-sample merges all see prev_count=0
        # (meta isn't written until after the loop) and the .npy ends up
        # holding only the last sample, defeating the multi-sample average.
        out: list[dict] = []
        embs: list = []
        for s in samples:
            wav_path = Path(s["path"])
            if not wav_path.exists():
                continue
            try:
                emb = vp.extract_embedding(wav_path)
                embs.append(emb)
                out.append({"path": s["path"]})
            except Exception as e:
                logger.exception(f"embedding failed for {wav_path}: {e}")
        if embs:
            _avg, count = vp.merge_voiceprint(name, embs)
            for r in out:
                r["sample_count_after"] = count
        return out

    if samples:
        processed = await asyncio.to_thread(_embed_all)
        if not processed:
            raise HTTPException(500, "all embeddings failed")
    else:
        processed = []  # interview-only update

    meta = vp.load_meta(name)
    existing_samples = meta.get("samples") or []
    appended_samples = existing_samples + [
        {
            "path": s["path"],
            "duration_s": s.get("duration_s"),
            "recorded_at": s.get("recorded_at"),
        }
        for s in samples
    ]
    interview = state.get("interview") or {}
    merged_interview = dict(meta.get("interview") or {})
    merged_interview.update(interview)

    final_meta = {
        "name": name,
        "display_name": state.get("display_name") or meta.get("display_name") or name,
        "role": state.get("role") or meta.get("role"),
        "photo": state.get("photo") or meta.get("photo"),
        "sample_count": len(appended_samples),
        "samples": appended_samples,
        "enrolled_at": meta.get("enrolled_at") or state.get("started_at") or _now_iso(),
        "last_updated_at": _now_iso(),
        "interview": merged_interview,
    }
    vp.write_meta(name, final_meta)

    md_path = MEMORY_DIR / f"{name}.md"
    today = date.today().isoformat()
    section = _build_memory_section(state, today)
    if md_path.exists():
        body = md_path.read_text()
        if not body.endswith("\n"):
            body += "\n"
        body += "\n" + section
        md_path.write_text(body)
    else:
        header = f"# {state.get('display_name') or name}\n\nFacts and context about {state.get('display_name') or name}, accumulated from past conversations.\n\n"
        md_path.write_text(header + section)

    avg_emb_path = vp._emb_path(name)
    avg_shape = None
    if avg_emb_path.exists():
        import numpy as np
        avg_shape = list(np.load(avg_emb_path).shape)

    return {
        "ok": True,
        "name": name,
        "sample_count": final_meta["sample_count"],
        "voiceprint_path": str(avg_emb_path),
        "voiceprint_shape": avg_shape,
        "metadata_path": str(vp._meta_path(name)),
        "memory_path": str(md_path),
    }


def _build_memory_section(state: dict, today: str) -> str:
    interview = state.get("interview") or {}
    role = state.get("role")
    display = state.get("display_name") or state.get("name")
    lines: list[str] = [f"## Enrollment {today}", ""]
    lines.append(f"- Display name: {display}")
    if role:
        lines.append(f"- Role: {role}")
    for q in INTERVIEW_QUESTIONS:
        a = interview.get(q["key"])
        if not a:
            continue
        lines.append(f"- {q['q']} — {a}")
    lines.append("")
    return "\n".join(lines)


# ─── Delete (keep memory .md) ────────────────────────────────────────────
@router.post("/{name}/delete")
async def delete_user(name: str) -> dict[str, Any]:
    slug = _slugify(name)
    if not slug:
        raise HTTPException(400, "bad name")
    _check_reserved(slug)
    result = await asyncio.to_thread(vp.delete, slug)
    return {"ok": True, **result}
