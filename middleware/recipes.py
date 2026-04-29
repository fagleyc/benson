"""Recipe ingestion: photo, video URL, manual.

Photos go to Claude Opus 4.7 vision (3.75 MP supported, handles
handwritten cards). Videos are downloaded with yt-dlp, transcribed with
openai-whisper (turbo model on CUDA), captions read from the platform's
metadata, then handed to Claude for recipe extraction.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json

from config import (
    PG_DSN,
    RECIPE_MEDIA_DIR,
)

logger = logging.getLogger("benson.recipes")


_EXTRACTION_PROMPT = (
    "Extract the recipe from this image. Verify your extraction is "
    "complete before responding. Return valid JSON with keys: title, "
    "ingredients (list of objects with text, name, quantity, unit), "
    "steps (list of strings), tags (list), notes. Return ONLY the JSON, "
    "no markdown."
)

_VIDEO_EXTRACTION_PROMPT = (
    "Extract a recipe from this cooking video. Combine information from "
    "BOTH the transcript and the caption — captions often have ingredient "
    "lists and quantities the narrator skips, while transcripts capture "
    "live commentary and technique tips.\n\n"
    "Return valid JSON with keys: title, ingredients (list of objects "
    "with text, name, quantity, unit), steps (list of strings), course "
    "(Main/Side/Sauce/Dessert/Drink/Other), prep_time (integer minutes, "
    "best estimate), tags (list of strings — cuisine, diet, etc.), notes "
    "(creator name + any tips). Return ONLY the JSON, no markdown.\n\n"
    "TITLE: {title}\nUPLOADER: {uploader}\n\nCAPTION:\n{caption}\n\n"
    "TRANSCRIPT:\n{transcript}"
)


# Module-level lazy whisper model (loaded once, cached for life of process).
_WHISPER_MODEL = None


def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import torch
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 'turbo' = large-v3-turbo, ~6x faster than large-v3 at near-equal accuracy.
        logger.info(f"loading whisper model 'turbo' on {device}")
        _WHISPER_MODEL = whisper.load_model("turbo", device=device)
    return _WHISPER_MODEL


def _media_type_for(image_path: str | Path) -> str:
    ext = str(image_path).lower().rsplit(".", 1)[-1]
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext, "image/jpeg")


def _extract_json_text(text: str) -> str:
    """Strip optional markdown fences and return the inner JSON."""
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence (```json / ```)
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


class RecipeIngester:
    def _conn(self):
        return psycopg2.connect(**PG_DSN)

    async def from_image(self, image_path: str | Path) -> dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # Vision via OAuth (Claude Code subscription quota, no per-token cost).
        from oauth_oneshot import ask_with_image
        try:
            text = await ask_with_image(
                str(image_path), _EXTRACTION_PROMPT, model="sonnet", timeout_s=120
            )
        except Exception:
            logger.exception(f"vision extract failed for {image_path}")
            raise
        if not text:
            raise RuntimeError(f"vision extract returned empty for {image_path}")
        try:
            recipe = json.loads(_extract_json_text(text))
        except json.JSONDecodeError as e:
            logger.exception(f"vision extract returned non-JSON for {image_path}: {text[:300]}")
            raise RuntimeError(f"Claude vision response wasn't parseable JSON: {e}") from e

        try:
            recipe_id = await asyncio.to_thread(
                self._save, recipe, "photo", image_path=str(image_path)
            )
        except Exception:
            logger.exception(f"DB save failed for recipe from {image_path}: recipe={recipe!r}")
            raise

        recipe["id"] = recipe_id
        return recipe

    async def from_video_url(self, url: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            audio = work / "audio.wav"
            meta = await asyncio.to_thread(self._download_media, url, audio, work)
            transcript = await asyncio.to_thread(self._transcribe, audio)

        prompt = _VIDEO_EXTRACTION_PROMPT.format(
            title=meta.get("title", "") or "",
            uploader=meta.get("uploader", "") or meta.get("channel", "") or "",
            caption=(meta.get("description") or "")[:4000],
            transcript=(transcript or "")[:8000],
        )
        # OAuth path — no API charge. Use Sonnet for accurate JSON extraction.
        from oauth_oneshot import ask as oauth_ask
        text = await oauth_ask(prompt, model="sonnet", timeout_s=90)
        if not text:
            raise RuntimeError("OAuth Claude call returned empty for recipe extraction")
        recipe = json.loads(_extract_json_text(text))
        recipe_id = await asyncio.to_thread(
            self._save, recipe, "video",
            source_url=url,
            image_url=meta.get("thumbnail"),
        )
        recipe["id"] = recipe_id
        recipe["source_url"] = url
        recipe["transcript_chars"] = len(transcript or "")
        recipe["caption_chars"] = len(meta.get("description") or "")
        return recipe

    def _download_media(self, url: str, audio_path: Path, work_dir: Path) -> dict:
        """Download audio + metadata. Returns yt-dlp info dict (caption, title, uploader, thumbnail)."""
        subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--write-info-json",
                "--no-playlist",
                "--no-warnings",
                "-o", str(audio_path.with_suffix(".%(ext)s")),
                url,
            ],
            check=True,
            timeout=180,
        )
        # yt-dlp writes the info file next to the audio with .info.json suffix.
        info_path = audio_path.with_suffix(".info.json")
        if info_path.exists():
            try:
                return json.loads(info_path.read_text())
            except Exception:
                return {}
        return {}

    def _transcribe(self, audio_path: Path) -> str:
        import torch
        model = _get_whisper_model()
        use_fp16 = torch.cuda.is_available()
        result = model.transcribe(str(audio_path), fp16=use_fp16)
        return (result.get("text") or "").strip()


    def _save(
        self,
        recipe: dict,
        source: str,
        *,
        source_url: str | None = None,
        image_path: str | None = None,
        image_url: str | None = None,
    ) -> int:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recipes
                    (title, source, source_url, ingredients, steps, tags,
                     image_path, image_url, course, prep_time, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    recipe.get("title", "Untitled"),
                    source,
                    source_url,
                    Json(recipe.get("ingredients", [])),
                    Json(recipe.get("steps", [])),
                    Json(recipe.get("tags", [])),
                    image_path,
                    image_url,
                    recipe.get("course"),
                    recipe.get("prep_time"),
                    recipe.get("notes", "") or None,
                ),
            )
            conn.commit()
            return cur.fetchone()[0]
