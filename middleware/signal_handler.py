"""Signal inbound handler + outbound helper.

Talks to bbernhard/signal-cli-rest-api running on 127.0.0.1:8201.
Polls /v1/receive/{number} for inbound messages, routes recognized
senders' messages through /conversation, and sends replies via /v2/send.

Registration flow (one-time, GV number via voice):
  1. Casey solves captcha at https://signalcaptchas.org/registration/generate.html
  2. POSTs captcha + number to /signal/register → container calls Signal,
     Signal voice-calls the GV number with a 6-digit code
  3. Casey enters code, POST /signal/verify → done
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
from contextlib import suppress
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("benson.signal")
router = APIRouter()

SIGNAL_API = os.environ.get("SIGNAL_API_URL", "http://localhost:8201")


def _benson_number() -> str:
    """The Signal-registered phone number Benson speaks as."""
    return (os.environ.get("SIGNAL_BENSON_NUMBER", "") or "").strip()


def _allowed() -> set[str]:
    raw = os.environ.get("SIGNAL_ALLOWED_NUMBERS", "")
    out: set[str] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # Signal uses E.164: "+15551234567"
        if not tok.startswith("+"):
            tok = "+" + "".join(c for c in tok if c.isdigit())
        out.add(tok)
    return out


def _allowed_groups() -> set[str]:
    """Group base64 IDs (raw, as Signal returns them)."""
    raw = os.environ.get("SIGNAL_ALLOWED_GROUPS", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


def _allowed_uuids() -> set[str]:
    """Optional explicit UUID allowlist. Signal now delivers ACI UUIDs
    instead of phone numbers when a contact has phone-number privacy on,
    so the legacy `_allowed()` set may not match. Populate this via
    SIGNAL_ALLOWED_UUIDS env (comma-separated)."""
    raw = os.environ.get("SIGNAL_ALLOWED_UUIDS", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


def _phone_to_speaker() -> dict[str, str]:
    raw = os.environ.get("SIGNAL_SPEAKER_MAP", "")
    m: dict[str, str] = {}
    for tok in raw.split(","):
        if ":" in tok:
            num, name = tok.split(":", 1)
            num = num.strip()
            if not num.startswith("+"):
                num = "+" + "".join(c for c in num if c.isdigit())
            m[num] = name.strip()
    return m


def _known_first_names() -> dict[str, str]:
    """Lowercased first-name → display name from the speaker map. Used to
    match against Signal `sourceName` when the source is a UUID we don't
    have explicitly listed."""
    out: dict[str, str] = {}
    for display in _phone_to_speaker().values():
        first = display.strip().split()[0].lower() if display.strip() else ""
        if first:
            out[first] = display
    return out


def _resolve_speaker_by_name(source_name: str) -> str | None:
    """Match Signal sourceName ('Lindsey Schultz', 'Casey') against the
    speaker map by first name."""
    if not source_name:
        return None
    sn = source_name.strip().lower()
    if not sn:
        return None
    first = sn.split()[0]
    return _known_first_names().get(first)


# ─── Registration ────────────────────────────────────────────────────────
@router.post("/signal/register")
async def signal_register(request: Request) -> dict:
    """Kick off Signal registration. Default SMS; opt-in voice fallback.

    Voice path requires SMS-first-then-voice (Signal gates abuse with a
    60s wait). Only use voice if your number genuinely can't receive SMS.
    """
    body = await request.json()
    number = (body.get("number") or "").strip()
    captcha = (body.get("captcha") or "").strip()
    use_voice = bool(body.get("use_voice", False))
    if not number or not captcha:
        raise HTTPException(400, "number and captcha required")
    if not number.startswith("+"):
        number = "+" + "".join(c for c in number if c.isdigit())

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            if not use_voice:
                # Standard SMS path
                r = await client.post(
                    f"{SIGNAL_API}/v1/register/{number}",
                    json={"use_voice": False, "captcha": captcha},
                )
                if r.status_code >= 400:
                    return {"ok": False, "step": "sms", "status": r.status_code, "error": r.text}
                return {"ok": True, "number": number, "method": "sms",
                        "next": "check your phone/GV for an SMS code, enter it below"}

            # Voice path: SMS request first, wait 60s, then voice.
            r1 = await client.post(
                f"{SIGNAL_API}/v1/register/{number}",
                json={"use_voice": False, "captcha": captcha},
            )
            if r1.status_code >= 400:
                return {"ok": False, "step": "sms", "status": r1.status_code, "error": r1.text}

            await asyncio.sleep(65)

            r2 = await client.post(
                f"{SIGNAL_API}/v1/register/{number}",
                json={"use_voice": True},
            )
            if r2.status_code >= 400:
                r2 = await client.post(
                    f"{SIGNAL_API}/v1/register/{number}",
                    json={"use_voice": True, "captcha": captcha},
                )
            if r2.status_code >= 400:
                return {"ok": False, "step": "voice", "status": r2.status_code, "error": r2.text}

            return {"ok": True, "number": number, "method": "voice",
                    "next": "answer the voice call from Signal, then enter the 6-digit code below"}
    except Exception as e:
        raise HTTPException(503, f"signal-cli error: {e}")


@router.post("/signal/verify")
async def signal_verify(request: Request) -> dict:
    body = await request.json()
    number = (body.get("number") or "").strip()
    code = (body.get("code") or "").strip().replace("-", "")
    if not number or not code:
        raise HTTPException(400, "number and code required")
    if not number.startswith("+"):
        number = "+" + "".join(c for c in number if c.isdigit())
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{SIGNAL_API}/v1/register/{number}/verify/{code}")
        if r.status_code >= 400:
            return {"ok": False, "status": r.status_code, "error": r.text}
        return {"ok": True, "number": number, "hint": "set SIGNAL_BENSON_NUMBER=" + number + " and restart benson"}
    except Exception as e:
        raise HTTPException(503, f"signal-cli unreachable: {e}")


# ─── Status ──────────────────────────────────────────────────────────────
@router.get("/signal/status")
async def signal_status() -> dict:
    out: dict = {
        "benson_number": _benson_number() or None,
        "registered": bool(_benson_number()),
        "allowlist": sorted(_allowed()),
        "allowed_uuids": sorted(_allowed_uuids()),
        "allowed_groups": sorted(_allowed_groups()),
        "speaker_map": _phone_to_speaker(),
        "name_fallback": _known_first_names(),
    }
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{SIGNAL_API}/v1/about")
        out["api"] = r.json() if r.status_code == 200 else {"error": r.text[:300]}
    except Exception as e:
        out["api"] = {"error": f"unreachable: {e}"}
    return out


# ─── Outbound ────────────────────────────────────────────────────────────
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


async def _upload_attachment(client: httpx.AsyncClient, file_path: str) -> str:
    """Upload a local file to the Signal REST API and return its attachment ID.

    Raises FileNotFoundError if the file doesn't exist on disk, or
    RuntimeError if the API returns an error status.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"file not found: {file_path!r}")
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    r = await client.post(
        f"{SIGNAL_API}/v1/attachments",
        files={"file": (p.name, p.read_bytes(), mime)},
    )
    if r.status_code >= 400:
        raise RuntimeError(
            f"attachment upload failed for {p.name!r}: HTTP {r.status_code} — {r.text[:200]}"
        )
    # signal-cli-rest-api returns the attachment ID as a plain string
    return r.text.strip().strip('"')


async def send_signal_message(
    to: str,
    text: str,
    file_paths: list[str] | None = None,
) -> dict:
    """Send a Signal message. `to` is E.164 (+1…), an ACI UUID, or a
    group internal_id (signal-cli's raw base64).

    signal-cli-rest-api accepts:
      - phone numbers as-is (E.164)
      - UUIDs as-is
      - groups as `group.{base64(internal_id)}`

    file_paths is an optional list of local file paths to attach as
    images or documents.  Each file is uploaded to the Signal REST API
    before sending; if any upload fails the entire call is aborted and
    a clear error is returned indicating which file failed and why.
    """
    num = _benson_number()
    if not num:
        return {"ok": False, "error": "SIGNAL_BENSON_NUMBER not set"}
    if to.startswith("+") or to.startswith("group.") or _UUID_RE.match(to):
        recipient = to
    else:
        # Treat as a raw group internal_id; wrap into the API form.
        import base64
        encoded = base64.b64encode(to.encode("utf-8")).decode("ascii")
        recipient = f"group.{encoded}"

    attachment_ids: list[str] = []
    if file_paths:
        async with httpx.AsyncClient(timeout=30) as upload_client:
            for fp in file_paths:
                try:
                    att_id = await _upload_attachment(upload_client, fp)
                    attachment_ids.append(att_id)
                except FileNotFoundError as exc:
                    return {"ok": False, "error": str(exc)}
                except RuntimeError as exc:
                    return {"ok": False, "error": str(exc)}
                except Exception as exc:
                    return {"ok": False, "error": f"unexpected error uploading {fp!r}: {exc}"}

    payload: dict = {
        "number": num,
        "message": text,
        "recipients": [recipient],
    }
    if attachment_ids:
        payload["attachments"] = attachment_ids

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(f"{SIGNAL_API}/v2/send", json=payload)
        if r.status_code >= 400:
            logger.warning(f"signal /v2/send {r.status_code}: payload={payload} body={r.text[:500]}")
        return {"ok": r.status_code < 400, "status": r.status_code, "body": r.text[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/signal/send")
async def signal_send(request: Request) -> dict:
    body = await request.json()
    to = (body.get("to") or "").strip()
    text = body.get("text") or body.get("message") or ""
    if not to or not text:
        raise HTTPException(400, "to and text required")
    if to.startswith("+") is False and to.replace("+", "").isdigit():
        to = "+" + to.lstrip("+")
    return await send_signal_message(to, text)


# ─── Inbound polling ─────────────────────────────────────────────────────
ATTACHMENT_DIR = "/tmp/benson-attachments"


async def _download_attachment(att_id: str, filename: str | None, content_type: str) -> str | None:
    """Pull a Signal attachment from the container and save it locally.
    Returns the local path or None on failure."""
    import os
    os.makedirs(ATTACHMENT_DIR, exist_ok=True)
    ext = ""
    if content_type:
        ext = "." + content_type.split("/")[-1].split(";")[0].strip()
        if ext == ".jpeg":
            ext = ".jpg"
    safe = "".join(c for c in att_id if c.isalnum() or c in "-_")[:40]
    out = f"{ATTACHMENT_DIR}/{safe}{ext}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(f"{SIGNAL_API}/v1/attachments/{att_id}")
        if r.status_code != 200:
            logger.warning(f"signal attachment fetch failed {r.status_code} for {att_id}")
            return None
        with open(out, "wb") as f:
            f.write(r.content)
        return out
    except Exception as e:
        logger.warning(f"signal attachment download error: {e}")
        return None


async def _process_envelope(env: dict) -> None:
    import json as _json
    logger.info(f"signal envelope: {_json.dumps(env)[:800]}")
    msg = env.get("envelope", {})
    source = msg.get("source") or msg.get("sourceNumber") or ""
    source_name = msg.get("sourceName") or ""
    data_msg = msg.get("dataMessage") or {}
    text = (data_msg.get("message") or "").strip()
    group_info = data_msg.get("groupInfo") or {}
    group_id = group_info.get("groupId") or ""

    # Download any attachments and append their local paths to the message.
    attachments = data_msg.get("attachments") or []
    attachment_lines: list[str] = []
    for att in attachments:
        att_id = att.get("id") or ""
        if not att_id:
            continue
        ctype = att.get("contentType") or "application/octet-stream"
        path = await _download_attachment(att_id, att.get("filename"), ctype)
        if path:
            attachment_lines.append(f"[attachment: {ctype} at {path}]")
    if attachment_lines:
        text = (text + "\n\n" + "\n".join(attachment_lines)).strip() if text else "\n".join(attachment_lines)

    if not text:
        return

    # Allowlist
    allowed_nums = _allowed()
    allowed_grps = _allowed_groups()
    allowed_uuids = _allowed_uuids()
    name_match = _resolve_speaker_by_name(source_name)
    if group_id:
        if allowed_grps and group_id not in allowed_grps:
            logger.info(f"signal ignore: group {group_id[:12]}… not in allowlist")
            return
    else:
        # Signal delivers UUIDs in `source` when contacts have phone-number
        # privacy enabled. Accept by phone, UUID, or known sender name.
        in_allowlist = (
            (allowed_nums and source in allowed_nums)
            or (allowed_uuids and source in allowed_uuids)
            or bool(name_match)
        )
        any_gate = bool(allowed_nums or allowed_uuids or _known_first_names())
        if any_gate and not in_allowlist:
            logger.info(
                f"signal ignore: {source} ({source_name!r}) not in allowlist"
            )
            return

    # Speaker
    speakers = _phone_to_speaker()
    speaker = speakers.get(source) or name_match or source_name or "Unknown"

    # Route
    from main import handle_conversation

    class _FakeReq:
        async def json(self_inner):
            return {
                "text": text,
                "speaker": speaker,
                "room": "signal_group" if group_id else "signal",
            }

    try:
        result = await handle_conversation(_FakeReq())  # type: ignore[arg-type]
    except Exception as e:
        logger.exception("signal → /conversation failed")
        await send_signal_message(group_id or source, f"(error: {type(e).__name__})")
        return

    response_text = result.get("response", "(no response)")
    reply_to = group_id or source
    await send_signal_message(reply_to, response_text)


async def _poll_loop() -> None:
    """Background task: WebSocket-receive incoming Signal messages.

    Container runs in json-rpc mode; /v1/receive/{number} is a WebSocket
    that pushes envelopes as they arrive. Avoids serializing with /send
    and /v1/about calls (which would block in normal mode).
    """
    import json
    import websockets

    while True:
        num = _benson_number()
        if not num:
            await asyncio.sleep(10)
            continue
        ws_url = SIGNAL_API.replace("http://", "ws://").replace("https://", "wss://")
        url = f"{ws_url}/v1/receive/{num}"
        try:
            async with websockets.connect(url, ping_interval=30, ping_timeout=10) as ws:
                logger.info(f"signal: ws connected to {url}")
                async for raw in ws:
                    try:
                        env = json.loads(raw)
                    except Exception:
                        continue
                    with suppress(Exception):
                        await _process_envelope(env)
        except Exception as e:
            logger.warning(f"signal ws error: {e} — reconnecting in 5s")
            await asyncio.sleep(5)


def start_poller() -> None:
    """Called from main.py on app startup."""
    asyncio.create_task(_poll_loop())
