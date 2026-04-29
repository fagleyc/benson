"""WhatsApp inbound handler + outbound helper.

Listens on /whatsapp/inbound for posts from the Baileys bridge
(loopback only, port 8200). Routes recognized senders' messages through
the chat path (/conversation) and sends the response back via the
bridge's /send.
"""
from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("benson.whatsapp")
router = APIRouter()

WA_BRIDGE = os.environ.get("WA_BRIDGE_URL", "http://localhost:8200")


def _allowed_jids() -> set[str]:
    raw = os.environ.get("WHATSAPP_ALLOWED_NUMBERS", "")
    out: set[str] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # accept "+15551234" or "15551234@s.whatsapp.net" or "...@g.us"
        digits = "".join(c for c in tok if c.isdigit() or c == "@")
        if "@" in tok:
            out.add(tok)
        else:
            out.add(f"{digits}@s.whatsapp.net")
    return out


def _phone_to_speaker() -> dict[str, str]:
    """Optional mapping: phone digits → speaker name. Format in env:
    WHATSAPP_SPEAKER_MAP=15551234:Casey,15555678:Lindsey
    """
    raw = os.environ.get("WHATSAPP_SPEAKER_MAP", "")
    m: dict[str, str] = {}
    for tok in raw.split(","):
        if ":" in tok:
            num, name = tok.split(":", 1)
            digits = "".join(c for c in num if c.isdigit())
            m[digits] = name.strip()
    return m


@router.post("/whatsapp/inbound")
async def whatsapp_inbound(request: Request) -> dict:
    """Called by the Baileys bridge for every incoming message."""
    data = await request.json()
    from_jid = data.get("from", "")
    sender_jid = data.get("sender_jid", from_jid)
    sender_name = data.get("sender_name", "")
    text = (data.get("text") or "").strip()
    is_group = bool(data.get("is_group", False))

    if not text:
        return {"ignored": True, "reason": "no text body"}

    # Allowlist: per-sender (DMs) OR per-group (group chats)
    allowed = _allowed_jids()
    if allowed:
        check = from_jid if is_group else sender_jid
        if check not in allowed and sender_jid not in allowed:
            logger.info(f"whatsapp ignore: {check} not in allowlist")
            return {"ignored": True, "reason": "not in allowlist"}

    # Resolve speaker name
    speakers = _phone_to_speaker()
    digits = sender_jid.split("@")[0]
    speaker = speakers.get(digits) or sender_name or "Unknown"

    # Route through main /conversation
    from main import handle_conversation

    class _FakeReq:
        async def json(self_inner):
            return {
                "text": text,
                "speaker": speaker,
                "room": "whatsapp_group" if is_group else "whatsapp",
            }

    try:
        result = await handle_conversation(_FakeReq())  # type: ignore[arg-type]
    except Exception as e:
        logger.exception("whatsapp → /conversation failed")
        return {"error": f"{type(e).__name__}: {e}"}

    response_text = result.get("response", "(no response)")

    # Reply
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{WA_BRIDGE}/send",
                json={"to": from_jid, "text": response_text},
            )
            ok = r.status_code == 200
    except Exception as e:
        logger.warning(f"whatsapp send failed: {e}")
        ok = False

    return {
        "ok": ok,
        "tier": result.get("tier"),
        "speaker": speaker,
        "is_group": is_group,
    }


@router.get("/whatsapp/status")
async def whatsapp_status() -> dict:
    """Pass-through to the bridge's /status."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{WA_BRIDGE}/status")
        return r.json()
    except Exception as e:
        return {"state": "bridge_unreachable", "error": str(e)}


@router.get("/whatsapp/qr")
async def whatsapp_qr():
    """Pass-through to the bridge's QR PNG."""
    from fastapi.responses import Response
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{WA_BRIDGE}/qr")
        if r.status_code != 200:
            raise HTTPException(404, "no QR pending")
        return Response(content=r.content, media_type="image/png")
    except Exception as e:
        raise HTTPException(503, f"bridge unreachable: {e}")


@router.post("/whatsapp/send")
async def whatsapp_send(request: Request) -> dict:
    """Send a manual WhatsApp message — for the admin page test box."""
    body = await request.json()
    to = body.get("to") or ""
    text = body.get("text") or body.get("message") or ""
    if not to or not text:
        raise HTTPException(400, "to and text required")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(f"{WA_BRIDGE}/send", json={"to": to, "text": text})
        return r.json()
    except Exception as e:
        raise HTTPException(503, f"bridge unreachable: {e}")
