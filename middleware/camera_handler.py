"""iPad-as-eyeball camera handler.

Architecture: a tablet/phone in a fixed location (kitchen, etc.) opens
/camera/<name>/page in Safari. The page captures the camera locally
(no upload until requested) and holds a WebSocket connection to this
server. When Benson needs to "look", we send a snapshot trigger over
the WS, the page draws the current video frame to a canvas, encodes
JPEG, and POSTs it to /camera/<name>/frame. The agent then runs
analyze_image on the saved file.

State is in-memory (CONNECTED) — restart drops registrations and the
iPad page reconnects automatically.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

logger = logging.getLogger("benson.camera")
router = APIRouter()

FRAMES_DIR = Path("/tmp/benson-cameras")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# cam_name -> {"ws": WebSocket, "connected_at": ts, "user_agent": str}
CONNECTED: dict[str, dict[str, Any]] = {}

# cam_name -> deque of recent events (max 50)
EVENTS: dict[str, deque] = {}


def _log(cam: str, kind: str, **extra) -> None:
    if cam not in EVENTS:
        EVENTS[cam] = deque(maxlen=50)
    EVENTS[cam].append({"ts": time.time(), "kind": kind, **extra})
    logger.info(f"camera '{cam}' {kind} {extra if extra else ''}")


def latest_frame_path(cam: str) -> Path:
    return FRAMES_DIR / f"{cam}.jpg"


@router.websocket("/camera/{cam}/ws")
async def camera_ws(websocket: WebSocket, cam: str):
    await websocket.accept()
    ua = websocket.headers.get("user-agent", "")
    CONNECTED[cam] = {"ws": websocket, "connected_at": time.time(), "user_agent": ua}
    _log(cam, "connected", ua=ua[:80])
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        _log(cam, "disconnected")
    except Exception as e:
        _log(cam, "ws_error", error=str(e)[:200])
    finally:
        if CONNECTED.get(cam, {}).get("ws") is websocket:
            CONNECTED.pop(cam, None)


@router.post("/camera/{cam}/frame")
async def upload_frame(cam: str, request: Request) -> dict:
    """The iPad page POSTs a JPEG here in response to a snapshot trigger."""
    body = await request.body()
    if not body or len(body) < 1000:
        _log(cam, "frame_rejected", bytes=len(body), reason="too small")
        raise HTTPException(400, f"frame too small ({len(body)} bytes)")
    path = latest_frame_path(cam)
    path.write_bytes(body)
    _log(cam, "frame_uploaded", bytes=len(body))
    return {"ok": True, "cam": cam, "bytes": len(body)}


async def trigger_snapshot(cam: str, timeout: float = 5.0, source: str = "agent") -> Path | None:
    """Ask the connected iPad for a fresh frame; wait up to `timeout`
    seconds for the resulting upload. Returns the file path or None.
    """
    info = CONNECTED.get(cam)
    if not info:
        _log(cam, "snapshot_no_camera", source=source)
        return None
    ws: WebSocket = info["ws"]
    path = latest_frame_path(cam)
    before = path.stat().st_mtime if path.exists() else 0.0
    _log(cam, "snapshot_requested", source=source)
    try:
        await ws.send_text("snapshot")
    except Exception as e:
        _log(cam, "snapshot_send_failed", error=str(e)[:200])
        CONNECTED.pop(cam, None)
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(0.1)
        if path.exists() and path.stat().st_mtime > before:
            elapsed_ms = round((time.time() - (deadline - timeout)) * 1000)
            _log(cam, "snapshot_delivered", elapsed_ms=elapsed_ms)
            return path
    _log(cam, "snapshot_timeout", waited_s=timeout)
    return None


@router.post("/camera/{cam}/snapshot")
async def manual_snapshot(cam: str) -> dict:
    p = await trigger_snapshot(cam, source="manual")
    if not p:
        return {"ok": False, "error": f"camera '{cam}' not connected or timed out"}
    return {"ok": True, "cam": cam, "path": str(p), "bytes": p.stat().st_size}


@router.get("/camera/{cam}/latest.jpg")
async def latest_jpeg(cam: str):
    p = latest_frame_path(cam)
    if not p.exists():
        raise HTTPException(404, f"no frame yet for '{cam}'")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/camera/status")
async def camera_status() -> dict:
    out: dict = {}
    # Include every camera that's connected OR has a cached frame OR has events
    cams = set(CONNECTED.keys()) | {p.stem for p in FRAMES_DIR.glob("*.jpg")} | set(EVENTS.keys())
    for cam in cams:
        info = CONNECTED.get(cam) or {}
        path = latest_frame_path(cam)
        events = list(EVENTS.get(cam, []))
        # Aggregate stats
        snapshots = sum(1 for e in events if e["kind"] == "snapshot_delivered")
        timeouts = sum(1 for e in events if e["kind"] == "snapshot_timeout")
        out[cam] = {
            "connected": cam in CONNECTED,
            "connected_at": info.get("connected_at"),
            "user_agent": (info.get("user_agent") or "")[:80] or None,
            "last_frame_age_s": (time.time() - path.stat().st_mtime) if path.exists() else None,
            "last_frame_bytes": path.stat().st_size if path.exists() else 0,
            "stats": {
                "snapshots_delivered": snapshots,
                "snapshots_timed_out": timeouts,
                "events_recorded": len(events),
            },
            "recent_events": events[-20:],
        }
    return {"cameras": out, "now": time.time()}


# ─── iPad-side page ──────────────────────────────────────────────────────
_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Benson Eye">
<link rel="manifest" href="/camera/{cam}/manifest.json">
<title>Benson Eye — {cam}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; height: 100vh; width: 100vw;
    background: #0a0a0a; color: #ddd; overflow: hidden;
    font-family: -apple-system, sans-serif;
  }}
  video {{
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    object-fit: cover; opacity: 0.55;
  }}
  .panel {{
    position: fixed; bottom: env(safe-area-inset-bottom, 16px); left: 0; right: 0;
    padding: 16px 20px; text-align: center;
    background: rgba(0,0,0,0.55); backdrop-filter: blur(12px);
  }}
  h1 {{ font-size: 1.05rem; margin: 0 0 4px; opacity: 0.85; letter-spacing: 0.04em; }}
  .status {{ font-size: 0.85rem; opacity: 0.65; }}
  .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
  .dot.ok {{ background:#7fdb9b; }} .dot.bad {{ background:#e07070; }} .dot.snap {{ background:#f7c873; }}
  .flash {{
    position: fixed; inset: 0; background: white; opacity: 0;
    pointer-events: none; transition: opacity 80ms;
  }}
  .flash.on {{ opacity: 0.4; }}
</style>
</head>
<body>
  <video id="v" autoplay playsinline muted></video>
  <div class="flash" id="flash"></div>
  <div class="panel">
    <h1>Benson Eye — {cam_title}</h1>
    <div class="status"><span class="dot bad" id="dot"></span><span id="msg">starting…</span></div>
  </div>
<script>
const CAM = "{cam}";
const v = document.getElementById('v');
const dot = document.getElementById('dot');
const msg = document.getElementById('msg');
const flash = document.getElementById('flash');
const canvas = document.createElement('canvas');

async function lockWake() {{
  try {{ if ('wakeLock' in navigator) await navigator.wakeLock.request('screen'); }} catch(e){{}}
}}
lockWake();
document.addEventListener('visibilitychange', () => {{ if (!document.hidden) lockWake(); }});

async function startCamera() {{
  try {{
    const facing = new URLSearchParams(location.search).get('facing') || 'user';
    const stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode: facing, width: {{ ideal: 1280 }}, height: {{ ideal: 720 }} }},
      audio: false,
    }});
    v.srcObject = stream;
    await v.play();
    dot.className = 'dot ok'; msg.textContent = 'camera ready';
    return true;
  }} catch (e) {{
    dot.className = 'dot bad'; msg.textContent = 'camera blocked: ' + e.message;
    return false;
  }}
}}

function snapshot() {{
  if (!v.videoWidth) return null;
  canvas.width = v.videoWidth;
  canvas.height = v.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(v, 0, 0);
  // brief flash so anyone in the kitchen sees the snap
  flash.classList.add('on');
  setTimeout(() => flash.classList.remove('on'), 150);
  return new Promise(res => canvas.toBlob(b => res(b), 'image/jpeg', 0.85));
}}

async function uploadSnapshot() {{
  const blob = await snapshot();
  if (!blob) {{ dot.className = 'dot bad'; msg.textContent = 'no video frame yet'; return; }}
  dot.className = 'dot snap'; msg.textContent = 'uploading…';
  try {{
    const r = await fetch('/camera/' + CAM + '/frame', {{ method: 'POST', headers: {{ 'Content-Type': 'image/jpeg' }}, body: blob }});
    const j = await r.json();
    dot.className = 'dot ok';
    msg.textContent = 'snapshot uploaded · ' + (j.bytes || 0) + ' B · ' + new Date().toLocaleTimeString();
  }} catch (e) {{
    dot.className = 'dot bad'; msg.textContent = 'upload failed: ' + e.message;
  }}
}}

let ws, pingTimer;
function connectWS() {{
  if (pingTimer) {{ clearInterval(pingTimer); pingTimer = null; }}
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/camera/' + CAM + '/ws');
  ws.onopen = () => {{
    dot.className = 'dot ok'; msg.textContent = 'connected to Benson';
    // Aggressive keepalive — 5s, well under any NAT/proxy idle reap
    try {{ ws.send('ping'); }} catch(e){{}}
    pingTimer = setInterval(() => {{
      if (ws && ws.readyState === 1) {{ try {{ ws.send('ping'); }} catch(e){{}} }}
    }}, 5000);
  }};
  ws.onmessage = (e) => {{
    if (e.data === 'snapshot') uploadSnapshot();
  }};
  ws.onclose = () => {{
    if (pingTimer) {{ clearInterval(pingTimer); pingTimer = null; }}
    dot.className = 'dot bad'; msg.textContent = 'reconnecting…';
    setTimeout(connectWS, 2000);
  }};
  ws.onerror = () => {{}};
}}

(async () => {{
  if (await startCamera()) connectWS();
}})();
</script>
</body>
</html>
"""


@router.get("/camera/{cam}/page", response_class=HTMLResponse)
async def camera_page(cam: str):
    html = _PAGE_HTML.format(cam=cam, cam_title=cam.replace("_", " ").title())
    # Permissions-Policy declares camera intent on this origin — helps
    # browsers (and iOS Safari) treat the permission grant as durable.
    return HTMLResponse(
        html,
        headers={
            "Permissions-Policy": "camera=(self), microphone=()",
            "Feature-Policy": "camera 'self'",
        },
    )


@router.get("/camera/{cam}/manifest.json")
async def camera_manifest(cam: str) -> dict:
    """Web app manifest — when added to Home Screen, iOS treats this as a
    proper PWA with sticky permissions (separate from Safari proper)."""
    title = cam.replace("_", " ").title()
    return {
        "name": f"Benson Eye — {title}",
        "short_name": "Benson Eye",
        "start_url": f"/camera/{cam}/page",
        "scope": f"/camera/{cam}/",
        "display": "standalone",
        "background_color": "#0a0a0a",
        "theme_color": "#0a0a0a",
        "orientation": "any",
        "permissions": ["camera"],
    }
