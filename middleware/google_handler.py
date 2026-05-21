"""Google OAuth + Calendar/Gmail clients for Benson.

Per-user tokens stored in `oauth_tokens` table. OAuth uses the
"manual paste" pattern since household members aren't on the Benson
server: user clicks the auth URL, grants consent, gets redirected to a
http://localhost URL that fails to load, copies the URL from the
address bar, pastes it back into Benson's admin page, and we extract
the code from the URL.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
from fastapi import APIRouter, HTTPException, Request
from psycopg2.extras import RealDictCursor

from config import PG_DSN

logger = logging.getLogger("benson.google")
router = APIRouter()

CLIENT_SECRETS_PATH = os.environ.get(
    "GOOGLE_CLIENT_SECRETS", "/etc/benson/google_client_secret.json"
)

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",  # read + write events
    "https://www.googleapis.com/auth/calendar.readonly",  # read calendar list
    "https://www.googleapis.com/auth/gmail.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

DEFAULT_TZ = "America/Denver"

# Manual-paste flow: redirect to localhost so any device can be used —
# Google won't deliver to localhost from a remote browser, but the user
# can copy the URL out of their address bar.
REDIRECT_URI = "http://localhost"

# PKCE code_verifier per pending OAuth flow, keyed by user_name (state).
# Held in-memory; user typically completes auth in <5 min.
_PENDING_VERIFIERS: dict[str, str] = {}


# ─── DB helpers ──────────────────────────────────────────────────────────
def _conn():
    return psycopg2.connect(**PG_DSN)


def _save_tokens(
    user_name: str,
    *,
    access_token: str,
    refresh_token: str | None,
    expiry: datetime | None,
    scopes: str,
    email: str | None,
) -> None:
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO oauth_tokens
                (user_name, provider, email, access_token, refresh_token,
                 token_expiry, scopes, updated_at)
            VALUES (%s, 'google', %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_name, provider) DO UPDATE SET
                email = EXCLUDED.email,
                access_token = EXCLUDED.access_token,
                refresh_token = COALESCE(EXCLUDED.refresh_token, oauth_tokens.refresh_token),
                token_expiry = EXCLUDED.token_expiry,
                scopes = EXCLUDED.scopes,
                updated_at = NOW()
            """,
            (user_name, email, access_token, refresh_token, expiry, scopes),
        )


def _load_token_row(user_name: str) -> dict | None:
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM oauth_tokens WHERE user_name = %s AND provider = 'google'",
            (user_name,),
        )
        r = cur.fetchone()
        return dict(r) if r else None


def list_linked_users() -> list[dict]:
    with _conn() as c, c.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT t.user_name, t.email, t.scopes, t.token_expiry, t.updated_at,
                   t.default_calendar_id, t.default_calendar_name,
                   (SELECT COUNT(*) FROM calendar_events e WHERE e.user_name = t.user_name) AS event_count,
                   (SELECT MAX(last_synced) FROM calendar_events e WHERE e.user_name = t.user_name) AS last_sync
            FROM oauth_tokens t
            WHERE t.provider = 'google'
            ORDER BY t.user_name
            """
        )
        return [dict(r) for r in cur.fetchall()]


def _get_default_calendar(user_name: str) -> str | None:
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT default_calendar_id FROM oauth_tokens WHERE user_name = %s AND provider = 'google'",
            (user_name,),
        )
        r = cur.fetchone()
        return (r[0] if r else None) or None


# ─── OAuth flow ──────────────────────────────────────────────────────────
def _flow():
    from google_auth_oauthlib.flow import Flow
    return Flow.from_client_secrets_file(
        CLIENT_SECRETS_PATH,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )


@router.post("/google/oauth/start")
async def oauth_start(request: Request) -> dict:
    """Build an OAuth URL for a household user to grant consent."""
    body = await request.json()
    user_name = (body.get("user_name") or "").strip()
    if not user_name:
        raise HTTPException(400, "user_name required")
    if not os.path.exists(CLIENT_SECRETS_PATH):
        raise HTTPException(500, f"client secrets not found at {CLIENT_SECRETS_PATH}")

    flow = _flow()
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",  # force refresh_token return
        state=user_name,
    )
    # Stash the PKCE verifier so /oauth/finish can re-use it.
    if getattr(flow, "code_verifier", None):
        _PENDING_VERIFIERS[user_name] = flow.code_verifier
    return {"ok": True, "user_name": user_name, "auth_url": auth_url}


@router.post("/google/oauth/finish")
async def oauth_finish(request: Request) -> dict:
    """Take a pasted callback URL (or raw code) and exchange for tokens."""
    body = await request.json()
    user_name = (body.get("user_name") or "").strip()
    pasted = (body.get("callback_url") or body.get("code") or "").strip()
    if not user_name or not pasted:
        raise HTTPException(400, "user_name and callback_url (or code) required")

    # Accept either the full callback URL or just the code.
    if pasted.startswith("http"):
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(pasted).query)
        code = (q.get("code") or [""])[0]
    else:
        code = pasted

    if not code:
        raise HTTPException(400, "could not find ?code= in pasted value")

    flow = _flow()
    verifier = _PENDING_VERIFIERS.pop(user_name, None)
    if verifier:
        flow.code_verifier = verifier
    try:
        flow.fetch_token(code=code)
    except Exception as e:
        raise HTTPException(400, f"token exchange failed: {e}")

    creds = flow.credentials

    # Identify the actual Google account email
    email: str | None = None
    try:
        from googleapiclient.discovery import build
        oauth2 = build("oauth2", "v2", credentials=creds, cache_discovery=False)
        info = oauth2.userinfo().get().execute()
        email = info.get("email")
    except Exception as e:
        logger.warning(f"userinfo lookup failed: {e}")

    _save_tokens(
        user_name,
        access_token=creds.token,
        refresh_token=creds.refresh_token,
        expiry=creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else None,
        scopes=" ".join(creds.scopes or []),
        email=email,
    )
    return {"ok": True, "user_name": user_name, "email": email}


@router.post("/google/disconnect")
async def disconnect(request: Request) -> dict:
    body = await request.json()
    user_name = (body.get("user_name") or "").strip()
    if not user_name:
        raise HTTPException(400, "user_name required")
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "DELETE FROM oauth_tokens WHERE user_name = %s AND provider = 'google'",
            (user_name,),
        )
        cur.execute(
            "DELETE FROM calendar_events WHERE user_name = %s",
            (user_name,),
        )
    return {"ok": True}


@router.get("/google/status")
async def status() -> dict:
    return {
        "ok": True,
        "client_secrets_present": os.path.exists(CLIENT_SECRETS_PATH),
        "users": list_linked_users(),
    }


# ─── Credential helpers ──────────────────────────────────────────────────
def _credentials_from_row(row: dict):
    from google.oauth2.credentials import Credentials
    with open(CLIENT_SECRETS_PATH) as f:
        secrets = json.load(f)["installed"]
    creds = Credentials(
        token=row["access_token"],
        refresh_token=row.get("refresh_token"),
        token_uri=secrets["token_uri"],
        client_id=secrets["client_id"],
        client_secret=secrets["client_secret"],
        scopes=(row.get("scopes") or "").split(),
    )
    if row.get("token_expiry"):
        creds.expiry = row["token_expiry"].replace(tzinfo=None)
    return creds


def get_credentials(user_name: str):
    """Load creds, refreshing if expired. Returns None on any failure;
    callers wanting the failure REASON should use get_credentials_with_status."""
    creds, _status = get_credentials_with_status(user_name)
    return creds


def get_credentials_with_status(user_name: str) -> tuple:
    """Load creds + a status string. Status is one of:
        'ok'           — creds usable
        'no_row'       — user has never linked their Google account
        'refresh_failed:<reason>' — token row exists but Google rejected
                          the refresh (most often 'invalid_grant: Token
                          has been expired or revoked' — user needs to
                          re-link at /admin/google).
    """
    row = _load_token_row(user_name)
    if not row:
        return None, "no_row"
    creds = _credentials_from_row(row)
    if not creds.valid:
        try:
            from google.auth.transport.requests import Request as _G
            creds.refresh(_G())
            _save_tokens(
                user_name,
                access_token=creds.token,
                refresh_token=creds.refresh_token,
                expiry=creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else None,
                scopes=" ".join(creds.scopes or []),
                email=row.get("email"),
            )
        except Exception as e:
            reason = str(e).splitlines()[0][:200]
            logger.warning(f"refresh failed for {user_name}: {e}")
            return None, f"refresh_failed:{reason}"
    return creds, "ok"


def calendar_service(user_name: str):
    from googleapiclient.discovery import build
    creds = get_credentials(user_name)
    if not creds:
        return None
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def gmail_service(user_name: str):
    from googleapiclient.discovery import build
    creds = get_credentials(user_name)
    if not creds:
        return None
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ─── Calendar sync ───────────────────────────────────────────────────────
_PERSON_KEYWORDS = ["casey", "lindsey", "cole", "zander", "sherry"]


def _derive_person(calendar_summary: str, account_user_name: str) -> str:
    s = (calendar_summary or "").lower()
    for name in _PERSON_KEYWORDS:
        if name in s:
            return name.capitalize()
    if any(k in s for k in ("family", "c&l", "household", "shared")):
        return "Family"
    if "holiday" in s:
        return "Holiday"
    return account_user_name


def sync_calendar_for(user_name: str, days_ahead: int = 14) -> dict:
    svc = calendar_service(user_name)
    if not svc:
        return {"ok": False, "error": "no credentials"}
    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=days_ahead)).isoformat()

    # Iterate every calendar this account can see (owned, writer, reader).
    cals = svc.calendarList().list().execute().get("items", [])
    from psycopg2.extras import execute_values
    rows: list = []
    seen_event_ids: set[tuple[str, str]] = set()  # (user_name, event_id) to dedupe across calendars
    cal_summaries: list[str] = []
    for cal in cals:
        cid = cal.get("id")
        csum = cal.get("summary") or cid
        cal_summaries.append(csum)
        person = _derive_person(csum, user_name)
        page_token = None
        while True:
            try:
                resp = svc.events().list(
                    calendarId=cid,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=250,
                    pageToken=page_token,
                ).execute()
            except Exception as e:
                logger.warning(f"sync error on calendar {csum}: {e}")
                break
            for e in resp.get("items", []):
                eid = e.get("id")
                if not eid or (user_name, eid) in seen_event_ids:
                    continue
                seen_event_ids.add((user_name, eid))
                start = e.get("start", {})
                end = e.get("end", {})
                all_day = "date" in start and "dateTime" not in start
                rows.append((
                    user_name,
                    eid,
                    cid,
                    csum,
                    person,
                    (e.get("summary") or "")[:500],
                    (e.get("description") or "")[:2000],
                    (e.get("location") or "")[:500],
                    start.get("dateTime") or start.get("date"),
                    end.get("dateTime") or end.get("date"),
                    all_day,
                    e.get("status"),
                ))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    if rows:
        with _conn() as c, c.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO calendar_events
                    (user_name, google_event_id, calendar_id, calendar_summary,
                     person, title, description, location, starts_at, ends_at,
                     all_day, status)
                VALUES %s
                ON CONFLICT (user_name, google_event_id) DO UPDATE SET
                    calendar_id = EXCLUDED.calendar_id,
                    calendar_summary = EXCLUDED.calendar_summary,
                    person = EXCLUDED.person,
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    location = EXCLUDED.location,
                    starts_at = EXCLUDED.starts_at,
                    ends_at = EXCLUDED.ends_at,
                    all_day = EXCLUDED.all_day,
                    status = EXCLUDED.status,
                    last_synced = NOW()
                """,
                rows,
            )

    # Drop events that fell out of the window OR that we own but no longer see (deleted/moved out).
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            "DELETE FROM calendar_events WHERE user_name = %s AND starts_at < %s",
            (user_name, now - timedelta(days=1)),
        )
        # Stale-prune: if we have an event ID that wasn't returned this sync, drop it.
        cur.execute(
            "DELETE FROM calendar_events WHERE user_name = %s AND last_synced < NOW() - interval '5 minutes'",
            (user_name,),
        )

    return {
        "ok": True,
        "synced": len(rows),
        "calendars_seen": len(cals),
        "user_name": user_name,
    }


@router.post("/google/sync/{user_name}")
async def trigger_sync(user_name: str) -> dict:
    import asyncio
    return await asyncio.to_thread(sync_calendar_for, user_name)


@router.get("/google/calendars/{user_name}")
async def list_user_calendars(user_name: str) -> dict:
    import asyncio
    def _run() -> dict:
        svc = calendar_service(user_name)
        if not svc:
            return {"ok": False, "error": f"{user_name} not linked"}
        cals = svc.calendarList().list().execute().get("items", [])
        out = [
            {
                "id": c.get("id"),
                "summary": c.get("summary"),
                "primary": bool(c.get("primary")),
                "access_role": c.get("accessRole"),
                "color": c.get("backgroundColor"),
            } for c in cals
        ]
        # Mark which is the user's stored default
        with _conn() as cn, cn.cursor() as cur:
            cur.execute(
                "SELECT default_calendar_id FROM oauth_tokens WHERE user_name = %s AND provider = 'google'",
                (user_name,),
            )
            r = cur.fetchone()
            default_id = r[0] if r else None
        for c in out:
            c["is_default"] = (c["id"] == default_id) if default_id else c["primary"]
        return {"ok": True, "user_name": user_name, "default_calendar_id": default_id, "calendars": out}
    return await asyncio.to_thread(_run)


@router.post("/google/default-calendar")
async def set_default_calendar(request: Request) -> dict:
    body = await request.json()
    user_name = (body.get("user_name") or "").strip()
    cal_id = body.get("calendar_id")
    cal_name = body.get("calendar_name")
    if not user_name or not cal_id:
        raise HTTPException(400, "user_name and calendar_id required")
    with _conn() as c, c.cursor() as cur:
        cur.execute(
            """
            UPDATE oauth_tokens
            SET default_calendar_id = %s, default_calendar_name = %s, updated_at = NOW()
            WHERE user_name = %s AND provider = 'google'
            """,
            (cal_id, cal_name, user_name),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, f"{user_name} not linked")
    return {"ok": True, "user_name": user_name, "default_calendar_id": cal_id, "default_calendar_name": cal_name}


# ─── Calendar writes ─────────────────────────────────────────────────────
def _resolve_calendar_id(svc, hint: str | None, user_name: str | None = None) -> tuple[str, str]:
    """Map a hint ('Cole', 'Family', 'primary', or a real id) to a calendar
    id. Returns (id, summary). When hint is None/empty, falls back to the
    user's stored default_calendar_id, then to primary."""
    cals = svc.calendarList().list().execute().get("items", [])

    def _find_id(cid: str) -> tuple[str, str] | None:
        for c in cals:
            if c.get("id") == cid:
                return c["id"], c.get("summary") or cid
        return None

    def _find_primary() -> tuple[str, str]:
        for c in cals:
            if c.get("primary"):
                return c["id"], c.get("summary") or "primary"
        return "primary", "primary"

    if not hint:
        if user_name:
            default_id = _get_default_calendar(user_name)
            if default_id:
                hit = _find_id(default_id)
                if hit:
                    return hit
        return _find_primary()

    if hint == "primary":
        return _find_primary()

    # Exact id
    hit = _find_id(hint)
    if hit:
        return hit
    # Substring on summary
    h = hint.strip().lower()
    for c in cals:
        if h in (c.get("summary") or "").lower():
            return c["id"], c.get("summary") or hint
    # No match → user's default → primary
    if user_name:
        default_id = _get_default_calendar(user_name)
        if default_id:
            hit = _find_id(default_id)
            if hit:
                return hit
    return _find_primary()


def _build_event_body(
    title: str,
    start: str,
    end: str | None,
    description: str | None,
    location: str | None,
    timezone: str,
) -> dict:
    """Build a Google Calendar event body. Detects all-day by absence of 'T'."""
    is_all_day = "T" not in start
    body: dict = {"summary": title}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if is_all_day:
        # all-day: end is exclusive — if not given, default to next day
        from datetime import date, timedelta
        s_date = date.fromisoformat(start)
        e_date = date.fromisoformat(end) if end else (s_date + timedelta(days=1))
        body["start"] = {"date": s_date.isoformat()}
        body["end"] = {"date": e_date.isoformat()}
    else:
        from datetime import datetime, timedelta
        s_dt = datetime.fromisoformat(start)
        e_dt = datetime.fromisoformat(end) if end else (s_dt + timedelta(hours=1))
        body["start"] = {"dateTime": s_dt.isoformat(), "timeZone": timezone}
        body["end"] = {"dateTime": e_dt.isoformat(), "timeZone": timezone}
    return body


def create_event(
    user_name: str,
    title: str,
    start: str,
    end: str | None = None,
    *,
    calendar: str | None = None,
    description: str | None = None,
    location: str | None = None,
    timezone: str = DEFAULT_TZ,
) -> dict:
    svc = calendar_service(user_name)
    if not svc:
        return {"ok": False, "error": f"{user_name} has not linked Google"}
    cal_id, cal_name = _resolve_calendar_id(svc, calendar, user_name=user_name)
    body = _build_event_body(title, start, end, description, location, timezone)
    try:
        result = svc.events().insert(calendarId=cal_id, body=body).execute()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    # Trigger immediate sync so the widget updates
    try:
        sync_calendar_for(user_name)
    except Exception as e:
        logger.warning(f"post-create sync failed: {e}")
    return {
        "ok": True,
        "event_id": result.get("id"),
        "html_link": result.get("htmlLink"),
        "calendar": cal_name,
        "summary": result.get("summary"),
        "start": result.get("start"),
        "end": result.get("end"),
    }


def update_event(
    user_name: str,
    event_id: str,
    *,
    calendar: str | None = None,
    title: str | None = None,
    start: str | None = None,
    end: str | None = None,
    description: str | None = None,
    location: str | None = None,
    timezone: str = DEFAULT_TZ,
) -> dict:
    svc = calendar_service(user_name)
    if not svc:
        return {"ok": False, "error": f"{user_name} has not linked Google"}
    cal_id, cal_name = _resolve_calendar_id(svc, calendar, user_name=user_name)
    try:
        existing = svc.events().get(calendarId=cal_id, eventId=event_id).execute()
    except Exception:
        # Search every calendar for this event id (the agent may not know
        # which calendar a given event lives on).
        existing = None
        for c in svc.calendarList().list().execute().get("items", []):
            try:
                existing = svc.events().get(calendarId=c["id"], eventId=event_id).execute()
                cal_id, cal_name = c["id"], c.get("summary") or c["id"]
                break
            except Exception:
                continue
        if not existing:
            return {"ok": False, "error": f"event {event_id} not found"}

    patch: dict = {}
    if title is not None:
        patch["summary"] = title
    if description is not None:
        patch["description"] = description
    if location is not None:
        patch["location"] = location
    if start is not None:
        is_all_day = "T" not in start
        if is_all_day:
            from datetime import date, timedelta
            s_date = date.fromisoformat(start)
            e_date = date.fromisoformat(end) if end else (s_date + timedelta(days=1))
            patch["start"] = {"date": s_date.isoformat()}
            patch["end"] = {"date": e_date.isoformat()}
        else:
            from datetime import datetime, timedelta
            s_dt = datetime.fromisoformat(start)
            e_dt = datetime.fromisoformat(end) if end else (s_dt + timedelta(hours=1))
            patch["start"] = {"dateTime": s_dt.isoformat(), "timeZone": timezone}
            patch["end"] = {"dateTime": e_dt.isoformat(), "timeZone": timezone}
    elif end is not None:
        # end-only update
        is_all_day = "T" not in end
        if is_all_day:
            from datetime import date
            patch["end"] = {"date": date.fromisoformat(end).isoformat()}
        else:
            from datetime import datetime
            patch["end"] = {"dateTime": datetime.fromisoformat(end).isoformat(), "timeZone": timezone}

    if not patch:
        return {"ok": False, "error": "no fields to update"}
    try:
        result = svc.events().patch(calendarId=cal_id, eventId=event_id, body=patch).execute()
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        sync_calendar_for(user_name)
    except Exception:
        pass
    return {"ok": True, "event_id": result.get("id"), "summary": result.get("summary"), "calendar": cal_name}


def delete_event(user_name: str, event_id: str, *, calendar: str | None = None) -> dict:
    svc = calendar_service(user_name)
    if not svc:
        return {"ok": False, "error": f"{user_name} has not linked Google"}
    cal_id, cal_name = _resolve_calendar_id(svc, calendar, user_name=user_name)
    try:
        svc.events().delete(calendarId=cal_id, eventId=event_id).execute()
    except Exception:
        # Search across calendars
        deleted = False
        for c in svc.calendarList().list().execute().get("items", []):
            try:
                svc.events().delete(calendarId=c["id"], eventId=event_id).execute()
                cal_id, cal_name = c["id"], c.get("summary") or c["id"]
                deleted = True
                break
            except Exception:
                continue
        if not deleted:
            return {"ok": False, "error": f"event {event_id} not found"}
    try:
        sync_calendar_for(user_name)
    except Exception:
        pass
    return {"ok": True, "event_id": event_id, "calendar": cal_name}


# ─── Token-health nudge ─────────────────────────────────────────────────
# Google rotates refresh tokens for unverified-app sensitive scopes on a
# variable cadence (~monthly in practice for this app). Rather than fix
# the cadence (requires Google verification or Workspace), we detect the
# flip from `ok` → `refresh_failed:*` in the existing 15-min sync loop
# and Signal the affected user with a one-tap re-link URL. Throttled to
# one nudge per user per NUDGE_COOLDOWN_S so a stuck failure doesn't spam.

NUDGE_COOLDOWN_S = 6 * 3600

# Where each user's nudge goes. Falls back to Casey's number for users
# without a known Signal contact (he'll forward).
USER_SIGNAL_TO = {
    "casey": "+15056208470",
    "lindsey": "+15056208470",  # Casey forwards; replace when Lindsey is on Signal
}
NUDGE_FALLBACK = "+15056208470"

# user_name -> (last_status, last_nudge_ts_epoch)
_TOKEN_STATE: dict[str, tuple[str, float]] = {}


def _public_base_url() -> str:
    # The reverse-proxy hostname Casey can tap from a Signal link.
    return os.environ.get("BENSON_PUBLIC_URL", "https://benson.local")


async def _send_oauth_nudge(user_name: str, status: str) -> None:
    """Signal the user with a one-tap re-link URL for /admin/google."""
    import time as _time
    last = _TOKEN_STATE.get(user_name)
    now = _time.time()
    if last and last[0] == status and (now - last[1]) < NUDGE_COOLDOWN_S:
        return
    _TOKEN_STATE[user_name] = (status, now)
    to = USER_SIGNAL_TO.get(user_name.lower(), NUDGE_FALLBACK)
    url = f"{_public_base_url()}/admin/google"
    reason = status.split(":", 1)[1] if ":" in status else status
    text = (
        f"Benson here — {user_name.title()}'s Google OAuth needs re-linking "
        f"(Google revoked the refresh token: {reason[:80]}).\n\n"
        f"Tap to re-authorize: {url}\n\n"
        "Once you finish the flow, calendar + email tools start working "
        "again automatically. You won't hear from me again on this until "
        "the next time Google revokes."
    )
    try:
        from signal_handler import send_signal_message
        result = await send_signal_message(to, text)
        logger.info(
            f"oauth nudge sent to {to} for {user_name} "
            f"(status={status}, ok={result.get('ok')})"
        )
    except Exception:
        logger.exception(f"oauth nudge send failed for {user_name}")


def _record_token_ok(user_name: str) -> None:
    """Mark this user's token as healthy so the next failure re-triggers a nudge."""
    import time as _time
    _TOKEN_STATE[user_name] = ("ok", _time.time())


# ─── Background sync loop ────────────────────────────────────────────────
async def _sync_loop() -> None:
    import asyncio
    first_pass = True
    while True:
        try:
            users = list_linked_users()
            for u in users:
                user_name = u["user_name"]
                try:
                    # Check token health regardless of sync outcome so we
                    # nudge on refresh_failed even if sync would silently skip.
                    _, status = await asyncio.to_thread(
                        get_credentials_with_status, user_name
                    )
                    if status == "ok":
                        _record_token_ok(user_name)
                        await asyncio.to_thread(sync_calendar_for, user_name)
                    elif status.startswith("refresh_failed"):
                        # Don't nudge on the very first cold-start pass — we
                        # don't know whether this is a long-standing failed
                        # state or a fresh flip. Only nudge on subsequent
                        # passes (or after we've seen ok at least once).
                        last = _TOKEN_STATE.get(user_name)
                        if not first_pass or (last and last[0] == "ok"):
                            await _send_oauth_nudge(user_name, status)
                except Exception as e:
                    logger.warning(f"sync error for {user_name}: {e}")
        except Exception as e:
            logger.warning(f"sync loop error: {e}")
        first_pass = False
        await asyncio.sleep(15 * 60)  # 15 min


def start_sync_loop() -> None:
    import asyncio
    asyncio.create_task(_sync_loop())
