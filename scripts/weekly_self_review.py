#!/usr/bin/env python3
"""Weekly self-review.

Pulls Benson's last 7 days of conversations, asks Claude (OAuth quota,
no API charge) to flag failures and rough edges, and posts a punch
list to the household admin via Signal. Casey reviews the list and
either replies "fix #1, #3" (which Benson can act on by calling
propose_change) or pastes a more specific instruction back.

Run by: systemd timer `benson-self-review.timer` (Sundays 7am MT).
Can also be invoked manually: `sudo -u benson python3 weekly_self_review.py`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Make the middleware importable.
sys.path.insert(0, "/opt/benson/middleware")

# Load env from /etc/benson/env (the systemd unit also does this, but
# allow ad-hoc CLI invocation to work).
ENV_FILE = Path("/etc/benson/env")
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from oauth_oneshot import ask as oauth_ask  # noqa: E402
from self_modify import read_my_conversations, read_my_logs  # noqa: E402
from signal_handler import send_signal_message  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benson.weekly_review")


REVIEW_PROMPT = """You are Benson, reviewing your own behavior over the past week.

Below are conversations from the last 7 days. Look for:

1. Failures: places you said "I don't have access", "I can't see that",
   "the integration isn't working", or otherwise refused something you
   actually CAN do. (You have tools for calendar, gmail, recipes, music,
   home automation, memory, lists, events, image vision, URL fetch,
   chores, weather.)
2. Repetition: questions or corrections the same person had to repeat
   to you across turns.
3. Tool errors visible in your responses (apologetic about a crash,
   weird output, etc).
4. Missing capabilities: things people kept asking for that you couldn't
   do at all.

Return a JSON array (and ONLY a JSON array — no markdown, no prose) of
at most 6 items. Each item:
{
  "title": "<terse one-line summary>",
  "evidence": "<which conversation id(s), what you said wrong>",
  "fix": "<concrete proposal — file/function to edit, or 'tool gap: <name>'>",
  "severity": "low|medium|high"
}

If nothing meaningful went wrong, return [].

CONVERSATIONS:
{conversations}
"""


def _format_conversations(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(
            f"[#{r['id']} {r['created_at']} {r['speaker']}@{r['room'] or '?'}]\n"
            f"  user: {r['user_text']}\n"
            f"  benson: {r['benson_response']}"
        )
    return "\n\n".join(out)


def _review_recipient() -> str | None:
    """Where to send the punch list. Prefer SIGNAL_REVIEW_RECIPIENT;
    fall back to the first SIGNAL_ALLOWED_NUMBERS entry."""
    explicit = (os.environ.get("SIGNAL_REVIEW_RECIPIENT") or "").strip()
    if explicit:
        return explicit
    allowed = (os.environ.get("SIGNAL_ALLOWED_NUMBERS") or "").strip()
    if allowed:
        return allowed.split(",")[0].strip()
    return None


async def main() -> int:
    conv_result = await read_my_conversations(days_back=7, limit=200)
    rows = conv_result.get("conversations", [])
    if not rows:
        log.info("no conversations in the last 7 days; skipping review")
        return 0

    log.info(f"reviewing {len(rows)} conversations")
    prompt = REVIEW_PROMPT.replace("{conversations}", _format_conversations(rows))

    review = await oauth_ask(prompt, model="sonnet", timeout_s=180)
    if not review:
        log.warning("oauth_ask returned empty; aborting")
        return 1

    # Strip optional ```json fences.
    blob = review.strip()
    if blob.startswith("```"):
        blob = blob.split("\n", 1)[1] if "\n" in blob else blob
        if blob.endswith("```"):
            blob = blob.rsplit("```", 1)[0]
    blob = blob.strip()

    # Format for Signal — compact, scannable, no JSON dump in the user's face.
    import json as _json
    try:
        items = _json.loads(blob)
    except Exception:
        log.warning(f"non-JSON review output, raw text follows:\n{blob[:1000]}")
        items = []

    if not items:
        msg = "Weekly self-review: nothing notable surfaced. Carrying on."
    else:
        lines = [f"Weekly self-review ({len(items)} item{'s' if len(items)!=1 else ''}):", ""]
        for i, it in enumerate(items, 1):
            title = it.get("title", "(untitled)")
            sev = it.get("severity", "medium")
            fix = it.get("fix", "")
            lines.append(f"{i}. [{sev}] {title}")
            if fix:
                lines.append(f"   → {fix}")
        lines.append("")
        lines.append("Reply 'fix N' to have me open a propose_change for that item.")
        msg = "\n".join(lines)

    recipient = _review_recipient()
    if not recipient:
        log.warning("no review recipient configured; printing instead:\n" + msg)
        print(msg)
        return 0

    log.info(f"sending review to {recipient}")
    result = await send_signal_message(recipient, msg)
    if not result.get("ok"):
        log.warning(f"send failed: {result}")
        return 2

    log.info("review delivered")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
