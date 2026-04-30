"""Benson's short-term memory (STM) — auto-curated working notes.

Benson writes here whenever something nontrivial happens (tool failure,
correction from Casey, successful side-effect tool, diagnosis). The
files are plain markdown, allowed to grow and recategorize. Recent
contents prepend to Benson's system prompt every turn.

Each night, the indexer consolidates STM into pgvector LTM
(memory_index) so this knowledge persists past the rolling window.

Layout:
  /opt/benson/memory/short_term/
    inbox.md                   # default scratch
    YYYY-MM-DD.md              # today's working journal (auto-created)
    topics/
      tool_caveats.md
      proposal_outcomes.md
      household_patterns.md
      open_questions.md

Topic names map directly to filenames under topics/. Special topics:
  'today'  → YYYY-MM-DD.md  (auto-rolled)
  'inbox'  → inbox.md
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger("benson.stm")

STM_ROOT = Path("/opt/benson/memory/short_term")
TOPICS_DIR = STM_ROOT / "topics"
INBOX_FILE = STM_ROOT / "inbox.md"

# Per-file size cap — when exceeded, stm_tidy splits or rotates.
MAX_TOPIC_BYTES = 10_000

# How many recent days to load into the system prompt each turn.
PROMPT_DAYS_BACK = 2
# Hard cap on STM content injected into system prompt (chars, not tokens).
PROMPT_BYTES_CAP = 3_000

_VALID_TOPIC = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _today_file() -> Path:
    return STM_ROOT / f"{date.today().isoformat()}.md"


def _resolve_topic(topic: str) -> Path:
    """Map a topic name to a file path under STM_ROOT."""
    t = topic.strip().lower()
    if t in ("today", ""):
        return _today_file()
    if t == "inbox":
        return INBOX_FILE
    if not _VALID_TOPIC.match(t):
        raise ValueError(
            f"invalid topic name {topic!r} — use lowercase letters, "
            f"digits, underscores; max 40 chars; must start with a letter"
        )
    return TOPICS_DIR / f"{t}.md"


def stm_append(topic: str, content: str) -> dict:
    """Append a timestamped entry to a topic file. Creates the file if
    missing. Returns the path written and the new total file size."""
    if not content or not content.strip():
        return {"ok": False, "error": "content is required"}
    try:
        path = _resolve_topic(topic)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H:%M")
    block = f"\n- [{ts}] {content.strip()}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)
    size = path.stat().st_size
    return {
        "ok": True,
        "path": str(path),
        "topic": topic,
        "size_bytes": size,
        "needs_tidy": size > MAX_TOPIC_BYTES,
    }


def stm_read(topic: str | None = None, days_back: int = 1) -> dict:
    """Read STM contents.

    - topic=None: returns all recent files (today + last N days + topic
      files), capped to PROMPT_BYTES_CAP. Use this for prompt-loading.
    - topic=<name>: returns the full named topic file (or 'today').
    """
    if topic:
        try:
            path = _resolve_topic(topic)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not path.exists():
            return {"ok": True, "topic": topic, "content": "", "exists": False}
        return {
            "ok": True,
            "topic": topic,
            "path": str(path),
            "content": path.read_text(errors="replace"),
        }

    # Aggregate: today's journal + last N daily journals + all topic files.
    chunks: list[tuple[str, str]] = []
    today = date.today()
    for d in range(0, max(1, days_back) + 1):
        day = today - timedelta(days=d)
        f = STM_ROOT / f"{day.isoformat()}.md"
        if f.exists() and f.stat().st_size > 0:
            chunks.append((f"### {day.isoformat()}", f.read_text(errors="replace")))

    if INBOX_FILE.exists() and INBOX_FILE.stat().st_size > 0:
        chunks.append(("### inbox", INBOX_FILE.read_text(errors="replace")))

    if TOPICS_DIR.exists():
        for fp in sorted(TOPICS_DIR.glob("*.md")):
            if fp.stat().st_size > 0:
                chunks.append((f"### {fp.stem}", fp.read_text(errors="replace")))

    return {"ok": True, "chunks": chunks, "count": len(chunks)}


def stm_list() -> dict:
    """List all STM files with their size + last-modified."""
    items: list[dict] = []

    def _add(p: Path) -> None:
        if not p.exists():
            return
        st = p.stat()
        items.append({
            "path": str(p),
            "name": p.relative_to(STM_ROOT).as_posix(),
            "bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })

    _add(INBOX_FILE)
    for fp in sorted(STM_ROOT.glob("*.md")):
        if fp != INBOX_FILE:
            _add(fp)
    if TOPICS_DIR.exists():
        for fp in sorted(TOPICS_DIR.glob("*.md")):
            _add(fp)
    return {"ok": True, "files": items, "count": len(items)}


def stm_for_prompt() -> str:
    """Format recent STM as a system-prompt section. Capped at
    PROMPT_BYTES_CAP. Empty string when nothing meaningful exists."""
    out = stm_read(topic=None, days_back=PROMPT_DAYS_BACK)
    chunks = out.get("chunks") or []
    if not chunks:
        return ""

    lines = ["--- short-term memory (your own working notes) ---"]
    used = len(lines[0]) + 1
    for header, body in chunks:
        body_stripped = body.strip()
        if not body_stripped:
            continue
        block = f"\n{header}\n{body_stripped}"
        if used + len(block) > PROMPT_BYTES_CAP:
            # Truncate this chunk to fit
            remaining = PROMPT_BYTES_CAP - used - len(header) - 5
            if remaining > 200:
                block = f"\n{header}\n{body_stripped[:remaining]}…"
                lines.append(block)
                used += len(block)
            break
        lines.append(block)
        used += len(block)
    lines.append("\n--- end short-term memory ---")
    return "".join(lines)


async def stm_tidy() -> dict:
    """Run a Sonnet-driven dedupe/merge/split pass over STM files. Used
    when files exceed MAX_TOPIC_BYTES or by manual trigger.

    Implementation note: this is intentionally minimal — it asks Sonnet
    via OAuth to rewrite each oversized file with redundancy removed
    and stable headers. Files that already fit stay untouched.
    """
    from oauth_oneshot import ask as oauth_ask

    listing = stm_list()
    over_cap: list[dict] = [f for f in listing["files"] if f["bytes"] > MAX_TOPIC_BYTES]
    if not over_cap:
        return {"ok": True, "tidied": [], "reason": "no files over cap"}

    tidied: list[dict] = []
    for f in over_cap:
        path = Path(f["path"])
        original = path.read_text(errors="replace")
        prompt = (
            "You are tidying one of Benson's short-term-memory note files. "
            "Goal: dedupe, merge near-duplicates, group related entries, "
            "preserve all distinct facts and lessons. Output is ONLY the "
            "rewritten markdown content of the file — no preamble, no "
            "fences, no commentary.\n\n"
            f"Filename: {path.name}\n\n"
            f"Current content:\n{original}"
        )
        cleaned = await oauth_ask(prompt, model="sonnet", timeout_s=120)
        if not cleaned or len(cleaned) < 100:
            tidied.append({"file": f["name"], "ok": False, "reason": "tidy returned empty/short output"})
            continue
        # Write atomically — keep an original.bak in case of regret.
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(original)
        path.write_text(cleaned)
        tidied.append({
            "file": f["name"],
            "ok": True,
            "before_bytes": f["bytes"],
            "after_bytes": path.stat().st_size,
            "backup": str(bak),
        })
    return {"ok": True, "tidied": tidied}
