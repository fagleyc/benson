"""Self-awareness + self-modification primitives for Benson.

Three families of tools:

  AWARENESS (read-only)
    - read_my_conversations(days_back, search?) → recent turns from
      `conversations`, scoped to one speaker or all.
    - read_my_logs(lines?, since?, level?) → journalctl output for
      benson.service.
    - list_my_tools() → tool registry summary so Benson can describe
      his own capability surface.

  PROPOSAL (writes a git branch, doesn't apply)
    - propose_change(rationale, instructions) → spawns a Claude Code
      SDK session in /opt/benson with bypassPermissions, lets it edit
      and commit to a `proposal/<timestamp>-<slug>` branch. Returns
      branch name + summary. Casey reviews on /admin/proposals and
      clicks "merge" to actually apply.

The apply path lives in scripts/apply_proposal.sh — it fast-forwards
the proposal branch to main, syntax-checks, and restarts benson via
the sudoers carveout, rolling back if startup fails.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

from config import PG_DSN

logger = logging.getLogger("benson.self_modify")

REPO_DIR = Path("/opt/benson")
APPLY_SCRIPT = REPO_DIR / "scripts" / "apply_proposal.sh"


# ─── Awareness ────────────────────────────────────────────────────────────

async def read_my_conversations(
    days_back: int = 7,
    speaker: str | None = None,
    search: str | None = None,
    limit: int = 50,
) -> dict:
    """Pull recent (speaker, user_text, benson_response) turns from the
    conversations table. Use when reflecting on past behavior — what
    you said, what failed, what someone keeps having to repeat.
    """
    sql = (
        "SELECT id, speaker, room, user_text, benson_response, tier, "
        "created_at FROM conversations WHERE created_at >= NOW() - "
        "(%s || ' days')::interval"
    )
    params: list = [days_back]
    if speaker:
        sql += " AND speaker = %s"
        params.append(speaker)
    if search:
        sql += " AND (user_text ILIKE %s OR benson_response ILIKE %s)"
        params.extend([f"%{search}%", f"%{search}%"])
    sql += " ORDER BY id DESC LIMIT %s"
    params.append(min(limit, 200))
    rows = await asyncio.to_thread(_query, sql, tuple(params))
    return {
        "ok": True,
        "count": len(rows),
        "conversations": [
            {
                "id": r["id"],
                "speaker": r["speaker"],
                "room": r["room"],
                "user_text": r["user_text"],
                "benson_response": r["benson_response"],
                "tier": r["tier"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ],
    }


async def read_my_logs(lines: int = 100, since: str | None = None) -> dict:
    """Read the most recent benson.service log entries. Use when
    diagnosing why a tool failed or a request didn't go through.
    `since` accepts journalctl-style relative times like '1 hour ago'
    or '2026-04-28 12:00'.
    """
    cmd = ["journalctl", "-u", "benson", "--no-pager", "-n", str(min(lines, 500))]
    if since:
        cmd += ["--since", since]
    try:
        out = await asyncio.to_thread(
            subprocess.check_output, cmd, stderr=subprocess.STDOUT, timeout=15
        )
        return {"ok": True, "lines": out.decode("utf-8", errors="replace").splitlines()}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": e.output.decode("utf-8", errors="replace")}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


async def list_my_tools() -> dict:
    """Return a summary of every tool currently registered in this
    Benson instance — names + one-line descriptions. Useful when
    deciding whether a capability already exists before proposing a
    new one.
    """
    # Late import: agent_tools registers self_modify tools at module
    # load, so importing it here would be circular at module import.
    from agent_tools import TOOLS
    return {
        "ok": True,
        "count": len(TOOLS),
        "tools": [
            {"name": t["name"], "description": t["description"]}
            for t in TOOLS
        ],
    }


# ─── Proposal ─────────────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 30) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "change"


def _proposal_branch_name(rationale: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"proposal/{ts}-{_slug(rationale)}"


def _git(args: list[str], check: bool = True) -> str:
    """Run a git command in the Benson repo and return stdout."""
    env = os.environ.copy()
    # Identity for commits made by this process. /opt/benson/.gitconfig
    # isn't writable by the benson user, so we set per-invocation.
    env.setdefault("GIT_AUTHOR_NAME", "Benson")
    env.setdefault("GIT_AUTHOR_EMAIL", "benson@fagley.home")
    env.setdefault("GIT_COMMITTER_NAME", "Benson")
    env.setdefault("GIT_COMMITTER_EMAIL", "benson@fagley.home")
    out = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), *args],
        env=env,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    return out.decode("utf-8", errors="replace")


async def propose_change(rationale: str, instructions: str) -> dict:
    """Open a self-modification proposal.

    Spawns a Claude Code SDK session against /opt/benson with edit
    permissions, gives it the rationale + instructions, lets it edit
    the necessary files and commit to a fresh `proposal/<timestamp>`
    branch. Returns the branch name + a one-paragraph summary.

    The branch is NOT merged automatically. Casey reviews diffs on the
    /admin/proposals page and clicks "merge" to apply.
    """
    if not rationale.strip():
        return {"ok": False, "error": "rationale is required"}
    if not instructions.strip():
        return {"ok": False, "error": "instructions are required"}

    branch = _proposal_branch_name(rationale)
    logger.info(f"propose_change: opening branch {branch}")

    # Branch from main. Working tree must be clean — if it isn't,
    # something else is mid-edit and we should refuse rather than
    # mix changes.
    try:
        status = await asyncio.to_thread(_git, ["status", "--porcelain"])
        if status.strip():
            return {
                "ok": False,
                "error": f"working tree is dirty, refusing: {status[:200]}",
            }
        await asyncio.to_thread(_git, ["checkout", "main"])
        await asyncio.to_thread(_git, ["checkout", "-b", branch])
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"git setup failed: {e.output[:300]}"}

    # Spawn Claude Code SDK session. Same engine Casey uses to talk
    # to me — we point it at /opt/benson with full edit permissions
    # and feed it the proposal instructions.
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )
    options = ClaudeAgentOptions(
        cwd=str(REPO_DIR),
        model="sonnet",
        max_turns=30,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Edit", "Write", "Grep", "Glob", "Bash"],
    )
    full_prompt = (
        "You are improving Benson, a household AI assistant. You are running "
        "in /opt/benson, a git repo currently checked out on a fresh proposal "
        "branch. Your job:\n\n"
        "1. Read the relevant code under middleware/ to understand the "
        "current behavior.\n"
        "2. Make the smallest sensible change that addresses the rationale. "
        "Don't refactor adjacent code unless directly necessary.\n"
        "3. Verify Python syntax with `python -m py_compile <file>` after "
        "edits via the Bash tool.\n"
        "4. Commit your change(s) on this branch with a clear message that "
        "starts with the rationale and explains *why*. Use:\n"
        "   git add <changed files>\n"
        "   GIT_AUTHOR_NAME=Benson GIT_AUTHOR_EMAIL=benson@fagley.home "
        "GIT_COMMITTER_NAME=Benson GIT_COMMITTER_EMAIL=benson@fagley.home "
        "git commit -m \"<message>\"\n"
        "5. End your response with a one-paragraph summary of what you "
        "changed and why, prefixed with 'SUMMARY:'.\n\n"
        "Do NOT restart the service, edit /etc/, modify .git config, push "
        "to a remote, or touch files outside /opt/benson. Do NOT merge to "
        "main — Casey reviews and merges.\n\n"
        f"RATIONALE: {rationale}\n\n"
        f"INSTRUCTIONS: {instructions}"
    )

    summary_lines: list[str] = []
    full_transcript: list[str] = []

    async def _run():
        async for msg in query(prompt=full_prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        full_transcript.append(block.text)

    try:
        await asyncio.wait_for(_run(), timeout=900)  # 15 min hard cap
    except asyncio.TimeoutError:
        logger.warning(f"propose_change: SDK session timed out on {branch}")
    except Exception as e:
        logger.warning(f"propose_change SDK failed: {type(e).__name__}: {e}")

    # Extract the SUMMARY: block from the transcript.
    transcript_blob = "\n\n".join(full_transcript)
    m = re.search(r"SUMMARY:\s*(.+?)(?:\n\n|$)", transcript_blob, re.DOTALL)
    summary = m.group(1).strip() if m else (transcript_blob[:600] or "(no summary)")

    # Did the session actually commit anything on this branch?
    try:
        log_out = await asyncio.to_thread(
            _git, ["log", "main..HEAD", "--oneline"], check=False
        )
        commits = [ln for ln in log_out.strip().splitlines() if ln.strip()]
    except Exception as e:
        commits = []
        logger.warning(f"git log failed on {branch}: {e}")

    # Always switch back to main so the live workspace isn't pinned to
    # the proposal branch (the apply script will handle the merge).
    try:
        await asyncio.to_thread(_git, ["checkout", "main"])
    except Exception as e:
        logger.warning(f"git checkout main after proposal failed: {e}")

    if not commits:
        # No commits — kill the empty branch so the dashboard doesn't
        # show an apparent proposal that does nothing.
        try:
            await asyncio.to_thread(_git, ["branch", "-D", branch])
        except Exception:
            pass
        return {
            "ok": False,
            "error": "session ended without committing any changes",
            "branch": None,
            "summary": summary,
        }

    return {
        "ok": True,
        "branch": branch,
        "commits": commits,
        "summary": summary,
        "review_url": "/admin/proposals",
    }


# ─── Apply (called by dashboard merge button) ─────────────────────────────

def apply_proposal(branch: str) -> dict:
    """Run scripts/apply_proposal.sh <branch>. Returns stdout/stderr +
    exit code. Called from the FastAPI handler when Casey clicks merge.
    """
    if not branch.startswith("proposal/"):
        return {"ok": False, "error": "branch must start with 'proposal/'"}
    try:
        out = subprocess.run(
            [str(APPLY_SCRIPT), branch],
            cwd=str(REPO_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "ok": out.returncode == 0,
            "exit_code": out.returncode,
            "stdout": out.stdout,
            "stderr": out.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "apply script timed out after 120s"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def reject_proposal(branch: str) -> dict:
    """Delete a proposal branch without merging."""
    if not branch.startswith("proposal/"):
        return {"ok": False, "error": "branch must start with 'proposal/'"}
    try:
        _git(["branch", "-D", branch])
        return {"ok": True, "deleted": branch}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": e.output[:300] if e.output else str(e)}


def list_open_proposals() -> list[dict]:
    """List proposal/* branches not yet merged to main."""
    try:
        # for-each-ref gives us branch + last commit info in one shot
        fmt = "%(refname:short)|%(committerdate:iso8601)|%(contents:subject)"
        out = _git([
            "for-each-ref",
            "--sort=-committerdate",
            f"--format={fmt}",
            "refs/heads/proposal/",
        ])
    except subprocess.CalledProcessError:
        return []

    proposals = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        branch, date_s, subject = parts
        try:
            diffstat = _git(["diff", "--stat", f"main..{branch}"]).strip().splitlines()
            stat_line = diffstat[-1] if diffstat else ""
            commit_count = len(_git(["log", "--oneline", f"main..{branch}"]).strip().splitlines())
        except subprocess.CalledProcessError:
            stat_line = "(diff failed)"
            commit_count = 0
        proposals.append({
            "branch": branch,
            "date": date_s,
            "subject": subject,
            "diffstat": stat_line,
            "commit_count": commit_count,
        })
    return proposals


def proposal_diff(branch: str) -> str:
    """Full unified diff of a proposal branch vs main."""
    if not branch.startswith("proposal/"):
        raise ValueError("branch must start with 'proposal/'")
    return _git(["diff", f"main..{branch}"])


# ─── Internal ─────────────────────────────────────────────────────────────

def _query(sql: str, params: tuple = ()) -> list[dict]:
    with psycopg2.connect(**PG_DSN) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
