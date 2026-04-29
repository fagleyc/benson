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
PROPOSAL_LOG_DIR = Path("/tmp/benson-proposals")

# Module-level lock — only one propose_change session at a time.
# The bundled Claude Code CLI subprocess doesn't tolerate concurrent
# sessions on the same machine, and this morning Benson opened three
# in 9 minutes. Second one crashed with "Fatal error in message
# reader: exit code 1". Lock is asyncio (lives inside the event loop).
_PROPOSAL_LOCK: asyncio.Lock | None = None


def _proposal_lock() -> asyncio.Lock:
    """Lazy-init so we bind to the right loop."""
    global _PROPOSAL_LOCK
    if _PROPOSAL_LOCK is None:
        _PROPOSAL_LOCK = asyncio.Lock()
    return _PROPOSAL_LOCK


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


async def read_my_source(path: str, max_lines: int = 400) -> dict:
    """Read a file from /opt/benson (read-only). Use this when the user
    asks where something lives or whether some capability already exists.
    `path` may be relative to /opt/benson or absolute under /opt/benson.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_DIR / path
    try:
        resolved = p.resolve()
    except Exception as e:
        return {"ok": False, "error": f"could not resolve path: {e}"}
    # Refuse paths outside the repo or inside .git config.
    repo_resolved = REPO_DIR.resolve()
    try:
        resolved.relative_to(repo_resolved)
    except ValueError:
        return {"ok": False, "error": f"refusing to read outside /opt/benson: {resolved}"}
    if any(part in {".git"} for part in resolved.parts):
        return {"ok": False, "error": "refusing to read inside .git/"}
    if not resolved.exists():
        return {"ok": False, "error": f"not found: {resolved}"}
    if not resolved.is_file():
        return {"ok": False, "error": f"not a file: {resolved}"}
    if resolved.stat().st_size > 500_000:
        return {"ok": False, "error": f"file too large ({resolved.stat().st_size} bytes); use grep_my_source instead"}
    try:
        text = resolved.read_text(errors="replace")
    except Exception as e:
        return {"ok": False, "error": f"read failed: {e}"}
    lines = text.splitlines()
    truncated = len(lines) > max_lines
    if truncated:
        lines = lines[:max_lines]
    return {
        "ok": True,
        "path": str(resolved),
        "lines": lines,
        "line_count": len(lines),
        "truncated": truncated,
    }


async def grep_my_source(pattern: str, path_glob: str = "**/*.py", max_results: int = 60) -> dict:
    """Search /opt/benson for a regex pattern. Use this to find where a
    capability is defined or whether something already exists. `path_glob`
    defaults to '**/*.py' but can be 'middleware/templates/*.html', etc.
    """
    import re as _re
    try:
        regex = _re.compile(pattern)
    except _re.error as e:
        return {"ok": False, "error": f"bad regex: {e}"}

    # Match files under REPO_DIR using the glob, skipping ignored noise.
    skip_parts = {".git", "venv", ".venv", "__pycache__", ".cache", ".nv",
                  "node_modules", "memory", "recipes", "context", "logs",
                  "ha", "music_assistant", "signal", "whatsapp", "kokoro",
                  "openwakeword", "whisper", "backup", "backups"}
    results: list[dict] = []
    truncated = False
    try:
        candidates = list(REPO_DIR.glob(path_glob))
    except Exception as e:
        return {"ok": False, "error": f"bad glob: {e}"}

    for fp in candidates:
        if any(part in skip_parts for part in fp.parts):
            continue
        if not fp.is_file():
            continue
        try:
            if fp.stat().st_size > 1_000_000:
                continue
            for i, line in enumerate(fp.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    results.append({
                        "path": str(fp.relative_to(REPO_DIR)),
                        "line": i,
                        "text": line[:240],
                    })
                    if len(results) >= max_results:
                        truncated = True
                        break
        except Exception:
            continue
        if truncated:
            break

    return {
        "ok": True,
        "pattern": pattern,
        "path_glob": path_glob,
        "match_count": len(results),
        "results": results,
        "truncated": truncated,
    }


_LOCAL_FILE_PREFIXES = (
    "/tmp/benson-",
    "/home/casey/Benson/",
)


async def write_local_file(path: str, content: str, append: bool = False) -> dict:
    """Write a file under /tmp/benson-* or /home/casey/Benson/*. Use this when
    the household asks you to save logs, notes, debug captures, images, documents,
    etc. — instead of claiming you saved a file you can't actually save.

    Path-locked to /tmp/benson-* or /home/casey/Benson/* (anything else is
    refused); real source edits go through propose_change.
    """
    if not any(path.startswith(pfx) for pfx in _LOCAL_FILE_PREFIXES):
        allowed = " or ".join(f"'{p}'" for p in _LOCAL_FILE_PREFIXES)
        return {
            "ok": False,
            "error": f"path must start with {allowed} (got {path!r})",
        }
    if len(content) > 1_000_000:
        return {"ok": False, "error": f"content too large ({len(content)} bytes); cap is 1MB"}
    p = Path(path)
    try:
        # Resolve and re-check — defends against path traversal tricks
        # (e.g. /tmp/benson-../etc/passwd or /home/casey/Benson/../../root).
        resolved = p.resolve()
        if not any(str(resolved).startswith(pfx) for pfx in _LOCAL_FILE_PREFIXES):
            return {"ok": False, "error": f"resolved path escapes allowed prefixes: {resolved}"}
        # If the prefix is followed by a /, the parent must exist or we create it.
        # If the prefix is followed by other characters (e.g. /tmp/benson-foo.log),
        # the parent is /tmp which already exists.
        resolved.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(resolved, mode) as f:
            f.write(content)
        return {
            "ok": True,
            "path": str(resolved),
            "bytes_written": len(content),
            "mode": "append" if append else "write",
        }
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

    Concurrency: only one propose_change session runs at a time. The
    bundled Claude Code CLI subprocess crashes under concurrent
    sessions ("Fatal error in message reader: exit code 1"). A second
    caller gets a clean refusal — wait for the first to finish.
    """
    if not rationale.strip():
        return {"ok": False, "error": "rationale is required"}
    if not instructions.strip():
        return {"ok": False, "error": "instructions are required"}

    lock = _proposal_lock()
    if lock.locked():
        return {
            "ok": False,
            "error": (
                "another propose_change is already running. Wait for it "
                "to finish (check /admin/proposals), then retry. If you "
                "want to refine an existing proposal, read its branch "
                "with read_my_source / grep_my_source instead of opening "
                "a new one."
            ),
        }

    async with lock:
        return await _propose_change_locked(rationale, instructions)


PROPOSAL_META_FILE = ".benson-proposal-meta.json"
PROPOSAL_META_TAG = "[proposal-meta]"


async def _propose_change_locked(rationale: str, instructions: str) -> dict:
    import json as _json
    branch = _proposal_branch_name(rationale)
    logger.info(f"propose_change: opening branch {branch}")

    PROPOSAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PROPOSAL_LOG_DIR / f"{branch.replace('/', '_')}.log"

    def _append_log(text: str) -> None:
        try:
            with open(log_path, "a") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        except Exception as e:
            logger.warning(f"failed to append proposal log: {e}")

    _append_log(f"=== {branch} @ {datetime.now(timezone.utc).isoformat()} ===")
    _append_log(f"RATIONALE: {rationale}")
    _append_log(f"INSTRUCTIONS: {instructions}")

    # Branch from main. Working tree must be clean — if it isn't, give a
    # useful diagnostic instead of just printing the porcelain output.
    try:
        status = await asyncio.to_thread(_git, ["status", "--porcelain"])
        head_branch = (await asyncio.to_thread(_git, ["branch", "--show-current"])).strip()
        if status.strip() or head_branch != "main":
            _append_log(f"REFUSED: dirty tree (head={head_branch}): {status[:300]}")
            return {
                "ok": False,
                "error": (
                    f"workspace is not clean — head is '{head_branch}', "
                    f"changes:\n{status[:400]}\n\n"
                    f"A previous proposal likely crashed mid-edit. Recovery: "
                    f"`cd /opt/benson && git checkout main && git checkout -- .` "
                    f"OR commit/preserve the changes on the existing branch."
                ),
                "log_path": str(log_path),
            }
        await asyncio.to_thread(_git, ["checkout", "-b", branch])
    except subprocess.CalledProcessError as e:
        out = e.output.decode() if isinstance(e.output, bytes) else str(e.output)
        _append_log(f"GIT SETUP FAILED: {out}")
        return {"ok": False, "error": f"git setup failed: {out[:300]}", "log_path": str(log_path)}

    # Write proposal metadata to the branch root and commit it BEFORE the
    # SDK runs. This guarantees rationale + instructions are visible on
    # /admin/proposals even if the SDK crashes mid-edit (the previous
    # behavior left 0-commit branches with no readable context).
    meta_path = REPO_DIR / PROPOSAL_META_FILE
    meta = {
        "rationale": rationale,
        "instructions": instructions,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
    }
    try:
        meta_path.write_text(_json.dumps(meta, indent=2))
        await asyncio.to_thread(_git, ["add", PROPOSAL_META_FILE])
        rationale_subject = rationale.replace("\n", " ").strip()[:60]
        meta_msg = (
            f"{PROPOSAL_META_TAG} {rationale_subject}\n\n"
            f"RATIONALE: {rationale}\n\n"
            f"INSTRUCTIONS: {instructions}"
        )
        await asyncio.to_thread(_git, ["commit", "-m", meta_msg])
        _append_log("META: committed proposal metadata")
    except subprocess.CalledProcessError as e:
        out = e.output.decode() if isinstance(e.output, bytes) else str(e.output)
        _append_log(f"META COMMIT FAILED: {out}")
        # Best-effort recovery — switch back to main, delete branch, return.
        try:
            await asyncio.to_thread(_git, ["checkout", "--", "."])
            await asyncio.to_thread(_git, ["checkout", "main"])
            await asyncio.to_thread(_git, ["branch", "-D", branch])
        except Exception:
            pass
        return {"ok": False, "error": f"meta commit failed: {out[:300]}", "log_path": str(log_path)}

    # Spawn Claude Code SDK session.
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    def _stderr_to_log(line: str) -> None:
        # The SDK pipes the bundled CLI's stderr through this callback only
        # when set — without it, "Check stderr output for details" is opaque.
        _append_log(f"[stderr] {line.rstrip()}")

    options = ClaudeAgentOptions(
        cwd=str(REPO_DIR),
        model="opus",  # self-mod planning needs Opus per Casey 2026-04-29
        max_turns=30,
        permission_mode="bypassPermissions",
        allowed_tools=["Read", "Edit", "Write", "Grep", "Glob", "Bash"],
        stderr=_stderr_to_log,
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

    full_transcript: list[str] = []
    sdk_error: str | None = None

    async def _run():
        async for msg in query(prompt=full_prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        full_transcript.append(block.text)

    try:
        await asyncio.wait_for(_run(), timeout=900)  # 15 min hard cap
    except asyncio.TimeoutError:
        sdk_error = "SDK session timed out after 900s"
        logger.warning(f"propose_change: {sdk_error} on {branch}")
    except Exception as e:
        sdk_error = f"{type(e).__name__}: {e}"
        logger.warning(f"propose_change SDK failed: {sdk_error}")

    transcript_blob = "\n\n".join(full_transcript)
    _append_log("--- SDK transcript ---")
    _append_log(transcript_blob if transcript_blob else "(empty transcript)")
    if sdk_error:
        _append_log(f"--- SDK ERROR ---\n{sdk_error}")

    m = re.search(r"SUMMARY:\s*(.+?)(?:\n\n|$)", transcript_blob, re.DOTALL)
    summary = m.group(1).strip() if m else (transcript_blob[:600] or "(no summary)")

    # Crash recovery: if the SDK crashed mid-edit, the working tree may
    # have uncommitted changes. Reset before switching back to main —
    # otherwise the next propose_change refuses with "dirty tree".
    try:
        await asyncio.to_thread(_git, ["checkout", "--", "."])
    except Exception as e:
        logger.warning(f"git checkout -- . cleanup failed: {e}")

    # Count SDK commits on this branch — exclude the proposal-meta commit
    # we made before the SDK ran. If only meta exists, the SDK contributed
    # nothing and we should drop the branch.
    sdk_commits: list[str] = []
    try:
        log_out = await asyncio.to_thread(
            _git, ["log", "main..HEAD", "--oneline"], check=False
        )
        for ln in log_out.strip().splitlines():
            if ln.strip() and PROPOSAL_META_TAG not in ln:
                sdk_commits.append(ln)
    except Exception as e:
        logger.warning(f"git log failed on {branch}: {e}")

    try:
        await asyncio.to_thread(_git, ["checkout", "main"])
    except Exception as e:
        logger.warning(f"git checkout main after proposal failed: {e}")

    if not sdk_commits:
        try:
            await asyncio.to_thread(_git, ["branch", "-D", branch])
        except Exception:
            pass
        _append_log("OUTCOME: SDK contributed no commits — branch deleted")
        return {
            "ok": False,
            "error": (
                f"SDK session ended without committing any code changes"
                + (f" (sdk_error: {sdk_error})" if sdk_error else "")
                + f". Full log: {log_path}"
            ),
            "branch": None,
            "summary": summary,
            "log_path": str(log_path),
        }

    _append_log(f"OUTCOME: {len(sdk_commits)} SDK commit(s) on {branch}")
    return {
        "ok": True,
        "branch": branch,
        "commits": sdk_commits,
        "summary": summary,
        "review_url": "/admin/proposals",
        "log_path": str(log_path),
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
    """List proposal/* branches not yet merged to main, with rationale +
    instructions sourced from each branch's .benson-proposal-meta.json
    when present (older branches without the file degrade gracefully)."""
    import json as _json
    try:
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

        # Pull rationale + instructions from the proposal-meta file on the
        # branch. Falls back to empty strings for older branches that
        # predate the metadata mechanism.
        rationale = ""
        instructions = ""
        try:
            meta_raw = _git(["show", f"{branch}:{PROPOSAL_META_FILE}"])
            meta = _json.loads(meta_raw)
            rationale = (meta.get("rationale") or "").strip()
            instructions = (meta.get("instructions") or "").strip()
        except Exception:
            pass

        # Count "real" SDK commits — exclude the meta commit so 0-commit
        # branches that only contain metadata are clearly visible as such.
        sdk_commit_count = max(0, commit_count - (1 if rationale else 0))

        proposals.append({
            "branch": branch,
            "date": date_s,
            "subject": subject,
            "diffstat": stat_line,
            "commit_count": commit_count,
            "sdk_commit_count": sdk_commit_count,
            "rationale": rationale,
            "instructions": instructions,
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
