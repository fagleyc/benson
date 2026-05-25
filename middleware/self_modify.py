"""Self-awareness + self-modification primitives for Benson.

Four families of tools:

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

  TIER 1 AUTONOMY (applies + commits directly, audited)
    - autofix(rationale, files=[{path, find, replace}]) → for trivial
      comment / docstring / log-string / markdown edits only. Validates
      the diff is ≤5 files, ≤20 lines, no blocklisted paths, no AST
      structural change, then applies + commits + audits + Signal-nudges
      Casey. Anything beyond the trivial class falls through to
      propose_change.
    - autofix_list / autofix_revert → audit-trail surface for the
      /admin/benson dashboard.

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
import sys
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

    # In-flight marker: the meta commit is on the branch BEFORE the SDK
    # runs, so the proposal shows on /admin/proposals immediately. To
    # prevent Casey from merging a meta-only branch while the SDK is
    # still composing the actual fix (the 2026-05-01 chore-recurring
    # incident), drop a marker file and have list_open_proposals + the
    # template hide the merge button while it exists.
    inflight_marker = PROPOSAL_LOG_DIR / f"{branch.replace('/', '_')}.inflight"
    try:
        inflight_marker.write_text(
            f"started={datetime.now(timezone.utc).isoformat()}\nbranch={branch}\n"
        )
    except Exception as e:
        logger.warning(f"could not write inflight marker: {e}")

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
    finally:
        # Clear the in-flight marker so the dashboard re-enables Merge.
        try:
            inflight_marker.unlink(missing_ok=True)
        except Exception:
            pass

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
    Refuses if the proposal's SDK session is still in-flight or if the
    branch contains only the meta commit (no actual code changes).
    """
    if not branch.startswith("proposal/"):
        return {"ok": False, "error": "branch must start with 'proposal/'"}

    # Refuse if the SDK is still composing the fix.
    marker = PROPOSAL_LOG_DIR / f"{branch.replace('/', '_')}.inflight"
    if marker.exists():
        return {
            "ok": False,
            "error": (
                "SDK session is still in-flight for this proposal — the "
                "actual fix code hasn't been committed yet. Wait for the "
                "session to finish (the dashboard refreshes every 30s), "
                "then merge."
            ),
        }

    # Refuse if there are no SDK commits beyond the meta — there's
    # literally nothing to merge except a JSON metadata file. The
    # 2026-05-01 chore-recurring incident merged exactly this and got
    # nothing.
    try:
        log_out = _git(["log", f"main..{branch}", "--oneline"])
        non_meta_commits = [
            ln for ln in log_out.splitlines()
            if ln.strip() and PROPOSAL_META_TAG not in ln
        ]
        if not non_meta_commits:
            return {
                "ok": False,
                "error": (
                    "this proposal has no code commits — only the "
                    "metadata commit is on the branch. The SDK session "
                    "either crashed or is still working. Reject the "
                    "branch and re-prompt Benson if it's stuck."
                ),
            }
    except Exception as e:
        logger.warning(f"apply_proposal could not enumerate commits: {e}")
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

        # In-flight marker: SDK still composing? Hide the merge button on
        # the dashboard until the marker disappears (cleared in the
        # propose_change finally block when SDK exits).
        marker = PROPOSAL_LOG_DIR / f"{branch.replace('/', '_')}.inflight"
        in_flight = marker.exists()

        proposals.append({
            "branch": branch,
            "date": date_s,
            "subject": subject,
            "diffstat": stat_line,
            "commit_count": commit_count,
            "sdk_commit_count": sdk_commit_count,
            "rationale": rationale,
            "instructions": instructions,
            "in_flight": in_flight,
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


# ─── Tier 1 autonomous fix ───────────────────────────────────────────────
import ast as _ast

# Allowed roots for autofix targets. Anything outside these refuses.
_TIER1_ALLOWED_ROOTS = (
    REPO_DIR / "middleware",
    REPO_DIR / "microwakeword" / "scripts",
    REPO_DIR / "scripts",
    REPO_DIR / "docs",
)

# Paths that must NEVER be touched autonomously. Expand as needed; keep
# in code (not config) so the rules ship with the binary.
TIER1_BLOCKLIST: list[str] = [
    "middleware/oauth_agent.py",
    "middleware/main.py",
    "middleware/benson_mcp.py",
    "middleware/benson_prompt.txt",
    "middleware/agent_tools.py",
    "middleware/self_modify.py",
    "middleware/voiceprint.py",
]

# Prefix-based blocks (any path starting with one of these is refused).
_TIER1_BLOCKED_PREFIXES: tuple[str, ...] = (
    "middleware/wyoming_",
    "middleware/scheduled_actions",
    "middleware/sql/",
    "ha/",
    "microwakeword/models/",
)

_TIER1_MAX_FILES = 5
_TIER1_MAX_LINES = 20
_AUTOFIX_SCHEMA_PATH = Path(__file__).parent / "sql" / "autonomous_changes.sql"
_AUTOFIX_LOCK: asyncio.Lock | None = None
_CASEY_SIGNAL_NUMBER = "+15056208470"


def _autofix_lock() -> asyncio.Lock:
    global _AUTOFIX_LOCK
    if _AUTOFIX_LOCK is None:
        _AUTOFIX_LOCK = asyncio.Lock()
    return _AUTOFIX_LOCK


def ensure_autofix_schema() -> None:
    try:
        sql = _AUTOFIX_SCHEMA_PATH.read_text()
    except FileNotFoundError:
        logger.error(f"autonomous_changes schema missing at {_AUTOFIX_SCHEMA_PATH}")
        return
    try:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
    except Exception:
        logger.exception("autonomous_changes: ensure_schema failed")


def _tier1_resolve(path: str) -> tuple[Path | None, str | None]:
    """Resolve a user-supplied path against REPO_DIR. Returns (resolved_path,
    error) — exactly one is None. Refuses absolute paths outside the repo,
    symlink escapes, blocklisted paths, and paths outside allowed roots."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_DIR / path
    try:
        resolved = p.resolve()
    except Exception as e:
        return None, f"cannot resolve {path!r}: {e}"
    repo_resolved = REPO_DIR.resolve()
    try:
        rel = resolved.relative_to(repo_resolved)
    except ValueError:
        return None, f"refusing path outside /opt/benson: {resolved}"
    rel_str = str(rel)
    if rel_str in TIER1_BLOCKLIST:
        return None, f"blocklist: {rel_str}"
    for pfx in _TIER1_BLOCKED_PREFIXES:
        if rel_str.startswith(pfx):
            return None, f"blocklist prefix {pfx!r}: {rel_str}"
    # Must live under one of the allowed roots.
    allowed = False
    for root in _TIER1_ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        return None, (
            f"outside Tier 1 allowed roots (middleware/, microwakeword/scripts/, "
            f"scripts/, docs/): {rel_str}"
        )
    return resolved, None


_LOG_CALL_RE = re.compile(
    r"^\s*(?:logger|log|logging)\.(?:debug|info|warning|error|exception|critical)\s*\("
)
_STRING_LIT_FMT_RE = re.compile(r"%[sdifr]|\{[^{}]*\}")


def _classify_line(line: str) -> str:
    """Return a one-word classification for a source line. Used by the
    pre-edit check to reject anything that isn't pure prose/whitespace
    BEFORE we even consider AST-equivalence. Classes:
      'blank', 'comment', 'docstring_marker', 'in_block',
      'logger_string', 'markdown', 'html_comment', 'jinja_comment',
      'code'.
    The autofix policy only accepts pre/post pairs where BOTH lines are
    non-'code'.
    """
    stripped = line.strip()
    if not stripped:
        return "blank"
    if stripped.startswith("#"):
        return "comment"
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return "docstring_marker"
    if stripped.startswith("{#") or stripped.endswith("#}"):
        return "jinja_comment"
    if stripped.startswith("<!--") or stripped.endswith("-->"):
        return "html_comment"
    if _LOG_CALL_RE.match(line):
        return "logger_string"
    # Bare markdown bullet / heading / paragraph: only safe to classify
    # this way when the file is *.md (caller checks suffix).
    return "code"


def _diffstat(before: str, after: str) -> tuple[int, int]:
    """Return (added, removed) line counts comparing two file contents."""
    import difflib as _difflib
    a = before.splitlines()
    b = after.splitlines()
    added = removed = 0
    for line in _difflib.unified_diff(a, b, lineterm=""):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _ast_dump_stripped(src: str) -> str:
    """Parse `src` and return an AST dump with all string-literal Constant
    values and docstring positions normalized away — so two files that
    differ only in string contents / docstrings / comments produce the
    same dump. Whitespace and comments are already invisible to ast.
    """
    tree = _ast.parse(src)
    # Strip every Constant.value (str/bytes only) so log-string and
    # docstring edits don't show up; numbers/booleans MUST still be
    # compared (they're real semantics).
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Constant) and isinstance(node.value, (str, bytes)):
            node.value = ""
    return _ast.dump(tree, include_attributes=False)


def _validate_line_classes(before: str, after: str, suffix: str) -> str | None:
    """Walk the unified diff and ensure every changed line is non-'code'
    per _classify_line. Returns None on success, or an error string."""
    import difflib as _difflib
    a = before.splitlines()
    b = after.splitlines()
    is_md = suffix.lower() == ".md"
    is_html = suffix.lower() in (".html", ".jinja", ".j2")
    for line in _difflib.unified_diff(a, b, lineterm=""):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if not line.startswith(("+", "-")):
            continue
        body = line[1:]
        cls = _classify_line(body)
        if cls != "code":
            continue
        # Markdown files: any prose line is fair game.
        if is_md:
            continue
        # HTML/Jinja templates: prose between tags is fine; the AST check
        # below doesn't run on these (no .py). Allow.
        if is_html:
            continue
        # Whitespace-only changes (e.g. trailing-space cleanup that produced
        # an empty rstrip on one side) — _classify_line already returns
        # 'blank' for that, so reaching here means it's real code.
        return f"non-trivial change in {suffix} file: {body.rstrip()[:160]!r}"
    return None


async def autofix(rationale: str, files: list[dict]) -> dict:
    """Tier 1 autonomous fix: apply per-file find/replace edits directly,
    commit, audit. Refuses anything that isn't a pure prose / comment /
    log-string / docstring / markdown change. See module docstring for
    the full eligibility rules.

    `files` is a list of {path, find, replace}. `find` must appear
    exactly once in the file. All edits are validated against the same
    rules; if any one fails, NONE are applied.
    """
    if not rationale.strip():
        return {"ok": False, "reason": "rationale is required"}
    if not isinstance(files, list) or not files:
        return {"ok": False, "reason": "files must be a non-empty list"}
    if len(files) > _TIER1_MAX_FILES:
        return {
            "ok": False,
            "reason": f"too many files ({len(files)} > {_TIER1_MAX_FILES})",
        }

    lock = _autofix_lock()
    if lock.locked():
        return {"ok": False, "reason": "another autofix is in flight; retry"}

    async with lock:
        return await asyncio.to_thread(_autofix_sync, rationale, files)


def _autofix_sync(rationale: str, files: list[dict]) -> dict:
    # ─── 1. Resolve + validate inputs, pre-compute before/after blobs ────
    plan: list[dict] = []
    for entry in files:
        if not isinstance(entry, dict):
            return {"ok": False, "reason": f"file entry not an object: {entry!r}"}
        path_in = entry.get("path")
        find = entry.get("find")
        replace = entry.get("replace")
        if not isinstance(path_in, str) or not isinstance(find, str) or not isinstance(replace, str):
            return {
                "ok": False,
                "reason": "each file entry needs string path, find, replace",
            }
        resolved, err = _tier1_resolve(path_in)
        if err:
            return {"ok": False, "reason": err}
        try:
            before = resolved.read_text()
        except Exception as e:
            return {"ok": False, "reason": f"read failed for {path_in}: {e}"}
        occurrences = before.count(find)
        if occurrences == 0:
            return {
                "ok": False,
                "reason": f"find string not present in {resolved.relative_to(REPO_DIR)}",
            }
        if occurrences > 1:
            return {
                "ok": False,
                "reason": (
                    f"find string appears {occurrences} times in "
                    f"{resolved.relative_to(REPO_DIR)} — must be unique"
                ),
            }
        after = before.replace(find, replace, 1)
        if after == before:
            return {
                "ok": False,
                "reason": f"replace equals find for {resolved.relative_to(REPO_DIR)} — no-op",
            }
        plan.append(
            {
                "path": resolved,
                "rel": str(resolved.relative_to(REPO_DIR)),
                "before": before,
                "after": after,
                "suffix": resolved.suffix,
            }
        )

    # ─── 2. Per-file class + AST checks, cumulative diff budget ──────────
    total_added = 0
    total_removed = 0
    for p in plan:
        cls_err = _validate_line_classes(p["before"], p["after"], p["suffix"])
        if cls_err:
            return {"ok": False, "reason": cls_err}
        if p["suffix"] == ".py":
            try:
                before_ast = _ast_dump_stripped(p["before"])
                after_ast = _ast_dump_stripped(p["after"])
            except SyntaxError as e:
                return {"ok": False, "reason": f"syntax error in {p['rel']}: {e}"}
            if before_ast != after_ast:
                return {
                    "ok": False,
                    "reason": f"AST structural change in {p['rel']}",
                }
        added, removed = _diffstat(p["before"], p["after"])
        total_added += added
        total_removed += removed
    if total_added + total_removed > _TIER1_MAX_LINES:
        return {
            "ok": False,
            "reason": (
                f"diff exceeds {_TIER1_MAX_LINES}-line cap "
                f"(+{total_added}/-{total_removed})"
            ),
        }

    # ─── 3. Confirm working tree clean + on main ─────────────────────────
    try:
        status = _git(["status", "--porcelain"])
        head = _git(["branch", "--show-current"]).strip()
    except subprocess.CalledProcessError as e:
        return {"ok": False, "reason": f"git probe failed: {e.output[:200] if e.output else e}"}
    if status.strip():
        return {
            "ok": False,
            "reason": f"working tree dirty; refuse to autofix on top: {status[:200]}",
        }
    if head != "main":
        return {"ok": False, "reason": f"not on main (head={head!r})"}

    # ─── 4. Apply edits to disk ──────────────────────────────────────────
    written: list[Path] = []
    try:
        for p in plan:
            p["path"].write_text(p["after"])
            written.append(p["path"])
    except Exception as e:
        # Best-effort rollback for files we already wrote.
        for p in plan:
            try:
                if p["path"] in written:
                    p["path"].write_text(p["before"])
            except Exception:
                pass
        return {"ok": False, "reason": f"write failed: {e}"}

    # ─── 5. py_compile each touched .py + pytest if tests dir exists ─────
    py_files = [p for p in plan if p["suffix"] == ".py"]
    compile_err: str | None = None
    for p in py_files:
        try:
            subprocess.check_output(
                [sys.executable, "-m", "py_compile", str(p["path"])],
                stderr=subprocess.STDOUT,
                timeout=15,
            )
        except subprocess.CalledProcessError as e:
            compile_err = (
                f"py_compile failed on {p['rel']}: "
                f"{e.output.decode('utf-8', errors='replace')[:300]}"
            )
            break
        except Exception as e:
            compile_err = f"py_compile invocation failed on {p['rel']}: {type(e).__name__}: {e}"
            break

    if compile_err is None:
        tests_dir = REPO_DIR / "middleware" / "tests"
        if tests_dir.exists():
            try:
                subprocess.check_output(
                    [sys.executable, "-m", "pytest", str(tests_dir), "-q"],
                    stderr=subprocess.STDOUT,
                    timeout=120,
                )
            except subprocess.CalledProcessError as e:
                compile_err = (
                    f"pytest failed: "
                    f"{e.output.decode('utf-8', errors='replace')[:300]}"
                )
            except FileNotFoundError:
                pass
            except Exception as e:
                compile_err = f"pytest invocation failed: {type(e).__name__}: {e}"

    if compile_err is not None:
        for p in plan:
            try:
                p["path"].write_text(p["before"])
            except Exception:
                pass
        return {"ok": False, "reason": compile_err}

    # ─── 6. git add + commit ─────────────────────────────────────────────
    try:
        _git(["add", *[p["rel"] for p in plan]])
    except subprocess.CalledProcessError as e:
        for p in plan:
            try:
                p["path"].write_text(p["before"])
            except Exception:
                pass
        return {
            "ok": False,
            "reason": f"git add failed: {e.output[:300] if e.output else e}",
        }

    one_line = rationale.strip().splitlines()[0][:100]
    commit_msg = (
        f"autofix: {one_line}\n\n"
        f"Tier 1 autonomous fix (+{total_added}/-{total_removed}, {len(plan)} file(s)).\n"
        f"Eligibility: comment/docstring/log-string/markdown only; no AST change.\n\n"
        f"Co-Authored-By: Benson Tier-1 Autonomous <benson@fagley.home>\n"
    )
    try:
        _git(["commit", "-m", commit_msg])
        commit_sha = _git(["rev-parse", "HEAD"]).strip()
    except subprocess.CalledProcessError as e:
        for p in plan:
            try:
                p["path"].write_text(p["before"])
            except Exception:
                pass
        try:
            _git(["reset", "HEAD"])
        except Exception:
            pass
        return {
            "ok": False,
            "reason": f"git commit failed: {e.output[:300] if e.output else e}",
        }

    # ─── 7. Audit-row insert ─────────────────────────────────────────────
    try:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO autonomous_changes "
                "(rationale, paths, commit_sha, diff_added, diff_removed) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (
                    rationale.strip(),
                    [p["rel"] for p in plan],
                    commit_sha,
                    total_added,
                    total_removed,
                ),
            )
            audit_id = int(cur.fetchone()[0])
            conn.commit()
    except Exception as e:
        logger.exception("autofix: audit insert failed")
        # Don't roll back the commit — the change is real; just flag.
        audit_id = -1
        logger.warning(f"autofix audit insert failed: {e}")

    # ─── 8. Signal nudge to Casey ────────────────────────────────────────
    try:
        from signal_handler import send_signal_message
        diffstat_str = f"{len(plan)} file{'s' if len(plan) != 1 else ''}, +{total_added}/-{total_removed} lines"
        nudge = (
            f"Benson auto-fixed: {one_line} · {diffstat_str} · "
            f"revert via /admin/benson"
        )
        # send_signal_message is async — fire-and-forget on the running loop.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    send_signal_message(_CASEY_SIGNAL_NUMBER, nudge), loop
                )
            else:
                loop.run_until_complete(send_signal_message(_CASEY_SIGNAL_NUMBER, nudge))
        except RuntimeError:
            asyncio.run(send_signal_message(_CASEY_SIGNAL_NUMBER, nudge))
    except Exception as e:
        logger.warning(f"autofix: Signal nudge failed: {e}")

    return {
        "ok": True,
        "audit_id": audit_id,
        "commit_sha": commit_sha,
        "diffstat": {"added": total_added, "removed": total_removed, "files": len(plan)},
        "paths": [p["rel"] for p in plan],
    }


def autofix_list(limit: int = 20) -> list[dict]:
    """Return the most recent autonomous_changes rows for the dashboard."""
    try:
        rows = _query(
            "SELECT id, created_at, rationale, paths, commit_sha, "
            "diff_added, diff_removed, actor, reverted_at, reverted_by, "
            "revert_commit FROM autonomous_changes "
            "ORDER BY id DESC LIMIT %s",
            (min(int(limit), 200),),
        )
    except Exception as e:
        logger.warning(f"autofix_list failed: {e}")
        return []
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "created_at": r["created_at"].isoformat(timespec="minutes") if r["created_at"] else "",
                "rationale": r["rationale"],
                "paths": list(r["paths"] or []),
                "commit_sha": r["commit_sha"] or "",
                "commit_sha_short": (r["commit_sha"] or "")[:8],
                "diff_added": r["diff_added"],
                "diff_removed": r["diff_removed"],
                "actor": r["actor"] or "",
                "reverted_at": r["reverted_at"].isoformat(timespec="minutes") if r["reverted_at"] else None,
                "reverted_by": r["reverted_by"],
                "revert_commit": r["revert_commit"] or "",
                "revert_commit_short": (r["revert_commit"] or "")[:8] if r["revert_commit"] else "",
            }
        )
    return out


def _autofix_remote_url() -> str | None:
    try:
        url = _git(["remote", "get-url", "origin"]).strip()
    except Exception:
        return None
    # Normalize ssh-style git@github.com:foo/bar.git → https://github.com/foo/bar
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:"):]
        if path.endswith(".git"):
            path = path[:-4]
        return f"https://github.com/{path}"
    if url.startswith("https://github.com/"):
        if url.endswith(".git"):
            url = url[:-4]
        return url
    return None


def autofix_remote_commit_url(sha: str) -> str | None:
    base = _autofix_remote_url()
    if not base or not sha:
        return None
    return f"{base}/commit/{sha}"


def autofix_revert(audit_id: int, reverted_by: str = "casey") -> dict:
    """Revert a Tier 1 autonomous change. Resolves the commit_sha from the
    audit row, `git revert <sha> --no-edit`, runs py_compile on touched
    files, commits, updates the audit row. If the revert conflicts, leaves
    the working tree dirty and returns ok=false with status 409 semantics."""
    try:
        row = _query(
            "SELECT id, commit_sha, paths, reverted_at FROM autonomous_changes "
            "WHERE id = %s",
            (int(audit_id),),
        )
    except Exception as e:
        return {"ok": False, "status": 500, "error": f"audit lookup failed: {e}"}
    if not row:
        return {"ok": False, "status": 404, "error": f"audit id {audit_id} not found"}
    r = row[0]
    if r["reverted_at"]:
        return {
            "ok": False,
            "status": 409,
            "error": f"already reverted at {r['reverted_at'].isoformat()}",
        }
    sha = r["commit_sha"]
    paths = list(r["paths"] or [])

    try:
        status = _git(["status", "--porcelain"])
        head = _git(["branch", "--show-current"]).strip()
    except subprocess.CalledProcessError as e:
        return {
            "ok": False,
            "status": 500,
            "error": f"git probe failed: {e.output[:200] if e.output else e}",
        }
    if status.strip():
        return {
            "ok": False,
            "status": 409,
            "error": f"working tree dirty; cannot revert: {status[:200]}",
        }
    if head != "main":
        return {"ok": False, "status": 409, "error": f"not on main (head={head!r})"}

    try:
        _git(["revert", sha, "--no-edit"])
    except subprocess.CalledProcessError as e:
        out = e.output.decode() if isinstance(e.output, bytes) else str(e.output)
        # Conflicted revert → abort to clean state, leave audit row alone.
        try:
            _git(["revert", "--abort"])
        except Exception:
            pass
        return {
            "ok": False,
            "status": 409,
            "error": f"git revert failed (likely conflict): {out[:300]}",
        }

    # py_compile any .py files that were touched, in their reverted state.
    for rel in paths:
        if not rel.endswith(".py"):
            continue
        fp = REPO_DIR / rel
        if not fp.exists():
            continue
        try:
            subprocess.check_output(
                ["python", "-m", "py_compile", str(fp)],
                stderr=subprocess.STDOUT,
                timeout=15,
            )
        except subprocess.CalledProcessError as e:
            # Compile failure after revert is very surprising — surface it.
            return {
                "ok": False,
                "status": 500,
                "error": (
                    f"py_compile after revert failed on {rel}: "
                    f"{e.output.decode('utf-8', errors='replace')[:300]}"
                ),
            }

    try:
        revert_sha = _git(["rev-parse", "HEAD"]).strip()
    except subprocess.CalledProcessError as e:
        revert_sha = ""
        logger.warning(f"could not read revert sha: {e}")

    try:
        with psycopg2.connect(**PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE autonomous_changes "
                "SET reverted_at = now(), reverted_by = %s, revert_commit = %s "
                "WHERE id = %s",
                (reverted_by, revert_sha, int(audit_id)),
            )
            conn.commit()
    except Exception as e:
        logger.exception("autofix_revert: audit update failed")
        return {
            "ok": True,
            "status": 200,
            "revert_commit": revert_sha,
            "warning": f"revert committed but audit update failed: {e}",
        }

    return {"ok": True, "status": 200, "revert_commit": revert_sha}
