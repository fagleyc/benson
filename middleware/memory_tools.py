"""File-based memory backend for Benson.

Replaces the pgvector MemoryStore with simple markdown files Claude
curates itself. Files live under MEMORY_DIR (default /opt/benson/memory).

Five tools the agent uses:
  memory_list()                       — list files with size + first line
  memory_read(path)                   — read one file
  memory_write(path, content)         — create/overwrite
  memory_append(path, content)        — append (one or more lines)
  memory_delete(path)                 — remove a file

Path safety: paths are restricted to MEMORY_DIR. No traversal, no abs paths.
"""
from __future__ import annotations

import os
from pathlib import Path

MEMORY_DIR = Path(os.environ.get("BENSON_MEMORY_DIR", "/opt/benson/memory"))


def _safe_path(path: str) -> Path:
    """Resolve `path` inside MEMORY_DIR. Raises ValueError on traversal."""
    p = Path(path.strip().lstrip("/"))
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"path must be relative inside memory dir: {path}")
    full = (MEMORY_DIR / p).resolve()
    if MEMORY_DIR.resolve() not in full.parents and full != MEMORY_DIR.resolve():
        raise ValueError(f"path escapes memory dir: {path}")
    return full


def memory_list() -> dict:
    """List every memory file with size + first non-empty line as a hint."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    for p in sorted(MEMORY_DIR.rglob("*.md")):
        rel = p.relative_to(MEMORY_DIR).as_posix()
        try:
            text = p.read_text(errors="replace")
        except Exception:
            text = ""
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        files.append({
            "path": rel,
            "bytes": p.stat().st_size,
            "first_line": first[:200],
        })
    return {"ok": True, "memory_dir": str(MEMORY_DIR), "files": files, "count": len(files)}


def memory_read(path: str) -> dict:
    p = _safe_path(path)
    if not p.exists():
        return {"ok": False, "error": f"not found: {path}"}
    return {"ok": True, "path": path, "content": p.read_text(errors="replace")}


def memory_write(path: str, content: str) -> dict:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"ok": True, "path": path, "bytes": p.stat().st_size}


def memory_append(path: str, content: str) -> dict:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = p.read_text(errors="replace") if p.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    p.write_text(existing + content + ("\n" if not content.endswith("\n") else ""))
    return {"ok": True, "path": path, "bytes": p.stat().st_size}


def memory_delete(path: str) -> dict:
    p = _safe_path(path)
    if not p.exists():
        return {"ok": False, "error": f"not found: {path}"}
    p.unlink()
    return {"ok": True, "path": path}
