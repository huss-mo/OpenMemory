"""
Low-level Markdown file read/write operations.
All writes are atomic (write to temp → rename) to prevent partial writes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from openmemory.core.workspace import Workspace


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically using a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def read_file(path: Path, start_line: int = 0, end_line: Optional[int] = None) -> str:
    """
    Read the file at *path*, optionally slicing to [start_line, end_line).
    Returns empty string if the file does not exist (graceful degradation).
    Lines are 0-indexed.
    """
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if start_line == 0 and end_line is None:
        return text
    lines = text.splitlines(keepends=True)
    sliced = lines[start_line:end_line]
    return "".join(sliced)


def file_hash(path: Path) -> str:
    """Return SHA-256 of the file's content, or empty string if missing."""
    if not path.exists():
        return ""
    return sha256(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def write_long_term(workspace: Workspace, content: str) -> dict:
    """
    Append *content* to MEMORY.md under a timestamped section header.
    Returns metadata about the write.
    """
    path = workspace.memory_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {timestamp}\n\n{content.strip()}\n"

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write(path, existing + entry)

    return {
        "file": str(path.relative_to(workspace.path)),
        "tier": "long_term",
        "timestamp": timestamp,
        "chars_written": len(entry),
    }


def write_daily(workspace: Workspace, content: str, day: Optional[date] = None) -> dict:
    """
    Append *content* to today's (or *day*'s) daily log.
    The daily log is append-only; entries are separated by timestamped headers.
    Returns metadata about the write.
    """
    path = workspace.daily_file(day)
    timestamp = datetime.now().strftime("%H:%M")
    d = (day or date.today()).isoformat()

    if not path.exists():
        header = f"# Daily Log — {d}\n"
        entry = f"\n{header}\n## {timestamp}\n\n{content.strip()}\n"
    else:
        entry = f"\n## {timestamp}\n\n{content.strip()}\n"

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_write(path, existing + entry)

    return {
        "file": f"daily/{d}.md",
        "tier": "daily",
        "timestamp": f"{d} {timestamp}",
        "chars_written": len(entry),
    }


def delete_lines(path: Path, start_line: int, end_line: int) -> dict:
    """
    Remove lines [start_line, end_line) from *path* (0-indexed).
    Appends an audit comment instead of silently erasing.
    Returns metadata.
    """
    if not path.exists():
        return {"error": f"File not found: {path}"}

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    removed = "".join(lines[start_line:end_line]).strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    tombstone = f"\n<!-- deleted {timestamp}: {removed[:80]}{'...' if len(removed)>80 else ''} -->\n"

    new_lines = lines[:start_line] + [tombstone] + lines[end_line:]
    _atomic_write(path, "".join(new_lines))

    return {
        "file": str(path),
        "deleted_lines": f"{start_line}-{end_line}",
        "preview": removed[:80],
    }


def list_daily_files(workspace: Workspace) -> list[str]:
    """Return sorted list of daily log filenames (newest first)."""
    if not workspace.daily_dir.exists():
        return []
    return sorted(
        [f.name for f in workspace.daily_dir.glob("*.md")],
        reverse=True,
    )