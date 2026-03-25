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

from groundmemory.core.workspace import Workspace


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

    Deduplication: if the exact same content body already exists anywhere in
    the file, the write is skipped and the existing entry is reported.  This
    makes the operation idempotent on retries (e.g. from MCP clients that
    retry on apparent errors).

    Returns metadata about the write.
    """
    path = workspace.memory_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = content.strip()
    entry = f"\n## {timestamp}\n\n{body}\n"

    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    # Skip write if this exact content body is already present in the file
    if body and body in existing:
        return {
            "file": str(path.relative_to(workspace.path)),
            "tier": "long_term",
            "timestamp": timestamp,
            "chars_written": 0,
            "deduplicated": True,
        }

    _atomic_write(path, existing + entry)

    return {
        "file": str(path.relative_to(workspace.path)),
        "tier": "long_term",
        "timestamp": timestamp,
        "chars_written": len(entry),
        "deduplicated": False,
    }


def write_user(workspace: Workspace, content: str) -> dict:
    """
    Append *content* to USER.md under a timestamped section header.

    Deduplication: skips write if the exact content body is already present.

    Returns metadata about the write.
    """
    path = workspace.user_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = content.strip()
    entry = f"\n## {timestamp}\n\n{body}\n"

    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    if body and body in existing:
        return {
            "file": str(path.relative_to(workspace.path)),
            "tier": "user",
            "timestamp": timestamp,
            "chars_written": 0,
            "deduplicated": True,
        }

    _atomic_write(path, existing + entry)

    return {
        "file": str(path.relative_to(workspace.path)),
        "tier": "user",
        "timestamp": timestamp,
        "chars_written": len(entry),
        "deduplicated": False,
    }


def write_agents(workspace: Workspace, content: str) -> dict:
    """
    Append *content* to AGENTS.md under a timestamped section header.

    Deduplication: skips write if the exact content body is already present.

    Returns metadata about the write.
    """
    path = workspace.agents_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = content.strip()
    entry = f"\n## {timestamp}\n\n{body}\n"

    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    if body and body in existing:
        return {
            "file": str(path.relative_to(workspace.path)),
            "tier": "agent",
            "timestamp": timestamp,
            "chars_written": 0,
            "deduplicated": True,
        }

    _atomic_write(path, existing + entry)

    return {
        "file": str(path.relative_to(workspace.path)),
        "tier": "agent",
        "timestamp": timestamp,
        "chars_written": len(entry),
        "deduplicated": False,
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
        header = f"# Daily Log - {d}\n"
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


def hard_delete_lines(path: Path, start_line: int, end_line: int) -> dict:
    """
    Physically remove lines [start_line, end_line] from *path* (1-indexed, inclusive).
    No tombstone is left - content is cleanly erased.
    Returns metadata.
    """
    if not path.exists():
        return {"error": f"File not found: {path}"}

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    total = len(lines)

    if start_line < 1 or start_line > total:
        return {"error": f"start_line {start_line} out of range (file has {total} lines)"}
    if end_line < start_line or end_line > total:
        return {"error": f"end_line {end_line} out of range (start={start_line}, total={total})"}

    # Convert to 0-indexed
    s = start_line - 1
    e = end_line  # exclusive upper bound

    removed_preview = "".join(lines[s:e]).strip()
    new_lines = lines[:s] + lines[e:]
    _atomic_write(path, "".join(new_lines))

    return {
        "file": str(path),
        "deleted_lines": f"{start_line}-{end_line}",
        "preview": removed_preview[:80],
    }


def replace_text(path: Path, search: str, replacement: str) -> dict:
    """
    Replace the first occurrence of *search* in *path* with *replacement*.

    Returns metadata including whether a match was found.
    """
    if not path.exists():
        return {"error": f"File not found: {path}"}
    if not search:
        return {"error": "search text cannot be empty"}

    content = path.read_text(encoding="utf-8")
    if search not in content:
        return {"error": f"Search text not found in {path.name}", "found": False}

    new_content = content.replace(search, replacement, 1)
    _atomic_write(path, new_content)

    chars_delta = len(replacement) - len(search)
    return {
        "file": str(path),
        "replaced": True,
        "chars_delta": chars_delta,
    }


def replace_lines(path: Path, start_line: int, end_line: int, replacement: str) -> dict:
    """
    Replace lines [start_line, end_line] in *path* (1-indexed, inclusive) with
    *replacement* text.

    Returns metadata about the replacement.
    """
    if not path.exists():
        return {"error": f"File not found: {path}"}

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    total = len(lines)

    if start_line < 1 or start_line > total:
        return {"error": f"start_line {start_line} out of range (file has {total} lines)"}
    if end_line < start_line or end_line > total:
        return {"error": f"end_line {end_line} out of range (start={start_line}, total={total})"}

    # Convert to 0-indexed
    s = start_line - 1
    e = end_line  # exclusive upper bound for slice

    replaced_preview = "".join(lines[s:e]).strip()
    replacement_block = replacement if replacement.endswith("\n") else replacement + "\n"

    new_lines = lines[:s] + [replacement_block] + lines[e:]
    _atomic_write(path, "".join(new_lines))

    chars_delta = len(replacement_block) - len("".join(lines[s:e]))
    return {
        "file": str(path),
        "replaced_lines": f"{start_line}-{end_line}",
        "replaced_preview": replaced_preview[:80],
        "chars_delta": chars_delta,
    }


def list_daily_files(workspace: Workspace) -> list[str]:
    """Return sorted list of daily log filenames (newest first)."""
    if not workspace.daily_dir.exists():
        return []
    return sorted(
        [f.name for f in workspace.daily_dir.glob("*.md")],
        reverse=True,
    )