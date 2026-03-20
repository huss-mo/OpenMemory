"""memory_list tool — list memory files or daily log entries."""
from __future__ import annotations

from openmemory.tools.base import ok, err
from openmemory.core import storage

SCHEMA = {
    "name": "memory_list",
    "description": (
        "List available memory files or the entries in a specific file. "
        "Use this to discover what memory exists before reading or deleting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["files", "daily"],
                "description": (
                    "'files' lists all memory files in the workspace "
                    "(MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, daily logs). "
                    "'daily' lists all daily log file names."
                ),
                "default": "files",
            },
            "file": {
                "type": "string",
                "description": (
                    "Optional. When set, return a summary (line count + first 20 lines) "
                    "of this specific file instead of listing files. "
                    "Relative paths like 'MEMORY.md' or 'daily/2025-01-01.md' are accepted."
                ),
            },
        },
        "required": [],
    },
}


def run(session, target: str = "files", file: str | None = None) -> dict:
    """
    Parameters
    ----------
    session : MemorySession
    target  : 'files' | 'daily'
    file    : optional specific file to peek at
    """
    ws = session.workspace

    # --- peek at a specific file ---
    if file:
        try:
            resolved = ws.resolve_file(file)
        except FileNotFoundError:
            return err(f"File not found: {file}")
        lines = storage.read_file(resolved)
        return ok(
            {
                "file": file,
                "line_count": len(lines),
                "preview": lines[:20],
            }
        )

    # --- list daily files ---
    if target == "daily":
        daily_files = storage.list_daily_files(ws.daily_dir)
        return ok(
            {
                "daily_files": [str(p.name) for p in daily_files],
                "count": len(daily_files),
            }
        )

    # --- list all memory files ---
    all_files = ws.all_memory_files()
    entries = []
    for path in all_files:
        try:
            lines = storage.read_file(path)
            rel = path.relative_to(ws.workspace_path)
        except Exception:
            rel = path
            lines = []
        entries.append(
            {
                "file": str(rel).replace("\\", "/"),
                "line_count": len(lines),
            }
        )

    return ok({"files": entries, "count": len(entries)})