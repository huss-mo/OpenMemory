"""
memory_get tool — read a specific memory file or line range.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from openmemory.tools.base import err, ok

if TYPE_CHECKING:
    from openmemory.session import MemorySession

SCHEMA = {
    "name": "memory_get",
    "description": (
        "Read a specific memory file or a range of lines from it. "
        "Use 'MEMORY.md' for long-term memory, 'daily/YYYY-MM-DD.md' for a daily log, "
        "or 'RELATIONS.md' for the entity relation graph. "
        "Returns empty string (not an error) if the file doesn't exist yet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "File path relative to workspace root. E.g. 'MEMORY.md', 'daily/2026-03-20.md'.",
            },
            "start_line": {
                "type": "integer",
                "description": "0-indexed start line (inclusive). Omit to read from the beginning.",
                "minimum": 0,
            },
            "end_line": {
                "type": "integer",
                "description": "0-indexed end line (exclusive). Omit to read to the end of file.",
                "minimum": 1,
            },
        },
        "required": ["file"],
    },
}


def run(
    session: "MemorySession",
    file: str,
    start_line: int = 0,
    end_line: Optional[int] = None,
) -> dict:
    from openmemory.core.storage import read_file

    path = session.workspace.resolve_file(file)
    content = read_file(path, start_line=start_line, end_line=end_line)

    return ok({
        "file": file,
        "start_line": start_line,
        "end_line": end_line,
        "exists": path.exists(),
        "content": content,
        "chars": len(content),
    })