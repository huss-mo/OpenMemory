"""
memory_read tool - unified search + file-read tool.

Dispatch logic:
  - query provided            → hybrid semantic+keyword search
  - file provided, no query   → read file content (optionally a line range)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from groundmemory.tools.base import err, ok

if TYPE_CHECKING:
    from groundmemory.session import MemorySession

# Filename → source filter name used by hybrid_search
_FILE_TO_SOURCE: dict[str, str] = {
    "MEMORY.md": "long_term",
    "USER.md": "user",
    "AGENTS.md": "agents",
    "RELATIONS.md": "relations",
    "daily": "daily",
}


def _file_to_source(file: str) -> Optional[str]:
    """Map a filename/prefix to a search source filter, or None if unrecognised."""
    name = file.strip().upper()
    # Exact match for top-level files
    for fname, src in _FILE_TO_SOURCE.items():
        if name == fname.upper():
            return src
    # daily/YYYY-MM-DD.md → "daily"
    if name.startswith("DAILY"):
        return "daily"
    return None


SCHEMA = {
    "name": "memory_read",
    "description": (
        "Read from memory. Two modes:\n\n"
        "  SEARCH mode - provide `query` to run hybrid semantic+keyword search across memory.\n"
        "    Optional: `top_k` to limit result count; `file` to restrict search to one tier.\n\n"
        "  GET mode - provide `file` (no query) to read a memory file directly.\n"
        "    Optional: `start_line` and `end_line` (1-based, inclusive) to read a line range.\n\n"
        "Files: 'MEMORY.md' (long-term), 'USER.md' (user facts), 'AGENTS.md' (agent rules), "
        "'RELATIONS.md' (entity graph), 'daily' (today's log), 'daily/YYYY-MM-DD.md' (specific day)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language search query (SEARCH mode). "
                    "Omit to use GET mode."
                ),
            },
            "file": {
                "type": "string",
                "description": (
                    "File path relative to workspace root, e.g. 'MEMORY.md', 'USER.md', "
                    "'AGENTS.md', 'RELATIONS.md', 'daily', 'daily/2026-03-20.md'. "
                    "In SEARCH mode, restricts results to this file's tier. "
                    "In GET mode (no query), this is required - it selects the file to read."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Max search results to return (SEARCH mode only, default: configured top_k).",
                "minimum": 1,
                "maximum": 20,
            },
            "start_line": {
                "type": "integer",
                "description": "1-based first line to return (GET mode only, inclusive). Omit to read from the beginning.",
                "minimum": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "1-based last line to return (GET mode only, inclusive). Omit to read to end of file.",
                "minimum": 1,
            },
        },
        "required": [],
    },
}


def run(
    session: "MemorySession",
    query: Optional[str] = None,
    file: Optional[str] = None,
    top_k: Optional[int] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> dict:
    if query and query.strip():
        return _run_search(session, query.strip(), file, top_k)
    elif file:
        return _run_get(session, file, start_line, end_line)
    else:
        return err("Provide either 'query' (search mode) or 'file' (get mode).")


def _run_search(
    session: "MemorySession",
    query: str,
    file: Optional[str],
    top_k: Optional[int],
) -> dict:
    from groundmemory.core.search import hybrid_search

    source_filter: Optional[str] = None
    if file:
        source_filter = _file_to_source(file)

    results = hybrid_search(
        query=query,
        index=session.index,
        provider=session.provider,
        config=session.config.search,
        source_filter=source_filter,
        top_k=top_k,
    )

    return ok({
        "mode": "search",
        "query": query,
        "source": source_filter,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    })


def _run_get(
    session: "MemorySession",
    file: str,
    start_line: Optional[int],
    end_line: Optional[int],
) -> dict:
    from groundmemory.core.storage import read_file

    # Resolve "daily" shorthand to today's daily log
    resolved_file = file
    if file.strip().lower() == "daily":
        from datetime import date
        resolved_file = f"daily/{date.today().isoformat()}.md"

    path = session.workspace.resolve_file(resolved_file)

    if not path.exists():
        return err(f"File not found: {resolved_file}")

    # Read full file first to get total_lines
    full_content = read_file(path)
    all_lines = full_content.splitlines()
    total_lines = len(all_lines)

    # Convert 1-indexed inclusive → 0-indexed for storage.read_file
    sl = (start_line - 1) if start_line is not None else 0
    el = end_line if end_line is not None else None

    content = read_file(path, start_line=sl, end_line=el)

    return ok({
        "mode": "get",
        "file": resolved_file,
        "start_line": start_line,
        "end_line": end_line,
        "exists": True,
        "content": content,
        "chars": len(content),
        "total_lines": total_lines,
    })
