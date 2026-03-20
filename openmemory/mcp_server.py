"""OpenMemory MCP server — exposes all 6 memory tools over HTTP (streamable-http transport)."""
from __future__ import annotations

import atexit
import json

from mcp.server.fastmcp import FastMCP

from openmemory.config import OpenMemoryConfig

# ---------------------------------------------------------------------------
# Lazy session — created once on first tool call, not at import time.
# This avoids filesystem side-effects when the module is merely imported
# (e.g. during testing or static analysis).
# ---------------------------------------------------------------------------
_session = None


def _get_session():
    global _session
    if _session is None:
        from openmemory.session import MemorySession

        cfg = OpenMemoryConfig.auto()
        _session = MemorySession.create(cfg.workspace, config=cfg)
        atexit.register(_session.close)
    return _session


mcp = FastMCP("OpenMemory")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _unwrap(result: dict) -> str:
    """Return JSON string of result, raising ValueError on tool errors."""
    if not result.get("ok"):
        raise ValueError(result.get("error", "unknown error"))
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_write(
    content: str,
    tier: str = "long_term",
    tags: list[str] | None = None,
) -> str:
    """Write a memory to persistent storage.

    Args:
        content: The text to store.
        tier: Storage tier — "long_term" (MEMORY.md) or "daily" (today's log).
        tags: Optional list of string tags to attach to the memory.
    """
    return _unwrap(_get_session().execute_tool("memory_write", content=content, tier=tier, tags=tags))


@mcp.tool()
def memory_search(
    query: str,
    top_k: int | None = None,
    source: str | None = None,
) -> str:
    """Search stored memories using hybrid semantic + keyword search.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return (uses config default when omitted).
        source: Restrict search to a specific tier: "long_term", "daily",
                "relations", "user", or "agents". Omit to search all tiers.
    """
    kwargs: dict = {"query": query}
    if top_k is not None:
        kwargs["top_k"] = top_k
    if source is not None:
        kwargs["source"] = source
    return _unwrap(_get_session().execute_tool("memory_search", **kwargs))


@mcp.tool()
def memory_get(
    file: str,
    start_line: int = 0,
    end_line: int | None = None,
) -> str:
    """Retrieve a slice of a workspace memory file by line range.

    Args:
        file: Workspace-relative file path (e.g. "MEMORY.md" or "daily/2025-03-20.md").
        start_line: 0-indexed first line to return (inclusive).
        end_line: 0-indexed last line to return (exclusive). Omit to read to end of file.
    """
    kwargs: dict = {"file": file, "start_line": start_line}
    if end_line is not None:
        kwargs["end_line"] = end_line
    return _unwrap(_get_session().execute_tool("memory_get", **kwargs))


@mcp.tool()
def memory_list(
    target: str = "files",
    file: str | None = None,
) -> str:
    """List workspace memory files or preview a specific file.

    Args:
        target: "files" to list all workspace files, or "file" to preview a specific one.
        file: Required when target is "file" — workspace-relative path to preview.
    """
    kwargs: dict = {"target": target}
    if file is not None:
        kwargs["file"] = file
    return _unwrap(_get_session().execute_tool("memory_list", **kwargs))


@mcp.tool()
def memory_delete(
    file: str,
    start_line: int,
    end_line: int,
    reason: str = "deleted by agent",
) -> str:
    """Delete a range of lines from a workspace memory file.

    Args:
        file: Workspace-relative file path containing the memory to delete.
        start_line: 0-indexed first line to delete (inclusive).
        end_line: 0-indexed last line to delete (exclusive).
        reason: Human-readable reason recorded in the tombstone comment.
    """
    return _unwrap(
        _get_session().execute_tool(
            "memory_delete",
            file=file,
            start_line=start_line,
            end_line=end_line,
            reason=reason,
        )
    )


@mcp.tool()
def memory_relate(
    subject: str,
    predicate: str,
    object: str,
    note: str = "",
    source_file: str = "RELATIONS.md",
    confidence: float = 1.0,
) -> str:
    """Record a typed entity relationship (subject → predicate → object).

    Near-duplicate triples are suppressed automatically via cosine similarity
    deduplication before any write occurs.

    Args:
        subject: The source entity (e.g. "Alice").
        predicate: The relationship type (e.g. "works_at").
        object: The target entity (e.g. "Acme Corp").
        note: Optional free-text annotation stored alongside the triple.
        source_file: Workspace-relative file where the triple is appended (default: RELATIONS.md).
        confidence: Confidence score between 0.0 and 1.0 (default: 1.0).
    """
    return _unwrap(
        _get_session().execute_tool(
            "memory_relate",
            subject=subject,
            predicate=predicate,
            object=object,
            note=note,
            source_file=source_file,
            confidence=confidence,
        )
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = OpenMemoryConfig.auto()
    mcp.run(transport="streamable-http", host=cfg.mcp.host, port=cfg.mcp.port)


if __name__ == "__main__":
    main()