"""groundmemory MCP server - exposes memory tools over HTTP (streamable-http transport)."""
from __future__ import annotations

import atexit
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from groundmemory.config import groundmemoryConfig

# ---------------------------------------------------------------------------
# Lazy session - created once on first tool call, not at import time.
# ---------------------------------------------------------------------------
_session = None


def _get_session():
    global _session
    if _session is None:
        from groundmemory.session import MemorySession

        cfg = groundmemoryConfig.auto()
        _session = MemorySession.create(cfg.workspace, config=cfg)
        atexit.register(_session.close)
    return _session


def _build_mcp() -> FastMCP:
    cfg = groundmemoryConfig.auto()
    allowed_hosts = [h.strip() for h in cfg.mcp.allowed_hosts.split(",") if h.strip()]
    return FastMCP(
        "groundmemory",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
        ),
    )


mcp = _build_mcp()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _unwrap(result: dict) -> str:
    """Return JSON string of result, raising ValueError on tool errors."""
    if result.get("status") != "ok":
        raise ValueError(result.get("message", "unknown error"))
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool functions - defined at module level so tests can import them directly.
#
# Registration (mcp.tool()) is done conditionally inside _register_tools()
# based on config flags, so only the right tools appear in the MCP registry.
# ---------------------------------------------------------------------------


def memory_tool(
    action: str,
    args: Optional[dict] = None,
) -> str:
    """Unified memory tool dispatcher. Pass `action` to select the operation:
      
      bootstrap: Return the full memory bootstrap context (no args needed). Must be called **once at the very start of every session** before doing anything else.
      describe:  Return full schema for an action (args: {"action": "<name>"}). Call this once before invoking action (other than bootstrap) to understand what args it needs.
      read:      Search memory or read a file.
      write:     Append/replace/delete memory.
      relate:    Add an entity relation.
      list:      List all memory files with sizes.

    Args:
        action: Which memory operation to perform.
        args: Arguments for the selected action.
    """
    kwargs: dict = {"action": action}
    if args is not None:
        kwargs["args"] = args
    return _unwrap(_get_session().execute_tool("memory_tool", **kwargs))


def memory_bootstrap() -> str:
    """Load the full memory context for this workspace into the conversation.

    Must be called **once at the very start of every session** before doing
    anything else. It assembles MEMORY.md (long-term facts), USER.md (user
    profile), AGENTS.md (agent instructions), RELATIONS.md (entity graph),
    and the last two daily logs into a single formatted block.

    Returns the Markdown-formatted context, or an empty placeholder when the
    workspace has no memory yet.
    """
    text = _get_session().bootstrap()
    return text or "(No memory context yet. Use memory_write to start building your memory.)"


def memory_read(
    query: Optional[str] = None,
    file: Optional[str] = None,
    top_k: Optional[int] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read from memory. Two modes:

    SEARCH mode - provide `query` to run hybrid semantic+keyword search across
    memory. Optionally supply `top_k` to limit result count, and `file` to
    restrict the search to one memory tier.

    GET mode - provide `file` (no query) to read a memory file directly.
    Optionally supply `start_line` and `end_line` (1-based, inclusive) to
    read a specific line range.

    Files: 'MEMORY.md', 'USER.md', 'AGENTS.md', 'RELATIONS.md',
           'daily' (today), 'daily/YYYY-MM-DD.md' (specific day).

    Args:
        query: Natural-language search query (SEARCH mode).
        file: File path relative to workspace root. In SEARCH mode,
              restricts results to this file's tier. In GET mode,
              reads the file directly (required).
        top_k: Max search results to return (SEARCH mode only).
        start_line: 1-based first line to return (GET mode only).
        end_line: 1-based last line to return, inclusive (GET mode only).
    """
    kwargs: dict = {}
    if query is not None:
        kwargs["query"] = query
    if file is not None:
        kwargs["file"] = file
    if top_k is not None:
        kwargs["top_k"] = top_k
    if start_line is not None:
        kwargs["start_line"] = start_line
    if end_line is not None:
        kwargs["end_line"] = end_line
    return _unwrap(_get_session().execute_tool("memory_read", **kwargs))


def memory_write(
    file: str = "MEMORY.md",
    content: str = "",
    search: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    tags: Optional[list] = None,
) -> str:
    """Write to memory. Four modes selected by the parameters you supply:

    APPEND        - omit start_line/end_line/search; appends `content` to `file`.
    REPLACE_TEXT  - supply `search`; replaces first exact occurrence with `content`.
    REPLACE_LINES - supply start_line + end_line + non-empty content; replaces that range.
    DELETE        - supply start_line + end_line + content=""; hard-deletes those lines.

    Append targets: 'MEMORY.md' (long-term), 'USER.md' (user profile),
                    'AGENTS.md' (agent rules), 'daily' (today's log).
    Edit targets:   'USER.md', 'AGENTS.md', 'RELATIONS.md'.
    MEMORY.md and daily/*.md are append-only (cannot be edited or deleted).

    Before appending to MEMORY.md, USER.md, or AGENTS.md, call memory_read
    with a query to check for near-duplicates first.

    Args:
        file: Target file (see targets above).
        content: Text to write, replacement text, or "" to delete lines.
        search: REPLACE_TEXT mode - exact string to find (first occurrence).
        start_line: 1-based first line for REPLACE_LINES / DELETE modes.
        end_line: 1-based last line (inclusive) for REPLACE_LINES / DELETE modes.
        tags: Optional tags for APPEND to MEMORY.md or daily.
    """
    kwargs: dict = {"file": file, "content": content}
    if search is not None:
        kwargs["search"] = search
    if start_line is not None:
        kwargs["start_line"] = start_line
    if end_line is not None:
        kwargs["end_line"] = end_line
    if tags is not None:
        kwargs["tags"] = tags
    return _unwrap(_get_session().execute_tool("memory_write", **kwargs))


def memory_relate(
    subject: str,
    predicate: str,
    object: str,
    supersedes: bool = False,
    note: str = "",
    source_file: str = "RELATIONS.md",
    confidence: float = 1.0,
) -> str:
    """Record a typed entity relationship: subject - predicate - object.

    Near-duplicate triples are suppressed automatically via cosine similarity
    deduplication before any write occurs.

    Args:
        subject: The source entity (e.g. "Alice").
        predicate: The relationship type (e.g. "works_at").
        object: The target entity (e.g. "Acme Corp").
        supersedes: When True, all existing (subject, predicate) triples are
                    deleted before writing the new one (use when a relation
                    replaces an outdated one, e.g. person changed employer).
        note: Optional free-text annotation stored alongside the triple.
        source_file: Workspace-relative file for the triple (default: RELATIONS.md).
        confidence: Confidence score 0.0-1.0 (default: 1.0).
    """
    return _unwrap(
        _get_session().execute_tool(
            "memory_relate",
            subject=subject,
            predicate=predicate,
            object=object,
            supersedes=supersedes,
            note=note,
            source_file=source_file,
            confidence=confidence,
        )
    )


def memory_list() -> str:
    """List all memory files in the workspace with sizes and line counts."""
    return _unwrap(_get_session().execute_tool("memory_list"))


# ---------------------------------------------------------------------------
# Tool registration - conditional on config
#
# dispatcher_mode=True  -> only memory_tool is registered (1 tool total).
#                          bootstrap is available via action="bootstrap".
# dispatcher_mode=False -> 4 core tools registered, plus memory_list when
#                          expose_memory_list=True.
# ---------------------------------------------------------------------------


def _register_tools(cfg: groundmemoryConfig) -> None:
    """Register MCP tools based on config flags.

    dispatcher_mode and expose_memory_list are top-level groundmemoryConfig
    fields - not MCP-specific - so they apply to both the MCP server and the
    Python API.
    """
    if cfg.dispatcher_mode:
        # Dispatcher mode: single tool replaces all four core tools.
        mcp.tool()(memory_tool)
    else:
        # Normal mode: four core tools.
        mcp.tool()(memory_bootstrap)
        mcp.tool()(memory_read)
        mcp.tool()(memory_write)
        mcp.tool()(memory_relate)
        # Optional: memory_list gated by config.
        if cfg.expose_memory_list:
            mcp.tool()(memory_list)


# Register all tools at module load (config is read once)
_register_tools(groundmemoryConfig.auto())


# ---------------------------------------------------------------------------
# Bootstrap prompt (MCP Prompts primitive)
# ---------------------------------------------------------------------------


@mcp.prompt()
def memory_bootstrap_prompt() -> str:
    """Return the workspace memory context as an MCP Prompt.

    MCP clients that support the Prompts primitive (e.g. Cline, Claude Desktop)
    can invoke this prompt at session start to inject the full memory context
    without requiring the agent to call a tool first.

    Returns the same content as the memory_bootstrap tool.
    """
    text = _get_session().bootstrap()
    return text or "(No memory context yet. Use memory_write to start building your memory.)"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    from groundmemory.config import _seed_example_config

    _seed_example_config()
    cfg = groundmemoryConfig.auto()
    app = mcp.streamable_http_app()
    uvicorn.run(
        app,
        host=cfg.mcp.host,
        port=cfg.mcp.port,
        forwarded_allow_ips=cfg.mcp.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()