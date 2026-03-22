"""OpenMemory MCP server — exposes all memory tools over HTTP (streamable-http transport)."""
from __future__ import annotations

import atexit
import json

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

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
        # cfg.workspace is the logical workspace name (e.g. "default").
        # MemorySession.create() resolves the full path as cfg.root_dir / workspace_name,
        # so the data directory is always a single level: ~/.openmemory/<workspace>.
        atexit.register(_session.close)
    return _session


# DNS rebinding protection (added in mcp 1.24) rejects any Host header that
# isn't localhost/127.0.0.1, which breaks LAN / Docker / remote access where
# the client connects via an IP address or hostname.  We disable it here
# because OpenMemory is a self-hosted, single-user service - callers must
# already have network access to reach port 4242, so the DNS rebinding attack
# vector does not apply.  If you expose the server to untrusted networks,
# re-enable this and add your allowed hosts to TransportSecuritySettings
# (allowed_hosts=["your-host:4242"]) instead.
mcp = FastMCP(
    "OpenMemory",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _unwrap(result: dict) -> str:
    """Return JSON string of result, raising ValueError on tool errors."""
    if result.get("status") != "ok":
        raise ValueError(result.get("message", "unknown error"))
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
    """Write a memory to persistent storage. Choose the tier carefully:

    - "long_term" → MEMORY.md. Curated facts, decisions, and preferences that
      should persist indefinitely. Use for general knowledge about the world,
      project decisions, or anything that doesn't fit a narrower tier.
    - "daily"     → daily/YYYY-MM-DD.md. Running session notes, task progress,
      and short-lived context. Auto-dated; not carried across days.
    - "user"      → USER.md. Stable facts *about the user*: their name, role,
      organisation, location, communication style, and stated preferences.
      Use this whenever the user reveals something about themselves.
    - "agent"     → AGENTS.md. Behavioural rules or instructions that should
      govern *how the agent operates* in future sessions (e.g. "always search
      memory before answering", "prefer bullet points over prose").

    Before writing to "long_term", "user", or "agent" tiers, call memory_search
    with top_k=1 to check whether a closely related fact is already stored in
    that tier. If a near-duplicate exists, prefer memory_replace_text or
    memory_replace_lines to update the existing entry rather than appending a
    new one.

    Args:
        content: The text to store. Be concise and specific.
        tier: One of "long_term", "daily", "user", or "agent" (see above).
        tags: Optional list of string tags (applied to long_term and daily only).
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
    """Delete a range of lines from a mutable memory file (tombstone-style).

    MEMORY.md and daily/*.md are append-only history and cannot be modified.
    Only USER.md, AGENTS.md, and RELATIONS.md are editable.

    The deleted content is replaced by an HTML audit comment so the change
    is always reversible by a human.

    Args:
        file: Workspace-relative path of the mutable file to edit (e.g. "USER.md").
        start_line: 1-based line number of the first line to delete.
        end_line: 1-based line number of the last line to delete (inclusive).
                  Pass the same value as start_line to delete a single line.
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
    supersedes: bool = False,
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
        supersedes: When True, all existing (subject, predicate) triples are
                    deleted from storage before the new triple is written.
                    Use this when a relation replaces an outdated one
                    (e.g. a person changed employer or city).
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
            supersedes=supersedes,
            note=note,
            source_file=source_file,
            confidence=confidence,
        )
    )


@mcp.tool()
def memory_replace_text(
    file: str,
    search: str,
    replacement: str,
) -> str:
    """Replace the first exact-string match in a mutable memory file.

    MEMORY.md and daily/*.md are append-only history and cannot be modified.
    Only USER.md, AGENTS.md, and RELATIONS.md are editable.

    Use memory_get first to read the file and confirm the exact text to
    replace (including whitespace and newlines).

    Args:
        file: Workspace-relative path of the mutable file to edit (e.g. "USER.md").
        search: Exact string to search for. Must match character-for-character.
                Only the first occurrence is replaced.
        replacement: Text to substitute in place of the matched string.
    """
    return _unwrap(
        _get_session().execute_tool(
            "memory_replace_text",
            file=file,
            search=search,
            replacement=replacement,
        )
    )


@mcp.tool()
def memory_replace_lines(
    file: str,
    start_line: int,
    end_line: int,
    replacement: str,
) -> str:
    """Replace a line range in a mutable memory file with new text.

    MEMORY.md and daily/*.md are append-only history and cannot be modified.
    Only USER.md, AGENTS.md, and RELATIONS.md are editable.

    Use memory_get first to identify the target line numbers.

    Args:
        file: Workspace-relative path of the mutable file to edit (e.g. "USER.md").
        start_line: 1-based line number of the first line to replace.
        end_line: 1-based line number of the last line to replace (inclusive).
                  Pass the same value as start_line to replace a single line.
        replacement: Text that replaces the specified line range.
    """
    return _unwrap(
        _get_session().execute_tool(
            "memory_replace_lines",
            file=file,
            start_line=start_line,
            end_line=end_line,
            replacement=replacement,
        )
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@mcp.tool()
def memory_bootstrap() -> str:
    """Load the full memory context for this workspace into the conversation.

    Call this tool **once at the very start of every session** before doing
    anything else.  It assembles the contents of MEMORY.md (long-term facts),
    USER.md (user profile), AGENTS.md (agent instructions), RELATIONS.md
    (entity graph), and the last two daily logs into a single formatted
    string that you should treat as persistent background context for the
    rest of the session.

    This is the MCP equivalent of the Python-API ``session.bootstrap()``
    call.  MCP clients that do not support the Prompts primitive (e.g. n8n,
    custom agents) should call this tool instead of - or in addition to -
    the ``memory_bootstrap`` prompt.

    Returns:
        A Markdown-formatted system-prompt block, or an empty string if the
        workspace has no memory files yet.
    """
    bootstrap_text = _get_session().bootstrap()
    # Return a plain string; _unwrap is not used here because bootstrap never
    # fails - an empty workspace simply returns an empty string.
    return bootstrap_text or "(No memory context yet. Use memory_write to start building your memory.)"


@mcp.prompt()
def memory_bootstrap_prompt() -> str:
    """Return the workspace memory context as an MCP Prompt.

    This is the Prompts-primitive counterpart of the ``memory_bootstrap``
    tool.  MCP clients that support the Prompts primitive (e.g. Cline,
    Claude Desktop) can invoke this prompt at session start from their UI
    to inject the full memory context into the conversation without
    requiring the agent to call a tool first.

    The content is identical to what ``memory_bootstrap`` returns - it
    assembles MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, and the last
    two daily logs into a single formatted block, respecting the character
    budgets configured in ``BootstrapConfig``.

    Returns:
        A Markdown-formatted memory context block ready to be prepended to
        the system prompt, or a placeholder if the workspace is empty.
    """
    bootstrap_text = _get_session().bootstrap()
    return bootstrap_text or "(No memory context yet. Use memory_write to start building your memory.)"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    from openmemory.config import _seed_example_config

    _seed_example_config()
    cfg = OpenMemoryConfig.auto()
    app = mcp.streamable_http_app()
    uvicorn.run(app, host=cfg.mcp.host, port=cfg.mcp.port, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()