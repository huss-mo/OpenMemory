"""groundmemory MCP server - exposes memory tools over HTTP (streamable-http transport)."""
from __future__ import annotations

import atexit
import json
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from groundmemory.config import groundmemoryConfig
from groundmemory.tools import (
    memory_bootstrap as _t_bootstrap,
    memory_compact as _t_compact,
    memory_dispatcher as _t_dispatcher,
    memory_list as _t_list,
    memory_read as _t_read,
    memory_relate as _t_relate,
    memory_write as _t_write,
)

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
    kwargs: dict = {"action": action}
    if args is not None:
        kwargs["args"] = args
    return _unwrap(_get_session().execute_tool("memory_tool", **kwargs))


def memory_bootstrap() -> str:
    text = _get_session().bootstrap()
    return text or "(No memory context yet. Use memory_write to start building your memory.)"


def memory_read(
    query: Optional[str] = None,
    file: Optional[str] = None,
    top_k: Optional[int] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
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
    return _unwrap(_get_session().execute_tool("memory_list"))


def memory_compact(tier: str, content: str) -> str:
    return _unwrap(_get_session().execute_tool("memory_compact", tier=tier, content=content))


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
        # memory_compact: only register when compaction is configured.
        # Gated on threshold > 0 so the tool is invisible (no token cost) when
        # compaction is disabled (the default).
        if cfg.bootstrap.compaction_token_threshold > 0:
            import copy
            compact_schema = copy.deepcopy(_t_compact.SCHEMA)
            compact_schema["parameters"]["properties"]["tier"]["enum"] = list(
                cfg.bootstrap.compaction_tiers
            )
            memory_compact.__doc__ = compact_schema["description"]
            mcp.tool()(memory_compact)


# ---------------------------------------------------------------------------
# Pull descriptions from the canonical SCHEMA in each tool module.
# FastMCP uses __doc__ as the MCP tool description, so assigning here means
# the tool modules are the single source of truth - no duplication.
# ---------------------------------------------------------------------------

memory_tool.__doc__       = _t_dispatcher.SCHEMA["description"]
memory_bootstrap.__doc__  = _t_bootstrap.SCHEMA["description"]
memory_read.__doc__       = _t_read.SCHEMA["description"]
memory_write.__doc__      = _t_write.SCHEMA["description"]
memory_relate.__doc__     = _t_relate.SCHEMA["description"]
memory_list.__doc__       = _t_list.SCHEMA["description"]
memory_compact.__doc__    = _t_compact.SCHEMA["description"]

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
# Bearer token authentication middleware
# ---------------------------------------------------------------------------


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces a static bearer token on every request.

    Requests must include the header:
        Authorization: Bearer <api_key>

    Any request with a missing or incorrect token receives a 401 response.
    This middleware is only added to the app when ``mcp.api_key`` is configured.
    """

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._expected = f"Bearer {api_key}"

    async def dispatch(self, request, call_next):
        if request.headers.get("Authorization") != self._expected:
            return Response("Unauthorized", status_code=401)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    from groundmemory.config import _seed_example_config

    _seed_example_config()
    cfg = groundmemoryConfig.auto()
    app = mcp.streamable_http_app()
    if cfg.mcp.api_key:
        app = BearerTokenMiddleware(app, cfg.mcp.api_key)
    uvicorn.run(
        app,
        host=cfg.mcp.host,
        port=cfg.mcp.port,
        forwarded_allow_ips=cfg.mcp.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()