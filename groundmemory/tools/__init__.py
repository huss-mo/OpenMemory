"""Tool registry - builds config-dependent tool sets for MCP server and Python API."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groundmemory.config import groundmemoryConfig


def build_tool_registry(
    config: "groundmemoryConfig",
) -> tuple[list[tuple[dict, object]], dict[str, object], dict[str, dict]]:
    """
    Build the tool registry based on the supplied config.

    Returns
    -------
    all_tools : list[tuple[schema, run]]
        Ordered list of (schema, runner) pairs for all active tools.
    tool_runners : dict[str, callable]
        Mapping tool_name → runner function.
    tool_schemas : dict[str, dict]
        Mapping tool_name → JSON schema dict.
    """
    from groundmemory.tools import (
        memory_read,
        memory_write,
        memory_bootstrap,
        memory_relate,
        memory_list,
        memory_dispatcher,
        memory_compact,
    )

    if config.dispatcher_mode:
        # Single dispatcher tool only - replaces all individual tools.
        # bootstrap is available via action="bootstrap" inside the dispatcher.
        all_tools: list[tuple[dict, object]] = [
            (memory_dispatcher.SCHEMA, memory_dispatcher.run),
        ]
    else:
        # Core tools
        all_tools = [
            (memory_bootstrap.SCHEMA, memory_bootstrap.run),
            (memory_read.SCHEMA, memory_read.run),
            (memory_write.SCHEMA, memory_write.run),
            (memory_relate.SCHEMA, memory_relate.run),
        ]
        # Optional: memory_list (gated by config)
        if config.expose_memory_list:
            all_tools.append((memory_list.SCHEMA, memory_list.run))
        # memory_compact is always registered - its availability is communicated
        # through the bootstrap notice, not by hiding the tool.
        all_tools.append((memory_compact.SCHEMA, memory_compact.run))

    tool_runners: dict[str, object] = {schema["name"]: run for schema, run in all_tools}
    tool_schemas: dict[str, dict] = {schema["name"]: schema for schema, _ in all_tools}

    return all_tools, tool_runners, tool_schemas


# ---------------------------------------------------------------------------
# Module-level convenience exports (use default config for back-compat).
# Code that does `from groundmemory.tools import TOOL_RUNNERS` still works
# with the default config (no mcp config = both flags False).
# ---------------------------------------------------------------------------

class _LazyRegistry:
    """Lazily builds the registry on first attribute access using default config."""

    def __init__(self) -> None:
        self._all: list | None = None
        self._runners: dict | None = None
        self._schemas: dict | None = None

    def _build(self) -> None:
        if self._all is None:
            from groundmemory.config import groundmemoryConfig
            cfg = groundmemoryConfig()
            self._all, self._runners, self._schemas = build_tool_registry(cfg)

    @property
    def ALL_TOOLS(self) -> list:
        self._build()
        return self._all  # type: ignore[return-value]

    @property
    def TOOL_RUNNERS(self) -> dict:
        self._build()
        return self._runners  # type: ignore[return-value]

    @property
    def TOOL_SCHEMAS(self) -> dict:
        self._build()
        return self._schemas  # type: ignore[return-value]


_registry = _LazyRegistry()


def __getattr__(name: str):
    if name in ("ALL_TOOLS", "TOOL_RUNNERS", "TOOL_SCHEMAS"):
        return getattr(_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_tool_registry", "ALL_TOOLS", "TOOL_RUNNERS", "TOOL_SCHEMAS"]