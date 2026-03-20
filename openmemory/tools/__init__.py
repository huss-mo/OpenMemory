"""Tool registry — maps tool names to (SCHEMA, run) pairs."""
from __future__ import annotations

from openmemory.tools import (
    memory_write,
    memory_search,
    memory_get,
    memory_list,
    memory_delete,
    memory_relate,
)

# Ordered list of all tools exposed by OpenMemory.
ALL_TOOLS: list[tuple[dict, object]] = [
    (memory_write.SCHEMA, memory_write.run),
    (memory_search.SCHEMA, memory_search.run),
    (memory_get.SCHEMA, memory_get.run),
    (memory_list.SCHEMA, memory_list.run),
    (memory_delete.SCHEMA, memory_delete.run),
    (memory_relate.SCHEMA, memory_relate.run),
]

# Convenience lookup: tool_name → run function
TOOL_RUNNERS: dict[str, object] = {schema["name"]: run for schema, run in ALL_TOOLS}

# Convenience lookup: tool_name → schema dict
TOOL_SCHEMAS: dict[str, dict] = {schema["name"]: schema for schema, _ in ALL_TOOLS}

__all__ = ["ALL_TOOLS", "TOOL_RUNNERS", "TOOL_SCHEMAS"]