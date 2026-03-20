"""
memory_search tool — hybrid semantic + keyword search across all memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from openmemory.tools.base import err, ok

if TYPE_CHECKING:
    from openmemory.session import MemorySession

SCHEMA = {
    "name": "memory_search",
    "description": (
        "Search memory using natural language. Combines semantic (meaning-based) and keyword search. "
        "Use this to recall past decisions, preferences, context, or facts — even if you don't remember "
        "the exact wording. Results include relation context when relevant entities are found."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (default: configured top_k).",
                "minimum": 1,
                "maximum": 20,
            },
            "source": {
                "type": "string",
                "enum": ["long_term", "daily", "relations", "user", "agents"],
                "description": "Restrict search to a specific memory tier. Omit to search all tiers.",
            },
        },
        "required": ["query"],
    },
}


def run(
    session: "MemorySession",
    query: str,
    top_k: Optional[int] = None,
    source: Optional[str] = None,
) -> dict:
    if not query or not query.strip():
        return err("query cannot be empty")

    from openmemory.core.search import hybrid_search

    results = hybrid_search(
        query=query,
        index=session.index,
        provider=session.provider,
        config=session.config.search,
        source_filter=source,
        top_k=top_k,
    )

    return ok({
        "query": query,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    })