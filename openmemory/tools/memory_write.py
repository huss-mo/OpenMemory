"""
memory_write tool — write a memory to long-term (MEMORY.md) or daily log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from openmemory.tools.base import err, ok

if TYPE_CHECKING:
    from openmemory.session import MemorySession

#: JSON schema for OpenAI / Anthropic function calling
SCHEMA = {
    "name": "memory_write",
    "description": (
        "Write a memory to persistent storage. "
        "Use tier='long_term' for facts, decisions, and preferences that should persist across sessions. "
        "Use tier='daily' for running notes, task progress, and session context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to store in memory. Be concise and specific.",
            },
            "tier": {
                "type": "string",
                "enum": ["long_term", "daily"],
                "description": (
                    "'long_term' → written to MEMORY.md (persists forever). "
                    "'daily' → written to today's daily log (date-stamped journal)."
                ),
                "default": "daily",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to categorize this memory (e.g. ['preference', 'project-x']).",
            },
        },
        "required": ["content"],
    },
}


def run(
    session: "MemorySession",
    content: str,
    tier: Literal["long_term", "daily"] = "daily",
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Execute the memory_write tool.

    After writing, immediately syncs the changed file into the search index
    so the content is retrievable in the same session.
    """
    if not content or not content.strip():
        return err("content cannot be empty")

    # Optionally prepend tags as a Markdown label
    body = content.strip()
    if tags:
        tag_str = " ".join(f"#{t}" for t in tags)
        body = f"{tag_str}\n{body}"

    from openmemory.core import storage, sync

    if tier == "long_term":
        result = storage.write_long_term(session.workspace, body)
        sync.sync_file(session.workspace.memory_file, session.index, session.provider, session.config.chunking)
    else:
        result = storage.write_daily(session.workspace, body)
        daily_path = session.workspace.daily_file()
        sync.sync_file(daily_path, session.index, session.provider, session.config.chunking)

    return ok(result)