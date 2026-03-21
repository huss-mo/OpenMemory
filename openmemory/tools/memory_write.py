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
        "Write a memory to persistent storage. Choose the tier carefully:\n"
        "  'long_term' — curated facts, decisions, and preferences that should persist forever (MEMORY.md).\n"
        "  'daily'     — running notes, task progress, and session context (daily log, date-stamped).\n"
        "  'user'      — stable facts about the user: name, role, location, preferences (USER.md).\n"
        "  'agent'     — instructions or rules that should govern future agent behaviour (AGENTS.md).\n\n"
        "Before writing to 'long_term', 'user', or 'agent' tiers, call memory_search with top_k=1 "
        "to check whether a closely related fact is already stored in that tier. "
        "If a near-duplicate exists, prefer memory_replace_text or memory_replace_lines to update "
        "the existing entry rather than appending a new one."
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
                "enum": ["long_term", "daily", "user", "agent"],
                "description": (
                    "'long_term' → MEMORY.md — curated facts and decisions.\n"
                    "'daily'     → daily/YYYY-MM-DD.md — session notes and task progress.\n"
                    "'user'      → USER.md — who the user is (name, role, preferences).\n"
                    "'agent'     → AGENTS.md — behavioural rules for future agent sessions."
                ),
                "default": "long_term",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags to categorize this memory (e.g. ['preference', 'project-x']). Not applied to 'user' or 'agent' tiers.",
            },
        },
        "required": ["content"],
    },
}


def run(
    session: "MemorySession",
    content: str,
    tier: Literal["long_term", "daily", "user", "agent"] = "long_term",
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Execute the memory_write tool.

    After writing, immediately syncs the changed file into the search index
    so the content is retrievable in the same session.
    """
    if not content or not content.strip():
        return err("content cannot be empty")

    from openmemory.core import storage, sync

    if tier in ("user", "agent"):
        # USER.md and AGENTS.md store structured facts — tags are not applied
        # to keep the files clean and readable.
        body = content.strip()
        if tier == "user":
            result = storage.write_user(session.workspace, body)
            sync.sync_file(session.workspace.user_file, session.index, session.provider, session.config.chunking)
        else:
            result = storage.write_agents(session.workspace, body)
            sync.sync_file(session.workspace.agents_file, session.index, session.provider, session.config.chunking)
    else:
        # Optionally prepend tags as a Markdown label for long_term / daily
        body = content.strip()
        if tags:
            tag_str = " ".join(f"#{t}" for t in tags)
            body = f"{tag_str}\n{body}"

        if tier == "long_term":
            result = storage.write_long_term(session.workspace, body)
            sync.sync_file(session.workspace.memory_file, session.index, session.provider, session.config.chunking)
        else:
            result = storage.write_daily(session.workspace, body)
            daily_path = session.workspace.daily_file()
            sync.sync_file(daily_path, session.index, session.provider, session.config.chunking)

    return ok(result)
