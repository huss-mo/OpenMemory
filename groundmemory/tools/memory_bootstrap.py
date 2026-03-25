"""
memory_bootstrap tool - returns the full workspace memory context as a string.

This is always the first tool called at the start of a session. It assembles
MEMORY.md, USER.md, AGENTS.md, RELATIONS.md, and the last two daily logs into
a single formatted Markdown block for injection into the system prompt.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from groundmemory.tools.base import ok

if TYPE_CHECKING:
    from groundmemory.session import MemorySession

SCHEMA = {
    "name": "memory_bootstrap",
    "description": (
        "Load the full memory context for this workspace into the conversation. "
        "Must be called once at the very start of every session before doing anything else. "
        "It assembles MEMORY.md (long-term facts), USER.md (user profile), "
        "AGENTS.md (agent instructions), RELATIONS.md (entity graph), "
        "and the last two daily logs into a single formatted block."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_PLACEHOLDER = "(No memory context yet. Use memory_write to start building your memory.)"


def run(session: "MemorySession") -> dict:
    """Return the workspace memory context."""
    text = session.bootstrap()
    content = text or _PLACEHOLDER
    return ok({"content": content, "chars": len(content)})