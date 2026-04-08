"""
memory_compact tool - overwrites a memory tier with a compacted version.

This tool is only meaningful when the bootstrap token count exceeds the
configured compaction threshold. At that point the bootstrap string includes
an explicit compaction notice telling the agent which tiers to compact and
in what order. The agent reads each tier with memory_read, produces a compact
version, and calls this tool to write it back.

Unlike memory_write (which is append-only for MEMORY.md), memory_compact
is allowed to overwrite the file entirely. A backup of the workspace is taken
once per session in session.bootstrap() before the agent can call this tool -
not inside this tool - so repeated calls do not produce redundant backups.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from groundmemory.session import MemorySession

# Tiers that may ever be compacted (hard safety list)
_ALLOWED_TIERS = {"MEMORY.md", "USER.md", "AGENTS.md"}

SCHEMA: dict = {
    "name": "memory_compact",
    "description": (
        "Overwrite a memory tier with a compacted version. "
        "Only call this tool when explicitly instructed to do so by the memory context - "
        "it will tell you which tiers to compact and in what order. "
        "Read the tier first with memory_read, produce a tighter version "
        "(merge duplicates, remove outdated entries, summarise verbose sections), "
        "then call this tool with the result. Work one tier at a time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tier": {
                "type": "string",
                "description": (
                    "The memory file to compact. "
                    "Must be one of the tiers listed in the compaction notice."
                ),
                # enum is injected at registration time from config.bootstrap.compaction_tiers
            },
            "content": {
                "type": "string",
                "description": "The compacted replacement content for the file.",
            },
        },
        "required": ["tier", "content"],
    },
}


def run(
    session: "MemorySession",
    tier: str,
    content: str,
) -> dict:
    from groundmemory.tools.base import err, ok
    from groundmemory.core.sync import sync_file

    # 1. Validate tier against the hard safety list
    if tier not in _ALLOWED_TIERS:
        return err(
            f"'{tier}' is not a compactable tier. "
            f"Allowed: {sorted(_ALLOWED_TIERS)}"
        )

    # 2. Validate tier against the configured allow-list
    allowed = set(session.config.bootstrap.compaction_tiers)
    if tier not in allowed:
        return err(
            f"'{tier}' is not in the configured compaction_tiers. "
            f"Configured: {sorted(allowed)}"
        )

    # 3. Content must not be empty
    if not content or not content.strip():
        return err("'content' cannot be empty.")

    # 4. Resolve path
    try:
        target = session.workspace.resolve_file(tier)
    except ValueError as exc:
        return err(str(exc))

    if not target.exists():
        return err(f"File not found: {tier}")

    # 5. Overwrite the file atomically
    from groundmemory.core import storage as _storage
    _storage._atomic_write(target, content.strip() + "\n")

    # 6. Re-index the file so search reflects the new content
    try:
        sync_file(target, session.index, session.provider, session.config.chunking)
    except Exception as exc:  # noqa: BLE001
        return ok({"tier": tier, "chars_written": len(content), "warning": f"sync failed: {exc}"})

    return ok({"tier": tier, "chars_written": len(content)})