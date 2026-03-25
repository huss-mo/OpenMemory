"""
Base class and shared utilities for all groundmemory tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MemoryToolError(Exception):
    """Raised when a tool call fails with a user-visible error."""


def ok(data: Any) -> dict:
    """Wrap a successful tool result."""
    return {"status": "ok", **data} if isinstance(data, dict) else {"status": "ok", "result": data}


def err(message: str) -> dict:
    """Wrap a tool error result."""
    return {"status": "error", "message": message}


def is_immutable(file: str) -> bool:
    """
    Return True if *file* refers to an append-only immutable memory tier.

    MEMORY.md and daily/*.md are write-once history files - their existing
    content must never be mutated or deleted by the agent.  Only USER.md,
    AGENTS.md, and any other files are editable.
    """
    p = Path(file)
    # Normalise: strip leading separators so both "daily/x.md" and
    # "/abs/path/.../daily/x.md" are caught.
    parts = p.parts
    name = p.name
    return name == "MEMORY.md" or (len(parts) >= 2 and parts[-2] == "daily")


_IMMUTABLE_MSG = (
    "'{file}' is an append-only memory file and cannot be edited or deleted. "
    "Use memory_write to append new information instead."
)


def sync_after_edit(
    session,
    resolved: Path,
    is_relations: bool,
    base_payload: dict,
) -> dict:
    """
    Re-index *resolved* after an in-place edit and return ``ok(base_payload)``.

    * Calls ``sync_file`` to make the updated content immediately searchable.
    * When *is_relations* is True, also calls ``sync_relations_from_file`` and
      appends ``relations_format``, ``format_reminder``, and (if non-empty)
      ``relations_synced`` keys to the payload.
    * On sync failure the exception is swallowed and a ``warning`` key is added
      instead of propagating (the file edit already succeeded).
    """
    from groundmemory.core.sync import sync_file
    from groundmemory.core.relations import sync_relations_from_file, RELATIONS_FORMAT_REMINDER

    relation_sync_result = None
    try:
        sync_file(resolved, session.index, session.provider, session.config.chunking)
        if is_relations:
            relation_sync_result = sync_relations_from_file(resolved, session.index)
    except Exception as exc:  # noqa: BLE001
        base_payload["warning"] = f"Index sync failed: {exc}"
        if is_relations:
            base_payload["relations_format"] = "confirmed"
            base_payload["format_reminder"] = RELATIONS_FORMAT_REMINDER
        return ok(base_payload)

    if is_relations:
        base_payload["relations_format"] = "confirmed"
        base_payload["format_reminder"] = RELATIONS_FORMAT_REMINDER
        if relation_sync_result:
            base_payload["relations_synced"] = relation_sync_result
    return ok(base_payload)
