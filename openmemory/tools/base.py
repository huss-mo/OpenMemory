"""
Base class and shared utilities for all OpenMemory tools.
"""

from __future__ import annotations

from typing import Any


class MemoryToolError(Exception):
    """Raised when a tool call fails with a user-visible error."""


def ok(data: Any) -> dict:
    """Wrap a successful tool result."""
    return {"status": "ok", **data} if isinstance(data, dict) else {"status": "ok", "result": data}


def err(message: str) -> dict:
    """Wrap a tool error result."""
    return {"status": "error", "message": message}