"""
Token counting utilities for bootstrap context measurement.

Two counters are supported:
  "approx"   - fast approximation: len(text) // 4  (no extra deps)
  "tiktoken" - accurate BPE tokenisation via the tiktoken library
               (requires: pip install tiktoken; falls back to approx if not installed)
"""
from __future__ import annotations

from typing import Literal


def count_tokens(text: str, method: Literal["approx", "tiktoken"] = "approx") -> int:
    """
    Return the approximate token count for *text*.

    Parameters
    ----------
    text   : str
    method : "approx" | "tiktoken"
        When ``"tiktoken"`` is requested but the library is not installed,
        falls back to ``"approx"`` silently.

    Returns
    -------
    int  estimated token count
    """
    if method == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass  # fall through to approx
    return max(1, len(text) // 4)