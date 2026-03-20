"""
Compaction helpers — detect when the context window is nearly full and
provide prompts that instruct the agent to flush important information
to memory before the session is compacted.
"""
from __future__ import annotations

from openmemory.config import CompactionConfig

# ---------------------------------------------------------------------------
# Flush detection
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are about to run out of context window space. Before the session is \
compacted, you MUST save all important information to long-term memory using \
the memory_write tool. Focus on:

1. Key facts, decisions, and outcomes from this session.
2. Any new relationships between entities (use memory_relate).
3. Updated user preferences or project state.
4. Anything the user would want remembered in a future session.

Write concisely — prefer bullet points. Do NOT include information that is \
already recorded in previous memory entries unless it has changed.\
"""

_DEFAULT_USER_PROMPT = """\
Please save a summary of our conversation so far to memory before we continue. \
Use memory_write to store the key points, and memory_relate for any entity \
relationships you have discovered.\
"""


def should_flush(
    current_tokens: int,
    context_window: int,
    cfg: CompactionConfig,
) -> bool:
    """
    Return True when the agent should flush memory before compaction.

    Parameters
    ----------
    current_tokens  : int   Current token usage in the context window.
    context_window  : int   Total token capacity of the model's context window.
    cfg             : CompactionConfig

    The flush fires when::

        current_tokens >= context_window - cfg.reserve_floor_tokens
          OR
        current_tokens >= cfg.soft_threshold_tokens
    """
    if not cfg.enabled:
        return False

    hard_limit = context_window - cfg.reserve_floor_tokens
    return current_tokens >= min(cfg.soft_threshold_tokens, hard_limit)


def get_compaction_prompts(cfg: CompactionConfig) -> dict[str, str]:
    """
    Return ``{"system": str, "user": str}`` prompts for the pre-compaction flush.

    Values come from the config when set, otherwise fall back to sensible defaults.
    """
    return {
        "system": cfg.system_prompt or _DEFAULT_SYSTEM_PROMPT,
        "user": cfg.user_prompt or _DEFAULT_USER_PROMPT,
    }