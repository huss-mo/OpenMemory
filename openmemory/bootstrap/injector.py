"""
Bootstrap injector — builds the system-prompt string that loads an agent's
long-term memory context at the start of a session.

Design goals
------------
* Respect BootstrapConfig.max_chars_per_file and max_total_chars so
  the injection never blows the context window.
* Warn (with a visible marker) when a file is truncated so the agent knows
  it doesn't have the full picture.
* Optionally include the relation graph and daily logs.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

from openmemory.config import BootstrapConfig
from openmemory.core.workspace import Workspace
from openmemory.core import storage, graph as _graph
from openmemory.core.index import MemoryIndex


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_capped(path: Path, max_chars: int) -> tuple[str, bool]:
    """
    Read *path* and return (text, truncated).

    ``truncated`` is True when the file was longer than ``max_chars``.
    """
    if not path.exists():
        return "", False
    full = storage.read_file(path)
    if len(full) <= max_chars:
        return full, False
    return full[:max_chars], True


def _section(title: str, body: str, truncated: bool = False) -> str:
    """Wrap *body* in a labelled Markdown block."""
    marker = " [TRUNCATED — use memory_get to read the rest]" if truncated else ""
    return f"### {title}{marker}\n\n{body}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_bootstrap_prompt(
    workspace: Workspace,
    cfg: BootstrapConfig,
    index: MemoryIndex | None = None,
) -> str:
    """
    Build the full memory-injection string for the system prompt.

    Parameters
    ----------
    workspace : Workspace
    cfg       : BootstrapConfig  (from OpenMemoryConfig.bootstrap)
    index     : MemoryIndex | None
        Only needed when ``cfg.inject_relations`` is True and you want
        relation data from SQLite rather than only from RELATIONS.md.

    Returns
    -------
    str
        Ready-to-prepend system prompt block (may be empty string).
    """
    sections: list[str] = []
    total_chars = 0

    def _add(title: str, path: Path | None, body: str | None = None) -> bool:
        """Add a section, respecting the total char budget. Returns False if budget exhausted."""
        nonlocal total_chars
        remaining = cfg.max_total_chars - total_chars
        if remaining <= 0:
            return False

        if body is None:
            if path is None or not path.exists():
                return True
            raw, truncated = _read_capped(path, min(cfg.max_chars_per_file, remaining))
            if not raw.strip():
                return True
            body_text = raw
        else:
            truncated = len(body) > cfg.max_chars_per_file
            body_text = body[: cfg.max_chars_per_file] if truncated else body
            if len(body_text) > remaining:
                body_text = body_text[:remaining]
                truncated = True

        if not body_text.strip():
            return True

        sections.append(_section(title, body_text, truncated))
        total_chars += len(body_text)
        return total_chars < cfg.max_total_chars

    # 1. Long-term memory (MEMORY.md)
    if cfg.inject_long_term_memory:
        _add("Long-Term Memory", workspace.memory_file)

    # 2. User profile (USER.md)
    if cfg.inject_user_profile:
        _add("User Profile", workspace.user_file)

    # 3. Agent roster (AGENTS.md)
    if cfg.inject_agents:
        _add("Agent Roster", workspace.agents_file)

    # 4. Relation graph
    if cfg.inject_relations:
        if index is not None:
            relations = _graph.get_relations(index)
            if relations:
                rel_md = _graph.format_relations_for_context(relations)
                _add("Relation Graph", None, body=rel_md)
        else:
            _add("Relation Graph", workspace.relations_file)

    # 5. Daily logs: today + yesterday
    if cfg.inject_daily_logs:
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        for day in (yesterday, today):
            day_path = workspace.daily_file(day)
            label = f"Daily Log ({day.isoformat()})"
            if not _add(label, day_path):
                break  # budget exhausted

    if not sections:
        return ""

    header = (
        "<!-- OpenMemory bootstrap start -->\n"
        "## Your Memory Context\n\n"
        "The following information was loaded from your long-term memory store. "
        "Use it to maintain continuity across sessions.\n\n"
    )
    footer = "\n<!-- OpenMemory bootstrap end -->"
    return header + "\n".join(sections) + footer