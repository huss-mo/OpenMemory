"""
Bootstrap injector - builds the system-prompt string that loads an agent's
long-term memory context at the start of a session.

Design goals
------------
* Respect BootstrapConfig.max_chars_per_file and max_total_chars so
  the injection never blows the context window.
* Warn (with a visible marker) when a file is truncated so the agent knows
  it doesn't have the full picture.
* Optionally include the relation graph and daily logs.
* When the token count of the assembled prompt exceeds
  BootstrapConfig.compaction_token_threshold, inject a compaction notice
  block so the agent knows to compact memory using the memory_compact tool.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

from groundmemory.config import BootstrapConfig
from groundmemory.core.workspace import Workspace, _AGENT_TOOLS_DISPATCHER_MD, _COMPACTION_NOTICE_TEMPLATE
from groundmemory.core import storage, relations as _graph
from groundmemory.core.index import MemoryIndex


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


def _section(title: str, body: str, truncated: bool = False, source: str = "") -> str:
    """Wrap *body* in a labelled Markdown block."""
    marker = " [TRUNCATED - use memory_get to read the rest]" if truncated else ""
    source_block = f" ({source})" if source else ""
    return f"### {title}{marker}{source_block}\n\n{body}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_bootstrap_prompt(
    workspace: Workspace,
    cfg: BootstrapConfig,
    index: MemoryIndex | None = None,
    dispatcher_mode: bool = False,
    inject_compaction_notice: bool = False,
) -> str:
    """
    Build the full memory-injection string for the system prompt.

    Parameters
    ----------
    workspace    : Workspace
    cfg          : BootstrapConfig  (from groundmemoryConfig.bootstrap)
    index        : MemoryIndex | None
        Only needed when ``cfg.inject_relations`` is True and you want
        relation data from SQLite rather than only from RELATIONS.md.
    dispatcher_mode : bool
        When True, inject full tool usage instructions for the dispatcher.
    inject_compaction_notice : bool
        When True, append the compaction notice to the bootstrap string.

    Returns
    -------
    str
        Ready-to-prepend system prompt block (may be empty string).
    """
    sections: list[str] = []
    total_chars = 0

    def _add(title: str, path: Path | None, body: str | None = None, source: str = "") -> bool:
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
            if not source:
                source = path.name
        else:
            truncated = len(body) > cfg.max_chars_per_file
            body_text = body[: cfg.max_chars_per_file] if truncated else body
            if len(body_text) > remaining:
                body_text = body_text[:remaining]
                truncated = True

        if not body_text.strip():
            return True

        sections.append(_section(title, body_text, truncated, source=source))
        total_chars += len(body_text)
        return total_chars < cfg.max_total_chars

    # Check whether first-run onboarding is active (file exists and is non-empty).
    # When active, skip MEMORY.md and USER.md - they are empty/default and would
    # only confuse the model during onboarding.
    first_run_active = (
        workspace.first_run_file.exists()
        and workspace.first_run_file.read_text(encoding="utf-8").strip() != ""
    )

    # 1. Long-term memory (MEMORY.md) - skipped during first run
    if cfg.inject_long_term_memory and not first_run_active:
        _add("Long-Term Memory", workspace.memory_file)

    # 2. User profile (USER.md) - skipped during first run
    if cfg.inject_user_profile and not first_run_active:
        _add("User Profile", workspace.user_file)

    # 3. Agent roster (AGENTS.md)
    if cfg.inject_agents:
        _add("Agent Roster", workspace.agents_file)
        # In dispatcher mode, inject full tool usage instructions (the agent has only
        # one opaque tool and cannot see individual tool schemas).
        if dispatcher_mode:
            _add("Tool Usage", None, body=_AGENT_TOOLS_DISPATCHER_MD)

    # 4. Relation graph
    if cfg.inject_relations:
        if index is not None:
            relations = _graph.get_relations(index)
            if relations:
                rel_md = _graph.format_relations_for_context(relations)
                _add("Relation Graph", None, body=rel_md, source="RELATIONS.md")
        else:
            _add("Relation Graph", workspace.relations_file)

    # 5. Daily logs: inject cfg.daily_log_days files counting back from today
    if cfg.inject_daily_logs and cfg.daily_log_days > 0:
        today = datetime.date.today()
        days = [today - datetime.timedelta(days=i) for i in range(cfg.daily_log_days - 1, -1, -1)]
        for day in days:
            day_path = workspace.daily_file(day)
            label = f"Daily Log ({day.isoformat()})"
            if not _add(label, day_path, source=f"daily/{day_path.name}"):
                break  # budget exhausted

    # 6. First-run onboarding (FIRST_RUN.md) - injected last; skipped automatically once emptied
    if first_run_active:
        _add("First Run", workspace.first_run_file)

    if not sections:
        return ""

    header = (
        "<!-- groundmemory bootstrap start -->\n"
        "## Your Memory Context\n\n"
        "The following information was loaded from your long-term memory store. "
        "Use it to maintain continuity across sessions.\n\n"
    )
    footer = "\n<!-- groundmemory bootstrap end -->"
    body = header + "\n".join(sections)

    # 7. Compaction notice: inject when requested
    if inject_compaction_notice:
        tiers_str = ", ".join(f"`{t}`" for t in cfg.compaction_tiers)
        notice = _COMPACTION_NOTICE_TEMPLATE.format(tiers=tiers_str)
        body = body + _section("Memory Compaction Required", notice.strip())

    return body + footer